"""Official notices: registry validation (incl. adversarial cases), watcher, geometry."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np
import pytest

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.http import Response
from coastwatch_pipeline.models import OfficialDataset, OfficialRegistry
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.sources import official

from .conftest import failing

RAW = json.loads(official.REGISTRY.read_text())


def reg_from(raw: dict) -> OfficialRegistry:
    return OfficialRegistry.model_validate(raw)


def errors_for(mutate) -> list[str]:
    raw = copy.deepcopy(RAW)
    mutate(raw)
    errs, _ = official.validate_registry(reg_from(raw))
    return errs


def rec(raw: dict, rid: str) -> dict:
    return next(r for r in raw["records"] if r["id"] == rid)


# ---------------------------------------------------------------- committed registry
def test_committed_registry_is_valid_and_conflict_free():
    errs, conflicts = official.validate_registry(official.load_registry())
    assert errs == [] and conflicts == []


def test_committed_registry_review_status_is_honest():
    r = official.load_registry().review
    if re.search(r"\bAI\b|assistant|not reviewed by a person", r.reviewed_by, re.I):
        assert r.status == "pending_human_review", "an AI transcription must never be marked human_verified"
    assert set(r.source_fingerprints) == {w.id for w in official.load_registry().watched_sources}


def test_every_active_record_quotes_and_links_its_agency():
    for r in official.load_registry().records:
        assert r.official_text.strip()
        assert r.sources and all(s.url.startswith(("https://", "tel:")) for s in r.sources)
        assert any(("cdph.ca.gov" in s.url) or ("wildlife.ca.gov" in s.url) or ("oehha.ca.gov" in s.url) or ("dfg.ca.gov" in s.url) or ("arcgis.com" in s.url) for s in r.sources)


# ---------------------------------------------------------------- adversarial registry cases
@pytest.mark.parametrize(
    "name,mutate,expect",
    [
        ("contradictory: active but lifted", lambda raw: rec(raw, "cdfw-2024-razor-clam-humboldt").update(lifted_date="2026-09-01"), "must not have lifted_date"),
        ("lifted without a date", lambda raw: rec(raw, "cdfw-2024-razor-clam-humboldt").update(status="lifted"), "requires lifted_date"),
        ("missing effective date, no explanation", lambda raw: rec(raw, "cdfw-rock-crab-commercial-40n").update(effective_date_note=None), "effective_date_note"),
        ("ambiguous band: no southern latitude", lambda raw: rec(raw, "cdfw-rock-crab-commercial-40n")["area"].update(lat_south=None), "needs lat_north and lat_south"),
        ("inverted band", lambda raw: rec(raw, "cdfw-rock-crab-commercial-40n")["area"].update(lat_south=40.6), "not a valid California range"),
        ("band outside California", lambda raw: rec(raw, "cdfw-rock-crab-commercial-40n")["area"].update(lat_north=45.0), "not a valid California range"),
        ("unknown county", lambda raw: rec(raw, "cdph-2026-sn26-018-monterey-bivalves")["area"].update(counties=["Atlantis"]), "unknown counties"),
        ("end date before start", lambda raw: rec(raw, "cdph-2026-annual-mussel-quarantine").update(expected_end_date="2026-01-01"), "before effective_date"),
        ("end date without caveat", lambda raw: rec(raw, "cdph-2026-annual-mussel-quarantine").update(expected_end_note=None), "expected_end_note"),
        ("non-https source", lambda raw: rec(raw, "cdph-2026-sn26-018-monterey-bivalves")["sources"][0].update(url="http://example.com"), "https or tel"),
        ("statement calling an area safe", lambda raw: raw["statements"][0].update(statement="Monterey Bay is safe for crab."), "must not characterise"),
        ("duplicate ids", lambda raw: raw["records"].append(copy.deepcopy(raw["records"][0])), "duplicate ids"),
        ("not a date", lambda raw: rec(raw, "cdph-2026-sn26-018-monterey-bivalves").update(effective_date="Sept 14"), "not an ISO date"),
    ],
)
def test_adversarial_registry_entries_are_rejected(name, mutate, expect):
    errs = errors_for(mutate)
    assert any(expect in e for e in errs), (name, errs)


def test_unpublishable_registry_keeps_previous_dataset_and_reports_failure(out, tmp_path):
    first = run_pipeline(fixture_context(out))
    bad = copy.deepcopy(RAW)
    rec(bad, "cdfw-2024-razor-clam-humboldt").update(lifted_date="2026-09-01")
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    res = official.run(fixture_context(out), path=p)
    assert res.dataset is None and "must not have lifted_date" in res.errors[0]
    assert first.official_url  # the earlier valid dataset remains what the manifest references


def test_conflicting_records_are_published_as_conflicts():
    raw = copy.deepcopy(RAW)
    reopen = copy.deepcopy(rec(raw, "cdfw-2024-razor-clam-humboldt"))
    reopen.update(id="cdfw-2026-razor-clam-humboldt-reopening", action="reopening", title="Razor clam reopening")
    raw["records"].append(reopen)
    errs, conflicts = official.validate_registry(reg_from(raw))
    assert errs == []
    assert conflicts and "reopening" in conflicts[0] and "both cannot be current" in conflicts[0]


# ---------------------------------------------------------------- watcher
def test_watcher_matches_reviewed_fingerprints_on_unchanged_pages(ctx):
    results = official.watch(ctx, official.load_registry())
    assert all(w.ok for w in results)
    assert all(w.matches_review is True for w in results)
    assert all(w.new_items == [] for w in results)


def _altered(url: str, transform):
    body = FixtureFetcher()(url).body.decode()
    return lambda u: Response(url=u, status=200, content_type="text/html", body=transform(body).encode())


def test_watcher_flags_changed_page_without_touching_records(out):
    before = official.REGISTRY.read_bytes()
    url = "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories"
    fetcher = FixtureFetcher(overrides={re.escape(url): _altered(url, lambda b: re.sub(r"remains closed in(\s|<[^>]+>)+Humboldt County", "is now open in Humboldt County", b))})
    res = official.run(fixture_context(out, fetcher=fetcher))
    w = next(w for w in res.dataset.watch if w.source_id == "cdfw_health_advisories")
    assert w.ok and w.matches_review is False
    assert any("changed since the last review" in n for n in res.notes)
    # the record is NOT changed by the watcher: the closure stays as reviewed
    razor = next(r for r in res.dataset.registry.records if r.id == "cdfw-2024-razor-clam-humboldt")
    assert razor.status == "active"
    assert official.REGISTRY.read_bytes() == before


def test_watcher_detects_new_cdph_release(out):
    url = "https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx"
    inject = '<a href="/Programs/OPA/Pages/SN26-020.aspx">CDPH Lifts Warning for Monterey County Bivalves</a>'
    fetcher = FixtureFetcher(overrides={re.escape(url): _altered(url, lambda b: b.replace("<body", inject + "<body", 1) if "<body" in b else inject + b)})
    res = official.run(fixture_context(out, fetcher=fetcher))
    w = next(w for w in res.dataset.watch if w.source_id == "cdph_release_list")
    assert w.new_items == ["SN26-020"] and w.matches_review is False
    # a release titled 'Lifts' does not lift anything until a person edits the registry
    mon = next(r for r in res.dataset.registry.records if r.id == "cdph-2026-sn26-018-monterey-bivalves")
    assert mon.status == "active"


@pytest.mark.parametrize("override", [failing("HTTP 403"), lambda u: Response(url=u, status=200, content_type="text/html", body=b"<html>Access denied</html>")])
def test_unreadable_official_page_is_reported_not_ignored(out, override):
    fetcher = FixtureFetcher(overrides={r"wildlife\.ca\.gov/Fishing/Ocean/Health-Advisories": override})
    m = run_pipeline(fixture_context(out, fetcher=fetcher))
    st = next(s for s in m.sources if s.source_id == "official")
    assert st.outcome == "partial" and st.error
    ds = OfficialDataset.model_validate_json((out / m.official_url).read_text())
    w = next(w for w in ds.watch if w.source_id == "cdfw_health_advisories")
    assert w.ok is False and w.matches_review is None


def test_normalisation_ignores_html_comments_and_zero_width_characters():
    a = "<h1>T</h1><p>Closure in effect​ at 40° N</p>" + "x" * 300
    b = "<h1>T</h1><!-- old section: anchovy --><p>Closure in effect at 40⁰ N</p>" + "x" * 300
    assert official.main_text(a) == official.main_text(b)


def test_human_verified_review_requires_explicit_confirmation(capsys):
    from coastwatch_pipeline.cli import main

    assert main(["review-official", "--reviewer", "Someone"]) == 2
    assert "--confirm" in capsys.readouterr().err


def test_review_refuses_while_sources_unreadable(tmp_path):
    p = tmp_path / "reg.json"
    p.write_text(official.REGISTRY.read_text())
    ctx = fixture_context(tmp_path / "o", fetcher=FixtureFetcher(overrides={r"cdph\.ca\.gov": failing("HTTP 503")}))
    with pytest.raises(official.RegistryError, match="unreadable"):
        official.record_review(p, ctx, "Tester", "human_verified", "test")


# ---------------------------------------------------------------- geometry
def test_latitude_bands_stay_between_official_latitudes_over_nearshore_ocean():
    mask = official.OceanMask.load()
    north, south = 37.183333, 36.524333
    g = official.lat_band_feature(mask, north, south)
    ys = [pt[1] for poly in g["coordinates"] for ring in poly for pt in ring]
    xs = [pt[0] for poly in g["coordinates"] for ring in poly for pt in ring]
    assert min(ys) >= south - 1e-5 and max(ys) <= north + 1e-5  # coordinates rounded to 5 dp (~1 m)
    # every shaded rectangle's centre is an ocean cell within 20 km of land
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(mask.ocean, sampling=(abs(mask.lat_step) * 111.32, abs(mask.lon_step) * 111.32 * np.cos(np.radians(36.85))))
    for poly in g["coordinates"]:
        ring = poly[0]
        cy = (ring[0][1] + ring[2][1]) / 2
        r = int(round((cy - mask.lat_first) / mask.lat_step))
        c0 = int(round((ring[0][0] + abs(mask.lon_step) / 2 - mask.lon_first) / mask.lon_step))
        c1 = int(round((ring[1][0] - abs(mask.lon_step) / 2 - mask.lon_first) / mask.lon_step))
        assert mask.ocean[r, c0 : c1 + 1].all()
        assert (dist[r, c0 : c1 + 1] <= 20.0 + 1e-6).all()
    # Monterey Bay (inside the band) is covered; the Monterey Peninsula land is not
    assert min(xs) > -122.75 and max(xs) < -121.75


def test_official_geometry_and_failures(out):
    m = run_pipeline(fixture_context(out))
    ds = OfficialDataset.model_validate_json((out / m.official_url).read_text())
    kinds = {(f["properties"]["kind"], tuple(f["properties"]["record_ids"])) for f in ds.geometry["features"]}
    assert ("county", ("cdph-2026-sn26-018-monterey-bivalves",)) in kinds
    assert ("named_area", ("cdph-nci-bivalve-special-advisory",)) in kinds
    assert ("lat_band", ("cdph-2026-sn26-019-anchovy-central-coast", "cdfw-2026-anchovy-take-restriction-monterey-bay")) in kinds
    assert ds.geometry_errors == []
    # statewide records are not drawn as shapes
    assert not any("cdph-2026-annual-mussel-quarantine" in f["properties"]["record_ids"] for f in ds.geometry["features"])


def test_missing_official_geometry_is_reported_not_invented(out):
    fetcher = FixtureFetcher(overrides={r"arcgis\.com": failing("HTTP 500")})
    res = official.run(fixture_context(out, fetcher=fetcher))
    assert any("Monterey County polygon unavailable" in e for e in res.dataset.geometry_errors)
    assert not any(f["properties"]["kind"] == "county" for f in res.dataset.geometry["features"])
    # the record itself is still published (text + sources), just not drawn
    assert any(r.id == "cdph-2026-sn26-018-monterey-bivalves" for r in res.dataset.registry.records)


def test_official_dataset_never_contains_open_or_safe_language(out):
    m = run_pipeline(fixture_context(out))
    ds = json.loads((out / m.official_url).read_text())
    text = json.dumps(ds["registry"]["records"]) + json.dumps(ds["policy"])
    for bad in (r"\bis safe\b", r"\bsafe to\b", r"\ball clear\b", r"\bnow open\b", r"\bopen for\b"):
        assert not re.search(bad, text, re.I), bad
