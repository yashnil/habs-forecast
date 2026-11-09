"""CDPH domoic-acid PDFs -> v1/biotoxin/index.json. Fixtures are CDPH's own PDFs recorded on
2026-11-08 (crab: as of 11/2/2026; seafood: as of 10/21/2026); values below are read off the PDFs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.http import FetchError
from coastwatch_pipeline.pilot_models import BiotoxinIndex
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.sources import biotoxin as bt

FIX = Path(__file__).parent / "fixtures" / "cdph"
REPO = Path(__file__).resolve().parents[2]
CRAB = (FIX / "CrabDAResults-2026-11-02.pdf").read_bytes()
SEAFOOD = (FIX / "SeafoodDAResults082826to102226.pdf").read_bytes()
SEAFOOD_URL = bt.DOC_BASE + "SeafoodDAResults082826to102226.pdf"
NOW = "2026-11-08T18:00:00Z"


@pytest.fixture(scope="module")
def idx() -> BiotoxinIndex:
    return bt.build_index(NOW, (CRAB, bt.CRAB_URL, NOW), (SEAFOOD, SEAFOOD_URL, NOW))


def site(idx, sid):
    return next(s for s in idx.sites if s.id == sid)


def test_known_crab_values(idx):
    mb = site(idx, "bma-g-monterey-bay-dungeness-crab")
    s = mb.sets[0]
    assert (mb.bma, mb.bma_label, mb.site_label, mb.species) == ("G", "Monterey", "Monterey Bay", "Dungeness crab")
    assert (s.date, s.kind, s.tissue) == ("2026-09-28", "routine", "viscera")
    assert s.values_ppm == [4.0, 3.0, 4.5, 5.0, 9.8, 6.8] and s.avg_ppm == 5.5 and s.pct_over_action == 0 and s.max_ppm == 9.8
    assert s.action_level_ppm == 30
    pp = site(idx, "bma-f-pillar-point-dungeness-crab").sets[0]
    assert pp.date == "2026-10-09" and pp.avg_ppm == 3.0 and pp.values_ppm[-1] is None and pp.nd_count == 1 and pp.values_raw[-1] == "ND"
    usal = site(idx, "bma-c-usal-dungeness-crab").sets[0]
    assert usal.date == "2026-10-20" and usal.avg_ppm == 38.5 and usal.max_ppm == 99 and usal.pct_over_action == 50
    # paired results are "elevated viscera | meat": viscera in values_ppm, meat alongside
    assert usal.values_ppm == [11, 24, 8.8, 37, 99, 51] and usal.meat_values_ppm == [None, None, None, 6.6, 20, 13]
    assert usal.values_raw[3] == "37 | 6.6"


def test_below_reporting_limit_and_nd_are_null_never_invented(idx):
    rock = site(idx, "bma-g-monterey-bay-rock-crab").sets[0]
    assert rock.values_raw == ["<2.5", "ND", "ND", "20", "<2.5", "2.6"]
    assert rock.values_ppm == [None, None, None, 20.0, None, 2.6] and rock.nd_count == 4
    assert rock.avg_ppm == 3.8  # CDPH's printed average, not recomputed


def test_seafood_anchovy_rows(idx):
    b509 = site(idx, "block-509-northern-anchovy")
    visc = next(s for s in b509.sets if s.tissue == "viscera")
    assert visc.date == "2026-08-28" and visc.values_ppm == [1200] and visc.composite_of == 12 and visc.kind == "unknown"
    assert visc.action_level_ppm == 20 and visc.avg_ppm is None  # a composite is not an average of individuals
    b517 = site(idx, "block-517-northern-anchovy")
    assert {(s.tissue, s.values_ppm[0], s.kind) for s in b517.sets} == {("viscera", 47, "retest"), ("whole", 12, "retest")}
    assert all(s.note and s.note.startswith("RT") for s in b517.sets)
    assert b509.informs_places == [] and b509.lat is None  # blocks are not mapped to places


def test_sources_dates_and_action_levels(idx):
    assert idx.source.as_of == "2026-11-02" and idx.source.url == bt.CRAB_URL and len(idx.source.sha256) == 64
    assert idx.source.period == "September 28, 2026 – November 2, 2026"
    assert idx.additional_sources[0].as_of == "2026-10-21" and idx.additional_sources[0].url == SEAFOOD_URL
    assert {(a.tissue, a.ppm) for a in idx.action_levels} == {("viscera", 30), ("viscera or whole body", 20)}
    assert idx.parse_warnings == [] and idx.errors == []
    assert len([s for s in idx.sites if s.bma]) == 8 and len([s for s in idx.sites if not s.bma]) == 3


def test_places_are_mapped_conservatively(idx):
    places = {s.id: s.informs_places for s in idx.sites}
    assert places["bma-g-monterey-bay-dungeness-crab"] == ["port-592", "port-550", "port-593"]
    assert places["bma-f-pillar-point-dungeness-crab"] == ["port-452"] and places["bma-f-pigeon-point-dungeness-crab"] == ["port-452"]
    assert places["bma-f-duxbury-reef-dungeness-crab"] == []  # Marin, north of the Golden Gate
    assert places["bma-c-usal-dungeness-crab"] == [] and places["bma-a-klamath-river-dungeness-crab"] == []


def test_unreadable_rows_go_to_parse_warnings():
    text = bt.pdf_text(CRAB)
    broken = text.replace("38.5         50%", "38.5         ??%").replace("BMA A - Crescent City              Klamath River       R ", "BMA A - Crescent City              Klamath River       X ")
    p = bt.parse_crab(broken)
    assert len(p.rows) == 6 and len(p.warnings) == 2
    assert any("percent" in w for w in p.warnings) and any("routine/retest" in w for w in p.warnings)
    assert not any(r.site in ("Usal", "Klamath River") for r in p.rows)
    assert bt.parse_crab("nothing here").warnings == ["crab PDF: no result rows found"]


def test_not_a_pdf_is_an_error_not_data():
    idx = bt.build_index(NOW, (b"<html>maintenance</html>", bt.CRAB_URL, NOW), (SEAFOOD, SEAFOOD_URL, NOW))
    assert idx.errors and "not a PDF" in idx.errors[0]
    assert all(not s.bma for s in idx.sites)  # crab rows absent, seafood still there
    with pytest.raises(RuntimeError):
        bt.build_index(NOW, None, None, ["crab PDF: HTTP 503"])


def test_current_seafood_file_is_found_on_the_page():
    html = (FIX / "DomoicAcid-links.html").read_text()
    assert bt.seafood_url(html) == SEAFOOD_URL  # Aug 28 - Oct 22, 2026 is newer than Jun 5 - Jul 23
    assert bt._end_date("SeafoodDAResultsJune052026toJuly232026.pdf").isoformat() == "2026-07-23"
    assert bt.seafood_url("<a href='x.pdf'>none</a>") is None


def test_cdph_tls_uses_the_bundled_intermediate_never_unverified():
    from coastwatch_pipeline import http

    src = Path(bt.__file__).read_text() + Path(http.__file__).read_text()
    assert "CERT_NONE" not in src and "check_hostname = False" not in src and "_create_unverified_context" not in src
    assert any(p.name.startswith("sectigo_public_server_auth_ca_ov_r36") for p in http.CERT_DIR.glob("*.pem"))
    assert http.tls_context().verify_mode.name == "CERT_REQUIRED"


def test_committed_schema_has_the_contract_fields():
    # schemas/v1/biotoxin.schema.json is exported from the model (`cwp schema`; kept in sync by
    # test_committed_json_schema_matches_models); the required keys are contracts.ts BiotoxinIndex
    sch = json.loads((REPO / "schemas" / "v1" / "biotoxin.schema.json").read_text())
    assert set(sch["required"]) == {"version", "generated_at", "source", "action_levels", "sites", "parse_warnings"}
    defs = sch["$defs"]
    assert set(defs["BiotoxinSite"]["required"]) == {"id", "bma", "bma_label", "port_label", "site_label", "species", "lat", "lon", "informs_places", "sets"}
    assert set(defs["BiotoxinSet"]["required"]) == {"date", "kind", "tissue", "values_ppm", "nd_count", "avg_ppm", "max_ppm", "pct_over_action", "action_level_ppm"}


def test_pipeline_writes_the_sidecar_outside_the_manifest(tmp_path):
    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    data = json.loads((out / "biotoxin" / "index.json").read_text())
    BiotoxinIndex.model_validate(data)
    assert data["source"]["as_of"] == "2026-11-02" and data["sites"]
    assert "biotoxin" not in (out / "manifest.json").read_text()


def test_fetch_failures_keep_the_previous_file(tmp_path):
    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    before = (out / "biotoxin" / "index.json").read_text()
    down = lambda url: (_ for _ in ()).throw(FetchError(f"HTTP 503 for {url}"))  # noqa: E731
    run_pipeline(fixture_context(out, fetcher=FixtureFetcher(overrides={r"cdph\.ca\.gov": down})))
    assert (out / "biotoxin" / "index.json").read_text() == before


def test_check_published_validates_the_sidecars(tmp_path):
    from coastwatch_pipeline.publish.check import check_published

    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    rep = check_published(str(out))
    kinds = {c["kind"]: c for c in rep["checked"]} if isinstance(rep["checked"][0], dict) else {}
    assert rep["ok"] and "biotoxin" in kinds and "fisheries-status" in kinds
    (out / "biotoxin" / "index.json").write_text('{"version": 1}')
    assert not check_published(str(out))["ok"]
