"""Publishing integrity: references resolve, versions are atomic, pruning is safe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.models import Manifest
from coastwatch_pipeline.pipeline import referenced_paths, run_pipeline
from coastwatch_pipeline.publish.check import check_published, guard_publish_tree

from .conftest import failing


def test_every_manifest_reference_exists_and_decodes(out):
    run_pipeline(fixture_context(out))
    report = check_published(str(out))
    assert report["ok"], report["problems"]
    kinds = {}
    for c in report["checked"]:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    # C-HARM: 12 images + 12 grids; then ports, official, port intel, observations, fisheries;
    # plus every satellite grid chunk and sample tiles
    assert kinds["image"] == 12 and kinds["grid"] >= 12
    assert {"tiles", "age_tiles", "age_grid", "ports", "official", "port_intel", "observations", "fisheries"} <= set(kinds)


def test_asset_paths_are_content_addressed(out):
    m = run_pipeline(fixture_context(out))
    for lyr in m.layers:
        if lyr.image:
            parts = lyr.image.url.split("/")
            assert parts[0] == "charm" and parts[1] == "2026-10-08"
            assert parts[2].startswith(f"lead{lyr.time.lead_days}-") and len(parts[2]) == len("leadN-") + 10
    assert m.ports_url.startswith("ports-") and m.ports_url.endswith(".geojson")


def test_rerun_with_same_data_reuses_paths(out):
    a = run_pipeline(fixture_context(out))
    b = run_pipeline(fixture_context(out, now="2026-10-08T23:00:00Z"))
    charm = lambda m: {p for p in referenced_paths(m) if p.startswith("charm/")}  # noqa: E731
    assert charm(a) == charm(b)
    # ports carry a retrieval timestamp, so each run gets a new content-addressed name
    assert a.ports_url != b.ports_url


def test_pruning_keeps_current_and_previous_and_only_touches_pipeline_files(out):
    run_pipeline(fixture_context(out))
    # an older version referenced by the *previous* manifest must survive one more run
    prev = Manifest.model_validate_json((out / "manifest.json").read_text())
    old = "charm/2026-10-07/lead1-0123456789/particulate_domoic.png"
    (out / old).parent.mkdir(parents=True)
    (out / old).write_bytes(b"old")
    prev.layers[0].image.url = old
    (out / "manifest.json").write_text(prev.model_dump_json())
    # unreferenced leftovers the pipeline owns, and a file it does not own
    junk = out / "charm/2026-09-01/lead0-deadbeef00/x.png"
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"x")
    (out / "ports-old0000000.geojson").write_text("{}")
    (out / "README.txt").write_text("not pipeline output")

    m2 = run_pipeline(fixture_context(out, now="2026-10-09T18:00:00Z"))
    assert (out / old).exists(), "asset of the previous manifest must be retained"
    assert not junk.exists() and not (out / "charm/2026-09-01").exists()
    assert not (out / "ports-old0000000.geojson").exists()
    assert (out / "README.txt").exists(), "pruning must never touch files it does not own"
    assert check_published(str(out))["ok"]

    run_pipeline(fixture_context(out, now="2026-10-10T18:00:00Z"))
    assert not (out / old).exists(), "two versions back is no longer needed"
    assert referenced_paths(m2) <= {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}


def test_failed_refresh_republishes_last_good_without_relabelling(out):
    first = run_pipeline(fixture_context(out))
    fetcher = FixtureFetcher(overrides={r"wvcharmV3_": failing("HTTP 503")})
    later = run_pipeline(fixture_context(out, now="2026-10-11T18:00:00Z", fetcher=fetcher))
    charm = lambda m: {p for p in referenced_paths(m) if p.startswith("charm/")}  # noqa: E731
    assert charm(later) == charm(first)
    st = next(s for s in later.sources if s.source_id == "charm")
    assert st.outcome == "failed" and st.latest_issued_date == "2026-10-08"
    assert later.generated_at == "2026-10-11T18:00:00Z"
    assert check_published(str(out))["ok"]


def test_dangling_reference_is_detected(out):
    m = run_pipeline(fixture_context(out))
    (out / m.layers[0].grid.url).unlink()
    report = check_published(str(out))
    assert not report["ok"] and any("HTTP 404" in p for p in report["problems"])


def test_publish_guard_rejects_non_artifacts(tmp_path: Path):
    root = tmp_path / "data-branch"
    (root / "v1" / "charm").mkdir(parents=True)
    (root / "v1" / "manifest.json").write_text("{}")
    (root / "v1" / "charm" / "a.png").write_bytes(b"x")
    assert guard_publish_tree(root) == []
    (root / "pinn_model.py").write_text("research code")
    (root / "v1" / "notes.md").write_text("x")
    (root / ".env").write_text("SECRET=1")
    bad = guard_publish_tree(root)
    assert set(bad) == {"pinn_model.py", "v1/notes.md", ".env"}


def test_verification_report_is_not_required_but_allowed(tmp_path: Path):
    root = tmp_path / "d"
    (root / "v1" / "verification").mkdir(parents=True)
    (root / "v1" / "verification" / "charm-points.json").write_text(json.dumps({}))
    assert guard_publish_tree(root) == []


def test_m1_published_dataset_still_validates_with_current_schema():
    """Schema changes within schema_version 1 must be additive: the dataset the M1 pipeline
    published to GitHub Pages (recorded 2026-10-08) must still validate."""
    from coastwatch_pipeline.fixtures import FIXTURES
    from coastwatch_pipeline.models import Manifest, PortsCollection

    root = FIXTURES / "compat" / "m1"
    m = Manifest.model_validate_json((root / "manifest.json").read_text())
    assert m.official_url is None and m.port_intel_url is None
    PortsCollection.model_validate_json((root / m.ports_url).read_text())


def test_m2_published_dataset_still_validates_with_current_schema():
    """The dataset the M2 pipeline published to GitHub Pages (recorded 2026-10-08, run
    37846066753) must validate with the M3 models: M3 fields are additive."""
    from coastwatch_pipeline.fixtures import FIXTURES
    from coastwatch_pipeline.models import Manifest, OfficialDataset, PortIntelCollection, PortsCollection

    root = FIXTURES / "compat" / "m2"
    m = Manifest.model_validate_json((root / "manifest.json").read_text())
    assert m.pipeline_run_id == "37846066753"
    assert m.observations_url is None and m.fisheries_url is None
    PortsCollection.model_validate_json((root / m.ports_url).read_text())
    OfficialDataset.model_validate_json((root / m.official_url).read_text())
    PortIntelCollection.model_validate_json((root / m.port_intel_url).read_text())


def test_m3_pipeline_runs_on_top_of_an_m2_output_directory(tmp_path: Path):
    """Switch-over: the first M3 run finds an M2 manifest (no observation/fisheries
    artifacts) and must add them while keeping every M2 artifact it still references."""
    import shutil

    from coastwatch_pipeline.fixtures import FIXTURES

    out = tmp_path / "v1"
    shutil.copytree(FIXTURES / "compat" / "m2", out)
    m = run_pipeline(fixture_context(out))
    assert m.observations_url and m.fisheries_url
    ids = {s.source_id: s.outcome for s in m.sources}
    assert ids["calhabmap"] in ("updated", "partial") and ids["foss_landings"] == "updated"
    # files referenced by the previous (M2) manifest are retained for one run
    prev = json.loads((FIXTURES / "compat" / "m2" / "manifest.json").read_text())
    for rel in (prev["official_url"], prev["port_intel_url"], prev["ports_url"]):
        assert (out / rel).exists()


def test_a_new_rendering_gets_new_addresses_and_is_reported_updated(out, monkeypatch):
    from coastwatch_pipeline.models import Palette, PaletteStop
    from coastwatch_pipeline.sources import charm

    m1 = run_pipeline(fixture_context(out))
    m2 = run_pipeline(fixture_context(out))
    st2 = next(s for s in m2.sources if s.source_id == "charm")
    assert st2.outcome == "unchanged"  # same run, same palette: same URLs
    other = Palette(id="test-palette-v0", domain=[0.0, 1.0], stops=[PaletteStop(value=0.0, color="#000000"), PaletteStop(value=1.0, color="#ffffff")])
    monkeypatch.setattr(charm, "PROBABILITY", other)
    m3 = run_pipeline(fixture_context(out))
    old = {lyr.image.url for lyr in m1.layers if lyr.group_id == "charm"}
    new = {lyr.image.url for lyr in m3.layers if lyr.group_id == "charm"}
    assert old.isdisjoint(new)
    assert next(s for s in m3.sources if s.source_id == "charm").outcome == "updated"


def test_http_retries_transient_403_but_not_404(monkeypatch):
    import urllib.error
    import urllib.request

    from coastwatch_pipeline import http

    calls = {"n": 0}

    class R:
        status = 200
        headers = {"Content-Type": "text/csv"}

        def read(self):
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def flaky(req, timeout, context):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, None)
        return R()

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    assert http.fetch("https://example.test/a").body == b"ok" and calls["n"] == 2

    def missing(req, timeout, context):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", missing)
    with pytest.raises(http.FetchError, match="HTTP 404"):
        http.fetch("https://example.test/b")
