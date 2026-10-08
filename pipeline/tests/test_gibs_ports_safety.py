"""GIBS tile-date fallback, CDFW ports, attribution, and safety-copy invariants."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.models import SCHEMA_MODELS, Manifest, PortsCollection
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.sources import gibs, ports

from .conftest import failing, static

REPO = Path(__file__).resolve().parents[2]


def _rgba_noise_png() -> bytes:
    # mimics the corrupt striped RGBA tiles GIBS served for its newest date (test-only)
    a = (np.random.default_rng(3).integers(0, 255, (256, 256, 4))).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a, "RGBA").save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- GIBS
def test_newest_empty_date_is_skipped_for_previous(ctx):
    res = gibs.run(ctx)
    viirs = next(lyr for lyr in res.layers if lyr.layer_id == "gibs_viirs_noaa20_chl")
    assert viirs.time.observed_date == "2026-10-06"  # caps default 10-07 has empty tiles
    assert "2026-10-06" in viirs.tiles.url_template
    assert any("2026-10-07 rejected" in n for n in res.notes)


def test_http_500_newest_date_falls_back(out):
    fetcher = FixtureFetcher(overrides={r"/default/2026-10-07/": static(b"Internal error", status=500)})
    res = gibs.run(fixture_context(out, fetcher=fetcher))
    viirs = next(lyr for lyr in res.layers if lyr.layer_id == "gibs_viirs_noaa20_chl")
    assert viirs.time.observed_date == "2026-10-06"


def test_corrupt_rgba_tiles_are_rejected():
    ok, _ = gibs.tile_check(200, _rgba_noise_png())
    assert not ok
    assert gibs.tile_check(500, b"")[0] is False
    good = (Path(__file__).parent / "fixtures" / "gibs" / "tile_previous.png").read_bytes()
    ok, frac = gibs.tile_check(200, good)
    assert ok and frac > 0.2


def test_no_valid_date_produces_no_layer_and_an_error(out):
    fetcher = FixtureFetcher(overrides={r"gibs\.earthdata\.nasa\.gov/wmts/epsg3857/best/\w+/default": failing("HTTP 500")})
    res = gibs.run(fixture_context(out, fetcher=fetcher))
    assert res.layers == []
    assert len(res.errors) == 2


def test_legend_comes_from_capabilities_and_is_verified(ctx):
    res = gibs.run(ctx)
    for lyr in res.layers:
        assert lyr.tiles.legend_verified
        assert lyr.tiles.legend_url.endswith("_H.svg")
        assert "NOAA20_Chlorophyll_a_H" not in lyr.tiles.legend_url  # the old dead URL


def test_dead_legend_is_flagged_not_hidden(out):
    fetcher = FixtureFetcher(overrides={r"/legends/": failing("HTTP 404")})
    res = gibs.run(fixture_context(out, fetcher=fetcher))
    assert res.layers and all(not lyr.tiles.legend_verified for lyr in res.layers)


# ---------------------------------------------------------------- ports
def test_ports_from_cdfw_fix_known_errors(ctx):
    res = ports.run(ctx)
    assert res.errors == []
    fc = res.collection
    by_name = {f.properties.display_name: f for f in fc.features}
    lon, lat = by_name["Santa Barbara"].geometry["coordinates"]
    assert 34.38 < lat < 34.43  # old hand-placed point was 34.25 (in the channel)
    lon, lat = by_name["San Pedro"].geometry["coordinates"]
    assert -118.35 < lon < -118.2  # old point was -118.50
    codes = [f.properties.port_code for f in fc.features]
    names = [f.properties.display_name for f in fc.features]
    assert len(set(codes)) == len(codes) and len(set(names)) == len(names)
    assert "Pillar Point" not in " ".join(names)  # single identity: Princeton / Half Moon Bay
    assert len(fc.features) == 22


def test_unknown_port_code_fails_loudly(ctx, tmp_path):
    cur = json.loads(ports.CURATED.read_text())
    cur["ports"].append({"port_code": 99999, "display_name": "Nowhere", "expected_port_area": "EUREKA"})
    p = tmp_path / "ports.json"
    p.write_text(json.dumps(cur))
    res = ports.run(ctx, curated_path=p)
    assert res.collection is None and "99999" in res.errors[0]


def test_port_area_change_upstream_fails_loudly(ctx, tmp_path):
    cur = json.loads(ports.CURATED.read_text())
    cur["ports"][0]["expected_port_area"] = "SAN DIEGO"
    p = tmp_path / "ports.json"
    p.write_text(json.dumps(cur))
    assert ports.run(ctx, curated_path=p).collection is None


# ---------------------------------------------------------------- schema, attribution, safety
def test_committed_json_schema_matches_models():
    for name, model in SCHEMA_MODELS.items():
        committed = json.loads((REPO / "schemas" / "v1" / f"{name}.schema.json").read_text())
        expected = model.model_json_schema()
        expected["$id"] = committed["$id"]
        assert committed == json.loads(json.dumps(expected, sort_keys=True)), f"run `uv run cwp schema` ({name})"


def test_manifest_and_ports_validate(out):
    run_pipeline(fixture_context(out))
    m = Manifest.model_validate_json((out / "manifest.json").read_text())
    PortsCollection.model_validate_json((out / m.ports_url).read_text())
    assert m.schema_version == 1


def test_every_layer_is_attributed(out):
    m = run_pipeline(fixture_context(out))
    for lyr in m.layers:
        p = lyr.provenance
        assert p.source_url.startswith("https://") and p.license and p.retrieved_at
        assert p.pipeline_run_id and p.pipeline_version
        if lyr.group_id == "charm":
            assert p.dataset_id.startswith("wvcharmV3_") and p.product_version == "3.1"
            assert p.citation and "doi:10.1016/j.hal.2016.08.006" in p.citation
            assert all(u.startswith("https://coastwatch.pfeg.noaa.gov/erddap/") for u in p.request_urls)


SAFE_RE = re.compile(r"\bsafe(ly|ty)?\b", re.I)
FORBIDDEN = [r"all clear", r"go fish", r"safe to (eat|fish|harvest)", r"no risk", r"\bclosed\b", r"\bopen for\b"]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;])\s+", text) if s]


def test_artifact_copy_never_labels_areas_safe(out):
    m = run_pipeline(fixture_context(out))
    texts = []
    for lyr in m.layers:
        texts += [lyr.title, lyr.short_title, lyr.description, lyr.threshold_text or "", *lyr.caveats, lyr.freshness.note]
    for t in texts:
        for s in _sentences(t):
            if SAFE_RE.search(s):
                assert re.search(r"\bnot\b|\bdoes not\b", s, re.I), f"'safe' without negation: {s!r}"
            for pat in FORBIDDEN:
                assert not re.search(pat, s, re.I), f"forbidden phrase {pat!r} in {s!r}"


def test_forecast_layers_disclaim_regulatory_status(out):
    m = run_pipeline(fixture_context(out))
    for lyr in m.layers:
        joined = " ".join(lyr.caveats)
        if lyr.group_id == "charm":
            assert "not a closure or health decision" in joined
            assert "does not mean an area is safe" in joined
            assert "inferred" in joined  # issue date honesty
        if lyr.variable == "chlorophyll_a":
            assert "does not measure toxins" in joined
            assert "gap means no observation" in joined


def test_chlorophyll_is_never_an_official_or_forecast_product(out):
    m = run_pipeline(fixture_context(out))
    for lyr in m.layers:
        if lyr.variable == "chlorophyll_a":
            assert lyr.product_class == "observation"
        if lyr.product_class.startswith("official"):
            assert lyr.group_id == "charm"
