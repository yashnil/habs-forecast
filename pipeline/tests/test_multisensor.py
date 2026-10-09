"""Multi-sensor display: one sensor's own value per pixel, native grids kept, members kept,
unobserved areas kept, the rule applied as documented, agreement computed honestly."""

from __future__ import annotations

from collections import Counter
from datetime import date

import numpy as np
import pytest

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.http import FetchError
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.publish.check import check_published
from coastwatch_pipeline.sources import multisensor as ms
from coastwatch_pipeline.sources import satellite as sat
from coastwatch_pipeline.verify_satellite import verify_multisensor

D = date(2026, 10, 6).toordinal()
NAN = np.nan


def test_rule_table():
    #            S3 value, S3 date, VIIRS value, VIIRS date -> sensor
    cases = [
        (1.0, D, NAN, NAN, 1),  # only Sentinel-3
        (NAN, NAN, 2.0, D, 2),  # only VIIRS
        (1.0, D, 2.0, D, 1),  # same day: Sentinel-3 (finer)
        (1.0, D - 2, 2.0, D, 1),  # VIIRS 2 days newer: still Sentinel-3
        (1.0, D - 3, 2.0, D, 2),  # VIIRS more than 2 days newer: VIIRS
        (1.0, D, 2.0, D - 5, 1),  # VIIRS older: Sentinel-3
        (NAN, NAN, NAN, NAN, 0),  # nobody: nothing
    ]
    o, od, v, vd, want = (np.array(c, float) for c in zip(*cases))
    assert ms.pick(o, od, v, vd).tolist() == want.astype(int).tolist()


@pytest.mark.parametrize("region", ["monterey", "north_coast", "socal_bight"])
def test_display_matches_members_pixel_by_pixel(tmp_path, region):
    out = tmp_path / "v1"
    m = run_pipeline(fixture_context(out, fetcher=FixtureFetcher(satellite_region=region)))
    layer = next(lyr for lyr in m.layers if lyr.layer_id == ms.LAYER_ID)
    # members stay published, untouched
    assert {"olci300_chl_latest", "viirs750_chl_latest"} <= {lyr.layer_id for lyr in m.layers}
    assert layer.grid is None and layer.native_resolution_m is None  # no merged grid claiming one resolution
    rows = verify_multisensor(out)
    assert rows and all(r["status"] == "pass" for r in rows), [r for r in rows if r["status"] != "pass"][:3]
    # unobserved stays unobserved: every 'none' pixel is transparent in all three tile sets
    assert any(r["sensor"] == "none" for r in rows)
    cov = {c.region_id: c for c in layer.multisensor.coverage_comparison}["domain"]
    assert cov.combined_fraction >= max(cov.primary_fraction, cov.secondary_fraction) - 1e-9
    assert cov.combined_fraction < 1.0 or region == "socal_bight"
    assert check_published(str(out))["ok"]


def test_viirs_keeps_its_750_m_cells_in_the_display(tmp_path):
    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    rows = [r for r in verify_multisensor(out) if r["sensor"] == "VIIRS"]
    assert rows
    from coastwatch_pipeline.models import Manifest

    man = Manifest.model_validate_json((out / "manifest.json").read_text())
    viirs = next(lyr for lyr in man.layers if lyr.layer_id == "viirs750_chl_latest")
    vals = sat.load_published(out, viirs.grid)
    g = ms.target_of(viirs.grid).grid
    for r in rows[:50]:
        rr, cc = g.cell_index(np.array([r["lat"]]), np.array([r["lon"]]))
        assert r["value"] == pytest.approx(vals[rr[0], cc[0]], abs=1e-5)  # the VIIRS cell's own value (report rounds to 5 dp)


def test_not_built_without_both_members_and_members_survive(tmp_path):
    out = tmp_path / "v1"
    fx = FixtureFetcher(overrides={r"erdVHNchla1day": lambda url: (_ for _ in ()).throw(FetchError(f"HTTP 503 for {url}"))})
    m = run_pipeline(fixture_context(out, fetcher=fx))
    ids = {lyr.layer_id for lyr in m.layers}
    assert ms.LAYER_ID not in ids and "olci300_chl_latest" in ids
    st = next(s for s in m.sources if s.source_id == "satellite_chl")
    assert any("multi-sensor view needs both" in n for n in st.notes)


def test_agreement_recovers_a_known_ratio():
    t_v = sat.Target(37.0, -122.5, 0.0075, 40, 40)
    t_o = sat.Target(37.0 + 0.0025, -122.5 - 0.0025, 0.0025, 120, 120)  # 3x3 pixels per VIIRS cell
    rng = np.random.default_rng(1)
    v = 10 ** rng.uniform(-1, 1, (40, 40))
    rows, cols = ms.cell_map(t_o, t_v)
    o = 2.0 * ms._sample(v, rows, cols)  # Sentinel-3 reads exactly twice VIIRS
    vd = np.full((40, 40), float(D))
    res = {a.region_id: a for a in ms.agreement([(date.fromordinal(D).isoformat(), o)], t_o, v, vd, t_v)}["domain"]
    assert res.n_cells > 1000
    assert res.median_log10_ratio == pytest.approx(np.log10(2), abs=1e-3)
    assert res.pearson_r_log10 == pytest.approx(1.0, abs=1e-6)
    # a day VIIRS did not observe gives no pairs
    none = {a.region_id: a for a in ms.agreement([(date.fromordinal(D - 1).isoformat(), o)], t_o, v, vd, t_v)}["domain"]
    assert none.n_cells == 0 and none.median_log10_ratio is None


def test_sensor_mix_is_reported(tmp_path):
    out = tmp_path / "v1"
    m = run_pipeline(fixture_context(out))
    layer = next(lyr for lyr in m.layers if lyr.layer_id == ms.LAYER_ID)
    mix = Counter(r["sensor"] for r in verify_multisensor(out))
    assert mix["VIIRS"] > 0 and mix["Sentinel-3"] > 0
    assert any(c.name == "one_sensor_per_pixel" for c in layer.qc.checks)
    assert any("Multi-sensor display" in c for c in layer.caveats)
