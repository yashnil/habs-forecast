"""Derived chlorophyll layers (v1/derived/): gap-filled estimate and 8-day outlook."""

from __future__ import annotations

import json
import math
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from coastwatch_pipeline.derived import estimate as E
from coastwatch_pipeline.derived import outlook as O
from coastwatch_pipeline.derived import run as R
from coastwatch_pipeline.fixtures import fixture_context
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.publish.check import check_derived, check_published, guard_publish_tree

REF = Path(__file__).resolve().parents[2] / "data" / "reference"
NO_OFFSET = {"open": 0.0, "bay": 0.0}


# ---------------------------------------------------------------- background state
def test_background_is_a_decaying_weighted_mean():
    b = np.full((1, 3), np.nan)
    w = np.zeros((1, 3))
    b, w = E.update_background(b, w, 0, np.array([[1.0, np.nan, 0.0]]))
    assert b[0, 0] == 1.0 and np.isnan(b[0, 1]) and w[0, 1] == 0 and w[0, 0] == 1
    b, w = E.update_background(b, w, E.TAU_DAYS, np.array([[0.0, 2.0, np.nan]]))
    f = math.exp(-1)
    assert b[0, 0] == pytest.approx(f / (f + 1))  # older view down-weighted by e^-1
    assert b[0, 1] == 2.0 and w[0, 1] == 1
    assert b[0, 2] == 0.0 and w[0, 2] == pytest.approx(f)  # no view: value kept, weight decays


# ---------------------------------------------------------------- estimate
def _scene(n=80):
    water = np.ones((n, n), bool)
    water[:, -10:] = False  # land on the east
    dist = E.distance_km(water, 0.0025, 37.0)
    bay = np.zeros_like(water)
    b = np.where(water, 0.0, np.nan)
    w = np.where(water, 3.0, 0.0)
    obs = np.where(water, 0.3, np.nan)  # observed everywhere at +0.3 above the background
    return water, dist, bay, b, w, obs


def test_estimate_fills_only_gaps_and_never_touches_observations():
    water, dist, bay, b, w, obs = _scene()
    obs[20:40, 20:40] = np.nan  # a cloud
    est, code = E.estimate(obs, b, w, water, dist, bay, 0.278, 0.222, NO_OFFSET)
    assert not np.isfinite(est[np.isfinite(obs)]).any()
    assert np.isfinite(est[20:40, 20:40]).all()
    # background 0 + the surrounding anomaly (+0.3)
    assert np.allclose(est[20:40, 20:40], 0.3, atol=0.02)
    assert (code[20:40, 20:40] > 0).all() and (code[np.isfinite(obs)] == 0).all()
    assert not np.isfinite(est[~water]).any()


def test_strip_without_history_borrows_with_the_shoreward_offset():
    water, dist, bay, b, w, obs = _scene()
    strip = water & (dist <= 0.3)
    b[strip] = np.nan
    w[strip] = 0
    obs[strip] = np.nan
    est, code = E.estimate(obs, b, w, water, dist, bay, 0.278, 0.222, {"open": 0.5, "bay": 0.2})
    assert np.isfinite(est[strip]).all()
    assert np.allclose(est[strip], 0.0 + 0.5 + 0.3, atol=0.03)
    keys = {E.CLASS_KEYS[c - 1] for c in np.unique(code[strip])}
    assert keys == {"0-0.6 km|open|borrowed"}


def test_no_recent_data_nearby_stays_empty():
    water, dist, bay, b, w, obs = _scene()
    b[:] = np.nan
    w[:] = 0
    obs[:] = np.nan
    est, _ = E.estimate(obs, b, w, water, dist, bay, 0.278, 0.222, NO_OFFSET)
    assert not np.isfinite(est).any()


def test_bay_mask_finds_water_behind_a_narrow_mouth():
    n = 120
    water = np.zeros((n, n), bool)
    water[:, :40] = True  # open sea on the west
    water[40:80, 60:110] = True  # a bay
    water[58:62, 40:60] = True  # a 4-cell (about 1 km) mouth
    bay = E.bay_mask(water, 0.0025, 37.0)
    assert bay[60, 90] and not bay[60, 10] and not bay[5, 38]


def test_class_keys_cover_validation_table():
    v = json.loads((REF / "chl_estimate_validation.json").read_text())
    assert set(v["uncertainty_by_class"]) == set(E.CLASS_KEYS)
    for key, u in v["uncertainty_by_class"].items():
        assert 0.05 < u["rmse_log10"] < 1.5, key
    # the shipped numbers are those of the held-out test, and it beat the older view
    s = v["summary"]["clouds"]
    for band in ("0-0.6 km", "0.6-3 km", "over 3 km", "bays"):
        assert s[band]["estimate"]["rmse_log10"] < s[band]["older_view"]["rmse_log10"], band


# ---------------------------------------------------------------- outlook
def test_next_period_follows_nasa_8day_periods():
    assert O.next_period(date(2026, 10, 12)) == (date(2026, 10, 20), date(2026, 10, 27))
    assert O.next_period(date(2026, 12, 27)) == (date(2027, 1, 1), date(2027, 1, 8))
    assert O.next_period(date(2026, 12, 19)) == (date(2026, 12, 27), date(2026, 12, 31))
    assert O.period(date(2026, 1, 1)) == 0 and O.period(date(2026, 12, 31)) == 45


def test_outlook_is_climatology_plus_damped_anomaly():
    ref = O.load_reference()
    g = ref["grid"]
    px = ref["_px"]
    start = date(2026, 9, 26)
    p_now, p_next = O.period(start), O.period(O.next_period(start)[0])
    grid = np.full((g["height"], g["width"]), np.nan)
    grid[px[:, 0], px[:, 1]] = ref["_clim"][:, p_now] + 1.0  # +1 ln everywhere
    grid[px[0, 0], px[0, 1]] = np.nan  # one cloudy cell
    r = O.outlook(grid, start, ref)
    expect = (ref["_clim"][:, p_next] + ref["_phi8"] * 1.0) / O.LN10
    assert np.allclose(r["forecast_log10"][1:], expect[1:])
    assert r["forecast_log10"][0] == pytest.approx(ref["_clim"][0, p_next] / O.LN10)
    assert r["uncertainty_log10"][0] == pytest.approx(ref["_rmse_cl"][0] / O.LN10)
    assert r["cells_with_anomaly"] == r["cells"] - 1


def test_outlook_reference_is_compact_and_skilful():
    p = REF / "chl_outlook_reference.json"
    assert p.stat().st_size < 20_000_000
    ref = json.loads(p.read_text())
    sk = ref["skill_8d"]
    assert sk["model"] < sk["climatology"] < sk["persistence"]
    assert sk["climatology_minus_model_95ci"][0] > 0  # better than climatology, CI excludes 0
    assert sk["n"] > 100_000 and len(ref["pixels"]) == len(ref["climatology_ln"])


def test_parse_composite_reads_the_recorded_fixture():
    ref = O.load_reference()
    blob = (Path(__file__).parent / "fixtures" / "modis" / "composite-2026-09-26.nc").read_bytes()
    grid = O.parse_composite(blob, ref)
    vals = grid[ref["_px"][:, 0], ref["_px"][:, 1]]
    assert np.isfinite(vals).mean() > 0.5
    assert np.nanmin(vals) >= math.log(O.FLOOR_MG) - 1e-9


# ---------------------------------------------------------------- the pipeline step
@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("derived") / "v1"
    run_pipeline(fixture_context(out))
    return out


def test_pipeline_writes_derived_index_and_leaves_manifest_alone(dataset: Path, tmp_path: Path):
    idx = json.loads((dataset / "derived" / "index.json").read_text())
    ids = [lyr["id"] for lyr in idx["layers"]]
    assert ids == ["chl_estimate_latest", "chl_outlook_8d", "charm_nearshore_extension"]
    manifest = (dataset / "manifest.json").read_text()
    assert "derived/" not in manifest and "chl_estimate" not in manifest and "chl_outlook" not in manifest
    est = idx["layers"][0]
    assert est["validation"]["by_band"]["bays"] > 0 and est["validation"]["n"] > 0
    out = idx["layers"][1]
    assert out["valid_from"] == "2026-10-04" and out["valid_to"] == "2026-10-11"
    assert out["skill"]["model"] < out["skill"]["climatology"] < out["skill"]["persistence"]
    assert any("not covered" in c for c in out["caveats"])
    ext = idx["layers"][2]
    assert ext["kind"] == "validation" and ext["has_demonstrated_skill"] is False
    assert ext["readout"].startswith("Context only")
    # a re-run of derive alone leaves the manifest byte-identical
    copy = tmp_path / "v1"
    shutil.copytree(dataset, copy)
    R.run(fixture_context(copy))
    assert (copy / "manifest.json").read_text() == manifest


def test_estimate_grid_holds_values_only_in_observed_gaps(dataset: Path):
    from coastwatch_pipeline.models import Manifest
    from coastwatch_pipeline.sources import satellite

    m = Manifest.model_validate_json((dataset / "manifest.json").read_text())
    latest = next(lyr for lyr in m.layers if lyr.layer_id == "olci300_chl_latest")
    obs = satellite.load_published(dataset, latest.grid)
    idx = json.loads((dataset / "derived" / "index.json").read_text())
    est = R.read_grid(dataset, idx["layers"][0]["grid"])
    unc = R.read_grid(dataset, idx["layers"][0]["uncertainty_grid"])
    assert est.shape == obs.shape and np.isfinite(est).any()
    assert not (np.isfinite(est) & np.isfinite(obs)).any()
    assert np.array_equal(np.isfinite(est), np.isfinite(unc))
    assert idx["layers"][0]["grid"]["transform"] == "log10" and idx["layers"][0]["uncertainty_grid"]["transform"] == "none"


def test_published_tree_checks_pass_and_catch_a_missing_chunk(dataset: Path, tmp_path: Path):
    copy = tmp_path / "pub" / "v1"
    shutil.copytree(dataset, copy)
    assert guard_publish_tree(copy.parent) == []
    rep = check_published(str(copy))
    derived = [c for c in rep["checked"] if c["kind"].startswith("derived")]
    assert derived and all(c["ok"] for c in derived), rep["problems"]
    idx = json.loads((copy / "derived" / "index.json").read_text())
    g = idx["layers"][0]["grid"]
    k = g["chunks"]["present"][0]
    (copy / g["chunks"]["url_template"].format(row=k.split("_")[0], col=k.split("_")[1])).unlink()
    problems: list[str] = []
    check_derived(str(copy), False, lambda rel, kind, f, ok, detail: problems.append(rel) if not ok else None)
    assert problems


def test_state_advances_and_old_derived_folders_are_pruned(dataset: Path, tmp_path: Path):
    copy = tmp_path / "v1"
    shutil.copytree(dataset, copy)
    first = json.loads((copy / "derived" / "state.json").read_text())
    assert first["through"] == "2026-10-06"
    # a later run with no new days keeps the state (no rewrite) and builds new layer folders
    ctx = fixture_context(copy)
    ctx.run_id = "later-1"
    R.run(ctx)
    ctx.run_id = "later-2"
    R.run(ctx)
    dirs = sorted(p.name for p in (copy / "derived").iterdir() if p.is_dir())
    idx = json.loads((copy / "derived" / "index.json").read_text())
    state = json.loads((copy / "derived" / "state.json").read_text())
    assert state["through"] == "2026-10-06"
    keep = R.referenced_dirs(idx) | R.referenced_dirs(state)
    # this run's folders + the previous run's (clients may still hold the previous index)
    assert keep <= set(dirs) and len([d for d in dirs if d.startswith("estimate-")]) == 2


def test_missing_olci_still_writes_a_valid_index(tmp_path: Path, dataset: Path):
    copy = tmp_path / "v1"
    shutil.copytree(dataset, copy)
    m = json.loads((copy / "manifest.json").read_text())
    m["layers"] = [lyr for lyr in m["layers"] if not lyr["layer_id"].startswith("olci300_")]
    (copy / "manifest.json").write_text(json.dumps(m))
    R.run(fixture_context(copy))
    idx = json.loads((copy / "derived" / "index.json").read_text())
    assert "chl_estimate_latest" not in [lyr["id"] for lyr in idx["layers"]]
    assert any("no estimate" in n for n in idx["notes"])
