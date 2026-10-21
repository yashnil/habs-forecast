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


def test_plain_classes_follow_what_the_estimator_did():
    water, dist, bay, b, w, obs = _scene()
    bay[:, 50:70] = True  # an enclosed strip of water by the eastern shore
    strip = water & (dist <= 0.3)
    b[strip] = np.nan  # never observed next to the shore: borrowed background
    w[strip] = 0
    obs[strip] = np.nan
    obs[10:30, 10:30] = np.nan  # a cloud over open water with its own history
    est, code = E.estimate(obs, b, w, water, dist, bay, 0.278, 0.222, {"open": 0.5, "bay": 0.2})
    cls = E.plain_class_grid(obs, code)
    assert (cls[np.isfinite(obs)] == E.OBSERVED).all() and not np.isfinite(est[cls == E.OBSERVED]).any()
    assert np.array_equal(np.isfinite(est), np.isin(cls, [1, 2, 3]))
    assert (cls[10:30, 10:30] == E.FILLED).all()
    assert (cls[strip & ~bay] == E.SHOREWARD).all() and (cls[strip & bay] == E.BAY_BORROWED).all()
    # the plain class is a function of the detailed (uncertainty) class
    for c in np.unique(code[code > 0]):
        band, typ, tier = E.CLASS_KEYS[c - 1].split("|")
        want = E.FILLED if tier == "own" else E.BAY_BORROWED if typ == "bay" else E.SHOREWARD if band == "0-0.6 km" else E.FILLED
        assert (cls[code == c] == want).all()
    assert not np.isfinite(cls[~water]).any()


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


BAY_POINTS = {  # (lat, lon) inside each named bay
    "San Francisco Bay": (37.70, -122.28), "San Pablo Bay": (38.05, -122.38), "Suisun Bay": (38.08, -122.05),
    "Tomales Bay": (38.161, -122.902), "Humboldt Bay (north)": (40.832, -124.131), "Humboldt Bay (south)": (40.72, -124.24),
    "Morro Bay": (35.34, -120.84), "Elkhorn Slough": (36.815, -121.765), "Mission Bay": (32.78, -117.225),
    "San Diego Bay": (32.717, -117.202), "South San Diego Bay": (32.62, -117.12),
}
OPEN_POINTS = {  # open coast, including open embayments
    "Monterey Bay": (36.80, -121.90), "Half Moon Bay coast": (37.48, -122.50), "Gulf of the Farallones": (37.75, -122.60),
    "Bodega Bay": (38.29, -123.05), "Santa Monica Bay": (33.90, -118.50), "off Point Loma": (32.68, -117.26),
}


def _cell(t, lat, lon):
    return int(round((t.lat_first - lat) / t.step)), int(round((lon - t.lon_first) / t.step))


def test_bay_mask_on_the_marine_water_mask_holds_every_named_bay():
    from coastwatch_pipeline.sources import satellite

    t = satellite.target_for(satellite.OLCI, satellite.CALIFORNIA)
    _, _, bay, water = R.geometry(t)
    for name, (lat, lon) in BAY_POINTS.items():
        r, c = _cell(t, lat, lon)
        assert water[r, c] and bay[r, c], name
    for name, (lat, lon) in OPEN_POINTS.items():
        r, c = _cell(t, lat, lon)
        assert water[r, c] and not bay[r, c], name
    # most of SF + San Pablo Bay is bay (east of the Golden Gate closing line)
    lats, lons = t.lats()[:, None], t.lons()[None, :]
    box = water & (lats > 37.45) & (lats < 38.15) & (lons > -122.47) & (lons < -122.0)
    assert bay[box].mean() > 0.98


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
    r = O.outlook(grid, start, ref, masked=np.zeros(len(px), bool))
    expect = (ref["_clim"][:, p_next] + ref["_phi8"] * 1.0) / O.LN10
    assert np.allclose(r["forecast_log10"][1:], expect[1:])
    assert r["forecast_log10"][0] == pytest.approx(ref["_clim"][0, p_next] / O.LN10)
    assert r["uncertainty_log10"][0] == pytest.approx(ref["_rmse_cl"][0] / O.LN10)
    assert r["cells_with_anomaly"] == r["cells"] - 1


def test_outlook_is_masked_in_bays_and_unchanged_on_the_open_coast():
    ref = O.load_reference()
    g, px = ref["grid"], ref["_px"]
    start = date(2026, 9, 26)
    grid = np.full((g["height"], g["width"]), np.nan)
    grid[px[:, 0], px[:, 1]] = ref["_clim"][:, O.period(start)] + 0.5
    masked = O.bay_cells(ref)
    r = O.outlook(grid, start, ref)
    plain = O.outlook(grid, start, ref, masked=np.zeros(len(px), bool))
    # open coast: exactly the unmasked forecast; bays: nothing
    assert np.array_equal(r["forecast_log10"][~masked], plain["forecast_log10"][~masked])
    assert np.array_equal(r["uncertainty_log10"][~masked], plain["uncertainty_log10"][~masked])
    assert not np.isfinite(r["forecast_log10"][masked]).any() and not np.isfinite(r["uncertainty_log10"][masked]).any()
    assert r["cells_masked_bay"] == int(masked.sum()) and r["cells"] + r["cells_masked_bay"] == len(px)
    # every cell of the research grid over San Francisco and San Pablo Bays is masked (34 before)
    lat = g["lat_first"] + px[:, 0] * g["lat_step"]
    lon = g["lon_first"] + px[:, 1] * g["lon_step"]
    sf = (lat >= 37.45) & (lat <= 38.10) & (lon >= -122.42) & (lon <= -122.0)
    assert sf.sum() >= 30 and masked[sf].all()
    # no outlook value in a cell holding a named bay's interior
    out = O.to_grid(r["forecast_log10"], ref)
    for name, (la, lo) in BAY_POINTS.items():
        rr = int(np.floor((la - (g["lat_first"] - g["lat_step"] / 2)) / g["lat_step"]))
        cc = int(np.floor((lo - (g["lon_first"] - g["lon_step"] / 2)) / g["lon_step"]))
        if 0 <= rr < g["height"] and 0 <= cc < g["width"] and np.isfinite(out[rr, cc]):
            share = O.bay_share(ref)[(px[:, 0] == rr) & (px[:, 1] == cc)][0]
            assert share < O.BAY_SHARE_MASK, name  # a mostly open-coast cell that only touches the bay
    # no cell with mostly bay water keeps a value; cells without bay water are all kept
    share = O.bay_share(ref)
    assert not np.isfinite(r["forecast_log10"][np.nan_to_num(share) >= O.BAY_SHARE_MASK]).any()
    assert np.isfinite(r["forecast_log10"][np.nan_to_num(share) == 0]).all()


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
    assert ids == ["chl_estimate_latest", "chl_outlook_8d", "sst_latest", "charm_nearshore_extension"]
    manifest = (dataset / "manifest.json").read_text()
    assert "derived/" not in manifest and "chl_estimate" not in manifest and "chl_outlook" not in manifest
    est = idx["layers"][0]
    assert est["validation"]["by_band"]["bays"] > 0 and est["validation"]["n"] > 0
    out = idx["layers"][1]
    assert out["valid_from"] == "2026-10-04" and out["valid_to"] == "2026-10-11"
    assert out["skill"]["model"] < out["skill"]["climatology"] < out["skill"]["persistence"]
    assert any("not covered" in c for c in out["caveats"])
    assert any(c.startswith("Bays are masked") for c in out["caveats"]) and out["coverage"]["bay_cells_masked"] > 30
    ov = R.read_grid(dataset, out["grid"])
    gg = out["grid"]
    lats = gg["lat_first"] + gg["lat_step"] * np.arange(gg["height"])[:, None]
    lons = gg["lon_first"] + gg["lon_step"] * np.arange(gg["width"])[None, :]
    sf = (lats >= 37.52) & (lats <= 38.06) & (lons >= -122.40) & (lons <= -122.10)
    assert np.isfinite(ov).any() and not np.isfinite(ov[sf]).any()
    sst = idx["layers"][2]
    assert sst["kind"] == "sst" and sst["units"] == "degC" and sst["observed_dates"] == ["2026-10-07"]
    assert sst["grid"]["transform"] == "none" and any("analysis" in c.lower() for c in sst["caveats"])
    ext = idx["layers"][3]
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


def test_class_grid_matches_the_estimate_and_its_uncertainty(dataset: Path):
    from coastwatch_pipeline.models import Manifest
    from coastwatch_pipeline.sources import satellite

    m = Manifest.model_validate_json((dataset / "manifest.json").read_text())
    latest = next(lyr for lyr in m.layers if lyr.layer_id == "olci300_chl_latest")
    obs = satellite.load_published(dataset, latest.grid)
    lyr = json.loads((dataset / "derived" / "index.json").read_text())["layers"][0]
    est, unc = R.read_grid(dataset, lyr["grid"]), R.read_grid(dataset, lyr["uncertainty_grid"])
    cg = lyr["class_grid"]
    cls = R.read_grid(dataset, cg)
    assert cg["transform"] == "none" and cg["scale_factor"] == 1.0 and cg["add_offset"] == 0.0
    assert (cg["width"], cg["height"], cg["chunks"]["rows"], cg["chunks"]["cols"]) == (lyr["grid"]["width"], lyr["grid"]["height"], lyr["grid"]["chunks"]["rows"], lyr["grid"]["chunks"]["cols"])
    assert set(lyr["class_legend"]) == {"1", "2", "3"} == set(lyr["class_validation"])
    assert all(isinstance(v, str) and v for v in lyr["class_legend"].values())
    assert set(np.unique(cls[np.isfinite(cls)])) <= {0.0, 1.0, 2.0, 3.0}
    # observed pixels are class 0 and never get an estimate; estimated pixels are 1-3
    assert (cls[np.isfinite(obs)] == 0).all() and not np.isfinite(est[cls == 0]).any()
    assert np.array_equal(np.isfinite(est), np.isfinite(cls) & (cls > 0))
    # each class carries the uncertainty of the detailed classes it stands for
    v = R.load_validation()["uncertainty_by_class"]
    for k in (1, 2, 3):
        allowed = np.array([v[key]["rmse_log10"] for key in E.CLASS_KEYS if E.plain_class(key) == k])
        got = unc[cls == k]
        assert got.size and (np.abs(got[:, None] - allowed[None, :]).min(axis=1) < 1e-3).all(), k


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


def test_sst_is_clipped_to_the_coastal_scope(dataset: Path):
    from coastwatch_pipeline.derived import sst as S
    from coastwatch_pipeline.sources import satellite

    idx = json.loads((dataset / "derived" / "index.json").read_text())
    lyr = next(x for x in idx["layers"] if x["id"] == "sst_latest")
    v = R.read_grid(dataset, lyr["grid"])
    g = lyr["grid"]
    t = satellite.Target(g["lat_first"], g["lon_first"], S.STEP, g["height"], g["width"])
    scope = satellite.marine_water(t) & satellite.coastal_mask(t)
    assert np.isfinite(v).any() and not np.isfinite(v[~scope]).any()
    f = v[np.isfinite(v)]
    assert 5 < f.min() and f.max() < 30


# ---------------------------------------------------------------- last known good
def _refusing_fetcher():
    from coastwatch_pipeline.fixtures import FixtureFetcher
    from coastwatch_pipeline.http import FetchError

    def refuse(url):
        raise FetchError(f"coastwatch.pfeg.noaa.gov not asked again this run: it refused 3 datasets (HTTP 403 on every attempt): {url}", refused=True)

    return FixtureFetcher(overrides={r"erdMH1chla8day_R2022NRT|jplMURSST41": refuse})


def test_outlook_and_sst_are_carried_forward_when_noaa_refuses(dataset: Path, tmp_path: Path):
    copy = tmp_path / "v1"
    shutil.copytree(dataset, copy)
    before = {lyr["id"]: lyr for lyr in json.loads((copy / "derived" / "index.json").read_text())["layers"]}
    ctx = fixture_context(copy, fetcher=_refusing_fetcher())
    ctx.run_id = "refused-1"
    R.run(ctx)
    ctx.run_id = "refused-2"  # a second refused run carries the carried entry again
    R.run(ctx)
    idx = json.loads((copy / "derived" / "index.json").read_text())
    after = {lyr["id"]: lyr for lyr in idx["layers"]}
    assert list(after) == ["chl_estimate_latest", "chl_outlook_8d", "sst_latest", "charm_nearshore_extension"]
    for lid in ("chl_outlook_8d", "sst_latest"):
        a, b = after[lid], before[lid]
        # unchanged: same dates, same grid files (still on disk after pruning)
        for k in ("issued", "valid_from", "valid_to", "observed_dates", "grid", "uncertainty_grid", "method", "basis"):
            assert a.get(k) == b.get(k), (lid, k)
        assert a["caveats"][0].startswith("Not updated this run: NOAA's server refused the request")
        assert sum(c.startswith("Not updated this run") for c in a["caveats"]) == 1 and a["caveats"][1:] == b["caveats"]
        assert a["carried_forward"]["built_by_run"] == "fixture" and a["carried_forward"]["built_at"]
        assert np.isfinite(R.read_grid(copy, a["grid"])).any()
    o = after["chl_outlook_8d"]
    assert f"showing the outlook issued {o['issued']} for {o['valid_from']}..{o['valid_to']}" in o["caveats"][0]
    assert any("last known good" in n for n in idx["notes"])
    rep = check_published(str(copy))
    assert all(c["ok"] for c in rep["checked"] if c["kind"].startswith("derived")), rep["problems"]


def test_an_ended_outlook_and_an_old_sst_are_dropped_not_carried(dataset: Path, tmp_path: Path):
    copy = tmp_path / "v1"
    shutil.copytree(dataset, copy)
    idx0 = json.loads((copy / "derived" / "index.json").read_text())
    out0 = next(lyr for lyr in idx0["layers"] if lyr["id"] == "chl_outlook_8d")
    later = (date.fromisoformat(out0["valid_to"]) + __import__("datetime").timedelta(days=1)).isoformat() + "T12:00:00Z"
    R.run(fixture_context(copy, now=later, fetcher=_refusing_fetcher()))
    idx = json.loads((copy / "derived" / "index.json").read_text())
    ids = [lyr["id"] for lyr in idx["layers"]]
    assert "chl_outlook_8d" not in ids
    assert any("has ended: not shown" in n for n in idx["notes"])
    # the SST analysis (2026-10-07) is still within 7 days on 2026-10-12: kept; after that dropped
    assert ("sst_latest" in ids) == ((date.fromisoformat(later[:10]) - date(2026, 10, 7)).days <= R.SST_CARRY_MAX_AGE_DAYS)
    R.run(fixture_context(copy, now="2026-10-20T12:00:00Z", fetcher=_refusing_fetcher()))
    idx = json.loads((copy / "derived" / "index.json").read_text())
    assert "sst_latest" not in [lyr["id"] for lyr in idx["layers"]]
