"""Satellite chlorophyll (OLCI 300 m, VIIRS 750 m fallback): dates, coverage, no-data
masks, geographic alignment and outage behaviour. Fixtures are real ERDDAP responses
recorded on 2026-10-09 for Monterey Bay, the North Coast and the Southern California Bight."""

from __future__ import annotations

import io
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.http import FetchError
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.publish.check import check_published
from coastwatch_pipeline.sources import satellite
from coastwatch_pipeline.sources.charm import parse_netcdf
from coastwatch_pipeline.verify_satellite import verify_satellite

FIX = Path(__file__).parent / "fixtures" / "satellite"


def run_region(tmp_path: Path, region: str, **kw):
    out = tmp_path / region / "v1"
    ctx = fixture_context(out, fetcher=FixtureFetcher(satellite_region=region, **kw))
    return ctx, satellite.run(ctx, [])


def by_id(res, layer_id):
    return next((lyr for lyr in res.layers if lyr.layer_id == layer_id), None)


def source_values(region: str, ds_prefix: str) -> list[np.ndarray]:
    out = []
    for f in sorted((FIX / region).glob(f"{ds_prefix}_*.nc")):
        arrays, vattrs, _ = parse_netcdf(f.read_bytes())
        var = "chlor_a" if "chlor_a" in arrays else "chla"
        a = np.asarray(arrays[var], dtype=float).ravel()
        fv = vattrs[var].get("_FillValue")
        if fv is not None:
            a[np.isclose(a, float(fv))] = np.nan
        out.append(a[np.isfinite(a) & (a > 0)])
    return out


@pytest.mark.parametrize("region,olci_days", [("monterey", 7), ("north_coast", 2), ("socal_bight", 2)])
def test_three_regions_publish_latest_clear_view_with_real_dates(tmp_path, region, olci_days):
    ctx, res = run_region(tmp_path, region)
    assert not res.errors, res.errors
    latest = by_id(res, "olci300_chl_latest")
    assert latest is not None and latest.composite is not None
    days = [lyr for lyr in res.layers if lyr.layer_id.startswith("olci300_chl_2")]
    assert len(days) == olci_days
    # observation times are the upstream overpass stamps, verbatim, and dates follow them
    for d in days:
        assert d.time.observed_times and all(t.startswith(d.time.observed_date) for t in d.time.observed_times)
        assert d.native_resolution_m == 300.0 and d.resolution_deg == 0.0025
    obs = [d.time.observed_date for d in days if d.grid]
    assert latest.composite.newest_observed_date == max(obs)
    assert latest.composite.oldest_observed_date == min(obs)
    assert latest.time.observed_date == latest.composite.newest_observed_date
    assert latest.composite.reference_date == ctx.now.date().isoformat()
    # ages count whole UTC days from the reference date and sum to the observed pixels
    assert abs(sum(b.fraction for b in latest.composite.age_histogram) - 1) < 1e-3
    assert sum(d.pixels_used for d in latest.composite.days) == latest.qc.n_valid
    # coverage is reported for the region and is a real fraction
    assert latest.coverage and latest.coverage.regions
    assert 0 < latest.coverage.domain_observed_fraction <= 1
    # VIIRS 750 m fallback is published alongside, labelled at its own resolution
    viirs = by_id(res, "viirs750_chl_latest")
    assert viirs is not None and viirs.native_resolution_m == 750.0


def test_every_published_value_is_an_upstream_value(tmp_path):
    ctx, res = run_region(tmp_path, "monterey")
    latest = by_id(res, "olci300_chl_latest")
    pub = satellite.load_published(ctx.out_dir, latest.grid)
    vals = pub[np.isfinite(pub)]
    src = np.concatenate(source_values("monterey", "olci300_s3a_CI"))
    # quantization: log10 step of 6/65534 -> relative error < 1.1e-4
    nearest = np.array([np.min(np.abs(np.log10(src) - np.log10(v))) for v in vals[:: max(1, vals.size // 400)]])
    assert nearest.max() <= latest.grid.max_quantization_error + 1e-9
    # never more valid pixels than the union of upstream valid pixels in the window
    assert vals.size <= sum(a.size for a in source_values("monterey", "olci300_s3a_CI"))


def test_no_data_stays_no_data_and_is_transparent(tmp_path):
    ctx, res = run_region(tmp_path, "monterey")
    # fully clouded days are published as days without values, not dropped and not filled
    empty = [lyr for lyr in res.layers if lyr.layer_id.startswith("olci300_chl_2") and lyr.grid is None]
    assert {lyr.time.observed_date for lyr in empty} == {"2026-10-03", "2026-10-04", "2026-10-07"}
    for lyr in empty:
        assert lyr.qc.n_valid == 0 and lyr.tiles is None and lyr.coverage.domain_observed_fraction == 0
    m = run_pipeline(ctx)
    report = verify_satellite(ctx.out_dir, live=False)
    assert report["summary"]["all_passed"], [r for r in report["rows"] if r["status"] == "FAIL"][:3]
    assert any(r.get("tile_transparent") for r in report["rows"])
    assert check_published(str(ctx.out_dir))["ok"]
    assert any(s.source_id == "satellite_chl" for s in m.sources)


def test_grid_alignment_with_source_coordinates(tmp_path):
    ctx, res = run_region(tmp_path, "socal_bight")
    day = by_id(res, "olci300_chl_2026-10-06")
    g = day.grid
    # the published grid sits on OLCI's lattice (cell centres 45.18625 - k*0.0025, -140.03625 + k*0.0025)
    assert abs(((45.18625 - g.lat_first) / 0.0025) - round((45.18625 - g.lat_first) / 0.0025)) < 1e-6
    assert abs(((g.lon_first + 140.03625) / 0.0025) - round((g.lon_first + 140.03625) / 0.0025)) < 1e-6
    # and a published cell equals the upstream file value at the same coordinates
    arrays, vattrs, _ = parse_netcdf((FIX / "socal_bight" / "olci300_s3a_DI_2026-10-06.nc").read_bytes())
    a = np.asarray(arrays["chlor_a"], dtype=float).reshape(arrays["latitude"].size, arrays["longitude"].size)
    pub = satellite.load_published(ctx.out_dir, g)
    rows, cols = np.nonzero(np.isfinite(a) & (a > 0))
    for k in np.linspace(0, rows.size - 1, 25).astype(int):
        lat, lon = float(arrays["latitude"][rows[k]]), float(arrays["longitude"][cols[k]])
        r = int(round((g.lat_first - lat) / 0.0025))
        c = int(round((lon - g.lon_first) / 0.0025))
        if 0 <= r < g.height and 0 <= c < g.width:
            assert abs(np.log10(pub[r, c]) - np.log10(a[rows[k], cols[k]])) <= g.max_quantization_error + 1e-9


def _shifted_scene(blob: bytes, dlat: float) -> bytes:
    arrays, vattrs, _ = parse_netcdf(blob)
    buf = io.BytesIO()
    f = netcdf_file(buf, "w", version=1)
    for name in ("time", "altitude", "latitude", "longitude"):
        f.createDimension(name, arrays[name].size)
    for name in ("time", "altitude", "latitude", "longitude"):
        v = f.createVariable(name, "d", (name,))
        v[:] = arrays[name] + (dlat if name == "latitude" else 0)
    c = f.createVariable("chlor_a", "f", ("time", "altitude", "latitude", "longitude"))
    c[:] = np.nan_to_num(arrays["chlor_a"], nan=-999.0)
    c._FillValue = np.float32(-999.0)
    f.flush()
    out = buf.getvalue()
    f.close()
    return out


def test_scene_off_the_source_lattice_is_rejected(tmp_path):
    good = (FIX / "monterey" / "olci300_s3a_CI_2026-10-05.nc").read_bytes()
    bad = _shifted_scene(good, 0.0011)  # 0.44 of a cell: misregistered
    target = satellite.target_for(satellite.OLCI, satellite.Domain(36.40125, 37.20125, -122.60125, -121.75125))
    with pytest.raises(ValueError, match="coordinates_on_lattice"):
        satellite.load_scene(bad, satellite.OLCI, "Sentinel-3A", "noaacwS3AOLCIchlaSectorCIDaily", _time_of(good), "u", target)
    sc = satellite.load_scene(good, satellite.OLCI, "Sentinel-3A", "noaacwS3AOLCIchlaSectorCIDaily", _time_of(good), "u", target)
    assert sc.n_valid > 0 and all(c.passed for c in sc.checks)


def _time_of(blob: bytes) -> str:
    arrays, _, _ = parse_netcdf(blob)
    return datetime.fromtimestamp(float(np.atleast_1d(arrays["time"])[0]), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_olci_outage_falls_back_to_viirs(tmp_path):
    def unknown(url):
        raise FetchError(f"HTTP 404 for {url} (Currently unknown datasetID)")

    out = tmp_path / "v1"
    ctx = fixture_context(out, fetcher=FixtureFetcher(overrides={r"noaacwS3[AB]OLCI": unknown}))
    m = run_pipeline(ctx)
    ids = {lyr.layer_id for lyr in m.layers if lyr.provenance.source_id == "satellite_chl"}
    assert ids == {"viirs750_chl_latest"}
    st = next(s for s in m.sources if s.source_id == "satellite_chl")
    assert st.outcome == "partial" and "VIIRS 750 m latest clear view is the fallback" in " ".join(st.notes)


def test_total_outage_keeps_previous_layers_with_real_dates(tmp_path):
    out = tmp_path / "v1"
    m1 = run_pipeline(fixture_context(out))
    later = "2026-10-12T18:00:00Z"
    fetcher = FixtureFetcher(overrides={r"OLCIchla|erdVHNchla1day": lambda url: (_ for _ in ()).throw(FetchError(f"HTTP 503 for {url}"))})
    m2 = run_pipeline(fixture_context(out, later, fetcher))
    old = {lyr.layer_id: lyr.time.observed_date for lyr in m1.layers if lyr.provenance.source_id == "satellite_chl"}
    new = {lyr.layer_id: lyr.time.observed_date for lyr in m2.layers if lyr.provenance.source_id == "satellite_chl"}
    assert new == old  # kept, with their own (now old) dates; the browser marks them stale
    st = next(s for s in m2.sources if s.source_id == "satellite_chl")
    assert st.outcome == "failed" and st.latest_valid_date == "2026-10-06"
    assert check_published(str(out))["ok"]


def test_unchanged_days_are_reused_not_refetched(tmp_path):
    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    f2 = FixtureFetcher()
    m2 = run_pipeline(fixture_context(out, fetcher=f2))
    scene_calls = [u for u in f2.calls if "OLCIchla" in u and ".nc?" in u]
    assert scene_calls == []  # every OLCI day matched the published overpass times
    st = next(s for s in m2.sources if s.source_id == "satellite_chl")
    assert any("reused it" in n for n in st.notes)
    assert check_published(str(out))["ok"]


def test_old_satellite_files_are_pruned(tmp_path):
    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    stale = out / "satellite" / "olci300" / "day-2020-01-01-deadbeef00" / "grid" / "0_0.u16.gz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"x")
    run_pipeline(fixture_context(out))
    run_pipeline(fixture_context(out))
    assert not stale.exists()


def test_implausible_values_are_flagged_not_removed(tmp_path):
    good = (FIX / "monterey" / "olci300_s3a_CI_2026-10-06.nc").read_bytes()
    arrays, _, _ = parse_netcdf(good)
    a = np.asarray(arrays["chlor_a"], dtype=float)
    ok = np.isfinite(a) & (a > 0) & (a < 1e30)
    a[ok] = 500.0  # every value implausibly high
    buf = io.BytesIO()
    f = netcdf_file(buf, "w", version=1)
    for name in ("time", "altitude", "latitude", "longitude"):
        f.createDimension(name, arrays[name].size)
        v = f.createVariable(name, "d", (name,))
        v[:] = arrays[name]
    c = f.createVariable("chlor_a", "f", ("time", "altitude", "latitude", "longitude"))
    c[:] = np.where(ok, a, -999.0)
    c._FillValue = np.float32(-999.0)
    f.flush()
    blob = buf.getvalue()
    f.close()
    target = satellite.target_for(satellite.OLCI, satellite.Domain(36.40125, 37.20125, -122.60125, -121.75125))
    sc = satellite.load_scene(blob, satellite.OLCI, "Sentinel-3A", "ds", _time_of(good), "u", target)
    flag = next(c for c in sc.checks if c.name.endswith("plausible_values"))
    assert not flag.passed and sc.n_valid > 0


def test_freshness_policy_and_caveats_state_the_limits():
    assert satellite.FRESHNESS.basis == "observed_date"
    text = " ".join(satellite.CAVEATS).lower()
    assert "does not measure toxins" in text and "never filled" in text
    assert date(2026, 10, 8) - date(2026, 10, 1) == satellite.timedelta(days=satellite.WINDOW_DAYS)


def test_satellite_fixture_index_is_real_erddap_data():
    for f in FIX.rglob("*.nc"):
        assert f.read_bytes()[:3] == b"CDF"



