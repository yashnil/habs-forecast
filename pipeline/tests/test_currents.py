"""Observed HF-radar currents: real recorded ERDDAP responses (2026-10-07/08) for Monterey Bay,
the North Coast and the Southern California Bight. Dates, alignment, QC, gaps, the 24-hour
mean, reuse and outages."""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.http import FetchError
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.publish.check import check_published
from coastwatch_pipeline.sources import currents as cur
from coastwatch_pipeline.sources.charm import parse_netcdf
from coastwatch_pipeline.verify_currents import verify_currents

FIX = Path(__file__).parent / "fixtures" / "currents"


def run_region(tmp_path: Path, region: str, **kw):
    ctx = fixture_context(tmp_path / region / "v1", fetcher=FixtureFetcher(satellite_region=region, **kw))
    return ctx, cur.run(ctx, [])


def recorded(region: str):
    arrays, vattrs, _ = parse_netcdf((FIX / f"{region}.nc").read_bytes())
    t = [datetime.fromtimestamp(float(x), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") for x in arrays["time"]]
    u = np.asarray(arrays["water_u"], float)
    u[np.isclose(u, float(vattrs["water_u"]["_FillValue"]))] = np.nan
    return t, u, np.asarray(arrays["latitude"], float), np.asarray(arrays["longitude"], float)


@pytest.mark.parametrize("region", ["monterey", "socal_bight"])
def test_hourly_layers_carry_real_hours_and_upstream_values(tmp_path, region):
    ctx, res = run_region(tmp_path, region)
    assert not res.errors
    hourly = [lyr for lyr in res.layers if lyr.layer_id.startswith("hfr2km_currents_2")]
    assert len(hourly) == cur.WINDOW_HOURS
    times, u_up, lat_up, lon_up = recorded(region)
    for lyr in hourly:
        assert lyr.group_id == "currents" and lyr.product_class == "observation" and lyr.native_resolution_m == 2000
        assert lyr.time.valid_time in times and lyr.time.observed_times == [lyr.time.valid_time]
        assert lyr.vectors and lyr.vectors.u_grid.width == lon_up.size and lyr.vectors.u_grid.height == lat_up.size
    # every published u equals the upstream value at that hour and cell (within quantization)
    newest = hourly[-1]
    u, _ = cur.load_published_field(ctx, newest)
    k = times.index(newest.time.valid_time)
    tol = newest.vectors.u_grid.max_quantization_error + 1e-9
    both = np.isfinite(u) & np.isfinite(u_up[k])
    assert both.sum() > 100
    assert np.all(np.abs(u[both] - u_up[k][both]) <= tol)
    assert np.array_equal(np.isfinite(u), np.isfinite(u_up[k]))  # nothing filled, nothing lost (no cell failed QC here)
    assert newest.vectors.u_grid.lat_first == pytest.approx(lat_up[0], abs=1e-5)  # float32 axis precision


def test_window_is_the_newest_24_hours_even_though_erddap_snaps_the_start(tmp_path):
    _, res = run_region(tmp_path, "monterey")
    hourly = sorted(lyr.time.valid_time for lyr in res.layers if lyr.layer_id.startswith("hfr2km_currents_2"))
    assert hourly[-1] == "2026-10-08T12:00:00Z" and hourly[0] == "2026-10-07T13:00:00Z"
    assert res.latest_time == "2026-10-08T12:00:00Z"


def test_mean_uses_only_cells_with_enough_hours(tmp_path):
    ctx, res = run_region(tmp_path, "monterey")
    mean = next(lyr for lyr in res.layers if lyr.layer_id == "hfr2km_currents_mean24h")
    hourly = [lyr for lyr in res.layers if lyr.layer_id.startswith("hfr2km_currents_2")]
    U = np.stack([cur.load_published_field(ctx, h)[0] for h in hourly])
    n = np.isfinite(U).sum(axis=0)
    mu, _ = cur.load_published_field(ctx, mean)
    assert np.array_equal(np.isfinite(mu), n >= cur.MEAN_MIN_HOURS)
    ok = np.isfinite(mu)
    assert np.allclose(mu[ok], np.nanmean(U, axis=0)[ok], atol=2 * mean.vectors.u_grid.max_quantization_error)
    assert any("not a forecast" in c for c in mean.caveats)
    assert mean.time.observed_times == [hourly[0].time.valid_time, hourly[-1].time.valid_time]


def _synthetic_batch(ax: cur.Axes, times: list[str], sites=3.0, hdop=0.5, u=0.2, v=0.1, lat_shift=0.0) -> bytes:
    buf = io.BytesIO()
    f = netcdf_file(buf, "w", version=1)
    ny, nx = ax.lats.size, ax.lons.size
    for d, n in (("time", len(times)), ("latitude", ny), ("longitude", nx)):
        f.createDimension(d, n)
    tv = f.createVariable("time", "d", ("time",))
    tv[:] = [datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() for t in times]
    f.createVariable("latitude", "f", ("latitude",))[:] = ax.lats + lat_shift
    f.createVariable("longitude", "f", ("longitude",))[:] = ax.lons
    fields = {"water_u": u, "water_v": v, "hdop": hdop, "number_of_sites": sites}
    for name, val in fields.items():
        var = f.createVariable(name, "f", ("time", "latitude", "longitude"))
        var[:] = np.broadcast_to(np.asarray(val, dtype="f"), (len(times), ny, nx))
        var._FillValue = np.float32(-327.67)
        if name.startswith("water"):
            var.units = b"m s-1"
    f.flush()
    out = buf.getvalue()
    f.close()
    return out


def _axes() -> cur.Axes:
    lat = np.array([36.5, 36.518, 36.536, 36.554])
    lon = np.array([-122.0, -121.979, -121.958])
    return cur.Axes(lat, lon, 0, 3, 0, 2)


def test_quality_checks_drop_cells_and_count_them():
    ax, t = _axes(), ["2026-10-08T12:00:00Z"]
    sites = np.array([[[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]])
    hdop = np.array([[[0.5, 2.0, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]])
    u = np.array([[[0.2, 0.2, 3.0], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [-327.67, 0.2, 0.2]]])
    (h,) = cur.load_batch(_synthetic_batch(ax, t, sites=sites, hdop=hdop, u=u), ax, t, "test")
    assert np.isnan(h.u[0, 0]) and np.isnan(h.u[0, 1]) and np.isnan(h.u[0, 2]) and np.isnan(h.u[3, 0])
    assert int(np.isfinite(h.u).sum()) == 12 - 4
    detail = {c.name: c.detail for c in h.checks}
    assert detail["min_sites"].startswith("1 ") and detail["hdop"].startswith("1 ") and detail["speed_plausible"].startswith("1 ")


def test_misaligned_or_mistimed_responses_are_rejected():
    ax, t = _axes(), ["2026-10-08T12:00:00Z"]
    with pytest.raises(ValueError, match="latitude/longitude"):
        cur.load_batch(_synthetic_batch(ax, t, lat_shift=0.009), ax, t, "test")
    with pytest.raises(ValueError, match="times"):
        cur.load_batch(_synthetic_batch(ax, t), ax, ["2026-10-08T13:00:00Z"], "test")


def test_direction_convention_is_toward_clockwise_from_north():
    assert cur.direction_deg(np.array([0.0, 1.0, 0.0, -1.0]), np.array([1.0, 0.0, -1.0, 0.0])).tolist() == [0.0, 90.0, 180.0, 270.0]


def test_a_region_without_radar_coverage_publishes_nothing(tmp_path):
    # North Coast, 2026-10-07/08: the recorded field has no valid cell in any hour
    _, res = run_region(tmp_path, "north_coast")
    assert res.layers == [] and res.errors and "no valid HF-radar cell" in res.errors[0]
    assert any("without a single valid cell" in n for n in res.notes)


def test_older_hours_are_reused_and_only_recent_hours_fetched(tmp_path):
    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    fx = FixtureFetcher()
    m2 = run_pipeline(fixture_context(out, fetcher=fx))
    nc = [u for u in fx.calls if "ucsdHfrW2.nc" in u]
    assert len(nc) == 1 and "2026-10-08T07:00:00Z" in nc[0]  # only the newest REFETCH_HOURS hours
    st = next(s for s in m2.sources if s.source_id == "hf_radar")
    assert st.outcome == "updated" and any("reused" in n for n in st.notes)
    assert check_published(str(out))["ok"]


def test_outage_keeps_previous_currents_with_their_real_times(tmp_path):
    out = tmp_path / "v1"
    m1 = run_pipeline(fixture_context(out))
    refuse = lambda url: (_ for _ in ()).throw(FetchError(f"Failed after 4 attempts: {url} (HTTP Error 403: )"))  # noqa: E731
    m2 = run_pipeline(fixture_context(out, "2026-10-09T18:00:00Z", FixtureFetcher(overrides={r"ucsdHfrW2": refuse})))
    old = {lyr.layer_id: lyr.time.valid_time for lyr in m1.layers if lyr.group_id == "currents"}
    new = {lyr.layer_id: lyr.time.valid_time for lyr in m2.layers if lyr.group_id == "currents"}
    assert new == old and len(new) == 25
    st = next(s for s in m2.sources if s.source_id == "hf_radar")
    assert st.outcome == "failed" and st.latest_valid_date == "2026-10-08" and "403" in (st.error or "")
    assert check_published(str(out))["ok"]


def test_offline_verification_passes_and_freshness_is_stated(tmp_path):
    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    rep = verify_currents(out, live=False)
    assert rep["summary"]["all_passed"] and any(r["kind"] == "mean" for r in rep["rows"])
    assert cur.FRESHNESS.basis == "observed_date" and cur.FRESHNESS.current_max_age_days == 1
    assert any("not a forecast" in c or "nothing here is a forecast" in c for c in cur.CAVEATS)


def test_fixtures_are_genuine_erddap_files():
    for region in ("monterey", "north_coast", "socal_bight"):
        body = (FIX / f"{region}.nc").read_bytes()
        assert body.startswith(b"CDF")
        _, _, g = parse_netcdf(body)
        assert "HFRNet" in " ".join(g.values()) or "hfrnet" in " ".join(g.values()).lower()

