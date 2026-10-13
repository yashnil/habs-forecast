"""VIIRS 750 m after Suomi-NPP: NOAA-21 first, NOAA-20 only where NOAA-21 has no value, one
platform per cell (recorded per cell), S-NPP only as a labelled fallback, and the cut-off
(docs/coastwatch/33-suomi-npp-continuity.md). Fixtures are real NOAA-21/NOAA-20 ERDDAP
responses recorded 2026-10-13 (scripts/record_jpss_fixtures.py)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from coastwatch_pipeline import health
from coastwatch_pipeline.fixtures import FIXTURES, FixtureFetcher, fixture_context
from coastwatch_pipeline.http import FetchError
from coastwatch_pipeline.models import PortIntelCollection
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.publish.check import check_published
from coastwatch_pipeline.sources import port_intel
from coastwatch_pipeline.sources import satellite as sat
from coastwatch_pipeline.verify_satellite import verify_satellite

N21, N20, SNPP = "noaacwN21VIIRSchlaSectorUSDaily", "noaacwN20VIIRSchlaSectorUSDaily", "erdVHNchla1day"


def fail(status: int = 503):
    def f(url):
        raise FetchError(f"HTTP {status} for {url}")

    return f


def nothing_newer(url):
    # what ERDDAP answers once a dataset has no time after the requested start
    if ".csv0?" in url:
        raise FetchError(f"HTTP 404 for {url}", detail='Error { code=404; message="Not Found: Your query produced no matching results."; }')
    raise FetchError(f"HTTP 404 for {url}")


def viirs(res_or_manifest):
    layers = res_or_manifest.layers
    return next((lyr for lyr in layers if lyr.layer_id == "viirs750_chl_latest"), None)


def run_sat(tmp_path: Path, region: str = "monterey", now: str | None = None, **overrides):
    ctx = fixture_context(tmp_path / region / "v1", **({"now": now} if now else {}), fetcher=FixtureFetcher(satellite_region=region, overrides=overrides or None))
    return ctx, sat.run(ctx, [])


def scenes_for(region: str, ds: str, target: sat.Target) -> dict[str, sat.Scene]:
    idx = json.loads((FIXTURES / "satellite" / "index.json").read_text())
    plat = "NOAA-21" if ds == N21 else "NOAA-20"
    out = {}
    for e in idx[region][ds]:
        blob = (FIXTURES / "satellite" / e["file"]).read_bytes()
        out[e["time"][:10]] = sat.load_scene(blob, sat.VIIRS_JPSS, plat, ds, e["time"], "u", target)
    return out


@pytest.mark.parametrize("region", ["monterey", "north_coast", "socal_bight"])
def test_noaa21_first_noaa20_only_where_noaa21_has_no_value(tmp_path, region):
    ctx, res = run_sat(tmp_path, region)
    lyr = viirs(res)
    assert lyr is not None and not [e for e in res.errors if "VIIRS" in e or "NOAA" in e]
    # labelled by sensor, then the satellites actually used, in priority order
    assert lyr.platforms == ["VIIRS", "NOAA-21", "NOAA-20"]
    assert lyr.provenance.dataset_id == f"{N21},{N20}" and "coastwatch.noaa.gov" in lyr.provenance.source_url
    md = lyr.provenance.upstream_metadata
    assert md["processing"] == sat.PROCESSING_VERSION == "satellite-processing-3"
    assert "-0.020" in md["offsets"] and "not corrected" in md["offsets"]
    assert lyr.native_resolution_m == 750.0 and lyr.resolution_deg == 0.0075
    # overpass-time stamps, verbatim, ~19 UTC on the observation date
    assert all(t[11:13] in ("18", "19", "20") for t in lyr.time.observed_times)
    # the grid sits on the erdVHNchla1day lattice (cell 42.49875 N, -127.00125 E)
    g = lyr.grid
    assert abs((42.49875 - g.lat_first) / 0.0075 - round((42.49875 - g.lat_first) / 0.0075)) < 1e-6
    assert abs((g.lon_first + 127.00125) / 0.0075 - round((g.lon_first + 127.00125) / 0.0075)) < 1e-6

    vals = sat.load_published(ctx.out_dir, g)
    age = sat.load_published(ctx.out_dir, lyr.composite.age_grid)
    codes, order = sat.load_platform_grid(ctx.out_dir, lyr)
    assert order == ["NOAA-21", "NOAA-20"]
    # every value has exactly one platform; cells without a value have none
    assert ((codes > 0) == np.isfinite(vals)).all()
    counts = {k: int(v) for k, v in (x.rsplit(" ", 1) for x in md["pixels_by_platform"].split(", "))}
    assert counts["NOAA-21"] == int((codes == 1).sum()) and counts["NOAA-20"] == int((codes == 2).sum())
    assert counts["NOAA-21"] + counts["NOAA-20"] == lyr.qc.n_valid

    # independent recomputation from the recorded upstream files
    target = sat.Target(g.lat_first, g.lon_first, 0.0075, g.height, g.width)
    s21, s20 = scenes_for(region, N21, target), scenes_for(region, N20, target)
    ref = datetime.fromisoformat(lyr.composite.reference_date).date()
    rr, cc = np.nonzero(np.isfinite(vals))
    assert rr.size > 50
    tol = g.max_quantization_error + 1e-9
    n20_cells = 0
    for r, c in zip(rr, cc):
        d = (ref - sat.timedelta(days=int(age[r, c]))).isoformat()
        a21 = s21[d].values[r, c] if d in s21 else np.nan
        if np.isfinite(a21):
            assert codes[r, c] == 1 and abs(np.log10(vals[r, c]) - np.log10(a21)) <= tol
        else:
            n20_cells += 1
            assert codes[r, c] == 2 and abs(np.log10(vals[r, c]) - np.log10(s20[d].values[r, c])) <= tol
    assert n20_cells == counts["NOAA-20"]


def test_noaa20_fills_cells_on_monterey_and_the_files_are_kept_and_checked(tmp_path):
    out = tmp_path / "v1"
    m = run_pipeline(fixture_context(out))
    lyr = viirs(m)
    md = lyr.provenance.upstream_metadata
    assert "NOAA-20 0" not in md["pixels_by_platform"]  # the fixture days do need NOAA-20
    prefix = md["platform_grid"].split("{row}")[0]
    files = list((out / prefix).glob("*.u16.gz"))
    assert files and len(files) == len(lyr.grid.chunks.present)
    rep = check_published(str(out))
    assert rep["ok"] and any(r["kind"] == "platform_grid" and r["ok"] for r in rep["checked"])
    # a later run keeps the files a still-served manifest references, then prunes them
    run_pipeline(fixture_context(out))
    assert all(f.exists() for f in files)
    # the S-NPP fallback was not asked while NOAA-21/NOAA-20 answered
    f = FixtureFetcher()
    run_pipeline(fixture_context(out, fetcher=f))
    assert not [u for u in f.calls if SNPP in u]
    # verification checks the layer against its own datasets and reports the platform
    rows = [r for r in verify_satellite(out, live=False)["rows"] if r["layer_id"] == "viirs750_chl_latest" and r.get("value") is not None]
    assert rows and {r["platform"] for r in rows} <= {"NOAA-21", "NOAA-20"}


def test_noaa21_fails_alone_noaa20_carries_the_layer(tmp_path):
    _, res = run_sat(tmp_path, **{N21: fail(503)})
    lyr = viirs(res)
    assert lyr.platforms == ["VIIRS", "NOAA-20"] and lyr.provenance.dataset_id == N20
    assert "NOAA-21 0" in lyr.provenance.upstream_metadata["pixels_by_platform"]
    assert any(N21 in e for e in res.errors)  # partial: the source status says what failed


def test_noaa20_fails_alone_noaa21_carries_the_layer(tmp_path):
    _, res = run_sat(tmp_path, **{N20: fail(403)})
    lyr = viirs(res)
    assert lyr.platforms == ["VIIRS", "NOAA-21"] and lyr.provenance.dataset_id == N21
    assert any(N20 in e for e in res.errors)


def test_both_fail_before_the_cutoff_the_snpp_composite_is_the_labelled_fallback(tmp_path):
    _, res = run_sat(tmp_path, **{r"noaacwN2[01]VIIRS": fail(503)})
    lyr = viirs(res)
    assert lyr.platforms == ["VIIRS", "S-NPP"] and lyr.provenance.dataset_id == SNPP
    assert "Suomi-NPP" in lyr.provenance.source_name and "platform_grid" not in lyr.provenance.upstream_metadata
    assert any("S-NPP composite (erdVHNchla1day), labelled S-NPP, is the fallback" in n for n in res.notes)


def test_after_the_cutoff_snpp_has_nothing_newer_and_noaa_carries_on(tmp_path):
    # S-NPP's listing answers "no matching results" from now on: it is never needed
    _, res = run_sat(tmp_path, **{SNPP: nothing_newer})
    assert viirs(res).platforms == ["VIIRS", "NOAA-21", "NOAA-20"] and not res.errors


def test_after_the_cutoff_both_noaa_fail_last_good_is_kept_and_health_alerts(tmp_path):
    out = tmp_path / "v1"
    m1 = run_pipeline(fixture_context(out))
    prev = viirs(m1)
    fx = FixtureFetcher(overrides={r"noaacwN2[01]VIIRS": fail(503), SNPP: nothing_newer})
    m2 = run_pipeline(fixture_context(out, fetcher=fx))
    kept = viirs(m2)
    assert kept.model_dump() == prev.model_dump()  # same files, same real dates and platforms
    st = next(s for s in m2.sources if s.source_id == "satellite_chl")
    assert st.outcome == "partial" and N21 in st.error and "S-NPP fallback" in st.error
    assert any("kept the previously published layers" in n for n in st.notes)
    assert check_published(str(out))["ok"]
    # three such runs in a row: the existing degraded-source alert, as for any partial outage
    h = None
    for _ in range(3):
        h = health.evaluate(m2, h, datetime(2026, 10, 8, 18, tzinfo=timezone.utc))
    assert "source_degraded:satellite_chl" in {a["key"] for a in h["alerts"]}


def test_snpp_fallback_never_replaces_a_newer_noaa_view(tmp_path):
    out = tmp_path / "v1"
    m1 = run_pipeline(fixture_context(out))
    prev = viirs(m1)
    m2 = run_pipeline(fixture_context(out, fetcher=FixtureFetcher(overrides={r"noaacwN2[01]VIIRS": fail(503)})))
    kept = viirs(m2)
    # S-NPP's newest fixture day (10-04) is not newer than NOAA's (10-04): NOAA's view is kept
    assert kept.provenance.dataset_id == prev.provenance.dataset_id and kept.time == prev.time
    st = next(s for s in m2.sources if s.source_id == "satellite_chl")
    assert any("is not newer than the published NOAA-21/NOAA-20 view" in n for n in st.notes)


def test_total_outage_keeps_last_good_and_fails_the_source(tmp_path):
    out = tmp_path / "v1"
    m1 = run_pipeline(fixture_context(out))
    fx = FixtureFetcher(overrides={r"OLCIchla|noaacwN2[01]VIIRS|erdVHNchla1day": fail(503)})
    m2 = run_pipeline(fixture_context(out, fetcher=fx))
    assert viirs(m2).model_dump() == viirs(m1).model_dump()
    assert next(s for s in m2.sources if s.source_id == "satellite_chl").outcome == "failed"


def test_published_snpp_artifacts_keep_their_provenance(tmp_path):
    """A layer published before the migration (S-NPP, platforms ["VIIRS"]) is carried with
    its own dataset id and dates, and verified against S-NPP, not NOAA-21/NOAA-20."""
    out = tmp_path / "v1"
    m1 = run_pipeline(fixture_context(out, fetcher=FixtureFetcher(overrides={r"noaacwN2[01]VIIRS": fail(503)})))
    old = viirs(m1)
    assert old.provenance.dataset_id == SNPP
    assert sat.product_for(old.provenance.dataset_id) is sat.VIIRS_SNPP
    assert sat.product_for(f"{N21},{N20}") is sat.VIIRS_JPSS
    assert sat.product_for("noaacwS3AOLCIchlaSectorCIDaily,noaacwS3BOLCIchlaSectorCIDaily") is sat.OLCI
    # a total outage later: the S-NPP layer stays as it was
    m2 = run_pipeline(fixture_context(out, fetcher=FixtureFetcher(overrides={r"OLCIchla|noaacwN2[01]VIIRS|erdVHNchla1day": fail(503)})))
    assert viirs(m2).model_dump() == old.model_dump()


def test_viirs_health_threshold_follows_noaa_latency(tmp_path):
    assert health.VIIRS_MAX_AGE_DAYS == 10
    m = run_pipeline(fixture_context(tmp_path / "v1"))
    obs = datetime.fromisoformat(viirs(m).time.observed_date).date()

    def keys(days: int) -> set[str]:
        now = datetime(obs.year, obs.month, obs.day, 12, tzinfo=timezone.utc) + sat.timedelta(days=days)
        # OLCI kept fresh: alerts are one per source, and OLCI's own would take the slot
        fresh = lambda lyr: lyr.model_copy(update={"time": lyr.time.model_copy(update={"observed_date": now.date().isoformat()})})  # noqa: E731
        only = m.model_copy(update={"layers": [fresh(lyr) if lyr.layer_id == "olci300_chl_latest" else lyr for lyr in m.layers]})
        return {a["key"] for a in health.evaluate(only, None, now)["alerts"]}

    assert "satellite_stale:viirs750_chl_latest" not in keys(8)  # NOAA-20/21 were 8 days behind on 2026-10-13
    assert "satellite_stale:viirs750_chl_latest" not in keys(10)
    assert "satellite_stale:viirs750_chl_latest" in keys(11)


# ---------------------------------------------------------------- port chlorophyll
def _ports(out, fetcher=None):
    m = run_pipeline(fixture_context(out, fetcher=fetcher) if fetcher else fixture_context(out))
    return m, PortIntelCollection.model_validate_json((out / m.port_intel_url).read_text())


def _port(coll, name):
    return next(p for p in coll.ports if p.display_name == name)


def test_port_window_median_from_daily_noaa21_noaa20_pixels(tmp_path):
    _, coll = _ports(tmp_path / "v1")
    chl = _port(coll, "Monterey").chlorophyll
    assert chl.error is None and chl.dataset_id == f"{N21},{N20}" and chl.composite_days == 8
    assert chl.latest and chl.latest.n > 0 and 0 < chl.latest_valid_fraction <= 1
    dates = [p.date for p in chl.history]
    assert dates == sorted(dates) and len(set(dates)) == len(dates) and len(dates) <= port_intel.CHL_DAYS
    # independent recomputation of the latest window from the recorded responses
    idx = json.loads((FIXTURES / "recorded" / "index.json").read_text())
    p = _port(coll, "Monterey")
    daily = []
    for ds in (N21, N20):
        url = next(u for u in idx if ds in u and "(36.7450)" in u)
        arrays, vattrs, _ = sat.parse_netcdf((FIXTURES / "recorded" / idx[url]["file"]).read_bytes())
        a = np.asarray(arrays["chlor_a"], float)[:, 0]
        a[np.isclose(a, -999.0) | (a <= 0) | (a > 200)] = np.nan
        days = [datetime.fromtimestamp(float(t), tz=timezone.utc).date().isoformat() for t in arrays["time"]]
        daily.append(dict(zip(days, a)))
        LA, LO = np.meshgrid(arrays["latitude"], arrays["longitude"], indexing="ij")
    within = port_intel.km(p.lat, p.lon, LA, LO) <= 15
    last = max(max(d) for d in daily)
    centre = datetime.fromisoformat(chl.history[-1].date).date()
    assert (centre + sat.timedelta(days=4)).isoformat() <= last  # complete windows only
    stack = []
    for k in range(-3, 5):
        d = (datetime.fromisoformat(chl.latest_center_date).date() + sat.timedelta(days=k)).isoformat()
        n21, n20 = daily[0].get(d), daily[1].get(d)
        if n21 is None and n20 is None:
            continue
        stack.append(np.where(np.isfinite(n21), n21, n20) if n21 is not None and n20 is not None else (n21 if n21 is not None else n20))
    per_pixel = np.nanmedian(np.stack(stack), axis=0)
    v = per_pixel[within]
    v = v[np.isfinite(v)]
    assert chl.latest.n == v.size and chl.latest.median == pytest.approx(float(np.median(v)), abs=1e-6)
    # provenance marks the change point and the measured offset; the method text says it too
    prov = next(x for x in coll.provenance if x.source_id == port_intel.CHL_SOURCE_ID)
    assert "erdVHNchla8day" in prov.upstream_metadata["change_point"] and "-0.020" in prov.upstream_metadata["offsets"]
    assert "about 4% lower" in coll.method["chlorophyll_source_change"]
    assert not any(x.source_id == "viirs_chl_8day" for x in coll.provenance)  # S-NPP not used


def test_port_noaa21_fails_alone(tmp_path):
    _, coll = _ports(tmp_path / "v1", FixtureFetcher(overrides={rf"{N21}\.nc\?chlor_a": fail(503)}))
    chl = _port(coll, "Monterey").chlorophyll
    assert chl.dataset_id == N20 and chl.latest and chl.error is None


def test_port_both_fail_uses_the_snpp_8day_fallback_then_says_unavailable(tmp_path):
    _, coll = _ports(tmp_path / "v1", FixtureFetcher(overrides={r"noaacwN2[01]VIIRSchlaSectorUSDaily\.nc\?chlor_a": fail(503)}))
    chl = _port(coll, "Monterey").chlorophyll
    assert chl.dataset_id == "erdVHNchla8day" and chl.latest and chl.error is None
    assert any(x.source_id == "viirs_chl_8day" for x in coll.provenance)
    # after the cut-off nothing answers: an error, never a zero
    _, coll = _ports(tmp_path / "v2", FixtureFetcher(overrides={r"noaacwN2[01]VIIRSchlaSectorUSDaily\.nc\?chlor_a|erdVHNchla8day": fail(503)}))
    chl = _port(coll, "Monterey").chlorophyll
    assert chl.latest is None and "NOAA-21/NOAA-20" in chl.error and "S-NPP fallback" in chl.error
    assert chl.dataset_id == f"{N21},{N20}" and not chl.history


def test_port_history_reuses_complete_windows_and_fetches_two_weeks(tmp_path):
    """ERDDAP takes 10-16 s per port and platform for 64 days; a later run keeps the complete
    windows of the previous summary and asks only for the recent days, with the same result."""
    out = tmp_path / "v1"
    _, first = _ports(out)
    f = FixtureFetcher()
    _, second = _ports(out, f)
    a, b = _port(first, "Monterey").chlorophyll, _port(second, "Monterey").chlorophyll
    assert b.history == a.history and b.latest == a.latest and b.latest_center_date == a.latest_center_date
    asked = [u for u in f.calls if N21 in u and "(36.7450)" in u]
    start = asked[0].split("%5B(")[1][:10]
    newest = datetime.fromisoformat(a.history[-1].date).date()
    assert start == (newest - sat.timedelta(days=port_intel.CHL_REFRESH_DAYS + 2)).isoformat()
    # a summary made from one platform is not reused as a two-platform one
    one = a.model_copy(update={"dataset_id": N20})
    chl, _ = port_intel.chlorophyll_jpss(fixture_context(tmp_path / "x"), _port(first, "Monterey").lat, _port(first, "Monterey").lon, datetime(2026, 10, 8).date(), one)
    assert chl.history == a.history  # full fetch, same values


def test_noaa_stays_primary_when_it_lags_more_than_a_week(tmp_path):
    """Staging, 2026-10-13: NOAA-21/NOAA-20 newest 10-05, S-NPP newest 10-08. With a 7-day
    window NOAA had no pass in reach and the S-NPP fallback became the layer. The VIIRS window
    covers NOAA's latency, so NOAA-21 stays primary and S-NPP is not even asked."""
    f = FixtureFetcher()
    ctx = fixture_context(tmp_path / "v1", "2026-10-12T18:00:00Z", f)  # fixture NOAA days 10-01..04: 8-11 days old
    res = sat.run(ctx, [])
    lyr = viirs(res)
    assert lyr.platforms[:2] == ["VIIRS", "NOAA-21"] and lyr.provenance.dataset_id.startswith(N21)
    assert lyr.composite.window_days == sat.VIIRS_WINDOW_DAYS == 10
    assert lyr.composite.oldest_observed_date >= "2026-10-02"
    assert max(b.age_days for b in lyr.composite.age_histogram) <= 10
    assert abs(sum(b.fraction for b in lyr.composite.age_histogram) - 1) < 1e-3
    assert not [u for u in f.calls if SNPP in u]
    # OLCI keeps its own 7-day window
    olci = next(x for x in res.layers if x.layer_id == "olci300_chl_latest")
    assert olci.composite.window_days == sat.WINDOW_DAYS == 7
