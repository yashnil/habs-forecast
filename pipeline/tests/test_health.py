"""Pipeline health alerts: real problems alert, ordinary source behaviour does not."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from coastwatch_pipeline import health as hl
from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.http import FetchError
from coastwatch_pipeline.models import Manifest
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.publish.check import guard_publish_tree

NOW = datetime(2026, 10, 8, 18, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def base(tmp_path_factory) -> Manifest:
    return run_pipeline(fixture_context(tmp_path_factory.mktemp("h") / "v1"))


def keys(h):
    return {a["key"] for a in h["alerts"]}


def with_status(m: Manifest, source_id: str, outcome: str) -> Manifest:
    return m.model_copy(update={"sources": [s.model_copy(update={"outcome": outcome, "error": "HTTP Error 403" if outcome == "failed" else None}) if s.source_id == source_id else s for s in m.sources]})


# The satellite fixtures were recorded 2026-10-01..07, after Sentinel-3B stopped (its last CI scene
# is 2026-10-01): they hold Sentinel-3A only, so the one alert on them is that real outage.
S3B_SILENT = "platform_silent:olci300:Sentinel-3B"


def test_healthy_fixture_has_no_alerts(base):
    h = hl.evaluate(base, None, NOW)
    assert keys(h) == {S3B_SILENT} and h["new_alerts"] == [S3B_SILENT]


def test_a_noaa_source_alerts_only_after_consecutive_failures(base):
    failing = with_status(base, "charm", "failed")
    h = None
    for i in range(1, hl.MIN_CONSECUTIVE_FAILURES + 1):
        h = hl.evaluate(failing, h, NOW + timedelta(hours=6 * i))
        assert (("source_failing:charm" in keys(h)) == (i >= hl.MIN_CONSECUTIVE_FAILURES)), i
    assert [k for k in h["new_alerts"] if k.startswith("source_failing")] == ["source_failing:charm"]
    since = next(a["since"] for a in h["alerts"] if a["key"] == "source_failing:charm")
    h2 = hl.evaluate(failing, h, NOW + timedelta(hours=30))
    assert "source_failing:charm" not in h2["new_alerts"] and next(a["since"] for a in h2["alerts"] if a["key"] == "source_failing:charm") == since
    h3 = hl.evaluate(base, h2, NOW + timedelta(hours=36))  # one success clears it
    assert "source_failing:charm" not in keys(h3) and "source_failing:charm" in h3["cleared_alerts"]
    assert h3["sources"]["charm"]["consecutive_failures"] == 0


def test_satellite_age_alerts_follow_latency_not_clouds(base):
    olci = next(lyr for lyr in base.layers if lyr.layer_id == "olci300_chl_latest")
    newest = datetime.fromisoformat(olci.time.observed_date + "T12:00:00+00:00")
    assert "satellite_stale:olci300_chl_latest" not in keys(hl.evaluate(base, None, newest + timedelta(days=hl.OLCI_MAX_AGE_DAYS)))
    assert "satellite_stale:olci300_chl_latest" in keys(hl.evaluate(base, None, newest + timedelta(days=hl.OLCI_MAX_AGE_DAYS + 1)))
    # a cloudy week (low coverage) with a recent pixel does not alert
    cloudy = base.model_copy(update={"layers": [lyr.model_copy(update={"coverage": lyr.coverage.model_copy(update={"domain_observed_fraction": 0.01})}) if lyr.layer_id == "olci300_chl_latest" else lyr for lyr in base.layers]})
    assert not any(k.startswith("satellite_stale:olci") for k in keys(hl.evaluate(cloudy, None, NOW)))


def test_missing_charm_runs_and_stale_hf_radar(base):
    run = next(r for r in base.forecast_runs if r.group_id == "charm")
    issued = datetime.fromisoformat(run.issued_date + "T12:00:00+00:00")
    # what the app labels "Stale" alerts (data_stale supersedes the older 3-day charm_missing)
    assert "data_stale:charm" in keys(hl.evaluate(base, None, issued + timedelta(days=2)))
    assert "charm_missing" not in keys(hl.evaluate(base, None, issued + timedelta(days=hl.CHARM_MAX_AGE_DAYS + 1)))
    assert not any(k.endswith(":charm") for k in keys(hl.evaluate(base, None, issued + timedelta(days=1))))
    assert "hfr_stale" in keys(hl.evaluate(base, None, NOW + timedelta(hours=hl.HFR_MAX_AGE_HOURS)))  # newest hour 12:00, now 18:00 + 18 h
    assert "hfr_stale" not in keys(hl.evaluate(base, None, NOW))


def test_hf_radar_coverage_drop_and_lost_region(base):
    hourly = sorted((lyr for lyr in base.layers if lyr.layer_id.startswith("hfr2km_currents_2")), key=lambda x: x.time.valid_time)
    newest = hourly[-1]
    thin = newest.model_copy(update={"qc": newest.qc.model_copy(update={"n_valid": int(newest.qc.n_valid * 0.3)})})
    m = base.model_copy(update={"layers": [thin if lyr.layer_id == newest.layer_id else lyr for lyr in base.layers]})
    assert "hfr_coverage_drop" in keys(hl.evaluate(m, None, NOW))
    # a region seen before and now empty alerts; a region never covered does not
    h1 = hl.evaluate(base, None, NOW)
    assert "monterey_bay" in h1["hfr_regions_last_covered"]

    def zero(lyr):
        if not lyr.layer_id.startswith("hfr2km_currents_2"):
            return lyr
        return lyr.model_copy(update={"coverage": lyr.coverage.model_copy(update={"regions": [r.model_copy(update={"observed_fraction": 0.0}) for r in lyr.coverage.regions]})})

    lost = base.model_copy(update={"layers": [zero(lyr) for lyr in base.layers]})
    assert "hfr_region_lost:monterey_bay" in keys(hl.evaluate(lost, h1, NOW + timedelta(hours=6)))
    assert not any(k.startswith("hfr_region_lost") for k in keys(hl.evaluate(lost, None, NOW)))  # no baseline: no alert


def test_missed_runs_alert(base):
    assert "publish_stale" in keys(hl.evaluate(base, None, NOW, previous_generated_at="2026-10-07T18:00:00Z"))
    assert "publish_stale" not in keys(hl.evaluate(base, None, NOW, previous_generated_at="2026-10-08T12:00:00Z"))


def test_health_file_is_published_and_kept(tmp_path):
    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    h = hl.run_health(out, None, NOW)
    assert (out / hl.HEALTH_FILE).exists() and keys(h) == {S3B_SILENT}
    # survives the next run (not pruned) and carries the failure count forward
    refuse = lambda url: (_ for _ in ()).throw(FetchError(f"Failed after 4 attempts: {url} (HTTP Error 403: )"))  # noqa: E731
    run_pipeline(fixture_context(out, fetcher=FixtureFetcher(overrides={r"wvcharmV3_": refuse})))
    assert (out / hl.HEALTH_FILE).exists()
    h2 = hl.run_health(out, None, NOW + timedelta(hours=6))
    assert h2["sources"]["charm"]["consecutive_failures"] == 1 and keys(h2) == {S3B_SILENT}
    assert guard_publish_tree(tmp_path) == []
    assert "All checks clear" in hl.markdown({**h, "alerts": [], "new_alerts": []}) and "[platform_silent]" in hl.markdown(h)


def test_a_partly_failing_source_alerts_after_consecutive_runs(base):
    part = base.model_copy(update={"sources": [s.model_copy(update={"outcome": "partial", "error": "Sentinel-3A ...: time listing failed: HTTP 404 (Currently unknown datasetID)"}) if s.source_id == "satellite_chl" else s for s in base.sources]})
    h = None
    for i in range(1, hl.MIN_CONSECUTIVE_FAILURES + 1):
        h = hl.evaluate(part, h, NOW + timedelta(hours=6 * i))
    # satellite chlorophyll alerts per product (product_failing), not as one degraded source
    assert "product_failing:olci300" in keys(h) and "product_failing:viirs750" not in keys(h)
    assert "product_failing:olci300" not in keys(hl.evaluate(part, None, NOW))



# ---------------------------------------------------------------- 2026-10-13: three failures, no alert


def failing_on(m: Manifest, ids: tuple[str, ...], host: str = "coastwatch.pfeg.noaa.gov") -> Manifest:
    """The 00:42Z run: several sources refused by one host in the same run."""
    return m.model_copy(update={"sources": [
        s.model_copy(update={"outcome": "failed", "error": f"probe failed: https://{host}/erddap/griddap/x.csv0 (HTTP Error 403: )"}) if s.source_id in ids else s
        for s in m.sources
    ]})


def test_one_host_failing_for_several_sources_alerts_on_the_second_run_once(base):
    m = failing_on(base, ("charm", "hf_radar", "satellite_chl"))
    h1 = hl.evaluate(m, None, NOW)
    assert not any(k.startswith("upstream_outage") for k in keys(h1))  # one run: NOAA often refuses a single run
    assert h1["failing_hosts"] == {"coastwatch.pfeg.noaa.gov": ["charm", "hf_radar", "satellite_chl"]}
    h2 = hl.evaluate(m, h1, NOW + timedelta(hours=8))
    assert "upstream_outage:coastwatch.pfeg.noaa.gov" in h2["new_alerts"]
    # one alert for the host, not one per source
    assert not any(k.startswith(("source_failing", "source_not_updating", "source_degraded")) for k in keys(h2))
    h3 = hl.evaluate(m, h2, NOW + timedelta(hours=16))
    assert h3["new_alerts"] == [] or all(not k.startswith("upstream_outage") for k in h3["new_alerts"])
    # a single source failing does not count as a host outage
    one = failing_on(base, ("charm",))
    assert not any(k.startswith("upstream_outage") for k in keys(hl.evaluate(one, hl.evaluate(one, None, NOW), NOW + timedelta(hours=6))))


def test_a_host_refusing_several_sources_in_single_runs_again_and_again_alerts_once(base):
    """The replayed 2026-10-09..13 pattern: pfeg refused C-HARM, HF radar and VIIRS together in
    single runs, recovering in between, so neither the consecutive nor the outage check fired."""
    bad, good = failing_on(base, ("charm", "hf_radar", "satellite_chl")), base
    pattern = [bad, good, good, bad, good, bad, good, good, good, good, bad, good]
    h, opened, t = None, [], NOW
    for i, m in enumerate(pattern):
        h = hl.evaluate(m, h, t)
        opened += [(i, k) for k in h["new_alerts"] if k.startswith("upstream_")]
        t += timedelta(hours=8)
    # opens on the third burst, once, and stays open across the recoveries (no flapping)
    assert opened == [(5, "upstream_intermittent:coastwatch.pfeg.noaa.gov")]
    assert "upstream_intermittent:coastwatch.pfeg.noaa.gov" in keys(h)
    # clears after a whole window without such a failure
    for _ in range(hl.INTERMITTENT_WINDOW):
        h = hl.evaluate(good, h, t)
        t += timedelta(hours=8)
    assert not any(k.startswith("upstream_intermittent") for k in keys(h))
    # two isolated bursts are ordinary NOAA behaviour
    h = None
    for m in [bad, good, good, good, bad, good]:
        h = hl.evaluate(m, h, t)
        t += timedelta(hours=8)
        assert not any(k.startswith("upstream_intermittent") for k in keys(h))


def test_no_success_for_too_long_alerts_by_time_even_when_runs_drift(base):
    m = failing_on(base, ("hf_radar",))
    last = next(s.last_success_at for s in base.sources if s.source_id == "hf_radar")
    t0 = datetime.fromisoformat(last.replace("Z", "+00:00"))
    # two failed runs 10 h apart: under the 3-in-a-row count, but over the 18 h budget
    h1 = hl.evaluate(m, None, t0 + timedelta(hours=10))
    h2 = hl.evaluate(m, h1, t0 + timedelta(hours=20))
    assert any(a["key"].endswith(":hf_radar") or a["kind"] == "hfr_stale" for a in h2["alerts"])
    assert len([a for a in h2["alerts"] if a["key"].endswith(":hf_radar") or a["kind"].startswith("hfr")]) == 1  # one per source


def test_every_run_writes_a_freshness_and_coverage_table_with_the_sources_own_dates(base):
    h = hl.evaluate(base, None, NOW)
    f = h["freshness"]
    assert {k for k in f if not k.startswith("satellite_chl:")} == {s.source_id for s in base.sources}
    assert {"satellite_chl:olci300", "satellite_chl:viirs750"} <= set(f)  # each product its own row
    charm = f["charm"]
    assert charm["basis"] == "issued_date" and charm["newest_date"] == next(s.latest_issued_date for s in base.sources if s.source_id == "charm")
    assert 0 < charm["coverage"]["valid_fraction"] <= 1
    assert "| Source | Last attempt |" in hl.markdown(h)


def with_charm_input(m: Manifest, **md) -> Manifest:
    def upd(lyr):
        if lyr.group_id != "charm" or not lyr.provenance:
            return lyr
        return lyr.model_copy(update={"provenance": lyr.provenance.model_copy(update={"upstream_metadata": {**(lyr.provenance.upstream_metadata or {}), **md}})})
    return m.model_copy(update={"layers": [upd(lyr) for lyr in m.layers]})


def test_a_change_in_charm_s_declared_satellite_input_alerts_once(base):
    snpp = with_charm_input(base, satellite="S-NPP", product_name="C-HARM v3.1 model output with SNPP VIIRS and WCOFS data", product_version="3.1")
    h1 = hl.evaluate(snpp, None, NOW)
    assert h1["charm_input"]["satellite"] == "S-NPP" and "charm_input_changed" not in keys(h1)
    h2 = hl.evaluate(snpp, h1, NOW + timedelta(hours=6))
    assert "charm_input_changed" not in keys(h2)
    n20 = with_charm_input(base, satellite="NOAA-20", product_name="C-HARM v3.2 model output with NOAA-20 VIIRS and WCOFS data", product_version="3.2")
    h3 = hl.evaluate(n20, h2, NOW + timedelta(hours=12))
    assert "charm_input_changed" in h3["new_alerts"]
    assert "S-NPP" in next(a["message"] for a in h3["alerts"] if a["key"] == "charm_input_changed")


def test_charm_still_declaring_snpp_after_the_end_of_snpp_data_is_flagged(base):
    snpp = with_charm_input(base, satellite="S-NPP")
    after = datetime(2026, 11, 5, 18, 0, tzinfo=timezone.utc)
    assert "charm_input_ageing" in keys(hl.evaluate(snpp, None, after))
    assert "charm_input_ageing" not in keys(hl.evaluate(snpp, None, NOW))


# ---------------------------------------------------------------- satellite products, each on its own
def _sat_error(m: Manifest, error: str) -> Manifest:
    return m.model_copy(update={"sources": [s.model_copy(update={"outcome": "partial", "error": error}) if s.source_id == "satellite_chl" else s for s in m.sources]})


def test_a_failing_product_alerts_under_its_own_key_and_does_not_mask_the_other(base):
    viirs_down = _sat_error(base, "VIIRS erdVHNchla1day: time listing failed: Failed after 4 attempts (HTTP Error 403: )")
    h = None
    for i in range(hl.MIN_CONSECUTIVE_FAILURES):
        h = hl.evaluate(viirs_down, h, NOW + timedelta(hours=6 * i))
    k = keys(h)
    assert "product_failing:viirs750" in k and "product_failing:olci300" not in k
    assert S3B_SILENT in k  # the OLCI platform alert is still reported beside the VIIRS one
    assert not any(x.startswith("source_degraded:satellite_chl") for x in k)
    rows = h["freshness"]
    assert rows["satellite_chl:viirs750"]["outcome"] == "failed" and rows["satellite_chl:olci300"]["newest_date"]


def test_viirs_is_stale_after_five_days_now_that_latency_is_measured(base):
    newest = datetime.fromisoformat(next(lyr for lyr in base.layers if lyr.layer_id == "viirs750_chl_latest").time.observed_date + "T12:00:00+00:00")
    assert hl.VIIRS_MAX_AGE_DAYS == 5
    assert "satellite_stale:viirs750_chl_latest" in keys(hl.evaluate(base, None, newest + timedelta(days=6)))
    assert "satellite_stale:viirs750_chl_latest" not in keys(hl.evaluate(base, None, newest + timedelta(days=5)))


def test_a_platform_that_comes_back_clears(base):
    h = hl.evaluate(base, None, NOW)
    h["satellite_products"]["olci300"]["platform_last_seen"]["Sentinel-3B"] = "2026-10-07"
    assert S3B_SILENT in h["alerts"][0]["key"]
    assert S3B_SILENT not in keys(hl.evaluate(base, h, NOW + timedelta(hours=6)))


def test_coverage_collapse_needs_a_history_and_a_real_drop(base):
    lyr = next(x for x in base.layers if x.layer_id == "olci300_chl_latest")
    frac = lyr.coverage.domain_observed_fraction
    h = None
    for i in range(hl.COVERAGE_MIN_RUNS):
        h = hl.evaluate(base, h, NOW + timedelta(hours=6 * i))
    assert "coverage_collapse:olci300" not in keys(h)
    low = base.model_copy(update={"layers": [x.model_copy(update={"coverage": x.coverage.model_copy(update={"domain_observed_fraction": frac * 0.1})}) if x.layer_id == lyr.layer_id else x for x in base.layers]})
    assert "coverage_collapse:olci300" in keys(hl.evaluate(low, h, NOW + timedelta(hours=30)))
    assert "coverage_collapse:olci300" not in keys(hl.evaluate(low, None, NOW))  # no history: no alert


def test_ports_without_any_chlorophyll_alert(base):
    ports = [{"port_code": i, "chlorophyll": {"latest": None, "error": "satellite chlorophyll unavailable: HTTP 403"}} for i in range(22)]
    assert "port_chl_empty" in keys(hl.evaluate(base, None, NOW, port_intel={"ports": ports}))
    ports[3]["chlorophyll"] = {"latest": {"median": 1.2}}
    assert "port_chl_empty" not in keys(hl.evaluate(base, None, NOW, port_intel={"ports": ports}))
