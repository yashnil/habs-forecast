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


def test_healthy_fixture_has_no_alerts(base):
    h = hl.evaluate(base, None, NOW)
    assert h["alerts"] == [] and h["new_alerts"] == []


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
    assert "charm_missing" in keys(hl.evaluate(base, None, issued + timedelta(days=hl.CHARM_MAX_AGE_DAYS + 1)))
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
    assert (out / hl.HEALTH_FILE).exists() and h["alerts"] == []
    # survives the next run (not pruned) and carries the failure count forward
    refuse = lambda url: (_ for _ in ()).throw(FetchError(f"Failed after 4 attempts: {url} (HTTP Error 403: )"))  # noqa: E731
    run_pipeline(fixture_context(out, fetcher=FixtureFetcher(overrides={r"wvcharmV3_": refuse})))
    assert (out / hl.HEALTH_FILE).exists()
    h2 = hl.run_health(out, None, NOW + timedelta(hours=6))
    assert h2["sources"]["charm"]["consecutive_failures"] == 1 and not h2["alerts"]
    assert guard_publish_tree(tmp_path) == []
    assert "All checks clear" in hl.markdown(h)


def test_a_partly_failing_source_alerts_after_consecutive_runs(base):
    part = base.model_copy(update={"sources": [s.model_copy(update={"outcome": "partial", "error": "Sentinel-3A ...: time listing failed: HTTP 404 (Currently unknown datasetID)"}) if s.source_id == "satellite_chl" else s for s in base.sources]})
    h = None
    for i in range(1, hl.MIN_CONSECUTIVE_FAILURES + 1):
        h = hl.evaluate(part, h, NOW + timedelta(hours=6 * i))
    assert "source_degraded:satellite_chl" in keys(h)
    assert "source_degraded:satellite_chl" not in keys(hl.evaluate(part, None, NOW))

