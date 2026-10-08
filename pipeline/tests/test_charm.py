from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pytest

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.models import Manifest
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.sources import charm

from .conftest import failing, make_nc, static

T0 = datetime(2026, 10, 7, 12, tzinfo=timezone.utc)


# ---------------------------------------------------------------- valid / issue time
@pytest.mark.parametrize(
    "lead,valid,issued",
    [
        (0, date(2026, 10, 7), date(2026, 10, 8)),
        (1, date(2026, 10, 8), date(2026, 10, 8)),
        (2, date(2026, 10, 9), date(2026, 10, 8)),
        (3, date(2026, 10, 10), date(2026, 10, 8)),
        (0, date(2026, 12, 31), date(2027, 1, 1)),
    ],
)
def test_issue_date_derivation(lead, valid, issued):
    assert charm.issue_date_for(valid, lead) == issued


def test_fixture_run_publishes_all_leads(ctx):
    res = charm.run(ctx)
    assert res.errors == []
    assert res.run is not None
    assert res.run.issued_date == "2026-10-08" and res.run.issued_date_derived
    assert res.run.leads_available == [0, 1, 2, 3] and res.run.leads_missing == []
    assert len(res.layers) == 12
    valid = {lyr.time.lead_days: lyr.time.valid_date for lyr in res.layers}
    assert valid == {0: "2026-10-07", 1: "2026-10-08", 2: "2026-10-09", 3: "2026-10-10"}
    for lyr in res.layers:
        assert lyr.product_class == "official_forecast"
        assert lyr.time.issued_date == "2026-10-08"
        assert lyr.time.valid_time.endswith("T12:00:00Z")
        assert lyr.qc and all(c.passed for c in lyr.qc.checks)
        assert (ctx.out_dir / lyr.image.url).exists() and (ctx.out_dir / lyr.grid.url).exists()


def test_lead_from_older_run_is_not_mixed_in(out):
    # lead 3 lags: its newest valid day (10-09) belongs to the run issued 10-07
    fetcher = FixtureFetcher(overrides={r"wvcharmV3_3day\.csv0": static(b"2026-10-09T12:00:00Z\n")})
    res = charm.run(fixture_context(out, fetcher=fetcher))
    assert res.run.issued_date == "2026-10-08"
    assert res.run.leads_available == [0, 1, 2]
    assert res.run.leads_missing == [3]
    assert not any(lyr.time.lead_days == 3 for lyr in res.layers)
    assert any("Lead 3" in n and "2026-10-07" in n for n in res.run.notes)
    # and the stale lead's data was never downloaded
    assert not any("wvcharmV3_3day.nc" in u for u in fetcher.calls)


def test_one_lead_download_failure_is_reported_missing(out):
    fetcher = FixtureFetcher(overrides={r"wvcharmV3_2day\.nc": failing()})
    res = charm.run(fixture_context(out, fetcher=fetcher))
    assert res.run.leads_missing == [2]
    assert any("lead 2" in e for e in res.errors)


# ---------------------------------------------------------------- malformed inputs
def _load(blob, expected=T0):
    return charm.load_lead(blob, 0, expected, "test://")


def test_html_error_page_is_rejected():
    with pytest.raises(charm.ValidationFailure, match="unreadable"):
        _load(b"<html><body>Error {\n code=500;\n}</body></html>")


def test_missing_variable_is_rejected():
    with pytest.raises(charm.ValidationFailure, match="missing variables"):
        _load(make_nc(drop=("cellular_domoic",)))


def test_out_of_range_probability_is_rejected():
    v = {n: np.full((10, 10), 0.5) for n in ("pseudo_nitzschia", "particulate_domoic", "cellular_domoic")}
    v["particulate_domoic"][3, 3] = 1.7
    with pytest.raises(charm.ValidationFailure, match="particulate_domoic_probability_range"):
        _load(make_nc(values=v))


def test_irregular_grid_is_rejected():
    lat = np.array([36.01, 36.04, 36.08, 36.11, 36.14, 36.17, 36.20, 36.23, 36.26, 36.29])
    with pytest.raises(charm.ValidationFailure, match="regular_grid"):
        _load(make_nc(lat=lat))


def test_outside_domain_is_rejected():
    with pytest.raises(charm.ValidationFailure, match="within_published_domain"):
        _load(make_nc(lon=np.round(250.01 + 0.03 * np.arange(10), 4)))


def test_unexpected_product_version_is_rejected():
    with pytest.raises(charm.ValidationFailure, match="product_version"):
        _load(make_nc(version="4.0"))


def test_time_mismatch_is_rejected():
    with pytest.raises(charm.ValidationFailure, match="valid_time_matches_probe"):
        _load(make_nc(), expected=datetime(2026, 10, 6, 12, tzinfo=timezone.utc))


def test_fill_values_become_no_data():
    v = {n: np.full((10, 10), 0.25) for n in ("pseudo_nitzschia", "particulate_domoic", "cellular_domoic")}
    for n in v:
        v[n][:4, :] = np.nan  # written as -99999 fill
    ld, _ = _load(make_nc(values=v))
    a = ld.values["particulate_domoic"]
    assert np.isnan(a[:4]).all() and np.allclose(a[4:], 0.25)


def test_mostly_empty_field_is_rejected():
    v = {n: np.full((10, 10), np.nan) for n in ("pseudo_nitzschia", "particulate_domoic", "cellular_domoic")}
    for n in v:
        v[n][0, 0] = 0.4
    with pytest.raises(charm.ValidationFailure, match="valid_fraction"):
        _load(make_nc(values=v))


def test_longitudes_converted_to_plus_minus_180():
    ld, _ = _load(make_nc())
    assert (ld.lon < 0).all() and ld.lon.min() > -123 and ld.lon.max() < -121


# ---------------------------------------------------------------- pipeline-level failure handling
def test_total_failure_keeps_last_good_run_with_real_dates(out):
    first = run_pipeline(fixture_context(out))
    good_ids = {lyr.layer_id for lyr in first.layers if lyr.group_id == "charm"}
    fetcher = FixtureFetcher(overrides={r"wvcharmV3_": failing("HTTP 502")})
    later = run_pipeline(fixture_context(out, now="2026-10-12T18:00:00Z", fetcher=fetcher))
    st = next(s for s in later.sources if s.source_id == "charm")
    assert st.outcome == "failed" and "HTTP 502" in st.error
    assert st.last_success_at == "2026-10-08T18:00:00Z"
    assert st.last_attempt_at == "2026-10-12T18:00:00Z"
    kept = [lyr for lyr in later.layers if lyr.group_id == "charm"]
    assert {lyr.layer_id for lyr in kept} == good_ids
    # dates are the true ones, never relabelled
    assert {lyr.time.issued_date for lyr in kept} == {"2026-10-08"}
    # other sources still updated
    assert next(s for s in later.sources if s.source_id == "gibs_chl").outcome != "failed"


def test_first_ever_failure_publishes_no_charm_layers(out):
    fetcher = FixtureFetcher(overrides={r"wvcharmV3_": failing()})
    m = run_pipeline(fixture_context(out, fetcher=fetcher))
    assert not [lyr for lyr in m.layers if lyr.group_id == "charm"]
    st = next(s for s in m.sources if s.source_id == "charm")
    assert st.outcome == "failed" and st.last_success_at is None
    # manifest still written and valid
    Manifest.model_validate_json((out / "manifest.json").read_text())


def test_output_is_deterministic(tmp_path):
    a = run_pipeline(fixture_context(tmp_path / "a"))
    b = run_pipeline(fixture_context(tmp_path / "b"))
    assert a.model_dump() == b.model_dump()
    for lyr in a.layers:
        if lyr.image:
            assert (tmp_path / "a" / lyr.image.url).read_bytes() == (tmp_path / "b" / lyr.image.url).read_bytes()
            assert (tmp_path / "a" / lyr.grid.url).read_bytes() == (tmp_path / "b" / lyr.grid.url).read_bytes()
