"""CalHABMAP observations: missing vs zero, censoring, QC, isolation, C-HARM context."""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
from datetime import date

import pytest

from coastwatch_pipeline.fixtures import FIXTURES, FixtureFetcher, fixture_context
from coastwatch_pipeline.http import FetchError, Response
from coastwatch_pipeline.models import ObservationDataset
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.sources import observations as obs

IDX = json.loads((FIXTURES / "recorded" / "index.json").read_text())


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr(obs, "RETRY_SLEEP_S", 0)


def recorded_csv(key: str) -> list[dict]:
    url = obs.request_url(key)
    rows = list(csv.reader(io.StringIO((FIXTURES / "recorded" / IDX[url]["file"]).read_text(encoding="latin-1"))))
    return [dict(zip(rows[0], r)) for r in rows[2:]]


def run(out, fetcher=None, now=None):
    kw = {"fetcher": fetcher} if fetcher else {}
    if now:
        kw["now"] = now
    m = run_pipeline(fixture_context(out, **kw))
    return m, ObservationDataset.model_validate_json((out / m.observations_url).read_text())


def station(ds, name):
    return next(s for s in ds.stations if s.name == name)


def series(st, var):
    return next(s for s in st.series if s.variable == var)


def summary(st, var):
    return next(s for s in st.summaries if s.variable == var)


def test_all_seventeen_stations_published_with_units_and_sources(out):
    m, ds = run(out)
    assert len(ds.stations) == 17 and all(s.status == "updated" for s in ds.stations)
    src = next(s for s in m.sources if s.source_id == "calhabmap")
    assert src.product_class == "observation" and src.freshness.basis == "observed_date"
    units = {v.id: v.units for v in ds.variables}
    assert units == {"pDA": "ng/mL", "dDA": "ng/mL", "tDA": "ng/mL", "pn_seriata": "cells/L", "pn_delicatissima": "cells/L", "chl_extracted": "mg/m3", "temp": "degree_C"}
    assert all(v.matrix == "seawater" for v in ds.variables)
    assert all(v.detection_limit is None for v in ds.variables), "no limit is published upstream; never invent one"


def test_not_measured_is_null_and_never_zero(out):
    _, ds = run(out)
    rows = recorded_csv("SantaCruzWharf")
    st = station(ds, "Santa Cruz Wharf")
    assert len(st.sample_times) == len(rows)
    for var, col in (("pDA", "pDA"), ("tDA", "tDA"), ("pn_seriata", "Pseudo_nitzschia_seriata_group")):
        nan_upstream = sum(1 for r in rows if r[col] in ("NaN", ""))
        vals = series(st, var).values
        assert sum(v is None for v in vals) == nan_upstream
        for r, v in zip(rows, vals):
            if r[col] in ("NaN", ""):
                assert v is None
            else:
                assert v is not None and math.isclose(v, float(r[col]), rel_tol=1e-6, abs_tol=1e-9)


def test_reported_zero_is_qualified_not_absent(out):
    _, ds = run(out)
    rows = recorded_csv("SantaCruzWharf")
    st = station(ds, "Santa Cruz Wharf")
    s = series(st, "pDA")
    zeros = [i for i, r in enumerate(rows) if r["pDA"] not in ("NaN", "") and float(r["pDA"]) == 0]
    assert zeros, "fixture should contain reported zeros"
    assert {int(k) for k, q in s.qualifiers.items() if q == "reported_zero"} == set(zeros)
    assert summary(st, "pDA").n_reported_zero == len(zeros)
    var = next(v for v in ds.variables if v.id == "pDA")
    assert "not that toxin was absent" in var.zero_policy


def test_summary_matches_independent_recomputation(out):
    _, ds = run(out)
    rows = recorded_csv("SantaCruzWharf")
    st = station(ds, "Santa Cruz Wharf")
    measured = [(r["time"][:10], float(r["pDA"])) for r in rows if r["pDA"] not in ("NaN", "")]
    sm = summary(st, "pDA")
    assert sm.last_date == measured[-1][0] and math.isclose(sm.last_value, measured[-1][1], rel_tol=1e-6)
    cutoff = "2025-10-08"
    recent = [d for d, _ in measured if d >= cutoff]
    gaps = [(date.fromisoformat(b) - date.fromisoformat(a)).days for a, b in zip(recent, recent[1:])]
    assert sm.n_last_365d == len(recent) and sm.median_interval_days_365d == statistics.median(gaps)
    assert sm.days_since_last == (date(2026, 10, 8) - date.fromisoformat(sm.last_date)).days


def test_monterey_wharf_long_absence_is_reported_as_absence(out):
    """Monterey Wharf has had no pDA since 2022: the summary must say so, not show zero."""
    _, ds = run(out)
    sm = summary(station(ds, "Monterey Wharf"), "pDA")
    assert sm.last_date and sm.last_date < "2023-01-01"
    assert sm.n_last_365d == 0 and sm.max_last_365d is None and sm.days_since_last > 1000


def test_fractions_and_groups_are_never_combined(out):
    _, ds = run(out)
    ids = [v.id for v in ds.variables]
    assert len(ids) == len(set(ids))
    st = station(ds, "Trinidad Pier")
    p, d, t = (series(st, k).values for k in ("pDA", "dDA", "tDA"))
    # tDA is published only where measured upstream; never derived from pDA + dDA
    rows = recorded_csv("TrinidadPier")
    assert [v is None for v in t] == [r["tDA"] in ("NaN", "") for r in rows]


def test_station_charm_context_only_where_recorded(out):
    _, ds = run(out)
    sc = station(ds, "Santa Cruz Wharf").charm
    assert sc and sc.error is None and sc.history_days == 180
    pts = sc.history["particulate_domoic"]
    assert pts[-1].date == "2026-10-07" and all(p.value is None or 0 <= p.value <= 1 for p in pts)
    other = station(ds, "Scripps Pier (La Jolla)").charm
    assert other and other.error and other.history == {}


def test_nearest_port_and_region(out):
    _, ds = run(out)
    st = station(ds, "Santa Cruz Wharf")
    assert st.nearest_port_name == "Santa Cruz" and st.nearest_port_km < 5 and st.region == "monterey_bay"


# ---------------------------------------------------------------- adversarial
HEADER = ",".join(obs.COLUMNS)
UNITS = ",".join(obs.EXPECTED_UNITS.get(c, "") for c in obs.COLUMNS)


def csv_body(rows: list[str], units: str = UNITS, header: str = HEADER) -> bytes:
    return ("\n".join([header, units, *rows]) + "\n").encode()


def row(t, pda="NaN", dda="NaN", tda="NaN", ser="NaN", deli="NaN", chl="NaN", temp="NaN", lat="36.958", lon="-122.017"):
    return f"{t},{lat},{lon},1.0,HAB_SCW,1,{pda},{dda},{tda},{ser},{deli},{chl},{temp}"


def only_station(key, body=None, fn=None):
    url = obs.request_url(key)

    def serve(u):
        if fn:
            return fn(u)
        return Response(url=u, status=200, content_type="text/csv", body=body)

    return FixtureFetcher(overrides={__import__("re").escape(url): serve})


def test_adversarial_values_negative_duplicate_future_high(out):
    body = csv_body(
        [
            row("2026-09-01T16:00:00Z", pda="-0.5", ser="1000"),
            row("2026-09-08T16:00:00Z", pda="0.0", ser="0"),
            row("2026-09-08T16:00:00Z", pda="0.0", ser="0"),  # exact duplicate
            row("2026-09-15T16:00:00Z", pda="900", ser="2e9", temp="45"),  # implausible
            row("2027-01-01T00:00:00Z", pda="1.0"),  # future
        ]
    )
    _, ds = run(out, only_station("SantaCruzWharf", body))
    st = station(ds, "Santa Cruz Wharf")
    assert len(st.sample_times) == 3
    p = series(st, "pDA")
    assert p.values == [None, 0.0, 900.0]
    assert p.qualifiers == {"0": "rejected_negative", "1": "reported_zero", "2": "flag_high"}
    assert series(st, "pn_seriata").qualifiers == {"1": "reported_zero", "2": "flag_high"}
    assert series(st, "temp").qualifiers == {"2": "flag_high"}
    qc = {q.name: q for q in st.qc}
    assert not qc["negative_values"].passed and not qc["plausibility"].passed and not qc["time"].passed
    assert "1 exact duplicate" in qc["duplicates"].detail
    assert summary(st, "pDA").n_rejected == 1 and summary(st, "pDA").n_measured == 2


def test_unit_change_upstream_fails_station_not_mislabelled(out):
    bad_units = UNITS.replace("ng/mL", "ug/L", 1)
    _, ds = run(out, only_station("SantaCruzWharf", csv_body([row("2026-09-01T16:00:00Z", pda="1")], units=bad_units)))
    st = station(ds, "Santa Cruz Wharf")
    assert st.status == "failed" and "unexpected units" in st.error and st.series == []
    assert station(ds, "Monterey Wharf").status == "updated"


def test_missing_column_fails_station(out):
    header = HEADER.replace(",Temp", ",Temperature")
    _, ds = run(out, only_station("SantaCruzWharf", csv_body([row("2026-09-01T16:00:00Z")], header=header)))
    assert "missing columns: Temp" in station(ds, "Santa Cruz Wharf").error


def test_failed_station_keeps_previous_data_with_original_dates(out):
    _, first = run(out)
    prev = station(first, "Santa Cruz Wharf")

    def fail(u):
        raise FetchError(f"HTTP 502 for {u}")

    m, ds = run(out, only_station("SantaCruzWharf", fn=fail), now="2026-10-12T18:00:00Z")
    st = station(ds, "Santa Cruz Wharf")
    assert st.status == "carried_forward" and "502" in st.error
    assert st.sample_times == prev.sample_times and st.retrieved_at == prev.retrieved_at
    src = next(s for s in m.sources if s.source_id == "calhabmap")
    assert src.outcome == "partial" and any("previous data kept" in n for n in src.notes)


def test_transient_errors_are_retried(out):
    url = obs.request_url("SantaCruzWharf")
    calls = {"n": 0}
    real = FixtureFetcher()

    def flaky(u):
        calls["n"] += 1
        if calls["n"] < 3:
            raise FetchError(f"HTTP 404 for {u}")  # SCCOOS: 'unknown datasetID' while reloading
        return real._recorded(u)

    _, ds = run(out, FixtureFetcher(overrides={__import__("re").escape(url): flaky}))
    assert calls["n"] == 3 and station(ds, "Santa Cruz Wharf").status == "updated"


def test_all_stations_failing_marks_source_failed_and_keeps_previous_artifact(out):
    m1, _ = run(out)

    def fail(u):
        raise FetchError("HTTP 503")

    m2 = run_pipeline(fixture_context(out, now="2026-10-09T18:00:00Z", fetcher=FixtureFetcher(overrides={r"erddap\.sccoos\.org": fail})))
    # every station carried forward -> still published, partial
    ds = ObservationDataset.model_validate_json((out / m2.observations_url).read_text())
    assert all(s.status == "carried_forward" for s in ds.stations)
    assert next(s for s in m2.sources if s.source_id == "calhabmap").outcome == "partial"


def test_first_run_with_all_stations_failing_is_a_failed_source(out):
    def fail(u):
        raise FetchError("HTTP 503")

    m = run_pipeline(fixture_context(out, fetcher=FixtureFetcher(overrides={r"erddap\.sccoos\.org": fail})))
    st = next(s for s in m.sources if s.source_id == "calhabmap")
    assert st.outcome == "failed" and m.observations_url is None


def test_caveats_cover_integrity_rules(out):
    _, ds = run(out)
    text = " ".join(ds.caveats)
    for phrase in ("not the same as no toxin", "not 'absent'", "not domoic acid in seafood", "does not describe nearby", "not a toxic bloom"):
        assert phrase in text


def test_region_from_official_region_bounds_first(out):
    _, ds = run(out)
    # Inner Tomales Bay lies inside the San Francisco & Farallones bounds, although its
    # nearest landing port (Bodega Bay) is in Mendocino-Sonoma
    st = station(ds, "Inner Tomales Bay")
    assert st.region == "sf_bay_farallones" and st.nearest_port_name == "Bodega Bay"
