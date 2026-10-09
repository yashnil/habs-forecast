"""Historical fisheries exposure: deflation, suppression, double counting, honest gaps."""

from __future__ import annotations

import json
import re

import pytest

from coastwatch_pipeline.fixtures import FIXTURES, FixtureFetcher, fixture_context
from coastwatch_pipeline.http import FetchError
from coastwatch_pipeline.models import FisheriesDataset
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.sources import fisheries as fish

IDX = json.loads((FIXTURES / "recorded" / "index.json").read_text())


def foss_rows(year: int) -> list[dict]:
    body = (FIXTURES / "recorded" / IDX[fish.foss_url(year)]["file"]).read_bytes()
    return json.loads(body)["items"]


def run(out, fetcher=None, now=None):
    kw = {}
    if fetcher:
        kw["fetcher"] = fetcher
    if now:
        kw["now"] = now
    m = run_pipeline(fixture_context(out, **kw))
    ds = FisheriesDataset.model_validate_json((out / m.fisheries_url).read_text()) if m.fisheries_url else None
    return m, ds


def group(ds, gid):
    return next(g for g in ds.groups if g.id == gid)


def yv(rows, year):
    return next(r for r in rows if r.year == year)


def test_years_and_honest_unavailable_year(out):
    m, ds = run(out)
    assert ds.years == list(range(2015, 2025)) and ds.years_requested_unavailable == [2025]
    st = next(s for s in m.sources if s.source_id == "foss_landings")
    assert st.product_class == "historical_context" and st.latest_valid_date == "2024-12-31"
    assert any("2025" in n for n in st.notes)


def test_cpi_annual_averages_and_base_year(out):
    _, ds = run(out)
    d = ds.deflator
    # BLS-published CPI-U annual averages (U.S. city average, all items)
    assert d.annual_index["2015"] == 237.017 and d.annual_index["2023"] == 304.702 and d.annual_index["2024"] == 313.689
    # Oct 2025 was never published (lapse in appropriations): 2025 cannot be the base year
    assert d.base_year == 2024 and d.months_used["2025"] == 11 and "2025" not in d.annual_index
    assert any("missing M10" in n for n in d.notes)


def test_dungeness_values_match_source_and_deflation(out):
    _, ds = run(out)
    raw = {y: next(r for r in foss_rows(y) if r["ts_afs_name"] == "CRAB, DUNGENESS") for y in (2015, 2024)}
    g = group(ds, "dungeness_crab")
    assert yv(g.annual, 2024).dollars_nominal == pytest.approx(raw[2024]["dollars"], abs=0.01)
    assert yv(g.annual, 2024).dollars_real == yv(g.annual, 2024).dollars_nominal  # base year
    v15 = yv(g.annual, 2015)
    assert v15.dollars_real == pytest.approx(raw[2015]["dollars"] * 313.689 / 237.017, abs=0.05)


def test_no_double_counting_and_totals_reconcile(out):
    _, ds = run(out)
    for y in ds.years:
        rows = foss_rows(y)
        excluded = {(e.source_name, e.pounds, e.dollars_nominal) for e in ds.excluded_rows if e.year == y}
        kept = [r for r in rows if (r["ts_afs_name"], r["pounds"], r["dollars"]) not in excluded]
        species = [r for r in kept if r["ts_afs_name"] != fish.WITHHELD]
        assigned = [fish.assign_group(r["ts_afs_name"]) for r in species]
        group_sum = sum(yv(g.annual, y).dollars_nominal or 0 for g in ds.groups)
        grouped_rows = sum(r["dollars"] or 0 for r, a in zip(species, assigned) if a)
        assert group_sum == pytest.approx(grouped_rows, abs=0.1)
        # statewide total = every kept row including the withheld row (NOAA's definition)
        assert yv(ds.statewide_total, y).dollars_nominal == pytest.approx(sum(r["dollars"] or 0 for r in kept), abs=0.1)
        assert sum(yv(g.annual, y).n_rows for g in ds.groups) == sum(1 for a in assigned if a)


# NOAA, Fisheries of the United States 2022 and 2023, Table 4 (California, thousands of dollars)
FUS_TABLE4 = {2021: 209_783, 2022: (207_919, 207_849), 2023: 170_330}


def test_statewide_totals_reproduce_noaa_published_state_totals(out):
    """NOAA's published California totals equal FOSS rows with the KSTR duplicate counted once
    (within routine revisions); a plain sum of all rows overshoots by up to 2.7%."""
    _, ds = run(out)
    for y, published in FUS_TABLE4.items():
        for p in published if isinstance(published, tuple) else (published,):
            ours = yv(ds.statewide_total, y).dollars_nominal / 1e3
            assert abs(ours - p) / p < 0.002, (y, ours, p)
            naive = sum(r["dollars"] or 0 for r in foss_rows(y)) / 1e3
            assert (naive - p) / p > 0.003


def test_generic_bivalve_categories_are_matched_but_generic_crabs_are_not():
    for n in ("SCALLOPS **", "CLAMS **", "MUSSELS **", "OYSTER, KUMAMOTO **", "OYSTER, PACIFIC"):
        assert fish.assign_group(n) == "bivalves"
    for n in ("CRABS, DECAPODA (ORDER) **", "MOLLUSKS **", "ANCHOVIES", "SQUID, CALIFORNIA MARKET", "CRAB, KING **"):
        assert fish.assign_group(n) is None


def test_duplicate_oyster_listing_counted_once(out):
    _, ds = run(out)
    assert len(ds.excluded_rows) == 10
    assert {(e.source_name, e.duplicate_of) for e in ds.excluded_rows} == {("OYSTER, KUMAMOTO **", "OYSTER, PACIFIC")}
    assert "OYSTER, KUMAMOTO **" not in group(ds, "bivalves").source_names


def test_withheld_kept_separate_and_never_attributed(out):
    _, ds = run(out)
    for y in ds.years:
        w = next(r for r in foss_rows(y) if r["ts_afs_name"] == fish.WITHHELD)
        sv = next(s for s in ds.withheld if s.year == y)
        assert sv.dollars_nominal == pytest.approx(w["dollars"], abs=0.01)
        assert fish.WITHHELD not in {n for g in ds.groups for n in g.source_names}
    assert any("Courtesy: National Oceanic and Atmospheric Administration" in (p.citation or "") for p in ds.provenance)
    assert any("aquaculture" in c for c in ds.caveats) and any("meat weight" in c for c in ds.caveats)


def test_rows_without_value_are_not_zero():
    cpi = fish.Deflator(series_id="x", title="x", source_url="x", base_year=2024, annual_index={"2024": 100.0}, months_used={"2024": 12}, method="x")
    rows = {2024: [{"ts_afs_name": "CRAB, DUNGENESS", "year": 2024, "pounds": None, "dollars": None}]}
    _, groups, total, _, _ = fish.aggregate(rows, cpi)
    v = yv(next(g for g in groups if g.id == "dungeness_crab").annual, 2024)
    assert v.dollars_nominal is None and v.dollars_real is None and v.n_rows_without_value == 1
    assert yv(total, 2024).dollars_nominal is None


def test_ambiguous_species_name_is_rejected(monkeypatch):
    monkeypatch.setattr(fish, "GROUPS", fish.GROUPS + [{**fish.GROUPS[0], "id": "dup", "names": [r"CRAB, DUNGENESS"]}])
    with pytest.raises(ValueError, match="more than one group"):
        fish.assign_group("CRAB, DUNGENESS")


def test_port_level_is_unavailable_with_reasons(out):
    _, ds = run(out)
    assert ds.port_level.status == "unavailable"
    text = " ".join(ds.port_level.reasons)
    assert "MFDE" in text and "permission" in text
    assert {a["id"]: a["status"] for a in ds.port_level.adapters}["cdfw_mfde"] == "disabled"


def test_terminology_is_exposure_never_predicted_loss(out):
    _, ds = run(out)
    assert ds.terminology.startswith("Historical fisheries exposure")
    blob = json.dumps(ds.model_dump())
    # 'loss' may appear only negated ("not a prediction of losses", "not a forecast of losses")
    for m in re.finditer(r"[^.]*\blosse?s?\b[^.]*", blob):
        assert re.search(r"\bnot\b", m.group(0)), m.group(0)
    assert "await HAB scientific review" in blob


def test_foss_failure_keeps_previous_dataset(out):
    m1, ds1 = run(out)

    def fail(u):
        raise FetchError("HTTP 503")

    m2, _ = run(out, FixtureFetcher(overrides={r"apps-st\.fisheries": fail}), now="2026-10-20T18:00:00Z")
    st = next(s for s in m2.sources if s.source_id == "foss_landings")
    assert st.outcome == "failed" and "FOSS" in st.error and st.latest_valid_date == "2024-12-31"
    assert m2.fisheries_url == m1.fisheries_url and (out / m2.fisheries_url).exists()


def test_cpi_failure_fails_source_instead_of_publishing_nominal_as_real(out):
    def fail(u):
        raise FetchError("HTTP 503")

    m, ds = run(out, FixtureFetcher(overrides={r"api\.bls\.gov": fail}))
    st = next(s for s in m.sources if s.source_id == "foss_landings")
    assert st.outcome == "failed" and "CPI" in st.error and ds is None


def test_weekly_refresh_does_not_refetch(out):
    run(out)
    f = FixtureFetcher()
    m, _ = run(out, f, now="2026-10-10T18:00:00Z")
    assert not any("fisheries.noaa.gov" in c or "bls.gov" in c for c in f.calls)
    assert next(s for s in m.sources if s.source_id == "foss_landings").outcome == "unchanged"


# ---------------------------------------------------------------- disclosure-safe aggregation (synthetic)
def test_suppression_propagates_and_complementary_suppression():
    """Synthetic port-level cells (test-only; no real port data)."""
    C = fish.Cell
    cells = [
        C(("PortA", "crab", 2024), 100.0, False),
        C(("PortA", "lobster", 2024), 50.0, False),
        C(("PortB", "crab", 2024), 30.0, False),
        C(("PortB", "lobster", 2024), None, True),
        C(("PortC", "crab", 2024), None, True),
        C(("PortD", "crab", 2024), 10.0, False),
        C(("PortD", "lobster", 2024), None, True),
        C(("PortD", "urchin", 2024), None, True),
    ]
    by_port = {a.key: a for a in fish.aggregate_cells(cells, lambda k: (k[0], k[2]), source_totals={("PortB", 2024): 80.0})}
    assert by_port[("PortA", 2024)].value == 150.0 and by_port[("PortA", 2024)].publishable
    # one suppressed component and a published total -> subtraction would reveal it
    assert not by_port[("PortB", 2024)].publishable and by_port[("PortB", 2024)].value is None
    assert not by_port[("PortC", 2024)].publishable
    d = by_port[("PortD", 2024)]
    assert d.includes_suppressed and d.publishable and d.value == 10.0 and "At least" in d.note
    # each cell counted once
    assert sum(1 for c in cells for k in by_port if (c.key[0], c.key[2]) == k) == len(cells)


def test_new_pipeline_version_rebuilds_within_the_refresh_window(out):
    """A code change may change the method, so a cached artifact from another version is not reused."""
    run(out)
    ctx = fixture_context(out, now="2026-10-10T18:00:00Z")
    ctx.pipeline_version = "next-version"
    m = run_pipeline(ctx)
    st = next(s for s in m.sources if s.source_id == "foss_landings")
    assert st.outcome == "updated"
    assert any("fisheries.noaa.gov" in c for c in ctx.fetcher.calls)
