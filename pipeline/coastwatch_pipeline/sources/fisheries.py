"""Historical fisheries exposure: California commercial landings by species and year.

Source: NOAA Fisheries FOSS (Fisheries One Stop Shop) commercial landings, which are
compiled from PacFIN for California. FOSS publishes statewide totals by species and year;
confidential values are aggregated by FOSS into one "WITHHELD FOR CONFIDENTIALITY" row
per year, which is kept separate here and never redistributed back to species.

Values are nominal ex-vessel dollars upstream. Real dollars use the BLS CPI-U (U.S. city
average, all items, not seasonally adjusted, CUUR0000SA0); annual averages are computed
from the twelve published monthly values, and the base year is the most recent year with
all twelve months published.

"Historical fisheries exposure" is the value of past landings of species that can be
affected by marine toxins. It is not a prediction of losses and says nothing about any
current or future season.

Port-level values are not published by CoastWatch (see PORT_LEVEL): the only 2021+ source
(CDFW MFDE) has not granted permission for automated extraction or republication.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import quote

from ..context import RunContext
from ..models import (
    Deflator,
    ExcludedRow,
    FisheriesDataset,
    PortLevelStatus,
    Provenance,
    SpeciesGroup,
    SuppressedValue,
    YearValue,
)

SOURCE_ID = "foss_landings"
FOSS_URL = "https://apps-st.fisheries.noaa.gov/ods/foss/landings/"
FOSS_PAGE = "https://www.fisheries.noaa.gov/foss"
BLS_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
CPI_SERIES = "CUUR0000SA0"
FIRST_YEAR = 2015
PAGE_LIMIT = 1000
WITHHELD = "WITHHELD FOR CONFIDENTIALITY"
FOSS_LICENSE = "U.S. Government work (NOAA Fisheries); public domain within the United States. FOSS asks users to cite the query and access date."
BLS_LICENSE = "U.S. Government work (Bureau of Labor Statistics); public domain."

TIER_REVIEW = "Tier assignments are CoastWatch's editorial grouping and await HAB scientific review."

# id, label, tier, basis, official record ids (CoastWatch registry), exact upstream names / regex
GROUPS: list[dict] = [
    {
        "id": "dungeness_crab",
        "label": "Dungeness crab",
        "tier": 1,
        "basis": "CDFW has delayed or closed the commercial Dungeness crab fishery because of domoic acid in past seasons (for example 2015–16). CDFW currently states there are no marine-toxin closures in this fishery.",
        "records": [],
        "names": [r"CRAB, DUNGENESS"],
    },
    {
        "id": "rock_crab",
        "label": "Rock crabs",
        "tier": 1,
        "basis": "A CDFW commercial rock crab closure for domoic acid is active north of 40° N (see the official notice).",
        "records": ["cdfw-rock-crab-commercial-40n"],
        "names": [r"CRAB, (RED|YELLOW|BROWN) ROCK", r"CRABS?, ROCK.*"],
    },
    {
        "id": "northern_anchovy",
        "label": "Northern anchovy",
        "tier": 1,
        "basis": "A CDFW commercial and recreational take restriction for domoic acid is active in Monterey Bay (see the official notice).",
        "records": ["cdfw-2026-anchovy-take-restriction-monterey-bay"],
        "names": [r"ANCHOVY, NORTHERN"],
    },
    {
        "id": "spiny_lobster",
        "label": "California spiny lobster",
        "tier": 2,
        "basis": "CDFW monitors and reports marine-toxin closures for this fishery; it currently states there are none.",
        "records": [],
        "names": [r"LOBSTER, CALIFORNIA SPINY"],
    },
    {
        "id": "pacific_sardine",
        "label": "Pacific sardine",
        "tier": 2,
        "basis": "A plankton-feeding fish that can carry domoic acid; no current commercial toxin closure is listed.",
        "records": [],
        "names": [r"SARDINE, PACIFIC"],
    },
    {
        "id": "bivalves",
        "label": "Bivalve shellfish (oysters, clams, mussels, scallops)",
        "tier": 2,
        "basis": "CDPH quarantines and advisories apply to sport-harvested bivalves; commercially sold bivalves come from certified growers tested under CDPH's program.",
        "records": ["cdph-2026-annual-mussel-quarantine", "cdph-2026-sn26-018-monterey-bivalves", "cdph-nci-bivalve-special-advisory"],
        "names": [r"OYSTERS?(,.*)?", r"CLAMS?(,.*)?( \*\*)?", r"MUSSELS?(,.*)?", r"SCALLOPS?(,.*)?"],
    },
]


def foss_url(year: int, offset: int = 0) -> str:
    q = json.dumps({"state_name": "CALIFORNIA", "year": year, "collection": "Commercial"}, separators=(",", ":"))
    return f"{FOSS_URL}?q={quote(q, safe='')}&limit={PAGE_LIMIT}&offset={offset}"


def bls_body(start: int, end: int) -> bytes:
    return json.dumps({"seriesid": [CPI_SERIES], "startyear": str(start), "endyear": str(end)}, separators=(",", ":")).encode()


def fetch_foss_year(ctx: RunContext, year: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        d = json.loads(ctx.fetcher(foss_url(year, offset)).body)
        items = d.get("items", [])
        rows.extend({k: v for k, v in i.items() if k != "links"} for i in items)
        if not d.get("hasMore") or not items:
            return rows
        offset += len(items)


def fetch_cpi(ctx: RunContext, first: int, last: int) -> dict[int, dict[str, float]]:
    """year -> {period: value}. The v1 API returns at most 10 years per request."""
    out: dict[int, dict[str, float]] = defaultdict(dict)
    start = first
    while start <= last:
        end = min(start + 9, last)
        d = json.loads(ctx.poster(BLS_URL, bls_body(start, end)).body)
        if d.get("status") != "REQUEST_SUCCEEDED":
            raise ValueError(f"BLS API: {d.get('status')} {d.get('message')}")
        for x in d["Results"]["series"][0]["data"]:
            if re.fullmatch(r"M(0[1-9]|1[0-2])", x["period"]) and x["value"] not in ("-", ""):
                out[int(x["year"])][x["period"]] = float(x["value"])
        start = end + 1
    return out


def build_deflator(monthly: dict[int, dict[str, float]], years: list[int]) -> Deflator:
    annual, used, notes = {}, {}, []
    for y in sorted(monthly):
        m = monthly[y]
        if not m:
            continue
        used[str(y)] = len(m)
        if len(m) == 12:
            annual[str(y)] = round(sum(m.values()) / 12, 3)
    complete = [int(y) for y in annual]
    if not complete:
        raise ValueError("no complete CPI year")
    base = max(complete)
    for y in sorted(monthly):
        n = len(monthly[y])
        if 0 < n < 12 and y < max(monthly):
            missing = sorted(set(f"M{i:02d}" for i in range(1, 13)) - set(monthly[y]))
            notes.append(f"{y}: {n} of 12 monthly values published (missing {', '.join(missing)}); no annual average is computed for {y}, so it cannot be the base year.")
    lacking = [y for y in years if str(y) not in annual]
    if lacking:
        raise ValueError(f"no complete CPI year for {lacking}")
    return Deflator(
        series_id=CPI_SERIES,
        title="CPI-U, U.S. city average, all items, not seasonally adjusted (BLS)",
        source_url=f"https://data.bls.gov/timeseries/{CPI_SERIES}",
        base_year=base,
        annual_index={k: annual[k] for k in sorted(annual)},
        months_used=used,
        method=f"Annual average = mean of the 12 published monthly values, rounded to 3 decimals (as BLS publishes). Real value = nominal × CPI({base}) / CPI(year). Base year = most recent year with all 12 monthly values published.",
        notes=notes,
    )


def assign_group(name: str) -> str | None:
    hits = [g["id"] for g in GROUPS if any(re.fullmatch(p, name) for p in g["names"])]
    if len(hits) > 1:
        raise ValueError(f"species name {name!r} matches more than one group: {hits}")
    return hits[0] if hits else None


def find_duplicates(rows: list[dict]) -> list[tuple[dict, dict]]:
    """Rows in the same year with different names but identical non-zero pounds and dollars.
    Returns (excluded, kept). Keeps the name without the '**' marker, else the first by name."""
    by: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["ts_afs_name"] == WITHHELD or not r.get("pounds") or not r.get("dollars"):
            continue
        by[(r["year"], r["pounds"], r["dollars"])].append(r)
    out = []
    for group in by.values():
        names = {r["ts_afs_name"] for r in group}
        if len(group) < 2 or len(names) < 2:
            continue
        ordered = sorted(group, key=lambda r: ("**" in r["ts_afs_name"], r["ts_afs_name"]))
        keep = ordered[0]
        out.extend((r, keep) for r in ordered[1:])
    return out


def _yv(year: int, rows: list[dict], cpi: Deflator) -> YearValue:
    with_value = [r for r in rows if r.get("dollars") is not None]
    pounds = [r["pounds"] for r in rows if r.get("pounds") is not None]
    nominal = round(sum(r["dollars"] for r in with_value), 2) if with_value else None
    return YearValue(
        year=year,
        pounds=round(sum(pounds), 1) if pounds else None,
        dollars_nominal=nominal,
        dollars_real=round(nominal * cpi.annual_index[str(cpi.base_year)] / cpi.annual_index[str(year)], 2) if nominal is not None else None,
        n_rows=len(rows),
        n_rows_without_value=len(rows) - len(with_value),
    )


def aggregate(rows_by_year: dict[int, list[dict]], cpi: Deflator):
    years = sorted(y for y, rows in rows_by_year.items() if rows)
    all_rows = [r for y in years for r in rows_by_year[y]]
    dups = find_duplicates(all_rows)
    excluded_ids = {id(r) for r, _ in dups}
    groups: dict[str, dict[int, list[dict]]] = {g["id"]: defaultdict(list) for g in GROUPS}
    names: dict[str, set[str]] = {g["id"]: set() for g in GROUPS}
    total: dict[int, list[dict]] = defaultdict(list)
    withheld: dict[int, list[dict]] = defaultdict(list)
    for r in all_rows:
        if id(r) in excluded_ids:
            continue
        name = r["ts_afs_name"] or ""
        if name == WITHHELD:
            withheld[r["year"]].append(r)
            continue
        total[r["year"]].append(r)
        gid = assign_group(name)
        if gid:
            groups[gid][r["year"]].append(r)
            names[gid].add(name)
    out_groups = [
        SpeciesGroup(
            id=g["id"],
            label=g["label"],
            tier=g["tier"],
            tier_basis=g["basis"],
            official_record_ids=g["records"],
            source_names=sorted(names[g["id"]]),
            annual=[_yv(y, groups[g["id"]].get(y, []), cpi) for y in years],
        )
        for g in GROUPS
    ]
    out_total = [_yv(y, total.get(y, []), cpi) for y in years]
    out_withheld = []
    for y in years:
        v = _yv(y, withheld.get(y, []), cpi)
        out_withheld.append(
            SuppressedValue(
                year=y,
                pounds=v.pounds,
                dollars_nominal=v.dollars_nominal,
                dollars_real=v.dollars_real,
                note="Landings FOSS withholds for confidentiality, aggregated across species by FOSS. Not attributable to any species or group and not included in group or statewide values.",
            )
        )
    excluded = [
        ExcludedRow(
            year=r["year"],
            source_name=r["ts_afs_name"],
            duplicate_of=k["ts_afs_name"],
            pounds=r["pounds"],
            dollars_nominal=r["dollars"],
            reason="Identical pounds and dollars to another species row in the same year; treated as a duplicate listing and counted once.",
        )
        for r, k in dups
    ]
    return years, out_groups, out_total, out_withheld, excluded


# ------------------------------------------------------------------ port level
PORT_LEVEL = PortLevelStatus(
    status="unavailable",
    reasons=[
        "Port-level landings for 2021 onward are published by CDFW only through the interactive Marine Fisheries Data Explorer (MFDE). CDFW asks users to consult it before using MFDE data, and permission for automated extraction or republication has not been obtained.",
        "The public CALFISH compilation (Dryad, CC0) has port-level landings only to 2019, and its download requires a browser challenge or an API token that this pipeline does not use.",
        "Statewide values below are not divided among ports: any split would be an estimate, and confidential values cannot be redistributed.",
    ],
    adapters=[
        {"id": "cdfw_mfde", "status": "disabled", "note": "Awaiting CDFW permission. When granted, port-level rows enter through the same disclosure-safe aggregation (suppression propagates; complementary suppression applied)."},
        {"id": "calfish_dryad", "status": "disabled", "note": "Needs a manually downloaded, checksum-verified file; covers 1941–2019 only."},
        {"id": "pacfin", "status": "not_used", "note": "PacFIN data-use terms not verified for republication."},
    ],
)


@dataclass
class Cell:
    """One port × species × year value from a port-level source. suppressed=True means the
    source withheld it (value is None)."""

    key: tuple
    value: float | None
    suppressed: bool


@dataclass
class Aggregate:
    key: tuple
    value: float | None
    includes_suppressed: bool
    publishable: bool
    note: str


def aggregate_cells(cells: list[Cell], group_of, source_totals: dict[tuple, float] | None = None) -> list[Aggregate]:
    """Disclosure-safe aggregation for port-level sources (used when a port-level source is
    authorised; tested with synthetic data).

    * A suppressed component is never estimated: an aggregate containing one is marked
      `includes_suppressed`, its value is the disclosed part only, and it is labelled as a
      lower bound.
    * Complementary suppression: if the source publishes a total for the aggregate and
      exactly one component is suppressed, publishing the total would reveal that component
      by subtraction, so the aggregate is not publishable.
    * Each cell contributes to exactly one aggregate (group_of returns one key)."""
    buckets: dict[tuple, list[Cell]] = defaultdict(list)
    for c in cells:
        buckets[group_of(c.key)].append(c)
    out = []
    for k, cs in sorted(buckets.items()):
        n_sup = sum(1 for c in cs if c.suppressed)
        disclosed = [c.value for c in cs if not c.suppressed and c.value is not None]
        value = round(sum(disclosed), 2) if disclosed else None
        if n_sup == 0:
            out.append(Aggregate(k, value, False, True, "All components disclosed."))
        elif n_sup == len(cs):
            out.append(Aggregate(k, None, True, False, "All components withheld by the source."))
        elif n_sup == 1 and source_totals is not None and k in source_totals:
            out.append(Aggregate(k, None, True, False, "Withheld: the published total minus the disclosed components would reveal a confidential value."))
        else:
            out.append(Aggregate(k, value, True, True, f"At least this value; {n_sup} component(s) withheld by the source are not included."))
    return out


# ------------------------------------------------------------------ run
@dataclass
class FisheriesResult:
    dataset: FisheriesDataset | None
    errors: list[str]
    notes: list[str]
    latest_year: int | None


def run(ctx: RunContext) -> FisheriesResult:
    last = ctx.now.year - 1
    requested = list(range(FIRST_YEAR, last + 1))
    notes: list[str] = []
    rows_by_year: dict[int, list[dict]] = {}
    try:
        for y in requested:
            rows_by_year[y] = fetch_foss_year(ctx, y)
    except Exception as e:
        return FisheriesResult(None, [f"FOSS landings unavailable: {e}"[:300]], notes, None)
    unavailable = [y for y in requested if not rows_by_year.get(y)]
    years_present = [y for y in requested if rows_by_year.get(y)]
    if not years_present:
        return FisheriesResult(None, ["FOSS returned no California commercial landings"], notes, None)
    if unavailable:
        notes.append(f"FOSS has not published California commercial landings for {', '.join(map(str, unavailable))} yet.")
    try:
        monthly = fetch_cpi(ctx, years_present[0], ctx.now.year)
        cpi = build_deflator(monthly, years_present)
    except Exception as e:
        return FisheriesResult(None, [f"CPI deflator unavailable: {e}"[:300]], notes, None)
    years, groups, total, withheld, excluded = aggregate({y: rows_by_year[y] for y in years_present}, cpi)
    if excluded:
        notes.append(f"{len(excluded)} duplicate species row(s) counted once (see excluded_rows).")
    ds = FisheriesDataset(
        generated_at=ctx.now_iso,
        scope="California commercial landings, statewide, by species group and year (all ports and gears combined).",
        years=years,
        years_requested_unavailable=unavailable,
        deflator=cpi,
        groups=groups,
        statewide_total=total,
        withheld=withheld,
        excluded_rows=excluded,
        port_level=PORT_LEVEL,
        terminology="Historical fisheries exposure: the reported value of past commercial landings of species that marine toxins can affect. It is not a prediction of losses, not an estimate of harm, and says nothing about any current or future season.",
        method={
            "source": "NOAA Fisheries FOSS commercial landings for California (compiled from PacFIN), one request per year; every row is kept with its upstream species name.",
            "groups": "Each upstream species row is assigned to at most one group by exact name; rows in no group count only toward the statewide total.",
            "withheld": "FOSS's 'WITHHELD FOR CONFIDENTIALITY' row is reported separately for each year and never assigned to a species or group.",
            "missing": "Rows without a published value are counted (n_rows_without_value) and never treated as zero.",
            "duplicates": "Rows in the same year with different names but identical non-zero pounds and dollars are counted once; the excluded rows are listed.",
            "inflation": cpi.method,
            "tiers": "Tier 1: fisheries with commercial closures or take restrictions for marine toxins (current or past). Tier 2: species under consumption advisories, sport-harvest quarantines or monitoring. " + TIER_REVIEW,
        },
        caveats=[
            "Historical fisheries exposure is not a forecast of losses. Past landings say nothing about whether a fishery will be closed.",
            "Values are ex-vessel revenue reported for landings; they exclude processing, retail, tourism and other economic activity.",
            "Statewide values cannot be divided among ports. Confidential landings are withheld by the source and not included in species values.",
            "Upstream species categories are market categories; some combine several species.",
            TIER_REVIEW,
        ],
        provenance=[
            Provenance(
                source_id=SOURCE_ID,
                source_name="NOAA Fisheries FOSS — Commercial Landings (California, PacFIN)",
                source_url=FOSS_PAGE,
                dataset_id="foss/landings",
                institution="NOAA Fisheries Office of Science and Technology",
                license=FOSS_LICENSE,
                citation=f"NOAA Fisheries Office of Science and Technology, Commercial Landings Query, available at www.fisheries.noaa.gov/foss, accessed {ctx.now.date().isoformat()}.",
                retrieved_at=ctx.now_iso,
                request_urls=[foss_url(y) for y in requested[-3:]],
                pipeline_version=ctx.pipeline_version,
                pipeline_run_id=ctx.run_id,
            ),
            Provenance(
                source_id="bls_cpi_u",
                source_name="BLS Consumer Price Index for All Urban Consumers (CPI-U), CUUR0000SA0",
                source_url=f"https://data.bls.gov/timeseries/{CPI_SERIES}",
                dataset_id=CPI_SERIES,
                institution="U.S. Bureau of Labor Statistics",
                license=BLS_LICENSE,
                retrieved_at=ctx.now_iso,
                request_urls=[BLS_URL],
                pipeline_version=ctx.pipeline_version,
                pipeline_run_id=ctx.run_id,
            ),
        ],
    )
    return FisheriesResult(ds, [], notes, years[-1])
