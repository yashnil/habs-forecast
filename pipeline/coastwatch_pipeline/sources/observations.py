"""Measured HAB observations: CalHABMAP shore stations (SCCOOS ERDDAP tabledap).

One weekly-ish water sample per station visit: particulate / dissolved / total domoic
acid (ng/mL of seawater), Pseudo-nitzschia cell abundance by size group (cells/L),
extracted chlorophyll (mg/m3) and water temperature. Rules this module enforces:

* NaN upstream means "not measured in this sample" -> null. Never zero.
* A reported 0 is kept as 0 with the qualifier `reported_zero`. The dataset publishes no
  detection or quantification limit, so 0 means "not quantified", not "absent".
* Negative values are rejected (null + `rejected_negative`), never silently dropped.
* Values above a plausibility ceiling are kept and flagged (`flag_high`) for review.
* Fractions (pDA/dDA/tDA), size groups and variables are never combined.
* Units are checked against the expected units on every request; a mismatch fails the
  station instead of publishing mislabelled numbers.
* A station whose request fails keeps its previously published data with their real
  dates (status `carried_forward`); it never blocks the other stations.

A shore station describes the water sampled at that pier on that day. It does not
describe surrounding waters, seafood, or any other location.
"""

from __future__ import annotations

import csv
import io
import math
import statistics
import time as _time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np

from ..context import RunContext
from ..http import FetchError
from ..models import (
    LayerArtifact,
    ObservationDataset,
    ObsSeries,
    ObsStation,
    ObsVariable,
    ObsVariableSummary,
    PortsCollection,
    Provenance,
    QCCheck,
    StationCharm,
)
from . import port_intel

SOURCE_ID = "calhabmap"
SERVER = "https://erddap.sccoos.org/erddap"
WINDOW_START = "2014-01-01"
CHARM_HISTORY_DAYS = 180
RETRY_ATTEMPTS = 3
RETRY_SLEEP_S = 20.0  # SCCOOS ERDDAP briefly answers 404/502 while it reloads datasets
LICENSE = (
    "The data may be used and redistributed for free but is not intended for legal use, since it may contain "
    "inaccuracies. Neither the data Contributor, ERD, NOAA, nor the United States Government, nor any of their "
    "employees or contractors, makes any warranty, express or implied, including warranties of merchantability "
    "and fitness for a particular purpose, or assumes any legal liability for the accuracy, completeness, or "
    "usefulness, of this information."
)

# dataset id suffix -> display name (all 17 CalHABMAP datasets on SCCOOS ERDDAP, 2026-10-08)
STATIONS: dict[str, str] = {
    "TrinidadPier": "Trinidad Pier",
    "Humboldt": "Humboldt Bay",
    "HumboldtSouthBay": "Humboldt Bay, South Bay",
    "BodegaMarineLab": "Bodega Marine Lab",
    "BodegaMarineLabBuoy": "Bodega Marine Lab buoy",
    "TomalesBayMouth": "Tomales Bay mouth",
    "TomalesBayMid-ChannelBuoy": "Tomales Bay mid-channel buoy",
    "InnerTomalesBay": "Inner Tomales Bay",
    "SantaCruzWharf": "Santa Cruz Wharf",
    "MontereyWharf": "Monterey Wharf",
    "MorroBayFrontBay": "Morro Bay, front bay",
    "MorroBayBackBay": "Morro Bay, back bay",
    "CalPolyPier": "Cal Poly Pier (Avila Beach)",
    "StearnsWharf": "Stearns Wharf (Santa Barbara)",
    "SantaMonicaPier": "Santa Monica Pier",
    "NewportBeachPier": "Newport Beach Pier",
    "ScrippsPier": "Scripps Pier (La Jolla)",
}

_NOT_PUBLISHED_DL = "No detection or quantification limit is published in the dataset metadata."
_ZERO_TOXIN = (
    "A reported 0 is shown as 'reported 0 (not quantified)'. Because no detection limit is published, "
    "0 means the value was at or below the laboratory's limit, not that toxin was absent."
)
_ZERO_CELLS = (
    "A reported 0 means no cells of this group were counted in the volume examined; cells may still be "
    "present below the counting limit."
)

VARIABLES: list[ObsVariable] = [
    ObsVariable(
        id="pDA",
        source_variable="pDA",
        label="Particulate domoic acid",
        upstream_long_name="Domoic Acid-pDA",
        units="ng/mL",
        kind="toxin",
        matrix="seawater",
        fraction="particulate",
        method="Toxin retained on a filter from a measured volume of seawater (DA_Volume_Filtered, mL). The analytical method is not recorded in the dataset metadata.",
        detection_limit_note=_NOT_PUBLISHED_DL,
        zero_policy=_ZERO_TOXIN,
        plausible_max=500.0,
    ),
    ObsVariable(
        id="dDA",
        source_variable="dDA",
        label="Dissolved domoic acid",
        upstream_long_name="Domoic Acid-dDA",
        units="ng/mL",
        kind="toxin",
        matrix="seawater",
        fraction="dissolved",
        method="Dissolved fraction of domoic acid in seawater. The analytical method is not recorded in the dataset metadata.",
        detection_limit_note=_NOT_PUBLISHED_DL,
        zero_policy=_ZERO_TOXIN,
        plausible_max=500.0,
    ),
    ObsVariable(
        id="tDA",
        source_variable="tDA",
        label="Total domoic acid",
        upstream_long_name="Domoic Acid-tDA",
        units="ng/mL",
        kind="toxin",
        matrix="seawater",
        fraction="total",
        method="Total (particulate + dissolved) domoic acid in seawater. The analytical method is not recorded in the dataset metadata.",
        detection_limit_note=_NOT_PUBLISHED_DL,
        zero_policy=_ZERO_TOXIN,
        plausible_max=1000.0,
    ),
    ObsVariable(
        id="pn_seriata",
        source_variable="Pseudo_nitzschia_seriata_group",
        label="Pseudo-nitzschia, seriata group (larger cells)",
        upstream_long_name="Pseudo-nitzschia seriata group",
        units="cells/L",
        kind="cell_abundance",
        matrix="seawater",
        fraction=None,
        method="Microscope count of settled seawater (Volume_Settled_for_Counting, mL). Counts a genus size group, not toxicity.",
        detection_limit_note="The counting limit depends on the settled volume, which varies by sample; it is not published as a limit.",
        zero_policy=_ZERO_CELLS,
        plausible_max=1.0e8,
    ),
    ObsVariable(
        id="pn_delicatissima",
        source_variable="Pseudo_nitzschia_delicatissima_group",
        label="Pseudo-nitzschia, delicatissima group (smaller cells)",
        upstream_long_name="Pseudo-nitzschia delicatissima group",
        units="cells/L",
        kind="cell_abundance",
        matrix="seawater",
        fraction=None,
        method="Microscope count of settled seawater (Volume_Settled_for_Counting, mL). Counts a genus size group, not toxicity.",
        detection_limit_note="The counting limit depends on the settled volume, which varies by sample; it is not published as a limit.",
        zero_policy=_ZERO_CELLS,
        plausible_max=1.0e8,
    ),
    ObsVariable(
        id="chl_extracted",
        source_variable="Avg_Chloro",
        label="Chlorophyll-a (extracted, average of replicates)",
        upstream_long_name="Avg Chloro",
        units="mg/m3",
        kind="pigment",
        matrix="seawater",
        fraction=None,
        method="Chlorophyll extracted from filtered seawater (Chl_Volume_Filtered, mL); average of the replicate measurements. Measures algae biomass, not toxins.",
        detection_limit_note=_NOT_PUBLISHED_DL,
        zero_policy="A reported 0 is shown as 'reported 0 (not quantified)'.",
        plausible_max=500.0,
    ),
    ObsVariable(
        id="temp",
        source_variable="Temp",
        label="Water temperature",
        upstream_long_name="Sea water temperature",
        units="degree_C",
        kind="physical",
        matrix="seawater",
        fraction=None,
        method="Temperature recorded at sampling.",
        detection_limit_note="Not applicable.",
        zero_policy="0 °C is a real temperature value.",
        plausible_max=35.0,
    ),
]
VAR_BY_ID = {v.id: v for v in VARIABLES}
COLUMNS = ["time", "latitude", "longitude", "depth", "Location_Code", "SampleID"] + [v.source_variable for v in VARIABLES]
EXPECTED_UNITS = {"time": "UTC", "latitude": "degrees_north", "longitude": "degrees_east", "depth": "m"} | {
    v.source_variable: v.units for v in VARIABLES
}

CAVEATS = [
    "Each value describes one water sample taken at that pier on that day. It does not describe nearby beaches, offshore waters or fishing grounds.",
    "Domoic acid in seawater is not domoic acid in seafood. Only official agency testing decides whether seafood can be harvested or eaten.",
    "A blank means the quantity was not measured in that sample. No measurement is not the same as no toxin.",
    "A reported 0 means 'not quantified' (at or below an unpublished laboratory limit), not 'absent'.",
    "Pseudo-nitzschia counts are cells of a genus size group; not every Pseudo-nitzschia produces toxin.",
    "Chlorophyll measures algae biomass. High chlorophyll is not a toxic bloom, and low chlorophyll does not rule one out.",
    "Measurements are published after laboratory analysis and can arrive days to weeks after sampling.",
]


def dataset_id(key: str) -> str:
    return f"HABs-{key}"


def request_url(key: str, start: str = WINDOW_START) -> str:
    q = ",".join(COLUMNS) + "&time>=" + f"{start}T00:00:00Z" + '&orderBy("time")'
    return f"{SERVER}/tabledap/{dataset_id(key)}.csv?" + quote(q, safe=',&=()"').replace(">=", "%3E=")


def _num(s: str) -> float | None:
    s = s.strip()
    if s == "" or s.lower() == "nan":
        return None
    v = float(s)
    return None if math.isnan(v) else v


@dataclass
class Parsed:
    times: list[str]
    depths: list[float | None]
    lats: list[float]
    lons: list[float]
    codes: list[str]
    values: dict[str, list[float | None]]
    qc: list[QCCheck]


def parse_csv(body: bytes, now: datetime) -> Parsed:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:  # ERDDAP labels its CSV ISO-8859-1
        text = body.decode("latin-1")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        raise ValueError("response has no header/units rows")
    header, units = rows[0], rows[1]
    missing = [c for c in COLUMNS if c not in header]
    if missing:
        raise ValueError(f"missing columns: {', '.join(missing)}")
    idx = {c: header.index(c) for c in COLUMNS}
    bad_units = [f"{c}: '{units[idx[c]]}' (expected '{u}')" for c, u in EXPECTED_UNITS.items() if units[idx[c]] != u]
    if bad_units:
        raise ValueError("unexpected units: " + "; ".join(bad_units))
    qc = [QCCheck(name="units", passed=True, detail="Units row matches the expected units for every variable.")]

    seen: set[tuple] = set()
    dup = future = 0
    recs = []
    for r in rows[2:]:
        if not r or len(r) < len(header):
            continue
        key = tuple(r)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        t = datetime.fromisoformat(r[idx["time"]].replace("Z", "+00:00"))
        if t > now + timedelta(hours=1):
            future += 1
            continue
        recs.append((t, r))
    recs.sort(key=lambda x: x[0])
    qc.append(QCCheck(name="duplicates", passed=True, detail=f"{dup} exact duplicate row(s) dropped."))
    qc.append(QCCheck(name="time", passed=future == 0, detail=f"{future} row(s) with a sample time in the future dropped."))

    out = Parsed([], [], [], [], [], {v.id: [] for v in VARIABLES}, qc)
    for t, r in recs:
        out.times.append(t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        out.depths.append(_num(r[idx["depth"]]))
        out.lats.append(float(r[idx["latitude"]]))
        out.lons.append(float(r[idx["longitude"]]))
        out.codes.append(r[idx["Location_Code"]])
        for v in VARIABLES:
            out.values[v.id].append(_num(r[idx[v.source_variable]]))
    return out


def qualify(var: ObsVariable, values: list[float | None]) -> tuple[list[float | None], dict[str, str], int, int]:
    """Apply the zero / negative / plausibility rules. Returns (values, qualifiers, n_rejected, n_flagged)."""
    vals: list[float | None] = []
    quals: dict[str, str] = {}
    rejected = flagged = 0
    for i, v in enumerate(values):
        if v is None:
            vals.append(None)
            continue
        if var.kind != "physical" and v < 0:
            vals.append(None)
            quals[str(i)] = "rejected_negative"
            rejected += 1
            continue
        if v == 0 and var.kind != "physical":
            quals[str(i)] = "reported_zero"
        elif var.plausible_max is not None and v > var.plausible_max:
            quals[str(i)] = "flag_high"
            flagged += 1
        vals.append(float(f"{v:.7g}"))  # 7 significant digits: float32 upstream precision, small concentrations intact
    return vals, quals, rejected, flagged


def summarise(var_id: str, times: list[str], series: ObsSeries, today: date) -> ObsVariableSummary:
    measured = [(times[i], v, series.qualifiers.get(str(i))) for i, v in enumerate(series.values) if v is not None]
    rejected = sum(1 for q in series.qualifiers.values() if q == "rejected_negative")
    zeros = sum(1 for _, _, q in measured if q == "reported_zero")
    if not measured:
        return ObsVariableSummary(
            variable=var_id, n_measured=0, n_reported_zero=0, n_rejected=rejected, first_date=None, last_date=None,
            last_value=None, n_last_365d=0, max_last_365d=None,
        )  # fmt: skip
    cutoff = (today - timedelta(days=365)).isoformat()
    recent = [(t, v) for t, v, _ in measured if t[:10] >= cutoff]
    dts = [date.fromisoformat(t[:10]) for t, _ in recent]
    gaps = [(b - a).days for a, b in zip(dts, dts[1:])]
    last_t, last_v, last_q = measured[-1]
    return ObsVariableSummary(
        variable=var_id,
        n_measured=len(measured),
        n_reported_zero=zeros,
        n_rejected=rejected,
        first_date=measured[0][0][:10],
        last_date=last_t[:10],
        last_value=last_v,
        last_qualifier=last_q,
        days_since_last=(today - date.fromisoformat(last_t[:10])).days,
        n_last_365d=len(recent),
        median_interval_days_365d=float(statistics.median(gaps)) if gaps else None,
        max_last_365d=max(v for _, v in recent) if recent else None,
    )


def _fetch_retry(ctx: RunContext, url: str):
    last: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return ctx.fetcher(url)
        except FetchError as e:
            last = e
            if attempt < RETRY_ATTEMPTS - 1 and RETRY_SLEEP_S:
                _time.sleep(RETRY_SLEEP_S)
    assert last is not None
    raise last


def _nearest_port(ports: PortsCollection | None, lat: float, lon: float):
    if not ports or not ports.features:
        return None
    best = min(ports.features, key=lambda f: float(port_intel.km(lat, lon, f.geometry["coordinates"][1], f.geometry["coordinates"][0])))
    d = float(port_intel.km(lat, lon, best.geometry["coordinates"][1], best.geometry["coordinates"][0]))
    return best.properties, round(d, 1)


def build_station(ctx: RunContext, key: str, body: bytes, url: str, ports: PortsCollection | None) -> ObsStation:
    p = parse_csv(body, ctx.now)
    qc = list(p.qc)
    if not p.times:
        raise ValueError("no samples in the requested window")
    # position: report the most recent; flag when the station moved
    spread = max(max(p.lats) - min(p.lats), max(p.lons) - min(p.lons))
    codes = sorted(set(c for c in p.codes if c))
    qc.append(
        QCCheck(
            name="position",
            passed=spread <= 0.01,
            detail=(
                f"Sampling position varies by up to {spread:.3f}° across the record (location codes: {', '.join(codes) or 'none'}); the most recent position is shown."
                if spread > 0.01
                else f"Single sampling position (location code {', '.join(codes) or 'none'})."
            ),
        )
    )
    series, summaries = [], []
    n_rej = n_flag = 0
    today = ctx.now.date()
    for v in VARIABLES:
        vals, quals, rej, flag = qualify(v, p.values[v.id])
        n_rej += rej
        n_flag += flag
        s = ObsSeries(variable=v.id, values=vals, qualifiers=quals)  # type: ignore[arg-type]
        series.append(s)
        summaries.append(summarise(v.id, p.times, s, today))
    qc.append(QCCheck(name="negative_values", passed=n_rej == 0, detail=f"{n_rej} negative value(s) rejected (shown as rejected, not as missing or zero)."))
    qc.append(QCCheck(name="plausibility", passed=n_flag == 0, detail=f"{n_flag} value(s) above the plausibility ceiling kept and flagged for review."))
    lat, lon = p.lats[-1], p.lons[-1]
    near = _nearest_port(ports, lat, lon)
    region = next(
        (r.id for r in (ports.regions if ports else []) if r.bounds[0][0] <= lon <= r.bounds[1][0] and r.bounds[0][1] <= lat <= r.bounds[1][1]),
        near[0].region if near else None,
    )
    return ObsStation(
        station_id=dataset_id(key),
        name=STATIONS[key],
        location_code=p.codes[-1] or None,
        lat=lat,
        lon=lon,
        region=region,
        nearest_port_code=near[0].port_code if near else None,
        nearest_port_name=near[0].display_name if near else None,
        nearest_port_km=near[1] if near else None,
        status="updated",
        retrieved_at=ctx.now_iso,
        source_url=f"{SERVER}/tabledap/{dataset_id(key)}.html",
        request_url=url,
        sample_times=p.times,
        depths_m=p.depths,
        series=series,
        summaries=summaries,
        qc=qc,
    )


def station_charm(ctx: RunContext, st: ObsStation, layers: list[LayerArtifact]) -> StationCharm:
    out = StationCharm(radius_km=port_intel.RADIUS_KM, nearest_cell_km=None, history_days=CHARM_HISTORY_DAYS)
    grids = port_intel.load_grids(ctx, layers)
    nowcast = grids.get(("pseudo_nitzschia", 0))
    if not nowcast:
        out.error = "No C-HARM run available in this dataset."
        return out
    assert st.lat is not None and st.lon is not None
    near = port_intel.charm_near(grids, st.lat, st.lon)
    out.nearest_cell_km = near.nearest_cell_km if near else None
    end = date.fromisoformat(nowcast.layer.time.valid_date) if nowcast.layer.time.valid_date else ctx.now.date()
    try:
        out.history = port_intel.charm_history(ctx, st.lat, st.lon, end, days=CHARM_HISTORY_DAYS)
    except Exception as e:
        out.error = f"C-HARM nowcast history unavailable: {e}"[:300]
    return out


@dataclass
class ObservationResult:
    dataset: ObservationDataset | None
    errors: list[str]
    notes: list[str]
    latest_sample: str | None


def run(
    ctx: RunContext,
    ports: PortsCollection | None,
    layers: list[LayerArtifact],
    previous: ObservationDataset | None,
    keys: list[str] | None = None,
) -> ObservationResult:
    prev_by_id = {s.station_id: s for s in previous.stations} if previous else {}
    stations: list[ObsStation] = []
    notes: list[str] = []
    errors: list[str] = []
    for key in keys or list(STATIONS):
        url = request_url(key)
        try:
            body = _fetch_retry(ctx, url).body
            st = build_station(ctx, key, body, url, ports)
        except Exception as e:
            msg = f"{STATIONS[key]}: {e}"[:300]
            prev = prev_by_id.get(dataset_id(key))
            if prev and prev.sample_times:
                st = prev.model_copy(update={"status": "carried_forward", "error": msg})
                notes.append(f"{msg} (previous data kept with their original dates)")
            else:
                errors.append(msg)
                st = ObsStation(
                    station_id=dataset_id(key), name=STATIONS[key], lat=None, lon=None, status="failed",
                    error=msg, source_url=f"{SERVER}/tabledap/{dataset_id(key)}.html", request_url=url,
                    sample_times=[], depths_m=[], series=[], summaries=[], qc=[],
                )  # fmt: skip
        else:
            st.charm = station_charm(ctx, st, layers)
            if st.charm.error:
                notes.append(f"{st.name}: {st.charm.error}")
        stations.append(st)
    if all(s.status == "failed" for s in stations):
        return ObservationResult(None, errors or ["no station data"], notes, None)
    latest = max((s.sample_times[-1] for s in stations if s.sample_times), default=None)
    ds = ObservationDataset(
        generated_at=ctx.now_iso,
        window_start=WINDOW_START,
        program="California Harmful Algal Bloom Monitoring and Alert Program (CalHABMAP)",
        variables=VARIABLES,
        stations=stations,
        method={
            "sampling": "Shore-station water samples, typically weekly, collected and analysed by CalHABMAP member laboratories and published on SCCOOS ERDDAP.",
            "missing": "A blank (null) means the quantity was not measured in that sample. It is never shown as zero.",
            "zeros": "A reported 0 is kept as 0 with the qualifier 'reported_zero' and shown as 'reported 0 (not quantified)'. No detection or quantification limit is published.",
            "qc": "Units are checked on every request; exact duplicate rows and future-dated rows are dropped; negative concentrations or counts are rejected and labelled; values above a plausibility ceiling are kept and flagged.",
            "frequency": "Measurement frequency is the median number of days between measurements of that variable in the last 365 days.",
            "charm": f"C-HARM nowcast median over model cells within {port_intel.RADIUS_KM:.0f} km of the station, last {CHARM_HISTORY_DAYS} days. A model probability, shown beside the measurements on the same time axis; never compared with them numerically.",
            "failures": "A station whose request fails keeps its previously published values with their original dates and is marked as carried forward.",
        },
        caveats=CAVEATS,
        provenance=Provenance(
            source_id=SOURCE_ID,
            source_name="CalHABMAP shore-station HAB monitoring (SCCOOS ERDDAP)",
            source_url="https://calhabmap.org",
            dataset_id="HABs-* (17 datasets)",
            institution="CalHABMAP; served by SCCOOS",
            license=LICENSE,
            citation="California Harmful Algal Bloom Monitoring and Alert Program (CalHABMAP), via SCCOOS ERDDAP (erddap.sccoos.org).",
            retrieved_at=ctx.now_iso,
            request_urls=[request_url(k) for k in (keys or list(STATIONS))][:3],
            upstream_metadata={"summary": "CalHABMAP collects weekly phytoplankton and water quality data at piers along the California coast."},
            pipeline_version=ctx.pipeline_version,
            pipeline_run_id=ctx.run_id,
        ),
    )
    return ObservationResult(ds, errors, notes, latest)

