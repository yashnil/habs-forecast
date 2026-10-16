"""Port-level summaries for the 22 CDFW landing ports.

For each port:
* C-HARM near the port: statistics over forecast cells within RADIUS_KM, per lead and
  variable, read from the *published* value grids (so they match the map and inspector).
* C-HARM nowcast history: the last HISTORY_DAYS of nowcasts from ERDDAP, median over the
  same neighbourhood; missing runs stay missing (no interpolation).
* Satellite chlorophyll: an 8-day window computed by Neritic from the daily NOAA-21 VIIRS
  750 m merges, with NOAA-20 in cells NOAA-21 has no value for (no NOAA-20/21 750 m 8-day
  product exists): each pixel's median over the window, then the median of those pixels within
  RADIUS_KM; cloud/land pixels excluded. Until S-NPP data end (2026-11-02) the S-NPP 8-day
  composite (erdVHNchla8day) is the fallback when neither NOAA-21 nor NOAA-20 answers, and the
  port's dataset_id says which was used.
* Official records whose stated area may include waters near the port, with the reason.

A neighbourhood statistic describes the forecast grid near a port. It does not describe
conditions at the port or at any fishing ground.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np

from ..context import RunContext
from ..models import (
    LayerArtifact,
    Manifest,
    OfficialDataset,
    PortChlorophyll,
    PortCharm,
    PortIntel,
    PortIntelCollection,
    PortLeadSummary,
    PortOfficialRelation,
    PortsCollection,
    Provenance,
    SeriesPoint,
    Stat,
)
from ..process import grid as gridcodec
from . import charm

SOURCE_ID = "port_intel"
RADIUS_KM = 15.0
HISTORY_DAYS = 30
CHL_DAYS = 60
# NOAA-21 first, NOAA-20 only where NOAA-21 has no value (docs/coastwatch/33-suomi-npp-continuity.md)
CHL_PLATFORMS = (("NOAA-21", "noaacwN21VIIRSchlaSectorUSDaily"), ("NOAA-20", "noaacwN20VIIRSchlaSectorUSDaily"))
CHL_JPSS_SERVER = "https://coastwatch.noaa.gov/erddap"
CHL_WINDOW_DAYS = 8
# window of a point dated D: days D-3 .. D+4 (8 days; the centre falls between D and D+1)
CHL_WINDOW_BEFORE, CHL_WINDOW_AFTER = 3, 4
CHL_SOURCE_ID = "viirs_jpss_chl_8day"
# ERDDAP opens one file per day, so a 64-day request takes 10-16 s per platform and port
# (2026-10-13). Complete windows older than this many days before the newest published point
# are kept from the previous port summary (same datasets); newer ones are recomputed, so late
# near-real-time passes are picked up. A normal run then asks for about two weeks of days.
CHL_REFRESH_DAYS = 7
# S-NPP 8-day composites: fallback until S-NPP data end, then they stop updating
CHL_DATASET = "erdVHNchla8day"
CHL_SERVER = "https://coastwatch.pfeg.noaa.gov/erddap"
NAMED_AREA_KM = 60.0
KM_PER_DEG = 111.32

CAVEATS = [
    "Values summarise forecast or satellite cells within 15 km of the CDFW port location. They do not describe conditions at the dock or at any particular fishing ground.",
    "C-HARM probabilities describe the water column, not toxin in seafood; low values do not mean an area is safe.",
    "Chlorophyll measures algae biomass, not toxins.",
    "Official records are listed when their stated area may include nearby waters; the agency's wording decides where they apply.",
]


def km(lat1: float, lon1: float, lat2: np.ndarray | float, lon2: np.ndarray | float):
    dy = (np.asarray(lat2) - lat1) * KM_PER_DEG
    dx = (np.asarray(lon2) - lon1) * KM_PER_DEG * math.cos(math.radians(lat1))
    return np.hypot(dx, dy)


def stat(values: np.ndarray) -> Stat:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return Stat(n=0, median=None, min=None, max=None)
    return Stat(n=int(v.size), median=round(float(np.median(v)), 6), min=round(float(v.min()), 6), max=round(float(v.max()), 6))


@dataclass
class Grid:
    layer: LayerArtifact
    values: np.ndarray
    lats: np.ndarray
    lons: np.ndarray


def load_grids(ctx: RunContext, manifest_layers: list[LayerArtifact]) -> dict[tuple[str, int], Grid]:
    out = {}
    for lyr in manifest_layers:
        if lyr.group_id != "charm" or not lyr.grid:
            continue
        g = lyr.grid
        vals = gridcodec.decode((ctx.out_dir / g.url).read_bytes(), g.width, g.height, g.scale_factor, g.add_offset, g.nodata)
        lats = g.lat_first + np.arange(g.height) * g.lat_step
        lons = g.lon_first + np.arange(g.width) * g.lon_step
        out[(lyr.variable, lyr.time.lead_days)] = Grid(lyr, vals, lats, lons)
    return out


def charm_near(grids: dict[tuple[str, int], Grid], lat: float, lon: float) -> PortCharm | None:
    if not grids:
        return None
    any_grid = next(iter(grids.values()))
    LA, LO = np.meshgrid(any_grid.lats, any_grid.lons, indexing="ij")
    d = km(lat, lon, LA, LO)
    within = d <= RADIUS_KM
    pn0 = grids.get(("pseudo_nitzschia", 0)) or any_grid
    valid = np.isfinite(pn0.values)
    nearest = float(d[valid].min()) if valid.any() else None
    leads: dict[int, PortLeadSummary] = {}
    for (var, lead), g in sorted(grids.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ls = leads.setdefault(lead, PortLeadSummary(lead_days=lead, valid_date=g.layer.time.valid_date or "", variables={}))
        ls.variables[var] = stat(g.values[within])
    return PortCharm(
        issued_date=any_grid.layer.time.issued_date,
        radius_km=RADIUS_KM,
        cells_in_radius=int((within & valid).sum()),
        nearest_cell_km=round(nearest, 2) if nearest is not None else None,
        leads=[leads[k] for k in sorted(leads)],
    )


# A nowcast history that could not be refreshed keeps the previously published one when it
# reaches to within this many days of the current run (a missed 6-hourly refresh or two), so
# a NOAA outage does not blank it. An older history is not shown as recent: the error stands.
CARRY_HISTORY_MAX_GAP_DAYS = 2


def carried_history(prev: dict[str, list[SeriesPoint]] | None, end: date) -> dict[str, list[SeriesPoint]] | None:
    """The previous history, if it is recent enough to stand in for one that failed to refresh.
    Points keep their own dates; nothing is extended or filled."""
    if not prev:
        return None
    last = max((p.date for pts in prev.values() for p in pts if p.value is not None), default=None)
    if last is None or date.fromisoformat(last[:10]) < end - timedelta(days=CARRY_HISTORY_MAX_GAP_DAYS):
        return None
    return prev


def charm_history(ctx: RunContext, lat: float, lon: float, end: date, days: int = HISTORY_DAYS) -> dict[str, list[SeriesPoint]]:
    from .charm import parse_netcdf

    start = end - timedelta(days=days - 1)
    dlat, dlon = RADIUS_KM / KM_PER_DEG + 0.02, RADIUS_KM / (KM_PER_DEG * math.cos(math.radians(lat))) + 0.02
    lon360 = lon + 360
    # clip to the published C-HARM domain; ERDDAP rejects out-of-range requests
    d = charm.DOMAIN
    la0, la1 = max(lat - dlat, d["lat_min"]), min(lat + dlat, d["lat_max"])
    lo0, lo1 = max(lon360 - dlon, d["lon_min"]), min(lon360 + dlon, d["lon_max"])
    sel = (
        f"[({start.isoformat()}T12:00:00Z):({end.isoformat()}T12:00:00Z)]"
        f"[({la0:.3f}):({la1:.3f})][({lo0:.3f}):({lo1:.3f})]"
    )
    q = ",".join(f"{v.name}{sel}" for v in charm.VARIABLES)
    url = f"{charm.SERVER}/griddap/{charm.dataset_id(0)}.nc?" + quote(q, safe=",():")
    arrays, vattrs, _ = parse_netcdf(ctx.fetcher(url).body)
    times = [datetime.fromtimestamp(float(t), tz=timezone.utc).date() for t in np.atleast_1d(arrays["time"])]
    lats, lons = arrays["latitude"].astype(float), arrays["longitude"].astype(float) - 360
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    within = km(lat, lon, LA, LO) <= RADIUS_KM
    out: dict[str, list[SeriesPoint]] = {}
    for v in charm.VARIABLES:
        a = arrays[v.name].astype(float)
        fv = vattrs[v.name].get("_FillValue")
        if isinstance(fv, (int, float)):
            a[np.isclose(a, float(fv))] = np.nan
        a[(a < 0) | (a > 1)] = np.nan
        series = []
        for i, t in enumerate(times):
            s = stat(a[i][within])
            series.append(SeriesPoint(date=t.isoformat(), value=s.median, n=s.n))
        out[v.name] = series
    return out


def _jpss_daily(ctx: RunContext, ds: str, lat: float, lon: float, start: date) -> tuple[dict[date, np.ndarray], np.ndarray, np.ndarray]:
    """Daily merges of one platform near a port: {UTC date: values}, latitudes, longitudes."""
    from .charm import parse_netcdf

    dlat, dlon = RADIUS_KM / KM_PER_DEG + 0.01, RADIUS_KM / (KM_PER_DEG * math.cos(math.radians(lat))) + 0.01
    # latitude is descending; stride 2 (~1.5 km) as before. The two platforms share the
    # lattice, so the same bounds give the same cells (checked by the caller).
    sel = (
        f"[({start.isoformat()}T00:00:00Z):(last)][(0.0)]"
        f"[({lat + dlat:.4f}):2:({lat - dlat:.4f})][({lon - dlon:.4f}):2:({lon + dlon:.4f})]"
    )
    url = f"{CHL_JPSS_SERVER}/griddap/{ds}.nc?" + quote(f"chlor_a{sel}", safe=",():")
    arrays, vattrs, _ = parse_netcdf(ctx.fetcher(url).body)
    a = np.asarray(arrays["chlor_a"], dtype=float)
    if a.ndim == 4:
        a = a[:, 0]
    for key in ("_FillValue", "missing_value"):
        fv = vattrs["chlor_a"].get(key)
        if isinstance(fv, (int, float)):
            a[np.isclose(a, float(fv))] = np.nan
    a[(a <= 0) | (a > 200)] = np.nan
    out: dict[date, np.ndarray] = {}
    for i, t in enumerate(np.atleast_1d(arrays["time"])):
        out[datetime.fromtimestamp(float(t), tz=timezone.utc).date()] = a[i]  # one merge per UTC day
    return out, arrays["latitude"].astype(float), arrays["longitude"].astype(float)


def chlorophyll_jpss(ctx: RunContext, lat: float, lon: float, end: date, prev: PortChlorophyll | None = None) -> tuple[PortChlorophyll | None, list[str]]:
    """8-day windows of the daily NOAA-21/NOAA-20 pixels near a port. Returns (None, errors) if
    neither platform answered. Complete windows of `prev` (the previous summary, if made from
    both platforms) up to CHL_REFRESH_DAYS before its newest point are kept, not refetched."""
    full_ids = ",".join(ds for _, ds in CHL_PLATFORMS)
    first = end - timedelta(days=CHL_DAYS - 1)  # first window centre shown
    keep: list[SeriesPoint] = []
    cutoff = None
    if prev and not prev.error and prev.dataset_id == full_ids and prev.history:
        cutoff = date.fromisoformat(prev.history[-1].date) - timedelta(days=CHL_REFRESH_DAYS)
        keep = [pt for pt in prev.history if first.isoformat() <= pt.date <= cutoff.isoformat()]
        if not keep:
            cutoff = None
    start = (cutoff + timedelta(days=1 - CHL_WINDOW_BEFORE)) if cutoff else end - timedelta(days=CHL_DAYS + CHL_WINDOW_BEFORE)
    got: list[tuple[str, str, dict[date, np.ndarray]]] = []
    errors: list[str] = []
    coords = None
    for plat, ds in CHL_PLATFORMS:
        try:
            days, la, lo = _jpss_daily(ctx, ds, lat, lon, start)
        except Exception as e:
            errors.append(f"{plat} {ds}: {e}"[:300])
            continue
        if coords is not None and (la.shape != coords[0].shape or lo.shape != coords[1].shape or np.abs(la - coords[0]).max() > 1e-6 or np.abs(lo - coords[1]).max() > 1e-6):
            errors.append(f"{plat} {ds}: cells differ from {got[0][0]}'s; not combined")
            continue
        coords = coords or (la, lo)
        got.append((plat, ds, days))
    if not got or coords is None:
        return None, errors
    LA, LO = np.meshgrid(coords[0], coords[1], indexing="ij")
    within = km(lat, lon, LA, LO) <= RADIUS_KM
    # per day, per pixel: the first platform with a value (never an average of platforms)
    all_days = sorted({d for _, _, days in got for d in days})
    merged: dict[date, np.ndarray] = {}
    for d in all_days:
        out = np.full(LA.shape, np.nan)
        for _, _, days in got:
            if d in days:
                fill = np.isnan(out) & np.isfinite(days[d])
                out[fill] = days[d][fill]
        merged[d] = out
    last = all_days[-1] if all_days else None
    if ",".join(ds for _, ds, _ in got) != full_ids:
        keep, cutoff = [], None  # kept points came from both platforms; this run's do not
    base = PortChlorophyll(
        dataset_id=",".join(ds for _, ds, _ in got),
        source_url=f"{CHL_JPSS_SERVER}/griddap/{got[0][1]}.html",
        radius_km=RADIUS_KM,
        composite_days=CHL_WINDOW_DAYS,
        latest_center_date=None,
        latest=None,
        latest_valid_fraction=None,
    )
    hist = list(keep)
    if prev and cutoff and prev.latest and prev.latest_center_date and prev.latest_center_date <= cutoff.isoformat() and keep:
        base.latest_center_date, base.latest, base.latest_valid_fraction = prev.latest_center_date, prev.latest, prev.latest_valid_fraction
    if last is not None:
        # only complete windows: the last day of the window must be in the archive already
        centre = max(first, all_days[0] + timedelta(days=CHL_WINDOW_BEFORE), *( [cutoff + timedelta(days=1)] if cutoff else []))
        while centre + timedelta(days=CHL_WINDOW_AFTER) <= last:
            window = [merged[d] for k in range(-CHL_WINDOW_BEFORE, CHL_WINDOW_AFTER + 1) if (d := centre + timedelta(days=k)) in merged]
            if window:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels stay NaN
                    per_pixel = np.nanmedian(np.stack(window), axis=0)
            else:
                per_pixel = np.full(LA.shape, np.nan)
            s = stat(per_pixel[within])
            hist.append(SeriesPoint(date=centre.isoformat(), value=s.median, n=s.n))
            if s.n > 0:
                base.latest_center_date = centre.isoformat()
                base.latest = s
                base.latest_valid_fraction = round(s.n / int(within.sum()), 4) if within.sum() else None
            centre += timedelta(days=1)
    base.history = hist
    return base, errors


# The newest complete NOAA-21/NOAA-20 window is centred about 6 days back (4 days after the centre,
# plus ~1.5 days latency). Centred further back than this, NOAA is lagging (an ERDDAP index froze
# at 2026-10-05 for 9 days) and the S-NPP composite is asked whether it has something newer.
CHL_LAG_DAYS = 8


def chlorophyll_near(ctx: RunContext, lat: float, lon: float, end: date, prev: PortChlorophyll | None = None) -> PortChlorophyll:
    """NOAA-21/NOAA-20 8-day windows; the S-NPP 8-day composite if neither answers, or if NOAA
    lags and S-NPP's newest composite is strictly newer: an older summary never hides a newer one."""
    chl, errors = chlorophyll_jpss(ctx, lat, lon, end, prev)
    if chl is not None:
        newest = chl.latest_center_date
        if newest is None or newest < (end - timedelta(days=CHL_LAG_DAYS)).isoformat():
            snpp = chlorophyll_snpp(ctx, lat, lon, end)
            if not snpp.error and snpp.latest_center_date and snpp.latest_center_date > (newest or ""):
                return snpp
        return chl
    snpp = chlorophyll_snpp(ctx, lat, lon, end)
    if snpp.error:
        # nothing answered: say so for both (never a zero); the dataset is the primary one
        snpp.dataset_id = ",".join(ds for _, ds in CHL_PLATFORMS)
        snpp.source_url = f"{CHL_JPSS_SERVER}/griddap/{CHL_PLATFORMS[0][1]}.html"
        snpp.error = (
            "satellite chlorophyll unavailable from NOAA-21/NOAA-20 and from the S-NPP fallback: "
            + "; ".join(e[:80] for e in errors) + f"; S-NPP fallback: {snpp.error.removeprefix('satellite chlorophyll unavailable: ')[:80]}"
        )[:300]
    return snpp


def chlorophyll_snpp(ctx: RunContext, lat: float, lon: float, end: date) -> PortChlorophyll:
    from .charm import parse_netcdf

    base = PortChlorophyll(
        dataset_id=CHL_DATASET,
        source_url=f"{CHL_SERVER}/griddap/{CHL_DATASET}.html",
        radius_km=RADIUS_KM,
        composite_days=8,
        latest_center_date=None,
        latest=None,
        latest_valid_fraction=None,
    )
    start = end - timedelta(days=CHL_DAYS)
    dlat, dlon = RADIUS_KM / KM_PER_DEG + 0.01, RADIUS_KM / (KM_PER_DEG * math.cos(math.radians(lat))) + 0.01
    # latitude is descending in this dataset; stride 2 (~1.5 km) keeps requests small
    sel = (
        f"[({start.isoformat()}T00:00:00Z):(last)][(0.0)]"
        f"[({lat + dlat:.4f}):2:({lat - dlat:.4f})][({lon - dlon:.4f}):2:({lon + dlon:.4f})]"
    )
    url = f"{CHL_SERVER}/griddap/{CHL_DATASET}.nc?" + quote(f"chla{sel}", safe=",():")
    try:
        arrays, vattrs, _ = parse_netcdf(ctx.fetcher(url).body)
    except Exception as e:
        base.error = f"satellite chlorophyll unavailable: {e}"[:300]
        return base
    a = arrays["chla"].astype(float)
    if a.ndim == 4:
        a = a[:, 0]
    fv = vattrs["chla"].get("_FillValue")
    if isinstance(fv, (int, float)):
        a[np.isclose(a, float(fv))] = np.nan
    a[(a <= 0) | (a > 200)] = np.nan
    times = [datetime.fromtimestamp(float(t), tz=timezone.utc).date() for t in np.atleast_1d(arrays["time"])]
    LA, LO = np.meshgrid(arrays["latitude"].astype(float), arrays["longitude"].astype(float), indexing="ij")
    within = km(lat, lon, LA, LO) <= RADIUS_KM
    hist = []
    for i, t in enumerate(times):
        s = stat(a[i][within])
        hist.append(SeriesPoint(date=t.isoformat(), value=s.median, n=s.n))
    base.history = hist
    # latest composite with any valid pixel near the port
    for i in range(len(times) - 1, -1, -1):
        s = stat(a[i][within])
        if s.n > 0:
            base.latest_center_date = times[i].isoformat()
            base.latest = s
            base.latest_valid_fraction = round(s.n / int(within.sum()), 4) if within.sum() else None
            break
    return base


def satellite_offsets() -> str:
    from .satellite import VIIRS_OFFSETS

    return VIIRS_OFFSETS


def relations(port: dict, lat: float, lon: float, official: OfficialDataset | None) -> list[PortOfficialRelation]:
    if not official:
        return []
    out = []
    geoms = {rid: f for f in official.geometry.get("features", []) for rid in f["properties"]["record_ids"]}
    for r in official.registry.records:
        if r.status != "active":
            continue
        a = r.area
        if a.type == "statewide":
            out.append(PortOfficialRelation(record_id=r.id, relation="statewide", note="Applies along the whole California coast."))
        elif a.type == "county" and port["county"] in a.counties:
            out.append(PortOfficialRelation(record_id=r.id, relation="same_county", note=f"Port is in {port['county']} County."))
        elif a.type == "lat_band" and a.lat_south <= lat <= a.lat_north:  # type: ignore[operator]
            out.append(
                PortOfficialRelation(
                    record_id=r.id,
                    relation="port_latitude_within_stated_range",
                    note="The port lies between the notice's official latitudes.",
                )
            )
        elif a.type == "named_area" and r.id in geoms:
            g = geoms[r.id]["geometry"]
            pts = np.array([p for ring in _rings(g) for p in ring])
            d = float(km(lat, lon, pts[:, 1], pts[:, 0]).min())
            if d <= NAMED_AREA_KM:
                out.append(
                    PortOfficialRelation(
                        record_id=r.id,
                        relation="named_area_nearby",
                        note=f"{a.description} is about {d:.0f} km from the port.",
                    )
                )
    return out


def _rings(g: dict):
    if g["type"] == "Polygon":
        return g["coordinates"]
    return [ring for poly in g["coordinates"] for ring in poly]


@dataclass
class PortIntelResult:
    collection: PortIntelCollection | None
    errors: list[str]
    notes: list[str]


def run(
    ctx: RunContext,
    manifest_layers: list[LayerArtifact],
    ports: PortsCollection | None,
    official: OfficialDataset | None,
    fetch_history: bool = True,
    previous: PortIntelCollection | None = None,
) -> PortIntelResult:
    if not ports:
        return PortIntelResult(None, ["no ports available"], [])
    grids = load_grids(ctx, manifest_layers)
    nowcast = grids.get(("pseudo_nitzschia", 0))
    end = date.fromisoformat(nowcast.layer.time.valid_date) if nowcast and nowcast.layer.time.valid_date else ctx.now.date()
    notes: list[str] = []
    prev_hist = {q.port_code: q.charm.history for q in previous.ports if q.charm and q.charm.history} if previous else {}
    prev_chl = {q.port_code: q.chlorophyll for q in previous.ports if q.chlorophyll} if previous else {}
    out = []
    for f in ports.features:
        p = f.properties
        lon, lat = f.geometry["coordinates"]
        ch = charm_near(grids, lat, lon)
        if ch and fetch_history:
            try:
                ch.history = charm_history(ctx, lat, lon, end)
            except Exception as e:
                kept = carried_history(prev_hist.get(p.port_code), end)
                if kept:
                    ch.history = kept
                    notes.append(f"{p.display_name}: nowcast history not refreshed ({str(e)[:160]}); previous history kept with its own dates")
                else:
                    ch.history_error = f"nowcast history unavailable: {e}"[:300]
                    notes.append(f"{p.display_name}: {ch.history_error}")
        chl = chlorophyll_near(ctx, lat, lon, ctx.now.date(), prev_chl.get(p.port_code)) if fetch_history else None
        if chl and chl.error:
            notes.append(f"{p.display_name}: {chl.error}")
        out.append(
            PortIntel(
                port_code=p.port_code,
                display_name=p.display_name,
                county=p.county,
                region=p.region,
                lon=lon,
                lat=lat,
                charm=ch,
                chlorophyll=chl,
                official_relations=relations({"county": p.county}, lat, lon, official),
                caveats=CAVEATS,
            )
        )
    coll = PortIntelCollection(
        generated_at=ctx.now_iso,
        ports=out,
        method={
            "neighbourhood": f"Cells whose centres lie within {RADIUS_KM:.0f} km of the CDFW port point.",
            "charm_leads": "Statistics from the published C-HARM value grids of the current run (same values as the map).",
            "charm_history": f"Median over the same neighbourhood for each C-HARM nowcast in the last {HISTORY_DAYS} days available on ERDDAP; days without a run are absent.",
            "chlorophyll": (
                "Computed by Neritic from NOAA's daily VIIRS 750 m merges: NOAA-21, with NOAA-20 only in pixels NOAA-21 has no value for "
                f"(each pixel from one satellite, never averaged). For each date, every pixel's median over an {CHL_WINDOW_DAYS}-day window "
                f"({CHL_WINDOW_BEFORE} days before to {CHL_WINDOW_AFTER} after; successive windows overlap), then the median of those pixels within {RADIUS_KM:.0f} km. "
                "Only complete windows are shown. Cloud and land pixels are excluded; a missing value means no clear observation."
            ),
            "chlorophyll_source_change": (
                "Earlier port summaries used NOAA's S-NPP 8-day composites (erdVHNchla8day); NOAA ends S-NPP data on 2026-11-02. "
                "Over California NOAA-21 reads about 4% lower than S-NPP and NOAA-20 about 10% lower (98 days of overlap, Feb-Oct 2026); "
                "no correction is applied, so values from before and after the change may differ by about that much. "
                "A port whose dataset is erdVHNchla8day still uses the S-NPP composites (fallback while NOAA-21 and NOAA-20 are unavailable)."
            ),
            "official": "A record is related to a port if it is statewide, names the port's county, has official latitudes that include the port, or its official polygon lies within 60 km.",
        },
        provenance=[
            Provenance(
                source_id="charm",
                source_name="C-HARM v3.1 (NOAA CoastWatch West Coast ERDDAP)",
                source_url=f"{charm.SERVER}/griddap/{charm.dataset_id(0)}.html",
                dataset_id="wvcharmV3_0day..3day",
                product_version="3.1",
                license=charm.LICENSE,
                retrieved_at=ctx.now_iso,
                pipeline_version=ctx.pipeline_version,
                pipeline_run_id=ctx.run_id,
            ),
            Provenance(
                source_id=CHL_SOURCE_ID,
                source_name="NOAA VIIRS chlorophyll-a, 750 m daily merges from NOAA-21 (NOAA-20 where NOAA-21 has no value), 8-day windows computed by Neritic (NOAA CoastWatch ERDDAP)",
                source_url=f"{CHL_JPSS_SERVER}/griddap/{CHL_PLATFORMS[0][1]}.html",
                dataset_id=",".join(ds for _, ds in CHL_PLATFORMS),
                license=charm.LICENSE,
                retrieved_at=ctx.now_iso,
                upstream_metadata={
                    "change_point": (
                        "Port chlorophyll switched from erdVHNchla8day (S-NPP 8-day composites) to 8-day windows of the daily "
                        "NOAA-21/NOAA-20 pixels with this pipeline version; every port's dataset_id names the source of its own values and history."
                    ),
                    "offsets": satellite_offsets(),
                    "platform_rule": "NOAA-21 first; NOAA-20 only in pixels NOAA-21 has no value for. No averaging, no bias correction.",
                    "window": f"{CHL_WINDOW_DAYS} days, D-{CHL_WINDOW_BEFORE} to D+{CHL_WINDOW_AFTER}; per-pixel median, then the median within {RADIUS_KM:.0f} km",
                },
                pipeline_version=ctx.pipeline_version,
                pipeline_run_id=ctx.run_id,
            ),
        ]
        + (
            [
                Provenance(
                    source_id="viirs_chl_8day",
                    source_name="NOAA VIIRS S-NPP chlorophyll-a, 750 m, 8-day composite (NOAA CoastWatch West Coast ERDDAP); fallback while NOAA-21 and NOAA-20 are unavailable",
                    source_url=f"{CHL_SERVER}/griddap/{CHL_DATASET}.html",
                    dataset_id=CHL_DATASET,
                    license=charm.LICENSE,
                    retrieved_at=ctx.now_iso,
                    pipeline_version=ctx.pipeline_version,
                    pipeline_run_id=ctx.run_id,
                )
            ]
            if any(q.chlorophyll and q.chlorophyll.dataset_id == CHL_DATASET and not q.chlorophyll.error for q in out)
            else []
        ),
    )
    return PortIntelResult(coll, [], notes)
