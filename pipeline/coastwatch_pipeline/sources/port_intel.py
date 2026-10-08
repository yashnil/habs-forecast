"""Port-level summaries for the 22 CDFW landing ports.

For each port:
* C-HARM near the port: statistics over forecast cells within RADIUS_KM, per lead and
  variable, read from the *published* value grids (so they match the map and inspector).
* C-HARM nowcast history: the last HISTORY_DAYS of nowcasts from ERDDAP, median over the
  same neighbourhood; missing runs stay missing (no interpolation).
* Satellite chlorophyll: NOAA VIIRS (S-NPP) 750 m 8-day composites from ERDDAP
  (erdVHNchla8day), median of valid pixels within RADIUS_KM; cloud/land pixels excluded.
* Official records whose stated area may include waters near the port, with the reason.

A neighbourhood statistic describes the forecast grid near a port. It does not describe
conditions at the port or at any fishing ground.
"""

from __future__ import annotations

import math
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


def charm_history(ctx: RunContext, lat: float, lon: float, end: date) -> dict[str, list[SeriesPoint]]:
    from .charm import parse_netcdf

    start = end - timedelta(days=HISTORY_DAYS - 1)
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


def chlorophyll_near(ctx: RunContext, lat: float, lon: float, end: date) -> PortChlorophyll:
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
) -> PortIntelResult:
    if not ports:
        return PortIntelResult(None, ["no ports available"], [])
    grids = load_grids(ctx, manifest_layers)
    nowcast = grids.get(("pseudo_nitzschia", 0))
    end = date.fromisoformat(nowcast.layer.time.valid_date) if nowcast and nowcast.layer.time.valid_date else ctx.now.date()
    notes: list[str] = []
    out = []
    for f in ports.features:
        p = f.properties
        lon, lat = f.geometry["coordinates"]
        ch = charm_near(grids, lat, lon)
        if ch and fetch_history:
            try:
                ch.history = charm_history(ctx, lat, lon, end)
            except Exception as e:
                ch.history_error = f"nowcast history unavailable: {e}"[:300]
                notes.append(f"{p.display_name}: {ch.history_error}")
        chl = chlorophyll_near(ctx, lat, lon, ctx.now.date()) if fetch_history else None
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
            "chlorophyll": "Median of valid NOAA VIIRS S-NPP 750 m pixels (8-day composites, time = centre of the 8-day window, successive values overlap). Cloud and land pixels are excluded; a missing value means no clear observation.",
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
                source_id="viirs_chl_8day",
                source_name="NOAA VIIRS S-NPP chlorophyll-a, 750 m, 8-day composite (NOAA CoastWatch West Coast ERDDAP)",
                source_url=f"{CHL_SERVER}/griddap/{CHL_DATASET}.html",
                dataset_id=CHL_DATASET,
                license=charm.LICENSE,
                retrieved_at=ctx.now_iso,
                pipeline_version=ctx.pipeline_version,
                pipeline_run_id=ctx.run_id,
            ),
        ],
    )
    return PortIntelResult(coll, [], notes)
