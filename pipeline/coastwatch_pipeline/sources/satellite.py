"""High-resolution satellite chlorophyll-a: Sentinel-3 OLCI 300 m, with VIIRS 750 m fallback.

Verified 2026-10-09 (docs/coastwatch/forecast-upgrade/01-source-inventory.md):
- NOAA CoastWatch (central) ERDDAP serves Sentinel-3A and -3B OLCI chlorophyll-a,
  near real time, 0.0025 deg, as longitude sectors: CI (140-120 W) and DI (120-100 W).
  California needs both. The time axis holds one overpass per sector per day, ~2 days
  behind real time, in a rolling 90-day window. On 2026-10-09 the server also reloaded
  and the sector datasets were briefly "unknown"; that is handled as an outage.
- VIIRS 750 m (0.0075 deg) is the fallback. NOAA ends Suomi-NPP data on 2026-11-02
  (docs/coastwatch/33-suomi-npp-continuity.md), so it comes from the NOAA-21 and NOAA-20 daily
  merges on NOAA CoastWatch (central) ERDDAP (noaacwN21/N20VIIRSchlaSectorUSDaily: same lattice
  as erdVHNchla1day, one merge per day stamped at its overpass time, about 1.5 days after the
  overpass: measured 2026-10-14 from ERDDAP's /files/ listing; the "3-8 days" seen 2026-10-09..13
  was ERDDAP's time index stuck at 10-05 until the server restarted). NOAA-21 is primary; NOAA-20 fills only the cells NOAA-21 has no value for, so
  every cell comes from one platform, recorded per cell (platform grid). Nothing is averaged or
  bias-corrected; the measured offsets to S-NPP are stated in the provenance. The S-NPP 1-day
  composite (erdVHNchla1day, West Coast ERDDAP) is used, labelled S-NPP, when it holds a strictly
  newer day than NOAA-21/NOAA-20 (or they cannot be updated): an older observation never hides a
  newer one. Whole layers, one lineage each; once S-NPP data end it has nothing new and drops out.

Rules (docs/coastwatch/05-science-and-safety.md, forecast-upgrade/04):
- every value published is an upstream value (quantized); missing pixels stay missing
- every pixel keeps its observation date; composites carry a per-pixel age grid
- cloud, land and quality masks from upstream are preserved as no-value
- no smoothing, no gap-filling, no resampling beyond nearest-cell display tiles
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

from ..context import RunContext
from ..http import FetchError
from ..models import (
    AgeBin,
    CompositeDay,
    CompositeInfo,
    Coverage,
    FreshnessPolicy,
    GridChunks,
    LayerArtifact,
    Provenance,
    QCCheck,
    QualityControl,
    RegionCoverage,
    TileLayer,
    TimeInfo,
    ValueGrid,
)
from ..process import grid as gridcodec
from ..process.mercator import SourceGrid
from ..process.palette import AGE_COLOURS, CHLOROPHYLL, palette_indices
from ..process.tiles import render_tiles, write_chunks, write_index
from .charm import parse_netcdf

SOURCE_ID = "satellite_chl"
GROUP_ID = "satellite_chlorophyll_hr"
REPO = Path(__file__).resolve().parents[3]

WINDOW_DAYS = 7
# VIIRS 750 m from NOAA-21/NOAA-20 is published about 1.5 days after observation (ERDDAP /files/,
# 2026-10-14). A 10-day window was tried for a "3-8 day latency" that was in fact ERDDAP's index
# stuck for 9 days; it let 9-day-old NOAA data fill the layer silently. The same 7 days as OLCI;
# a newer S-NPP day wins over older NOAA days (run_viirs), and health alerts after 5 days.
VIIRS_WINDOW_DAYS = WINDOW_DAYS
# Bump whenever processing changes what a published day contains (scope, masks, encoding, source):
# published days are reused across runs only when made by the same processing version.
# 3: VIIRS from NOAA-21/NOAA-20 (per-cell platform grid) instead of S-NPP; no S-NPP day is
# ever reused as a NOAA-21/NOAA-20 one.
# 4: scope adds the marine bays and estuaries connected to the coastal ocean (MARINE_WATER_MASK).
PROCESSING_VERSION = "satellite-processing-4"
CHUNK = 512
LOG_RANGE = (-3.0, 3.0)  # quantization domain for log10(chl): 0.001 .. 1000 mg m-3
IMPLAUSIBLE_MG_M3 = 200.0  # flagged in QC, never silently removed
# Product scope: the coastal ocean plus the marine bays and estuaries connected to it.
# - The C-HARM ocean domain (3 km) extended this many cells (about 12 km) landward keeps the
#   offshore extent and the open-coast nearshore strip, and drops inland lakes, reservoirs and the
#   mixed land/water pixels the upstream L3 product carries inland.
# - Enclosed water farther from the model's ocean cells (95% of San Francisco Bay, with San Pablo
#   and Suisun Bays) is added from MARINE_WATER_MASK: OpenStreetMap ocean water (bays and
#   estuaries up to the OSM coastline; never lakes or reservoirs) connected to that scope, built
#   by pipeline/scripts/build_marine_water_mask.py. The C-HARM layers are not affected.
COASTAL_DILATE_CELLS = 4
MARINE_WATER_MASK = "marine_water_mask.json"
MAX_IMPLAUSIBLE_FRACTION = 0.01

LICENSE_OLCI = (
    "Contains modified Copernicus Sentinel-3 data (EUMETSAT), processed by NOAA CoastWatch/OceanWatch; "
    "free and open. NOAA data: may be used and redistributed for free."
)
LICENSE_VIIRS = "NOAA CoastWatch: may be used and redistributed for free but is not intended for legal use."

FRESHNESS = FreshnessPolicy(
    basis="observed_date",
    current_max_age_days=3,
    stale_max_age_days=7,
    note=(
        "Satellite overpasses are published about 1.5-2 days after observation (OLCI; VIIRS 750 m from NOAA-21/NOAA-20). "
        "Current: newest observation within 3 days. Stale: 4-7 days. Historical: older. "
        "In a 'latest clear view' every pixel also carries its own date."
    ),
)

CAVEATS_BASE = [
    "Chlorophyll-a measures algae biomass near the surface. It does not measure toxins, does not identify Pseudo-nitzschia and does not predict where fish are.",
    "Clouds and fog leave gaps. A gap means no observation, not low chlorophyll. Gaps are never filled.",
    "Values very close to shore, in river plumes and in shallow water are less reliable (bottom reflectance, sediment, land adjacency).",
]
CAVEAT_SCOPE = (
    "Shown for the coastal ocean and the bays and estuaries that open onto it (OpenStreetMap ocean water); "
    "inland lakes, reservoirs and inland land/water pixels are left out."
)
CAVEAT_TURBID = (
    "In bays and estuaries (San Francisco, San Pablo, Suisun, Humboldt, Tomales and Morro Bays, Elkhorn Slough) the water is "
    "often turbid. The near-real-time chlorophyll algorithms were made for open-ocean water; sediment and dissolved organic "
    "matter there can make them read too high or too low, so treat bay values as approximate and do not compare them "
    "directly with open-coast values."
)
CAVEAT_COMPOSITE = (
    "Latest clear view: each pixel shows its most recent valid observation within {n} days; neighbouring pixels can come "
    "from different days. Check the observation date in the readout."
)
CAVEATS = CAVEATS_BASE + [CAVEAT_SCOPE, CAVEAT_TURBID]


@dataclass(frozen=True)
class Domain:
    lat_s: float
    lat_n: float
    lon_w: float
    lon_e: float


CALIFORNIA = Domain(32.40, 42.10, -125.50, -117.00)


@dataclass(frozen=True)
class Product:
    key: str
    server: str
    # (platform, ((dataset, lon_min, lon_max, lat_max), ...)); lat_max trims requests to water
    platforms: tuple[tuple[str, tuple[tuple[str, float, float, float], ...]], ...]
    variable: str
    step: float
    lattice_lat: float  # any cell-centre latitude on the source lattice
    lattice_lon: float
    native_m: float
    title: str
    short_title: str
    license: str
    source_name: str
    sensor_text: str
    # A sensor name listed before the satellites in `platforms` ("VIIRS", then "NOAA-21"...):
    # the app names the sensor from it, and the released app knows only "VIIRS".
    sensor: str | None = None
    # Publish a per-cell platform grid with the latest clear view (referenced from provenance)
    platform_grid: bool = False
    # Offsets measured against other platforms, stated in provenance (never applied)
    offsets: str | None = None
    # Latest clear view: days back from the run date
    window_days: int = WINDOW_DAYS


SPLIT = -120.0
OLCI = Product(
    key="olci300",
    server="https://coastwatch.noaa.gov/erddap",
    platforms=(
        # East of 120 W, California's coast lies south of 35 N; north of that the DI sector is land.
        ("Sentinel-3A", (("noaacwS3AOLCIchlaSectorCIDaily", -180.0, SPLIT, 90.0), ("noaacwS3AOLCIchlaSectorDIDaily", SPLIT, 180.0, 35.0))),
        ("Sentinel-3B", (("noaacwS3BOLCIchlaSectorCIDaily", -180.0, SPLIT, 90.0), ("noaacwS3BOLCIchlaSectorDIDaily", SPLIT, 180.0, 35.0))),
    ),
    variable="chlor_a",
    step=0.0025,
    lattice_lat=45.18625,
    lattice_lon=-140.03625,
    native_m=300.0,
    title="Satellite chlorophyll-a, Sentinel-3 OLCI 300 m",
    short_title="Chlorophyll · OLCI 300 m",
    license=LICENSE_OLCI,
    source_name="Sentinel-3 OLCI chlorophyll-a, near real time (NOAA CoastWatch)",
    sensor_text="the OLCI sensor on Sentinel-3A, with Sentinel-3B filling gaps",
)
# Measured on 98 days with all three sensors, February-October 2026, California, same cells
# (research/viirs-jpss-overlap/report.md): median log10 ratio, pooled.
VIIRS_OFFSETS = (
    "Measured, not corrected (98 days with all three, Feb-Oct 2026, California, same cells; "
    "research/viirs-jpss-overlap/report.md): NOAA-21 reads about 4% lower than S-NPP (median log10 ratio -0.020), "
    "NOAA-20 about 10% lower (-0.045); NOAA-21 about 10% higher than NOAA-20 (+0.040). Agreement is poorer "
    "within 5 km of shore and above 1 mg m-3."
)
VIIRS_JPSS = Product(
    key="viirs750",  # the layer the app selects as "VIIRS"; provenance names the satellites
    server="https://coastwatch.noaa.gov/erddap",
    platforms=(
        ("NOAA-21", (("noaacwN21VIIRSchlaSectorUSDaily", -180.0, 180.0, 90.0),)),
        ("NOAA-20", (("noaacwN20VIIRSchlaSectorUSDaily", -180.0, 180.0, 90.0),)),
    ),
    variable="chlor_a",
    step=0.0075,
    lattice_lat=42.49875,  # the erdVHNchla1day lattice (checked 2026-10-13)
    lattice_lon=-127.00125,
    native_m=750.0,
    title="Satellite chlorophyll-a, VIIRS 750 m",
    short_title="Chlorophyll · VIIRS 750 m",
    license=LICENSE_VIIRS,
    source_name="VIIRS chlorophyll-a, 750 m daily merges from NOAA-21, with NOAA-20 where NOAA-21 has no value (NOAA CoastWatch / NESDIS STAR, near real time)",
    sensor_text="VIIRS on NOAA-21, with NOAA-20 in cells NOAA-21 has no value for",
    sensor="VIIRS",
    platform_grid=True,
    offsets=VIIRS_OFFSETS,
    window_days=VIIRS_WINDOW_DAYS,
)
# Suomi-NPP: fallback only, until its data end (2026-11-02, 13:00 UTC).
VIIRS_SNPP = Product(
    key="viirs750",
    server="https://coastwatch.pfeg.noaa.gov/erddap",
    platforms=(("S-NPP", (("erdVHNchla1day", -180.0, 180.0, 90.0),)),),
    variable="chla",
    step=0.0075,
    lattice_lat=0.0,  # resolved from the response
    lattice_lon=0.0,
    native_m=750.0,
    title="Satellite chlorophyll-a, VIIRS 750 m",
    short_title="Chlorophyll · VIIRS 750 m",
    license=LICENSE_VIIRS,
    source_name="VIIRS chlorophyll-a, 750 m daily composite from Suomi-NPP (NOAA CoastWatch West Coast); fallback while NOAA-21 and NOAA-20 are unavailable",
    sensor_text="VIIRS on Suomi-NPP (daily composite of available passes)",
    sensor="VIIRS",
    offsets=VIIRS_OFFSETS,
    window_days=VIIRS_WINDOW_DAYS,
)
VIIRS = VIIRS_JPSS
PRODUCTS = (OLCI, VIIRS_JPSS, VIIRS_SNPP)


def product_for(dataset_id: str | None) -> Product | None:
    """The product a published layer came from, by its own provenance dataset ids."""
    ids = set((dataset_id or "").split(","))
    for p in PRODUCTS:
        if ids & {sec[0] for _, sectors in p.platforms for sec in sectors}:
            return p
    return None


def platform_labels(p: Product, platforms: list[str]) -> list[str]:
    """`platforms` as published: the sensor (if the product names one) then the satellites."""
    return ([p.sensor] if p.sensor else []) + [pl for pl in platforms if pl != p.sensor]


# ---------------------------------------------------------------- target grid
@dataclass(frozen=True)
class Target:
    lat_first: float  # northernmost row centre
    lon_first: float  # westernmost column centre
    step: float
    height: int
    width: int

    @property
    def grid(self) -> SourceGrid:
        return SourceGrid(self.lat_first, -self.step, self.lon_first, self.step, self.height, self.width)

    def lats(self) -> np.ndarray:
        return self.lat_first - self.step * np.arange(self.height)

    def lons(self) -> np.ndarray:
        return self.lon_first + self.step * np.arange(self.width)


def target_for(p: Product, d: Domain, lattice: tuple[float, float] | None = None) -> Target:
    la0, lo0 = lattice or (p.lattice_lat, p.lattice_lon)
    s = p.step
    lat_first = la0 - math.ceil(round((la0 - d.lat_n) / s, 6)) * s
    lon_first = lo0 + math.ceil(round((d.lon_w - lo0) / s, 6)) * s
    height = int(math.floor(round((lat_first - d.lat_s) / s, 6))) + 1
    width = int(math.floor(round((d.lon_e - lon_first) / s, 6))) + 1
    return Target(round(lat_first, 6), round(lon_first, 6), s, height, width)


# ---------------------------------------------------------------- upstream access
def times_url(server: str, ds: str, start: date) -> str:
    q = f"time[({start.isoformat()}T00:00:00Z):1:(last)]"
    return f"{server}/griddap/{ds}.csv0?" + quote(q, safe=":(),")


def scene_url(server: str, ds: str, var: str, t: str, target: Target, lon_lo: float, lon_hi: float, lat_max: float) -> str:
    """Request the part of the target grid this sector serves, with bounds on exact cell
    centres (ERDDAP picks the cell nearest each bound, so a bound between two cells is
    ambiguous). The cell on a sector split belongs to exactly one sector."""
    lats, lons = target.lats(), target.lons()
    lons = lons[(lons > lon_lo) & (lons < lon_hi)] if lon_lo > -180 else lons[lons < lon_hi]
    lats = lats[lats <= lat_max]
    if not lats.size or not lons.size:
        raise ValueError("sector does not intersect the domain")
    q = f"{var}[({t})][(0.0)][({lats[0]:.5f}):({lats[-1]:.5f})][({lons[0]:.5f}):({lons[-1]:.5f})]"
    return f"{server}/griddap/{ds}.nc?" + quote(q, safe=",():")


def no_matching_results(e: Exception) -> bool:
    """ERDDAP's 404 'Your query produced no matching results' is an answer (nothing new);
    its 404 'Currently unknown datasetID' (dataset reloading or down) is an outage."""
    if "HTTP 404" not in str(e):
        return False
    detail = getattr(e, "detail", None)
    text = detail or str(e)
    if "unknown datasetid" in text.lower():
        return False
    return detail is None or "no matching results" in text.lower()


def masked_404(e: Exception) -> bool:
    """A 404 whose body is a web page, not ERDDAP's error: coastwatch.noaa.gov's front end replaces
    ERDDAP's 'no matching results' text with a site-wide HTML 404 (seen 2026-10-15 for every
    Sentinel-3B window, whose feed had stalled), so the body cannot say which 404 it is."""
    if "HTTP 404" not in str(e):
        return False
    detail = (getattr(e, "detail", None) or "").lower()
    return bool(detail) and ("<html" in detail or "<!doctype" in detail or "noaa coastwatch" in detail) and "no matching results" not in detail and "unknown datasetid" not in detail


def list_times(ctx: RunContext, server: str, ds: str, start: date) -> list[str]:
    try:
        r = ctx.fetcher(times_url(server, ds, start))
    except FetchError as e:
        # ERDDAP answers 404 "Your query produced no matching results" when nothing is newer
        # than start; a 404 for "Currently unknown datasetID" is an outage, not "no data"
        if no_matching_results(e):
            return []
        # a front-end HTML 404 hides which it was: the dataset's own metadata settles it
        # (reachable: nothing newer than start; not reachable: the outage it looks like)
        if masked_404(e):
            try:
                ctx.fetcher(f"{server}/griddap/{ds}.das")
            except FetchError:
                raise e
            return []
        raise
    out = []
    for line in r.body.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if line:
            out.append(line if line.endswith("Z") else line + "Z")
    return out


@dataclass
class Scene:
    platform: str
    dataset: str
    time: str
    url: str
    values: np.ndarray  # on the target grid; NaN outside the scene's sector
    n_valid: int
    checks: list[QCCheck]
    content_hash: str


def load_scene(blob: bytes, p: Product, platform: str, ds: str, t: str, url: str, target: Target) -> Scene:
    checks: list[QCCheck] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append(QCCheck(name=f"{ds}:{name}", passed=bool(ok), detail=detail))

    arrays, vattrs, _ = parse_netcdf(blob)
    missing = [n for n in ("time", "latitude", "longitude", p.variable) if n not in arrays]
    check("required_variables", not missing, f"missing {missing}" if missing else "all present")
    if missing:
        raise ValueError(f"{ds} {t}: missing {missing}")
    tt = np.atleast_1d(arrays["time"]).astype(np.float64)
    ft = datetime.fromtimestamp(float(tt[0]), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if tt.size else None
    check("time_matches_listing", ft == t, f"file {ft}, listed {t}")
    lat = arrays["latitude"].astype(np.float64)
    lon = arrays["longitude"].astype(np.float64)
    lon = np.where(lon > 180, lon - 360, lon)
    a = np.asarray(arrays[p.variable], dtype=np.float64).reshape(lat.size, lon.size)
    for key in ("_FillValue", "missing_value"):
        fv = vattrs[p.variable].get(key)
        if isinstance(fv, (int, float)) and math.isfinite(float(fv)):
            a[np.isclose(a, float(fv))] = np.nan
    a[a <= 0] = np.nan  # non-positive chlorophyll is not a measurement
    # Coordinates must sit on the target lattice: that is what makes alignment exact.
    rows = np.round((target.lat_first - lat) / target.step).astype(np.int64)
    cols = np.round((lon - target.lon_first) / target.step).astype(np.int64)
    off_lat = np.abs(target.lat_first - rows * target.step - lat).max() if lat.size else 0.0
    off_lon = np.abs(target.lon_first + cols * target.step - lon).max() if lon.size else 0.0
    on_lattice = off_lat < 0.01 * target.step and off_lon < 0.01 * target.step
    check("coordinates_on_lattice", on_lattice, f"max offset lat {off_lat:.2e}, lon {off_lon:.2e} deg")
    # ERDDAP may return an edge cell just beyond a bound; cells outside the domain are dropped.
    keep_r = (rows >= 0) & (rows < target.height)
    keep_c = (cols >= 0) & (cols < target.width)
    check("within_domain", bool(keep_r.any() and keep_c.any()), f"{int((~keep_r).sum())} edge rows and {int((~keep_c).sum())} edge columns outside the domain dropped")
    if not (on_lattice and keep_r.any() and keep_c.any() and not missing and ft == t):
        raise ValueError(f"{ds} {t}: failed checks {[c.name for c in checks if not c.passed]}")
    a, rows, cols = a[np.ix_(keep_r, keep_c)], rows[keep_r], cols[keep_c]
    finite = a[np.isfinite(a)]
    implaus = float((finite > IMPLAUSIBLE_MG_M3).mean()) if finite.size else 0.0
    check("plausible_values", implaus <= MAX_IMPLAUSIBLE_FRACTION, f"{implaus:.4%} of values > {IMPLAUSIBLE_MG_M3} mg m-3")
    out = np.full((target.height, target.width), np.nan)
    out[np.ix_(rows, cols)] = a
    return Scene(platform, ds, t, url, out, int(finite.size), checks, hashlib.sha256(blob).hexdigest()[:10])


# ---------------------------------------------------------------- day mosaics
@dataclass
class Day:
    date: date
    values: np.ndarray
    scenes: list[Scene] = field(default_factory=list)
    observed_times: list[str] = field(default_factory=list)
    platforms: list[str] = field(default_factory=list)
    checks: list[QCCheck] = field(default_factory=list)
    reused: LayerArtifact | None = None
    # per cell: 1 + index in the product's platform order, 0 without a value (None if unknown)
    source: np.ndarray | None = None

    @property
    def key(self) -> str:
        return hashlib.sha256(("|".join(sorted(self.observed_times)) + PROCESSING_VERSION).encode()).hexdigest()[:10]


def mosaic(target: Target, scenes: list[Scene], platform_order: list[str], scope: np.ndarray | None = None) -> tuple[np.ndarray, int, np.ndarray]:
    """Primary platform first; a later platform only fills cells the earlier one left empty,
    so every cell is one platform's own value. Cells outside the coastal scope are dropped
    (returned count), never altered. Also returns, per cell, 1 + the index in platform_order
    of the platform the value came from (0: no value)."""
    out = np.full((target.height, target.width), np.nan)
    src = np.zeros((target.height, target.width), dtype=np.uint8)
    for k, plat in enumerate(platform_order):
        for sc in scenes:
            if sc.platform == plat:
                fill = np.isnan(out) & np.isfinite(sc.values)
                out[fill] = sc.values[fill]
                src[fill] = k + 1
    dropped = 0
    if scope is not None:
        outside = np.isfinite(out) & ~scope
        dropped = int(outside.sum())
        out[outside] = np.nan
        src[outside] = 0
    return out, dropped, src


# ---------------------------------------------------------------- coverage
def _load_reference():
    mask = json.loads((REPO / "data" / "reference" / "charm_ocean_mask.json").read_text())
    ocean = np.zeros((mask["height"], mask["width"]), dtype=bool)
    for r, runs in enumerate(mask["rows_rle"]):
        for s, n in runs:
            ocean[r, s : s + n] = True
    regions = json.loads((REPO / "data" / "curated" / "ports.json").read_text()).get("regions", [])
    return mask, ocean, regions


def _on_target(mask_meta: dict, cells: np.ndarray, target: Target) -> np.ndarray:
    """Sample a reference mask (nearest cell) at each target cell centre; False outside it."""
    ref = SourceGrid(mask_meta["lat_first"], mask_meta["lat_step"], mask_meta["lon_first"], mask_meta["lon_step"], mask_meta["height"], mask_meta["width"])
    lats, lons = target.lats(), target.lons()
    r, _ = ref.cell_index(lats, np.full_like(lats, mask_meta["lon_first"]))
    _, c = ref.cell_index(np.full_like(lons, mask_meta["lat_first"]), lons)
    ok = (r[:, None] >= 0) & (c[None, :] >= 0)
    out = np.zeros((target.height, target.width), dtype=bool)
    out[ok] = cells[np.broadcast_to(r[:, None], ok.shape)[ok], np.broadcast_to(c[None, :], ok.shape)[ok]]
    return out


def charm_scope(target: Target) -> np.ndarray:
    """True where a target cell lies within COASTAL_DILATE_CELLS of a C-HARM ocean cell."""
    from scipy.ndimage import binary_dilation

    mask, ocean, _ = _load_reference()
    return _on_target(mask, binary_dilation(ocean, iterations=COASTAL_DILATE_CELLS), target)


@functools.lru_cache(maxsize=1)
def _load_marine() -> tuple[dict, np.ndarray]:
    m = json.loads((REPO / "data" / "reference" / MARINE_WATER_MASK).read_text())
    cells = np.zeros((m["height"], m["width"]), dtype=bool)
    for r, runs in enumerate(m["rows_rle"]):
        for s, n in runs:
            cells[r, s : s + n] = True
    return m, cells


def marine_water(target: Target) -> np.ndarray:
    """True where a target cell is a marine bay or estuary connected to the coastal ocean."""
    m, cells = _load_marine()
    return _on_target(m, cells, target)


def coastal_mask(target: Target) -> np.ndarray:
    """True where a target cell lies in the satellite scope: the dilated C-HARM ocean
    (COASTAL_DILATE_CELLS) or connected marine water (MARINE_WATER_MASK)."""
    return charm_scope(target) | marine_water(target)


def coverage(values: np.ndarray, target: Target) -> Coverage:
    """Share of C-HARM ocean cells (3 km) that contain at least one valid pixel."""
    rr, cc = np.nonzero(np.isfinite(values))
    lat = target.lat_first - rr * target.step
    lon = target.lon_first + cc * target.step
    px_km2 = (target.step * 111.32) ** 2 * np.cos(np.radians(lat))
    south, north = target.lat_first - target.step * (target.height - 1), target.lat_first
    west, east = target.lon_first, target.lon_first + target.step * (target.width - 1)
    return coverage_points(lat, lon, px_km2, (south, north, west, east))


def coverage_points(lat: np.ndarray, lon: np.ndarray, px_km2: np.ndarray, bounds: tuple[float, float, float, float]) -> Coverage:
    """Coverage from the centres (and areas) of valid cells of any regular grid;
    `bounds` = (south, north, west, east) of the grid's cell centres."""
    mask, ocean, regions = _load_reference()
    ref = SourceGrid(mask["lat_first"], mask["lat_step"], mask["lon_first"], mask["lon_step"], mask["height"], mask["width"])
    observed = np.zeros_like(ocean)
    if lat.size:
        r3, c3 = ref.cell_index(lat, lon)
        ok = (r3 >= 0) & (c3 >= 0)
        observed[r3[ok], c3[ok]] = True
    observed &= ocean
    # restrict the reference to the target domain
    lats = mask["lat_first"] + mask["lat_step"] * np.arange(mask["height"])
    lons = mask["lon_first"] + mask["lon_step"] * np.arange(mask["width"])
    south, north, west, east = bounds
    in_dom = ((lats >= south) & (lats <= north))[:, None] & ((lons >= west) & (lons <= east))[None, :]

    def frac(sel: np.ndarray) -> tuple[float, int]:
        n = int((ocean & sel).sum())
        return (float((observed & sel).sum() / n) if n else 0.0), n

    out_regions = []
    for reg in regions:
        (w, s), (e, n) = reg["bounds"]
        sel = in_dom & ((lats >= s) & (lats <= n))[:, None] & ((lons >= w) & (lons <= e))[None, :]
        f, n_ref = frac(sel)
        if n_ref == 0:
            continue
        in_reg = (lat >= s) & (lat <= n) & (lon >= w) & (lon <= e)
        out_regions.append(
            RegionCoverage(
                region_id=reg["id"], label=reg["label"], observed_fraction=round(f, 4),
                observed_km2=round(float(px_km2[in_reg].sum()), 1), reference_cells=n_ref,
            )
        )
    f_dom, n_dom = frac(in_dom)
    return Coverage(
        reference="C-HARM v3.1 ocean cells (0.03°, about 3 km) inside the layer's domain; a cell counts as observed if any pixel inside it has a value. Nearshore pixels outside the C-HARM mask are not counted.",
        domain_observed_fraction=round(f_dom, 4), domain_reference_cells=n_dom,
        regions=out_regions,
    )


# ---------------------------------------------------------------- publishing
def encode_log(block: np.ndarray) -> bytes:
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(np.isfinite(block) & (block > 0), np.log10(np.where(block > 0, block, 1.0)), np.nan)
    return gridcodec.encode(q, *LOG_RANGE)[0]


def encode_age(block: np.ndarray) -> bytes:
    return gridcodec.encode(block, 0.0, float(gridcodec.MAX_CODE))[0]


def _grid_meta(target: Target, base: str, present: list[str], log: bool) -> ValueGrid:
    scale = (LOG_RANGE[1] - LOG_RANGE[0]) / gridcodec.MAX_CODE if log else 1.0
    tmpl = f"{base}/{{row}}_{{col}}.u16.gz"
    return ValueGrid(
        url=tmpl, width=target.width, height=target.height,
        lat_first=target.lat_first, lat_step=-target.step, lon_first=target.lon_first, lon_step=target.step,
        scale_factor=scale, add_offset=LOG_RANGE[0] if log else 0.0, max_quantization_error=scale / 2,
        transform="log10" if log else "none",
        chunks=GridChunks(rows=CHUNK, cols=CHUNK, url_template=tmpl, present=present),
    )


# Display tiles (M5 rendering audit): 512 px images in 256 px slots (@2x) and one zoom past
# the cell size, so a 300 m Sentinel-3 cell (or 750 m VIIRS cell) spans an even number of
# screen pixels when zoomed in. Values, georeferencing and no-data masks are unchanged.
TILE_SCALE = 2
OLCI_ZOOMS = range(5, 12)
VIIRS_ZOOMS = range(5, 11)


def _bounds(target: Target) -> tuple[float, float, float, float]:
    h = target.step / 2
    return (target.lon_first - h, target.lat_first - target.step * (target.height - 1) - h, target.lon_first + target.step * (target.width - 1) + h, target.lat_first + h)


def publish_values(ctx: RunContext, values: np.ndarray, target: Target, base: str, zooms: range) -> tuple[ValueGrid, TileLayer, int]:
    present, nbytes = write_chunks(encode_log, values, CHUNK, CHUNK, ctx.out_dir / base / "grid")
    grid = _grid_meta(target, f"{base}/grid", present, log=True)
    # tiles are rendered from the published (quantized) values, so a tile colour always
    # matches the value a reader gets back from the grid
    q = np.full(values.shape, np.nan)
    ok = np.isfinite(values) & (values > 0)
    codes = np.clip(np.round((np.log10(values[ok]) - LOG_RANGE[0]) / grid.scale_factor), 0, gridcodec.MAX_CODE)
    q[ok] = 10 ** (codes * grid.scale_factor + LOG_RANGE[0])
    ts = render_tiles(q, target.grid, _bounds(target), zooms, lambda v: palette_indices(v, CHLOROPHYLL)[0], palette_indices(np.array([1.0]), CHLOROPHYLL)[1], ctx.out_dir / base / "tiles", TILE_SCALE)
    write_index(ts, ctx.out_dir / base / "tiles")
    w, s, e, n = _bounds(target)
    tiles = TileLayer(
        url_template=f"{base}/tiles/{{z}}/{{x}}/{{y}}.png", relative=True, max_native_zoom=zooms.stop - 1, min_zoom=zooms.start,
        tile_size=256 * TILE_SCALE,
        legend_url=None, legend_verified=False, bounds_lnglat=[round(w, 5), round(s, 5), round(e, 5), round(n, 5)],
        n_tiles=ts.n_tiles, sample_tiles=ts.tiles[:: max(1, len(ts.tiles) // 5)][:5],
        date_selection="Rendered by Neritic from the published value grid: one source cell per tile pixel (nearest), cells without a value transparent.",
    )
    return grid, tiles, nbytes + ts.bytes


def _provenance(ctx: RunContext, p: Product, urls: list[str], meta: dict[str, str], platforms: list[str] | None = None) -> Provenance:
    """dataset_id lists the product's datasets; for a product with a named sensor (VIIRS) only
    those of the platforms actually used, so a layer made from NOAA-20 alone says so."""
    used = [(pl, sectors) for pl, sectors in p.platforms if not (p.sensor and platforms is not None) or pl in platforms] or list(p.platforms)
    first_ds = used[0][1][0][0]
    if p.offsets:
        meta = {**meta, "offsets": p.offsets}
    return Provenance(
        source_id=SOURCE_ID, source_name=p.source_name, source_url=f"{p.server}/griddap/{first_ds}.html",
        dataset_id=",".join(sec[0] for _, sectors in used for sec in sectors), institution="NOAA CoastWatch",
        license=p.license, retrieved_at=ctx.now_iso, request_urls=urls[:12], upstream_metadata=meta,
        pipeline_version=ctx.pipeline_version, pipeline_run_id=ctx.run_id,
    )


def _qc(values: np.ndarray, checks: list[QCCheck]) -> QualityControl:
    f = values[np.isfinite(values)]
    return QualityControl(
        checks=checks, n_cells=int(values.size), n_valid=int(f.size),
        valid_fraction=round(f.size / values.size, 6) if values.size else 0.0,
        value_min=float(f.min()) if f.size else None, value_max=float(f.max()) if f.size else None,
    )


def day_layer(ctx: RunContext, p: Product, target: Target, day: Day, zooms: range) -> tuple[LayerArtifact, int]:
    base = f"satellite/{p.key}/day-{day.date.isoformat()}-{day.key}"
    grid, tiles, nbytes = publish_values(ctx, day.values, target, base, zooms)
    has = bool(np.isfinite(day.values).any())
    layer = LayerArtifact(
        layer_id=f"{p.key}_chl_{day.date.isoformat()}", group_id=GROUP_ID, product_class="observation",
        title=f"{p.title}, single day", short_title=p.short_title, variable="chlorophyll_a", units="mg m-3",
        description=f"Chlorophyll-a from {p.sensor_text}, as published for {day.date.isoformat()}. Cells without a clear observation have no value.",
        resolution_deg=p.step, native_resolution_m=p.native_m, platforms=platform_labels(p, day.platforms),
        time=TimeInfo(observed_date=day.date.isoformat(), valid_date=day.date.isoformat(), observed_times=sorted(day.observed_times)),
        freshness=FRESHNESS, grid=grid if has else None, tiles=tiles if has else None, palette=CHLOROPHYLL,
        caveats=list(CAVEATS), qc=_qc(day.values, day.checks), coverage=coverage(day.values, target),
        provenance=_provenance(ctx, p, [s.url for s in day.scenes], {"scenes": ", ".join(f"{s.platform} {s.dataset} {s.time}" for s in day.scenes), "processing": PROCESSING_VERSION}, day.platforms),
    )
    return layer, nbytes


def composite_layer(ctx: RunContext, p: Product, target: Target, days: list[Day], zooms: range) -> tuple[LayerArtifact | None, int]:
    ref = ctx.now.date()
    comp = np.full((target.height, target.width), np.nan)
    age = np.full((target.height, target.width), np.nan)
    track = p.platform_grid and all(d.source is not None for d in days)
    src = np.zeros((target.height, target.width), dtype=np.uint8)
    used: list[CompositeDay] = []
    contributing: list[Day] = []
    for d in sorted(days, key=lambda x: x.date, reverse=True):
        fill = np.isnan(comp) & np.isfinite(d.values)
        comp[fill] = d.values[fill]
        age[fill] = (ref - d.date).days
        if track:
            src[fill] = d.source[fill]  # type: ignore[index]
        used.append(CompositeDay(date=d.date.isoformat(), platforms=platform_labels(p, d.platforms), observed_times=sorted(d.observed_times), pixels_used=int(fill.sum())))
        if fill.any():
            contributing.append(d)
    if not np.isfinite(comp).any():
        return None, 0
    # The stated dates, overpass times and platforms are those of the days whose pixels the view
    # shows: a day entirely covered by newer days is listed (pixels_used 0) but dates nothing.
    observed_days = contributing
    newest, oldest = max(d.date for d in observed_days), min(d.date for d in observed_days)
    h = hashlib.sha256((p.key + "|".join(f"{d.date}:{d.key}" for d in days) + ref.isoformat()).encode()).hexdigest()[:10]
    base = f"satellite/{p.key}/latest-{ref.isoformat()}-{h}"
    grid, tiles, nbytes = publish_values(ctx, comp, target, base, zooms)
    present, abytes = write_chunks(encode_age, age, CHUNK, CHUNK, ctx.out_dir / base / "age")
    age_grid = _grid_meta(target, f"{base}/age", present, log=False)
    colours = [(0, 0, 0)] + [tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in AGE_COLOURS]
    ats = render_tiles(age, target.grid, _bounds(target), range(zooms.start, zooms.stop), lambda v: np.where(np.isfinite(v), np.clip(np.nan_to_num(v), 0, len(AGE_COLOURS) - 1) + 1, 0).astype(np.uint8), colours, ctx.out_dir / base / "age-tiles", TILE_SCALE)
    write_index(ats, ctx.out_dir / base / "age-tiles")
    age_tiles = tiles.model_copy(update={"url_template": f"{base}/age-tiles/{{z}}/{{x}}/{{y}}.png", "n_tiles": ats.n_tiles, "sample_tiles": ats.tiles[:: max(1, len(ats.tiles) // 5)][:5], "date_selection": "Age in days of each pixel of the latest clear view (categorical)."})
    a = age[np.isfinite(age)]
    hist = [AgeBin(age_days=int(k), fraction=round(float((a == k).mean()), 4)) for k in range(p.window_days + 1) if (a == k).any()]
    order = [pl for pl, _ in p.platforms]
    platforms = [pl for pl in order if any(pl in d.platforms for d in observed_days)]
    meta = {"days": ", ".join(sorted(d.date.isoformat() for d in days)), "processing": PROCESSING_VERSION}
    if p.sensor:
        # which dataset and overpass each day used (the verifier queries exactly these)
        meta["scenes"] = ", ".join(f"{s.platform} {s.dataset} {s.time}" for d in sorted(days, key=lambda x: x.date) for s in d.scenes)
    if track:
        pbytes = write_chunks(encode_age, np.where(src > 0, src.astype(float), np.nan), CHUNK, CHUNK, ctx.out_dir / base / "platform")[1]
        abytes += pbytes
        meta.update(platform_meta(base, order, src))
    layer = LayerArtifact(
        layer_id=f"{p.key}_chl_latest", group_id=GROUP_ID, product_class="observation",
        title=f"{p.title}, latest clear view", short_title=p.short_title, variable="chlorophyll_a", units="mg m-3",
        description=f"Most recent valid chlorophyll-a observation from {p.sensor_text} at each pixel within {p.window_days} days, with the date of each pixel.",
        resolution_deg=p.step, native_resolution_m=p.native_m, platforms=platform_labels(p, platforms),
        time=TimeInfo(observed_date=newest.isoformat(), valid_date=newest.isoformat(), observed_times=sorted(t for d in observed_days for t in d.observed_times)),
        freshness=FRESHNESS, grid=grid, tiles=tiles, palette=CHLOROPHYLL,
        caveats=[CAVEAT_COMPOSITE.format(n=p.window_days)] + list(CAVEATS), qc=_qc(comp, [c for d in days for c in d.checks]),
        coverage=coverage(comp, target),
        composite=CompositeInfo(
            window_days=p.window_days, reference_date=ref.isoformat(), newest_observed_date=newest.isoformat(),
            oldest_observed_date=oldest.isoformat(), days=used, age_histogram=hist, age_grid=age_grid, age_tiles=age_tiles,
        ),
        provenance=_provenance(ctx, p, [s.url for d in days for s in d.scenes], meta, platforms),
    )
    return layer, nbytes + abytes + ats.bytes


# ---------------------------------------------------------------- per-cell platform
# The latest clear view of a multi-platform product (VIIRS: NOAA-21, NOAA-20) records which
# platform each cell's value came from, as a chunked grid next to the values (same chunking,
# codec and cell lattice as the age grid). It is referenced from provenance.upstream_metadata
# (a free-form dict), so the published schema does not change.
PLATFORM_GRID_KEY = "platform_grid"


def platform_meta(base: str, order: list[str], src: np.ndarray) -> dict[str, str]:
    counts = {pl: int((src == k + 1).sum()) for k, pl in enumerate(order)}
    return {
        PLATFORM_GRID_KEY: f"{base}/platform/{{row}}_{{col}}.u16.gz",
        "platform_codes": ", ".join(f"{k + 1}={pl}" for k, pl in enumerate(order)) + "; a cell without a value has no code. Same chunks (512 x 512 cells) and lattice as the value grid; one platform per cell, never averaged.",
        "platform_rule": f"{order[0]} first; " + "; ".join(f"{pl} only in cells the platforms before it have no value for" for pl in order[1:]) + ". No averaging, no bias correction.",
        "pixels_by_platform": ", ".join(f"{pl} {n}" for pl, n in counts.items()),
    }


def load_platform_grid(out_dir: Path, lyr: LayerArtifact) -> tuple[np.ndarray, list[str]] | None:
    """(codes per cell, platform order) of a published latest clear view, if it has one."""
    md = lyr.provenance.upstream_metadata
    tmpl, codes = md.get(PLATFORM_GRID_KEY), md.get("platform_codes", "")
    if not tmpl or not lyr.grid or not lyr.grid.chunks:
        return None
    order = [c.split("=", 1)[1] for c in codes.split(";")[0].split(", ") if "=" in c]
    g = lyr.grid.model_copy(update={"url": tmpl, "scale_factor": 1.0, "add_offset": 0.0, "transform": "none", "chunks": lyr.grid.chunks.model_copy(update={"url_template": tmpl})})
    v = load_published(out_dir, g)
    return np.where(np.isfinite(v), v, 0).astype(np.uint8), order


# ---------------------------------------------------------------- reuse of published days
def load_published(out_dir: Path, grid: ValueGrid) -> np.ndarray:
    out = np.full((grid.height, grid.width), np.nan)
    assert grid.chunks is not None
    for key in grid.chunks.present:
        r, c = (int(x) for x in key.split("_"))
        h = min(grid.chunks.rows, grid.height - r * grid.chunks.rows)
        w = min(grid.chunks.cols, grid.width - c * grid.chunks.cols)
        blob = (out_dir / grid.chunks.url_template.format(row=r, col=c)).read_bytes()
        q = gridcodec.decode(blob, w, h, grid.scale_factor, grid.add_offset)
        out[r * grid.chunks.rows : r * grid.chunks.rows + h, c * grid.chunks.cols : c * grid.chunks.cols + w] = 10**q if grid.transform == "log10" else q
    return out


def _published_intact(out_dir: Path, lyr: LayerArtifact) -> bool:
    if lyr.grid is None or lyr.grid.chunks is None:
        return lyr.grid is None
    ch = lyr.grid.chunks
    return all((out_dir / ch.url_template.format(row=k.split("_")[0], col=k.split("_")[1])).exists() for k in ch.present)


# ---------------------------------------------------------------- run
@dataclass
class SatelliteResult:
    layers: list[LayerArtifact]
    errors: list[str]
    notes: list[str]
    latest_date: str | None
    bytes_written: int = 0


def run_product(ctx: RunContext, p: Product, domain: Domain, prev_layers: list[LayerArtifact], zooms: range, per_day: bool) -> SatelliteResult:
    errors: list[str] = []
    notes: list[str] = []
    start = ctx.now.date() - timedelta(days=p.window_days)
    listings: dict[tuple[str, str], list[str]] = {}
    for plat, sectors in p.platforms:
        for ds, _, _, _ in sectors:
            try:
                # ERDDAP snaps the start bound to the nearest time, which can be the evening
                # before the window; keep only times inside it
                listings[(plat, ds)] = [t for t in list_times(ctx, p.server, ds, start) if start.isoformat() <= t[:10] <= ctx.now.date().isoformat()]
            except Exception as e:
                errors.append(f"{plat} {ds}: time listing failed: {e}")
    if not listings:
        return SatelliteResult([], errors, notes, None)
    # OLCI's lattice is fixed and known; VIIRS's is read from the first scene.
    target = target_for(p, domain) if p.lattice_lat or p.lattice_lon else None
    scope: np.ndarray | None = None

    by_day: dict[date, list[tuple[str, str, str]]] = {}
    for (plat, ds), ts in listings.items():
        for t in ts:
            by_day.setdefault(date.fromisoformat(t[:10]), []).append((plat, ds, t))
    prev_days = {lyr.time.observed_date: lyr for lyr in prev_layers if lyr.layer_id.startswith(f"{p.key}_chl_2")}
    days: list[Day] = []
    order = [plat for plat, _ in p.platforms]
    for d in sorted(by_day):
        entries = by_day[d]
        times = sorted(t for _, _, t in entries)
        prev = prev_days.get(d.isoformat())
        same_processing = prev is not None and prev.provenance.upstream_metadata.get("processing") == PROCESSING_VERSION
        if prev and target is not None and same_processing and sorted(prev.time.observed_times) == times and _published_intact(ctx.out_dir, prev):
            vals = load_published(ctx.out_dir, prev.grid) if prev.grid else np.full((target.height, target.width), np.nan)
            if vals.shape == (target.height, target.width):
                days.append(Day(d, vals, observed_times=times, platforms=[pl for pl in prev.platforms if pl != p.sensor], checks=list(prev.qc.checks) if prev.qc else [], reused=prev))
                notes.append(f"{p.key} {d}: same overpasses as the published day; reused it.")
                continue
        scenes: list[Scene] = []
        for plat, ds, t in entries:
            sec = next(s for pl, ss in p.platforms if pl == plat for s in ss if s[0] == ds)
            if target is None:
                # VIIRS: lattice from a one-cell probe of the first scene
                probe = f"{p.server}/griddap/{ds}.nc?" + quote(f"{p.variable}[({t})][(0.0)][({domain.lat_n}):({domain.lat_n})][({domain.lon_w}):({domain.lon_w})]", safe=",():")
                arr, _, _ = parse_netcdf(ctx.fetcher(probe).body)
                lattice = (float(np.atleast_1d(arr["latitude"])[0]), float(np.atleast_1d(arr["longitude"])[0]))
                target = target_for(p, domain, lattice)
            try:
                url = scene_url(p.server, ds, p.variable, t, target, sec[1], sec[2], sec[3])
                r = ctx.fetcher(url)
                scenes.append(load_scene(r.body, p, plat, ds, t, url, target))
            except Exception as e:
                errors.append(f"{plat} {ds} {t}: {e}")
        if not scenes or target is None:
            continue
        if scope is None:
            scope = coastal_mask(target)
        vals, dropped, src = mosaic(target, scenes, order, scope)
        checks = [c for s in scenes for c in s.checks] + [
            QCCheck(name="coastal_scope", passed=True, detail=f"{dropped} valid pixels outside the coastal-ocean scope (inland lakes, reservoirs, land/water mix) left out")
        ]
        day = Day(d, vals, scenes=scenes, observed_times=sorted(s.time for s in scenes),
                  platforms=[pl for pl in order if any(s.platform == pl for s in scenes)], checks=checks, source=src)
        days.append(day)
    if not days or target is None:
        return SatelliteResult([], errors or [f"{p.key}: no scenes in the last {p.window_days} days"], notes, None)

    layers: list[LayerArtifact] = []
    nbytes = 0
    comp, b = composite_layer(ctx, p, target, days, zooms)
    nbytes += b
    if comp:
        layers.append(comp)
    if per_day:
        for d in days:
            if d.reused is not None:
                layers.append(d.reused)
                continue
            lyr, b = day_layer(ctx, p, target, d, zooms)
            layers.append(lyr)
            nbytes += b
    latest = max((d.date for d in days if np.isfinite(d.values).any()), default=None)
    notes.append(f"{p.key}: {len(days)} day(s) in window; newest observed {latest}.")
    return SatelliteResult(layers, errors, notes, latest.isoformat() if latest else None, nbytes)


def newest_listed(ctx: RunContext, p: Product) -> tuple[str | None, list[str]]:
    """Newest day a product's datasets list in its window (one small time listing each), or None,
    and the listings that failed (coastwatch.pfeg.noaa.gov refuses runners with HTTP 403 at
    times: a failed listing must be reported, not read as "nothing newer")."""
    start = ctx.now.date() - timedelta(days=p.window_days)
    newest = None
    failed: list[str] = []
    for _, sectors in p.platforms:
        for ds, _, _, _ in sectors:
            try:
                ts = [t for t in list_times(ctx, p.server, ds, start) if start.isoformat() <= t[:10] <= ctx.now.date().isoformat()]
            except Exception as e:
                failed.append(f"{ds}: time listing failed: {e}")
                continue
            if ts:
                newest = max(newest or "", ts[-1][:10])
    return newest, failed


def run_viirs(ctx: RunContext, domain: Domain, prev_layers: list[LayerArtifact]) -> SatelliteResult:
    """The newest VIIRS lineage, as whole layers, never S-NPP pixels mixed into NOAA-21/20 ones:
    NOAA-21/NOAA-20 unless the S-NPP composite holds a strictly newer day (then S-NPP, labelled
    S-NPP), or NOAA-21/NOAA-20 cannot be updated. An older observation inside the window must
    not hide a newer one (2026-10-05..14: NOAA's index froze at 10-05 while S-NPP had 10-12).
    S-NPP's scenes are fetched only when its listing is newer. After S-NPP data end
    (2026-11-02) its listing has nothing new and it drops out."""
    res = run_product(ctx, VIIRS_JPSS, domain, prev_layers, VIIRS_ZOOMS, per_day=False)
    if res.layers:
        listed, failed = newest_listed(ctx, VIIRS_SNPP)
        if failed:
            # NOAA-21/NOAA-20 updated; whether S-NPP holds a newer day is unknown this run
            res.notes.append(f"VIIRS: could not check the S-NPP composite for a day newer than {res.latest_date}: " + "; ".join(failed)[:300])
        if not listed or listed <= (res.latest_date or ""):
            return res
        snpp = run_product(ctx, VIIRS_SNPP, domain, prev_layers, VIIRS_ZOOMS, per_day=False)
        if snpp.layers and (snpp.latest_date or "") > (res.latest_date or ""):
            note = f"VIIRS: the S-NPP composite has a newer day ({snpp.latest_date}) than NOAA-21/NOAA-20 ({res.latest_date}); the newer S-NPP view, labelled S-NPP, is the layer this run."
            return SatelliteResult(snpp.layers, res.errors + snpp.errors, res.notes + [note] + snpp.notes, snpp.latest_date, snpp.bytes_written)
        return SatelliteResult(res.layers, res.errors, res.notes + [f"VIIRS: S-NPP listed {listed} but gave no newer layer; NOAA-21/NOAA-20 kept."] + [f"S-NPP: {e}" for e in snpp.errors], res.latest_date, res.bytes_written)
    snpp = run_product(ctx, VIIRS_SNPP, domain, prev_layers, VIIRS_ZOOMS, per_day=False)
    prev = next((lyr for lyr in prev_layers if lyr.layer_id == f"{VIIRS_JPSS.key}_chl_latest" and _published_intact(ctx.out_dir, lyr)), None)
    if snpp.layers and prev and product_for(prev.provenance.dataset_id) is VIIRS_JPSS and (prev.time.observed_date or "") >= (snpp.latest_date or ""):
        # the last NOAA-21/NOAA-20 view is as new as S-NPP's: keep it (the caller does)
        note = f"VIIRS: NOAA-21 and NOAA-20 unavailable this run; the S-NPP composite (newest {snpp.latest_date}) is not newer than the published NOAA-21/NOAA-20 view ({prev.time.observed_date}), which is kept."
        return SatelliteResult([], res.errors, res.notes + [note], None)
    if snpp.layers:
        snpp.notes.insert(0, "VIIRS: NOAA-21 and NOAA-20 unavailable this run; the S-NPP composite (erdVHNchla1day), labelled S-NPP, is the fallback.")
        return SatelliteResult(snpp.layers, res.errors + snpp.errors, res.notes + snpp.notes, snpp.latest_date, snpp.bytes_written)
    return SatelliteResult([], res.errors + [f"S-NPP fallback: {e}" for e in snpp.errors], res.notes + snpp.notes, None)


def run(ctx: RunContext, prev_layers: list[LayerArtifact], domain: Domain | None = None) -> SatelliteResult:
    """OLCI 300 m (primary) and VIIRS 750 m (fallback). Each runs independently: if OLCI is
    unavailable the VIIRS latest clear view is still published, and the reverse."""
    domain = domain or ctx.options.get("satellite_domain", CALIFORNIA)
    res_o = run_product(ctx, OLCI, domain, prev_layers, OLCI_ZOOMS, per_day=True)
    res_v = run_viirs(ctx, domain, prev_layers)
    notes = res_o.notes + res_v.notes
    # A product that failed this run keeps its previously published layers, with their real
    # observation dates, as long as the other product still updated (a total outage is
    # carried over by the pipeline as a failed source).
    if res_o.layers or res_v.layers:
        for p, res in ((OLCI, res_o), (VIIRS, res_v)):
            if not res.layers:
                kept = [lyr for lyr in prev_layers if lyr.layer_id.startswith(f"{p.key}_chl_") and _published_intact(ctx.out_dir, lyr)]
                if kept:
                    res.layers = kept
                    notes.append(f"{p.title} unavailable this run: kept the previously published layers with their observation dates.")
    if not res_o.layers and res_v.layers:
        notes.append("OLCI unavailable this run: the VIIRS 750 m latest clear view is the fallback.")
    elif not res_o.layers:
        notes.append("Neither OLCI nor VIIRS could be updated this run.")
    extra: list[LayerArtifact] = []
    nbytes = 0
    o_latest = next((lyr for lyr in res_o.layers if lyr.layer_id == f"{OLCI.key}_chl_latest"), None)
    v_latest = next((lyr for lyr in res_v.layers if lyr.layer_id == f"{VIIRS.key}_chl_latest"), None)
    if o_latest and v_latest:
        from . import multisensor

        try:
            days = [lyr for lyr in res_o.layers if lyr.layer_id.startswith(f"{OLCI.key}_chl_2")]
            ms, nbytes = multisensor.build(ctx, o_latest, v_latest, days, OLCI_ZOOMS)
            extra.append(ms)
        except Exception as e:  # the members stay published; only the display is missing
            notes.append(f"multi-sensor view not built: {e}")
    else:
        notes.append("multi-sensor view needs both the Sentinel-3 and the VIIRS latest clear view; not built this run.")
    latest = max([d for d in (res_o.latest_date, res_v.latest_date) if d], default=None)
    return SatelliteResult(res_o.layers + res_v.layers + extra, res_o.errors + res_v.errors, notes, latest, res_o.bytes_written + res_v.bytes_written + nbytes)
