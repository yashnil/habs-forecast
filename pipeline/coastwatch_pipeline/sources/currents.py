"""Observed surface currents from the US West Coast HF-radar network (HFRNet totals, 2 km).

Source: NOAA CoastWatch ERDDAP `ucsdHfrW2` (Scripps / IOOS HFRNet real-time vectors, hourly).
When that ERDDAP does not serve it (HTTP 403 to cloud runners, "Currently unknown datasetID"
while it reloads, no answer), the same product is read from NOAA NDBC's THREDDS server
(`hfradar_uswc_2km`, OPeNDAP). Checked 2026-10-14 on the two newest hours published from
ERDDAP (2026-10-12T03Z and 10Z): same lattice and cell indices, and after the checks below the
same 9,869 and 10,127 cells with values equal within the published quantization (3.8e-5 m/s).
Each hour records the server it came from.
Upstream already requires at least two contributing radars and limits the geometric
dilution of precision (observed: no published cell has HDOP above 1.24). This module:

- requests exact grid indices (no nearest-bound snapping) and checks the returned
  latitude, longitude and time axes against the dataset's own axes;
- re-checks each cell: fill values, >= MIN_SITES radars, HDOP <= HDOP_MAX, speed <=
  SPEED_MAX (physically implausible beyond that for this coast); failing cells become
  no-value and are counted, never repaired;
- publishes each of the last WINDOW_HOURS hours as its own layer (u, v grids on the
  source lattice; the browser draws arrows from them), plus a 24-hour mean where a cell has at least
  MEAN_MIN_HOURS valid hours; nothing is interpolated in space or time, and gaps stay gaps;
- reuses already-published hours older than REFETCH_HOURS (upstream rarely revises
  them), so a run fetches only the newest hours.

These are observations of the past. Nothing here forecasts, and the hourly fields are
snapshots: the map must not imply motion between them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np

from ..context import RunContext
from ..http import FetchError
from ..models import FreshnessPolicy, LayerArtifact, Provenance, QCCheck, QualityControl, TimeInfo, ValueGrid, VectorField
from ..process import grid as gridcodec
from ..process import opendap
from . import satellite as sat
from .charm import parse_netcdf

SOURCE_ID = "hf_radar"
GROUP_ID = "currents"
SERVER = "https://coastwatch.pfeg.noaa.gov/erddap"
DATASET = "ucsdHfrW2"
VARS = ("water_u", "water_v", "hdop", "number_of_sites")
NDBC_URL = "https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_uswc_2km"
NDBC_DATASET = "hfradar_uswc_2km"
# the same variables on NDBC's server, in the order of VARS
NDBC_VARS = ("u", "v", "hdop", "number_of_sites")


@dataclass(frozen=True)
class Host:
    key: str
    dataset: str
    source_url: str
    served_by: str


ERDDAP = Host("erddap", DATASET, f"{SERVER}/griddap/{DATASET}.html", "NOAA CoastWatch West Coast (ERDDAP)")
NDBC = Host("ndbc", NDBC_DATASET, f"{NDBC_URL}.html", "NOAA National Data Buoy Center (THREDDS)")
HOSTS = (ERDDAP, NDBC)


def host_for(dataset_id: str | None) -> Host:
    return NDBC if (dataset_id or "").split(",")[0] == NDBC_DATASET else ERDDAP
WINDOW_HOURS = 24
REFETCH_HOURS = 6
BATCH_HOURS = 6
MEAN_MIN_HOURS = 18
MIN_SITES = 2
HDOP_MAX = 1.25
SPEED_MAX = 2.0  # m s-1
V_RANGE = (-2.5, 2.5)  # m s-1, fixed quantization range (contract)
NATIVE_M = 2000
PROCESSING_VERSION = "currents-processing-1"

FRESHNESS = FreshnessPolicy(
    basis="observed_date",
    current_max_age_days=1,
    stale_max_age_days=3,
    note="HF-radar totals are published about 4-15 hours after observation. Current: observed today or yesterday (UTC). Stale: 2-3 days. Historical: older.",
)
CAVEATS = [
    "Observed surface water movement (top metre or so), not where a bloom will be, not bloom growth, not toxin.",
    "Each hour is a separate snapshot; the map does not show what happened between hours, and nothing here is a forecast.",
    "Coverage varies hour to hour; gaps are no data, not calm water. Near the coast, inside bays and at the edge of coverage the uncertainty is higher.",
]
CAVEAT_MEAN = (
    f"The 24-hour mean averages each cell's valid hours (at least {MEAN_MIN_HOURS} of 24). It suppresses most of the daily "
    "sea-breeze and tidal back-and-forth; it is not a forecast and does not describe any single hour."
)


@dataclass
class Axes:
    lat: np.ndarray  # ascending
    lon: np.ndarray  # ascending
    i0: int
    i1: int
    j0: int
    j1: int
    host: Host = ERDDAP
    # NDBC only: variable attributes (scale, fill, units) and the index of each listed time
    attrs: dict = field(default_factory=dict)
    time_index: dict[str, int] = field(default_factory=dict)

    @property
    def lats(self) -> np.ndarray:
        return self.lat[self.i0 : self.i1 + 1]

    @property
    def lons(self) -> np.ndarray:
        return self.lon[self.j0 : self.j1 + 1]


@dataclass
class Hour:
    time: str
    u: np.ndarray
    v: np.ndarray
    checks: list[QCCheck] = field(default_factory=list)
    reused: LayerArtifact | None = None
    url: str = ""
    host: Host = ERDDAP


def _csv(ctx: RunContext, url: str) -> list[str]:
    return [x.strip() for x in ctx.fetcher(url).body.decode().splitlines() if x.strip()]


def axes(ctx: RunContext, d: sat.Domain, host: Host = ERDDAP) -> Axes:
    attrs: dict = {}
    if host is NDBC:
        a = opendap.parse_dods(ctx.fetcher(f"{NDBC_URL}.dods?lat,lon").body)
        # Float32 axes, written the way ERDDAP's CSV writes the same values (32.4076, not
        # 32.40760040283203), so published grids are identical whichever server they came from
        lat, lon = (np.array([float(str(np.float32(x))) for x in a[k]]) for k in ("lat", "lon"))
        attrs = opendap.parse_das(ctx.fetcher(f"{NDBC_URL}.das").body.decode("utf-8", "replace"))
    else:
        lat = np.array([float(x) for x in _csv(ctx, f"{SERVER}/griddap/{DATASET}.csv0?latitude")])
        lon = np.array([float(x) for x in _csv(ctx, f"{SERVER}/griddap/{DATASET}.csv0?longitude")])
    if not (np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0)):
        raise ValueError("HF-radar axes are not ascending")
    ii = np.nonzero((lat >= d.lat_s) & (lat <= d.lat_n))[0]
    jj = np.nonzero((lon >= d.lon_w) & (lon <= d.lon_e))[0]
    if not ii.size or not jj.size:
        raise ValueError("domain outside the HF-radar grid")
    return Axes(lat, lon, int(ii[0]), int(ii[-1]), int(jj[0]), int(jj[-1]), host, attrs)


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_hours(ctx: RunContext, ax: Axes | None = None) -> list[str]:
    start = ctx.now - timedelta(hours=WINDOW_HOURS + 12)
    lo = f"{start:%Y-%m-%dT%H}:00:00Z"
    if ax is not None and ax.host is NDBC:
        units = str((ax.attrs.get("time") or {}).get("units", ""))
        if not units.startswith("seconds since 1970-01-01"):
            raise ValueError(f"unexpected NDBC time units {units!r}")
        t = opendap.parse_dods(ctx.fetcher(f"{NDBC_URL}.dods?time").body)["time"]
        ax.time_index = {_iso(x): k for k, x in enumerate(np.atleast_1d(t))}
        return sorted(x for x in ax.time_index if lo <= x <= ctx.now_iso)
    q = f"time[({start:%Y-%m-%dT%H}:00:00Z):1:(last)]"
    try:
        ts = _csv(ctx, f"{SERVER}/griddap/{DATASET}.csv0?" + quote(q, safe=":(),"))
    except FetchError as e:
        if sat.no_matching_results(e):
            return []
        raise
    ts = [t if t.endswith("Z") else t + "Z" for t in ts]
    # ERDDAP snaps the start bound to the nearest time; keep what is inside the window
    return [t for t in ts if lo <= t <= ctx.now_iso]


def batch_url(ax: Axes, t0: str, t1: str) -> str:
    if ax.host is NDBC:
        sel = f"[{ax.time_index[t0]}:1:{ax.time_index[t1]}][{ax.i0}:1:{ax.i1}][{ax.j0}:1:{ax.j1}]"
        return f"{NDBC_URL}.dods?" + quote(",".join(v + sel for v in NDBC_VARS), safe=",:")
    sel = f"[({t0}):1:({t1})][{ax.i0}:1:{ax.i1}][{ax.j0}:1:{ax.j1}]"
    return f"{SERVER}/griddap/{DATASET}.nc?" + quote(",".join(v + sel for v in VARS), safe=",():")


def _fill(a: np.ndarray, attrs: dict) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    fv = attrs.get("_FillValue")
    if fv is not None:
        a = np.where(np.isclose(a, float(fv)), np.nan, a)
    return a


def _ndbc_arrays(body: bytes, ax: Axes) -> tuple[dict, dict]:
    """NDBC's OPeNDAP response under ERDDAP's names, unpacked with the dataset's own attributes
    (packed integers: scale_factor, add_offset, _FillValue; DAP2 bytes are unsigned, so a negative
    byte fill value is compared as a signed byte)."""
    raw = opendap.parse_dods(body)
    missing = [v for v in NDBC_VARS + ("time", "lat", "lon") if v not in raw]
    if missing:
        raise ValueError(f"response lacks {missing}")
    arrays: dict = {"time": raw["time"], "latitude": raw["lat"], "longitude": raw["lon"]}
    vattrs: dict = {}
    for name, src in zip(VARS, NDBC_VARS):
        at = ax.attrs.get(src) or {}
        a = np.asarray(raw[src])
        fv = at.get("_FillValue")
        if a.dtype == np.uint8 and fv is not None and float(fv) < 0:
            a = a.astype(np.int8)
        a = a.astype(float)
        if fv is not None:
            a[a == float(fv)] = np.nan
        a = a * float(at.get("scale_factor", 1.0)) + float(at.get("add_offset", 0.0))
        arrays[name] = a
        vattrs[name] = {"units": at.get("units")}
    return arrays, vattrs


def load_batch(body: bytes, ax: Axes, times: list[str], url: str) -> list[Hour]:
    if ax.host is NDBC:
        arrays, vattrs = _ndbc_arrays(body, ax)
    else:
        arrays, vattrs, _ = parse_netcdf(body)
    missing = [v for v in VARS + ("time", "latitude", "longitude") if v not in arrays]
    if missing:
        raise ValueError(f"response lacks {missing}")
    lat, lon = np.asarray(arrays["latitude"], float), np.asarray(arrays["longitude"], float)
    if lat.shape != ax.lats.shape or lon.shape != ax.lons.shape or not np.allclose(lat, ax.lats, atol=1e-6) or not np.allclose(lon, ax.lons, atol=1e-6):
        raise ValueError("returned latitude/longitude do not match the requested grid indices")
    got = [_iso(t) for t in np.atleast_1d(arrays["time"])]
    if got != times:
        raise ValueError(f"returned times {got[:3]}... do not match requested {times[:3]}...")
    units = {v: (vattrs.get(v) or {}).get("units") for v in ("water_u", "water_v")}
    if any(u not in ("m s-1", "m/s") for u in units.values()):
        raise ValueError(f"unexpected velocity units {units}")
    shape = (len(times), lat.size, lon.size)
    u, v = (_fill(arrays[k], vattrs[k]).reshape(shape) for k in ("water_u", "water_v"))
    hd = _fill(arrays["hdop"], vattrs["hdop"]).reshape(shape)
    ns = _fill(arrays["number_of_sites"], vattrs["number_of_sites"]).reshape(shape)
    out = []
    for k, t in enumerate(times):
        uu, vv = u[k].copy(), v[k].copy()
        valid = np.isfinite(uu) & np.isfinite(vv)
        few = valid & ~(ns[k] >= MIN_SITES)
        geo = valid & ~few & ~(hd[k] <= HDOP_MAX)
        sp = np.hypot(uu, vv)
        fast = valid & ~few & ~geo & (sp > SPEED_MAX)
        drop = few | geo | fast
        uu[drop | ~valid], vv[drop | ~valid] = np.nan, np.nan
        n_ok = int(np.isfinite(uu).sum())
        checks = [
            QCCheck(name="grid_alignment", passed=True, detail=f"rows {ax.i0}-{ax.i1}, cols {ax.j0}-{ax.j1} of {ax.host.dataset}; returned axes equal the dataset's"),
            QCCheck(name="min_sites", passed=True, detail=f"{int(few.sum())} cells with fewer than {MIN_SITES} radars dropped"),
            QCCheck(name="hdop", passed=True, detail=f"{int(geo.sum())} cells with HDOP above {HDOP_MAX} (or missing) dropped"),
            QCCheck(name="speed_plausible", passed=int(fast.sum()) <= max(5, n_ok // 1000), detail=f"{int(fast.sum())} cells faster than {SPEED_MAX} m/s dropped"),
        ]
        out.append(Hour(t, uu, vv, checks, url=url, host=ax.host))
    return out


# ---------------------------------------------------------------- publishing
def _vgrid(ax: Axes, url: str) -> ValueGrid:
    scale = (V_RANGE[1] - V_RANGE[0]) / gridcodec.MAX_CODE
    lats, lons = ax.lats, ax.lons
    return ValueGrid(
        url=url, width=lons.size, height=lats.size, lat_first=float(lats[0]), lat_step=float(np.mean(np.diff(lats))),
        lon_first=float(lons[0]), lon_step=float(np.mean(np.diff(lons))), scale_factor=scale, add_offset=V_RANGE[0],
        max_quantization_error=scale / 2,
    )


def quantize(a: np.ndarray) -> np.ndarray:
    """The values a reader decodes from the published grid."""
    scale = (V_RANGE[1] - V_RANGE[0]) / gridcodec.MAX_CODE
    q = np.full(a.shape, np.nan)
    ok = np.isfinite(a)
    q[ok] = np.clip(np.round((a[ok] - V_RANGE[0]) / scale), 0, gridcodec.MAX_CODE) * scale + V_RANGE[0]
    return q


def direction_deg(u, v):
    """Direction the water moves TOWARD, degrees clockwise from north (oceanographic convention)."""
    return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0


def coverage(ax: Axes, u: np.ndarray):
    rr, cc = np.nonzero(np.isfinite(u))
    lats, lons = ax.lats, ax.lons
    lat, lon = lats[rr], lons[cc]
    dlat, dlon = float(np.mean(np.diff(lats))), float(np.mean(np.diff(lons)))
    px_km2 = (dlat * 111.32) * (dlon * 111.32 * np.cos(np.radians(lat)))
    return sat.coverage_points(lat, lon, px_km2, (float(lats[0]), float(lats[-1]), float(lons[0]), float(lons[-1])))


def _provenance(ctx: RunContext, urls: list[str], meta: dict[str, str], hosts: list[Host] | None = None) -> Provenance:
    """`hosts`: the servers the values came from, in HOSTS order (the first is source_url)."""
    hosts = [h for h in HOSTS if h in (hosts or [ERDDAP])]
    return Provenance(
        source_id=SOURCE_ID, source_name="US West Coast HF-radar surface currents (HFRNet real-time vectors, 2 km, hourly)",
        source_url=hosts[0].source_url, dataset_id=",".join(h.dataset for h in hosts),
        institution="Scripps Institution of Oceanography / IOOS HFRNet; served by " + " and ".join(h.served_by for h in hosts),
        license="Public domain (U.S. Government and IOOS data); credit HFRNet and the operating institutions",
        retrieved_at=ctx.now_iso, request_urls=urls[:12], upstream_metadata=meta,
        pipeline_version=ctx.pipeline_version, pipeline_run_id=ctx.run_id,
    )


def write_field(ctx: RunContext, ax: Axes, base: str, u: np.ndarray, v: np.ndarray) -> tuple[VectorField, int]:
    out = ctx.out_dir / base
    out.mkdir(parents=True, exist_ok=True)
    bu, *_ = gridcodec.encode(u, *V_RANGE)
    bv, *_ = gridcodec.encode(v, *V_RANGE)
    (out / "u.u16.gz").write_bytes(bu)
    (out / "v.u16.gz").write_bytes(bv)
    # no arrows file: the browser draws arrows from these two grids (about 15 KB each per
    # hour, against about 1.4 MB for an arrows GeoJSON of the same cells)
    sp = np.hypot(quantize(u), quantize(v))
    vf = VectorField(
        u_grid=_vgrid(ax, f"{base}/u.u16.gz"), v_grid=_vgrid(ax, f"{base}/v.u16.gz"), depth_m=0.0,
        speed_max=round(float(np.nanmax(sp)), 3) if np.isfinite(sp).any() else 0.0, arrows_url=None,
    )
    return vf, len(bu) + len(bv)


def _layer_common(ax: Axes, u: np.ndarray, v: np.ndarray, checks: list[QCCheck]) -> dict:
    """Shared fields; QC value range is the current speed (m/s) of the published values."""
    n_ok = int(np.isfinite(u).sum())
    sp = np.hypot(quantize(u), quantize(v))
    return dict(
        group_id=GROUP_ID, product_class="observation", variable="surface_current", units="m s-1",
        resolution_deg=float(np.mean(np.diff(ax.lats))), native_resolution_m=NATIVE_M, platforms=["HF radar (HFRNet)"], freshness=FRESHNESS,
        qc=QualityControl(
            checks=checks, n_cells=int(u.size), n_valid=n_ok, valid_fraction=round(n_ok / u.size, 6) if u.size else 0.0,
            value_min=round(float(np.nanmin(sp)), 4) if n_ok else None, value_max=round(float(np.nanmax(sp)), 4) if n_ok else None,
        ),
        coverage=coverage(ax, u),
    )


def hour_layer(ctx: RunContext, ax: Axes, h: Hour) -> tuple[LayerArtifact, int]:
    key = hashlib.sha256(np.nan_to_num(quantize(h.u), nan=9).tobytes() + np.nan_to_num(quantize(h.v), nan=9).tobytes() + PROCESSING_VERSION.encode()).hexdigest()[:10]
    stamp = h.time[:13].replace("-", "").replace("T", "T")
    base = f"currents/hfr2km/{stamp}Z-{key}"
    vf, nbytes = write_field(ctx, ax, base, h.u, h.v)
    layer = LayerArtifact(
        layer_id=f"hfr2km_currents_{stamp}Z", title="Observed surface currents, HF radar 2 km, one hour", short_title="Currents · HF radar 2 km",
        description=f"Surface current vectors observed by the HF-radar network for the hour {h.time} (HFRNet total vectors, 2 km). Cells failing quality checks or without enough radar coverage have no value.",
        time=TimeInfo(observed_date=h.time[:10], valid_date=h.time[:10], valid_time=h.time, observed_times=[h.time]),
        vectors=vf, caveats=list(CAVEATS),
        provenance=_provenance(ctx, [h.url], {"hour": h.time, "processing": PROCESSING_VERSION}, [h.host]),
        **_layer_common(ax, h.u, h.v, h.checks),
    )
    return layer, nbytes


def mean_layer(ctx: RunContext, ax: Axes, hours: list[Hour]) -> tuple[LayerArtifact | None, int]:
    if len(hours) < MEAN_MIN_HOURS:
        return None, 0
    U = np.stack([h.u for h in hours])
    V = np.stack([h.v for h in hours])
    n = np.isfinite(U).sum(axis=0)
    with np.errstate(invalid="ignore"):
        um = np.where(n >= MEAN_MIN_HOURS, np.nansum(U, axis=0) / np.maximum(n, 1), np.nan)
        vm = np.where(n >= MEAN_MIN_HOURS, np.nansum(V, axis=0) / np.maximum(n, 1), np.nan)
    if not np.isfinite(um).any():
        return None, 0
    times = sorted(h.time for h in hours)
    key = hashlib.sha256(("|".join(times) + PROCESSING_VERSION).encode() + np.nan_to_num(quantize(um), nan=9).tobytes()).hexdigest()[:10]
    base = f"currents/hfr2km/mean24h-{times[-1][:13].replace('-', '')}Z-{key}"
    vf, nbytes = write_field(ctx, ax, base, um, vm)
    checks = [QCCheck(name="mean_min_hours", passed=True, detail=f"mean published where a cell has at least {MEAN_MIN_HOURS} of {len(hours)} valid hours ({int((n >= MEAN_MIN_HOURS).sum())} cells); others left without a value")]
    layer = LayerArtifact(
        layer_id="hfr2km_currents_mean24h", title="Observed surface currents, HF radar 2 km, 24-hour mean", short_title="Currents · 24 h mean",
        description=f"Mean of the hourly HF-radar surface currents from {times[0]} to {times[-1]} at each cell with at least {MEAN_MIN_HOURS} valid hours.",
        time=TimeInfo(observed_date=times[-1][:10], valid_date=times[-1][:10], valid_time=times[-1], observed_times=[times[0], times[-1]]),
        vectors=vf, caveats=[CAVEAT_MEAN] + list(CAVEATS),
        provenance=_provenance(
            ctx, sorted({h.url for h in hours if h.url}), {"hours": f"{len(hours)} hours {times[0]}..{times[-1]}", "processing": PROCESSING_VERSION}, [h.host for h in hours]
        ),
        **_layer_common(ax, um, vm, checks),
    )
    return layer, nbytes


def load_published_field(ctx: RunContext, lyr: LayerArtifact) -> tuple[np.ndarray, np.ndarray]:
    assert lyr.vectors
    out = []
    for g in (lyr.vectors.u_grid, lyr.vectors.v_grid):
        q = gridcodec.decode((ctx.out_dir / g.url).read_bytes(), g.width, g.height, g.scale_factor, g.add_offset)
        out.append(q)
    return out[0], out[1]


# ---------------------------------------------------------------- run
@dataclass
class CurrentsResult:
    layers: list[LayerArtifact]
    errors: list[str]
    notes: list[str]
    latest_time: str | None
    bytes_written: int = 0


def run(ctx: RunContext, prev_layers: list[LayerArtifact]) -> CurrentsResult:
    d = ctx.options.get("currents_domain") or ctx.options.get("satellite_domain") or sat.CALIFORNIA
    errors: list[str] = []
    notes: list[str] = []
    opened: dict[str, tuple[Axes, list[str]] | None] = {}
    failures: list[str] = []

    def open_host(host: Host) -> tuple[Axes, list[str]] | None:
        """Axes and listed hours of one server, once per run (None if it fails)."""
        if host.key not in opened:
            try:
                a = axes(ctx, d, host)
                opened[host.key] = (a, list_hours(ctx, a))
            except Exception as e:
                failures.append(f"{host.dataset}: axes or time listing failed: {e}")
                opened[host.key] = None
        return opened[host.key]

    # Both servers are listed (three small requests each) and the one with the newer hour is
    # used, ERDDAP on a tie: a server that refuses, is reloading, or whose index has stopped
    # moving never hides newer hours that the other one has.
    got = open_host(ERDDAP)
    alt = open_host(NDBC)
    newest = lambda o: o[1][-1] if o and o[1] else ""  # noqa: E731
    if newest(alt) > newest(got):
        why = next((f for f in failures if f.startswith(DATASET)), None) or (f"newest hour listed {newest(got)}" if newest(got) else "no recent hour listed")
        notes.append(f"{DATASET} on CoastWatch ERDDAP: {why[:200]}; the same HFRNet product was read from NOAA NDBC ({NDBC_DATASET}, newest {newest(alt)}).")
        got = alt
    elif alt is None and got is not None:
        notes.append(f"NOAA NDBC copy ({NDBC_DATASET}) not available this run: {failures[-1][:160] if failures else ''}")
    if got is None:
        return CurrentsResult([], failures[:2], notes, None)
    ax, listed = got
    if not listed:
        return CurrentsResult([], failures + [f"{ax.host.dataset}: no hourly field in the last {WINDOW_HOURS + 12} hours"], notes, None)
    newest = datetime.fromisoformat(listed[-1].replace("Z", "+00:00"))
    window = [t for t in listed if datetime.fromisoformat(t.replace("Z", "+00:00")) > newest - timedelta(hours=WINDOW_HOURS)]
    prev = {lyr.time.valid_time: lyr for lyr in prev_layers if lyr.layer_id.startswith("hfr2km_currents_2") and lyr.vectors}
    hours: dict[str, Hour] = {}
    fetch_times = []
    for t in window:
        p = prev.get(t)
        old = datetime.fromisoformat(t.replace("Z", "+00:00")) <= newest - timedelta(hours=REFETCH_HOURS)
        if p and old and p.provenance.upstream_metadata.get("processing") == PROCESSING_VERSION and all((ctx.out_dir / g.url).exists() for g in (p.vectors.u_grid, p.vectors.v_grid)):
            try:
                u, v = load_published_field(ctx, p)
                if u.shape == (ax.i1 - ax.i0 + 1, ax.j1 - ax.j0 + 1):
                    hours[t] = Hour(t, u, v, list(p.qc.checks) if p.qc else [], reused=p, host=host_for(p.provenance.dataset_id))
                    continue
            except Exception:
                pass
        fetch_times.append(t)
    # contiguous runs of hours, at most BATCH_HOURS per request
    batches: list[list[str]] = []
    for t in fetch_times:
        tt = datetime.fromisoformat(t.replace("Z", "+00:00"))
        if batches and len(batches[-1]) < BATCH_HOURS and datetime.fromisoformat(batches[-1][-1].replace("Z", "+00:00")) + timedelta(hours=1) == tt:
            batches[-1].append(t)
        else:
            batches.append([t])
    for b in batches:
        url = batch_url(ax, b[0], b[-1])
        try:
            for h in load_batch(ctx.fetcher(url).body, ax, b, url):
                hours[h.time] = h
            continue
        except Exception as e:
            first = f"{ax.host.dataset} {b[0]}..{b[-1]}: {e}"
        # the same hours from the other server, if it lists them (same lattice, checked per batch)
        other = next((h for h in HOSTS if h is not ax.host), None)
        alt = open_host(other) if other else None
        if alt is None or not set(b) <= set(alt[1]):
            errors.append(first)
            continue
        ax2 = alt[0]
        if (ax2.i0, ax2.i1, ax2.j0, ax2.j1) != (ax.i0, ax.i1, ax.j0, ax.j1) or not (np.allclose(ax2.lats, ax.lats, atol=1e-6) and np.allclose(ax2.lons, ax.lons, atol=1e-6)):
            errors.append(first + f" ({other.dataset} grid differs; not used)")
            continue
        url2 = batch_url(ax2, b[0], b[-1])
        try:
            for h in load_batch(ctx.fetcher(url2).body, ax2, b, url2):
                hours[h.time] = h
            notes.append(f"{b[0]}..{b[-1]} read from {other.dataset} after {first[:160]}")
        except Exception as e2:
            errors.append(first)
            errors.append(f"{other.dataset} {b[0]}..{b[-1]}: {e2}")
    empty = sorted(t for t, h in hours.items() if not np.isfinite(h.u).any())
    for t in empty:
        del hours[t]  # no valid cell anywhere in the domain: missing, not an empty field
    if empty:
        notes.append(f"{len(empty)} hours without a single valid cell in the domain left out.")
    if not hours:
        return CurrentsResult([], errors or [f"{ax.host.dataset}: no valid HF-radar cell in the domain in the last {WINDOW_HOURS} hours"], notes, None)
    if len(hours) < len(window):
        notes.append(f"{len(window) - len(hours)} of {len(window)} hours could not be loaded this run; they are missing, not filled.")
    n_reused = sum(1 for h in hours.values() if h.reused)
    if n_reused:
        notes.append(f"{n_reused} hours older than {REFETCH_HOURS} h reused from the published dataset.")
    layers: list[LayerArtifact] = []
    nbytes = 0
    ordered = [hours[t] for t in sorted(hours)]
    for h in ordered:
        if h.reused is not None:
            layers.append(h.reused)
            continue
        lyr, b = hour_layer(ctx, ax, h)
        layers.append(lyr)
        nbytes += b
    m, b = mean_layer(ctx, ax, ordered)
    if m:
        layers.append(m)
        nbytes += b
    else:
        notes.append(f"24-hour mean not published: fewer than {MEAN_MIN_HOURS} hours available.")
    notes.append(f"{len(ordered)} hourly fields {ordered[0].time} .. {ordered[-1].time}; newest {ordered[-1].time}.")
    return CurrentsResult(layers, errors, notes, ordered[-1].time, nbytes)
