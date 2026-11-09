"""8-day chlorophyll outlook: climatology + damped anomaly (AR(1)) on the research 4 km grid.

What it forecasts: the MEAN chlorophyll of the next NASA 8-day period at each 4 km cell, not a
day-8 map and not toxin. Skill was measured on held-out years (2019-2021-06) with the same
protocol as the research lead experiment (pipeline/scripts/build_chl_outlook_reference.py); the
coefficients and climatology are read from data/reference/chl_outlook_reference.json.

Live input: the newest MODIS-Aqua 8-day composite (NASA L3 mapped 4 km, R2022 near real time, on
NOAA CoastWatch West Coast ERDDAP), i.e. the same sensor, product and 8-day periods as the
climatology, so its anomaly is like-for-like and the forecast is exactly the tested one: the
composite that follows. (Sentinel-3 OLCI was tried first: its departure from the MODIS
climatology is about -1 ln on average in Aug-Oct 2026 in five of six study regions, a sensor
difference that would bias the outlook, so it is not used here.) Cells the composite has no
value for get the climatology alone (and its larger error).
"""

from __future__ import annotations

import functools
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np

REPO = Path(__file__).resolve().parents[3]
REFERENCE = REPO / "data" / "reference" / "chl_outlook_reference.json"
LN10 = math.log(10.0)
MODIS_SERVER = "https://coastwatch.pfeg.noaa.gov/erddap"
MODIS_DATASET = "erdMH1chla8day_R2022NRT"
MODIS_VARIABLE = "chlorophyll"
FLOOR_MG = 0.056616  # the cube's ln(max(chl, 0.056616))
MAX_PREDICTOR_AGE_DAYS = 16  # newest composite must have started within this many days of the run
BAY_SHARE_MASK = 0.5  # a cell is masked when at least this share of its marine water is bay


@functools.lru_cache(maxsize=1)
def load_reference(path: str = str(REFERENCE)) -> dict:
    ref = json.loads(Path(path).read_text())
    ref["_clim"] = np.asarray(ref["climatology_ln"], dtype=np.float64)  # (P, periods)
    ref["_px"] = np.asarray(ref["pixels"], dtype=np.int64)  # (P, 2) row, col
    ref["_phi8"] = np.asarray(ref["ar1_phi_pixel_8d"] if ref.get("ar1_phi_pixel_8d") else [ref["ar1_phi_by_lag_days"]["8"]] * len(ref["pixels"]), dtype=np.float64)
    ref["_rmse_ar"] = np.asarray(ref["expected_rmse_ln_climAR1"], dtype=np.float64)
    ref["_rmse_cl"] = np.asarray(ref["expected_rmse_ln_climatology"], dtype=np.float64)
    return ref


def period(d: date, periods: int = 46) -> int:
    return min((d.timetuple().tm_yday - 1) // 8, periods - 1)


def next_period(start: date) -> tuple[date, date]:
    """(first, last day) of the NASA 8-day period after the one starting on `start`
    (periods start on day-of-year 1, 9, 17, ...; the last of a year ends on 31 December)."""
    nxt = start + timedelta(days=8)
    if nxt.year != start.year:
        nxt = date(start.year + 1, 1, 1)
    end = nxt + timedelta(days=7)
    if end.year != nxt.year:
        end = date(nxt.year, 12, 31)
    return nxt, end


# ---------------------------------------------------------------- live input
def times_url(start: date) -> str:
    return f"{MODIS_SERVER}/griddap/{MODIS_DATASET}.csv0?" + quote(f"time[({start.isoformat()}T00:00:00Z):1:(last)]", safe=":(),")


def composite_url(t: str, ref: dict) -> str:
    g = ref["grid"]
    lat0, lat1 = g["lat_first"], g["lat_first"] + g["lat_step"] * (g["height"] - 1)
    lon0, lon1 = g["lon_first"], g["lon_first"] + g["lon_step"] * (g["width"] - 1)
    q = f"{MODIS_VARIABLE}[({t})][({lat0:.5f}):({lat1:.5f})][({lon0:.5f}):({lon1:.5f})]"
    return f"{MODIS_SERVER}/griddap/{MODIS_DATASET}.nc?" + quote(q, safe=",():")


def parse_composite(blob: bytes, ref: dict) -> np.ndarray:
    """ERDDAP NetCDF -> ln chlorophyll on the reference grid (NaN where no value)."""
    from ..sources.charm import parse_netcdf

    arr, _, _ = parse_netcdf(blob)
    v = np.asarray(arr[MODIS_VARIABLE], dtype=np.float64).reshape(len(arr["latitude"]), len(arr["longitude"]))
    lat, lon = np.asarray(arr["latitude"], float), np.asarray(arr["longitude"], float)
    g = ref["grid"]
    r = np.round((lat - g["lat_first"]) / g["lat_step"]).astype(int)
    c = np.round((lon - g["lon_first"]) / g["lon_step"]).astype(int)
    if not (np.allclose(g["lat_first"] + r * g["lat_step"], lat, atol=1e-3) and np.allclose(g["lon_first"] + c * g["lon_step"], lon, atol=1e-3)):
        raise ValueError("MODIS composite is not on the reference 4 km lattice")
    out = np.full((g["height"], g["width"]), np.nan)
    okr, okc = (r >= 0) & (r < g["height"]), (c >= 0) & (c < g["width"])
    with np.errstate(invalid="ignore"):
        ln = np.where(np.isfinite(v) & (v > 0), np.log(np.maximum(v, FLOOR_MG)), np.nan)
    out[np.ix_(r[okr], c[okc])] = ln[np.ix_(okr, okc)]
    return out


def fetch_latest(ctx, ref: dict) -> tuple[date, np.ndarray, list[str]]:
    """(start date of the newest MODIS-Aqua 8-day composite, ln grid, request urls)."""
    u1 = times_url(ctx.now.date() - timedelta(days=40))
    lines = [ln.strip() for ln in ctx.fetcher(u1).body.decode().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("no MODIS-Aqua 8-day composite in the last 40 days")
    t = lines[-1]
    u2 = composite_url(t, ref)
    grid = parse_composite(ctx.fetcher(u2).body, ref)
    return datetime.fromisoformat(t.replace("Z", "+00:00")).date(), grid, [u1, u2]


# ---------------------------------------------------------------- bays
_BAY: dict[tuple, np.ndarray] = {}


def bay_share(ref: dict) -> np.ndarray:
    """Per footprint pixel: the share of the 4 km cell's marine water that is bay (estimate.bay_mask
    on the Sentinel-3 OLCI 300 m lattice of the marine water mask). A cell with no marine water of
    its own (a land-contaminated coastal cell) takes the share of its 3 x 3 neighbourhood; NaN when
    that has none either (outside the mask's domain)."""
    from scipy.ndimage import uniform_filter

    from ..sources import satellite
    from . import estimate as E

    g = ref["grid"]
    key = (g["lat_first"], g["lat_step"], g["lon_first"], g["lon_step"], g["height"], g["width"], ref["_px"].tobytes())
    if key not in _BAY:
        t = satellite.target_for(satellite.OLCI, satellite.CALIFORNIA)
        water = satellite.marine_water(t)
        bay = E.bay_mask(water, t.step, t.lat_first - t.step * t.height / 2, (t.lat_first, t.lon_first))
        # 4 km cell of each fine cell (cell edges half a step either side of the centres)
        rr = np.floor((t.lats() - (g["lat_first"] - g["lat_step"] / 2)) / g["lat_step"]).astype(int)
        cc = np.floor((t.lons() - (g["lon_first"] - g["lon_step"] / 2)) / g["lon_step"]).astype(int)
        okr, okc = (rr >= 0) & (rr < g["height"]), (cc >= 0) & (cc < g["width"])
        sub_w, sub_b = water[np.ix_(okr, okc)], bay[np.ix_(okr, okc)]
        R, C = np.meshgrid(rr[okr], cc[okc], indexing="ij")
        nw = np.zeros((g["height"], g["width"]))
        nb = np.zeros_like(nw)
        np.add.at(nw, (R[sub_w], C[sub_w]), 1.0)
        np.add.at(nb, (R[sub_b], C[sub_b]), 1.0)
        nw3, nb3 = uniform_filter(nw, 3, mode="constant"), uniform_filter(nb, 3, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(nw > 0, nb / np.maximum(nw, 1.0), np.where(nw3 > 0, nb3 / np.maximum(nw3, 1e-12), np.nan))
        px = ref["_px"]
        _BAY.clear()
        _BAY[key] = share[px[:, 0], px[:, 1]]
    return _BAY[key]


def bay_cells(ref: dict) -> np.ndarray:
    """True for footprint pixels masked as bay (bay share >= BAY_SHARE_MASK)."""
    s = bay_share(ref)
    return np.isfinite(s) & (s >= BAY_SHARE_MASK)


# ---------------------------------------------------------------- forecast
def outlook(latest_ln_grid: np.ndarray, start: date, ref: dict | None = None, masked: np.ndarray | None = None) -> dict:
    """Forecast of the next 8-day period after the composite that starts on `start`. Footprint
    pixels in `masked` (default: bay_cells) get NaN forecast and uncertainty."""
    ref = ref or load_reference()
    px = ref["_px"]
    masked = bay_cells(ref) if masked is None else masked
    obs = latest_ln_grid[px[:, 0], px[:, 1]]
    has = np.isfinite(obs) & ~masked
    p_now = period(start, ref["periods"])
    v0, v1 = next_period(start)
    p_next = period(v0, ref["periods"])
    anom = np.where(has, obs - ref["_clim"][:, p_now], 0.0)
    clim_t = ref["_clim"][:, p_next]
    fc_ln = clim_t + ref["_phi8"] * anom
    rmse_ln = np.where(has, ref["_rmse_ar"], ref["_rmse_cl"])
    return {
        "forecast_log10": np.where(masked, np.nan, fc_ln / LN10),
        "climatology_log10": np.where(masked, np.nan, clim_t / LN10),
        "uncertainty_log10": np.where(masked, np.nan, rmse_ln / LN10),
        "has_anomaly": has,
        "masked_bay": masked,
        "mean_anomaly_ln": float(anom[has].mean()) if has.any() else None,
        "cells": int((~masked).sum()),
        "cells_masked_bay": int(masked.sum()),
        "cells_with_anomaly": int(has.sum()),
        "valid_from": v0,
        "valid_to": v1,
        "predictor_window": (start, start + timedelta(days=7)),
    }


def to_grid(values: np.ndarray, ref: dict) -> np.ndarray:
    """Footprint pixel values onto the full 4 km grid (NaN elsewhere)."""
    g = ref["grid"]
    out = np.full((g["height"], g["width"]), np.nan)
    out[ref["_px"][:, 0], ref["_px"][:, 1]] = values
    return out
