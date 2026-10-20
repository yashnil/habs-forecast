"""Sea surface temperature for v1/derived/: JPL MUR SST v4.1 (0.01 deg, daily), via NOAA CoastWatch
West Coast ERDDAP (jplMURSST41), clipped to the satellite chlorophyll scope (coastal ocean, marine
bays and estuaries).

MUR is an ANALYSIS: a gap-free daily field blended from several satellite sensors (infrared and
microwave) and in-situ data, interpolated where clouds hide the sea. It is not a direct observation
of each 1 km cell; near shore, in bays and under persistent fog it leans on interpolation.
Values are published as received (degrees C, quantized); nothing is smoothed or filled here.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from urllib.parse import quote

import numpy as np

from ..sources import satellite

SERVER = "https://coastwatch.pfeg.noaa.gov/erddap"
DATASET = "jplMURSST41"
VARIABLE = "analysed_sst"
STEP = 0.01
RANGE_C = (-2.0, 40.0)
MAX_AGE_DAYS = 5


def times_url(start: date) -> str:
    return f"{SERVER}/griddap/{DATASET}.csv0?" + quote(f"time[({start.isoformat()}T00:00:00Z):1:(last)]", safe=":(),")


def target_for(domain: satellite.Domain) -> satellite.Target:
    """MUR cell centres lie on whole hundredths of a degree."""
    lat_first = round(float(np.floor(domain.lat_n / STEP)) * STEP, 2)
    lon_first = round(float(np.ceil(domain.lon_w / STEP)) * STEP, 2)
    h = int(round((lat_first - domain.lat_s) / STEP)) + 1
    w = int(round((domain.lon_e - lon_first) / STEP)) + 1
    return satellite.Target(lat_first, lon_first, STEP, h, w)


def grid_url(t: str, target: satellite.Target) -> str:
    lat_s = target.lat_first - STEP * (target.height - 1)
    lon_e = target.lon_first + STEP * (target.width - 1)
    q = f"{VARIABLE}[({t})][({lat_s:.2f}):({target.lat_first:.2f})][({target.lon_first:.2f}):({lon_e:.2f})]"
    return f"{SERVER}/griddap/{DATASET}.nc?" + quote(q, safe=",():")


def parse(blob: bytes, target: satellite.Target) -> np.ndarray:
    from ..sources.charm import parse_netcdf

    arr, attrs, _ = parse_netcdf(blob)
    v = np.asarray(arr[VARIABLE], dtype=np.float64)
    lat, lon = np.asarray(arr["latitude"], float), np.asarray(arr["longitude"], float)
    v = v.reshape(len(lat), len(lon))
    fill = attrs.get(VARIABLE, {}).get("_FillValue")
    bad = ~np.isfinite(v) | ((v == float(fill)) if fill is not None else False) | (v < RANGE_C[0]) | (v > RANGE_C[1])
    v = np.where(bad, np.nan, v)
    r = np.round((target.lat_first - lat) / STEP).astype(int)
    c = np.round((lon - target.lon_first) / STEP).astype(int)
    out = np.full((target.height, target.width), np.nan)
    okr, okc = (r >= 0) & (r < target.height), (c >= 0) & (c < target.width)
    out[np.ix_(r[okr], c[okc])] = v[np.ix_(okr, okc)]
    return out


def fetch_latest(ctx, domain: satellite.Domain) -> tuple[date, satellite.Target, np.ndarray, list[str]]:
    u1 = times_url(ctx.now.date() - timedelta(days=10))
    lines = [ln.strip() for ln in ctx.fetcher(u1).body.decode().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("no MUR SST day in the last 10 days")
    t = lines[-1]
    day = datetime.fromisoformat(t.replace("Z", "+00:00")).date()
    if (ctx.now.date() - day).days > MAX_AGE_DAYS:
        raise ValueError(f"newest MUR SST day {day} is older than {MAX_AGE_DAYS} days")
    target = target_for(domain)
    u2 = grid_url(t, target)
    vals = parse(ctx.fetcher(u2).body, target)
    scope = satellite.marine_water(target) & satellite.coastal_mask(target)
    return day, target, np.where(scope, vals, np.nan), [u1, u2]
