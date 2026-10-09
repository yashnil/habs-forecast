"""Ocean-currents proof of concept (P1 phase D). Not wired into the pipeline.

1. Reads WCOFS surface currents for California from NOAA's public S3 bucket, surface level
   only, with HTTP byte ranges (no 582 MB downloads).
2. Encodes them in the VectorField contract (models.py): u and v as quantized value grids.
3. Compares a 24-hour WCOFS forecast (the previous day's run, +24 h) with observed HF-radar
   surface currents (2 km, ucsdHfrW2) at its valid hour in Monterey Bay: complex
   correlation, RMSE, direction difference. WCOFS assimilates HF radar, so only a
   forecast (not the nowcast) is a meaningful check, and even it shares the radar's
   history through the initial state.
4. Writes thinned arrows GeoJSON.

    uv run --with xarray --with h5netcdf --with h5py --with fsspec --with aiohttp \
        python scripts/wcofs_poc.py <out-dir> [YYYYMMDD]

Nothing here is published. Results inform the currents phase (docs/coastwatch/p1/).
"""

from __future__ import annotations

import io
import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.http import fetch  # noqa: E402
from coastwatch_pipeline.models import ValueGrid, VectorField  # noqa: E402
from coastwatch_pipeline.process import grid as codec  # noqa: E402
from coastwatch_pipeline.sources.charm import parse_netcdf  # noqa: E402

BUCKET = "https://noaa-nos-ofs-pds.s3.amazonaws.com/wcofs/netcdf"
LAT, LON = (32.0, 42.6), (-126.5, -117.0)
MONTEREY = (36.45, 37.15, -122.45, -121.75)
HFR = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ucsdHfrW2"
V_RANGE = (-2.5, 2.5)  # m s-1, fixed


def read_step(day: str, tag: str):
    import fsspec
    import xarray as xr

    url = f"{BUCKET}/{day[:4]}/{day[4:6]}/{day[6:]}/wcofs.t03z.{day}.regulargrid.{tag}.nc"
    fs = fsspec.filesystem("https", block_size=256 * 1024)
    t0 = time.time()
    with fs.open(url, cache_type="readahead") as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        lat, lon = ds.Latitude.values, ds.Longitude.values
        rows = np.where((lat[:, 0] >= LAT[0]) & (lat[:, 0] <= LAT[1]))[0]
        cols = np.where((lon[0] >= LON[0]) & (lon[0] <= LON[1]))[0]
        sl = dict(ny=slice(rows[0], rows[-1] + 1), nx=slice(cols[0], cols[-1] + 1))
        sub = ds[["u_eastward", "v_northward"]].isel(Depth=0, **sl).load()
        valid = str(ds.time.values[0])[:19] + "Z"
        info = {"url": url, "valid_time": valid, "seconds": round(time.time() - t0, 1), "bytes_requested": getattr(f.cache, "total_requested_bytes", None)}
    return lat[sl["ny"], 0], lon[0, sl["nx"]], sub.u_eastward.values[0], sub.v_northward.values[0], info


def grid_meta(lat, lon, url) -> ValueGrid:
    scale = (V_RANGE[1] - V_RANGE[0]) / codec.MAX_CODE
    return ValueGrid(
        url=url, width=lon.size, height=lat.size, lat_first=float(lat[0]), lat_step=float(lat[1] - lat[0]),
        lon_first=float(lon[0]), lon_step=float(lon[1] - lon[0]), scale_factor=scale, add_offset=V_RANGE[0], max_quantization_error=scale / 2,
    )


def hfr(valid: str):
    s, n, w, e = MONTEREY
    q = f"water_u[({valid})][({s}):({n})][({w}):({e})],water_v[({valid})][({s}):({n})][({w}):({e})]"
    arrays, _, _ = parse_netcdf(fetch(f"{HFR}.nc?" + quote(q, safe=",():"), retries=4).body)
    u = np.asarray(arrays["water_u"], float).reshape(arrays["latitude"].size, arrays["longitude"].size)
    v = np.asarray(arrays["water_v"], float).reshape(arrays["latitude"].size, arrays["longitude"].size)
    for a in (u, v):
        a[np.abs(a) > 10] = np.nan
    return arrays["latitude"].astype(float), arrays["longitude"].astype(float), u, v


def compare(wlat, wlon, wu, wv, hlat, hlon, hu, hv) -> dict:
    """Sample WCOFS (4 km) at each valid HF-radar cell (nearest WCOFS cell)."""
    ri = np.clip(np.round((hlat - wlat[0]) / (wlat[1] - wlat[0])).astype(int), 0, wlat.size - 1)
    ci = np.clip(np.round((hlon - wlon[0]) / (wlon[1] - wlon[0])).astype(int), 0, wlon.size - 1)
    mu, mv = wu[np.ix_(ri, ci)], wv[np.ix_(ri, ci)]
    ok = np.isfinite(hu) & np.isfinite(hv) & np.isfinite(mu) & np.isfinite(mv)
    if ok.sum() < 10:
        return {"n": int(ok.sum()), "note": "too few overlapping cells"}
    a, b = hu[ok] + 1j * hv[ok], mu[ok] + 1j * mv[ok]
    a0, b0 = a - a.mean(), b - b.mean()
    complex_corr = (np.vdot(a0, b0) / np.sqrt(np.vdot(a0, a0).real * np.vdot(b0, b0).real))
    return {
        "n": int(ok.sum()),
        "hfr_mean_speed_m_s": round(float(np.abs(a).mean()), 3),
        "wcofs_mean_speed_m_s": round(float(np.abs(b).mean()), 3),
        "rmse_vector_m_s": round(float(np.sqrt(np.mean(np.abs(a - b) ** 2))), 3),
        "complex_correlation_magnitude": round(float(abs(complex_corr)), 3),
        "complex_correlation_angle_deg": round(float(np.degrees(np.angle(complex_corr))), 1),
        "mean_direction_difference_deg": round(float(np.degrees(np.angle(np.mean(b) / np.mean(a)))), 1),
    }


def arrows(lat, lon, u, v, stride=6, scale_deg=0.16):
    feats = []
    for i in range(0, lat.size, stride):
        for j in range(0, lon.size, stride):
            uu, vv = u[i, j], v[i, j]
            if not (np.isfinite(uu) and np.isfinite(vv)):
                continue
            sp = float(math.hypot(uu, vv))
            k = scale_deg * min(1.0, sp / 0.3) / max(sp, 1e-6)
            tip = [float(lon[j] + uu * k / math.cos(math.radians(lat[i]))), float(lat[i] + vv * k)]
            feats.append({"type": "Feature", "properties": {"speed_m_s": round(sp, 3)}, "geometry": {"type": "LineString", "coordinates": [[float(lon[j]), float(lat[i])], tip]}})
    return {"type": "FeatureCollection", "features": feats}


def main(out: Path, day: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"run": day, "steps": []}
    for tag in ("f003", "f024", "f048", "f072"):
        lat, lon, u, v, info = read_step(day, tag)
        blob_u, *_ = codec.encode(u, *V_RANGE)
        blob_v, *_ = codec.encode(v, *V_RANGE)
        (out / f"u_{tag}.u16.gz").write_bytes(blob_u)
        (out / f"v_{tag}.u16.gz").write_bytes(blob_v)
        vf = VectorField(
            u_grid=grid_meta(lat, lon, f"u_{tag}.u16.gz"), v_grid=grid_meta(lat, lon, f"v_{tag}.u16.gz"), depth_m=0.0,
            speed_max=round(float(np.nanmax(np.hypot(u, v))), 3), arrows_url=f"arrows_{tag}.geojson",
        )
        (out / f"arrows_{tag}.geojson").write_text(json.dumps(arrows(lat, lon, u, v)))
        (out / f"vectorfield_{tag}.json").write_text(vf.model_dump_json(indent=1))
        report["steps"].append({**info, "grid": [int(lat.size), int(lon.size)], "u_bytes": len(blob_u), "v_bytes": len(blob_v), "speed_max_m_s": vf.speed_max})
    try:
        # newest HF-radar hour that falls on a WCOFS 3-hourly forecast step of the run one
        # day earlier, at least 12 h into that forecast
        times = fetch(f"{HFR}.csv0?time%5Blast-48:1:last%5D").body.decode().split()
        for t in reversed(times):
            T = datetime.fromisoformat(t.replace("Z", "+00:00"))
            run = (T - timedelta(hours=12)).replace(hour=3, minute=0, second=0)
            if run > T - timedelta(hours=12):
                run -= timedelta(days=1)
            h = int((T - run).total_seconds() // 3600)
            if h % 3 == 0 and 12 <= h <= 72:
                break
        lat, lon, u, v, info = read_step(run.strftime("%Y%m%d"), f"f{h:03d}")
        assert info["valid_time"] == T.strftime("%Y-%m-%dT%H:%M:%SZ"), (info["valid_time"], T)
        hlat, hlon, hu, hv = hfr(info["valid_time"])
        report["hf_radar_comparison"] = {
            "forecast": f"WCOFS run {run:%Y-%m-%d} t03z, +{h} h", "valid_time": info["valid_time"], "region": "Monterey Bay",
            "hfr_dataset": "ucsdHfrW2", **compare(lat, lon, u, v, hlat, hlon, hu, hv),
        }
    except Exception as e:
        report["hf_radar_comparison"] = {"error": str(e)[:300]}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    d = sys.argv[2] if len(sys.argv) > 2 else (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y%m%d")
    main(Path(sys.argv[1]), d)
