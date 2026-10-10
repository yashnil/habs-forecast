"""WCOFS vs HF radar: a small, honest skill check (P2 phase D). Research only; nothing published.

For each WCOFS run (t03z) and each 3-hourly forecast step that HF radar has observed:
- sample WCOFS surface currents (0.04 deg regular grid, nearest cell) at every valid HF-radar
  2 km cell of a region, at the same hour;
- score vector RMSE, complex correlation (magnitude and angle), mean speeds;
- score two baselines at the same cells and hour:
    persistence: the HF-radar field observed at the run's issue hour (03 UTC) carried forward
                 (optimistic: real-time radar arrives hours late); for daily means, the radar's
                 mean over the 24 h before issue;
    zero:        predict no current;
- skill = 1 - RMSE_model / RMSE_baseline (positive = better than the baseline).
Also the lead-day-1 24-hour means (8 WCOFS steps vs 24 radar hours), which suppress the tide
and the daily sea breeze that HF radar resolves but a 3-hourly comparison aliases.

WCOFS assimilates HF radar, so its early leads are not independent of the radar.

    uv run --with xarray --with h5netcdf --with h5py --with fsspec --with aiohttp \
        python scripts/wcofs_eval.py <out.json> 20261006 20261007
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coastwatch_pipeline.http import fetch  # noqa: E402
from coastwatch_pipeline.sources.charm import parse_netcdf  # noqa: E402
from wcofs_poc import read_step  # noqa: E402

HFR = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ucsdHfrW2"
REGIONS = {  # (south, north, west, east)
    "monterey_bay": (36.45, 37.15, -122.45, -121.75),
    "gulf_of_farallones": (37.30, 38.10, -123.40, -122.45),
    "central_coast_shelf": (34.40, 35.60, -121.40, -120.55),
    "socal_bight": (33.30, 34.45, -119.90, -117.80),
}


def hfr_region(region, t0: str, t1: str):
    s, n, w, e = REGIONS[region]
    sel = f"[({t0}):1:({t1})][({s}):({n})][({w}):({e})]"
    url = f"{HFR}.nc?" + quote(f"water_u{sel},water_v{sel}", safe=",():")
    arrays, vattrs, _ = parse_netcdf(fetch(url, retries=4, timeout=300).body)
    lat, lon = np.asarray(arrays["latitude"], float), np.asarray(arrays["longitude"], float)
    tt = [datetime.fromtimestamp(float(x), tz=timezone.utc) for x in np.atleast_1d(arrays["time"])]
    out = {}
    for k in ("water_u", "water_v"):
        a = np.asarray(arrays[k], float).reshape(len(tt), lat.size, lon.size)
        a[np.abs(a) > 10] = np.nan
        out[k] = a
    return tt, lat, lon, out["water_u"], out["water_v"]


def sample(wlat, wlon, wu, wv, lat, lon):
    ri = np.clip(np.round((lat - wlat[0]) / (wlat[1] - wlat[0])).astype(int), 0, wlat.size - 1)
    ci = np.clip(np.round((lon - wlon[0]) / (wlon[1] - wlon[0])).astype(int), 0, wlon.size - 1)
    return wu[np.ix_(ri, ci)], wv[np.ix_(ri, ci)]


def scores(ou, ov, mu, mv, pu=None, pv=None) -> dict | None:
    ok = np.isfinite(ou) & np.isfinite(ov) & np.isfinite(mu) & np.isfinite(mv)
    if pu is not None:
        ok &= np.isfinite(pu) & np.isfinite(pv)
    if ok.sum() < 30:
        return None
    a, b = ou[ok] + 1j * ov[ok], mu[ok] + 1j * mv[ok]
    a0, b0 = a - a.mean(), b - b.mean()
    cc = np.vdot(a0, b0) / np.sqrt(np.vdot(a0, a0).real * np.vdot(b0, b0).real)
    rmse = float(np.sqrt(np.mean(np.abs(a - b) ** 2)))
    r = {"n": int(ok.sum()), "obs_speed": round(float(np.abs(a).mean()), 3), "model_speed": round(float(np.abs(b).mean()), 3), "rmse": round(rmse, 3),
         "cc": round(float(abs(cc)), 3), "cc_angle_deg": round(float(np.degrees(np.angle(cc))), 1),
         "rmse_zero": round(float(np.sqrt(np.mean(np.abs(a) ** 2))), 3)}
    r["skill_vs_zero"] = round(1 - rmse / r["rmse_zero"], 3) if r["rmse_zero"] else None
    if pu is not None:
        p = pu[ok] + 1j * pv[ok]
        rp = float(np.sqrt(np.mean(np.abs(a - p) ** 2)))
        r["rmse_persistence"] = round(rp, 3)
        r["skill_vs_persistence"] = round(1 - rmse / rp, 3) if rp else None
    return r


def main(out: Path, runs: list[str]) -> None:
    report: dict = {"method": __doc__.split("\n\n")[1], "regions": REGIONS, "runs": []}
    t_start = time.time()
    for run in runs:
        r0 = datetime.strptime(run, "%Y%m%d").replace(hour=3, tzinfo=timezone.utc)
        last = r0 + timedelta(hours=72)
        hfr = {reg: hfr_region(reg, (r0 - timedelta(hours=24)).strftime("%Y-%m-%dT%H:00:00Z"), min(last, datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%dT%H:00:00Z")) for reg in REGIONS}
        steps = []
        wfields = {}
        for h in range(3, 73, 3):
            T = r0 + timedelta(hours=h)
            if not any(T in hfr[reg][0] for reg in REGIONS):
                continue
            try:
                wlat, wlon, wu, wv, info = read_step(run, f"f{h:03d}")
            except Exception as e:  # missing step
                steps.append({"lead_h": h, "error": str(e)[:200]})
                continue
            wfields[h] = (wlat, wlon, wu, wv)
            row = {"lead_h": h, "valid": T.strftime("%Y-%m-%dT%H:%MZ"), "read_s": info["seconds"], "regions": {}}
            for reg, (tt, lat, lon, U, V) in hfr.items():
                if T not in tt or r0 not in tt:
                    continue
                k, k0 = tt.index(T), tt.index(r0)
                mu, mv = sample(wlat, wlon, wu, wv, lat, lon)
                sc = scores(U[k], V[k], mu, mv, U[k0], V[k0])
                if sc:
                    row["regions"][reg] = sc
            steps.append(row)
            print(run, h, {k: (v["skill_vs_persistence"], v["cc"]) for k, v in row["regions"].items()}, flush=True)
        # lead-day-1 daily mean: valid hours (r0+24h, r0+48h], 8 WCOFS steps vs 24 radar hours
        daily = {}
        hs = [h for h in range(27, 49, 3) if h in wfields]
        if len(hs) >= 7:
            for reg, (tt, lat, lon, U, V) in hfr.items():
                idx = [i for i, t in enumerate(tt) if r0 + timedelta(hours=24) < t <= r0 + timedelta(hours=48)]
                if len(idx) < 18:
                    continue
                with np.errstate(invalid="ignore"):
                    n = np.isfinite(U[idx]).sum(axis=0)
                    ou = np.where(n >= 18, np.nanmean(U[idx], axis=0), np.nan)
                    ov = np.where(n >= 18, np.nanmean(V[idx], axis=0), np.nan)
                ms = [sample(*wfields[h], lat, lon) for h in hs]
                mu = np.mean([m[0] for m in ms], axis=0)
                mv = np.mean([m[1] for m in ms], axis=0)
                # persistence of the daily mean: the radar's mean over the 24 h before issue
                # (known at issue time, apart from radar latency)
                idp = [i for i, t in enumerate(tt) if r0 - timedelta(hours=24) <= t < r0]
                pu = pv = None
                if len(idp) >= 18:
                    npv = np.isfinite(U[idp]).sum(axis=0)
                    pu = np.where(npv >= 18, np.nanmean(U[idp], axis=0), np.nan)
                    pv = np.where(npv >= 18, np.nanmean(V[idp], axis=0), np.nan)
                daily[reg] = scores(ou, ov, mu, mv, pu, pv)
        report["runs"].append({"run": f"{run} t03z", "steps": steps, "lead_day1_daily_mean": daily})
    report["seconds"] = round(time.time() - t_start)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", out, report["seconds"], "s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
