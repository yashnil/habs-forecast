#!/usr/bin/env python3
"""Moving-block bootstrap (over 8-day time steps) of test-period RMSE differences.

Uses the deltas saved by rescore.py. Blocks of 6 steps (48 days) keep the
seasonal autocorrelation inside a block. Reports mean and 95% interval of
RMSE(a) - RMSE(b) for the pairs that matter.
"""
import argparse, importlib.util, json, pathlib
import numpy as np, pandas as pd, xarray as xr

ap = argparse.ArgumentParser()
ap.add_argument("--freeze", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=2000); ap.add_argument("--block", type=int, default=6)
a = ap.parse_args(); out = pathlib.Path(a.out)
REPO = pathlib.Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("d", REPO / "pinn/diagnostics.py"); D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
ds = xr.open_dataset(a.freeze).load(); ok = D.build_pixel_ok_if_needed(ds)
la_any = np.where(ok.any("lon"))[0]; lo_any = np.where(ok.any("lat"))[0]
la, lo = slice(la_any.min(), la_any.max() + 1), slice(lo_any.min(), lo_any.max() + 1)
okv = ok.isel(lat=la, lon=lo).values; logc = ds.log_chl.isel(lat=la, lon=lo).values.astype(np.float32)
core = np.arange(5, ds.sizes["time"] - 1); t = ds.time.values[core + 1]
last, truth = logc[core], logc[core + 1]; mask = okv[None] & np.isfinite(truth)
tr = ds.time.values < np.datetime64("2016-01-01"); te = t > np.datetime64("2018-12-31")
mu, sd = float(np.nanmean(logc[tr])), float(np.nanstd(logc[tr]))   # == norm_stats on train
months = pd.DatetimeIndex(ds.time.values).month.values
clim = np.stack([np.nanmean(logc[np.where(tr & (months == m))[0]], 0) for m in range(1, 13)])
P = json.load(open(out / "baseline_params.json")); phi = P["ar1_clim_anom_phi"]
c_last, c_tgt = clim[months[core] - 1], clim[months[core + 1] - 1]
z_last = np.nan_to_num((last - mu) / sd, nan=0.0)
F = {"persistence": last, "ar1_clim_anom": c_tgt + phi * (last - c_last)}
for n in ("convlstm", "pinn_opt"):
    F[f"{n}:as_trained"] = z_last + np.load(out / f"delta_{n}.npy")
idx = np.where(te)[0]
se = {k: np.array([((f[i] - truth[i])[mask[i]] ** 2).sum() for i in idx]) for k, f in F.items()}
cnt = np.array([mask[i].sum() for i in idx])
rng = np.random.default_rng(0); T = len(idx); B = a.block
starts = np.arange(T - B + 1)
def rm(k, sel): return np.sqrt(se[k][sel].sum() / cnt[sel].sum())
pairs = [("pinn_opt:as_trained", "convlstm:as_trained"), ("convlstm:as_trained", "ar1_clim_anom"),
         ("pinn_opt:as_trained", "ar1_clim_anom"), ("ar1_clim_anom", "persistence"), ("convlstm:as_trained", "persistence")]
res = {}
for x, y in pairs:
    d = []
    for _ in range(a.n):
        sel = np.concatenate([np.arange(s, s + B) for s in rng.choice(starts, int(np.ceil(T / B)))])[:T]
        d.append(rm(x, sel) - rm(y, sel))
    full = np.arange(T)
    res[f"{x} - {y}"] = {"point": float(rm(x, full) - rm(y, full)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
json.dump({"test_steps": int(T), "block_steps": B, "n_boot": a.n, "differences": res}, open(out / "bootstrap.json", "w"), indent=1)
for k, v in res.items(): print(f"{k:50s} {v['point']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")
