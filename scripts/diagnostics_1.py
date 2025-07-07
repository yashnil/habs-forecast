#!/usr/bin/env python3
"""
Compute baselines and diagnostics for HAB ConvLSTM forecasts.

1. Persistence baseline (t+1 = t)
2. Climatology baseline (multi‐year day‐of‐year mean)
3. RMSE breakdown by season, latitude band, and chlorophyll quintile
4. Mean spatial residual map
5. Residual histogram & QQ‐plot
"""
from __future__ import annotations
import pathlib, yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as st

# ───────── paths & load cube ─────────────
root    = pathlib.Path(__file__).resolve().parents[1]
cfg     = yaml.safe_load(open(root/"config.yaml"))
cube_fp = pathlib.Path(cfg["data_root"]) / "HAB_cube_std0_2016_2021.nc"

ds      = xr.open_dataset(cube_fp)
chl     = ds["log_chl"]     # expect dims: (time, lat, lon)

# ─── ensure dims order is (time, lat, lon) ────────────────────
if tuple(chl.dims) != ("time","lat","lon"):
    chl = chl.transpose("time","lat","lon")

# ───────── split out test year ─────────────
test = chl.sel(time=chl.time.dt.year == 2021)
NT, NY, NX = len(test.time), len(test.lat), len(test.lon)

# ───────── 1. Persistence baseline ─────────
truth   = test.isel(time=slice(1, None)).values    # shape (NT-1, NY, NX)
persist = test.isel(time=slice(0, -1)).values       # shape (NT-1, NY, NX)
mask_p  = np.isfinite(truth) & np.isfinite(persist)
print(f"Persistence: valid grid-time pairs = {mask_p.sum():,} / {truth.size:,}")
rmse_p  = np.sqrt(((persist - truth)[mask_p]**2).mean())
print(f"Persistence RMSE (log-chl) = {rmse_p:.4f}\n")

# ───────── 2. Climatology baseline ─────────
# build day-of-year mean over all years < 2021
ds_all   = chl.sel(time=chl.time.dt.year < 2021)
doy_mean = ds_all.groupby("time.dayofyear").mean("time")   # dims (dayofyear, lat, lon)

# apply to each test date
doys      = test.time.dt.dayofyear.values                # length NT
clim_pred = np.stack([doy_mean.sel(dayofyear=int(d)).values
                      for d in doys], axis=0)             # shape (NT,NY,NX)
clim_truth= test.values                                   # shape (NT,NY,NX)
mask_c    = np.isfinite(clim_pred) & np.isfinite(clim_truth)
print(f"Climatology: valid grid-time pairs = {mask_c.sum():,} / {clim_truth.size:,}")
rmse_c    = np.sqrt(((clim_pred - clim_truth)[mask_c]**2).mean())
print(f"Climatology   RMSE (log-chl) = {rmse_c:.4f}\n")

# ───────── load your model’s test preds ─────────
# e.g. mdl_pred = np.load("Models/convLSTM_test_preds.npy")
# for now we reuse persistence so downstream code runs
mdl_pred = persist
mask_m   = mask_p
resid    = mdl_pred - truth     # shape (NT-1,NY,NX)

# ───────── 3. Error breakdown ────────────────────────────────
print("RMSE by season:")
seasons = dict(winter=[12,1,2], spring=[3,4,5],
               summer=[6,7,8],   fall=[9,10,11])
months = test.time.dt.month.values[1:]  # align with resid/time-slice
for name, mlist in seasons.items():
    sel = np.isin(months, mlist)
    nsel = sel.sum()
    if nsel == 0:
        print(f"  {name:6s}: N/A (no test days)")
    else:
        errs = resid[sel][mask_m[sel]]
        rm   = np.sqrt((errs**2).mean())
        print(f"  {name:6s}: {rm:.4f} (n={errs.size})")
print()

print("RMSE by latitude band:")
lats = ds["lat"].values
bands = [32,36,40,44,48,50]
for lo, hi in zip(bands[:-1], bands[1:]):
    lat_sel = (lats >= lo) & (lats < hi)   # length NY
    count   = lat_sel.sum() * resid.shape[0]
    if count == 0:
        print(f"  {lo:2.0f}–{hi:2.0f}°N: N/A")
    else:
        errs = resid[:, lat_sel, :][mask_m[:, lat_sel, :]]
        rm   = np.sqrt((errs**2).mean())
        print(f"  {lo:2.0f}–{hi:2.0f}°N: {rm:.4f} (n={errs.size})")
print()

flat_truth = truth[mask_m]
flat_res   = resid[mask_m]
df = pd.DataFrame({"truth": flat_truth, "resid": flat_res})
df["q"] = pd.qcut(df.truth, 5, labels=False, duplicates="drop")
print("RMSE by log-chl quintile:")
for q in sorted(df.q.unique()):
    sub = df.loc[df.q == q, "resid"].values
    rm  = np.sqrt((sub**2).mean()) if len(sub) else np.nan
    print(f"  Q{int(q)+1}: {rm:.4f} (n={len(sub)})")
print()

# ───────── 4. Mean spatial residual map ───────────────────────
mean_err = np.nanmean(resid, axis=0)     # dims (lat, lon)
plt.figure(figsize=(6,4))
plt.imshow(mean_err, origin="lower", cmap="RdBu_r",
           extent=[ds.lon.min(), ds.lon.max(), ds.lat.min(), ds.lat.max()])
plt.colorbar(label="mean residual")
plt.title("Mean spatial residual (test year)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.savefig("mean_residual_map.png", dpi=200, bbox_inches="tight")
plt.close()

# ───────── 5. Residual distribution ───────────────────────────
plt.figure()
plt.hist(flat_res, bins=100, density=True)
plt.title("Residual histogram")
plt.xlabel("prediction – truth")
plt.savefig("residual_hist.png", dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
st.probplot(flat_res, dist="norm", plot=plt)
plt.title("QQ-plot of residuals")
plt.savefig("residual_qq.png", dpi=200, bbox_inches="tight")
plt.close()

print("✅ Diagnostics complete; check the printed RMSEs and PNGs.")
