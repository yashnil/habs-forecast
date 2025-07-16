#!/usr/bin/env python3
"""
Gap-fill MODIS optical variables within the CA 16-km coastal stripe
using a climatology + clipping approach that preserves observed values
and limits synthetic fills to physically plausible ranges.

Vars filled: chlor_a, Kd_490, nflh
Derived   : log_chl  (natural log)

Input : HAB_master_8day_4km_coastal_CA.nc
Output: HAB_master_8day_4km_coastal_CA_climclip.nc
"""

from __future__ import annotations
import pathlib
import numpy as np
import xarray as xr
from scipy import ndimage

# ---------------------------------------------------------------------
# config / paths
# ---------------------------------------------------------------------
ROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
FIN  = ROOT / "Data" / "Finalized"

IN_FP  = FIN / "HAB_master_8day_4km_coastal_CA.nc"
OUT_FP = FIN / "HAB_master_8day_4km_coastal_CA_climclip.nc"

# temporal interpolation gap limit
MAX_GAP = "24D"         # linear bridge gaps <= 24 days
# spatial smoothing kernel (pixels)
SMOOTH_SIZE = 3         # 3x3 nanmedian on climatology
# percentile clip limits for synthetic fills (relative to observed stripe distrib)
CLIP_LO, CLIP_HI = 1.0, 99.0

VARS = ["chlor_a", "Kd_490", "nflh"]   # log_chl derived later

# ---------------------------------------------------------------------
# open dataset
# ---------------------------------------------------------------------
print(f"🔹 opening CA coastal cube …")
ds0 = xr.open_dataset(IN_FP).transpose("time", "lat", "lon")

nt, ny, nx = ds0.dims["time"], ds0.dims["lat"], ds0.dims["lon"]

# build coastal stripe mask from union of the three optics vars (time-collapsed)
stripe2d = (
    ds0["chlor_a"].notnull().any("time")
    | ds0["Kd_490"].notnull().any("time")
    | ds0["nflh"].notnull().any("time")
).astype(bool)
stripe2d.name = "coastal_stripe"
stripe3d = stripe2d.expand_dims(time=ds0.time).transpose("time", "lat", "lon")

n_stripe = int(stripe2d.sum())
print(f"   stripe pixels: {n_stripe:,} ({n_stripe / (ny*nx):.2%} of regional grid)\n")

# work copy we will modify
ds = ds0.copy(deep=True)

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def eightday_block(time_coord: xr.DataArray) -> xr.DataArray:
    """Return 8-day block index (0..44) for each timestamp."""
    blk = ((time_coord.dt.dayofyear - 1) // 8).astype(int)
    blk.name = "blk"
    return blk

def nanmedian2d(arr2d: np.ndarray, size: int = 3) -> np.ndarray:
    """Return 2-D nanmedian filtered array of same shape."""
    def _nm(v):
        return np.nanmedian(v)
    return ndimage.generic_filter(arr2d, _nm, size=size, mode="nearest")

def smooth_climatology(clim: xr.DataArray, size: int = 3) -> xr.DataArray:
    """
    Apply 2-D nanmedian smoothing to each block slice in a (blk,lat,lon) field.
    """
    out = clim.copy()
    for b in clim["blk"].values:
        arr = clim.sel(blk=b).values
        arr_sm = nanmedian2d(arr, size=size)
        out.loc[dict(blk=b)] = arr_sm
    return out

def build_block_clim(da: xr.DataArray, stripe3d: xr.DataArray) -> xr.DataArray:
    """
    Multi-year seasonal climatology median for each 8-day block.
    Returns DataArray(blk, lat, lon).
    """
    blk = eightday_block(da.time)
    da_s = da.where(stripe3d)  # restrict to stripe
    clim = da_s.groupby(blk).median("time", skipna=True)
    clim_sm = smooth_climatology(clim, size=SMOOTH_SIZE)
    return clim_sm

def fill_var(name: str) -> None:
    """
    Fill ds[name] (in place) using staged approach:
        1) temporal interp short gaps + f/b fill
        2) block climatology median (smoothed) fill
        3) clip synthetic to observed percentile range
    Observed values preserved.
    """
    print(f"▶ filling {name} …")
    da_orig = ds0[name]  # original
    da_mod  = ds[name]   # working

    # Observation flag (original non-NaN inside stripe)
    obs_flag = (da_orig.notnull() & stripe3d).astype(np.int8)
    ds[f"{name}_obs"] = obs_flag

    # Stage A: temporal interpolation within stripe
    da_t = (
        da_orig.where(stripe3d)
               .interpolate_na("time", method="linear", max_gap=MAX_GAP)
               .ffill("time")
               .bfill("time")
    )

    # Stage B: block climatology
    clim_blk = build_block_clim(da_orig, stripe3d)          # (blk,lat,lon)
    blk = eightday_block(ds0.time)
    clim_bcast = clim_blk.sel(blk=blk).drop_vars("blk")     # (time,lat,lon)

    da_tc = da_t.fillna(clim_bcast)

    # Stage C: clip synthetic values only
    # compute observed stripe percentiles
    obs_vals = da_orig.where(stripe3d)
    lo = float(obs_vals.quantile(CLIP_LO/100, skipna=True))
    hi = float(obs_vals.quantile(CLIP_HI/100, skipna=True))
    # where original was NaN but stripe True -> synthetic candidate
    synth_mask = stripe3d & da_orig.isnull()
    synth_vals = da_tc.where(synth_mask)
    synth_vals = synth_vals.clip(min=lo, max=hi)
    # merge back: keep original obs where present; else clipped synthetic
    da_new = xr.where(synth_mask, synth_vals, da_orig)

    # ensure off-stripe stays NaN
    da_new = da_new.where(stripe3d)

    ds[name] = da_new.astype("float32")

# ---------------------------------------------------------------------
# run fill for each variable
# ---------------------------------------------------------------------
for v in VARS:
    fill_var(v)
    print()

# ---------------------------------------------------------------------
# recompute log_chl from filled chlor_a
# ---------------------------------------------------------------------
print("▶ recomputing log_chl …")
chl     = ds["chlor_a"]
chl_obs = ds["chlor_a_obs"].astype(bool)

# floor based on positive observed chlor_a (stripe only)
chl_pos = ds0["chlor_a"].where(stripe3d & (ds0["chlor_a"] > 0))
chl_min = float(chl_pos.min(skipna=True))
if not np.isfinite(chl_min) or chl_min <= 0.0:
    chl_min = 0.01459
floor = max(chl_min, 1e-5)

log_chl = np.log(np.maximum(chl, floor))
log_chl.attrs.update(
    long_name="natural-log chlorophyll-a",
    note=f"log(max(chlor_a, {floor:.5g}))",
    units="ln(mg m-3)",
)
ds["log_chl"] = log_chl.astype("float32")
ds["log_chl_obs"] = chl_obs.astype(np.int8)

# ---------------------------------------------------------------------
# stripe NaN summary
# ---------------------------------------------------------------------
print("\nNaNs inside CA coastal stripe (post-fill; expect 0):")
stripe3d_bool = stripe3d
for v in VARS + ["log_chl"]:
    n_nan = int(ds[v].where(stripe3d_bool).isnull().sum())
    print(f"{v:8s} | {n_nan:,}")
    if n_nan != 0:
        print(f"  ⚠ Warning: {v} still has NaNs in stripe.")

# range sanity vs observed
print("\nRange comparison (observed stripe vs final):")
for v in VARS + ["log_chl"]:
    obs_vals = ds0[v].where(stripe3d_bool)
    fin_vals = ds[v].where(stripe3d_bool)
    print(f"{v:8s} obs[min,max]={float(obs_vals.min(skipna=True)):.3g},"
          f"{float(obs_vals.max(skipna=True)):.3g}  "
          f"filled[min,max]={float(fin_vals.min(skipna=True)):.3g},"
          f"{float(fin_vals.max(skipna=True)):.3g}")

# ---------------------------------------------------------------------
# write out
# ---------------------------------------------------------------------
enc = {k: {"zlib": True, "complevel": 4} for k in ds.data_vars}
ds.to_netcdf(OUT_FP, mode="w", engine="netcdf4", encoding=enc)
print(f"\n✅ wrote gap-filled CA coastal cube → {OUT_FP.name}")


'''

BEFORE
Stripe pixels (lat,lon): 1,626
Stripe time-cells     : 1,383,726 (=851 × 1626)

chlor_a: 137,416 NaNs in stripe ( 9.93%)
Kd_490 : 137,405 NaNs in stripe ( 9.93%)
nflh   : 245,328 NaNs in stripe (17.73%)
log_chl: 137,416 NaNs in stripe ( 9.93%)

AFTER
NaNs inside CA coastal stripe (should be 0):
var | NaNs_in_stripe | stripe_cells | pct
chlor_a | 0 | 1,383,726 |  0.00%
Kd_490  | 0 | 1,383,726 |  0.00%
nflh    | 0 | 1,383,726 |  0.00%
log_chl | 0 | 1,383,726 |  0.00%
'''