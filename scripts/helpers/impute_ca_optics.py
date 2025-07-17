#!/usr/bin/env python3
# scripts/helpers/impute_ca_optics_localseason.py
"""
Local log-space seasonal gap-fill for CA coastal stripe (chlor_a, Kd_490, nflh).

Approach:
  • Stripe = union of non-NaN across chlor_a, Kd_490, nflh (time-collapsed).
  • Pixel median (multi-year) sets local baseline.
  • Regional 8-day seasonal anomaly (median-removed) adds realistic seasonality.
  • Missing pixels' medians NN-filled from nearest observed pixel.
  • Reconstruction:
        logvar_filled = pix_med_log + seasonal_anom_reg
        var_filled     = exp(logvar_filled)
  • Soft guard vs. unrealistically large multipliers.
  • Observations preserved; fills only where NaN in stripe.
  • nflh handled in linear space (can be <0).

Outputs:
  Data/Finalized/HAB_master_8day_4km_coastal_CA_localseason.nc

Adds *_obs flags (1=observed in input; 0=gap-filled in output).
"""

from __future__ import annotations
import pathlib
import numpy as np
import xarray as xr
from scipy import ndimage

# ------------------------------------------------------------------ paths ---
ROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
FIN  = ROOT / "Data" / "Finalized"

IN_FP  = FIN / "HAB_master_8day_4km_coastal_CA.nc"     # unfilled CA coastal cube
OUT_FP = FIN / "HAB_master_8day_4km_coastal_CA_localseason.nc"

# tunables
SOFT_FACTOR_CHL   = 5.0   # max allowed (filled/median) ratio
SOFT_FACTOR_KD    = 3.0
SOFT_FACTOR_NFLH  = 5.0   # multiplier on pixel IQR
MIN_POS_FLOOR     = 1e-5  # lower bound before log

# ---------------------------------------------------------------- open -----
print("🔹 opening CA coastal cube …")
ds0 = xr.open_dataset(IN_FP)
ds0 = ds0.transpose("time", "lat", "lon")  # assure order
ds  = ds0.copy(deep=True)

time = ds.time

# ---------------------------------------------------------------- stripe mask (lat,lon) ----
stripe2d = (
    ds0["chlor_a"].notnull().any("time")
  | ds0["Kd_490"].notnull().any("time")
  | ds0["nflh"].notnull().any("time")
).astype(bool)
stripe2d.name = "coastal_stripe"
stripe3d = stripe2d.expand_dims(time=time)

n_stripe = int(stripe2d.sum())
print(f"   stripe pixels: {n_stripe:,}")

# ---------------------------------------------------------------- helpers ---
def edt_nn_fill2d(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Fill NaNs in arr with nearest non-NaN (restricted to valid_mask)."""
    work = arr.copy()
    work[~valid_mask] = np.nan
    need = np.isnan(work) & valid_mask
    if not need.any():
        return arr
    features = ~np.isnan(work)
    _, (iy, ix) = ndimage.distance_transform_edt(~features, return_indices=True)
    out = arr.copy()
    out[need] = work[iy[need], ix[need]]
    return out

def nn_fill_median2d(pix_med: xr.DataArray, valid2d: xr.DataArray) -> xr.DataArray:
    """Nearest-neighbour fill of per-pixel medians where missing."""
    filled = edt_nn_fill2d(pix_med.values, valid2d.values)
    return xr.DataArray(filled, coords=pix_med.coords, dims=pix_med.dims)

def block_index(times: xr.DataArray) -> xr.DataArray:
    """0..44 8-day seasonal index."""
    blk = ((times.dt.dayofyear - 1) // 8).astype(int)
    blk.name = "blk"
    return blk

# ---------------------------------------------------------------- seasonal anomaly (region-wide) ----
def regional_seasonal_anom(da: xr.DataArray) -> xr.DataArray:
    """
    Return regional 8-day seasonal anomaly series (time,) in same units as da.

    Steps: 
      1. Compute per-block regional *median* over stripe pixels (observed only).
      2. Interpolate to full time via block index.
      3. Convert to anomaly about overall regional median (over observed stripe).
    """
    blk = block_index(da.time)
    # collapse over space using stripe mask
    da_stripe = da.where(stripe3d)
    reg_med_all = da_stripe.median(dim=("time","lat","lon"), skipna=True)
    reg_block   = da_stripe.groupby(blk).median(("time","lat","lon"), skipna=True)
    # broadcast block medians to time
    reg_block_t = reg_block.sel(blk=blk).drop_vars("blk")
    # anomaly
    anom = reg_block_t - reg_med_all
    return anom

# ---------------------------------------------------------------- fill positive var in log space ----
def fill_pos_var(name: str, soft_factor: float):
    da_raw = ds0[name]
    obs_flag = (da_raw.notnull() & stripe3d).astype(np.int8)
    ds[f"{name}_obs"] = obs_flag

    # restrict to stripe
    da_s = da_raw.where(stripe3d)

    # floor then log
    da_floor = da_s.where(da_s > 0, MIN_POS_FLOOR)
    log_da = np.log(da_floor)

    # pixel median (log)
    pix_med_log = log_da.median("time", skipna=True)
    valid_pix   = pix_med_log.notnull() & stripe2d
    # NN-fill missing medians within stripe
    if bool((~valid_pix & stripe2d).any()):
        pix_med_log = nn_fill_median2d(pix_med_log, valid_pix)

    # regional seasonal anomaly (log units)
    reg_anom_log = regional_seasonal_anom(log_da)

    # reconstruct
    recon_log = pix_med_log.sel(lat=ds.lat, lon=ds.lon).broadcast_like(log_da) + reg_anom_log

    # where we had obs, keep obs; else fill
    log_filled = xr.where(da_s.notnull(), log_da, recon_log)

    # soft guard: cap at log(pix_med * soft_factor)
    pix_med_val = np.exp(pix_med_log)
    hi_cap = np.log(pix_med_val * soft_factor)
    lo_cap = np.log(pix_med_val / soft_factor)
    log_filled = xr.where(log_filled > hi_cap, hi_cap, log_filled)
    log_filled = xr.where(log_filled < lo_cap, lo_cap, log_filled)

    filled = np.exp(log_filled).where(stripe3d)  # off-stripe stays NaN
    ds[name] = filled.astype("float32")

# ---------------------------------------------------------------- fill nflh (linear; can be neg) ----
def fill_nflh(name: str, soft_factor: float):
    da_raw = ds0[name]
    obs_flag = (da_raw.notnull() & stripe3d).astype(np.int8)
    ds[f"{name}_obs"] = obs_flag

    da_s = da_raw.where(stripe3d)

    # pixel median + IQR
    pix_med = da_s.median("time", skipna=True)
    q25 = da_s.quantile(0.25, "time", skipna=True)
    q75 = da_s.quantile(0.75, "time", skipna=True)
    iqr = (q75 - q25).fillna(0)

    valid_pix = pix_med.notnull() & stripe2d
    if bool((~valid_pix & stripe2d).any()):
        pix_med = nn_fill_median2d(pix_med, valid_pix)

    reg_anom = regional_seasonal_anom(da_s)

    recon = pix_med.sel(lat=ds.lat, lon=ds.lon).broadcast_like(da_s) + reg_anom

    filled = xr.where(da_s.notnull(), da_s, recon)

    # soft guard: ± soft_factor * IQR around median
    hi_cap = (pix_med + soft_factor * iqr).broadcast_like(filled)
    lo_cap = (pix_med - soft_factor * iqr).broadcast_like(filled)
    filled = xr.where(filled > hi_cap, hi_cap, filled)
    filled = xr.where(filled < lo_cap, lo_cap, filled)

    ds[name] = filled.where(stripe3d).astype("float32")

# ---------------------------------------------------------------- run fills ----
print("\n▶ filling chlor_a …")
fill_pos_var("chlor_a", SOFT_FACTOR_CHL)

print("\n▶ filling Kd_490 …")
fill_pos_var("Kd_490", SOFT_FACTOR_KD)

print("\n▶ filling nflh …")
fill_nflh("nflh", SOFT_FACTOR_NFLH)

# ---------------------------------------------------------------- log_chl from filled chlor_a ----
print("\n▶ recomputing log_chl …")
chl = ds["chlor_a"]
pos = chl.where(chl > 0)
chl_min = float(pos.min(skipna=True))
if not np.isfinite(chl_min) or chl_min <= 0:
    chl_min = 0.01459
floor = max(chl_min, MIN_POS_FLOOR)

log_chl = np.log(np.maximum(chl, floor)).where(stripe3d)
ds["log_chl"] = log_chl.astype("float32")
# obs flag mirrors chlor_a
ds["log_chl_obs"] = ds["chlor_a_obs"].astype(np.int8)

# ---------------------------------------------------------------- report ----
print("\nNaNs inside CA coastal stripe (expect 0):")
for v in ["chlor_a","Kd_490","nflh","log_chl"]:
    n_nan = int(ds[v].where(stripe3d).isnull().sum())
    print(f"  {v:7s}: {n_nan:,}")

# ---------------------------------------------------------------- write ----
enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf(OUT_FP, mode="w", engine="netcdf4", encoding=enc)
print(f"\n✅ wrote CA local-season filled cube → {OUT_FP.name}")
