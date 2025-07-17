#!/usr/bin/env python3
"""
Report NaN counts *inside* (and outside) the CA coastal stripe
for **all** gridded variables in a coastal CA cube.

Edit the FILLED / RAW paths below as needed.
"""

from __future__ import annotations
import pathlib
import xarray as xr
import numpy as np
from tabulate import tabulate

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
FIN  = ROOT / "Data" / "Finalized"

# dataset you want to inspect (filled or not)
FILLED = FIN / "HAB_master_8day_4km_coastal_CA_climclip.nc"
# *raw* CA coastal cube used to derive the coastal stripe (before filling)
RAW    = FIN / "HAB_master_8day_4km_coastal_CA.nc"

# ── open datasets ─────────────────────────────────────────────────────────
print("🔹 opening datasets …")
ds_f = xr.open_dataset(FILLED).transpose("time", "lat", "lon")
ds_r = xr.open_dataset(RAW).transpose("time", "lat", "lon")

# ── 2‑D coastal stripe mask from union of raw MODIS optical vars ──────────
# (we intentionally DO NOT use other vars so that globally valid atmos/ocean
#  variables don't blow the stripe up to the full domain.)
stripe2d = (
    ds_r["chlor_a"].notnull().any("time")
  | ds_r["Kd_490"].notnull().any("time")
  | ds_r["nflh"].notnull().any("time")
).astype(bool)
stripe2d.name = "coastal_stripe"

n_px   = int(stripe2d.sum())
n_grid = ds_f.sizes["lat"] * ds_f.sizes["lon"]
print(f"   stripe pixels: {n_px:,} ({n_px / n_grid:.2%} of regional grid)")

# 3‑D broadcast (time,lat,lon)
stripe3d = stripe2d.expand_dims(time=ds_f.time).transpose("time", "lat", "lon")
stripe_cells3d = int(stripe3d.sum())
print(f"   stripe cells (time×px): {stripe_cells3d:,}\n")

# ── which vars to check? all gridded (lat & lon in dims) ──────────────────
vars_check = [
    v for v in ds_f.data_vars
    if {"lat", "lon"}.issubset(ds_f[v].dims)
]

rows = []
rows_off = []

for v in vars_check:
    da = ds_f[v]

    # boolean NaN mask
    nanmask = da.isnull()

    # inside stripe
    nan_in  = int((nanmask & stripe3d).sum())
    pct_in  = nan_in / stripe_cells3d * 100

    # outside stripe
    nan_out = int((nanmask & (~stripe3d)).sum())
    # number of off‑stripe cells in domain
    off_cells3d = int((~stripe3d).sum())
    pct_out = nan_out / off_cells3d * 100 if off_cells3d else np.nan

    rows.append([v, f"{nan_in:,}", f"{stripe_cells3d:,}", f"{pct_in:5.2f} %"])
    rows_off.append([v, f"{nan_out:,}", f"{off_cells3d:,}", f"{pct_out:5.2f} %"])

print("NaNs *inside* CA coastal stripe:")
print(tabulate(rows, headers=["var", "NaNs", "stripe cells", "NaN %"], tablefmt="github"))

print("\nNaNs OUTSIDE stripe (context):")
print(tabulate(rows_off, headers=["var", "NaNs", "off‑stripe cells", "NaN %"], tablefmt="github"))

print("\n✅ sanity_allvars done.")
