#!/usr/bin/env python3
# scripts/helpers/enforce_coastal_mask.py
"""
Flatten the cube, apply the coastal‑water mask to *every* variable,
summarise NaNs, and write one clean NetCDF file.

Output
------
Data/Finalized/HAB_master_8day_4km_coastal.nc   (everything in root group)
Data/Finalized/qc_coastal_stripe/*.nc           (1 frame per quarter)
"""

from __future__ import annotations
import pathlib, numpy as np, xarray as xr
from tabulate import tabulate          # pip install tabulate
from dask.diagnostics import ProgressBar

# ───────── paths ────────────────────────────────────────────────────────────
ROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
FIN  = ROOT / "Data" / "Finalized"

SRC_CUBE = FIN / "HAB_master_8day_4km.nc"          # original with groups
DST_CUBE = FIN / "HAB_master_8day_4km_coastal.nc"  # new flattened file
SNAP_DIR = FIN / "qc_coastal_stripe"; SNAP_DIR.mkdir(exist_ok=True)

# ───────── load mask & data ────────────────────────────────────────────────
print("▶ loading coastal mask …")
mask = xr.open_dataset(SRC_CUBE, group="masks")["coastal_water_mask"]

print("▶ loading root & derivatives groups …")
root_ds   = xr.open_dataset(SRC_CUBE, group="/")                # core bands
deriv_ds  = xr.open_dataset(SRC_CUBE, group="derivatives")      # curl, etc.

full_ds   = xr.merge([root_ds, deriv_ds])   # one Dataset, no groups

# ───────── apply mask (broadcasts automatically) ───────────────────────────
print("▶ applying mask to every band …")
coastal_ds = full_ds.where(mask == 1)

# ───────── quick NaN summary ───────────────────────────────────────────────
rows = []
for v, da in coastal_ds.data_vars.items():
    n_all   = int(np.isnan(da).sum())
    tot_all = int(np.prod(da.shape))
    pct_all = n_all / tot_all * 100

    in_strip = da  # already masked outside strip
    n_strip  = int(np.isnan(in_strip).sum())
    tot_strip= int(np.prod(in_strip.shape))
    pct_strip= n_strip / tot_strip * 100

    rows.append([v, f"{pct_all:5.1f} %", f"{pct_strip:5.1f} %"])

print("\nNaN fraction (entire grid vs. coastal strip)")
print(tabulate(rows,
               headers=["variable", "all pixels", "coastal strip"],
               tablefmt="github"))

# ───────── write single NetCDF file ----------------------------------------
print(f"\n▶ writing {DST_CUBE.name} …")
enc = {v: {"zlib": True, "complevel": 4, "chunksizes": (90, 144, 80)}
        for v in coastal_ds.data_vars}

with ProgressBar():
    coastal_ds.to_netcdf(DST_CUBE, encoding=enc, engine="netcdf4")

print("   ✓ done.")

# ───────── quarterly QC snapshots ------------------------------------------
print("▶ QC snapshots →", SNAP_DIR.name)
snap_enc = {v: {"zlib": True, "complevel": 4} for v in coastal_ds.data_vars}
for ts in coastal_ds.time.values[::46]:          # every ≈ quarter
    ts_str = np.datetime_as_string(ts, unit="D")
    coastal_ds.sel(time=ts).to_netcdf(
        SNAP_DIR / f"stripe_{ts_str}.nc",
        encoding=snap_enc
    )

print("✅  all set – inspect snapshots in Panoply to verify the coastal band.")
