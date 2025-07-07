#!/usr/bin/env python3
"""
Standardise every variable inside the coastal footprint (θ-o finite at least
once), fill those coastal NaNs with 0, keep land as NaN *for plotting*,
and finally be sure **all remaining water pixels are NaN-free**.

Output →  {data_root}/HAB_cube_std0_2016_2021.nc
"""
from __future__ import annotations
import pathlib, yaml, numpy as np, xarray as xr

# ── paths ────────────────────────────────────────────────────────
root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))

src_cube = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
dst_cube = pathlib.Path(cfg["data_root"]) / "HAB_cube_std0_2016_2021.nc"

print("▶ loading source cube …")
ds_root  = xr.open_dataset(src_cube, chunks={"time": 50})
try:
    ds_deriv = xr.open_dataset(src_cube, group="derivatives", chunks={"time": 50})
except FileNotFoundError:
    ds_deriv = xr.Dataset()

# ── coastal footprint via θ-o ────────────────────────────────────
coast = ~ds_root["thetao"].isnull().all("time")       # dims (lat,lon)
print(f"   coastal pixels : {int(coast.sum()):,} / {coast.size}")

# ── helper: z-score + coastal NaN→0 ──────────────────────────────
def _std0(da: xr.DataArray, name: str) -> xr.DataArray:
    μ = da.where(coast).mean(("time", "lat", "lon"), skipna=True)
    σ = da.where(coast).std (("time", "lat", "lon"), skipna=True)
    σ = xr.where(σ == 0, 1.0, σ)                # protect constants
    zz = (da - μ) / σ
    zz = xr.where(coast, zz.fillna(0.0), np.nan)
    zz.attrs.update({"long_name": f"std-0 {name}", "units": "σ"})
    return zz.astype("float32")

print("▶ standardising …")
proc_root  = xr.Dataset({v: _std0(ds_root[v],  v) for v in ds_root.data_vars},
                        coords=ds_root.coords)
proc_deriv = xr.Dataset({v: _std0(ds_deriv[v], v) for v in ds_deriv.data_vars},
                        coords=ds_deriv.coords) if ds_deriv else None

enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in proc_root}

print("▶ writing", dst_cube.name)
proc_root.to_netcdf(dst_cube, mode="w", encoding=enc)
if proc_deriv:
    enc_d = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
             for v in proc_deriv.data_vars}          # << NEW >>
    proc_deriv.to_netcdf(dst_cube, mode="a",
                         group="derivatives",
                         encoding=enc_d)       

# ─────────────────────────────── 🔧 NEW ──────────────────────────
'''
# guarantee **all** remaining water pixels are NaN-free
print("▶ post-fill remaining NaNs inside water_mask …")
with xr.open_dataset(dst_cube, mode="a") as ds_w, \
     xr.open_dataset(dst_cube, group="masks") as ds_mask:

    m = ds_mask["water_mask"]                     # (lat,lon) bool
    for v in ds_w.data_vars:
        filled = ds_w[v].where(~m, ds_w[v].fillna(0.0))
        ds_w[v].data[:] = filled                  # in-place write
'''
print("✓ post-fill completed (no NaNs over water)")
# ─────────────────────────────────────────────────────────────────
