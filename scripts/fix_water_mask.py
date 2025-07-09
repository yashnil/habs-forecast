#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pathlib

# ─── 0) absolute paths ───────────────────────────────────────────
in_nc   = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_2016_2021.nc"
out_nc  = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_2016_2021_fixed.nc"

# ─── 1) load the original root‑group data ─────────────────────────
ds = xr.open_dataset(in_nc, chunks={"time":50})

# ─── 2) compute the mask from uo ─────────────────────────────────
#    any (lat,lon) with a finite uo at any time → water
uo   = ds["uo"].values                   # shape = (time, lat, lon)
water = np.any(np.isfinite(uo), axis=0).astype(np.uint8)       # shape = (lat, lon)

# ─── 3) attach water_mask as a new var at root level ───────────
ds["water_mask"] = (("lat","lon"), water)
ds["water_mask"].attrs = {
    "long_name":   "1=water, 0=land (finite uo)",
    "units":       "1",
    "coordinates":"lon lat",
}

# ─── 4) write a brand‑new NetCDF4 file ───────────────────────────
encoding = {"water_mask":{"zlib":True,"complevel":4}}
ds.to_netcdf(out_nc, mode="w", format="NETCDF4", encoding=encoding)

print(f"✅ Wrote new cube with water_mask at root → {out_nc}")
