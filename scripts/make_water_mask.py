#!/usr/bin/env python3
# scripts/make_water_mask.py
"""
Build precise coastal-water mask and store it in the cube:
group='masks'/water_mask   (1=ocean, 0=land/invalid)
"""
from __future__ import annotations
import pathlib, yaml, xarray as xr

root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))
cube = pathlib.Path(cfg["data_root"]) / "HAB_cube_std0_2016_2021.nc"

print("▶ building strict water-mask …")
with xr.open_dataset(cube, chunks={"time": 50}) as ds:

    chl_foot   = ds["chlor_a"].isnull().all("time") == False
    theta_foot = ds["thetao"].isnull().all("time") == False
    water_mask = (chl_foot & theta_foot).load()      # (lat,lon) boolean

n_pix = int(water_mask.sum())
print(f"   ✓ water pixels: {n_pix:,} / {water_mask.size} "
      f"({n_pix/water_mask.size*100:.1f} %)")

# ─── write as its own Dataset so xarray can create the subgroup ───
ds_mask = xr.Dataset({"water_mask": water_mask.astype("int8")})
ds_mask.to_netcdf(
    cube, mode="a", group="masks", engine="netcdf4",
    encoding={"water_mask": {"zlib": True, "complevel": 4}}
)
print("✓ water_mask appended under group='masks'")
