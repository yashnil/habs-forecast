#!/usr/bin/env python3
"""
Subset the coastal cube to a California regional domain:

    lat: 32°N .. 42°N
    lon: -125° .. -115°

Keeps:
    • all time steps (851)
    • all root variables in the coastal cube
    • writes a new 2‑D /masks/coastal_water_mask reflecting the *subset* domain
      (union of non‑NaN across chlor_a, Kd_490, nflh)

Output:
    HAB_master_8day_4km_coastal_CA.nc
"""

from __future__ import annotations
import pathlib
import numpy as np
import xarray as xr
import netCDF4

# ------------------------------------------------------------------
# paths
# ------------------------------------------------------------------
ROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
FIN  = ROOT / "Data" / "Finalized"

SRC = FIN / "HAB_master_8day_4km_coastal.nc"
DST = FIN / "HAB_master_8day_4km_coastal_CA.nc"

LAT_MIN, LAT_MAX = 32.0, 42.0
LON_MIN, LON_MAX = -125.0, -115.0

# ------------------------------------------------------------------
# open coastal cube
# ------------------------------------------------------------------
print("🔹 opening source coastal cube …")
ds = xr.open_dataset(SRC)

# make sure dims are (time, lat, lon)
ds = ds.transpose("time", "lat", "lon")

# ------------------------------------------------------------------
# slice helper to cope w/ ascending or descending coordinate order
# ------------------------------------------------------------------
def coord_slice(coord, vmin, vmax):
    ascending = bool(coord[0] < coord[-1])
    return slice(vmin, vmax) if ascending else slice(vmax, vmin)

lat_slice = coord_slice(ds.lat, LAT_MIN, LAT_MAX)
lon_slice = coord_slice(ds.lon, LON_MIN, LON_MAX)

# subset
ds_ca = ds.sel(lat=lat_slice, lon=lon_slice)

print(f"   subset dims: time={ds_ca.sizes['time']}, "
      f"lat={ds_ca.sizes['lat']}, lon={ds_ca.sizes['lon']}")

# ------------------------------------------------------------------
# rebuild a coastal stripe mask for the subset domain
# ------------------------------------------------------------------
# Use union of non‑NaN across the three MODIS optics variables
optics = []
for v in ("chlor_a", "Kd_490", "nflh"):
    if v in ds_ca:
        optics.append(ds_ca[v].notnull().any("time"))
if not optics:
    raise RuntimeError("None of the expected optics variables found in subset!")

coastal_mask_2d = xr.zeros_like(optics[0], dtype=bool)
for o in optics:
    coastal_mask_2d = coastal_mask_2d | o

# convert to int8 (1=water stripe, 0=off stripe)
coastal_mask_2d_i = coastal_mask_2d.astype("int8")
coastal_mask_2d_i.name = "coastal_water_mask"
coastal_mask_2d_i.attrs.update(
    long_name="coastal water (≤16 km) mask – CA subset",
    flag_values=[0,1],
)

# counts
n_mask = int(coastal_mask_2d.sum())
n_grid = ds_ca.sizes["lat"] * ds_ca.sizes["lon"]
print(f"   CA coastal stripe pixels: {n_mask:,} "
      f"({n_mask/n_grid*100:.2f}% of regional grid)")

# ------------------------------------------------------------------
# write new NetCDF
#   root group = cropped dataset variables
#   masks group = 2D coastal_water_mask only (water_mask omitted here;
#                 add if you want by building union across all vars)
# ------------------------------------------------------------------
print(f"→ writing {DST.name} …")

# write the root dataset first
enc_root = {v: {"zlib": True, "complevel": 4,
                # pick chunk sizes scaled to new region
                "chunksizes": (90,
                               min(ds_ca.sizes["lat"], 144),
                               min(ds_ca.sizes["lon"], 80))}
            for v in ds_ca.data_vars}

ds_ca.to_netcdf(DST, mode="w", engine="netcdf4", encoding=enc_root)

# now append the mask group
with netCDF4.Dataset(DST, "a") as nc:
    g = nc.createGroup("masks")
    v = g.createVariable("coastal_water_mask", "i1",
                         ("lat", "lon"),
                         zlib=True, complevel=4,
                         chunksizes=(min(ds_ca.sizes["lat"], 144),
                                     min(ds_ca.sizes["lon"], 80)))
    v.setncatts(coastal_mask_2d_i.attrs)
    v[:] = coastal_mask_2d_i.values

print("✅  California coastal subset written.")
