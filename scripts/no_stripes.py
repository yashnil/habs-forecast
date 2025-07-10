#!/usr/bin/env python3
"""
Remove spurious three-column and one-row stripes from the variable-specific all‐filled HAB cube by refining the water mask:
 - Force all pixels with lon >= -116.0 to land (mask=0)
 - Force all pixels with lat <= 32.0 to land (mask=0)
Then re–mask all variables accordingly and save as a new NetCDF.
"""
import pathlib
import xarray as xr
import numpy as np

# 1) Paths
root    = pathlib.Path(__file__).resolve().parents[1]
cfg_fp = root / "config.yaml"
import yaml
cfg     = yaml.safe_load(open(cfg_fp))
data_root = pathlib.Path(cfg["data_root"])

in_fp   = data_root / "HAB_cube_2016_2021_varspec_allfilled.nc"
out_fp  = data_root / "HAB_cube_2016_2021_varspec_nostripes.nc"

# 2) Open dataset
ds = xr.open_dataset(in_fp)

# 3) Original water mask
water = ds["water_mask"].astype(bool)

# 4) Build lon/lat masks
lon = ds["lon"]  # dims: (lon,)
lat = ds["lat"]  # dims: (lat,)

# Mask: only keep ocean west of -116 and north of 32
lon_ok = lon < -116.0  # True for lon < -116
lat_ok = lat >  32.0   # True for lat > 32

# Broadcast to 2D
lon2d = lon_ok.broadcast_like(water)
lat2d = lat_ok.broadcast_like(water)

# 5) New water mask: require original water AND lon_ok AND lat_ok
new_water = water & lon2d & lat2d

# 6) Apply new mask to all 2D+time variables
for var in ds.data_vars:
    da = ds[var]
    if set(("lat","lon")).issubset(da.dims):
        # keep time coords if present
        ds[var] = da.where(new_water)

# 7) Overwrite water_mask
ds["water_mask"] = new_water.astype(np.uint8)

# 8) Save out
encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf(out_fp, mode="w", format="NETCDF4", encoding=encoding)

print(f"✅ Wrote stripped cube → {out_fp}")
print(f"Water pixels before: {int(water.sum().item())}")
print(f"Water pixels after : {int(new_water.sum().item())}")
