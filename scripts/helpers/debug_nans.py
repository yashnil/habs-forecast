#!/usr/bin/env python3
"""
Report NaN counts *inside* the coastal‑water mask.
"""

from __future__ import annotations
import pathlib, numpy as np, xarray as xr
from tabulate import tabulate

ROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
FIN  = ROOT / "Data" / "Finalized"

CUBE_COASTAL = FIN / "HAB_master_8day_4km_coastal.nc"
CUBE_FULL    = FIN / "HAB_master_8day_4km.nc"           # holds the mask

print("▶ opening datasets …")
ds      = xr.open_dataset(CUBE_COASTAL, engine="netcdf4")
mask2d  = xr.open_dataset(CUBE_FULL, group="masks")["coastal_water_mask"]

# build 3‑D mask (time,lat,lon)
mask3d  = mask2d.astype(bool).expand_dims(time=ds.time)

stripe_cells = mask3d.sum().item()
total_cells  = int(np.prod(list(ds.sizes.values())))

print(f"   coastal‑stripe cells: {stripe_cells:,} "
      f"({stripe_cells / total_cells * 100:.2f} % of the grid)\n")

rows = []
for var, da in ds.data_vars.items():
    nan_in_stripe = np.logical_and(np.isnan(da), mask3d).sum().item()
    pct_nan       = nan_in_stripe / stripe_cells * 100
    rows.append([var, f"{nan_in_stripe:,}", f"{stripe_cells:,}",
                 f"{pct_nan:6.2f} %"])

print("NaN fraction *inside* the 16‑km coastal stripe")
print(tabulate(rows,
               headers=["variable", "NaN count", "stripe cells", "NaN %"],
               tablefmt="github"))
