#!/usr/bin/env python3
import xarray as xr
import numpy as np

# ─── 0) file path ───────────────────────────────────────────
cube = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_2016_2021_fixed.nc"

# ─── 1) load dataset ───────────────────────────────────────
ds = xr.open_dataset(cube, chunks={"time": 50})

# ─── 2) extract water mask ─────────────────────────────────
water = ds["water_mask"] == 1
n_water = int(water.sum().values)
n_time  = ds.sizes["time"]

# ─── 3) report totals ──────────────────────────────────────
print(f"Water pixels per time slice: {n_water}")
print(f"Time steps             : {n_time}")
print(f"Total slots (water×time): {n_water * n_time}\n")

# ─── 4) quantify NaNs per variable ─────────────────────────
print(f"{'Variable':12s} {'#NaNs':>10s} {'% Missing':>12s}")
for var in ds.data_vars:
    if var == "water_mask":
        continue
    # count NaNs only on water pixels
    n_nan = int((ds[var].isnull() & water).sum().values)
    pct   = 100 * n_nan / (n_water * n_time)
    print(f"{var:12s} {n_nan:10d} {pct:12.2f}%")

'''
Results:

Variable          #NaNs    % Missing
chlor_a         1440290        21.42%
Kd_490          1435620        21.35%
nflh            2015944        29.98%
sst             1594112        23.71%
tp                    0         0.00%
avg_sdswrf            0         0.00%
t2m                   0         0.00%
d2m                   0         0.00%
u10                   0         0.00%
v10                   0         0.00%
uo                    0         0.00%
vo                    0         0.00%
zos                   0         0.00%
so                    0         0.00%
thetao                0         0.00%
log_chl         1701548        25.30%
sin_doy               0         0.00%
cos_doy               0         0.00%
'''