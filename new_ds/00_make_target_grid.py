#!/usr/bin/env python3
"""
Build the MODIS-Aqua 4-km plate-carrée grid
(lon: −125→−115, lat: 32→50, step ≈4 km) and store it in
modis_4km_grid.npz for align_utils.py.
"""

import numpy as np, pathlib

GRID_FILE = pathlib.Path(__file__).with_name("modis_4km_grid.npz")

lon_w, lon_e = -125.0, -115.0
lat_s, lat_n =   32.0,   50.0
ddeg         =   4 / 111.195          # 4 km in degrees

lat1d = np.arange(lat_s, lat_n + 1e-6, ddeg)
lon1d = np.arange(lon_w, lon_e + 1e-6, ddeg)
lats, lons = np.meshgrid(lat1d, lon1d, indexing="ij")   # shape (432,240)

np.savez(GRID_FILE, lats=lats.astype("float32"), lons=lons.astype("float32"))
print(f"✔︎ wrote {GRID_FILE}  –  shape {lats.shape}")
