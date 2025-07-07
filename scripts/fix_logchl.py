#!/usr/bin/env python3
# scripts/fix_logchl.py      ← overwrite the old version
"""
Add or replace variable 'log_chl' in the *raw* cube.

Definition
----------
log_chl = log10( chlor_a )             (chlor_a > 0)
          NaN                          (chlor_a ≤ 0  or  NaN)

Nothing else in the file is modified.
"""
from __future__ import annotations
import pathlib, yaml, numpy as np, xarray as xr

root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))

raw   = pathlib.Path(cfg["data_root"]) / "HAB_cube_RAW.nc"
print("▶ writing log_chl into", raw.name)

with xr.open_dataset(raw, mode="a") as ds:
    chl = ds["chlor_a"].load()            # bring into RAM once
    valid = chl > 0
    log_chl = xr.full_like(chl, np.nan, dtype="float32")
    log_chl.values[valid.values] = np.log10(chl.values[valid.values])

    log_chl.attrs = {
        "long_name": "log10 chlorophyll-a concentration",
        "units": "log10(mg m-3)"
    }

    # drop old var if it exists, then write the new one
    if "log_chl" in ds:
        ds = ds.drop_vars("log_chl")
    ds["log_chl"] = log_chl
    ds.log_chl.encoding.update({"zlib": True, "complevel": 4})
print("✓ log_chl saved.")
