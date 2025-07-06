
#!/usr/bin/env python3
"""Quick NaN scanner for the std-0 cube."""
import pathlib, yaml, xarray as xr, numpy as np

root  = pathlib.Path(__file__).resolve().parents[1]
cfg   = yaml.safe_load(open(root / "config.yaml"))
cube  = pathlib.Path(cfg["data_root"]) / "HAB_cube_std0_2016_2021.nc"

# ── load root + derivatives lazily (keeps memory small) ──────────
ds_root   = xr.open_dataset(cube,                           chunks={"time": 50})
try:
    ds_deriv  = xr.open_dataset(cube, group="derivatives", chunks={"time": 50})
except FileNotFoundError:
    ds_deriv = xr.Dataset()

ds = xr.merge([ds_root, ds_deriv])

print("╭───────────── NaN report ─────────────╮")
bad_vars = []
for v in ds.data_vars:
    n_bad = int(ds[v].isnull().sum().compute())
    if n_bad:
        bad_vars.append(v)
        print(f"⚠️  {v:18s}  NaNs = {n_bad:,}")
    else:
        print(f"✓ {v:18s}  OK")

# ── bail out if everything is fine ───────────────────────────────
if not bad_vars:
    print("✓ No NaNs remain — dataset is clean.")
    quit()

print("\nInspecting first offending location …")
for v in bad_vars:
    bad_mask = ds[v].isnull()
    if bad_mask.any():
        idx = np.argwhere(bad_mask.values)[0]      # (lat, lon, time) order
        lat_i, lon_i, time_i = idx
        print(f"{v}: NaN at time={int(time_i)}  lat_i={lat_i}  lon_i={lon_i}")
        break
