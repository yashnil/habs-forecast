#!/usr/bin/env python3
import pathlib, yaml
import xarray as xr

# ─── Paths ─────────────────────────────────────────────────────────
root      = pathlib.Path(__file__).resolve().parents[1]
cfg       = yaml.safe_load(open(root / "config.yaml"))
data_root = pathlib.Path(cfg["data_root"])
cube_fp   = data_root / "HAB_cube_2016_2021_varspec_allfilled.nc"

# ─── Load dataset & water mask ────────────────────────────────────
ds      = xr.open_dataset(cube_fp)
water_m = ds["water_mask"].astype(bool)

n_water = int(water_m.sum().item())
n_time  = ds.sizes.get("time", 1)

print(f"Checking NaNs over water pixels for\n  {cube_fp}\n")
print(f"{'Variable':20s} {'# NaNs on water':>15s} {'% missing on water':>20s}")
print("-"*60)

for var in ds.data_vars:
    da = ds[var]

    # 1) build a boolean array: True where da is NaN AND on water
    if "time" in da.dims:
        # broadcast water_m over time
        wm3 = water_m.expand_dims(time=da.sizes["time"]).transpose("time","lat","lon")
        missing = int((da.isnull() & wm3).sum().item())
        total   = n_water * n_time
    else:
        missing = int((da.isnull() & water_m).sum().item())
        total   = n_water

    pct = 100 * missing / total if total>0 else 0.0
    print(f"{var:20s} {missing:15,d} {pct:20.2f}%")
