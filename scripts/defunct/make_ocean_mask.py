# scripts/make_ocean_mask.py   (drop-in replacement)
import pathlib, yaml, xarray as xr, numpy as np

root  = pathlib.Path(__file__).resolve().parents[1]
cfg   = yaml.safe_load(open(root / "config.yaml"))
cube  = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"

print("▶ scanning cube for NaNs …")

# ── 1. open BOTH groups in a context so handles close afterwards ──
with xr.open_dataset(cube,                           chunks={"time": 50}) as ds_root, \
     xr.open_dataset(cube, group="derivatives", chunks={"time": 50}) as ds_deriv:

    ds_all = xr.merge([ds_root, ds_deriv])              # ≈ 18 vars total

    finite_every = (~ds_all.isnull()).all("time")       # (var,lat,lon)
    mask = finite_every.to_array("tmp").all("tmp")      # (lat,lon) true↔no-NaN

    ocean0 = ~ds_root["chlor_a"].isnull().all("time")   # coastal footprint
    mask   = (mask & ocean0).load()                     # bring into RAM *here*

# ── 2. now file handles are closed; safe to append ────────────────
n_keep  = int(mask.sum())          # numpy scalar → int
n_total = mask.size
print(f"   ✓ valid coastal pixels retained: {n_keep:,} / {n_total} "
      f"({n_keep/n_total*100:.1f} %)")

mask = mask.rename("ocean_mask")

mask.to_netcdf(
    cube,
    group    = "masks",
    mode     = "a",
    encoding = {"ocean_mask": {"zlib": True, "complevel": 4}},
)
print("✓ ocean mask saved → group='masks'/ocean_mask")
