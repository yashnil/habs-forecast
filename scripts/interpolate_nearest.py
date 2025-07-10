#!/usr/bin/env python3
import pathlib, yaml
import xarray as xr
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────
root      = pathlib.Path(__file__).resolve().parents[1]
cfg       = yaml.safe_load(open(root / "config.yaml"))
data_root = pathlib.Path(cfg["data_root"])
in_fp     = data_root / "HAB_cube_2016_2021_varspec_imputed.nc"
out_fp    = data_root / "HAB_cube_2016_2021_varspec_allfilled.nc"

# ─── 1) Load dataset & water mask ─────────────────────────────────
ds = xr.open_dataset(in_fp)
wm = ds["water_mask"].astype(bool)  # dims (lat, lon)

# ─── 2) Variables we know still have NaNs on water  ──────────────
to_fix = ["sst", "curl_uv", "log_chl"]

print("Imputing remaining NaNs on water for:", to_fix, "\n")

for v in to_fix:
    da   = ds[v]
    da_w = da.where(wm)  # broadcast wm over time dimension

    n0 = int(da_w.isnull().sum().item())
    print(f"{v:8s} before fill → {n0:,} NaNs on water")

    if v == "log_chl":
        # fill log10(0) holes by flatting up to the lowest finite log value
        min_val = float(da_w.min(skipna=True).item())
        da_f = da_w.fillna(min_val)
    else:
        # two rounds of lat→lon nearest-neighbour to close any islands
        da1 = da_w.interpolate_na(dim="lat", method="nearest")
        da2 = da1.interpolate_na(dim="lon", method="nearest")
        da3 = da2.interpolate_na(dim="lat", method="nearest")
        da4 = da3.interpolate_na(dim="lon", method="nearest")
        da_f = da4

    # stitch back: water pixels take da_f, land remain as-is
    ds[v] = xr.where(wm, da_f, da)

    n1 = int(ds[v].where(wm).isnull().sum().item())
    print(f"{v:8s} after  fill → {n1:,} NaNs on water\n")

# ─── 3) Save new cube with zero NaNs on water ───────────────────────
enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf(out_fp, mode="w", format="NETCDF4", encoding=enc)
print(f"✅ Wrote fully-filled cube →\n   {out_fp}")
