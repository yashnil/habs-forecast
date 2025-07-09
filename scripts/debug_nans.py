#!/usr/bin/env python3
#scripts/debug_nans.py
"""
NaN scanner for the std-0(F) cube
---------------------------------
* Text table of NaN counts (water pixels only)
* Optional spatial map (--plot <var> | ANY) saved as PNG
"""
from __future__ import annotations
import argparse, pathlib, yaml, numpy as np, xarray as xr, matplotlib.pyplot as plt

# ───────── CLI ────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--cube", default="HAB_cube_std0_2016_2021.nc",
               help="NetCDF file inside data_root to inspect")
p.add_argument("--plot", metavar="VAR",
               help="Make PNG of NaN locations for VAR (or 'ANY')")
args = p.parse_args()

# ───────── paths & load cube lazily ───────────────────────────────
root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))
cube = root / cfg["data_root"] / args.cube

ds_root  = xr.open_dataset(cube, chunks={"time": 50})
try:
    ds_deriv = xr.open_dataset(cube, group="derivatives", chunks={"time": 50})
except FileNotFoundError:
    ds_deriv = xr.Dataset()

ds = xr.merge([ds_root, ds_deriv])

# ───────── coast mask via thetao (finite anywhere) ───────────────
try:
    coast = xr.open_dataset(cube, group="masks")["water_mask"].load()
except FileNotFoundError:
    raise SystemExit("❌  masks/water_mask not found – run make_water_mask.py first")
n_water = int(coast.sum())
print(f"Water-grid cells : {n_water:,}\n")

# ───────── NaN count per var (water only) ────────────────────────
print("╭──────── NaN report (ocean only) ───────╮")
bad_vars = []
for v in ds.data_vars:
    nan_mask = ds[v].where(coast).isnull() & coast 
    n_bad = int(nan_mask.sum().compute())
    if n_bad:
        worst = int(nan_mask.sum(("lat","lon")).max().compute())
        bad_vars.append(v)
        print(f"⚠️  {v:18s} NaNs = {n_bad:,}  (worst composite: {worst:,})")
    else:
        print(f"✓ {v:18s} OK")

if not bad_vars:
    print("✓ No NaNs remain over water — dataset is clean.")
    exit(0)

# ───────── optional plot ─────────────────────────────────────────
if args.plot:
    if args.plot != "ANY" and args.plot not in ds:
        raise SystemExit(f"Variable '{args.plot}' not found in cube.")

    if args.plot == "ANY":
        nan2d = xr.concat([ds[v].where(coast).isnull() for v in ds.data_vars],
                          dim="tmp").any("tmp").any("time")
        title = "ANY variable"
        fname = "nanmap_ANY.png"
    else:
        nan2d = ds[args.plot].where(coast).isnull().any("time")
        title = args.plot
        fname = f"nanmap_{args.plot}.png"

    nan_img = nan2d.astype(int).plot.imshow(
        cmap="Reds", add_colorbar=False
    )
    plt.title(f"NaN locations for {title} (red=NaN)")
    plt.xlabel("lon index"); plt.ylabel("lat index")
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"🖼  saved → {fname}")
    plt.show()

'''
Results:

Water-grid cells : 32,642

╭──────── NaN report (ocean only) ───────╮
✓ chlor_a            OK
✓ Kd_490             OK
✓ nflh               OK
✓ sst                OK
✓ tp                 OK
✓ avg_sdswrf         OK
✓ t2m                OK
✓ d2m                OK
✓ u10                OK
✓ v10                OK
✓ uo                 OK
✓ vo                 OK
✓ zos                OK
✓ so                 OK
✓ thetao             OK
✓ log_chl            OK
✓ sin_doy            OK
✓ cos_doy            OK
✓ curl_uv            OK
✓ dist_river_km      OK
✓ log1p_dist_river   OK
✓ No NaNs remain over water — dataset is clean.

'''