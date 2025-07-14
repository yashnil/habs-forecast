#!/usr/bin/env python3
"""
sanity_check_modis.py
---------------------
Compare a handful of archived 4-km MODIS-Aqua 8-day composites with a
fresh re-projection of the original L3B bin files.

Pass criteria (rule-of-thumb):
    mean_diff  < 1e-6
    max_diff   < 1e-3

Edit the RAW_DIR / ARCHIVED paths if yours differ.
"""

import random, xarray as xr, numpy as np, pathlib, sys, xesmf as xe

# ─────────────────────────────── paths ──────────────────────────────────
RAW_DIR     = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/chlorophyll")
ARCHIVED_DIR = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/modis_l3m/chlorophyll")

if not RAW_DIR.is_dir() or not ARCHIVED_DIR.is_dir():
    sys.exit("❌  Check RAW_DIR and ARCHIVED_DIR paths at top of script")

# ─────────────────────── canonical 4-km grid (lon, lat) ─────────────────
try:
    # if you already have grid_modis.py
    from grid_modis import lons, lats                     # 2-D arrays
except ModuleNotFoundError:
    # build it quickly here (plate-carrée, 4 km ≈ 0.036° lat step)
    LON_W, LON_E, LAT_S, LAT_N = -125.0, -115.0, 32.0, 50.0
    STEP = 4 / 111.195                                     # ~0.036°
    lat1d = np.arange(LAT_S, LAT_N + STEP, STEP)
    lon1d = np.arange(LON_W, LON_E + STEP, STEP)
    lats, lons = np.meshgrid(lat1d, lon1d, indexing="ij")

target_grid = xr.Dataset({
    "lon": (("lat", "lon"), lons),
    "lat": (("lat", "lon"), lats),
})

# ─────────────────── choose four composites robustly ────────────────────
files_per_year = {
    yr: list(ARCHIVED_DIR.glob(f"AQUA_MODIS.{yr}*.nc"))
    for yr in range(2003, 2022)
}
valid_years = [yr for yr, lst in files_per_year.items() if len(lst) >= 2]
if len(valid_years) < 2:
    sys.exit("❌  Need at least two years with ≥2 archived composites each")

years_chosen = random.sample(valid_years, 2)
sample_files = sum(
    [random.sample(files_per_year[yr], 2) for yr in years_chosen],
    []
)

print(f"✔︎  Using composites:")
for fp in sample_files:
    print("   •", fp.name)

# ───────────────────────── helper: regrid raw L3B ───────────────────────
# ───────────────────────── helper: regrid raw L3B ───────────────────────
def regrid_raw(fp4km: pathlib.Path) -> xr.DataArray:
    """
    Given an archived 4-km composite path, locate the corresponding raw
    L3B bin file (any …L3b.*.nc), regrid it to the 4-km grid, and return
    a DataArray masked with nobs >= 3.
    """
    # e.g. fp4km.name → AQUA_MODIS.20190407_20190414_4km_L3m.nc
    date_tag = fp4km.name.split("_4km_")[0]      # AQUA_MODIS.20190407_20190414

    # find the raw file (could be CHL.x.nc, KD.x.nc, etc.)
    matches = list(RAW_DIR.glob(f"{date_tag}.L3b.*.nc"))
    if not matches:
        raise FileNotFoundError(f"No raw L3B file matching {date_tag} in {RAW_DIR}")
    fp_raw = matches[0]        # take the first hit

    ds = xr.open_dataset(fp_raw, group="level-3_binned_data")
    var = next(v for v in ds.data_vars if not v.endswith("_index"))
    da  = (ds[var]["sum"] / ds[var]["nobs"]).where(ds[var]["nobs"] >= 3)

    src_grid = xr.Dataset({
        "lon": (("y", "x"), ds["longitude"]),
        "lat": (("y", "x"), ds["latitude"]),
    })
    regridder = xe.Regridder(src_grid, target_grid,
                             method="bilinear", reuse_weights=False)
    return regridder(da)

# ───────────────────── run comparison & print stats ─────────────────────
print("\nΔ = |archived − fresh| over ocean pixels")
print("Composite             mean_diff     max_diff")

for fp4 in sample_files:
    comp_tag = fp4.stem.split(".")[1]                 # e.g., 20030101_20030108
    da_arch  = xr.open_dataset(fp4)["chlor_a"]
    da_fresh = regrid_raw(fp4)

    delta = np.abs(da_arch - da_fresh)
    print(f"{comp_tag:>17}   {float(delta.mean()):10.3e}   {float(delta.max()):10.3e}")
