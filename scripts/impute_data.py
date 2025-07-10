#!/usr/bin/env python3
import xarray as xr
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────
fixed_nc   = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_2016_2021_fixed.nc"
orig_nc    = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_2016_2021.nc"
imputed_nc = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_2016_2021_imputed.nc"

# ─── 1) Load fixed root‐group (vars + water_mask) & derivatives ────
ds_fixed = xr.open_dataset(fixed_nc)  
ds_deriv = xr.open_dataset(orig_nc, group="derivatives")
ds       = xr.merge([ds_fixed, ds_deriv])

# ─── 2) Compute water mask & print counts ─────────────────────────
water   = ds["water_mask"] == 1
n_water = int(water.sum())
n_time  = ds.sizes["time"]
print(f"Water pixels × time steps = {n_water} × {n_time}")

# ─── 3) Load everything into memory (no more Dask) ────────────────
ds = ds.load()

# ─── 4) Variables to impute ───────────────────────────────────────
vars_to_impute = ["chlor_a", "Kd_490", "nflh", "sst", "log_chl"]

# ─── 5) Two‐stage water‐only imputation w/o extrapolation ──────────
for var in vars_to_impute:
    orig = ds_fixed[var]        # original DataArray
    da   = ds[var]

    # a) mask out land → land stays NaN
    da_w = da.where(water)

    # b) linear interpolate only interior gaps
    da_lin = da_w.interpolate_na(dim="time", method="linear")

    # c) forward‐fill & back‐fill the time edges
    da_ff  = da_lin.ffill(dim="time")
    da_fb  = da_ff .bfill(dim="time")

    # d) spatial nearest‐neighbor fill of any residues
    da_s1 = da_fb.interpolate_na(dim="lat", method="nearest")
    da_s2 = da_s1.interpolate_na(dim="lon", method="nearest")

    # e) stitch back: water gets da_s2, land stays NaN
    filled = xr.where(water, da_s2, da)

    # f) clip to original observed range
    vmin, vmax = float(orig.min()), float(orig.max())
    ds[var] = filled.clip(min=vmin, max=vmax)

    print(f"  • `{var}`: filled & clipped to [{vmin:.2f}, {vmax:.2f}]")

# ─── 6) Apply water mask globally (grey‐out land everywhere) ───────
for v in list(ds.data_vars):
    if "lat" in ds[v].dims and "lon" in ds[v].dims and v != "water_mask":
        ds[v] = ds[v].where(water)
print("  • All lat×lon variables masked to water only")

# ─── 7) Create `<var>_obs` channels ────────────────────────────────
for var in vars_to_impute:
    orig = ds_fixed[var]
    obs  = (orig.notnull() & water)
    ds[f"{var}_obs"] = (orig.dims, obs.astype(np.int8).values)
    print(f"  • `{var}_obs` added")

# ─── 8) Save fully‐imputed NetCDF4 compressed ─────────────────────
encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf(imputed_nc, mode="w", format="NETCDF4", encoding=encoding)
print(f"\n✅ Wrote imputed dataset →\n   {imputed_nc}")
