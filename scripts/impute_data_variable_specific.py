
#!/usr/bin/env python3
import xarray as xr
import numpy as np
from scipy.ndimage import generic_filter

# ─── Paths ─────────────────────────────────────────────────────────
fixed_nc     = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_2016_2021_fixed.nc"
orig_nc      = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_2016_2021.nc"
out_nc       = "/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_2016_2021_varspec_imputed.nc"

# ─── Helper: local neighbor‐mean fill ──────────────────────────────
def local_mean_fill(arr, iterations=3):
    """
    Iteratively fill NaNs in `arr` (2D or 3D with time first) 
    by replacing each NaN with the mean of its 3×3 spatial neighbors.
    """
    def neigh_mean(window):
        # window is flattened 3×3 patch
        m = window.reshape((3,3))
        return np.nanmean(m)
    filled = arr.copy()
    for _ in range(iterations):
        # apply only on the spatial dims; leaves edges as-is
        if filled.ndim == 3:
            # time,lat,lon
            for t in range(filled.shape[0]):
                filled[t] = generic_filter(
                    filled[t],
                    function=neigh_mean,
                    size=3,
                    mode="constant",
                    cval=np.nan
                )
        elif filled.ndim == 2:
            filled = generic_filter(
                filled,
                function=neigh_mean,
                size=3,
                mode="constant",
                cval=np.nan
            )
    return filled

# ─── 1) Load fixed + derivatives ──────────────────────────────────
ds_fixed = xr.open_dataset(fixed_nc)
ds_deriv = xr.open_dataset(orig_nc, group="derivatives")
ds       = xr.merge([ds_fixed, ds_deriv])

# ─── 2) Water mask & counts ──────────────────────────────────────
water   = ds["water_mask"] == 1
n_water = int(water.sum()); n_time = ds.sizes["time"]
print(f"Water pixels × time steps = {n_water} × {n_time}")

# ─── 3) Bring into memory ─────────────────────────────────────────
ds = ds.load()

# ─── 4) Vars & processing rules ──────────────────────────────────
vars_to_impute = ["chlor_a", "Kd_490", "nflh", "sst", "log_chl"]

for var in vars_to_impute:
    orig = ds_fixed[var]     # original measurements
    da   = ds[var]

    # mask land
    da_w = da.where(water)

    # temporal interp (linear) + ffill/bfill on time ends
    da_lin = da_w.interpolate_na(dim="time", method="linear")
    da_ff  = da_lin.ffill(dim="time")
    da_fb  = da_ff .bfill(dim="time")

    # spatial fill: neighbor‐mean for chlor_a, Kd_490, nflh, log_chl
    if var != "sst":
        arr      = da_fb.values             # shape (time,lat,lon)
        arr_filt = local_mean_fill(arr, iterations=3)
        da_spatial = xr.DataArray(
            arr_filt, dims=da_fb.dims, coords=da_fb.coords
        )
    else:
        # for SST, skip spatial fill entirely
        da_spatial = da_fb

    # stitch back: water→imputed, land→NaN
    filled = xr.where(water, da_spatial, da)

    # clip to real observed range
    vmin, vmax = float(orig.min()), float(orig.max())
    ds[var] = filled.clip(min=vmin, max=vmax)

    print(f"  • `{var}` done; clipped to [{vmin:.2f}, {vmax:.2f}]")

# ─── 5) Grey‐out land on every grid var ───────────────────────────
for v in ds.data_vars:
    if v!="water_mask" and "lat" in ds[v].dims and "lon" in ds[v].dims:
        ds[v] = ds[v].where(water)
print("  • Land pixels greyed out on all lat×lon vars")

# ─── 6) Create `<var>_obs` flags ─────────────────────────────────
for var in vars_to_impute:
    orig = ds_fixed[var]
    obs  = (orig.notnull() & water)
    ds[f"{var}_obs"] = (orig.dims, obs.astype(np.int8).values)
    print(f"  • `{var}_obs` flag added")

# ─── 7) Write new NetCDF ─────────────────────────────────────────
encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf(out_nc, mode="w", format="NETCDF4", encoding=encoding)
print(f"\n✅ Wrote variable‐specific imputed cube →\n   {out_nc}")
