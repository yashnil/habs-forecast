#!/usr/bin/env python3
"""
Compute extra physical predictors and append them to HAB_cube_2016_2021.nc

Adds under group = “derivatives”
  • curl_uv           ∂v/∂x – ∂u/∂y        [s⁻¹]
  • dist_river_km     distance to nearest major river mouth  [km]
  • log1p_dist_river  log-scaled version                     [log-km]
"""
import pathlib, yaml, xarray as xr, numpy as np
from   scipy.ndimage import distance_transform_edt

# ── paths ─────────────────────────────────────────────────────────
root  = pathlib.Path(__file__).resolve().parents[1]
cfg   = yaml.safe_load(open(root / "config.yaml"))
cube  = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"

# ── open cube (auto-close after with-block) ───────────────────────
print("Opening cube …")
with xr.open_dataset(cube, chunks={"time": 50}) as ds:

    # -------------------------------------------------- #
    # 1. Curl of surface currents (uo, vo)               #
    # -------------------------------------------------- #
    R_earth = 6_371_000        # m
    deg2m   = np.pi/180 * R_earth
    dx = ds.lon.diff('lon').median().values * deg2m * np.cos(np.deg2rad(ds.lat))
    dy = ds.lat.diff('lat').median().values * deg2m

    du_dy = ds['uo'].differentiate('lat') / dy
    dv_dx = ds['vo'].differentiate('lon') / dx
    curl  = (dv_dx - du_dy).rename("curl_uv")          # dims (time,lat,lon)

    # -------------------------------------------------- #
    # 2. Distance to river mouths                        #
    # -------------------------------------------------- #
    river_pts = {                  # (lat ,  lon)
        "sacramento": (38.00 , -121.80),
        "columbia"  : (46.25 , -124.05),
    }

    river_mask = xr.zeros_like(ds['chlor_a'].isel(time=0), dtype=bool)
    # mark nearest grid-cells
    for lat_pt, lon_pt in river_pts.values():
        lat_near = ds.lat.sel(lat=lat_pt , method="nearest").item()
        lon_near = ds.lon.sel(lon=lon_pt, method="nearest").item()
        river_mask.loc[dict(lat=lat_near, lon=lon_near)] = True

    ocean   = ~ds['chlor_a'].isel(time=0).isnull()
    landsea = ~(river_mask & ocean)                    # True = background
    dist_px = xr.apply_ufunc(
                 distance_transform_edt, landsea,
                 input_core_dims =[('lat','lon')],
                 output_core_dims=[('lat','lon')],
                 dask="allowed"
             )

    cell_km = float(dx.mean()/1_000)                   # km per grid-cell
    dist_km = (dist_px * cell_km).broadcast_like(ds['chlor_a'])
    dist_km  = dist_km.rename("dist_river_km")
    dist_log = np.log1p(dist_km).rename("log1p_dist_river")

    # -------------------------------------------------- #
    # 3. Assemble derivatives Dataset                    #
    # -------------------------------------------------- #
    deriv_ds = xr.Dataset(
        {"curl_uv": curl,
         "dist_river_km": dist_km,
         "log1p_dist_river": dist_log}
    )

#  ── file is closed here; now append safely ───────────────────────
deriv_ds.to_netcdf(
    cube,
    group   = "derivatives",
    mode    = "a",                  # append
    engine  = "netcdf4",
    encoding={v: {"zlib": True, "complevel": 4}    # small file size
              for v in deriv_ds.data_vars},
)
print("✓ derivatives written to cube →", cube)
