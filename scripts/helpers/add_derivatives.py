#!/usr/bin/env python3
"""
add_derivatives.py
──────────────────────────────────────────────────────────────────────────────
Append extra predictors to HAB_master_8day_4km.nc

* ROOT group
    – log_chl               natural-log chlorophyll (floor at min positive)

* Group /derivatives
    – curl_uv               ∂v/∂x – ∂u/∂y              [s-1]
    – dist_river_km         distance to nearest river [km]
    – log1p_dist_river      log(1 + distance)         [log-km]
"""

# ── std / third-party ───────────────────────────────────────────────────────
import pathlib, numpy as np, xarray as xr, netCDF4
from   scipy.ndimage import distance_transform_edt

# ── paths ───────────────────────────────────────────────────────────────────
ROOT   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
CUBE   = ROOT / "Data" / "Finalized" / "HAB_master_8day_4km.nc"

# ── constants ───────────────────────────────────────────────────────────────
R_EARTH = 6_371_000            # m
DEG2M   = np.pi / 180 * R_EARTH

RIVER_MOUTHS = {               # lat , lon  (°)
    "sacramento": (38.06325 , -121.85274),
    "columbia"  : (46.246922, -124.09344),
}

# ════════════════════════════════════════════════════════════════════════════
# 1)  open cube (read-only) & pull needed fields into memory
# ════════════════════════════════════════════════════════════════════════════
print("🔹 opening cube read-only …")
with xr.open_dataset(CUBE, chunks={"time": 90}) as ds:

    lat, lon = ds["lat"], ds["lon"]

    # ── log_chl (predictand) ───────────────────────────────────────────────
    chl      = ds["chlor_a"].load()                         # bring into RAM
    chl_min  = float(chl.where(chl > 0).min())
    log_chl  = np.log(np.maximum(chl, chl_min)).astype("float32")
    log_chl_da = xr.DataArray(
        log_chl, dims=chl.dims, coords=chl.coords,
        name="log_chl",
        attrs=dict(long_name="natural-log chlorophyll-a",
                   note=f"log(max(chlor_a, {chl_min:.4g}))",
                   units="ln(mg m-3)")
    )
    print(f"   • log_chl computed   (floor = {chl_min:.4g})")

    # ── curl of surface currents (uo, vo) ──────────────────────────────────
    dlat, dlon = float(lat.diff("lat").mean()), float(lon.diff("lon").mean())
    dx_m = dlon * DEG2M * np.cos(np.deg2rad(lat))          # DataArray (lat)
    dx_m = xr.DataArray(dx_m, dims="lat", coords={"lat": lat})  # ensure broadcasting
    dy_m = dlat * DEG2M

    du_dy = ds["uo"].differentiate("lat") / dy_m
    dv_dx = ds["vo"].differentiate("lon") / dx_m
    curl  = (dv_dx - du_dy).load().astype("float32").rename("curl_uv")
    curl.attrs.update(units="s-1", long_name="surface current curl")
    print("   • curl_uv computed")


    # ── distance to nearest river mouth ────────────────────────────────────
    river_mask = xr.zeros_like(ds["chlor_a"].isel(time=0, drop=True), dtype=bool)
    for lat_pt, lon_pt in RIVER_MOUTHS.values():
        river_mask.loc[
            dict(
                lat=lat.sel(lat=lat_pt, method="nearest"),
                lon=lon.sel(lon=lon_pt, method="nearest"),
            )
        ] = True

    ocean     = ~ds["chlor_a"].isel(time=0).isnull()
    land_mask = ~(river_mask & ocean)
    dist_px   = xr.apply_ufunc(
        distance_transform_edt, land_mask,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
    )

    cell_km      = dy_m / 1_000
    dist_km_2d   = (dist_px * cell_km).astype("float32")
    dist_km_3d   = dist_km_2d.broadcast_like(ds["chlor_a"]).load()

    dist_da      = xr.DataArray(
        dist_km_3d, dims=ds["chlor_a"].dims, coords=ds["chlor_a"].coords,
        name="dist_river_km",
        attrs=dict(units="km", long_name="distance to nearest river mouth"),
    )
    log1p_da     = np.log1p(dist_da).astype("float32").rename("log1p_dist_river")
    log1p_da.attrs.update(units="log-km",
                          long_name="log(1 + distance to river mouth)")

    print("   • dist_river_km & log1p_dist_river computed")

# ds is now closed; all new variables live entirely in memory
# ════════════════════════════════════════════════════════════════════════════
# 2)  append variables safely with netCDF4 (single write handle)
# ════════════════════════════════════════════════════════════════════════════
print("🔸 writing variables back to cube …")
with netCDF4.Dataset(CUBE, mode="a") as nc:

    # ── root: log_chl ──────────────────────────────────────────────────────
    if "log_chl" not in nc.variables:
        v = nc.createVariable("log_chl", "f4",
                              dimensions=("time", "lat", "lon"),
                              zlib=True, complevel=4)
        v.setncatts(log_chl_da.attrs)
    nc.variables["log_chl"][:] = log_chl_da.values

    # ── /derivatives group ────────────────────────────────────────────────
    if "derivatives" not in nc.groups:
        der_grp = nc.createGroup("derivatives")
    else:
        der_grp = nc.groups["derivatives"]

    def _write(name, da):
        if name not in der_grp.variables:
            v = der_grp.createVariable(name, "f4",
                                       dimensions=("time", "lat", "lon"),
                                       zlib=True, complevel=4,
                                       chunksizes=(90, 144, 80))
            v.setncatts(da.attrs)
        der_grp.variables[name][:] = da.values

    _write("curl_uv",          curl)
    _write("dist_river_km",    dist_da)
    _write("log1p_dist_river", log1p_da)

print("✅  All derivatives appended to", CUBE.name)
