#!/usr/bin/env python3
# scripts/helpers/make_water_mask.py
"""
Append two masks to HAB_master_8day_4km.nc

 /masks/water_mask          1 = pixel ever ocean (any layer non‑NaN)
 /masks/coastal_water_mask  1 = ocean pixel ≤ 16 km from land
"""
from __future__ import annotations
import pathlib, numpy as np, xarray as xr, netCDF4
from  scipy.ndimage import distance_transform_edt

# ── config ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
CUBE = ROOT / "Data" / "Finalized" / "HAB_master_8day_4km.nc"

MAX_DIST_KM = 16                    # ≈10 mi (4 MODIS cells)
R_EARTH     = 6_371_000
DEG2M       = np.pi / 180 * R_EARTH

print("▶ building water / coastal‑water masks …")
with xr.open_dataset(CUBE, chunks={"time": 90}) as ds:
    lat, lon = ds["lat"], ds["lon"]

    # 1) ever‑water mask
    water_mask = (
        (~ds["chlor_a"].isnull()).any("time") &
        (~ds["thetao"  ].isnull()).any("time")
    ).load()                                           # (lat,lon) bool

    # 2) distance‑to‑land in pixels – run EDT once on the full NumPy array
    #    SciPy’s EDT returns the distance of EVERY pixel to the nearest
    #    **zero** pixel.  We want “distance from water → land”, so:
    #        • water = 1  (foreground)
    #        • land  = 0  (background)
    dist_px_np = distance_transform_edt(water_mask.values.astype("uint8"))
    dist_px    = xr.DataArray(dist_px_np,
                              coords=water_mask.coords,
                              dims=water_mask.dims)

    # 3) px → km  (mean grid‑cell ≈4.6 km)
    dlat  = float(abs(lat.diff("lat").mean()))      # ensure +ve
    dlon  = float(abs(lon.diff("lon").mean()))
    dy_km = dlat * DEG2M / 1_000
    dx_km = (dlon * DEG2M * np.cos(np.deg2rad(lat))).mean() / 1_000
    cell_km = (dx_km + dy_km) / 2.0

    dist_km = (dist_px * cell_km).astype("float32")

    coastal_water_mask = (water_mask & (dist_km <= MAX_DIST_KM)).astype("int8")

    # ── diagnostics ────────────────────────────────────────────────────
    sea_px   = int(water_mask.sum())
    coast_px = int(coastal_water_mask.sum())
    print(f"   water pixels        : {sea_px:,}")
    print(f"   coastal ≤{MAX_DIST_KM} km : {coast_px:,}  "
          f"({coast_px / sea_px * 100:.1f} %)")
    print(f"   max distance in domain: {dist_km.max().item():.1f} km")

    masks_ds = xr.Dataset(
        {"water_mask": water_mask.astype("int8"),
         "coastal_water_mask": coastal_water_mask}
    )
    masks_ds["water_mask"].attrs.update(
        long_name="ocean pixel ever valid", flag_values=[0, 1])
    masks_ds["coastal_water_mask"].attrs.update(
        long_name=f"ocean pixel ≤ {MAX_DIST_KM} km from land",
        flag_values=[0, 1])

print("✚ writing to cube …")
with netCDF4.Dataset(CUBE, "a") as nc:
    grp = nc.groups.get("masks") or nc.createGroup("masks")

    for name in ("water_mask", "coastal_water_mask"):
        da = masks_ds[name]
        if name not in grp.variables:
            var = grp.createVariable(name, "i1", ("lat", "lon"),
                                     zlib=True, complevel=4,
                                     chunksizes=(144, 80))
            var.setncatts(da.attrs)
        grp.variables[name][:] = da.values

print("✅  masks appended – done!")
