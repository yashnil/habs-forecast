#!/usr/bin/env python3
# scripts/helpers/recompute_curl.py
"""
• NN-fill tiny gaps in uo / vo  (within coastal band)
• recompute curl_uv             (and NN-fill its gaps)
• drop log1p_dist_river everywhere
• keep every other subgroup (masks, …)
• atomic replacement of the cube
"""

from __future__ import annotations
import pathlib, tempfile, shutil
import numpy as np, xarray as xr, netCDF4
from scipy import ndimage

ROOT  = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
CUBE  = ROOT / "Data" / "Finalized" / "HAB_master_8day_4km.nc"

# ────────────────────────────────────────────────────────────────────────────
print("🔹 opening cube …")
root_ds = xr.open_dataset(CUBE)

try:
    mask2d = xr.open_dataset(CUBE, group="masks")["coastal_water_mask"].astype(bool)
except OSError:
    raise RuntimeError(
        "❌  /masks group not found – rebuild it first with "
        "`make_water_mask.py` and rerun this script."
    )

mask3d = mask2d.expand_dims(time=root_ds.time)   # broadcast to (time,lat,lon)

# ── helper ─────────────────────────────────────────────────────────────────
def nn_fill_one(arr2d: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill of NaNs *within* <valid> mask."""
    want = np.logical_and(valid, np.isnan(arr2d))
    if not want.any():
        return arr2d
    _, (iy, ix) = ndimage.distance_transform_edt(
        np.isnan(arr2d), return_indices=True
    )
    out = arr2d.copy()
    out[want] = arr2d[iy[want], ix[want]]
    return out


def nn_fill_da(da: xr.DataArray) -> xr.DataArray:
    """Time-loop NN fill inside coastal mask."""
    filled = [
        xr.DataArray(
            nn_fill_one(da.sel(time=t).values, mask2d.values),
            dims=("lat", "lon"), coords={"lat": da.lat, "lon": da.lon}
        ).assign_coords(time=t)
        for t in da.time
    ]
    return xr.concat(filled, dim="time")


print("🔹 filling uo / vo …")
uo_f = nn_fill_da(root_ds["uo"])
vo_f = nn_fill_da(root_ds["vo"])

# ── curl ───────────────────────────────────────────────────────────────────
R        = 6_371_000.0
deg2m    = np.pi / 180.0 * R
dx_m     = deg2m * np.cos(np.deg2rad(root_ds.lat))
dy_m     = deg2m

curl = (vo_f.differentiate("lon") / dx_m
        - uo_f.differentiate("lat") / dy_m).where(mask3d)

curl_f = nn_fill_da(curl).astype("float32")
print("   NaNs still inside band →", int(np.isnan(curl_f.where(mask3d)).sum()))

# ── fresh /derivatives (without log1p_dist_river) ──────────────────────────
deriv_ds = xr.open_dataset(CUBE, group="derivatives") \
             .drop_vars("log1p_dist_river", errors="ignore")
deriv_ds["curl_uv"] = curl_f

# ── atomic rewrite ─────────────────────────────────────────────────────────
print("🔹 writing new NetCDF (atomic replace)…")
with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
    TMP = pathlib.Path(tmp.name)

enc_root = {v: {"zlib": True, "complevel": 4,
                "chunksizes": (90, 144, 80)}
            for v in root_ds.data_vars}
root_ds.to_netcdf(TMP, mode="w", engine="netcdf4", encoding=enc_root)

# copy all groups except derivatives (we add it next)
with netCDF4.Dataset(CUBE, "r") as src, netCDF4.Dataset(TMP, "a") as dst:
    for gname, src_grp in src.groups.items():
        if gname == "derivatives":
            continue
        dst_grp = dst.createGroup(gname)
        for vname, var in src_grp.variables.items():
            v_out = dst_grp.createVariable(
                vname, var.datatype, var.dimensions,
                zlib=var.filters().get("zlib", False),
                complevel=var.filters().get("complevel", 0),
                chunksizes=var.chunking() if var.filters() else None
            )
            v_out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
            v_out[:] = var[:]

# append updated derivatives
enc_deriv = {v: {"zlib": True, "complevel": 4,
                 "chunksizes": (90, 144, 80)}
             for v in deriv_ds.data_vars}
deriv_ds.to_netcdf(TMP, mode="a", group="derivatives",
                   engine="netcdf4", encoding=enc_deriv)

root_ds.close(); deriv_ds.close()
shutil.move(TMP, CUBE)

print("✅  curl_uv updated, log1p_dist_river removed, masks preserved\n")
