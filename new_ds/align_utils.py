"""
Common helpers:   to_datetime, resample_8day, regrid_to_target
"""
import numpy as np, xarray as xr, pandas as pd, pathlib, warnings

# -------------------------------------------------------------------
GRID_FILE = pathlib.Path(__file__).with_name("modis_4km_grid.npz")
if not GRID_FILE.is_file():
    raise FileNotFoundError(f"{GRID_FILE} missing – run 00_make_target_grid.py first")

g = np.load(GRID_FILE)
TGT = xr.Dataset({"lon": (("y", "x"), g["lons"]),
                  "lat": (("y", "x"), g["lats"])})

# -------------------------------------------------------------------
def to_datetime(ds, time_dim):
    ds[time_dim] = pd.to_datetime(ds[time_dim].values, unit="s")
    return ds

def resample_8day(da):
    return (da
            .resample(time="8D", label="left", closed="left")
            .mean())

# -------------------------------------------------------------------
def regrid_to_target(da, method="bilinear"):
    """
    Regrid (time,lat,lon) DataArray onto the 279×502 target grid (y,x).
    • Uses xesmf/esmpy if available.
    • Falls back to xarray.interp if not.
    """
    # normalise coordinate names
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.dims:
        da = da.rename({"longitude": "lon"})

    try:
        import xesmf as xe
        rg = xe.Regridder(da, TGT, method=method)
        da_i = rg(da)
    except (ImportError, ModuleNotFoundError):
        warnings.warn("xesmf / esmpy not available – using xarray.interp")
        da_i = da.interp(lat=TGT.lat, lon=TGT.lon, method="linear")

    # If the result still has lat/lon dims, convert them to y/x
    if {"lat", "lon"}.issubset(da_i.dims):
        da_i = da_i.swap_dims({"lat": "y", "lon": "x"})

    return da_i