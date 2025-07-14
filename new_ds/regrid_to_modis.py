#!/usr/bin/env python3
"""
regrid_to_modis.py
──────────────────
Interpolate ERA-5 (0.25°) and CMEMS (0.083°) 8-day cubes to the
MODIS-Aqua 4 km grid produced by build_source_cubes.py.

Outputs (compressed NetCDF):
  era5_8day_4km.nc
  cmems_8day_4km.nc
"""

# ── imports ──────────────────────────────────────────────────────────────
import pathlib, xarray as xr, numpy as np, xesmf as xe

# ── fixed paths ──────────────────────────────────────────────────────────
ROOT     = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
FIN      = ROOT / "Data" / "Finalized"          # native cubes live here
PROCESSED= ROOT / "Processed"                   # keep weight files here
PROCESSED.mkdir(exist_ok=True)

FP_MODIS = FIN / "modis_8day_native.nc"
FP_ERA   = FIN / "era5_8day_native.nc"
FP_CMEMS = FIN / "cmems_8day_native.nc"

OUT_ERA  = FIN / "era5_8day_4km.nc"
OUT_CMEMS= FIN / "cmems_8day_4km.nc"

# ── helper: load target grid once ────────────────────────────────────────
def load_modis_grid():
    ds = xr.open_dataset(FP_MODIS, engine="netcdf4", chunks={})
    # 1-D lat / lon are fine for xESMF target description
    return xr.Dataset({"lat": ("lat", ds.lat.values),
                       "lon": ("lon", ds.lon.values)})

TGT_GRID = load_modis_grid()

# ── helper: build / reuse an xESMF regridder ─────────────────────────────
def make_regridder(src, wfile):
    """
    Return a bilinear xESMF regridder from *src* grid → MODIS grid,
    using cached weights if possible.
    """
    wfile = pathlib.Path(wfile)
    try:
        return xe.Regridder(src, TGT_GRID, "bilinear",
                            filename=str(wfile), reuse_weights=wfile.exists())
    except ValueError:
        # cached matrix incompatible with new shapes → rebuild
        wfile.unlink(missing_ok=True)
        return xe.Regridder(src, TGT_GRID, "bilinear",
                            filename=str(wfile), reuse_weights=False)

# ── common post-processing after regridding ──────────────────────────────
def clean_coords(ds_rg):
    """drop 2-D helper coords, keep neat 1-D lat/lon/time only."""
    aux = [c for c in ds_rg.coords
           if c not in ("lat", "lon", "time") and ds_rg[c].ndim > 1]
    if aux:
        ds_rg = ds_rg.reset_coords(aux, drop=True)
    # overwrite lat/lon with the exact 1-D MODIS vectors
    ds_rg = ds_rg.assign_coords(lat=("lat", TGT_GRID.lat.values),
                                lon=("lon", TGT_GRID.lon.values))
    return ds_rg

# ── ERA-5 → MODIS grid ──────────────────────────────────────────────────
def regrid_era5():
    if OUT_ERA.exists():
        print("✔︎", OUT_ERA.name, "already exists – skip")
        return
    print("⏳  re-gridding ERA-5 …")

    ds = xr.open_dataset(FP_ERA, engine="netcdf4",
                         chunks={"time": 64})          # lazy loading

    # rename & sort lat so xESMF gets ascending coords
    ds = (ds.rename({"latitude": "lat", "longitude": "lon"})
            .sortby("lat"))

    # ‘number’ has no lat/lon dims – drop before interpolation
    ds_drop = ds.drop_vars("number", errors="ignore")

    rg = make_regridder(ds_drop, PROCESSED / "weights_era5_to_modis.nc")
    ds_rg = rg(ds_drop)                    # interpolate every var at once
    ds_rg = clean_coords(ds_rg)

    # add the realisation index back (unchanged) if wanted
    if "number" in ds:
        ds_rg = xr.merge([ds_rg, ds[["number"]]])

    comp = {v: {"zlib": True, "complevel": 4} for v in ds_rg.data_vars}
    ds_rg.to_netcdf(OUT_ERA, encoding=comp)
    print("✅  wrote", OUT_ERA.name)

# ── CMEMS → MODIS grid ──────────────────────────────────────────────────
def regrid_cmems():
    if OUT_CMEMS.exists():
        print("✔︎", OUT_CMEMS.name, "already exists – skip");  return
    print("⏳  re-gridding CMEMS …")

    ds = xr.open_dataset(FP_CMEMS, engine="netcdf4", chunks={"time": 64})

    # only rename if the long names are present
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items()
           if k in ds.dims or k in ds.coords}
    if ren:
        ds = ds.rename(ren)

    rg     = make_regridder(ds, PROCESSED / "weights_cmems_to_modis.nc")
    ds_rg  = clean_coords(rg(ds))

    comp   = {v: {"zlib": True, "complevel": 4} for v in ds_rg.data_vars}
    ds_rg.to_netcdf(OUT_CMEMS, encoding=comp)
    print("✅  wrote", OUT_CMEMS.name)

# ── main ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    regrid_era5()
    regrid_cmems()
