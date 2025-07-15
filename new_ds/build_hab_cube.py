
#!/usr/bin/env python3
"""
build_hab_cube.py
──────────────────────────────────────────────────────────────────────────────
Fuse the three re-gridded 8-day cubes (MODIS, ERA-5, CMEMS) into *one*
HAB master cube on the MODIS-Aqua 4-km grid.

Inputs   (already created by regrid_to_modis.py)
  Data/Finalized/modis_8day_native.nc   ← reference grid & timeline (851 × 240 × 432)
  Data/Finalized/era5_8day_4km.nc
  Data/Finalized/cmems_8day_4km.nc

Outputs  (written to the same folder)
  HAB_master_8day_4km.nc      (NetCDF, compressed)
  qc_snapshots/*.nc           (≈ one frame / quarter)
  # HAB_master_8day_4km.zarr  (chunked Zarr – lines left commented out)
"""

# ─────────────────────────────────────────────────────────────────────────────
import pathlib, numpy as np, xarray as xr
from dask.diagnostics import ProgressBar

ROOT      = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
FIN       = ROOT / "Data" / "Finalized"
OUT_NC    = FIN  / "HAB_master_8day_4km.nc"
OUT_ZARR  = FIN  / "HAB_master_8day_4km.zarr"    # currently unused
QC_DIR    = FIN  / "qc_snapshots";   QC_DIR.mkdir(exist_ok=True)

# ───────── helper ----------------------------------------------------------------
def open_cube(fp):
    """Open with eager decoding & ensure np.datetime64[ns] coordinates."""
    ds = xr.open_dataset(fp, engine="netcdf4", chunks={})
    # xarray decoded `time` already, but cast to ns precision to be safe
    ds["time"] = ds.time.astype("datetime64[ns]")
    return ds

# ───────── load ------------------------------------------------------------------
print("⏳  loading three cubes …")
modis  = open_cube(FIN / "modis_8day_native.nc")
era5   = open_cube(FIN / "era5_8day_4km.nc")
cmems  = open_cube(FIN / "cmems_8day_4km.nc")

# ───────── intersect timeline (they *should* all be 851, but we enforce) ---------
t_common = np.intersect1d(
    np.intersect1d(modis.time.values, era5.time.values),
    cmems.time.values
)
if len(t_common) != 851:
    print(f"⚠︎ common timeline is {len(t_common)} composites (expected 851) – trimming")
modis, era5, cmems = [ds.sel(time=t_common) for ds in (modis, era5, cmems)]

# ───────── merge -----------------------------------------------------------------
master = xr.merge([modis, era5, cmems], compat="override")

# ───────── write NetCDF with live progress-bar -----------------------------------
print(f"→ writing NetCDF  {OUT_NC.name}")
enc = {v: {"zlib": True, "complevel": 4, "chunksizes": (90, 144, 80)}
         for v in master.data_vars}

delayed = master.to_netcdf(OUT_NC, encoding=enc, engine="netcdf4", compute=False)
with ProgressBar():
    delayed.compute()

'''
# ───────── optional Zarr (left disabled) -----------------------------------------
if OUT_ZARR.exists():
    if OUT_ZARR.is_dir():
        shutil.rmtree(OUT_ZARR)          # <- works for directories
    else:
        OUT_ZARR.unlink()                # <- works for plain files
# uncomment if/when you really want the Zarr output
print(f"→ writing Zarr    {OUT_ZARR.name}")
master.chunk({"time": 90, "lat": 128}).to_zarr(OUT_ZARR, mode="w")
'''

# ───────── QC snapshots ----------------------------------------------------------
print("→ QC snapshots")
snap_enc = {v: {"zlib": True, "complevel": 4}   # no chunksizes
            for v in master.data_vars}

for ts in t_common[::46]:                       # every ≈ quarter
    ts_str = np.datetime_as_string(ts, unit="D")
    master.sel(time=ts).to_netcdf(
        QC_DIR / f"snapshot_{ts_str}.nc",
        encoding=snap_enc
    )

print("✅  done – cube dims", dict(master.sizes))
