
#!/usr/bin/env python3
# scripts/make_std0_cube.py
"""
Create a *new* NetCDF cube where all coastal pixels are z-scored
and NaNs are filled with 0 (mean after standardisation).

Output:  {data_root}/HAB_cube_std0_2016_2021.nc
"""

from __future__ import annotations
import pathlib, yaml, numpy as np, xarray as xr

# ── paths ─────────────────────────────────────────────────────────
root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))

src_cube  = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
dst_cube  = pathlib.Path(cfg["data_root"]) / "HAB_cube_std0_2016_2021.nc"

print("▶ loading source cube lazily …")
ds_root  = xr.open_dataset(src_cube, chunks={"time": 50})
try:
    ds_deriv = xr.open_dataset(src_cube, group="derivatives",
                               chunks={"time": 50})
except FileNotFoundError:
    ds_deriv = xr.Dataset()          # no derivatives yet -> empty

# ── coastal mask : thetao finite in *any* composite ───────────────
coast = ~ds_root["thetao"].isnull().all("time")     # dims (lat,lon)
print("   coastal pixels :", int(coast.sum()), "/", coast.size)

# ── helper to z-score + nan→0 inside coast ───────────────────────
def _std0(da: xr.DataArray, name: str) -> xr.DataArray:
    μ = da.where(coast).mean(("time", "lat", "lon"), skipna=True)
    σ = da.where(coast).std (("time", "lat", "lon"), skipna=True)
    σ = xr.where(σ == 0, 1.0, σ)     # protect constant vars
    zz = (da - μ) / σ
    zz = xr.where(coast, zz.fillna(0.0), np.nan)
    zz.attrs.update({"long_name": f"std-0 {name}", "units": "σ"})
    return zz.astype("float32")

# ── standardise every data-var in both groups ────────────────────
print("▶ standardising & zero-filling …")
proc_root  = xr.Dataset({v: _std0(ds_root[v], v)  for v in ds_root.data_vars},
                        coords=ds_root.coords)

proc_deriv = xr.Dataset({v: _std0(ds_deriv[v], v) for v in ds_deriv.data_vars},
                        coords=ds_deriv.coords) if ds_deriv else None

# ── write new NetCDF (root + derivatives group) ──────────────────
enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"}
       for v in proc_root.data_vars}
print("▶ writing", dst_cube.name)
proc_root.to_netcdf(dst_cube, mode="w", encoding=enc)

if proc_deriv:                       # append derivatives group
    enc_d = {v: enc["sst"] for v in proc_deriv.data_vars}  # same codec
    proc_deriv.to_netcdf(dst_cube, mode="a", group="derivatives",
                         encoding=enc_d)

print("✓ new cube saved at", dst_cube)
