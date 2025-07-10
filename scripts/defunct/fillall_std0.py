# scripts/fillall_std0.py   ← overwrite the previous version
#!/usr/bin/env python3
"""Replace all NaNs with 0 and drop _FillValue attrs (land & ocean)."""
import pathlib, yaml, xarray as xr, numpy as np, shutil

root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))

src = pathlib.Path(cfg["data_root"]) / "HAB_cube_std0_2016_2021.nc"
dst = src.with_name("HAB_cube_std0F_2016_2021.nc")   # F = filled-zeros

print("• loading source lazily …")
with xr.open_dataset(src, chunks={"time": 50}) as ds_root, \
     xr.open_dataset(src, group="derivatives",
                     chunks={"time": 50}) as ds_deriv:

    filled_root  = ds_root.fillna(0).copy(deep=False)
    filled_deriv = ds_deriv.fillna(0).copy(deep=False)

    # —— strip _FillValue so 0 is not re-interpreted as NaN ——
    for ds in (filled_root, filled_deriv):
        for v in ds.data_vars:
            ds[v].encoding.pop('_FillValue', None)
            ds[v].attrs.pop('_FillValue', None)

    tmp = dst.with_suffix(".tmp.nc")
    print("• writing zero-filled cube … (this may take a minute)")
    filled_root.to_netcdf(tmp, mode="w", engine="netcdf4",
                          encoding={v: {"zlib": True, "complevel": 4}
                                    for v in filled_root.data_vars})
    filled_deriv.to_netcdf(tmp, mode="a", group="derivatives",
                           engine="netcdf4",
                           encoding={v: {"zlib": True, "complevel": 4}
                                     for v in filled_deriv.data_vars})

shutil.move(tmp, dst)
print(f"✓ saved → {dst}")

# ---- quick sanity: count NaNs -----------------------------------
with xr.open_dataset(dst) as test_root, \
     xr.open_dataset(dst, group="derivatives") as test_deriv:
    n_nans = (test_root.isnull().to_array().sum() +
              test_deriv.isnull().to_array().sum()).item()
print("NaNs after rewrite :", int(n_nans))
