import xarray as xr, numpy as np, pathlib
cube = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Processed/HAB_cube_std0_2016_2021.nc")

ds = xr.open_dataset(cube, chunks={"time": 50})
nan_fraction = ds.isnull().mean("time")      # fraction per pixel

print("Kd_490  pixels that are EVER NaN:", int((nan_fraction["Kd_490"]>0).sum().compute()))
print("Kd_490  pixels NaN on ALL dates :", int((nan_fraction["Kd_490"]==1).sum().compute()))
