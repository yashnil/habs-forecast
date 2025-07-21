# quick_check.py
import numpy as np
import xarray as xr

path = "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc"
ds = xr.open_dataset(path)

print("Kd_490 max        :", float(ds.Kd_490.max()))
print("Kd_490 0.999 quant:", float(ds.Kd_490.quantile(0.999)))
