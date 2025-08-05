import xarray as xr
ds = xr.open_dataset("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
print(ds.data_vars.keys())
print(ds.coords['time'][0].values, "→", ds.coords['time'][-1].values)
