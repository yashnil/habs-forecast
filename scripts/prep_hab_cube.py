
#!/usr/bin/env python
# Scripts/prep_hab_cube.py
# -----------------------------------------------------------
# Trim the master dataset to the common 2016-01-09 – 2021-06-17
# window and save as HAB_cube_2016_2021.nc
# -----------------------------------------------------------
import pathlib, xarray as xr, numpy as np, pandas as pd

import pathlib, yaml, xarray as xr, numpy as np, pandas as pd

root = pathlib.Path(__file__).resolve().parents[1]      # repo root
cfg  = yaml.safe_load(open(root / "config.yaml"))

data_dir = pathlib.Path(cfg["data_root"])               # <- external folder
src = data_dir / "root_dataset_filled.nc"
dst = data_dir / "HAB_cube_2016_2021.nc"

print("Opening", src)
ds = xr.open_dataset(src, chunks={"time": 50})

# 0 ▸ confirm original range
print("Original time span:",
      pd.to_datetime(ds.time.values[0]).date(),
      "→",
      pd.to_datetime(ds.time.values[-1]).date(),
      f"({len(ds.time)} steps)")

# 1 ▸ trim to the common window
ds = ds.sel(time=slice("2016-01-09", "2021-06-17"))

# 2 ▸ quick NaN report on chlor_a
chl = ds["chlor_a"]
nan_frac = float(chl.isnull().sum().values) / chl.size
print(f"NaN fraction in chlor_a after trim: {nan_frac:.2%}")

# 3 ▸ add log10 chlorophyll (predictand) for convenience
ds["log_chl"] = np.log10(chl)

# 4 ▸ save
encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
ds.to_netcdf(dst, encoding=encoding)
print("✓ wrote", dst, "with", len(ds.time), "time steps")
