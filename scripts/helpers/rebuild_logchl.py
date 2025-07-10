#!/usr/bin/env python3
"""
Recompute log10(chlor_a) in the raw cube,
treating chlor_a<=0 as NaN, and overwrite log_chl.
"""
import pathlib, yaml
import numpy as np, xarray as xr

root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root/"config.yaml"))

src = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
with xr.open_dataset(src, mode="a") as ds:
    # compute log10, masking zeros/negatives
    chlor = ds["chlor_a"]
    log10chl = xr.where(chlor>0, np.log10(chlor), np.nan)
    log10chl.name = "log_chl"
    log10chl.attrs.update({
        "long_name": "log10 chlorophyll-a",
        "units": "log10(mg m⁻³)"
    })

    # write back into the file (root group)
    log10chl.to_netcdf(
        src, mode="a", group=None,
        encoding={"log_chl": {"zlib":True, "complevel":4}}
    )
print("✓ Rebuilt log_chl (no more -inf) in HAB_cube_2016_2021.nc")
