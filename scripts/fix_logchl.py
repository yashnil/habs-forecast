
#!/usr/bin/env python3
"""
Add (or overwrite) log10-chlorophyll in the *raw* cube.

→ group=None, var name = 'log_chl'
"""
import pathlib, yaml, xarray as xr, numpy as np

root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))

raw = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"

with xr.open_dataset(raw, mode="r+") as ds:        # open in append mode
    chl = ds["chlor_a"].clip(min=1e-6)             # avoid log(0)
    log = np.log10(chl)
    log.attrs.update(units="log10(µg/L)", long_name="log10 chlor_a")
    ds["log_chl"] = log.astype("float32")          # overwrites if exists
print("✓ log_chl written to", raw)
