
import xarray as xr
import numpy as np
import joblib
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# Load config and paths
repo_root = Path().resolve()
cfg = yaml.safe_load(open(repo_root/"config.yaml"))
lag = cfg["forecast_lag"]
cube_file = Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"

# Load dataset and model
ds = xr.open_dataset(cube_file, chunks={"time": 50})
reg = joblib.load(repo_root/"Models"/"xgb_chla_reg.joblib")

# Prepare predictors and target
pred_vars = ["sst","t2m","u10","v10","avg_sdswrf","Kd_490","nflh","so","thetao","uo","vo","zos"]
X = ds[pred_vars]
y = ds["log_chl"].shift(time=-lag)
X = X.isel(time=slice(None, -lag))
y = y.isel(time=slice(None, -lag))

# Stack space-time
X_stack = (X.to_array()
            .transpose("time","lat","lon","variable")
            .stack(sample=("time","lat","lon"))
            .transpose("sample","variable"))
y_stack = y.stack(sample=("time","lat","lon"))

# Mask valid samples
valid = (~np.isnan(X_stack).any("variable") & np.isfinite(y_stack)).compute()
mask = valid.values

# Extract arrays
X_vals = X_stack.values[mask].astype("float32")
y_vals = y_stack.values[mask].astype("float32")
times  = X_stack["time"].values[mask]
lats   = X_stack["lat"].values[mask]
lons   = X_stack["lon"].values[mask]

# Select 2021 test samples
test_mask = times >= np.datetime64("2021-01-01")
X_test = X_vals[test_mask]
y_test = y_vals[test_mask]
times_test = times[test_mask]
lats_test = lats[test_mask]
lons_test = lons[test_mask]

# Predictions
y_pred = reg.predict(X_test)

# Build DataArrays with MultiIndex for unstacking
import pandas as pd
mi = pd.MultiIndex.from_arrays([times_test, lats_test, lons_test], names=("time","lat","lon"))
pred_da = xr.DataArray(y_pred, coords={"sample": mi}, dims=["sample"])
true_da = xr.DataArray(y_test, coords={"sample": mi}, dims=["sample"])
pred_cube = pred_da.unstack("sample")
true_cube = true_da.unstack("sample")

# Compute maps
r2_map = 1 - ((pred_cube - true_cube)**2).sum("time") / ((true_cube - true_cube.mean("time"))**2).sum("time")
mean_true = true_cube.mean("time")
mean_pred = pred_cube.mean("time")

# Plot Observed
plt.figure(figsize=(6,8))
ax = plt.gca()
mean_true.plot(ax=ax)
ax.set_title("Mean Observed log₁₀(chlor_a), 2021")
plt.show()
plt.savefig(Path("/Users/yashnilmohanty/Desktop/habs-all-diagnostics")/"observed_chlor.png", dpi=300)

# Plot Predicted
plt.figure(figsize=(6,8))
ax = plt.gca()
mean_pred.plot(ax=ax)
ax.set_title("Mean Predicted log₁₀(chlor_a), 2021")
plt.show()
plt.savefig(Path("/Users/yashnilmohanty/Desktop/habs-all-diagnostics")/"predicted_chlor.png", dpi=300)

# Plot R² map
plt.figure(figsize=(6,8))
ax = plt.gca()
r2_map.plot(ax=ax, vmin=0, vmax=1)
ax.set_title("Per-pixel R², 2021 Test")
plt.show()
plt.savefig(Path("/Users/yashnilmohanty/Desktop/habs-all-diagnostics")/"r2_map.png", dpi=300)