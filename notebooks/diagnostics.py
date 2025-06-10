
#!/usr/bin/env python
# diagnostics/all_diagnostics.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pathlib
import yaml
import numpy as np
import xarray as xr
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────── setup ─────────────────

repo = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(repo/"config.yaml"))
lag  = cfg["forecast_lag"]
cube_file = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
model_reg  = repo/"Models"/"xgb_chla_reg.joblib"
model_clf  = repo/"Models"/"xgb_hab_clf.joblib"

out_dir = pathlib.Path("/Users/yashnilmohanty/Desktop/habs-all-diagnostics")
out_dir.mkdir(parents=True, exist_ok=True)

# ───────────────── load data & models ─────────────────
ds  = xr.open_dataset(cube_file, chunks={"time":50})
reg = joblib.load(model_reg)
clf = joblib.load(model_clf)

# predictors & target
PRED_VARS = [
    "sst","t2m","u10","v10","avg_sdswrf",
    "Kd_490","nflh","so","thetao","uo","vo","zos"
]
X = ds[PRED_VARS]
y = ds["log_chl"].shift(time=-lag)
X = X.isel(time=slice(None, -lag))
y = y.isel(time=slice(None, -lag))

# ───────────────── build test set ─────────────────
# flatten space‐time → samples
X_stack = (
    X.to_array()
     .transpose("time","lat","lon","variable")
     .stack(sample=("time","lat","lon"))
     .transpose("sample","variable")
)
y_stack = y.stack(sample=("time","lat","lon"))

valid = (~np.isnan(X_stack).any("variable") & np.isfinite(y_stack)).compute()
vals = valid.values

X_vals = X_stack.values[vals].astype("float32")
y_vals = y_stack.values[vals].astype("float32")
times  = X_stack["time"].values[vals]
lats   = X_stack["lat"].values[vals]
lons   = X_stack["lon"].values[vals]

# select 2021 test
mask_test = times >= np.datetime64("2021-01-01")
X_test = X_vals[mask_test]
y_test = y_vals[mask_test]
t_test = times[mask_test]
lat_test = lats[mask_test]
lon_test = lons[mask_test]

# predictions
y_pred = reg.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

# ───────────────── 1. Feature importance ─────────────────
fi = reg.feature_importances_
plt.figure(figsize=(6,4))
plt.barh(PRED_VARS, fi)
plt.xlabel("Gain")
plt.title("Feature Importance (Regressor)")
plt.tight_layout()
plt.savefig(out_dir/"feature_importance.png", dpi=300)
plt.close()

# ───────────────── 2. Pred vs Obs scatter ────────────────
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred, s=1, alpha=0.3)
lims = (min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()))
plt.plot(lims, lims, 'k--', label='1:1 line')
plt.xlabel("True log₁₀(chl­a)")
plt.ylabel("Predicted log₁₀(chl­a)")
plt.title("Predicted vs Observed (2021 Test)")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir/"scatter_pred_vs_obs.png", dpi=300)
plt.close()

# ───────────────── 3. Error distribution ─────────────────
errors = y_pred - y_test
plt.figure(figsize=(6,4))
plt.boxplot(errors, vert=False, showfliers=False)
plt.xlabel("Prediction error (pred – true)")
plt.title("Error Distribution (Boxplot)")
plt.tight_layout()
plt.savefig(out_dir/"error_boxplot.png", dpi=300)
plt.close()

plt.figure(figsize=(6,4))
plt.hist(errors, bins=100, density=True)
plt.xlabel("Prediction error")
plt.title("Error Distribution (Histogram)")
plt.tight_layout()
plt.savefig(out_dir/"error_histogram.png", dpi=300)
plt.close()

# ───────────────── 4. Spatial RMSE map ────────────────
# build a MultiIndex for the sample dimension
mi = pd.MultiIndex.from_arrays(
    [t_test, lat_test, lon_test],
    names=("time", "lat", "lon")
)

# create DataArrays with that MultiIndex
pred_da = xr.DataArray(
    y_pred,
    coords={"sample": mi},
    dims=["sample"]
)
true_da = xr.DataArray(
    y_test,
    coords={"sample": mi},
    dims=["sample"]
)

# now unstack back to 3D cubes
pred_cube = pred_da.unstack("sample")
true_cube = true_da.unstack("sample")

# compute per-pixel RMSE over the 2021 test period
rmse_map = np.sqrt(((pred_cube - true_cube) ** 2).mean(dim="time"))

# plot and save
plt.figure(figsize=(6, 8))
ax = plt.gca()
rmse_map.plot(ax=ax)
plt.title("Per-pixel RMSE (log₁₀ chlorophyll) for 2021 Test")
plt.tight_layout()
plt.savefig(out_dir/"spatial_rmse.png", dpi=300)
plt.close()
