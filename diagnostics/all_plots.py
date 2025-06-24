
#!/usr/bin/env python
import pathlib, yaml, xarray as xr, numpy as np, joblib, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ─── CONFIG ────────────────────────────────────────────────
repo = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(repo/"config.yaml"))
data_root = pathlib.Path(cfg["data_root"])
cube_file = data_root / "HAB_cube_2016_2021.nc"
model_file= repo/"Models"/"xgb_chla_reg.joblib"
out_dir   = pathlib.Path("/Users/yashnilmohanty/Desktop/habs-all-diagnostics")
out_dir.mkdir(parents=True, exist_ok=True)

# ─── LOAD ─────────────────────────────────────────────────
ds  = xr.open_dataset(cube_file, chunks={"time":50})
reg = joblib.load(model_file)

# ─── PREP TEST SET ────────────────────────────────────────
lag = cfg["forecast_lag"]
PVAR = ["sst","t2m","u10","v10","avg_sdswrf",
        "Kd_490","nflh","so","thetao","uo","vo","zos"]

X = ds[PVAR]
y = ds["log_chl"].shift(time=-lag)
X = X.isel(time=slice(None,-lag))
y = y.isel(time=slice(None,-lag))

Xst = (X.to_array()
         .transpose("time","lat","lon","variable")
         .stack(sample=("time","lat","lon"))
         .transpose("sample","variable"))
yst = y.stack(sample=("time","lat","lon"))

mask = (~np.isnan(Xst).any("variable") & np.isfinite(yst)).compute().values
Xv    = Xst.values[mask].astype("float32")
yv    = yst.values[mask].astype("float32")
ts    = Xst["time"].values[mask]
lats  = Xst["lat"].values[mask]
lons  = Xst["lon"].values[mask]

test = ts >= np.datetime64("2021-01-01")
X_test, y_test = Xv[test], yv[test]
t_test, lat_test, lon_test = ts[test], lats[test], lons[test]

# ─── PREDICT & METRICS ────────────────────────────────────
y_pred = reg.predict(X_test)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)

# 1) Scatter with metrics
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, s=1, alpha=0.3)
lims = (min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()))
plt.plot(lims, lims, 'k--', label='1:1')
plt.xlabel("Observed log₁₀(chl­a)")
plt.ylabel("Predicted log₁₀(chl­a)")
plt.title(f"Prediction vs Observation (2021)\nRMSE={rmse:.3f}, R²={r2:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir/"scatter_with_metrics.png", dpi=300)
plt.close()

# Build MultiIndex and unstack to cubes
mi = pd.MultiIndex.from_arrays([t_test, lat_test, lon_test],
                               names=("time","lat","lon"))
pred_da = xr.DataArray(y_pred, coords={"sample":mi}, dims=["sample"])
true_da = xr.DataArray(y_test, coords={"sample":mi}, dims=["sample"])
pred_cube = pred_da.unstack("sample")
true_cube = true_da.unstack("sample")

# 2) Mean observed & predicted maps
mean_true = true_cube.mean(dim="time")
mean_pred = pred_cube.mean(dim="time")

plt.figure(figsize=(6,8))
mean_true.plot()
plt.title("Mean Observed log₁₀(chl­a) (2021)")
plt.tight_layout()
plt.savefig(out_dir/"mean_observed.png", dpi=300)
plt.close()

plt.figure(figsize=(6,8))
mean_pred.plot()
plt.title("Mean Predicted log₁₀(chl­a) (2021)")
plt.tight_layout()
plt.savefig(out_dir/"mean_predicted.png", dpi=300)
plt.close()

# 3) Per-pixel R² map
r2_map = 1 - ((pred_cube - true_cube)**2).sum("time") / \
              ((true_cube - true_cube.mean("time"))**2).sum("time")

plt.figure(figsize=(6,8))
r2_map.plot(vmin=0, vmax=1)
plt.title("Per-pixel R² (2021)")
plt.tight_layout()
plt.savefig(out_dir/"spatial_r2.png", dpi=300)
plt.close()

print(f"Saved diagnostics to {out_dir}")
