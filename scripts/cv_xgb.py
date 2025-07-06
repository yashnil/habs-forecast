#!/usr/bin/env python3
# scripts/cv_xgb.py
"""
Year-blocked cross-validation for the tuned XGBoost HAB regressor.

Design matrix  : 67 features  (t, lags 1-3, 3-comp means, seasonality,
                               optical indices, curl, log-dist-river)
CV folds       : leave-one-year-out  (2016-2020) ; 2021 stays untouched
Output         : Models/cv_metrics.tsv   (per-year RMSE, MAE, R²)
"""
import pathlib, yaml, numpy as np, xarray as xr, pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from _feature_utils import build_design_matrix

# ── config & paths ────────────────────────────────────────────────
root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))
cube = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
out_tsv = root / "Models" / "cv_metrics.tsv"
out_tsv.parent.mkdir(exist_ok=True)

LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0
COAST_R      = int(cfg.get("coast_radius_cells", 4))
FORECAST_LAG = int(cfg.get("forecast_lag",       1))
PRED_VARS = [
    "sst","t2m","u10","v10","avg_sdswrf",
    "Kd_490","nflh","so","thetao","uo","vo","zos"
]

# ── load best Optuna params if they exist ─────────────────────────
tuned_yaml = root / "Models" / "xgb_tuned_params.yaml"
if tuned_yaml.exists():
    best_params = yaml.safe_load(open(tuned_yaml))
else:                                  # first-run fall-back
    best_params = dict(
        n_estimators    = 300,
        max_depth       = 4,
        learning_rate   = 0.015804704867439994,
        subsample       = 0.9885668717366809,
        colsample_bytree= 0.8659690419027274,
        min_child_weight= 4.1935572069285945,
        gamma           = 2.1888927171422337,
        reg_lambda      = 0.6880408368756581,
    )

# add fixed items
best_params.update(
    tree_method = "hist",
    objective   = "reg:squarederror",
    n_jobs      = -1,
    random_state= 42,
)

# ── helper: build design matrix & masks exactly like train_baseline ─
def build_design():
    ds = xr.open_dataset(cube, chunks={"time": 50})
    ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

    # coast mask
    ocean = ~ds["chlor_a"].isel(time=0).isnull()
    d2l   = xr.apply_ufunc(distance_transform_edt, ~ocean,
                           input_core_dims=[("lat","lon")],
                           output_core_dims=[("lat","lon")], dask="allowed")
    coast = (ocean & (d2l <= COAST_R)).broadcast_like(ds["chlor_a"])

    # full feature cube
    X_all, y = build_design_matrix(ds, PRED_VARS, FORECAST_LAG)
    coast    = coast.sel(time=X_all["time"])     # align time axis

    # stack to (sample, var)
    def stack(da):
        extra=[d for d in da.dims if d not in ("time","lat","lon")]
        return (da.transpose("time","lat","lon",*extra)
                 .stack(sample=("time","lat","lon"))
                 .transpose("sample",*extra))

    Xs, ys   = stack(X_all), stack(y)
    coast_m  = coast.transpose("time","lat","lon").stack(sample=("time","lat","lon"))

    finite_f = ~np.isnan(Xs).any("var")
    finite_y =  np.isfinite(ys)

    coast_m, finite_f, finite_y = xr.align(coast_m, finite_f, finite_y,
                                           join="outer", fill_value=False)
    mask = (coast_m & finite_f & finite_y).values

    X_np  = Xs.values[mask].astype("float32")
    y_np  = ys.values[mask].astype("float32")
    times = coast_m["time"].values[mask]
    return X_np, y_np, times

X, y, t = build_design()

# ── cross-validation ──────────────────────────────────────────────
metrics = []
for yr in range(2016, 2021):         # 2021 is hold-out
    train = t.astype("datetime64[Y]") != np.datetime64(f"{yr}")
    val   = ~train

    model = XGBRegressor(**best_params)
    model.fit(X[train], y[train])
    pred  = model.predict(X[val])

    rmse = np.sqrt(mean_squared_error(y[val], pred))
    mae  = mean_absolute_error(y[val], pred)
    r2   = r2_score(y[val], pred)

    metrics.append({"year": yr, "rmse": rmse, "mae": mae, "r2": r2})
    print(f"Fold {yr}: RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}")

df = pd.DataFrame(metrics)
df.to_csv(out_tsv, sep="\t", index=False)

print("\n── summary ──")
print(df.describe()[['rmse','mae','r2']])
print(f"\n✓ per-year metrics saved → {out_tsv}")

'''
Run with:
micromamba activate habs
cd ~/Desktop/habs-forecast
python scripts/cv_xgb.py
'''
