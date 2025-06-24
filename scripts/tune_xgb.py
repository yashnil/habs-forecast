#!/usr/bin/env python3
# scripts/tune_xgb.py
"""
Optuna hyper-parameter tuning for the XGBoost *regressor* that predicts
log10-chlorophyll one composite (≈8 days) ahead, restricted to a ≈10 mi coastal
strip.  Uses exactly the same data-prep as scripts/train_baseline.py.

After it finishes, the best params are written to
    Models/xgb_tuned_params.yaml
and printed to screen so you can paste them back into train_baseline.py.
"""
# ─────────────────────────────────────────────────────────────────────────────
import pathlib, yaml, numpy as np, xarray as xr, optuna, joblib, json, warnings
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", "The specified chunks")     # xarray chunks msg

# ╭───────────────────────── project paths & constants ───────────────────────╮
repo_root = pathlib.Path(__file__).resolve().parents[1]
cfg       = yaml.safe_load(open(repo_root / "config.yaml"))

CUBE_FILE  = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
FORECAST_LAG = int(cfg.get("forecast_lag",        1))
COAST_R      = int(cfg.get("coast_radius_cells",  4))
LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0

PRED_VARS = ["sst","t2m","u10","v10","avg_sdswrf",
             "Kd_490","nflh","so","thetao","uo","vo","zos"]

OUT_DIR = repo_root / "Models";  OUT_DIR.mkdir(exist_ok=True)

# ╭───────────────────────── helper: load & return train / val arrays ─────────╮
def get_train_val_arrays():
    ds = xr.open_dataset(CUBE_FILE, chunks={"time": 50})
    ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    ocean_mask = ~ds["chlor_a"].isel(time=0).isnull()
    dist2land  = xr.apply_ufunc(
                    distance_transform_edt, ~ocean_mask,
                    input_core_dims=[("lat","lon")],
                    output_core_dims=[("lat","lon")],
                    dask="allowed"
                 )
    coast_bool_2d = (ocean_mask & (dist2land <= COAST_R))
    coast_bool = coast_bool_2d.broadcast_like(ds["chlor_a"])
    coast_bool = coast_bool.isel(time=slice(None, -FORECAST_LAG))

    X = ds[PRED_VARS]
    y = ds["log_chl"].shift(time=-FORECAST_LAG)
    X = X.isel(time=slice(None, -FORECAST_LAG))
    y = y.isel(time=slice(None, -FORECAST_LAG))

    def stack(da):
        extra = [d for d in da.dims if d not in ("time","lat","lon")]
        return (da
                .transpose("time","lat","lon",*extra)
                .stack(sample=("time","lat","lon"))
                .transpose("sample",*extra))

    X_st = stack(X.to_array("var"))
    y_st = stack(y)

    flat_mask = coast_bool.stack(sample=("time","lat","lon")).values
    valid = (flat_mask &
             ~np.isnan(X_st).any("var") &
             np.isfinite(y_st))

    X_np  = X_st.values[valid].astype("float32")
    y_np  = y_st.values[valid].astype("float32")
    times = X_st["time"].values[valid]

    # temporal split: train <2020, val = 2020, test >=2021  (use train/val only)
    tr = times < np.datetime64("2020-01-01")
    va = (times >= np.datetime64("2020-01-01")) & (times < np.datetime64("2021-01-01"))

    def _sub(m): idx = np.where(m)[0] ; return X_np[idx], y_np[idx]
    return _sub(tr) + _sub(va)   # X_tr, y_tr, X_val, y_val

X_tr, y_tr, X_val, y_val = get_train_val_arrays()

# ╭───────────────────────── Optuna objective ────────────────────────────────╮
def objective(trial: optuna.Trial):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 300, 1000, step=100),
        "max_depth"        : trial.suggest_int("max_depth", 4, 10),
        "learning_rate"    : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight" : trial.suggest_float("min_child_weight", 0.0, 10.0),
        "gamma"            : trial.suggest_float("gamma", 0.0, 5.0),
        "lambda"           : trial.suggest_float("lambda", 0.0, 5.0),
        "tree_method"      : "hist",
        "n_jobs"           : -1,
        "eval_metric"      : "rmse",
        "early_stopping_rounds": 30,
        "verbosity"        : 0
    }
    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, pred) ** 0.5      # √MSE  →  RMSE
    return rmse

# ╭───────────────────────── run study ───────────────────────────────────────╮
study = optuna.create_study(direction="minimize", study_name="xgb_chl_reg")
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("\nBest RMSE :", study.best_value)
print("Best parameters:\n", json.dumps(study.best_params, indent=2))

# save params to YAML for convenience
best_yaml = OUT_DIR / "xgb_tuned_params.yaml"
yaml.safe_dump(study.best_params, open(best_yaml, "w"))
print(f"→ Saved best params to {best_yaml}")

# (optional) save the full study for later analysis
joblib.dump(study, OUT_DIR / "optuna_study.pkl")
