#!/usr/bin/env python3
# scripts/tune_xgb.py
"""
Optuna tuning for the XGBoost regressor that predicts log10-chlorophyll
one 8-day composite ahead.

Design matrix (build_design_matrix):
    • 12 current vars
    • lags 1-3  (36 cols)
    • 3-composite means (12)
    • sin_doy / cos_doy (2)
    • nflh_anom, flh_kd (2)
    • curl_uv, dist_river_km, log1p_dist_river (3)
      ------------------------------------------------
      total = 67 predictors
"""
import pathlib, yaml, numpy as np, xarray as xr, optuna, joblib, json, warnings
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from _feature_utils import build_design_matrix

warnings.filterwarnings("ignore", "The specified chunks")

# ─── paths & constants ────────────────────────────────────────────
root   = pathlib.Path(__file__).resolve().parents[1]
cfg    = yaml.safe_load(open(root / "config.yaml"))
cube_f = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"

FORE_LAG  = int(cfg.get("forecast_lag", 1))
COAST_R   = int(cfg.get("coast_radius_cells", 4))
LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0
PRED_VARS = [
    "sst","t2m","u10","v10","avg_sdswrf",
    "Kd_490","nflh","so","thetao","uo","vo","zos"
]

out_dir   = root / "Models"; out_dir.mkdir(exist_ok=True)
STORAGE   = f"sqlite:///{out_dir/'optuna_habs_xgb.db'}"

# ─── helper: build train / val arrays (same masking logic as train_baseline) ──
def get_train_val_arrays():
    ds = xr.open_dataset(cube_f, chunks={"time": 50})
    ds = ds.sel(lat=slice(LAT_MIN,LAT_MAX), lon=slice(LON_MIN,LON_MAX))

    # --- design matrix & target
    X_all, y = build_design_matrix(ds, PRED_VARS, FORE_LAG)

    # --- coastal mask (aligned in time)
    ocean = ~ds["chlor_a"].isel(time=0).isnull()
    d2l   = xr.apply_ufunc(distance_transform_edt, ~ocean,
                           input_core_dims=[("lat","lon")],
                           output_core_dims=[("lat","lon")], dask="allowed")
    coast = (ocean & (d2l <= COAST_R)).broadcast_like(ds["chlor_a"])
    coast = coast.sel(time=X_all["time"])

    # --- stack to 1-D “sample”
    def _stack(da):
        extra=[d for d in da.dims if d not in ("time","lat","lon")]
        return (da.transpose("time","lat","lon",*extra)
                  .stack(sample=("time","lat","lon"))
                  .transpose("sample",*extra))

    Xs, ys        = _stack(X_all), _stack(y)
    coast_mask    = coast.transpose("time","lat","lon").stack(sample=("time","lat","lon"))
    finite_feats  = (~np.isnan(Xs).any("var"))
    finite_targ   = np.isfinite(ys)

    # align → avoid shape mismatches
    coast_mask, finite_feats, finite_targ = xr.align(
        coast_mask, finite_feats, finite_targ,
        join="outer", fill_value=False
    )
    mask = (coast_mask & finite_feats & finite_targ).values

    X_np  = Xs.values[mask].astype("float32")
    y_np  = ys.values[mask].astype("float32")
    times = coast_mask["time"].values[mask]

    train = times < np.datetime64("2020-01-01")
    val   = (times >= np.datetime64("2020-01-01")) & (times < np.datetime64("2021-01-01"))

    def _sub(idx): rows = np.where(idx)[0]; return X_np[rows], y_np[rows]
    return _sub(train) + _sub(val)

X_tr, y_tr, X_val, y_val = get_train_val_arrays()

# ─── Optuna objective ────────────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators"     : trial.suggest_int( "n_estimators",   300, 1500, 100),
        "max_depth"        : trial.suggest_int( "max_depth",      4, 10),
        "learning_rate"    : trial.suggest_float("learning_rate", 3e-4, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample",     0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight" : trial.suggest_float("min_child_weight", 0.0, 10.0),
        "gamma"            : trial.suggest_float("gamma",         0.0, 5.0),
        "reg_lambda"       : trial.suggest_float("reg_lambda",    0.0, 5.0),
        "objective"        : "reg:squarederror",
        "tree_method"      : "hist",
        "eval_metric"      : "rmse",
        "early_stopping_rounds": 30,
        "n_jobs"           : -1,
        "verbosity"        : 0,
        "random_state"     : 42
    }
    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    rmse = np.sqrt(mean_squared_error(y_val, model.predict(X_val)))
    return rmse

# ─── run study ────────────────────────────────────────────────────
study = optuna.create_study(
    direction="minimize",
    storage  = STORAGE,
    study_name="xgb_chl_reg",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(objective, n_trials=40, show_progress_bar=True)

print("\nBest RMSE :", round(study.best_value, 4))
print("Best parameters:\n", json.dumps(study.best_params, indent=2))

# save artefacts
yaml.safe_dump(study.best_params, open(out_dir/"xgb_tuned_params.yaml", "w"))
joblib.dump(study,                 out_dir/"optuna_study.pkl")
study.trials_dataframe().to_csv(   out_dir/"optuna_trials.csv", index=False)
print(f"✓ Results saved in {out_dir}")

# (Optional) dashboard:
#   micromamba activate habs
#   optuna-dashboard sqlite:///Models/optuna_habs_xgb.db
