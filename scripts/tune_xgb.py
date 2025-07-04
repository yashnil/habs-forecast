#!/usr/bin/env python3
# scripts/tune_xgb.py
"""
Optuna hyper-parameter tuning for the XGBoost regressor that predicts
log10-chlorophyll one 8-day composite ahead.  *Design matrix now matches
train_baseline.py: 12 current-week vars  + 12 lag-1 vars  = 24 features.*
"""
import pathlib, yaml, numpy as np, xarray as xr, optuna, joblib, json, warnings
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from _feature_utils import build_design_matrix   # new
warnings.filterwarnings("ignore", "The specified chunks")

# ─── paths & constants ────────────────────────────────────────────
root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))
CUBE_FILE   = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
FORECAST_LAG = int(cfg.get("forecast_lag", 1))
COAST_R      = int(cfg.get("coast_radius_cells", 4))
LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0
PRED_VARS = [
    "sst","t2m","u10","v10","avg_sdswrf",
    "Kd_490","nflh","so","thetao","uo","vo","zos"
]
OUT_DIR   = root / "Models"; OUT_DIR.mkdir(exist_ok=True)
STUDY_DB  = OUT_DIR / "optuna_habs_xgb.db"
STORAGE   = f"sqlite:///{STUDY_DB}"

# ─── helper: load & split arrays ──────────────────────────────────
def get_train_val_arrays():
    ds = xr.open_dataset(CUBE_FILE, chunks={"time": 50})
    ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

    # coastal mask
    ocean = ~ds["chlor_a"].isel(time=0).isnull()
    d2l   = xr.apply_ufunc(distance_transform_edt, ~ocean,
                           input_core_dims=[("lat","lon")],
                           output_core_dims=[("lat","lon")], dask="allowed")
    coast = (ocean & (d2l <= COAST_R)).broadcast_like(ds["chlor_a"])
    coast = coast.isel(time=slice(None, -FORECAST_LAG-3))
    
    print("▶ building design matrix …")
    X_all, y = build_design_matrix(ds, PRED_VARS, FORECAST_LAG)
    print("   ✓ design matrix built; stacking …")

    # stack to (sample, var)
    def stack(da):
        extra = [d for d in da.dims if d not in ("time","lat","lon")]
        return (da.transpose("time","lat","lon",*extra)
                  .stack(sample=("time","lat","lon"))
                  .transpose("sample",*extra))
    
    print("   ✓ design matrix built; stacking …")
    X_st, y_st = stack(X_all), stack(y)
    
    mask = (
    coast.stack(sample=("time","lat","lon"))
            .compute()
            .values
        & ~np.isnan(X_st).any("var").compute()
        &  np.isfinite(y_st).compute()
    )
    
    print("   ✓ stacked arrays; valid mask size =", mask.sum())  # now an int

    X_np = X_st.data[mask].compute().astype("float32")
    y_np = y_st.data[mask].compute().astype("float32")
    times = X_st["time"].values[mask]

    train = times < np.datetime64("2020-01-01")
    valid = (times >= np.datetime64("2020-01-01")) & (times < np.datetime64("2021-01-01"))
    def _sub(m): idx = np.where(m)[0]; return X_np[idx], y_np[idx]
    return _sub(train) + _sub(valid)   # X_tr, y_tr, X_val, y_val

X_tr, y_tr, X_val, y_val = get_train_val_arrays()

# ─── Optuna objective ────────────────────────────────────────────
def objective(trial):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 300, 1000, 100),
        "max_depth"        : trial.suggest_int("max_depth", 4, 10),
        "learning_rate"    : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight" : trial.suggest_float("min_child_weight", 0.0, 10.0),
        "gamma"            : trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 0.0, 5.0),
        "objective"        : "reg:squarederror",
        "tree_method"      : "hist",        # change to "gpu_hist" if GPU available
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
            storage=STORAGE,
            study_name="xgb_chl_reg",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
)
study.optimize(objective, n_trials=30, show_progress_bar=True)

print("\nBest RMSE :", study.best_value)
print("Best parameters:\n", json.dumps(study.best_params, indent=2))

yaml.safe_dump(study.best_params, open(OUT_DIR / "xgb_tuned_params.yaml", "w"))
joblib.dump(study, OUT_DIR / "optuna_study.pkl")
study.trials_dataframe().to_csv(OUT_DIR / "optuna_trials.csv", index=False)
print(f"✓ Saved params, study object, and trials CSV in {OUT_DIR}")

# Dashboard reminder:
# micromamba activate habs
# cd ~/Desktop/habs-forecast
# optuna-dashboard sqlite:///Models/optuna_habs_xgb.db
