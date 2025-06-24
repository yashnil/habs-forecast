#!/usr/bin/env python3
# scripts/train_baseline.py   ◇ baseline XGBoost regressor + classifier
# ---------------------------------------------------------------------
#  * uses current-week met/ocean vars
#  * predicts log10-chlor_a one 8-day composite ahead
#  * trains only on ≈ 10 mi coastal strip (radius = coast_radius_cells)
#
#  Results:
#     • Models/xgb_chla_reg.joblib
#     • Models/xgb_hab_clf.joblib
#     • Models/metrics.tsv         ← easy to paste in a paper / notebook
# ---------------------------------------------------------------------
import warnings, xarray as xr
warnings.filterwarnings("ignore", module=r"xarray\.")
import pathlib, yaml, numpy as np, xarray as xr, joblib, pandas as pd
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, average_precision_score,
                             classification_report)
from scipy.ndimage import distance_transform_edt
from xgboost import XGBRegressor, XGBClassifier

# ╭───────────────────────── configuration ─────────────────────────╮
repo_root = pathlib.Path(__file__).resolve().parents[1]
cfg       = yaml.safe_load(open(repo_root / "config.yaml"))

CUBE_FILE   = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
OUT_DIR     = repo_root / "Models";  OUT_DIR.mkdir(exist_ok=True)
FORECAST_LAG = int(cfg.get("forecast_lag",        1))  # 1 composite ahead
COAST_R      = int(cfg.get("coast_radius_cells",  4))  # ~10 mi coastal strip
HAB_THRESH   = float(cfg.get("hab_threshold",     20)) # µg L⁻¹ → binarise
# CA/OR/WA domain
LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0
# predictors
PRED_VARS = ["sst","t2m","u10","v10","avg_sdswrf",
             "Kd_490","nflh","so","thetao","uo","vo","zos"]

# ╭───────────────────────── load & pre-process ─────────────────────╮
ds = xr.open_dataset(CUBE_FILE, chunks={"time": 50})
ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

# coastal ocean mask – take finite SST as “ocean”
ocean_mask = ~ds["chlor_a"].isel(time=0).isnull()  # 1 = ocean, 0 = land
dist2land  = xr.apply_ufunc(
                distance_transform_edt, ~ocean_mask,
                input_core_dims=[("lat", "lon")],
                output_core_dims=[("lat", "lon")],
                dask="allowed"
             )
coast_bool_2d = (ocean_mask & (dist2land <= COAST_R))

# ▶ broadcast across time so dims = (time, lat, lon)
coast_bool = coast_bool_2d.broadcast_like(ds["chlor_a"])   # or: .expand_dims(time=ds.time)
coast_bool = coast_bool.isel(time=slice(None, -FORECAST_LAG))   # <- add this

# ── predictors & lag-1 features ───────────────────────────────────
X_curr = ds[PRED_VARS]                   # t
X_prev = X_curr.shift(time=+1)           # t-1  (pad with NaNs)

# keep only slices where the target is defined
X_curr = X_curr.isel(time=slice(None, -FORECAST_LAG))
X_prev = X_prev.isel(time=slice(None, -FORECAST_LAG))

# rename lag-1 labels so they don’t collide
X_prev = (
    X_prev.to_array("var")
          .assign_coords(var=[v + "_lag1" for v in PRED_VARS])
)

X_curr = X_curr.to_array("var")

# concatenate on the feature axis  → (time, lat, lon, var)
X = xr.concat([X_curr, X_prev], dim="var")               # 24 features now
y = (ds["log_chl"]
        .shift(time=-FORECAST_LAG)
        .isel(time=slice(None, -FORECAST_LAG)))

def stack_to_samples(da):
    """(time,lat,lon,…) → stack to first-axis=sample"""
    extra = [d for d in da.dims if d not in ("time","lat","lon")]
    return (da
            .transpose("time","lat","lon",*extra)
            .stack(sample=("time","lat","lon"))
            .transpose("sample",*extra))

X_stack = stack_to_samples(X)  
y_stack = stack_to_samples(y)  

flat_mask = coast_bool.stack(sample=("time","lat","lon")).values

x_np = X_stack.values          # (samples , n_features)
y_np = y_stack.values          # (samples ,)

valid = (
    flat_mask
    & ~np.isnan(x_np).any(axis=1)
    & np.isfinite(y_np)
)

X_np  = x_np[valid].astype("float32")
y_np  = y_np[valid].astype("float32")
times = X_stack["time"].values[valid]

print(f"Final coastal samples : {len(y_np):,}")

# ╭───────────────────────── temporal split ─────────────────────────╮
train = times < np.datetime64("2020-01-01")
val   = (times >= np.datetime64("2020-01-01")) & (times < np.datetime64("2021-01-01"))
test  = times >= np.datetime64("2021-01-01")
def _sub(m): idx = np.where(m)[0] ; return X_np[idx], y_np[idx]
X_tr,y_tr = _sub(train) ; X_va,y_va = _sub(val) ; X_te,y_te = _sub(test)
print("Split sizes train/val/test :", [len(y_tr), len(y_va), len(y_te)])

# ╭───────────────────────── XGB Regressor ──────────────────────────╮

best_params = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.010986686201636925,
    "subsample": 0.6558178059090922,
    "colsample_bytree": 0.834021639480953,
    "min_child_weight": 0.7812818135862791,
    "gamma": 3.669227243241542,
    "reg_lambda": 4.940671977816998   # XGBoost uses *reg_lambda* not *lambda*
}
best_params.update({"random_state": 42})
reg = XGBRegressor(**best_params,  tree_method="hist", n_jobs=-1,
                   eval_metric="rmse", early_stopping_rounds=50, verbosity=1)

reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
y_hat  = reg.predict(X_te)
rmse_lg = np.sqrt(mean_squared_error(y_te, y_hat))
mae_lg  = mean_absolute_error(y_te, y_hat)
r2      = r2_score(y_te, y_hat)
rmse_lin = 10**rmse_lg - 1  # approximate linear-space error
print(f"[Reg] RMSE(log10)={rmse_lg:.3f}  RMSE(µg/L)≈{rmse_lin:.2f}  MAE={mae_lg:.3f}  R²={r2:.3f}")
joblib.dump(reg, OUT_DIR / "xgb_chla_reg.joblib")

# ╭───────────────────────── XGB Classifier ─────────────────────────╮
y_tr_bin = (10**y_tr > HAB_THRESH).astype(int)
y_va_bin = (10**y_va > HAB_THRESH).astype(int)
y_te_bin = (10**y_te > HAB_THRESH).astype(int)
pos_w = ((1 - y_tr_bin).sum() / y_tr_bin.sum()
         if cfg.get("pos_weight","auto")=="auto"
         else float(cfg["pos_weight"]))
clf = XGBClassifier(**best_params, 
        scale_pos_weight=pos_w,
        tree_method="hist", n_jobs=-1,
        eval_metric="logloss", early_stopping_rounds=50, verbosity=1)
clf.fit(X_tr, y_tr_bin, eval_set=[(X_va, y_va_bin)])
prob_te = clf.predict_proba(X_te)[:,1]

# --- metrics -------------------------------------------------------
roc_auc = roc_auc_score(y_te_bin, prob_te)
pr_auc  = average_precision_score(y_te_bin, prob_te)
# pick probability that maximises F1 on *validation* set
prob_va = clf.predict_proba(X_va)[:,1]
prec, rec, thr = precision_recall_curve(y_va_bin, prob_va)
f1_va = 2*prec*rec/(prec+rec+1e-9)
best_thr = thr[np.argmax(f1_va)]
pred_bin = (prob_te >= best_thr).astype(int)
report   = classification_report(y_te_bin, pred_bin, digits=3)
cm       = confusion_matrix(y_te_bin, pred_bin)

print("\n[Clf] optimum threshold (val) =", round(best_thr,3))
print("[Clf] ROC-AUC =", round(roc_auc,3), " PR-AUC =", round(pr_auc,3))
print(report)

joblib.dump(clf, OUT_DIR / "xgb_hab_clf.joblib")

# ╭───────────────────────── save metrics table ────────────────────╮
df = pd.DataFrame([{
    "rmse_log": rmse_lg, "mae_log": mae_lg, "rmse_lin": rmse_lin,
    "r2": r2, "roc_auc": roc_auc, "pr_auc": pr_auc,
    "thr": best_thr,
    "tp": cm[1,1], "fp": cm[0,1], "tn": cm[0,0], "fn": cm[1,0]
}])
df.to_csv(OUT_DIR / "metrics.tsv", sep="\t", index=False)
print(f"→ metrics written to {OUT_DIR / 'metrics.tsv'}")
