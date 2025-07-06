#!/usr/bin/env python3
# scripts/train_baseline.py
"""
Baseline XGBoost regressor & classifier for HAB forecasting.
Design matrix: current vars + lags 1-3 + 3-comp means + seasonality +
optical indices + curl/dist-river  → 67 predictors
"""
import warnings, pathlib, yaml, numpy as np, xarray as xr, joblib, pandas as pd
warnings.filterwarnings("ignore", module=r"xarray\.")
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, average_precision_score,
                             classification_report)
from scipy.ndimage import distance_transform_edt
from xgboost import XGBRegressor, XGBClassifier
from _feature_utils import build_design_matrix

# ── config ────────────────────────────────────────────────────────
repo = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(repo / "config.yaml"))

CUBE     = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
OUT_DIR  = repo / "Models"; OUT_DIR.mkdir(exist_ok=True)

FORE_LAG = int(cfg.get("forecast_lag", 1))
COAST_R  = int(cfg.get("coast_radius_cells", 4))
HAB_THR  = float(cfg.get("hab_threshold", 20))

LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0
PRED_VARS = [
    "sst","t2m","u10","v10","avg_sdswrf",
    "Kd_490","nflh","so","thetao","uo","vo","zos"
]

# ── load cube & build design matrix ───────────────────────────────
ds = xr.open_dataset(CUBE, chunks={"time": 50})
ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

X_all, y = build_design_matrix(ds, PRED_VARS, FORE_LAG)

# ── coastal-ocean mask (same time dimension as X_all) ─────────────
ocean = ~ds["chlor_a"].isel(time=0).isnull()
d2l   = xr.apply_ufunc(distance_transform_edt, ~ocean,
                       input_core_dims=[("lat","lon")],
                       output_core_dims=[("lat","lon")], dask="allowed")
coast_bool = (ocean & (d2l <= COAST_R)).broadcast_like(ds["chlor_a"])
coast_bool = coast_bool.sel(time=X_all["time"])          # align times

# ── flatten everything to a common “sample” dim ──────────────────
def _stack(da):
    extra = [d for d in da.dims if d not in ("time","lat","lon")]
    return (da.transpose("time","lat","lon",*extra)
              .stack(sample=("time","lat","lon"))
              .transpose("sample",*extra))

Xs, ys        = _stack(X_all), _stack(y)
# --- after Xs, ys are defined ------------------------------------
coast_mask = (
    coast_bool.transpose("time", "lat", "lon")
              .stack(sample=("time", "lat", "lon"))
)

finite_feats = (~np.isnan(Xs).any("var"))
finite_targ  =  np.isfinite(ys)

# ⇣ NEW: force all three DA's to share the same 'sample' index
coast_mask, finite_feats, finite_targ = xr.align(
    coast_mask, finite_feats, finite_targ,
    join="outer",                # take the union of sample labels
    fill_value=False             # missing entries → False
)

mask_da = coast_mask & finite_feats & finite_targ
mask    = mask_da.values         # ← now exactly len(Xs.sample)

X_np   = Xs.values[mask].astype("float32")
y_np   = ys.values[mask].astype("float32")
times  = mask_da["time"].values[mask]
print(f"Final coastal samples : {len(y_np):,}")

# ── temporal split (train / val / test) ───────────────────────────
train = times < np.datetime64("2020-01-01")
val   = (times >= np.datetime64("2020-01-01")) & (times < np.datetime64("2021-01-01"))
test  = times >= np.datetime64("2021-01-01")

def _sub(idx): i = np.where(idx)[0]; return X_np[i], y_np[i]
X_tr,y_tr = _sub(train);  X_va,y_va = _sub(val);  X_te,y_te = _sub(test)
print("Split sizes train/val/test :", [len(y_tr), len(y_va), len(y_te)])

# ── XGBoost parameters (tuned via Optuna) ─────────────────────────
best_params = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.015804704867439994,
    "subsample": 0.9885668717366809,
    "colsample_bytree": 0.8659690419027274,
    "min_child_weight": 4.1935572069285945,
    "gamma": 2.1888927171422337,
    "reg_lambda": 0.6880408368756581,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42
}

# ── Regressor ─────────────────────────────────────────────────────
reg = XGBRegressor(**best_params, eval_metric="rmse",
                   early_stopping_rounds=30)
reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

y_hat    = reg.predict(X_te)
rmse_lg  = np.sqrt(mean_squared_error(y_te, y_hat))
mae_lg   = mean_absolute_error(y_te, y_hat)
r2       = r2_score(y_te, y_hat)
rmse_lin = 10**rmse_lg - 1
print(f"[Reg] RMSE(log10)={rmse_lg:.3f}  RMSE(µg/L)≈{rmse_lin:.2f}  "
      f"MAE={mae_lg:.3f}  R²={r2:.3f}")
joblib.dump(reg, OUT_DIR / "xgb_chla_reg.joblib")

# ─── Classifier fit & metrics ─────────────────────────────────────
y_tr_bin = (10**y_tr > HAB_THR).astype(int)
y_va_bin = (10**y_va > HAB_THR).astype(int)
y_te_bin = (10**y_te > HAB_THR).astype(int)

pos_w = ((1 - y_tr_bin).sum() / y_tr_bin.sum())

# copy regressor params → classifier params
clf_params = best_params.copy()
clf_params.pop("objective", None)          # remove reg objective
clf_params["objective"] = "binary:logistic"

clf = XGBClassifier(**clf_params,
                    scale_pos_weight=pos_w,
                    eval_metric="logloss",
                    early_stopping_rounds=30)

clf.fit(X_tr, y_tr_bin, eval_set=[(X_va, y_va_bin)], verbose=False)

prob_te  = clf.predict_proba(X_te)[:,1]
roc_auc  = roc_auc_score(y_te_bin, prob_te)
pr_auc   = average_precision_score(y_te_bin, prob_te)

# choose threshold that maximises F1 on validation set
prob_va = clf.predict_proba(X_va)[:,1]
prec, rec, thr = precision_recall_curve(y_va_bin, prob_va)
best_thr = thr[np.argmax(2*prec*rec/(prec+rec+1e-9))]
pred_bin = (prob_te >= best_thr).astype(int)

report = classification_report(y_te_bin, pred_bin, digits=3)
cm     = confusion_matrix(y_te_bin, pred_bin)
print("\n[Clf] optimum threshold (val) =", round(best_thr,3))
print("[Clf] ROC-AUC =", round(roc_auc,3), " PR-AUC =", round(pr_auc,3))
print(report)
joblib.dump(clf, OUT_DIR / "xgb_hab_clf.joblib")

# ── save metrics table ────────────────────────────────────────────
pd.DataFrame([{
    "rmse_log": rmse_lg, "mae_log": mae_lg, "rmse_lin": rmse_lin,
    "r2": r2, "roc_auc": roc_auc, "pr_auc": pr_auc,
    "thr": best_thr,
    "tp": cm[1,1], "fp": cm[0,1], "tn": cm[0,0], "fn": cm[1,0]
}]).to_csv(OUT_DIR / "metrics.tsv", sep="\t", index=False)

print(f"→ metrics written to {OUT_DIR / 'metrics.tsv'}")


'''
micromamba activate habs
cd ~/Desktop/habs-forecast
python scripts/train_baseline.py      # new 2021 metrics
python scripts/cv_xgb.py              # updated cross-val summary
'''