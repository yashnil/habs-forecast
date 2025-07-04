#!/usr/bin/env python3
# scripts/train_baseline.py
"""
Baseline XGBoost regressor & classifier for HAB forecasting.
Design matrix: 12 current vars + 12 lag-1 + 12 lag-2 + sin/cos(DOY) + lag-3 + roll-3 + indices = 42 features
"""
import warnings, pathlib, yaml, numpy as np, xarray as xr, joblib, pandas as pd
warnings.filterwarnings("ignore", module=r"xarray\.")
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, average_precision_score,
                             classification_report)
from scipy.ndimage import distance_transform_edt
from xgboost import XGBRegressor, XGBClassifier
from _feature_utils import build_design_matrix   # ← NEW

# ─── configuration ────────────────────────────────────────────────
repo = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(repo / "config.yaml"))

CUBE     = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
OUT_DIR  = repo / "Models"; OUT_DIR.mkdir(exist_ok=True)
FORE_LAG = int(cfg.get("forecast_lag", 1))
COAST_R  = int(cfg.get("coast_radius_cells", 4))
HAB_THR  = float(cfg.get("hab_threshold", 20))
LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0
PRED_VARS = ["sst","t2m","u10","v10","avg_sdswrf",
             "Kd_490","nflh","so","thetao","uo","vo","zos"]

# ─── load cube & build features ───────────────────────────────────
ds = xr.open_dataset(CUBE, chunks={"time": 50})
ds = ds.sel(lat=slice(LAT_MIN,LAT_MAX), lon=slice(LON_MIN,LON_MAX))

# design matrix & target
X_all, y = build_design_matrix(ds, PRED_VARS, FORE_LAG)   # 42 vars

# coast mask, trimmed to lag-2 horizon
ocean = ~ds["chlor_a"].isel(time=0).isnull()
d2l   = xr.apply_ufunc(distance_transform_edt, ~ocean,
                       input_core_dims=[("lat","lon")],
                       output_core_dims=[("lat","lon")], dask="allowed")
coast_bool = (ocean & (d2l <= COAST_R)).broadcast_like(ds["chlor_a"])
coast_bool = coast_bool.isel(time=slice(None, -FORE_LAG - 3))

# stack samples
def _stack(da):
    extra=[d for d in da.dims if d not in ("time","lat","lon")]
    return (da.transpose("time","lat","lon",*extra)
              .stack(sample=("time","lat","lon"))
              .transpose("sample",*extra))

Xs, ys = _stack(X_all), _stack(y)
mask = (coast_bool.stack(sample=("time","lat","lon")).values &
        ~np.isnan(Xs).any("var") & np.isfinite(ys))

X_np = Xs.values[mask].astype("float32")
y_np = ys.values[mask].astype("float32")
times = Xs["time"].values[mask]
print(f"Final coastal samples : {len(y_np):,}")

# ─── temporal split ───────────────────────────────────────────────
train = times < np.datetime64("2020-01-01")
val   = (times >= np.datetime64("2020-01-01")) & (times < np.datetime64("2021-01-01"))
test  = times >= np.datetime64("2021-01-01")
def _sub(m): idx=np.where(m)[0]; return X_np[idx], y_np[idx]
X_tr,y_tr = _sub(train); X_va,y_va = _sub(val); X_te,y_te = _sub(test)
print("Split sizes train/val/test :", [len(y_tr), len(y_va), len(y_te)])

# ╭───────────────────────── XGB Regressor ──────────────────────────╮

best_params = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.015804704867439994,
    "subsample": 0.9885668717366809,
    "colsample_bytree": 0.8659690419027274,
    "min_child_weight": 4.1935572069285945,
    "gamma": 2.1888927171422337,
    "reg_lambda": 0.6880408368756581
}

# ─── Regressor fit & metrics ──────────────────────────────────────
reg = XGBRegressor(**best_params,
                   tree_method="hist", n_jobs=-1,
                   eval_metric="rmse", early_stopping_rounds=30)
reg.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

y_hat = reg.predict(X_te)
rmse_lg  = np.sqrt(mean_squared_error(y_te, y_hat))
mae_lg   = mean_absolute_error(y_te, y_hat)
r2       = r2_score(y_te, y_hat)
rmse_lin = 10**rmse_lg - 1
print(f"[Reg] RMSE(log10)={rmse_lg:.3f}  RMSE(µg/L)≈{rmse_lin:.2f}  MAE={mae_lg:.3f}  R²={r2:.3f}")

joblib.dump(reg, OUT_DIR / "xgb_chla_reg.joblib")

# ─── Classifier fit & metrics ─────────────────────────────────────
y_tr_bin = (10**y_tr > HAB_THR).astype(int)
y_va_bin = (10**y_va > HAB_THR).astype(int)
y_te_bin = (10**y_te > HAB_THR).astype(int)

pos_w = ((1 - y_tr_bin).sum() / y_tr_bin.sum()) if cfg.get("pos_weight","auto")=="auto" else float(cfg["pos_weight"])
clf = XGBClassifier(**best_params, scale_pos_weight=pos_w,
                    tree_method="hist", n_jobs=-1,
                    eval_metric="logloss", early_stopping_rounds=30)
clf.fit(X_tr, y_tr_bin, eval_set=[(X_va, y_va_bin)], verbose=False)

prob_te = clf.predict_proba(X_te)[:, 1]        # ← now defined

# --- metrics ------------------------------------------------------
roc_auc = roc_auc_score(y_te_bin, prob_te)
pr_auc  = average_precision_score(y_te_bin, prob_te)

prob_va = clf.predict_proba(X_va)[:, 1]
prec, rec, thr = precision_recall_curve(y_va_bin, prob_va)
f1_va   = 2*prec*rec / (prec + rec + 1e-9)
best_thr = thr[np.argmax(f1_va)]
pred_bin = (prob_te >= best_thr).astype(int)
report   = classification_report(y_te_bin, pred_bin, digits=3)
cm       = confusion_matrix(y_te_bin, pred_bin)

print("\n[Clf] optimum threshold (val) =", round(best_thr,3))
print("[Clf] ROC-AUC =", round(roc_auc,3), " PR-AUC =", round(pr_auc,3))
print(report)

joblib.dump(clf, OUT_DIR / "xgb_hab_clf.joblib")

# ─── save metrics table ───────────────────────────────────────────
df = pd.DataFrame([{
    "rmse_log": rmse_lg, "mae_log": mae_lg, "rmse_lin": rmse_lin,
    "r2": r2, "roc_auc": roc_auc, "pr_auc": pr_auc,
    "thr": best_thr,
    "tp": cm[1,1], "fp": cm[0,1], "tn": cm[0,0], "fn": cm[1,0]
}])
df.to_csv(OUT_DIR / "metrics.tsv", sep="\t", index=False)
print(f"→ metrics written to {OUT_DIR / 'metrics.tsv'}")

'''
micromamba activate habs
cd ~/Desktop/habs-forecast
python scripts/train_baseline.py      # new 2021 metrics
python scripts/cv_xgb.py              # updated cross-val summary
'''