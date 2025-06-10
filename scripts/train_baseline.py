
#!/usr/bin/env python
# scripts/train_baseline.py
"""
Flatten HAB_cube_2016_2021.nc into (samples × features),
split into train/val/test,
compute pos_weight automatically,
and train an XGBoost regressor + classifier baseline.
"""
import pathlib, yaml, numpy as np, xarray as xr, joblib, pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from xgboost import XGBRegressor, XGBClassifier

# ─────────────────── config & paths ───────────────────
repo_root = pathlib.Path(__file__).resolve().parents[1]
cfg       = yaml.safe_load(open(repo_root / "config.yaml"))
cube_file = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
out_dir   = repo_root / "Models"
out_dir.mkdir(exist_ok=True)

# ─────────────────── load cube ───────────────────
ds = xr.open_dataset(cube_file, chunks={"time": 50})

# select predictors
PRED_VARS = [
    "sst", "t2m", "u10", "v10", "avg_sdswrf",
    "Kd_490", "nflh", "so", "thetao", "uo", "vo", "zos"
]
X = ds[PRED_VARS]
y = ds["log_chl"].shift(time=-cfg["forecast_lag"])

# drop the last lagged slice (now NaN)
X = X.isel(time=slice(None, -cfg["forecast_lag"]))
y = y.isel(time=slice(None, -cfg["forecast_lag"]))

# ─────────────────── stack space → (sample) ───────────────────
X_stack = (
    X.to_array()                          # (variable, time, lat, lon)
      .transpose("time", "lat", "lon", "variable")
      .stack(sample=("time", "lat", "lon"))  #  (variable, sample)
      .transpose("sample", "variable")       #  ← sample axis first
)

y_stack = (
    y.stack(sample=("time", "lat", "lon"))   # dims: (sample)
)


# ── build a concrete boolean mask ──────────────────────────
valid = (
    ~np.isnan(X_stack).any("variable")
    & np.isfinite(y_stack)
).compute()

X_np  = X_stack.values[valid].astype("float32")   # shape: (N, 12)
y_np  = y_stack.values[valid].astype("float32")
times = X_stack["time"].values[valid]

print("Total valid samples:", len(y_np))

# ─────────────────── temporal split ───────────────────
train_mask = times < np.datetime64("2020-01-01")
val_mask   = (times >= np.datetime64("2020-01-01")) & (times < np.datetime64("2021-01-01"))
test_mask  = times >= np.datetime64("2021-01-01")

def subset(mask):
    idx = np.where(mask)[0]
    return X_np[idx], y_np[idx]

X_train, y_train = subset(train_mask)
X_val,   y_val   = subset(val_mask)
X_test,  y_test  = subset(test_mask)

print("Train / val / test:", len(y_train), len(y_val), len(y_test))

# ─────────────────── XGBoost regressor ───────────────────
reg = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="rmse",          # stays here
        early_stopping_rounds=50,    # ← moved up
        verbose=100                  # ← moved up
)

reg.fit(X_train, y_train,
        eval_set=[(X_val, y_val)])    # fit() now ONLY data args

pred = reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2   = r2_score(y_test, pred)
print(f"[Reg] test RMSE (log10 chl) = {rmse:.3f},  R² = {r2:.3f}")

joblib.dump(reg, out_dir / "xgb_chla_reg.joblib")

# ─────────────────── XGBoost HAB classifier ───────────────────
thresh = cfg["hab_threshold"]
y_train_bin = (10**y_train > thresh).astype(int)
y_val_bin   = (10**y_val   > thresh).astype(int)
y_test_bin  = (10**y_test  > thresh).astype(int)

if cfg.get("pos_weight", "auto") == "auto":
    pos_weight = (1 - y_train_bin).sum() / y_train_bin.sum()
else:
    pos_weight = cfg["pos_weight"]

clf = XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(pos_weight),
        tree_method="hist",
        n_jobs=-1,
        eval_metric="logloss",        # …
        early_stopping_rounds=50,     # ← moved up
        verbose=100
)

clf.fit(X_train, y_train_bin,
        eval_set=[(X_val, y_val_bin)])

print("[Clf] pos_weight used:", pos_weight)
print(classification_report(y_test_bin, clf.predict(X_test)))

joblib.dump(clf, out_dir / "xgb_hab_clf.joblib")
