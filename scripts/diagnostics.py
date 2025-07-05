#!/usr/bin/env python3
# scripts/diagnostics.py
"""
Generates SHAP interpretability plots for the tuned XGBoost regressor.

Outputs
-------
diagnostics/
 ├─ shap_summary.png
 ├─ dep_<feature_1>.png
 ├─ dep_<feature_2>.png
 └─ … (top-5 features)
"""
import warnings, pathlib, joblib, numpy as np, pandas as pd, shap
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ─────────────────────────────────────────────────────────
repo       = pathlib.Path(__file__).resolve().parents[1]
model_path = repo / "Models" / "xgb_chla_reg.joblib"
mat_path   = repo / "Models" / "X_test.npy"           # cached matrix
vars_path  = repo / "Models" / "feature_names.npy"    # cached names
out_dir    = repo / "diagnostics"; out_dir.mkdir(exist_ok=True)

PRED_VARS = [
    "sst","t2m","u10","v10","avg_sdswrf",
    "Kd_490","nflh","so","thetao","uo","vo","zos"
]

# ── 1. Load model ─────────────────────────────────────────────────
reg = joblib.load(model_path)

# ── 2. Load (or rebuild) test-set design matrix ───────────────────
if mat_path.exists() and vars_path.exists():
    X_te       = np.load(mat_path)
    feat_names = np.load(vars_path, allow_pickle=True)
else:
    import xarray as xr, yaml
    from _feature_utils import build_design_matrix

    cfg   = yaml.safe_load(open(repo / "config.yaml"))
    cube  = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"
    ds    = xr.open_dataset(cube)
    X_all, _ = build_design_matrix(
        ds,
        pred_vars=PRED_VARS,
        forecast_lag=int(cfg.get("forecast_lag", 1))
    )

    # helper to stack spatial samples
    def _stack(da):
        extra = [d for d in da.dims if d not in ("time","lat","lon")]
        return (
            da.transpose("time","lat","lon",*extra)
              .stack(sample=("time","lat","lon"))
              .transpose("sample",*extra)
        )

    Xs      = _stack(X_all)
    times   = Xs["time"].values
    tst_idx = times >= np.datetime64("2021-01-01")
    X_te    = Xs.values[tst_idx].astype("float32")
    feat_names = Xs["var"].values

    # cache for next time
    np.save(mat_path, X_te)
    np.save(vars_path, feat_names, allow_pickle=True)

# ── 3. Sub-sample for faster SHAP plotting ────────────────────────
rng  = np.random.default_rng(42)
m    = min(20_000, len(X_te))
subs = rng.choice(len(X_te), size=m, replace=False)
df   = pd.DataFrame(X_te[subs], columns=feat_names)

# ── 4. SHAP analysis & plots ──────────────────────────────────────
explainer   = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(df)
if isinstance(shap_values, list):      # safety for multi-output APIs
    shap_values = shap_values[0]

# 4-a Summary plot
shap.summary_plot(shap_values, df, show=False)
plt.gcf().savefig(out_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# 4-b Dependence plots for the top-5 features by mean |SHAP|
top5_idx = np.argsort(np.abs(shap_values).mean(0))[-5:][::-1]
for i in top5_idx:
    f = feat_names[i]
    shap.dependence_plot(f, shap_values, df, show=False)
    plt.gcf().savefig(out_dir / f"dep_{f}.png", dpi=300, bbox_inches="tight")
    plt.close()

print("✓ SHAP plots saved in", out_dir)
