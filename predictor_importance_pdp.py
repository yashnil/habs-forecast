#!/usr/bin/env python3
# analysis/predictor_importance_pdp_scatter.py
"""
Make SHAP, PDP, and bivariate scatter/anomaly plots for top predictors.

Inputs
------
- Freeze NetCDF with variables you trained on (e.g., HAB_convLSTM_core_v1_clean.nc),
  including: log_chl, Kd_490, nflh, u10, v10, wind_speed, tau_mag, avg_sdswrf,
  tp, t2m, d2m, uo, vo, cur_speed, cur_div, cur_vort, zos, ssh_grad_mag, so,
  thetao, chl_anom_monthly, chl_roll*, river_rank, dist_river_km, ocean_mask_static,
  and (optionally) pixel_ok.
- (Optional) A pre-trained XGBoost model path to skip fitting.

Outputs
-------
- figs/shap_beeswarm.png
- figs/pdp_<feature>.png (one per top feature)
- figs/scatter_<feature>_{raw,anom}.png (hexbin + trend)
- figs/pdp_grid_topK.png (compact multi-panel)
- data/shap_values_topK.parquet (for reproducibility)

Notes
-----
- Uses a 1-step (8-day) lead target: y = log_chl(t+1).
- Downsamples training rows to keep runtime manageable; adjust N_* constants.

Running
-----
python predictor_importance_pdp.py \
  --freeze /Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc \
  --model_out models/xgb_tabular.json \
  --outdir out/predictor_importance \
  --lead 1 \
  --topk 6 \
  --force_features "Kd_490,dist_river_km,river_rank,thetao,avg_sdswrf"
"""

from __future__ import annotations
import pathlib, argparse, warnings, json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import shap
import xgboost as xgb

# ------------------------------- config -------------------------------------
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos","ssh_grad_mag","so","thetao"]
DERIVED   = ["chl_anom_monthly","chl_roll24d_mean","chl_roll24d_std",
             "chl_roll40d_mean","chl_roll40d_std"]
STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]
ALL_VARS  = SATELLITE + METEO + OCEAN + DERIVED + STATIC

TOP_K = 6            # how many features to plot in PDP/Scatter
N_TRAIN_SAMP = 500_000   # max sampled rows for training
N_TEST_SAMP  = 250_000   # max sampled rows for SHAP/PDP/Scatter on test
RNG_SEED = 42

NICE_NAME = {
    "log_chl":          "Chlorophyll (log of mg m⁻³)",
    "Kd_490":           "Water clarity Kd₄₉₀ (m⁻¹)",
    "nflh":             "Fluorescence nFLH (mW cm⁻² µm⁻¹ sr⁻¹)",
    "u10":              "10 m wind U (m s⁻¹)",
    "v10":              "10 m wind V (m s⁻¹)",
    "wind_speed":       "Wind speed (m s⁻¹)",
    "tau_mag":          "Wind stress |τ| (N m⁻²)",
    "avg_sdswrf":       "Downward shortwave (W m⁻²)",
    "tp":               "Total precipitation (m)",
    "t2m":              "Air temperature 2 m (K)",
    "d2m":              "Dew point 2 m (K)",
    "uo":               "Surface current U (m s⁻¹)",
    "vo":               "Surface current V (m s⁻¹)",
    "cur_speed":        "Current speed (m s⁻¹)",
    "cur_div":          "Horizontal divergence (s⁻¹)",
    "cur_vort":         "Relative vorticity (s⁻¹)",
    "zos":              "Sea surface height (m)",
    "ssh_grad_mag":     "SSH gradient |∇SSH| (m m⁻¹)",
    "so":               "Sea surface salinity (psu)",
    "thetao":           "Sea surface temperature (°C)",
    "chl_anom_monthly": "Chlorophyll anomaly (mg m⁻³)",
    "chl_roll24d_mean": "Chl 24 d mean (mg m⁻³)",
    "chl_roll24d_std":  "Chl 24 d std (mg m⁻³)",
    "chl_roll40d_mean": "Chl 40 d mean (mg m⁻³)",
    "chl_roll40d_std":  "Chl 40 d std (mg m⁻³)",
    "river_rank":       "River influence index (unitless)",
    "dist_river_km":    "Distance to river (km)",
    "ocean_mask_static":"Ocean mask (unitless)"
}

# ------------------------------- helpers ------------------------------------

def _get_pdp_xy(model, X, j, grid_resolution=60):
    # scikit-learn compatibility shim: values vs grid_values, average vs averages
    pdp = partial_dependence(model, X, [j], kind="average", grid_resolution=grid_resolution)
    xkey = "values" if "values" in pdp else "grid_values"
    ykey = "average" if "average" in pdp else "averages"
    xs = pdp[xkey][0]
    ys = pdp[ykey][0]
    return xs, ys

def time_masks(times):
    t = np.asarray(times)
    train = t <  np.datetime64("2016-01-01")
    val   = (t >= np.datetime64("2016-01-01")) & (t <= np.datetime64("2018-12-31"))
    test  = t >  np.datetime64("2018-12-31")
    return train, val, test

def build_pixel_ok(ds, thresh=0.20, min_depth_m=10.0):
    pix_ok = ds["pixel_ok"].astype(bool) if "pixel_ok" in ds else \
             (np.isfinite(ds["log_chl"]).sum("time")/ds.sizes["time"] >= thresh)
    if "depth" in ds:
        pix_ok = pix_ok & (ds["depth"] > min_depth_m)
    return pix_ok

def stack_to_rows(ds, feat_vars, target_var="log_chl", lead=1, mask=None):
    if mask is None:
        mask = xr.ones_like(ds[feat_vars[0]].isel(time=0), dtype=bool)

    feats = []
    for v in feat_vars:
        arr = ds[v].reset_coords(drop=True)  # <-- add this
        if "time" in arr.dims:
            feats.append(arr.isel(time=slice(0, arr.sizes["time"]-lead)))
        else:
            feats.append(arr.expand_dims(time=ds.time.isel(time=slice(0, ds.sizes["time"]-lead))))

    y = ds[target_var].isel(time=slice(lead, ds.sizes["time"])).reset_coords(drop=True)  # <-- and here

    time_feat = ds.time.isel(time=slice(0, ds.sizes["time"]-lead))

    data_vars = {f"{v}": f for v,f in zip(feat_vars, feats)}
    data_vars["y_log_next"] = y
    combo = xr.Dataset(data_vars).transpose("time","lat","lon")

    mask3d = mask.expand_dims(time=time_feat)
    combo = combo.where(mask3d)

    df = combo.to_dataframe().reset_index(drop=False)
    df = df.dropna(axis=0, how="any")
    return df

def add_anomalies(df, feat_list, month_series):
    out = df.copy()
    out["month"] = month_series.values
    for v in feat_list:
        if v not in out.columns: continue
        # monthly climatology over TRAIN only added later; here, simple per-month z-score for visualization
        grp = out.groupby("month")[v]
        mu = grp.transform("mean")
        sd = grp.transform("std").replace(0, np.nan)
        out[f"{v}_anom"] = (out[v] - mu) / sd
    return out

def subsample_df(df, nmax, rng=RNG_SEED):
    if len(df) <= nmax: return df
    return df.sample(n=nmax, random_state=rng)

def ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# ------------------------------- plotting -----------------------------------
def hex_scatter(ax, x, y, xlabel, ylabel, title):
    hb = ax.hexbin(x, y, gridsize=60, bins='log', mincnt=1)
    limx = (np.nanpercentile(x, 1), np.nanpercentile(x, 99))
    limy = (np.nanpercentile(y, 1), np.nanpercentile(y, 99))
    ax.set_xlim(limx); ax.set_ylim(limy)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9)
    return hb

# Scatter panels
def plot_scatter_panels(df, feature, outdir):
    fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    xlab = NICE_NAME.get(feature, feature)
    hb0 = hex_scatter(axs[0], df[feature].values, df["y_log_next"].values,
                      xlab, "log chl (t+8d)", f"{xlab} vs. log chl (raw)")
    fig.colorbar(hb0, ax=axs[0], shrink=0.8)
    anom_col = f"{feature}_anom"
    if anom_col in df:
        hb1 = hex_scatter(axs[1], df[anom_col].values, df["y_log_next"].values,
                          f"{xlab} anomaly (z)", "log chl (t+8d)", f"{xlab} (anom) vs. log chl")
        fig.colorbar(hb1, ax=axs[1], shrink=0.8)
    fig.savefig(outdir / f"scatter_{feature}.png", dpi=200)
    plt.close(fig)

def plot_pdp(model, X, feat_names, feature, outdir):
    j = feat_names.index(feature)
    xs, ys = _get_pdp_xy(model, X, j, grid_resolution=60)
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(xs, ys, lw=2)
    ax.set_xlabel(NICE_NAME.get(feature, feature))
    ax.set_ylabel("PD: E[log chl (t+8d) | feature]")
    ax.set_title(f"PDP: {NICE_NAME.get(feature, feature)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / f"pdp_{feature}.png", dpi=200)
    plt.close(fig)

def plot_pdp_grid(model, X, feat_names, features, outdir):
    n = len(features)
    cols = min(3, n); rows = int(np.ceil(n/cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), constrained_layout=True)
    axs = np.atleast_2d(axs)
    for k, feat in enumerate(features):
        r, c = divmod(k, cols)
        j = feat_names.index(feat)
        xs, ys = _get_pdp_xy(model, X, j, grid_resolution=60)
        axs[r, c].plot(xs, ys, lw=2)
        axs[r, c].set_title(NICE_NAME.get(feat, feat), fontsize=10)
        axs[r, c].set_xlabel(NICE_NAME.get(feat, feat))
        axs[r, c].set_ylabel("E[log chl (t+8d)]")
    for k in range(n, rows*cols):
        r, c = divmod(k, cols)
        axs[r, c].axis("off")
    fig.suptitle("Partial dependence (top predictors)", y=1.02)
    fig.savefig(outdir / "pdp_grid_topK.png", dpi=200)
    plt.close(fig)

# ------------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True,
                    help="Path to freeze NetCDF (e.g., HAB_convLSTM_core_v1_clean.nc)")
    ap.add_argument("--model_out", default="models/xgb_tabular.json",
                    help="Where to save/load XGB model")
    ap.add_argument("--load_only", action="store_true",
                    help="Skip training and just load model_out")
    # argparse
    ap.add_argument(
        "--force_features",
        type=str, default="",
        help="Comma-separated features to use for SHAP/PDP/scatter display. "
            "If set, overrides SHAP topK selection."
    )
    ap.add_argument("--outdir", default="figs_predictor_importance")
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--topk", type=int, default=TOP_K)
    args = ap.parse_args()

    freeze = pathlib.Path(args.freeze).expanduser().resolve()
    outdir = pathlib.Path(args.outdir).expanduser().resolve()
    figdir = outdir / "figs"
    datadir = outdir / "data"
    ensure_dir(figdir); ensure_dir(datadir); ensure_dir(pathlib.Path(args.model_out).parent)

    print("[load] opening dataset …")
    ds = xr.open_dataset(freeze, chunks={"time": 1})
    feat_vars = [v for v in ALL_VARS if v in ds.data_vars]

    pix_ok = build_pixel_ok(ds)
    df_all = stack_to_rows(ds, feat_vars, target_var="log_chl", lead=args.lead, mask=pix_ok)

    # attach month for anomaly calc
    # align with features' time: times 0..T-lead-1 → y at +lead
    times_feat = pd.to_datetime(ds.time.values[:-args.lead])
    df_all = df_all.reset_index(drop=True)
    month_series = pd.to_datetime(df_all["time"]).dt.month
    df_all = add_anomalies(df_all, feat_vars, month_series=month_series)

    t_full  = pd.to_datetime(ds.time.values)
    dt      = t_full[1] - t_full[0]             # 1×8‑day step in your freeze
    feat_t  = pd.to_datetime(df_all["time"])    # feature time per row
    y_times = feat_t + dt * args.lead           # target time per row (t+lead)

    train_mask, val_mask, test_mask = time_masks(y_times.values)
    df_train = df_all[train_mask].copy()
    df_val   = df_all[val_mask].copy()
    df_test  = df_all[test_mask].copy()

    # subsample to keep things tractable
    df_train = subsample_df(df_train, N_TRAIN_SAMP)
    df_test  = subsample_df(df_test,  N_TEST_SAMP)

    X_train = df_train[feat_vars].values
    y_train = df_train["y_log_next"].values
    X_test  = df_test[feat_vars].values
    y_test  = df_test["y_log_next"].values

    # --------------------------- model --------------------------------------
    model_path = pathlib.Path(args.model_out)
    if args.load_only and model_path.is_file():
        print(f"[model] loading: {model_path}")
        xgbm = xgb.XGBRegressor()
        xgbm.load_model(str(model_path))
    else:
        print("[model] training XGBRegressor …")
        xgbm = xgb.XGBRegressor(
            n_estimators=600,
            max_depth=8,
            subsample=0.7,
            colsample_bytree=0.8,
            learning_rate=0.05,
            reg_lambda=1.0,
            random_state=RNG_SEED,
            n_jobs=8,
            tree_method="hist"
        )
        xgbm.fit(X_train, y_train,
                 eval_set=[(X_train, y_train)],
                 verbose=False)
        xgbm.save_model(str(model_path))

    # quick sanity
    y_hat = xgbm.predict(X_test)
    print(f"[eval] R^2 (test): {r2_score(y_test, y_hat):.3f}")

    # --------------------------- SHAP ---------------------------------------
    print("[shap] computing SHAP values on test sample …")
    explainer = shap.TreeExplainer(xgbm)
    shap_vals = explainer.shap_values(X_test)
    shap_abs_mean = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(shap_abs_mean)[::-1][:args.topk]
    top_features = [feat_vars[i] for i in top_idx]
    # after computing `top_features`
    forced = [s.strip() for s in args.force_features.split(",") if s.strip()]
    if forced:
        missing = [f for f in forced if f not in feat_vars]
        if missing:
            raise ValueError(f"--force_features includes unknown vars: {missing}")
        chosen_features = forced
    else:
        chosen_features = top_features

    print("[features] using for plots:", chosen_features)
    print("[features] pretty:", [NICE_NAME.get(f, f) for f in chosen_features])
    print("[shap] top features:", top_features)

    # beeswarm
   # SHAP beeswarm with pretty labels
    nice_feat_names = [NICE_NAME.get(v, v) for v in feat_vars]
    nice_feat_names = [NICE_NAME.get(v, v) for v in feat_vars]

    if forced:
        idx = [feat_vars.index(f) for f in chosen_features]
        shap.summary_plot(
            shap_vals[:, idx],
            features=X_test[:, idx],
            feature_names=[NICE_NAME.get(f, f) for f in chosen_features],
            show=False, max_display=len(chosen_features)
        )
    else:
        shap.summary_plot(
            shap_vals,
            features=X_test,
            feature_names=nice_feat_names,
            show=False, max_display=args.topk
        )

    plt.tight_layout(); plt.savefig(figdir/"shap_beeswarm.png", dpi=200); plt.close()

    # save SHAP table
    shap_df = pd.DataFrame({"feature": feat_vars, "mean_abs_shap": shap_abs_mean})
    shap_df.sort_values("mean_abs_shap", ascending=False).to_parquet(datadir/"shap_values_topK.parquet")

    print("[pdp] plotting PDPs …")
    for f in chosen_features:
        plot_pdp(xgbm, X_test, feat_vars, f, figdir)
    plot_pdp_grid(xgbm, X_test, feat_vars, chosen_features, figdir)

    print("[scatter] making raw & anomaly hexbin scatter …")
    df_test_local = df_test.copy()
    df_test_local["y_time"] = pd.Series(y_times, index=df_all.index).loc[df_test.index]
    for f in chosen_features:
        plot_scatter_panels(df_test_local, f, figdir)

    # summary
    meta = {
        "freeze": str(freeze),
        "model_path": str(model_path),
        "lead": args.lead,
        "top_features": top_features,
        "n_train": int(len(df_train)),
        "n_test":  int(len(df_test)),
    }
    json.dump(meta, open(outdir/"summary.json","w"), indent=2)
    print(f"[done] figures → {figdir}\n       metadata → {outdir/'summary.json'}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
