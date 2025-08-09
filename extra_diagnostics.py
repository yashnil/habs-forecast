#!/usr/bin/env python3
"""
extra_diagnostics.py – NEW plots only
-------------------------------------
✓ SHAP beeswarm (surrogate RandomForest) w/ nicenames + option to drop engineered CHL features
✓ Reliability curve for bloom ≥5 mg m⁻³ (Brier score + CSV)
✓ Q–Q plot (linear chl)  [kept, but feel free to ignore]
✓ Bland–Altman (log chl)
✓ Lead-time skill curve 8/16/24/32 d (across models) + per-model RMSE CSV
✓ Violin plots of error distribution vs lead
✓ Top-5 predictor scatter (Observed chlorophyll vs predictor) with latitude-band stratification  [PINN only]
Run:  python extra_diagnostics.py
"""

# ── imports ───────────────────────────────────────────────────────────
import pathlib, warnings
import numpy as np, pandas as pd, xarray as xr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, brier_score_loss
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
import shap
from scipy.special import expit  # logistic σ()
# after your imports
import re
import matplotlib as mpl
mpl.rcParams['font.family'] = 'DejaVu Sans'           # safe default
mpl.rcParams['mathtext.fontset'] = 'dejavusans'
mpl.rcParams['mathtext.default'] = 'regular'

SUPERS = {
    "⁻¹": "$^{-1}$",
    "⁻²": "$^{-2}$",
    "⁻³": "${-3}$",  # not strictly needed here but keep consistent
    "₄₉₀": "$_{490}$",
    "°C": r"$^\circ$C",
    "τ": r"$\tau$",
    "µ": r"$\mu$",
}
def to_mathtext(s: str) -> str:
    # targeted replacements
    for k, v in SUPERS.items():
        s = s.replace(k, v)
    # also fix patterns like 'm s^-1' if you ever use caret in text
    s = re.sub(r'\^-(\d+)', r'$^{-\1}$', s)
    return s

# ── file registry ─────────────────────────────────────────────────────
ROOT = pathlib.Path.cwd()
MODELS = {
    "Vanilla": dict(
        ckpt = pathlib.Path("~/HAB_Models/vanilla_best.pt").expanduser(),
        pred = ROOT / "Diagnostics_ConvLSTM/predicted_fields.nc"),
    "PINN": dict(
        ckpt = pathlib.Path("~/HAB_Models/convLSTM_best.pt").expanduser(),
        pred = ROOT / "Diagnostics_PINN/predicted_fields.nc"),
    "TFT": dict(
        ckpt = pathlib.Path("~/HAB_Models/convTFT_best.pt").expanduser(),
        pred = ROOT / "Diagnostics_TFT/predicted_fields.nc"),
}

FLOOR = 0.056616           # mg m-3
LEADS = [1,2,3,4]          # 8,16,24,32 d
OUT_ROOT = ROOT / "Diagnostics_Combo_New"; OUT_ROOT.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# ── predictor lists (same as freezer) ─────────────────────────────────
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos","ssh_grad_mag","so","thetao"]
DERIVED   = ["chl_anom_monthly",
             "chl_roll24d_mean","chl_roll24d_std",
             "chl_roll40d_mean","chl_roll40d_std"]
STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]
ALL_VARS  = SATELLITE + METEO + OCEAN + DERIVED + STATIC

# ── nicenames for every feature (with units) ─────────────────────────
NICE_NAME = {
    # Satellite / optics
    "log_chl":          "Chlorophyll (log of mg m⁻³)",        # log itself is unitless, but of mg m⁻³
    "Kd_490":           "Water clarity Kd₄₉₀ (m⁻¹)",
    "nflh":             "Fluorescence nFLH (mW cm⁻² µm⁻¹ sr⁻¹)",

    # Meteorology (ERA5-style defaults)
    "u10":              "10 m wind U (m s⁻¹)",
    "v10":              "10 m wind V (m s⁻¹)",
    "wind_speed":       "Wind speed (m s⁻¹)",
    "tau_mag":          "Wind stress |τ| (N m⁻²)",
    "avg_sdswrf":       "Downward shortwave (W m⁻²)",
    "tp":               "Total precipitation (m)", 
    "t2m":              "Air temperature 2 m (K)",  
    "d2m":              "Dew point 2 m (K)",

    # Ocean / dynamics (CMEMS-style defaults)
    "uo":               "Surface current U (m s⁻¹)",
    "vo":               "Surface current V (m s⁻¹)",
    "cur_speed":        "Current speed (m s⁻¹)",
    "cur_div":          "Horizontal divergence (s⁻¹)",
    "cur_vort":         "Relative vorticity (s⁻¹)",
    "zos":              "Sea surface height (m)",
    "ssh_grad_mag":     "SSH gradient |∇SSH| (m m⁻¹)",

    # Hydro/thermo
    "so":               "Sea surface salinity (psu)",
    "thetao":           "Sea surface temperature (°C)",

    # Derived chl history (linear space)
    "chl_anom_monthly": "Chlorophyll anomaly (mg m⁻³)",
    "chl_roll24d_mean": "Chl 24 d mean (mg m⁻³)",
    "chl_roll24d_std":  "Chl 24 d std (mg m⁻³)",
    "chl_roll40d_mean": "Chl 40 d mean (mg m⁻³)",
    "chl_roll40d_std":  "Chl 40 d std (mg m⁻³)",

    # Static
    "river_rank":       "River influence index (unitless)",
    "dist_river_km":    "Distance to river (km)",
    "ocean_mask_static":"Ocean mask (unitless)"
}


# path to the freeze with predictors
FREEZE_PATH = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc"
).expanduser()

# ── utils ─────────────────────────────────────────────────────────────
def _save(fig, path): fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)

def bloom_prob(chl_mg, method="logistic", scale=5.0):
    """Convert chlorophyll (mg m⁻³) to a pseudo-probability in [0,1]."""
    if method == "linear":
        return np.clip(chl_mg / scale, 0.0, 1.0)
    elif method == "logistic":
        TH = 5.0
        return expit((chl_mg - TH) / scale)
    else:
        raise ValueError("method must be 'linear' or 'logistic'")

# ── SHAP surrogate (beeswarm) + samples for downstream plots ─────────
def shap_surrogate_and_samples(
    ds_pred,
    freeze_path,
    out_png,
    n_samples=4000,
    topk=20,
    seed=0,
    drop_engineered=True,
    drop_log_chl=True,
    also_save_bar=True,
    verbose=True
):
    """
    Fit a small RandomForest on FREEZE predictors -> ds_pred.log_chl_pred.
    Return (rf, X, y_pred, y_obs, raw_names, pretty_names, lat_vals)
    and write SHAP beeswarm + (optional) bar plot + CSV of mean|SHAP|.
    """
    if verbose: print("[SHAP] opening:", str(freeze_path))
    dsf = xr.open_dataset(freeze_path)

    # choose features
    feat_names = [v for v in ALL_VARS if v in dsf.data_vars]
    if drop_engineered:
        ban = {"chl_anom_monthly", "chl_roll24d_mean","chl_roll24d_std",
               "chl_roll40d_mean","chl_roll40d_std"}
        feat_names = [v for v in feat_names if v not in ban]
    if drop_log_chl:
        feat_names = [v for v in feat_names if v != "log_chl"]

    if not feat_names:
        warnings.warn("No predictors found after filtering; skipping SHAP.")
        return None

    pretty_names = [to_mathtext(NICE_NAME.get(v, v)) for v in feat_names]
    
    if verbose:
        print(f"[SHAP] features kept: {len(feat_names)} -> {feat_names[:10]}{' …' if len(feat_names)>10 else ''}")

    # sampling valid points
    vm = ds_pred.valid_mask.astype(bool).values
    pred_ok = np.isfinite(ds_pred.log_chl_pred.values)
    good = vm & pred_ok
    if not good.any():
        warnings.warn("No valid points for sampling; skipping SHAP.")
        return None

    T, H, W = good.shape
    rng = np.random.default_rng(seed)
    want = min(n_samples, int(good.sum()))
    coords = []
    tries = 0
    while len(coords) < want and tries < 50 * want:
        t = int(rng.integers(0, T)); y = int(rng.integers(0, H)); x = int(rng.integers(0, W))
        if good[t, y, x]:
            coords.append((t, y, x))
        tries += 1
    if not coords:
        warnings.warn("Sampler failed to collect valid points; skipping SHAP.")
        return None

    # build matrices
    X = np.zeros((len(coords), len(feat_names)), dtype=np.float32)
    y_pred = np.zeros(len(coords), dtype=np.float32)
    y_obs  = np.zeros(len(coords), dtype=np.float32)
    lat_vals = np.zeros(len(coords), dtype=np.float32)

    for i, (t, yy, xx) in enumerate(coords):
        for j, v in enumerate(feat_names):
            da = dsf[v]
            val = da.isel(time=t, lat=yy, lon=xx).values if "time" in da.dims else da.isel(lat=yy, lon=xx).values
            if not np.isfinite(val):
                val = float(da.mean(skipna=True))
            X[i, j] = float(val)
        y_pred[i] = float(ds_pred.log_chl_pred.isel(time=t, lat=yy, lon=xx).values)
        # observed chl in linear space (mg m^-3)
        obs_log = float(ds_pred.log_chl_true.isel(time=t, lat=yy, lon=xx).values)
        y_obs[i] = max(np.exp(obs_log) - FLOOR, 0.0)
        lat_vals[i] = float(ds_pred.lat.values[yy])

    # fit RF + TreeSHAP
    if verbose: print(f"[SHAP] fitting RandomForest: {X.shape} → {y_pred.shape}")
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, n_jobs=-1, random_state=seed)
    rf.fit(X, y_pred)
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X, check_additivity=False)

    # beeswarm
    plt.figure(figsize=(7.8, 4.8))
    shap.summary_plot(sv, features=X, feature_names=pretty_names, max_display=topk,
                      show=False, plot_type="dot")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220); plt.close()
    if verbose: print("[SHAP] wrote", str(out_png))

    # optional bar
    if also_save_bar:
        plt.figure(figsize=(6.6, 3.0))
        shap.summary_plot(sv, features=X, feature_names=pretty_names,
                          max_display=topk, show=False, plot_type="bar")
        plt.tight_layout()
        plt.savefig(str(pathlib.Path(out_png).with_name(pathlib.Path(out_png).stem + "_bar.png")), dpi=220)
        plt.close()

    # CSV (mean|SHAP|)
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(mean_abs)[::-1][:topk]
    pd.DataFrame({
        "feature_raw": np.array(feat_names)[order],
        "feature_pretty": np.array(pretty_names)[order],
        "mean_abs_shap": mean_abs[order]
    }).to_csv(pathlib.Path(out_png).with_suffix(".csv"), index=False)

    return rf, X, y_pred, y_obs, feat_names, pretty_names, lat_vals

# ── Top-5 predictor scatter vs observed chl (lat stratified) ─────────
def plot_top5_feature_scatter_binned(
    rf, X, y_obs, lat_vals, feat_raw, feat_pretty, outdir, n_bins=20
):
    """
    For the RF top-5 features, plot Observed chlorophyll (mg m⁻³)
    vs predictor value, binned into n_bins, stratified by four
    latitude bands (quartiles).
    """
    imp = rf.feature_importances_
    top5 = np.argsort(imp)[::-1][:5]

    # make 4 roughly equal lat bands (quartiles)
    q = np.quantile(lat_vals, [0.25, 0.5, 0.75])
    bands = np.digitize(lat_vals, q)  # 0..3
    band_labels = [
        f"<{q[0]:.1f}°",
        f"{q[0]:.1f}–{q[1]:.1f}°",
        f"{q[1]:.1f}–{q[2]:.1f}°",
        f">={q[2]:.1f}°"
    ]
    colors = ["#D62728", "#FF7F0E", "#2CA02C", "#1F77B4"]

    # save band edges for methods section
    pd.DataFrame({"band": list(range(4)), "label": band_labels}).to_csv(outdir/"lat_bands.csv", index=False)

    for rank, f_idx in enumerate(top5, start=1):
        x_all = X[:, f_idx]
        edges = np.linspace(np.nanmin(x_all), np.nanmax(x_all), n_bins+1)
        centers = 0.5*(edges[:-1]+edges[1:])

        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        for b in range(4):
            m = bands == b
            if not np.any(m): continue
            # nonparametric association per band
            rho, p = spearmanr(x_all[m], y_obs[m])
            p_label = "p<0.01" if (p < 0.01) else (f"p={p:.2g}" if np.isfinite(p) else "p=NA")

            # bin & summarize
            ymu, ysd, xv = [], [], []
            for i in range(n_bins):
                sel = m & (x_all>=edges[i]) & (x_all<edges[i+1])
                if not np.any(sel): continue
                ymu.append(float(np.nanmean(y_obs[sel])))
                ysd.append(float(np.nanstd (y_obs[sel], ddof=0)))
                xv.append(centers[i])
            if not xv: continue
            ax.errorbar(xv, ymu, yerr=ysd, fmt='o-', ms=4, lw=1.25,
                        color=colors[b], alpha=0.85,
                        label=f"{band_labels[b]} (ρ={rho:.2f}, {p_label})")

        ax.set_xlabel(to_mathtext(feat_pretty[f_idx]))
        ax.set_ylabel("Observed chlorophyll (mg m$^{-3}$)")
        ax.set_title(f"Top-{rank}: {feat_pretty[f_idx]}")
        ax.legend(fontsize="x-small", ncols=2)
        _save(fig, outdir / f"top5_scatter_{rank}_{feat_raw[f_idx]}.png")

# ── main per-model diagnostics ────────────────────────────────────────
def main():
    lead_curves = {}

    for tag, cfg in MODELS.items():
        print(f"▶  {tag}")
        outdir = OUT_ROOT / tag; outdir.mkdir(exist_ok=True)

        ds       = xr.open_dataset(cfg["pred"])
        obs_log  = ds.log_chl_true
        pred_log = ds.log_chl_pred
        mask     = ds.valid_mask.astype(bool)
        pers_log = ds.log_chl_pers if "log_chl_pers" in ds else obs_log.shift(time=1)

        # linear
        obs_lin  = np.clip(np.exp(obs_log)  - FLOOR, 0, None)
        pred_lin = np.clip(np.exp(pred_log) - FLOOR, 0, None)
        pers_lin = np.clip(np.exp(pers_log) - FLOOR, 0, None)
        flat = lambda da: da.where(mask).values.flatten()

        # 1) SHAP + Top-5 scatter (PINN only)
        if tag == "PINN":
            res = shap_surrogate_and_samples(
                ds_pred=ds,
                freeze_path=FREEZE_PATH,
                out_png=outdir / "shap_beeswarm.png",
                n_samples=4000,
                topk=20,
                seed=0,
                drop_engineered=True,   # main fig excludes chl_roll* / anomaly
                drop_log_chl=True,      # exclude log_chl from importance
                also_save_bar=True,
                verbose=True
            )
            if res is not None:
                rf, X, y_pred, y_obs, feat_raw, feat_pretty, lat_vals = res
                # optional: SI version including engineered features
                shap_surrogate_and_samples(
                    ds_pred=ds, freeze_path=FREEZE_PATH,
                    out_png=outdir / "shap_beeswarm_with_engineered.png",
                    n_samples=3000, topk=20, seed=1,
                    drop_engineered=False, drop_log_chl=True,
                    also_save_bar=False, verbose=False
                )
                # Top-5 scatter (Observed vs predictor)
                plot_top5_feature_scatter_binned(
                    rf, X, y_obs, lat_vals, feat_raw, feat_pretty, outdir, n_bins=20
                )

        # 2) Reliability curve for bloom ≥ 5 mg m⁻³
        thresh = 5.0
        prob_model = bloom_prob(pred_lin, method="logistic", scale=2.5)
        prob_pers  = bloom_prob(pers_lin,  method="logistic", scale=2.5)

        y_true = (obs_lin >= thresh).where(mask).values.flatten().astype(int)
        pm = prob_model.where(mask).values.flatten()
        pp = prob_pers.where(mask).values.flatten()
        good = np.isfinite(y_true)&np.isfinite(pm)&np.isfinite(pp)
        y_true, pm, pp = y_true[good], pm[good], pp[good]
        bins = np.linspace(0,1,11); digit = np.digitize(pm, bins)-1
        obs_freq = [y_true[digit==i].mean() if np.any(digit==i) else np.nan for i in range(len(bins)-1)]
        fig,ax=plt.subplots(figsize=(4,4))
        ax.plot(bins[:-1]+.05, obs_freq, "-o", label="Model")
        ax.plot([0,1],[0,1],'k--',lw=.6)
        ax.set_xlabel("Forecast probability"); ax.set_ylabel("Observed frequency")
        ax.set_title(f"{tag} – bloom reliability")
        _save(fig, outdir/"reliability_bloom.png")
        pd.DataFrame(dict(bin_mid=bins[:-1]+.05, obs_freq=obs_freq)).to_csv(outdir/"metrics_reliability.csv", index=False)
        try:
            bs = brier_score_loss(y_true, pm); print(f"   Brier score: {bs:.3f}")
        except Exception as e:
            print("   Brier score failed:", e)

        # 3) Q–Q plot (linear)
        q = np.linspace(0.01,0.99,99)
        fig,ax=plt.subplots(figsize=(4,4))
        ax.plot(np.quantile(flat(obs_lin),q), np.quantile(flat(pred_lin),q),label="Model")
        ax.plot(np.quantile(flat(obs_lin),q), np.quantile(flat(pers_lin),q),label="Persistence")
        ax.plot([0,obs_lin.max()],[0,obs_lin.max()],'k--',lw=.6)
        ax.set_xlabel("Observed quantile"); ax.set_ylabel("Predicted quantile")
        ax.set_title(f"{tag} – Q–Q (linear)")
        ax.legend(); _save(fig,outdir/"qq_linear.png")

        # 4) Bland–Altman (log)
        avg = .5*(flat(obs_log)+flat(pred_log))
        diff= flat(pred_log)-flat(obs_log)
        sel = np.isfinite(avg)&np.isfinite(diff)
        fig,ax=plt.subplots(figsize=(5,3))
        ax.scatter(avg[sel], diff[sel], s=5, alpha=.3)
        ax.axhline(diff[sel].mean(), color='r')
        ax.set_xlabel("Mean (log)"); ax.set_ylabel("Pred−Obs (log)")
        ax.set_title(f"{tag} – Bland–Altman")
        _save(fig,outdir/"bland_altman_log.png")

        # 5) Lead-time RMSE & error violins
        lead_rmse=[]; violins=[]
        for k in LEADS:
            p_shift = pred_log[:-k] if k else pred_log
            o_shift = obs_log[k:]  if k else obs_log
            m_shift = mask[k:]     if k else mask
            err = (p_shift - o_shift).where(m_shift).values.flatten()
            err = err[np.isfinite(err)]
            violins.append(err)
            lead_rmse.append(float(np.sqrt(np.nanmean(err**2))))
        lead_curves[tag]=lead_rmse

        pd.DataFrame({"lead_days": [8*l for l in LEADS], "rmse_log": lead_rmse}).to_csv(outdir/"metrics_lead_rmse.csv", index=False)

        fig,ax=plt.subplots(figsize=(5,3))
        parts=ax.violinplot(violins, positions=[8*l for l in LEADS],
                            showmeans=False, showextrema=False)
        for pc in parts['bodies']: pc.set_alpha(0.6)
        ax.set_xlabel("Lead (days)"); ax.set_ylabel("Error (log-chl)")
        ax.set_title(f"{tag} – error distribution vs lead")
        _save(fig,outdir/"err_violin_by_lead.png")

    # cross-model lead curve
    fig,ax=plt.subplots(figsize=(5,3))
    for tag,rm in lead_curves.items():
        ax.plot([8*l for l in LEADS], rm, "-o", label=tag)
    ax.set_xlabel("Lead (days)"); ax.set_ylabel("RMSE (log-chl)")
    ax.set_title("Lead-time skill (all models)"); ax.legend()
    _save(fig, OUT_ROOT/"lead_time_skill.png")

    print("\n✓ NEW diagnostics written to", OUT_ROOT)

if __name__ == "__main__":
    main()
