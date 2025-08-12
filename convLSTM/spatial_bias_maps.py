#!/usr/bin/env python3
"""
spatial_bias_maps_plus.py – diagnostic bundle
====================================================
Covers every ask in the latest convo:

1. **Domain‑mean time‑series** (mg m‑3).
2. **Monthly RMSE curves** (linear space, model vs persistence).
3. **Taylor diagram** (monthly anomalies, log space).
4. **Spatial skill maps** – RMSE & % improvement vs persistence.
5. **Spatial mean maps** – Observed, Predicted, **Bias (Pred–Obs)** with land/deep‑ocean transparently masked.
6. **Hex‑bin scatter plot** (obs vs pred, log space) for an at‑a‑glance diagnostic of bias & spread.
7. **Global‑plus metrics** (RMSE, MAE, Bias, Pearson *r*, Spearman ρ, NSE, KGE) in **both** log & linear space **+** a per‑month metrics CSV.

All outputs drop into *OUTDIR* so the folder can be zipped for a manuscript supplement.

python convLSTM/spatial_bias_maps.py
"""
# pylint: disable=invalid-name
import pathlib, json, itertools, warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc
from scipy.stats import spearmanr, pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

# --- UNIVERSAL FIGURE STYLE (matches Blueprint conventions) ---
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": "0.85",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "savefig.dpi": 300,
    "figure.dpi": 200
})


# ────────────────────────────────────────────────────────────────
# paths – edit if needed
# ────────────────────────────────────────────────────────────────
FREEZE = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
PRED   = pathlib.Path(
     "/Users/yashnilmohanty/Desktop/Diagnostics_ConvLSTM/predicted_fields.nc")
OUTDIR = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_ConvLSTM")
OUTDIR.mkdir(exist_ok=True, parents=True)

TILER  = cimgt.GoogleTiles(style='satellite'); TILER.request_timeout = 5
EXTENT = [-125, -114, 31, 43]   # CA coast bbox

# ────────────────────────────────────────────────────────────────
# load & prep
# ────────────────────────────────────────────────────────────────
ds_pred  = xr.open_dataset(PRED)
obs_log  = ds_pred["log_chl_true"].load()      # (time,lat,lon)
pred_log = ds_pred["log_chl_pred"].load()

mask     = ds_pred["valid_mask"].astype(bool).load()

pers_log = ds_pred["log_chl_pers"] if "log_chl_pers" in ds_pred else obs_log.shift(time=1)

FLOOR = 0.056616       # mg m-3


# --- bias‐correction -----------------------------------
# compute the global mean bias in log‐space (only over valid pixels)
mean_bias_log = float(((pred_log - obs_log).where(mask).mean().item()))
print(f"Mean bias (log space): {mean_bias_log:.4f}")

# subtract it to get bias‐corrected predictions
pred_log_bc = pred_log - mean_bias_log

# re-derive linear‐space, floor‐corrected fields
pred_mg_bc = np.clip(np.exp(pred_log_bc) - FLOOR, 0, None)
# -----------------------------------------------------

pred_log = pred_log_bc
pred_mg  = pred_mg_bc

obs_mg  = np.exp(obs_log)  - FLOOR
pred_mg = np.exp(pred_log) - FLOOR
pers_mg = np.exp(pers_log) - FLOOR

obs_mg  = np.clip(obs_mg,  0, None)
pred_mg = np.clip(pred_mg, 0, None)
pers_mg = np.clip(pers_mg, 0, None)

# New plots

# choose threshold in mg m-3 (after floor correction)
THR = 5.0

obs_b  = (obs_mg  >= THR).where(mask)
pred_b = (pred_mg >= THR).where(mask)

hits  = int(((obs_b==1) & (pred_b==1)).sum())
miss  = int(((obs_b==1) & (pred_b==0)).sum())
falarm= int(((obs_b==0) & (pred_b==1)).sum())
corr  = int(((obs_b==0) & (pred_b==0)).sum())

pod   = hits / (hits+miss+1e-9)               # Probability of Detection
far   = falarm / (hits+falarm+1e-9)           # False Alarm Ratio
csi   = hits / (hits+miss+falarm+1e-9)        # Critical Success Index

bloom_row = pd.Series(dict(metric='bloom_≥5mg',
                           hits=hits, miss=miss, false_alarm=falarm,
                           POD=pod, FAR=far, CSI=csi))
bloom_row.to_frame().T.to_csv(OUTDIR/'metrics_bloom_threshold.csv', index=False)
print("\nBloom contingency (≥5 mg m-3):", bloom_row.to_dict())
# 

time = pd.to_datetime(obs_log.time.values)

# 2‑D coastal mask (True = valid ocean pixel)
coast_mask2d = mask.any("time").values

# ────────────────────────────────────────────────────────────────
# style & helpers
# ────────────────────────────────────────────────────────────────
# plt.style.use("seaborn-v0_8-whitegrid")
RES = 200


def _add_bg(ax, zoom=6):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    try:
        ax.add_image(TILER, zoom, interpolation="nearest")
    except Exception:
        ax.add_feature(cfeature.LAND, facecolor="lightgrey");
        ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.coastlines(resolution="10m", lw=0.6)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTDIR/name, dpi=RES)
    plt.close(fig)

flat = lambda arr: arr.where(mask).values.flatten()

# ────────────────────────────────────────────────────────────────
# 0) GLOBAL METRICS
# ────────────────────────────────────────────────────────────────

def _nse(o, p):
    return 1.0 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)

def _kge(o, p):
    r = pearsonr(o, p)[0]
    alpha = p.std(ddof=0) / o.std(ddof=0)
    beta  = p.mean() / o.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

metrics = {k: [] for k in ["space","RMSE","MAE","Bias","Pearson_r","Spearman_rho","NSE","KGE"]}
for tag, of, pf in [("linear", obs_mg, pred_mg), ("log", obs_log, pred_log)]:
    o, p = flat(of), flat(pf)
    sel  = np.isfinite(o) & np.isfinite(p)
    o, p = o[sel], p[sel]
    metrics["space"].append(tag)
    metrics["RMSE"].append(float(np.sqrt(mean_squared_error(o, p))))
    metrics["MAE"].append(float(np.mean(np.abs(o - p))))
    metrics["Bias"].append(float(np.mean(p - o)))
    metrics["Pearson_r"].append(float(pearsonr(o, p)[0]))
    metrics["Spearman_rho"].append(float(spearmanr(o, p)[0]))
    metrics["NSE"].append(float(_nse(o, p)))
    metrics["KGE"].append(float(_kge(o, p)))

df_global = pd.DataFrame(metrics)
df_global.to_csv(OUTDIR/"metrics_global_plus.csv", index=False)
print("\nGLOBAL METRICS (model vs obs):")
print(df_global.to_string(index=False))

# ────────────────────────────────────────────────────────────────
# 1) Domain‑mean time series
# ────────────────────────────────────────────────────────────────
mu_obs = obs_mg.where(mask).mean(["lat","lon"])
mu_mod = pred_mg.where(mask).mean(["lat","lon"])
mu_pers= pers_mg.where(mask).mean(["lat","lon"])

'''
# drop this into a notebook
mu_obs  = np.exp(obs_log).where(mask).mean().item()
mu_pred = np.exp(pred_log).where(mask).mean().item()
mu_pers = np.exp(pers_log).where(mask).mean().item()

print(mu_obs, mu_pers, mu_pred,
      mu_pred - mu_pers, mu_pred - mu_obs)
'''

fig, ax = plt.subplots(figsize=(10,3))
ax.plot(time, mu_obs,  label="Observed",     lw=1.3)
ax.plot(time, mu_mod,  label="Model",        lw=1.1)
ax.plot(time, mu_pers, label="Persistence", lw=1.0, alpha=.7)
ax.set_ylabel("Chl (mg m$^{-3}$)"); ax.set_title("Domain‑mean chlorophyll")
ax.legend(ncol=3)
_save(fig, "timeseries_domain_mean.png")

# ────────────────────────────────────────────────────────────────
# 2) Monthly RMSE & full metrics CSV (linear space)
# ────────────────────────────────────────────────────────────────
rows = []
for mth in range(1,13):
    idx = time.month == mth
    if idx.sum()==0: continue
    o, p = flat(obs_mg[idx]), flat(pred_mg[idx])
    sel  = np.isfinite(o) & np.isfinite(p)
    if not sel.any(): continue
    row = {
        "month": mth,
        "rmse": float(np.sqrt(mean_squared_error(o[sel], p[sel]))),
        "bias": float(np.mean(p[sel] - o[sel]))
    }
    rows.append(row)

df_month = pd.DataFrame(rows).set_index("month")
df_month.to_csv(OUTDIR/"metrics_monthly_linear.csv")

rmse_p = []
for mth in df_month.index:
    idx = time.month == mth
    o, q = flat(obs_mg[idx]), flat(pers_mg[idx])
    sel  = np.isfinite(o) & np.isfinite(q)
    rmse_p.append(float(np.sqrt(mean_squared_error(o[sel], q[sel]))))
df_month['rmse_pers'] = rmse_p        # save to CSV automatically

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(df_month.index, df_month.rmse,      "-o", label="Model")
ax.plot(df_month.index, df_month.rmse_pers, "-s", label="Persistence")

ax.set_xlabel("Month"); ax.set_ylabel("RMSE (mg m$^{-3}$)")
ax.set_title("Monthly RMSE – linear space"); ax.set_xticks(range(1,13))
ax.legend(); _save(fig, "rmse_monthly.png")

# --- Monthly RMSE (log space) for Model and Persistence ---
# Build monthly RMSE (log space) for Model and Persistence — SINGLE dataframe
def _rmse_np(a, b):
    a = np.asarray(a); b = np.asarray(b)
    sel = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[sel]-b[sel])**2))) if sel.any() else np.nan

rows_log = []
for mth in range(1,13):
    idx = (time.month == mth)
    if idx.sum() == 0: 
        continue
    o = obs_log[idx].where(mask[idx]).values.flatten()
    p = pred_log[idx].where(mask[idx]).values.flatten()
    r = pers_log[idx].where(mask[idx]).values.flatten()
    rows_log.append({
        "month": mth,
        "rmse_log_model": _rmse_np(o, p),
        "rmse_log_pers":  _rmse_np(o, r)
    })

df_month_log = (pd.DataFrame(rows_log)
                  .set_index("month")
                  .sort_index())
df_month_log.to_csv(OUTDIR/"metrics_monthly_log.csv")



# ────────────────────────────────────────────────────────────────
# 3) Taylor diagram (monthly anomalies, log space)
# ────────────────────────────────────────────────────────────────
class TaylorDiagram:  # minimal helper
    def __init__(self, ref_std, fig=None):
        self.ref_std = ref_std
        fig = fig or plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_direction(-1); ax.set_theta_zero_location('N')
        ax.set_ylim(0, ref_std*1.5)
        ax.set_xticks(np.deg2rad([0,30,60,90,120,150,180]))
        ax.grid(True); self.ax = ax
        self.ax.plot([0],[ref_std],'ko', label='Reference')
    def add(self, std, corr, label, **kw):
        theta = np.arccos(corr); self.ax.plot(theta, std, 'o', label=label, **kw)

def _monthly_anoms(a):
    clim = a.groupby('time.month').mean('time'); return a.groupby('time.month') - clim

a_obs, a_mod, a_pers = map(_monthly_anoms, (obs_log, pred_log, pers_log))
ref_sd = float(a_obs.where(mask).std())
std_mod, std_pers = [float(x.where(mask).std()) for x in (a_mod, a_pers)]
get_corr = lambda x: np.corrcoef(flat(a_obs), flat(x))[0,1]
TD = TaylorDiagram(ref_sd)
TD.add(std_pers, get_corr(a_pers), 'Persistence', color='C1')
TD.add(std_mod,  get_corr(a_mod),  'Model',       color='C0')
TD.ax.set_title('Taylor diagram – monthly anomalies'); TD.ax.legend(loc='upper right')
_save(TD.ax.figure, "taylor_diagram.png")

# ────────────────────────────────────────────────────────────────
# 4) Spatial skill maps
# ────────────────────────────────────────────────────────────────

def _rmse(a,b):
    return np.sqrt(((a - b) ** 2).mean('time'))

rmse_model = _rmse(pred_log.where(mask), obs_log.where(mask))
rmse_pers  = _rmse(pers_log.where(mask),  obs_log.where(mask))
skill_pct  = 100.0 * (1.0 - rmse_model / rmse_pers)

fig = plt.figure(figsize=(10,4))
for i, (fld, title, cmap, vmin, vmax) in enumerate([
        (rmse_model, 'RMSE model (log)', 'viridis', None, None),
        (rmse_pers,  'RMSE persistence (log)', 'viridis', None, None),
        (skill_pct,  'Skill % (improvement)', 'coolwarm', -100, 100)]):
    ax = fig.add_subplot(1,3,i+1, projection=ccrs.PlateCarree())
    _add_bg(ax, zoom=6)
    data = np.ma.masked_where(~coast_mask2d, fld)
    norm = TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax) if 'Skill' in title else None
    im = ax.pcolormesh(ds_pred.lon, ds_pred.lat, data, cmap=cmap, norm=norm, shading='nearest')
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.tick_params(labelsize=10)  # match xtick/ytick.labelsize
fig.suptitle('Spatial skill diagnostics', y=1.04)
_save(fig, "spatial_skill_maps.png")

# ────────────────────────────────────────────────────────────────
# 5) Spatial mean maps – Obs / Pred / Bias
# ────────────────────────────────────────────────────────────────
obs_mu, pred_mu = [x.where(mask).mean('time') for x in (obs_log, pred_log)]
bias_mu = pred_mu - obs_mu  # Pred – Obs

vmin, vmax = np.nanpercentile(np.concatenate([obs_mu.values[coast_mask2d], pred_mu.values[coast_mask2d]]), [2,98])
max_abs_bias = float(np.nanmax(np.abs(bias_mu.values[coast_mask2d])))

fig = plt.figure(figsize=(10,3.5))
for i, (fld, title, cmap, norm) in enumerate([
        (obs_mu,  'Mean observed chlorophyll', 'RdYlBu_r', plt.Normalize(vmin, vmax)),
        (pred_mu, 'Mean predicted chlorophyll','RdYlBu_r', plt.Normalize(vmin, vmax)),
        (bias_mu, 'Bias (Pred–Obs)',       'coolwarm', TwoSlopeNorm(vcenter=0, vmin=-max_abs_bias, vmax=max_abs_bias))]):
    ax = fig.add_subplot(1,3,i+1, projection=ccrs.PlateCarree())
    _add_bg(ax, zoom=6)
    data = np.ma.masked_where(~coast_mask2d, fld)
    im = ax.pcolormesh(ds_pred.lon, ds_pred.lat, data, cmap=cmap, norm=norm, shading='nearest')
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.tick_params(labelsize=10)  # match xtick/ytick.labelsize
fig.suptitle('Spatial mean & bias', y=1.04)
_save(fig, "spatial_mean_bias_maps.png")

# ────────────────────────────────────────────────────────────────
# 6) Hex‑bin scatter (log space)
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4,4))
x = flat(obs_log); y = flat(pred_log)
sel = np.isfinite(x) & np.isfinite(y)
x, y = x[sel], y[sel]
hb = ax.hexbin(x, y, gridsize=75, mincnt=1, bins='log', cmap='inferno')
lim = [np.nanmin([x.min(), y.min()]), np.nanmax([x.max(), y.max()])]
ax.plot(lim, lim, 'k--', lw=0.6)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Observed log‑chl'); ax.set_ylabel('Predicted log‑chl')
ax.set_title('Obs vs Pred – hex density')
cbar = plt.colorbar(hb, ax=ax)
cbar.set_label('log₁₀(count)')
cbar.ax.tick_params(labelsize=10)
_save(fig, "scatter_hex_log.png")

# --- Black & White scatter (log space), manuscript-friendly ---
def plot_scatter_hex_gray(outdir, name, pred_log, true_log, valid_m):
    m = np.asarray(valid_m, dtype=bool)
    x = np.asarray(true_log)[m]
    y = np.asarray(pred_log)[m]
    if x.size == 0:
        return

    fig, ax = plt.subplots(figsize=(3.2, 3.2))

    hb = ax.hexbin(x, y, gridsize=60, bins='log', mincnt=1,
                   cmap='Greys', linewidths=0.0)
    hb.set_rasterized(True)

    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("Obs log_chl"); ax.set_ylabel("Model log_chl")
    # ax.set_title(f"{name} scatter")

    # colorbar same height as the axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.06)
    cbar = fig.colorbar(hb, cax=cax)
    cbar.set_label("count")

    fig.tight_layout()
    fig.savefig(outdir / f"fig_scatter_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# build the B/W scatter over ALL times
# build a val mask over the predicted time axis
time_pd = pd.to_datetime(ds_pred.time.values)
va_sel = (time_pd >= pd.Timestamp("2016-01-01")) & (time_pd <= pd.Timestamp("2018-12-31"))

plot_scatter_hex_gray(
    OUTDIR, "val",
    pred_log.isel(time=va_sel),
    obs_log.isel(time=va_sel),
    mask.isel(time=va_sel)
)
plot_scatter_hex_gray(OUTDIR, "all", pred_log, obs_log, mask)




# --- RMSE by latitude (compare Model vs Persistence) ---
lat_bins = np.arange(30, 43.1, 1.0)  # 1° bands
rows_lat = []
for lo, hi in zip(lat_bins[:-1], lat_bins[1:]):
    band = (ds_pred.lat >= lo) & (ds_pred.lat < hi)
    if not band.any():
        continue
    o = obs_log.where(mask & band).values.flatten()
    p = pred_log.where(mask & band).values.flatten()
    r = pers_log.where(mask & band).values.flatten()
    sel_p = np.isfinite(o) & np.isfinite(p)
    sel_r = np.isfinite(o) & np.isfinite(r)
    rows_lat.append({
        "lat": f"{lo:.0f}-{hi:.0f}",
        "rmse_model": _rmse_np(o[sel_p], p[sel_p]),
        "rmse_pers":  _rmse_np(o[sel_r], r[sel_r])
    })

df_lat = pd.DataFrame(rows_lat)
df_lat.to_csv(OUTDIR/'rmse_by_lat.csv', index=False)

# replace the block that bars df_lat["rmse"]
fig, ax = plt.subplots(figsize=(4,3))
x = np.arange(len(df_lat))
w = 0.38
ax.bar(x - w/2, df_lat["rmse_model"], width=w, label="Model")
ax.bar(x + w/2, df_lat["rmse_pers"],  width=w, label="Persistence")
ax.set_xticks(x)
ax.set_xticklabels(df_lat["lat"], rotation=45, ha="right")
ax.set_ylabel('RMSE (log-chl)')
ax.set_xlabel('Latitude')
ax.set_title('RMSE by latitude')
ax.legend()
_save(fig, 'rmse_latitude.png')

# lead-time skill curve

# assume obs_log, pred_log_lead[k] are pre‐computed arrays for lead k
leads = [1,2,3,4]  # 8d,16d,24d,32d
rmse = []

for k in leads:
    np.save(f"pred_log_lead_{k}.npy", pred_log[:-k] if k>0 else pred_log)
    np.save(f"obs_log_lead_{k}.npy",  obs_log[k:]  if k>0 else obs_log)
    np.save(f"mask_lead_{k}.npy",     mask[k:]     if k>0 else mask)


for k in leads:
    # load your k‐step prediction arrays: pred_k, obs_k
    pred_k = np.load(f"pred_log_lead_{k}.npy")  # shape (T,H,W)
    obs_k  = np.load(f"obs_log_lead_{k}.npy")
    m      = np.load(f"mask_lead_{k}.npy")
    diff   = (pred_k - obs_k)[m]
    rmse.append(np.sqrt((diff*diff).mean()))

plt.figure(figsize=(5,3))
plt.plot([8*k for k in leads], rmse, marker='o')
plt.xlabel("Forecast lead (days)")
plt.ylabel("RMSE (log chl)")
plt.title("Lead-time skill curve")
# plt.tight_layout()
plt.savefig(OUTDIR / "fig_lead_time_skill.png")

# roc curve

threshold = 5.0
leads = [1,2,3]
plt.figure(figsize=(4,4))
for k in leads:
    pred_k = np.exp(np.load(f"pred_log_lead_{k}.npy")) - FLOOR
    obs_k  = np.exp(np.load(f"obs_log_lead_{k}.npy"))  - FLOOR
    m      = np.load(f"mask_lead_{k}.npy")
    y_true = (obs_k[m] >= threshold).astype(int)
    y_score = pred_k[m]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{8*k} d (AUC={roc_auc:.2f})")

plt.plot([0,1],[0,1],'k--',linewidth=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curves for bloom ≥ 5 mg m⁻³")
plt.legend(fontsize="x-small")
# plt.tight_layout()
plt.savefig(OUTDIR / "fig_bloom_roc.png")

# --- RMSE per time by observed log‑chl quartile (for Fig 3A) ---
o_all = obs_log.where(mask).values.flatten()
o_all = o_all[np.isfinite(o_all)]
q_edges = np.quantile(o_all, [0.25, 0.5, 0.75])

rmse_quart = {"<Q1": [], "Q1–Q2": [], "Q2–Q3": [], ">Q3": []}
for t_idx in range(obs_log.shape[0]):
    ov = obs_log.isel(time=t_idx).where(mask.isel(time=t_idx)).values
    pv = pred_log.isel(time=t_idx).where(mask.isel(time=t_idx)).values
    sel = np.isfinite(ov) & np.isfinite(pv)
    if not sel.any():
        for k in rmse_quart: rmse_quart[k].append(np.nan)
        continue
    ov, pv = ov[sel], pv[sel]
    d = np.digitize(ov, q_edges)  # 0..3
    groups = {"<Q1": (d==0), "Q1–Q2": (d==1), "Q2–Q3": (d==2), ">Q3": (d==3)}
    for k,gsel in groups.items():
        rmse_quart[k].append(_rmse_np(ov[gsel], pv[gsel]) if gsel.any() else np.nan)

pd.DataFrame(rmse_quart).to_csv(OUTDIR/"rmse_time_by_logquartile.csv", index=False)

def panel_label(ax, text, x=-0.10, y=1.05):
    ax.text(x, y, text, transform=ax.transAxes, weight="bold", fontsize=12)

def make_fig3(df_lat, rmse_quart, q_edges, df_month_log, outdir):
    import numpy as np
    fig = plt.figure(figsize=(9.5, 6.5), constrained_layout=True)
    gs  = fig.add_gridspec(2, 2)

    # A) Quartile RMSE (log-space) – distribution over time
    axA = fig.add_subplot(gs[0, 0])
    labels = ["<Q1", "Q1–Q2", "Q2–Q3", ">Q3"]
    data = [np.array(rmse_quart[l]) for l in labels]
    data = [d[np.isfinite(d)] for d in data]
    bp = axA.boxplot(data, widths=0.6, patch_artist=True,
                     medianprops={"linewidth":1.0, "color":"black"})
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("lightgray"); patch.set_alpha(0.7)
    axA.set_xticklabels([
        f"<Q1\n(<{q_edges[0]:.2f})",
        f"Q1–Q2\n({q_edges[0]:.2f}–{q_edges[1]:.2f})",
        f"Q2–Q3\n({q_edges[1]:.2f}–{q_edges[2]:.2f})",
        f">Q3\n(>{q_edges[2]:.2f})"
    ])
    axA.set_ylabel("RMSE (log‑chl)")
    axA.set_title("A  RMSE vs observed quartile", loc="left", weight="bold")

    # B) RMSE by latitude (log-space) – Model vs Persistence
    axB = fig.add_subplot(gs[0, 1])
    x = np.arange(len(df_lat))
    width = 0.38
    axB.bar(x - width/2, df_lat["rmse_model"], width, label="Model")
    axB.bar(x + width/2, df_lat["rmse_pers"],  width, label="Persistence")
    axB.set_xticks(x)
    axB.set_xticklabels(df_lat["lat"], rotation=45, ha="right")
    axB.set_ylabel("RMSE (log‑chl)")
    axB.set_title("B  RMSE by latitude", loc="left", weight="bold")
    axB.legend()

    # C) Monthly RMSE (log-space) – Model vs Persistence, on 2nd row
    axC = fig.add_subplot(gs[1, 0])
    axC.plot(df_month_log.index, df_month_log["rmse_log_model"], "-o", markersize=3, label="Model")
    axC.plot(df_month_log.index, df_month_log["rmse_log_pers"],  "-s", markersize=3, label="Persistence")
    axC.set_xticks(range(1,13))
    axC.set_xlabel("Month")
    axC.set_ylabel("RMSE (log‑chl)")
    axC.set_title("C  Monthly RMSE (log‑space)", loc="left", weight="bold")
    axC.legend()

    # D) leave a clean spacer (or future panel)
    axD = fig.add_subplot(gs[1, 1])
    axD.axis("off")

    fig.savefig(outdir / "fig3_error_strat_seasonality.png", dpi=300)
    fig.savefig(outdir / "fig3_error_strat_seasonality.pdf")
    plt.close(fig)


# case-study snapshot maps

lon   = ds_pred.lon.values
lat   = ds_pred.lat.values
times = ds_pred.time.values

# pick an index (e.g. bloom_peak_idx)
# find the time index of the largest bloom (max domain‐mean chlorophyll)
domain_mean = obs_mg.where(mask).mean(dim=("lat","lon"))
bloom_peak_idx = int(domain_mean.argmax(dim="time").item())
print("Case study at index", bloom_peak_idx, "which is date", ds_pred.time[bloom_peak_idx].values)

idx = bloom_peak_idx  
leads = [0,1,2]  # 0,8,16 d
fig = plt.figure(figsize=(9,3))
for i,k in enumerate(leads,1):
    ax = fig.add_subplot(1,3,i, projection=ccrs.PlateCarree())
    if k==0:
        field = np.exp(obs_log[idx])
        title = "Obs t₀"
    else:
        arr = np.load(f"pred_log_lead_{k}.npy")
        field = np.exp(arr[idx])
        title = f"Pred t₀+{8*k} d"
    im = ax.pcolormesh(lon, lat, field, shading='nearest')
    ax.coastlines()
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
fig.suptitle("Case study: bloom forecast vs. observation")
# plt.tight_layout()
plt.savefig(OUTDIR / "fig_case_study_maps.png")

# hovmoller diagram

# first back‐transform to linear
obs_lin   = np.exp(obs_log)
pred_lin  = np.exp(pred_log)

# compute anomalies relative to each frame’s spatial mean
obs_anom  = obs_lin - obs_lin.mean(dim=('lat','lon'))
pred_anom = pred_lin - pred_lin.mean(dim=('lat','lon'))

# apply mask (turn land/deep→NaN), then average over longitude
obs_hov   = obs_anom.where(mask).mean(dim='lon')
pred_hov  = pred_anom.where(mask).mean(dim='lon')

# now obs_hov & pred_hov are (time, lat).  Plot them against your time & lat coords:
t = pd.to_datetime(ds_pred.time.values)
lats = ds_pred.lat.values

fig, (ax0,ax1) = plt.subplots(1,2, figsize=(10,3), sharey=True)
pcm0 = ax0.pcolormesh(t, lats, obs_hov.T, shading='nearest')
ax0.set_title("Obs anomaly")
ax0.set_ylabel("Latitude")
pcm1 = ax1.pcolormesh(t, lats, pred_hov.T, shading='nearest')
ax1.set_title("Model anomaly")
for ax in (ax0,ax1):
    ax.set_xlabel("Time")
cbar = fig.colorbar(pcm0, ax=[ax0,ax1], orientation='horizontal',
                    pad=0.2, label="Chl anomaly (mg m⁻³)")
cbar.ax.tick_params(labelsize=10)
# plt.tight_layout()
plt.savefig(OUTDIR / "fig_hovmoller.png")
make_fig3(df_lat, rmse_quart, q_edges, df_month_log, OUTDIR)


print("\n✓ All diagnostics & figures written to", OUTDIR)
