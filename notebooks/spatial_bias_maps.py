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

python notebooks/spatial_bias_maps.py
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
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt

# ────────────────────────────────────────────────────────────────
# paths – edit if needed
# ────────────────────────────────────────────────────────────────
FREEZE = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_freeze_v1.nc")
PRED   = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_v0p4/predicted_fields.nc")
OUTDIR = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_v0p4")
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
plt.style.use("seaborn-v0_8-whitegrid")
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

fig, ax = plt.subplots(figsize=(10,3))
ax.plot(time, mu_obs,  label="Observed",     lw=1.3)
ax.plot(time, mu_mod,  label="Model",        lw=1.1)
ax.plot(time, mu_pers, label="Persistence", lw=1.0, alpha=.7)
ax.set_ylabel("Chl (mg m$^{-3}$)"); ax.set_title("Domain‑mean chlorophyll")
ax.legend(ncol=3, fontsize="small")
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
TD.ax.set_title('Taylor diagram – monthly anomalies'); TD.ax.legend(loc='upper right', fontsize='x-small')
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
    ax.set_title(title, fontsize='small'); plt.colorbar(im, ax=ax, shrink=0.75)
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
    ax.set_title(title, fontsize='small'); plt.colorbar(im, ax=ax, shrink=0.75)
fig.suptitle('Spatial mean & bias', y=1.04)
_save(fig, "spatial_mean_bias_maps.png")

# ────────────────────────────────────────────────────────────────
# 6) Hex‑bin scatter (log space)
# ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4,4))
hb = ax.hexbin(flat(obs_log), flat(pred_log), gridsize=75, mincnt=1, bins='log', cmap='inferno')
lim = [np.nanmin(hb.get_offsets()), np.nanmax(hb.get_offsets())]
ax.plot(lim, lim, 'k--', lw=0.6)
ax.set_xlabel('Observed log‑chl'); ax.set_ylabel('Predicted log‑chl')
ax.set_title('Obs vs Pred – hex density')
plt.colorbar(hb, ax=ax, label='log₁₀(count)'), _save(fig, "scatter_hex_log.png")

lat_bins = np.arange(30, 43.1, 1.0)         # 1° bands
rmse_lat = []
for lo, hi in zip(lat_bins[:-1], lat_bins[1:]):
    band = (ds_pred.lat >= lo) & (ds_pred.lat < hi)
    if not band.any(): continue
    o = obs_log.where(mask & band).values.flatten()
    p = pred_log.where(mask & band).values.flatten()
    sel = np.isfinite(o) & np.isfinite(p)
    rmse_lat.append(dict(lat=f'{lo:.0f}-{hi:.0f}',
                         rmse=np.sqrt(mean_squared_error(o[sel], p[sel]))))

df_lat = pd.DataFrame(rmse_lat)
df_lat.to_csv(OUTDIR/'rmse_by_lat.csv', index=False)

fig,ax = plt.subplots(figsize=(4,3))
ax.bar(df_lat.lat, df_lat.rmse)
ax.set_ylabel('RMSE (log-chl)'); ax.set_xlabel('Latitude band')
ax.set_title('RMSE by latitude'); ax.tick_params(axis='x', rotation=45)
_save(fig, 'rmse_latitude.png')


print("\n✓ All diagnostics & figures written to", OUTDIR)


'''
old results:

RMSE  (mg m-3) = 6.136
Bias  (mg m-3) = -1.476
ρ (Spearman)  = 0.786
KGE           = 0.539

Tier-wise metrics (linear space):
      bin      n      rmse      bias      rho        kge
      low 344306  2.203102 -0.847233 0.632948 -11.857232
      mid 344305  3.532932 -1.732169 0.245168 -12.505745
     high 344305  5.076493 -2.530042 0.283291  -5.173887
very_high 344306 10.367783 -0.794286 0.378148   0.373854

new results:

GLOBAL METRICS (model vs obs):
Bloom contingency (≥5 mg m-3): {'metric': 'bloom_≥5mg', 'hits': 224212, 'miss': 44446, 'false_alarm': 246032, 'POD': 0.8345629015328007, 'FAR': 0.52320072132765, 'CSI': 0.43562532786725905}

GLOBAL METRICS (model vs obs):
 space     RMSE     MAE     Bias  Pearson_r  Spearman_rho       NSE       KGE
linear 7.230919 3.59826 2.175986   0.565585      0.778074 -0.355453  0.198529
   log 0.994509 0.79026 0.608961   0.776783      0.778074  0.312010 -0.184123
'''