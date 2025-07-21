#!/usr/bin/env python3
# spatial_bias_maps.py  –  predicted vs observed chl maps + bias hist
# ---------------------------------------------------------
import pathlib, numpy as np, xarray as xr, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import pandas as pd

# ------------------------------------------------------------------ #
# paths – adjust if yours differ
FREEZE = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_freeze_v1.nc")
PRED   = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_v0p3/predicted_fields.nc")
OUTDIR = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_v0p3")
OUTDIR.mkdir(exist_ok=True, parents=True)
# helper: cheap, rate-limited Google tiles
TILER = cimgt.GoogleTiles(style='satellite')
TILER.request_timeout = 5
# bounding box for CA coast
EXTENT = [-125, -114, 31, 43]

# ------------------------------------------------------------------ #
# load & prep
ds_pred = xr.open_dataset(PRED)
obs_log  = ds_pred["log_chl_true"]      # (time,lat,lon)
pred_log = ds_pred["log_chl_pred"]
mask     = ds_pred["valid_mask"].astype(bool)

# linear-space arrays (mg m-3) – floor already removed in freeze
obs_mg  = np.exp(obs_log)
pred_mg = np.exp(pred_log)

# ---------------- global metrics (linear space) -------------------- #
obs_f   = obs_mg.values.flatten()
pred_f  = pred_mg.values.flatten()
mask_f  = mask.values.flatten() & np.isfinite(obs_f) & np.isfinite(pred_f)

rmse_lin = np.sqrt(mean_squared_error(obs_f[mask_f], pred_f[mask_f]))
bias_lin = (obs_f[mask_f] - pred_f[mask_f]).mean()

# Spearman rank correlation ρ (linear space)
rho_lin, _ = spearmanr(obs_f[mask_f], pred_f[mask_f])

# Kling-Gupta Efficiency (linear space)
μ_o, μ_p               = obs_f[mask_f].mean(), pred_f[mask_f].mean()
σ_o, σ_p               = obs_f[mask_f].std(ddof=0), pred_f[mask_f].std(ddof=0)
r_lin, _               = spearmanr(obs_f[mask_f], pred_f[mask_f])  # use Spearman ρ
alpha, beta            = σ_p / σ_o, μ_p / μ_o
kge_lin                = 1.0 - np.sqrt((r_lin-1)**2 + (alpha-1)**2 + (beta-1)**2)

print(f"RMSE  (mg m-3) = {rmse_lin:.3f}")
print(f"Bias  (mg m-3) = {bias_lin:.3f}")
print(f"ρ (Spearman)  = {rho_lin:.3f}")
print(f"KGE           = {kge_lin:.3f}")

# ---------------- domain-mean maps (log space, but colour swapped) -
obs_mu  = obs_log.where(mask).mean("time", skipna=True)
pred_mu = pred_log.where(mask).mean("time", skipna=True)
bias_mu = obs_mu - pred_mu          # +ve  →  model under-predicts

# ------------------------------------------------------------------ #
# simple Cartopy helper (sat-img background optional)
def _add_bg(ax, zoom=6):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    try:
        ax.add_image(TILER, zoom, interpolation="nearest")   # satellite
    except Exception:
        ax.add_feature(cfeature.LAND, facecolor="lightgrey")
        ax.add_feature(cfeature.OCEAN, facecolor="white")
    ax.coastlines(resolution="10m", lw=0.6, color="k")


def _density_scatter(x, y, fname, units, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(4,4))
    hb = ax.hexbin(x, y, gridsize=70, mincnt=1,
                   bins="log", cmap=cmap)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, 'k--', lw=0.7)
    ax.set_xlabel(f"Observed ({units})")
    ax.set_ylabel(f"Predicted ({units})")
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("log₁₀ (count)")
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=200)
    plt.close()


def _pcolormesh(field, fname, cmap, norm=None, label=""):
    fig = plt.figure(figsize=(6,5))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    _add_bg(ax)
    mesh = ax.pcolormesh(field.lon, field.lat, field,
                         cmap=cmap, norm=norm,
                         shading="nearest", transform=ccrs.PlateCarree())
    cb = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.8)
    cb.set_label(label)
    plt.title(fname.replace("_"," ").title())
    plt.tight_layout()
    plt.savefig(OUTDIR/f"{fname}.png", dpi=200)
    plt.close(fig)

# colour maps  (blue = high, red = low) -----------------------------
_pcolormesh(obs_mu,  "map_obs_mean",
            cmap="RdYlBu_r", label="Mean obs log-chl")

_pcolormesh(pred_mu, "map_pred_mean",
            cmap="RdYlBu_r", label="Mean pred log-chl")

norm_bias = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
_pcolormesh(bias_mu, "map_bias_mean",
            cmap="coolwarm_r", norm=norm_bias,
            label="Obs – Pred (log-chl)")

# 1) density-scatter plots
_density_scatter(obs_log.values[mask],  pred_log.values[mask],
                 "scatter_log.png",   "log-chl")
_density_scatter(obs_mg.values[mask],   pred_mg.values[mask],
                 "scatter_linear.png", "mg m$^{-3}$")

# 2) tier-wise metrics  (quartiles in *linear* space)
bins_lin = np.nanquantile(obs_mg.values[mask], [0.25, 0.50, 0.75])
labels   = ["low", "mid", "high", "very_high"]
tier     = np.digitize(obs_mg.values, bins_lin)   # 0-3
rows = []
for b, lbl in enumerate(labels):
    m = (tier == b) & mask.values
    if not m.any(): continue
    o, p = obs_mg.values[m], pred_mg.values[m]
    rmse  = np.sqrt(mean_squared_error(o, p))
    bias  = (o - p).mean()
    rho,_ = spearmanr(o, p)
    # KGE
    alpha = p.std(ddof=0) / o.std(ddof=0)
    beta  = p.mean()      / o.mean()
    r,_   = spearmanr(o, p)
    kge   = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    rows.append(dict(bin=lbl, n=m.sum(), rmse=rmse,
                     bias=bias, rho=rho, kge=kge))
df = pd.DataFrame(rows)
df.to_csv(OUTDIR / "tier_metrics.csv", index=False)
print("\nTier-wise metrics (linear space):")
print(df.to_string(index=False))

# ------------------------------------------------------------------ #
# histogram of bias  (linear space, mg m-3)
bias_all = (obs_mg - pred_mg).where(mask)
vals = bias_all.values.flatten()
vals = vals[np.isfinite(vals)]

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(vals, bins=60, alpha=0.8, color="tab:blue")
txt = (f"ρ={rho_lin:.2f}   KGE={kge_lin:.2f}   "
       f"RMSE={rmse_lin:.2f}   Bias={bias_lin:.2f}")
ax.set_title(txt)
ax.set_xlabel("Bias (obs – pred)  [mg m$^{-3}$]")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(OUTDIR/"hist_bias_linear.png", dpi=200)
plt.close()

print("✓ maps + histogram written to", OUTDIR)

'''
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

'''