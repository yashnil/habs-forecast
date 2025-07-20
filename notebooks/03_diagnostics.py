#!/usr/bin/env python3
"""
03_diagnostics.py  —  ConvLSTM v0.3 diagnostic suite
====================================================
Generates publication / science-fair quality diagnostics for the ConvLSTM
baseline trained in 02_baseline_model.py.

Key guarantees
--------------
* Uses the **same predictor list** that the model was trained with
  (the 30-variable ALL_VARS list in 02_baseline_model.py).
* Loads the checkpoint with strict shape checking (`strict=True`) so a
  mismatch is caught immediately.
* Same SEQ = 4 (32-day history) and LEAD = 1 (8-day forecast) defaults.
* Produces the usual CSV/NetCDF/PNG output bundle.

Run example
-----------
python 03_diagnostics.py \
  --freeze "/path/to/HAB_freeze_v1.nc" \
  --ckpt   "Models/convLSTM_best.pt"   \
  --out    "Diagnostics_v0p3"          \
  --batch  64            # 64–128 on GPU; 8–16 on CPU
"""

from __future__ import annotations
import argparse, json, math, pathlib, sys, warnings
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Predictor lists (identical to 02_baseline_model.py)
# ------------------------------------------------------------------ #
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos",
             "ssh_grad_mag","so","thetao"]
DERIVED   = ["chl_anom_monthly",
             "chl_roll24d_mean","chl_roll24d_std",
             "chl_roll40d_mean","chl_roll40d_std"]
STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]
ALL_VARS  = SATELLITE + METEO + OCEAN + DERIVED + STATIC
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def get_args():
    p = argparse.ArgumentParser(description="ConvLSTM v0.3 diagnostics")
    p.add_argument("--freeze", required=True, help="Path to HAB_freeze_v1.nc")
    p.add_argument("--ckpt",   required=True, help="Path to convLSTM_best.pt")
    p.add_argument("--out",    default="Diagnostics", help="Output directory")
    p.add_argument("--seq",    type=int, default=4, help="History length (default 4)")
    p.add_argument("--lead",   type=int, default=1, help="Lead steps (default 1)")
    p.add_argument("--batch",  type=int, default=1,
                   help="Number of core time steps per forward pass")
    p.add_argument("--no_mixed_prec", action="store_true", help="Disable AMP")
    p.add_argument("--no_gpu", action="store_true", help="Force CPU")
    return p.parse_args()

# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #
def discover_vars(ds):
    """Return (varlist, log_chl index) for variables present in BOTH freeze & ALL_VARS."""
    varlist = [v for v in ALL_VARS if v in ds.data_vars]
    if "log_chl" not in varlist:
        raise RuntimeError("Dataset missing 'log_chl' — cannot run diagnostics.")
    return varlist, varlist.index("log_chl")

def build_pixel_ok_if_needed(ds, thresh=0.20):
    if "pixel_ok" in ds:
        return ds.pixel_ok.astype(bool)
    frac = np.isfinite(ds.log_chl).sum("time") / ds.sizes["time"]
    return (frac >= thresh).astype("uint8").astype(bool)

def time_splits(times):
    t = np.asarray(times)
    train = t <  np.datetime64("2016-01-01")
    val   = (t >= np.datetime64("2016-01-01")) & (t <= np.datetime64("2018-12-31"))
    test  = t >  np.datetime64("2018-12-31")
    return train, val, test

def norm_stats(ds, train_mask, varlist):
    stats = {}
    idx = np.where(train_mask)[0]
    for v in varlist:
        da = ds[v].isel(time=idx) if "time" in ds[v].dims else ds[v]
        mu = float(da.mean(skipna=True))
        sd = float(da.std(skipna=True)) or 1.0
        stats[v] = (mu, sd)
    return stats

def z_np(arr, mu_sd):
    mu, sd = mu_sd
    return (arr - mu) / sd

# --------------------------------------------------
# Model definition (must match training)
# --------------------------------------------------
class PxLSTM(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.cell = nn.LSTMCell(ci, co)
        self.conv = nn.Conv2d(co, co, 1)

    def forward(self, x, hc=None):
        # x: (B,C,H,W)
        B,C,H,W = x.shape
        flat = x.permute(0,2,3,1).reshape(B*H*W, C)

        if hc is None:
            h = torch.zeros(flat.size(0), self.cell.hidden_size,
                            dtype=flat.dtype, device=flat.device)
            c = torch.zeros_like(h)
            hc = (h,c)

        h,c = self.cell(flat, hc)
        h_map = h.view(B,H,W,-1).permute(0,3,1,2)
        return self.conv(h_map), (h,c)

class ConvLSTM(nn.Module):
    """
    Channel reducer + 2-layer pixel-wise LSTM conv head, identical to v0.3 training.
    """
    def __init__(self, Cin):
        super().__init__()
        self.reduce = nn.Conv2d(Cin, 24, 1)
        self.l1     = PxLSTM(24, 48)
        self.l2     = PxLSTM(48, 64)
        self.head   = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # x: (B,L,C,H,W)
        h1=h2=None
        for t in range(x.size(1)):
            f = self.reduce(x[:,t])
            o1,h1 = self.l1(f,h1)
            o2,h2 = self.l2(o1,h2)
        return self.head(o2).squeeze(1)   # (B,H,W) ∆log_chl
    

# ------------------------------------------------------------------ #
# Build (seq,C,H,W) input block for time index k
# ------------------------------------------------------------------ #
def make_input(ds, varlist, stats, k, seq, lat_slice, lon_slice):
    frames = []
    for dt in range(seq):
        t = k - seq + 1 + dt
        bands = []
        for v in varlist:
            da = ds[v].isel(
                time=t, lat=lat_slice, lon=lon_slice) if "time" in ds[v].dims else \
                 ds[v].isel(lat=lat_slice, lon=lon_slice)
            bands.append(np.nan_to_num(z_np(da.values, stats[v]), nan=0.0))
        frames.append(np.stack(bands, 0))
    return np.stack(frames, 0).astype(np.float32)

# --------------------------------------------------
# Build input tensor for a given core time index k
# --------------------------------------------------
def build_input_block(ds, varlist, stats, k, seq, patch_slice_lat, patch_slice_lon):
    """
    Extract 1 sample covering full patch_slice from times [k-seq+1 .. k].
    Returns numpy array (seq, C, H, W).
    """
    frames = []
    for dt in range(seq):
        t = k - seq + 1 + dt
        bands=[]
        for v in varlist:
            darr = ds[v]
            if "time" in darr.dims:
                da = darr.isel(time=t, lat=patch_slice_lat, lon=patch_slice_lon)
            else:
                da = darr.isel(lat=patch_slice_lat, lon=patch_slice_lon)
            arr = da.values
            arr = np.nan_to_num(z_np(arr, stats[v]), nan=0.0)
            bands.append(arr)
        frames.append(np.stack(bands,0))
    return np.stack(frames,0).astype(np.float32)

# ------------------------------------------------------------------ #
# Full-grid inference loop
# ------------------------------------------------------------------ #
def run_inference(ds, varlist, log_idx, stats, pixel_ok,
                  seq, lead, batch_k, device, mixed_prec=True, verbose=True):
    ntime = ds.sizes["time"]
    H, W  = ds.sizes["lat"], ds.sizes["lon"]

    # bounding box of valid coast pixels to save work
    lat_any = np.where(pixel_ok.any("lon"))[0]
    lon_any = np.where(pixel_ok.any("lat"))[0]
    lat_slice = slice(lat_any.min(), lat_any.max()+1)
    lon_slice = slice(lon_any.min(), lon_any.max()+1)
    Hs = lat_any.max() - lat_any.min() + 1
    Ws = lon_any.max() - lon_any.min() + 1

    core_range = np.arange(seq-1, ntime-lead)
    Tpred = len(core_range)
    times_pred = ds.time.values[core_range + lead]

    pred_log = np.full((Tpred, Hs, Ws), np.nan, np.float32)
    pers_log = np.full_like(pred_log, np.nan)
    true_log = np.full_like(pred_log, np.nan)
    valid_m  = np.zeros_like(pred_log, bool)

    model = ConvLSTM(len(varlist)).to(device)
    model.load_state_dict(
        torch.load(args.ckpt, map_location=device), strict=True)
    model.eval()

    for i in range(0, Tpred, batch_k):
        ks = core_range[i:i+batch_k]
        X_np = np.stack([make_input(ds, varlist, stats, k, seq,
                                    lat_slice, lon_slice) for k in ks], 0)
        X_t = torch.from_numpy(X_np).to(device)

        with torch.no_grad(), autocast(enabled=mixed_prec):
            delta = model(X_t).cpu().numpy()   # (B, Hs, Ws)

        last_log = ds.log_chl.isel(
            time=ks, lat=lat_slice, lon=lon_slice).values.astype(np.float32)
        truth = ds.log_chl.isel(
            time=ks+lead, lat=lat_slice, lon=lon_slice).values.astype(np.float32)

        idx = slice(i, i+len(ks))
        pred_log[idx] = last_log + delta
        pers_log[idx] = last_log
        true_log[idx] = truth
        valid_m[idx]  = pixel_ok.isel(
            lat=lat_slice, lon=lon_slice).values & np.isfinite(truth)

        if verbose:
            print(f"[diag] processed {i+len(ks):>4}/{Tpred}", flush=True)

    # pad back to full grid for plotting convenience
    def pad(arr, fill=np.nan):
        out = np.full((Tpred, H, W),
                      fill if arr.dtype.kind == "f" else False,
                      arr.dtype)
        out[:, lat_any.min():lat_any.max()+1,
                lon_any.min():lon_any.max()+1] = arr
        return out

    return times_pred, pad(pred_log), pad(pers_log), pad(true_log), pad(valid_m, False)

# --------------------------------------------------
# Metric helpers
# --------------------------------------------------
def _masked_rmse(a,b,m):
    diff = (a-b)[m]
    if diff.size==0: return np.nan
    return np.sqrt(np.mean(diff*diff))

def _masked_mae(a,b,m):
    diff = np.abs(a-b)[m]
    if diff.size==0: return np.nan
    return np.mean(diff)

def _metrics_block(pred_log, pers_log, true_log, mask, floor_mg=None):
    """
    Return dict of metrics in log & mg space for arrays (T,H,W).
    mg conversion uses exp(log). If you want to subtract a floor_mg
    (as in freeze construction) supply it; otherwise raw exp().
    """
    m = mask
    p = pred_log; r = pers_log; y = true_log

    # log
    rmse_log_m   = _masked_rmse(p, y, m)
    rmse_log_p   = _masked_rmse(r, y, m)
    mae_log_m    = _masked_mae(p, y, m)
    mae_log_p    = _masked_mae(r, y, m)

    # mg space
    if floor_mg is not None:
        p_lin = np.exp(p) - floor_mg
        r_lin = np.exp(r) - floor_mg
        y_lin = np.exp(y) - floor_mg
        p_lin = np.clip(p_lin, 0, None)
        r_lin = np.clip(r_lin, 0, None)
        y_lin = np.clip(y_lin, 0, None)
    else:
        p_lin = np.exp(p); r_lin = np.exp(r); y_lin = np.exp(y)

    rmse_mg_m  = _masked_rmse(p_lin, y_lin, m)
    rmse_mg_p  = _masked_rmse(r_lin, y_lin, m)
    mae_mg_m   = _masked_mae(p_lin, y_lin, m)
    mae_mg_p   = _masked_mae(r_lin, y_lin, m)

    return {
        "rmse_log_model": rmse_log_m,
        "rmse_log_pers" : rmse_log_p,
        "mae_log_model" : mae_log_m,
        "mae_log_pers"  : mae_log_p,
        "rmse_mg_model" : rmse_mg_m,
        "rmse_mg_pers"  : rmse_mg_p,
        "mae_mg_model"  : mae_mg_m,
        "mae_mg_pers"   : mae_mg_p,
        "skill_log_pct" : 100.0 * (1.0 - (rmse_log_m / rmse_log_p)) if rmse_log_p>0 else np.nan,
        "skill_mg_pct"  : 100.0 * (1.0 - (rmse_mg_m  / rmse_mg_p))  if rmse_mg_p>0 else np.nan,
    }

# --------------------------------------------------
# Aggregations
# --------------------------------------------------
def subset_time_block(times_pred, mask, *arrays, mask_subset=None):
    """
    mask_subset is boolean mask over times_pred; returns arrays subset.
    """
    if mask_subset is None:
        return arrays + (mask,)
    idx = np.where(mask_subset)[0]
    arrs = tuple(a[idx] for a in arrays)
    m    = mask[idx]
    return arrs + (m,)

def monthly_skill(times_pred, pred_log, pers_log, true_log, valid_m):
    months = pd.DatetimeIndex(times_pred).month.values
    rows=[]
    for m in range(1,13):
        sel = months==m
        P,R,Y,M = pred_log[sel], pers_log[sel], true_log[sel], valid_m[sel]
        if P.size==0: continue
        rows.append({"month":m, **_metrics_block(P,R,Y,M)})
    return pd.DataFrame(rows).sort_values("month")

def quartile_bins_skill(times_pred, true_log, mask_all):
    """Use chl_lin (true) domain-median by time to define bins, then compute mask subset metrics later."""
    # median over spatial domain each time (valid_m to guard)
    chl_lin_time = np.exp(true_log)  # mg
    chl_lin_time = np.where(mask_all, chl_lin_time, np.nan)
    dom_med = np.nanmedian(chl_lin_time, axis=(1,2))
    q = np.nanquantile(dom_med, [0.25,0.5,0.75])
    bins = np.digitize(dom_med, q)  # 0-3
    return bins, q  # caller will compute metrics per bin

# --------------------------------------------------
# Figures
# --------------------------------------------------
def _common_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi":150,
        "savefig.dpi":150,
        "axes.titlesize":"medium",
        "axes.labelsize":"small",
        "legend.fontsize":"small",
        "xtick.labelsize":"x-small",
        "ytick.labelsize":"x-small",
    })

def plot_timeseries(outdir, times_pred, pred_log, pers_log, true_log, valid_m):
    _common_style()
    t = pd.to_datetime(times_pred)
    # domain-mean mg m-3
    y_lin = np.exp(true_log); p_lin = np.exp(pred_log); r_lin = np.exp(pers_log)
    m = valid_m
    y_mu = np.nanmean(np.where(m, y_lin, np.nan), axis=(1,2))
    p_mu = np.nanmean(np.where(m, p_lin, np.nan), axis=(1,2))
    r_mu = np.nanmean(np.where(m, r_lin, np.nan), axis=(1,2))
    plt.figure(figsize=(10,3))
    plt.plot(t, y_mu, label="Obs")
    plt.plot(t, p_mu, label="Model", alpha=.8)
    plt.plot(t, r_mu, label="Persistence", alpha=.8)
    plt.ylabel("Chl (mg m$^{-3}$)")
    plt.title("Domain-mean chlorophyll")
    plt.legend(ncol=3, fontsize="x-small")
    plt.tight_layout()
    plt.savefig(outdir/"fig_timeseries.png")
    plt.close()
    # also save CSV
    df = pd.DataFrame({"time":t, "chl_obs":y_mu, "chl_model":p_mu, "chl_pers":r_mu})
    df.to_csv(outdir/"metrics_timeseries.csv", index=False)

def plot_scatter(outdir, name, pred_log, true_log, valid_m):
    _common_style()
    # flatten
    m = valid_m
    x = true_log[m]; y = pred_log[m]
    if x.size==0:
        return
    plt.figure(figsize=(3,3))
    plt.hexbin(x, y, gridsize=60, bins='log', mincnt=1)
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    plt.plot(lim,lim,'k--',lw=0.5)
    plt.xlabel("Obs log_chl")
    plt.ylabel("Model log_chl")
    plt.title(f"{name} scatter")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(outdir/f"fig_scatter_{name}.png")
    plt.close()

def plot_month_skill(outdir, df_month):
    _common_style()
    plt.figure(figsize=(5,3))
    plt.plot(df_month.month, df_month.rmse_log_model, label="Model")
    plt.plot(df_month.month, df_month.rmse_log_pers,  label="Persistence")
    plt.xlabel("Month")
    plt.ylabel("RMSE log_chl")
    plt.title("Monthly RMSE (log)")
    plt.xticks(range(1,13))
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"fig_month_skill.png")
    plt.close()

def plot_bins_skill(outdir, df_bins):
    _common_style()
    width=0.35
    idx = np.arange(len(df_bins))
    plt.figure(figsize=(5,3))
    plt.bar(idx-width/2, df_bins.rmse_log_model, width, label="Model")
    plt.bar(idx+width/2, df_bins.rmse_log_pers,  width, label="Persistence")
    plt.xticks(idx, df_bins.bin_label, rotation=45, ha="right")
    plt.ylabel("RMSE log_chl")
    plt.title("Skill by bloom quartile (domain median)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"fig_bins_skill.png")
    plt.close()

def plot_residual_hist(outdir, name, pred_log, true_log, valid_m):
    _common_style()
    err = (pred_log - true_log)[valid_m]
    plt.figure(figsize=(4,3))
    plt.hist(err, bins=60, alpha=0.8, color="C0")
    plt.xlabel("Prediction error (log chl)")
    plt.ylabel("Count")
    plt.title(f"Residuals {name}")
    plt.tight_layout()
    plt.savefig(outdir/f"fig_residual_hist_{name}.png")
    plt.close()

def plot_skill_maps(outdir, rmse_model, rmse_pers, skill_pct, lat, lon):
    _common_style()
    fig,axs = plt.subplots(1,3,figsize=(10,3), constrained_layout=True)
    im0 = axs[0].pcolormesh(lon,lat, rmse_model, shading="nearest")
    axs[0].set_title("RMSE model (log)")
    fig.colorbar(im0, ax=axs[0], shrink=0.8)
    im1 = axs[1].pcolormesh(lon,lat, rmse_pers, shading="nearest")
    axs[1].set_title("RMSE persistence (log)")
    fig.colorbar(im1, ax=axs[1], shrink=0.8)
    im2 = axs[2].pcolormesh(lon,lat, skill_pct, shading="nearest", vmin=-100, vmax=100, cmap="coolwarm")
    axs[2].set_title("Skill % (1 - RMSEm/RMSEp)")
    fig.colorbar(im2, ax=axs[2], shrink=0.8)
    for ax in axs:
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.suptitle("Spatial skill maps", y=1.02)
    fig.savefig(outdir/"fig_skill_maps.png", bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    args = get_args()

    FREEZE = pathlib.Path(args.freeze).expanduser().resolve()
    CKPT   = pathlib.Path(args.ckpt  ).expanduser().resolve()
    OUTDIR = pathlib.Path(args.out   ).expanduser().resolve()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    OUT = OUTDIR

    if not FREEZE.is_file():
        sys.exit(f"[ERR] freeze file not found: {FREEZE}")
    if not CKPT.is_file():
        sys.exit(f"[ERR] checkpoint not found: {CKPT}")

    device = torch.device("cpu") if args.no_gpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_prec = (not args.no_mixed_prec) and device.type == "cuda"

    print("[diag] loading data …", flush=True)
    ds = xr.open_dataset(FREEZE, chunks={"time": -1})
    ds.load()  # pull into RAM once

    pixel_ok = build_pixel_ok_if_needed(ds)
    varlist, LOG_IDX = discover_vars(ds)
    Cin = len(varlist)              # MUST match checkpoint

    tr_mask, va_mask, te_mask = time_splits(ds.time.values)
    stats = norm_stats(ds, tr_mask, varlist)

    print("[diag] running inference …", flush=True)
    times_pred, pred_log, pers_log, true_log, valid_m = run_inference(
        ds, varlist, LOG_IDX, stats, pixel_ok,
        seq=args.seq, lead=args.lead, batch_k=args.batch,
        device=device, mixed_prec=mixed_prec)

    # Align train/val/test subsets to PRED times (k+lead)
    t_pred = times_pred
    tr_sel = t_pred <  np.datetime64("2016-01-01")
    va_sel = (t_pred >= np.datetime64("2016-01-01")) & (t_pred <= np.datetime64("2018-12-31"))
    te_sel = t_pred >  np.datetime64("2018-12-31")

    # Floor mg?  If you logged chl after adding detection floor, pass the floor you used
    # into _metrics_block. If unknown, just None (raw exp). For now:
    floor_mg = None

    # Global metrics
    blocks=[]
    for label,sel in [("train",tr_sel),("val",va_sel),("test",te_sel),("all",None)]:
        if sel is None:
            P,R,Y,M = pred_log, pers_log, true_log, valid_m
        else:
            P,R,Y,M = pred_log[sel], pers_log[sel], true_log[sel], valid_m[sel]
        row = {"subset":label, **_metrics_block(P,R,Y,M, floor_mg=floor_mg)}
        blocks.append(row)
    df_global = pd.DataFrame(blocks)
    df_global.to_csv(OUT/"metrics_global.csv", index=False)
    print("\nGlobal metrics:")
    print(df_global.to_string(index=False))

    # Monthly skill
    df_month = monthly_skill(t_pred, pred_log, pers_log, true_log, valid_m)
    df_month.to_csv(OUT/"metrics_month.csv", index=False)

    # Quartile-bin skill (by domain-median chl at each pred time)
    chl_bins, qvals = quartile_bins_skill(t_pred, true_log, valid_m)
    rows=[]
    labels=[]
    for b in range(4):
        sel = chl_bins==b
        P,R,Y,M = pred_log[sel], pers_log[sel], true_log[sel], valid_m[sel]
        row = {"bin":b, **_metrics_block(P,R,Y,M, floor_mg=floor_mg)}
        rows.append(row)
    # label strings
    qstr = [f"<Q1({qvals[0]:.2f})",
            f"Q1-Q2({qvals[0]:.2f}-{qvals[1]:.2f})",
            f"Q2-Q3({qvals[1]:.2f}-{qvals[2]:.2f})",
            f">=Q3({qvals[2]:.2f})"]
    df_bins = pd.DataFrame(rows)
    df_bins["bin_label"] = qstr
    df_bins.to_csv(OUT/"metrics_bins.csv", index=False)

    # Spatial skill maps (over ALL pred times)
    # SSE by pixel
    M = valid_m
    diff_m = (pred_log - true_log)
    diff_p = (pers_log - true_log)
    sse_m  = np.nansum((diff_m**2)*M, axis=0)
    sse_p  = np.nansum((diff_p**2)*M, axis=0)
    cnt    = M.sum(axis=0).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmse_m_grid = np.sqrt(sse_m / cnt)
        rmse_p_grid = np.sqrt(sse_p / cnt)
        skill_pct_grid = 100.0 * (1.0 - (rmse_m_grid / rmse_p_grid))
    np.savez(OUT/"skill_maps.npz",
             rmse_model=rmse_m_grid, rmse_pers=rmse_p_grid, skill_pct=skill_pct_grid)

    # Save predicted fields to NetCDF
    pred_da = xr.DataArray(pred_log, coords={"time":t_pred, "lat":ds.lat, "lon":ds.lon},
                           dims=("time","lat","lon"), name="log_chl_pred")
    pers_da = xr.DataArray(pers_log, coords=pred_da.coords, dims=pred_da.dims, name="log_chl_pers")
    true_da = xr.DataArray(true_log, coords=pred_da.coords, dims=pred_da.dims, name="log_chl_true")
    vm_da   = xr.DataArray(valid_m.astype("uint8"), coords=pred_da.coords, dims=pred_da.dims, name="valid_mask")
    ds_out  = xr.Dataset({"log_chl_pred":pred_da, "log_chl_pers":pers_da,
                          "log_chl_true":true_da, "valid_mask":vm_da})
    comp = {v:{"zlib":True,"complevel":4} for v in ds_out.data_vars}
    ds_out.to_netcdf(OUT/"predicted_fields.nc", encoding=comp)

    # --- FIGURES ---
    print("\nBuilding figures …", flush=True)
    plot_timeseries(OUT, t_pred, pred_log, pers_log, true_log, valid_m)
    plot_scatter   (OUT, "val",  pred_log[va_sel], true_log[va_sel], valid_m[va_sel])
    plot_month_skill(OUT, df_month)
    plot_bins_skill (OUT, df_bins)
    plot_residual_hist(OUT, "val", pred_log[va_sel], true_log[va_sel], valid_m[va_sel])
    plot_skill_maps(OUT, rmse_m_grid, rmse_p_grid, skill_pct_grid,
                    ds.lat.values, ds.lon.values)

    # Save summary JSON
    summary = {
        "freeze_path" : str(FREEZE),
        "ckpt_path"   : str(CKPT),
        "seq"         : args.seq,
        "lead"        : args.lead,
        "global"      : df_global.to_dict(orient="records"),
        "quartiles"   : df_bins.to_dict(orient="records"),
        "month"       : df_month.to_dict(orient="records"),
    }
    with open(OUT/"diagnostics_summary.json","w") as f:
        json.dump(summary, f, indent=2)

    print("\n✓ Diagnostics complete.")
    print(f"Outputs written to: {OUT}")

'''
To run the code:

python notebooks/03_diagnostics.py \
  --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_freeze_v1.nc" \
  --ckpt  "/Users/yashnilmohanty/Desktop/habs-forecast/Models/convLSTM_best.pt" \
  --out   "/Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_v0p3" \
  --batch 64

'''