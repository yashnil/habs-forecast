#!/usr/bin/env python3
"""
diagnostics.py — patch-based + full-grid diagnostics for convLSTM_best.pt

Outputs:
  - metrics_patch_val.csv
  - fig_scatter_val_patch.png
  - predicted_fields.nc
  - printed Global metrics table (train/val/test/all)

python pinn/diagnostics.py \
  --freeze /Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc \
  --ckpt   "/Users/yashnilmohanty/HAB_Models/convLSTM_best.pt" \
  --out    Diagnostics_patch \
  --seq    6 \
  --batch  32
"""

from __future__ import annotations
import argparse, json, pathlib, sys

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.cuda.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from testing import make_loaders, ConvLSTM, ALL_VARS, LOGCHL_IDX, MIXED_PREC
import testing
import torch.nn.functional as _F

def run_inference(ds, varlist, log_idx, stats, pixel_ok,
                  seq, lead, batch_k, device, mixed_prec, ckpt_path, verbose=True):
    # ——— set up —
    ntime = ds.sizes["time"]
    H, W  = ds.sizes["lat"], ds.sizes["lon"]
    # bounding box
    lat_any = np.where(pixel_ok.any("lon"))[0]
    lon_any = np.where(pixel_ok.any("lat"))[0]
    lat_slice = slice(lat_any.min(), lat_any.max()+1)
    lon_slice = slice(lon_any.min(), lon_any.max()+1)
    Hs = lat_any.max()-lat_any.min()+1
    Ws = lon_any.max()-lon_any.min()+1
    core = np.arange(seq-1, ntime-lead)
    Tpred = len(core)
    times = ds.time.values[core + lead]

    pred = np.full((Tpred,Hs,Ws), np.nan, np.float32)
    pers = pred.copy()
    true = pred.copy()
    m    = np.zeros_like(pred, bool)

    model = ConvLSTM(len(varlist)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.eval()

    for i in range(0, Tpred, batch_k):
        ks = core[i:i+batch_k]
        # build inputs
        # build inputs (handle static vs time‐varying)
        frames = []
        for k in ks:
            bands = []
            for v in varlist:
                da = ds[v]
                if "time" in da.dims:
                    arr = da.isel(time=k-seq+1, lat=lat_slice, lon=lon_slice).values
                else:
                    arr = da.isel(lat=lat_slice, lon=lon_slice).values
                µ, σ = stats[v]
                normed = (arr - µ) / σ
                bands.append(np.nan_to_num(normed, nan=0.0))
            frames.append(np.stack(bands, 0))  # shape (C, Hs, Ws)
        # stack into (B= len(ks), L=1, C, Hs, Ws)
        Xb = np.stack(frames, 0)[None,...] if len(frames)==1 else np.stack(frames,0)[:,None,:,:,:]
        Xb = Xb.astype(np.float32)

        Xt = torch.from_numpy(Xb).to(device)

        with torch.no_grad(), autocast(enabled=mixed_prec):
            Δ = model(Xt).cpu().numpy()

        last = ds.log_chl.isel(time=ks, lat=lat_slice, lon=lon_slice).values.astype(np.float32)
        # un-normalize Δ?  (your network predicts raw residuals on log space)
        pred[i:i+len(ks)] = last + Δ
        pers[i:i+len(ks)] = last
        true[i:i+len(ks)] = ds.log_chl.isel(time=ks+lead, lat=lat_slice, lon=lon_slice).values
        m[i:i+len(ks)]    = pixel_ok.isel(lat=lat_slice, lon=lon_slice).values & np.isfinite(true[i:i+len(ks)])
        if verbose:
            print(f"[diag] full-grid {i+len(ks)}/{Tpred}")

    return times, pred, pers, true, m

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", required=True, help="path to freeze .nc")
    p.add_argument("--ckpt",   required=True, help="path to convLSTM_best.pt")
    p.add_argument("--out",    required=True, help="output directory")
    p.add_argument("--seq",    type=int, default=6)
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--no_gpu", action="store_true")
    return p.parse_args()

def pad_full(sub, mask2d, H, W):
    """Pad a sub‐grid array (T, h, w) back into full (T,H,W)."""
    # find bounding box
    lat_any = mask2d.any("lon").values.nonzero()[0]
    lon_any = mask2d.any("lat").values.nonzero()[0]
    ymin, ymax = lat_any.min(), lat_any.max()+1
    xmin, xmax = lon_any.min(), lon_any.max()+1

    out = np.full((sub.shape[0], H, W), np.nan, dtype=sub.dtype)
    out[:, ymin:ymax, xmin:xmax] = sub
    return out

def main():
    args = get_args()
    FREEZE = pathlib.Path(args.freeze).expanduser().resolve()
    CKPT   = pathlib.Path(args.ckpt  ).expanduser().resolve()
    OUT    = pathlib.Path(args.out   ).expanduser().resolve()
    OUT.mkdir(exist_ok=True)
    if not FREEZE.is_file() or not CKPT.is_file():
        sys.exit("❌ freeze or ckpt not found")
    # global CKPT

    # ─────────────────────────────────────────────────────────────
    # 0) configure testing.py
    # ─────────────────────────────────────────────────────────────
    testing.FREEZE   = FREEZE
    testing.SEQ      = args.seq
    testing.LEAD_IDX = 1
    testing.BATCH    = args.batch

    # 1) load freeze to count channels & build mask2d
    ds = xr.open_dataset(FREEZE, chunks={"time":-1})
    ds.load()
    varlist = [v for v in ALL_VARS if v in ds.data_vars]
    Cin = len(varlist)
    pixel_ok = ds["pixel_ok"].astype(bool) if "pixel_ok" in ds else \
               ((~np.isnan(ds.log_chl)).sum("time")/ds.sizes["time"]>=0.2)
    full_H, full_W = ds.sizes["lat"], ds.sizes["lon"]

    # 2) load model
    device = torch.device("cpu") if args.no_gpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed = MIXED_PREC and device.type=="cuda"
    net = ConvLSTM(Cin).to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device), strict=True)
    net.eval()

    # 3) patch‐based val scatter + CSV
    print("[diag] patch‐based val scatter…")
    _, va_dl, _, _, _ = make_loaders()
    all_p, all_t = [], []
    for X,y,m in va_dl:
        X,y,m = X.to(device), y.to(device), m.to(device)
        with autocast(enabled=mixed):
            Δ = net(X)
            p = (X[:,-1,LOGCHL_IDX] + Δ)[m]
        all_p.append(p.detach().cpu().numpy())
        all_t.append(y[m].detach().cpu().numpy())
    preds = np.concatenate(all_p)
    trues = np.concatenate(all_t)
    # write CSV
    pd.DataFrame({"obs":trues,"pred":preds}) \
      .to_csv(OUT/"metrics_patch_val.csv", index=False)
    # scatter
    plt.hexbin(trues, preds, gridsize=60, bins="log", mincnt=1)
    mn,mx = preds.min(), preds.max()
    plt.plot([mn,mx],[mn,mx],"k--",lw=0.5)
    plt.xlabel("Obs log_chl"); plt.ylabel("Pred log_chl")
    plt.title("Patch‐based val scatter")
    plt.colorbar(label="count")
    plt.tight_layout()
    plt.savefig(OUT/"fig_scatter_val_patch.png")
    plt.close()

    # CKPT = str(pathlib.Path(args.ckpt).expanduser().resolve())

    # 4) full‐grid inference
    print("[diag] full‐grid inference…")
    # compute stats on train‐period:
    # compute stats on train‐period, handling static vs time‐varying vars
    tr_mask = ds.time < np.datetime64("2016-01-01")
    stats = {}
    for v in varlist:
        da = ds[v]
        if "time" in da.dims:
            sel = da.isel(time=tr_mask)
        else:
            sel = da
        μ = float(sel.mean(skipna=True).item())
        σ = float(sel.std(skipna=True).item()) or 1.0
        stats[v] = (μ, σ)
    times, pred_sub, pers_sub, true_sub, m_sub = run_inference(
        ds, varlist, LOGCHL_IDX, stats, pixel_ok,
        args.seq, 1, args.batch, device, mixed, str(CKPT))
    
    # pad back
    pred_full = pad_full(pred_sub, pixel_ok, full_H, full_W)
    pers_full = pad_full(pers_sub, pixel_ok, full_H, full_W)
    true_full = pad_full(true_sub, pixel_ok, full_H, full_W)
    m_full    = pad_full(m_sub.astype("uint8"), pixel_ok, full_H, full_W)

    # write NetCDF
    coords = {"time":times, "lat":ds.lat, "lon":ds.lon}
    ds_out = xr.Dataset({
      "log_chl_pred": (("time","lat","lon"), pred_full),
      "log_chl_pers": (("time","lat","lon"), pers_full),
      "log_chl_true": (("time","lat","lon"), true_full),
      "valid_mask":   (("time","lat","lon"), m_full.astype("uint8")),
    }, coords=coords)
    comp = {v:{"zlib":True,"complevel":4} for v in ds_out.data_vars}
    ncpath = OUT/"predicted_fields.nc"
    ds_out.to_netcdf(ncpath, encoding=comp)
    print(f"[diag] wrote full‐grid to {ncpath}")
    # 5) global metrics table
    print("\nGlobal metrics:")
    subsets = {
      "train": times <  np.datetime64("2016-01-01"),
      "val":   (times>=np.datetime64("2016-01-01")) & (times<=np.datetime64("2018-12-31")),
      "test":  times >  np.datetime64("2018-12-31"),
      "all":   np.ones_like(times, bool)
    }

    rows = []
    for name, mask_t in subsets.items():
        # first select the timeslice
        P_slice = pred_full[mask_t]      # shape (Nt, H, W)
        R_slice = pers_full[mask_t]
        Y_slice = true_full[mask_t]
        M_slice = m_full[mask_t] == 1     # boolean mask same shape

        # flatten only the valid pixels
        P_flat = P_slice[M_slice]
        R_flat = R_slice[M_slice]
        Y_flat = Y_slice[M_slice]

        # log‐space
        rm_m = np.sqrt(np.mean((P_flat - Y_flat)**2))
        rm_p = np.sqrt(np.mean((R_flat - Y_flat)**2))
        mae_m = np.mean(np.abs(P_flat - Y_flat))
        mae_p = np.mean(np.abs(R_flat - Y_flat))

        # linear‐space
        mgP = np.exp(P_flat); mgR = np.exp(R_flat); mgY = np.exp(Y_flat)
        rmM = np.sqrt(np.mean((mgP - mgY)**2))
        rmP = np.sqrt(np.mean((mgR - mgY)**2))
        maeM = np.mean(np.abs(mgP - mgY))
        maeP = np.mean(np.abs(mgR - mgY))

        skill_log = 100.0 * (1.0 - rm_m / rm_p) if rm_p>0 else np.nan
        skill_mg  = 100.0 * (1.0 - rmM / rmP)  if rmP>0 else np.nan

        rows.append({
            "subset": name,
            "rmse_log_model":  rm_m,  "rmse_log_pers": rm_p,
            "mae_log_model":   mae_m, "mae_log_pers": mae_p,
            "rmse_mg_model":   rmM,   "rmse_mg_pers":  rmP,
            "mae_mg_model":    maeM,  "mae_mg_pers":   maeP,
            "skill_log_pct":   skill_log, "skill_mg_pct": skill_mg
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT/"metrics_global.csv", index=False)
