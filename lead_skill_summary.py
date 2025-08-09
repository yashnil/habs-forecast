#!/usr/bin/env python3
"""
lead_skill_summary.py — 8/16/24/32 d lead metrics for ConvLSTM, PINN, TFT

Reads each model's predicted_fields.nc and computes per-lead metrics:
  - log space:   RMSE_log, MAE_log, Bias_log, r_log
  - linear space (mg m^-3): RMSE_mg, MAE_mg
  - skill vs persistence (%) in both spaces (1 - RMSE_model/RMSE_pers) * 100
  - N_valid

Persistence baseline for lead k uses obs at time t to predict obs at t+k:
  r_k(t+k) = y(t), i.e., r_log_shift = obs_log[:-k]

Usage:
  python lead_skill_summary.py --subset all
     where subset ∈ {all, train, val, test}

Paths assume your existing layout; edit below if needed.
"""

import argparse, pathlib, numpy as np, pandas as pd, xarray as xr

FLOOR = 0.056616  # mg m^-3
OUT_ROOT = pathlib.Path.cwd() / "Diagnostics_Combo_New"
OUT_ROOT.mkdir(exist_ok=True)

# Your corrected paths
BASE_MODELS = {
    "ConvLSTM": pathlib.Path("/Users/yashnilmohanty/Desktop/Diagnostics_ConvLSTM/predicted_fields.nc"),
    "PINN"    : pathlib.Path("/Users/yashnilmohanty/Desktop/Diagnostics_PINN/predicted_fields.nc"),
    "TFT"     : pathlib.Path("/Users/yashnilmohanty/Desktop/Diagnostics_TFT/predicted_fields.nc"),
}

LEAD_DAYS = [8, 16, 24, 32]  # human-friendly leads

def time_subset_mask(times):
    t = np.asarray(times)
    return {
        "train": t <  np.datetime64("2016-01-01"),
        "val"  : (t >= np.datetime64("2016-01-01")) & (t <= np.datetime64("2018-12-31")),
        "test" : t >  np.datetime64("2018-12-31"),
        "all"  : np.ones_like(t, bool),
    }

def _masked_rmse(a,b,m):
    d=(a-b)[m]; 
    return float(np.sqrt(np.mean(d*d))) if d.size else np.nan

def _masked_mae(a,b,m):
    d=np.abs(a-b)[m]
    return float(np.mean(d)) if d.size else np.nan

def _masked_bias(a,b,m):
    d=(a-b)[m]
    return float(np.mean(d)) if d.size else np.nan

def _masked_r(a,b,m):
    x=a[m]; y=b[m]
    if x.size<2: return np.nan
    x = x - np.nanmean(x); y = y - np.nanmean(y)
    denom = np.sqrt(np.nanmean(x*x)*np.nanmean(y*y))
    return float(np.nanmean(x*y)/denom) if denom and np.isfinite(denom) else np.nan

def compute_metrics_aligned(P, Y, R, M):
    """All arrays already aligned in time; shapes (T,H,W)."""
    rmse_log_m = _masked_rmse(P, Y, M)
    rmse_log_p = _masked_rmse(R, Y, M)
    mae_log_m  = _masked_mae (P, Y, M)
    bias_log_m = _masked_bias(P, Y, M)
    r_log_m    = _masked_r   (P, Y, M)

    # linear space
    P_lin = np.clip(np.exp(P) - FLOOR, 0, None)
    Y_lin = np.clip(np.exp(Y) - FLOOR, 0, None)
    R_lin = np.clip(np.exp(R) - FLOOR, 0, None)

    rmse_mg_m = _masked_rmse(P_lin, Y_lin, M)
    rmse_mg_p = _masked_rmse(R_lin, Y_lin, M)
    mae_mg_m  = _masked_mae (P_lin, Y_lin, M)

    skill_log = 100.0*(1.0 - rmse_log_m/rmse_log_p) if rmse_log_p and np.isfinite(rmse_log_p) else np.nan
    skill_mg  = 100.0*(1.0 - rmse_mg_m /rmse_mg_p ) if rmse_mg_p  and np.isfinite(rmse_mg_p ) else np.nan
    n_valid   = int(np.sum(M & np.isfinite(Y) & np.isfinite(P)))

    return dict(rmse_log=rmse_log_m, mae_log=mae_log_m, bias_log=bias_log_m, r_log=r_log_m,
                rmse_mg=rmse_mg_m, mae_mg=mae_mg_m,
                skill_log_pct=skill_log, skill_mg_pct=skill_mg,
                n_valid=n_valid)

def arrays_for_lead(ds, lead_days):
    """Return (P, Y, R, M, times_used) aligned for given lead.
       Uses per-lead file if present; else shifts obs by k=(lead/8 - 1)."""
    k = lead_days // 8 - 1  # 8->0, 16->1, 24->2, 32->3
    pred = ds["log_chl_pred"].values.astype(np.float32)  # (T,H,W)
    obs  = ds["log_chl_true"].values.astype(np.float32)
    mask = ds["valid_mask"].astype(bool).values

    # If this dataset already contains persistence aligned to its predictions, prefer it.
    pers = ds["log_chl_pers"].values.astype(np.float32) if "log_chl_pers" in ds else None

    if k == 0:
        P, Y, M = pred, obs, mask
        R = pers if pers is not None else obs
        times_used = ds["time"].values
    else:
        # shift obs/pers forward by k steps; predictions are from the +8 d file
        P = pred[:-k]
        Y = obs [ k:]
        M = mask[ k:]
        if pers is not None:
            R = pers[:-k]  # pers in file is last_log aligned to pred; keep same slicing
        else:
            R = obs[:-k]   # fallback
        times_used = ds["time"].values[k:]
    return P, Y, R, M, times_used

def try_open_perlead(model_name, lead_days):
    """If /Users/.../Diagnostics_{MODEL}_L{lead}/predicted_fields.nc exists, return that path, else None."""
    base = pathlib.Path(f"/Users/yashnilmohanty/Desktop/Diagnostics_{model_name}_L{lead_days}/predicted_fields.nc")
    return base if base.is_file() else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["all","train","val","test"], default="all")
    args = ap.parse_args()

    rows=[]
    for model, base_nc in BASE_MODELS.items():
        # pick per-lead file if available, else use base file for all leads
        for lead in LEAD_DAYS:
            nc_path = try_open_perlead(model, lead) or base_nc
            if not pathlib.Path(nc_path).is_file():
                print(f"[WARN] missing file for {model} lead {lead}: {nc_path}")
                continue
            ds = xr.open_dataset(nc_path)

            # build aligned arrays for this lead
            P, Y, R, M, times_used = arrays_for_lead(ds, lead)

            # subset on target (prediction) times
            submask = time_subset_mask(times_used)[args.subset]
            if args.subset != "all":
                P, Y, R, M = P[submask], Y[submask], R[submask], M[submask]

            met = compute_metrics_aligned(P, Y, R, M)
            rows.append({"model": model, "subset": args.subset, "lead_days": lead, **met})

    df = pd.DataFrame(rows).sort_values(["model","lead_days"])
    out_csv = OUT_ROOT / "lead_skill_summary.csv"
    df.to_csv(out_csv, index=False)

    show = ["model","subset","lead_days","rmse_log","skill_log_pct","rmse_mg","skill_mg_pct","r_log","n_valid"]
    with pd.option_context("display.max_rows", None, "display.width", 120, "display.precision", 3):
        print("\nLead-time metrics\n", df[show].to_string(index=False))
    print(f"\n✓ Wrote {out_csv}")

if __name__ == "__main__":
    main()