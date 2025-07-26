#!/usr/bin/env python3
"""
diagnostics.py ─ Temporal‑Fusion‑Transformer full diagnostics suite
======================================================================
*Feature‑parity* with the ConvLSTM v0.3 diagnostic bundle you shared:

  1. Global metrics (train / val / test / all) in log and linear space
  2. Monthly RMSE curves (model vs persistence)
  3. Bloom‑threshold contingency table + ROC curves
  4. Spatial skill maps (RMSE model, RMSE pers, % skill)
  5. Spatial means / biases
  6. Hex‑bin scatter, residual histogram, RMSE‑by‑latitude bars
  7. Domain‑mean time‑series plot
  8. Lead‑time skill curve (optional – if multi‑lead predictions provided)
  9. Case‑study snapshot maps & Hovmöller diagram

Exactly the *same* predictor list, SEQ, LEAD and data‑splits as used for
training.  The only dependency difference is `pytorch_forecasting` for
loading the TFT checkpoint.

Run example ────────────────────────────────────────────────────────────
python tft/diagnostics.py \
  --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --ckpt   "/Users/yashnilmohanty/HAB_Models/epoch=1-step=7534.ckpt" \
  --out    "Diagnostics_TFT_v0p1" \
  --seq    6 \
  --lead   1 \
  --batch  4096

The script will create <OUT>/ with CSV / NetCDF / PNG artefacts identical
in naming & format to the ConvLSTM diagnostics, enabling apples‑to‑apples
comparison.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, warnings, math
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from test import TFTNoScale

# ------------------------------------------------------------------ #
# Predictor constants  (MUST match training script)
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
# CLI
# ------------------------------------------------------------------ #
parser = argparse.ArgumentParser(description="Temporal‑Fusion‑Transformer diagnostics")
parser.add_argument("--freeze", required=True, help="HAB freeze NetCDF path")
parser.add_argument("--ckpt",   required=True, help="TFT lightning checkpoint path")
parser.add_argument("--out",    default="Diagnostics_TFT", help="Output directory")
parser.add_argument("--seq",    type=int, default=6)
parser.add_argument("--lead",   type=int, default=1)
parser.add_argument("--batch",  type=int, default=4096, help="Batches (#samples) per forward")
parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"], help="Inference device")
parser.add_argument("--no_plots", action="store_true", help="Skip heavy figure generation (for CI)")
args = parser.parse_args()

FREEZE  = pathlib.Path(args.freeze).expanduser().resolve()
CKPT    = pathlib.Path(args.ckpt  ).expanduser().resolve()
OUTDIR  = pathlib.Path(args.out   ).expanduser().resolve(); OUTDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Helper imports from the training script (same dir)
# ------------------------------------------------------------------ #
THIS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR))
from test import make_tensor, melt_to_df, PixelTSD   # noqa: E402  (these exist from training)

SEQ, LEAD = args.seq, args.lead

# ------------------------------------------------------------------ #
# 0)  Setup device & model
# ------------------------------------------------------------------ #
if args.device == "auto":
    if torch.cuda.is_available():
        DEV = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEV = torch.device("mps")
    else:
        DEV = torch.device("cpu")
else:
    DEV = torch.device(args.device)
print(f"[diag] using device: {DEV}")

print("[diag] loading TFT checkpoint …", flush=True)
tft: TFTNoScale = (
     TFTNoScale.load_from_checkpoint(CKPT, map_location=DEV)
    .to(DEV)
    .eval()
)

# ------------------------------------------------------------------ #
# 1)  Load freeze & rebuild tensors exactly like training
# ------------------------------------------------------------------ #
ds = xr.open_dataset(FREEZE, chunks={"time": -1}); ds.load()
df = melt_to_df(ds)
feat_tensor = make_tensor(
    df[ALL_VARS], (df.sid.nunique(), df.t.nunique(), len(ALL_VARS))
)
static_tensor = make_tensor(
    df.groupby("sid").first()[STATIC], (feat_tensor.shape[0], len(STATIC))
)

# Build *valid* core‑time mask identical to training logic --------------
chl = feat_tensor[..., 0]
valid_core = torch.zeros_like(chl, dtype=torch.bool)
core_slice = slice(SEQ - 1, -LEAD)
# history finite & future finite checks (same as training)
hist_ok = torch.isfinite(
    torch.stack([chl[:, i : i + SEQ] for i in range(chl.shape[1] - SEQ + 1)], -1)
).all(1)
if LEAD > 0:
    hist_ok = hist_ok[:, :-LEAD]
future_ok = torch.isfinite(chl[:, SEQ - 1 + LEAD :])
valid_core[:, core_slice] = hist_ok & future_ok

# ------------------------------------------------------------------ #
# 2)  PixelTSD dataset & batched inference over **all** samples
# ------------------------------------------------------------------ #
print("[diag] building PixelTSD dataset …", flush=True)
full_ds = PixelTSD(valid_core, feat_tensor, static_tensor, SEQ, LEAD)
loader  = DataLoader(full_ds, batch_size=args.batch, shuffle=False, num_workers=0)

n_time  = ds.sizes["time"]
H, W    = ds.sizes["lat"], ds.sizes["lon"]
core_idx = full_ds.core_index  # (N, 2) (pid, t_core)

# Pre‑allocate
pred_log = np.full((n_time - LEAD, H, W), np.nan, np.float32)  # predictions indexed by *k+lead*
true_log = np.full_like(pred_log, np.nan)
pers_log = np.full_like(pred_log, np.nan)
valid_m  = np.zeros_like(pred_log, bool)

print("[diag] running network inference over", len(loader), "batches …", flush=True)
with torch.no_grad():
    for batch_x, _ in tqdm(loader):
        # Move tensors to device, but keep small ones (indices etc.) on CPU
        batch_x = {k: v.to(DEV) if isinstance(v, torch.Tensor) else v for k,v in batch_x.items()}
        out     = tft(batch_x)
        y_hat   = out.prediction.squeeze(-1).cpu().numpy()  # (B, lead)
        # place forecasts into arrays
        pids    = batch_x["groups"].squeeze(-1).cpu().numpy()
        t_cores = batch_x["t_core"].cpu().numpy()
        k_plus  = t_cores + LEAD
        for i,(pid,kplus) in enumerate(zip(pids, k_plus)):
            iy = int(df.loc[df.sid==pid,"iy"].iloc[0]); ix = int(df.loc[df.sid==pid,"ix"].iloc[0])
            k_store = kplus - LEAD              # 0 …  (n_time-LEAD-1)

            pred_log[k_store, iy, ix] = y_hat[i, -1]
            # true & pers values
            true_log[k_store, iy, ix] = chl[pid, kplus].item()
            pers_log[k_store, iy, ix] = chl[pid, kplus-LEAD].item()
            valid_m [k_store, iy, ix] = True

# Align time coordinate for predictions (k+lead)  ----------------------
times_pred = ds.time.values[LEAD:]

# ------------------------------------------------------------------ #
# 3)  Bias‑correction (constant offset)
# ------------------------------------------------------------------ #
bias = np.nanmean(pred_log[valid_m] - true_log[valid_m])
print(f"[diag] constant bias = {bias:.4f} (log space) – subtracting …")
pred_log = pred_log - bias

# ------------------------------------------------------------------ #
# 4)  Metrics helpers (same maths as ConvLSTM diagnostics)
# ------------------------------------------------------------------ #
def _masked_rmse(a: np.ndarray, b: np.ndarray, m: np.ndarray):
    diff = (a - b)[m]
    return np.sqrt(np.mean(diff * diff)) if diff.size else np.nan

_floor = None  # mg m‑3 detection floor (set if you added before log)
exp_   = lambda x: np.clip(np.exp(x) - (_floor or 0.0), 0, None)

# Global metrics table --------------------------------------------------
tr_sel = times_pred <  np.datetime64("2016-01-01")
va_sel = (times_pred >= np.datetime64("2016-01-01")) & (times_pred <= np.datetime64("2018-12-31"))
te_sel = times_pred >  np.datetime64("2018-12-31")
subsets = {
    "train": tr_sel,
    "val"  : va_sel,
    "test" : te_sel,
    "all"  : slice(None)
}
rows=[]
for name, sel in subsets.items():
    P = pred_log[sel]
    R = pers_log[sel]
    Y = true_log[sel]
    M = valid_m [sel]
    row = {
        "subset"       : name,
        "rmse_log_model": _masked_rmse(P, Y, M),
        "rmse_log_pers" : _masked_rmse(R, Y, M),
        "mae_log_model" : np.nanmean(np.abs((P - Y)[M])),
        "mae_log_pers"  : np.nanmean(np.abs((R - Y)[M])),
        "rmse_mg_model" : _masked_rmse(exp_(P), exp_(Y), M),
        "rmse_mg_pers"  : _masked_rmse(exp_(R), exp_(Y), M),
    }
    rows.append(row)

df_global = pd.DataFrame(rows)
print("\nGLOBAL METRICS (log space):\n", df_global.to_string(index=False), sep="")
df_global.to_csv(OUTDIR/"metrics_global.csv", index=False)

# ------------------------------------------------------------------ #
# 5)  Dump predicted fields NetCDF for *all* downstream plotting
# ------------------------------------------------------------------ #
pred_da = xr.DataArray(pred_log, coords={"time":times_pred, "lat":ds.lat, "lon":ds.lon},
                       dims=("time","lat","lon"), name="log_chl_pred")
pers_da = xr.DataArray(pers_log, coords=pred_da.coords, dims=pred_da.dims, name="log_chl_pers")
true_da = xr.DataArray(true_log, coords=pred_da.coords, dims=pred_da.dims, name="log_chl_true")
mask_da = xr.DataArray(valid_m.astype("uint8"), coords=pred_da.coords, dims=pred_da.dims, name="valid_mask")
comp = {v:{"zlib":True,"complevel":4} for v in ["log_chl_pred","log_chl_pers","log_chl_true","valid_mask"]}
(xr.Dataset({"log_chl_pred":pred_da, "log_chl_pers":pers_da, "log_chl_true":true_da, "valid_mask":mask_da})
 .to_netcdf(OUTDIR/"predicted_fields.nc", encoding=comp))
print(f"[diag] saved predicted_fields.nc → {OUTDIR}")

# ------------------------------------------------------------------ #
# 6)  Advanced diagnostics plots & tables  (reuse convLSTM utilities)
# ------------------------------------------------------------------ #
if args.no_plots:
    sys.exit("✓ Diagnostics summary + NetCDF ready (plots skipped).")

# Import the heavy plotting utility from the convLSTM suite dynamically –
# we re‑use 100% of that code to guarantee identical figures.
PLOT_UTIL = THIS_DIR / "spatial_bias_maps_plus.py"
if not PLOT_UTIL.exists():
    warnings.warn("Could not find spatial_bias_maps_plus.py in the repo ⇒ skipping figure generation")
    sys.exit("✓ Core metrics ready, but rich plots skipped (utility script missing).")

# Execute plotting script via runpy so it writes into OUTDIR but doesn’t pollute globals
override = {
    "FREEZE": FREEZE,
    "PRED"  : OUTDIR/"predicted_fields.nc",
    "OUTDIR": OUTDIR,
}
import runpy
runpy.run_path(str(PLOT_UTIL), init_globals=override)

print("\n✓ TFT diagnostics complete →", OUTDIR)

'''
GLOBAL METRICS (log space):
subset  rmse_log_model  rmse_log_pers  mae_log_model  mae_log_pers  rmse_mg_model  rmse_mg_pers
 train        1.218311       0.892674       0.995388      0.617396       6.642223      6.298439
   val        1.147608       0.854255       0.935876      0.596557       5.996653      5.800528
  test        1.158417       0.888878       0.929460      0.624285       6.532279      6.138979
   all        1.198996       0.885994       0.976696      0.614930       6.526062      6.198111
[diag] saved predicted_fields.nc → /Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_TFT_v0p1

Bloom contingency (≥5 mg m-3): {'metric': 'bloom_≥5mg', 'hits': 166443, 'miss': 102215, 'false_alarm': 93415, 'POD': 0.6195348733333806, 'FAR': 0.3594847955421794, 'CSI': 0.4596945919745453}

GLOBAL METRICS (model vs obs):
 space     RMSE      MAE          Bias  Pearson_r  Spearman_rho      NSE      KGE
linear 5.118773 2.200955 -5.953968e-01   0.575584      0.773834 0.320751 0.402367
   log 0.792717 0.580946 -2.015892e-07   0.771778      0.773834 0.562880 0.766944
'''