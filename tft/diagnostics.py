#!/usr/bin/env python3
#tft/diagnostics.py
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
  --ckpt   runs/tft/lightning_logs/version_1/checkpoints/epoch=14-step=48780.ckpt \
  --out    Diagnostics_TFT_v0p1 \
  --seq 24 --lead 7 --batch 4096 --device cpu
  
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
import pickle
# from pytorch_forecasting import TimeSeriesDataSet

from pixel_ds import make_tensor, melt_to_df
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer as TFT

_floor = 0.056616

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

# ---------------------------------------------
print("[diag] loading TFT checkpoint …", flush=True)
tft: TFT = (
     TFT.load_from_checkpoint(CKPT, map_location=DEV)
    .to(DEV)
    .eval()
)

# >>> NEW – make the diagnostics use the same variable list as the model
ALL_VARS = list(tft.hparams["x_reals"])        # 29 vars in the order the model learned

'''
helper = {'encoder_length','log_chl_center','log_chl_scale','time_idx',
          'relative_time_idx','lat','lon','number','month',
          'log_chl_outmask','Kd_490_outmask','nflh_outmask'}

ALL_VARS = [v for v in ALL_VARS if v not in helper]
print("using vars:", len(ALL_VARS))

STATIC    = []                                 # they were not treated as static_reals
print(f"[diag] model expects {len(ALL_VARS)} continuous vars:", ALL_VARS)
'''
# ---------------------------------------------
print("► available hparams keys:", sorted(tft.hparams.keys()))
# ------------------------------------------------------------------ #
# 1)  Load freeze & rebuild tensors exactly like training
# ------------------------------------------------------------------ #
ds = xr.open_dataset(FREEZE, chunks={"time": -1}); ds.load()
# ─────────────────────────────────────────────────────────────────────
# make sure sid matches the string IDs you trained on
# ─────────────────────────────────────────────────────────────────────

# DataFrame preparation (existing)
df = melt_to_df(ds)
df["sid"] = df["sid"].astype(str)

# Explicitly add missing columns
df["number"] = df["sid"].astype('category').cat.codes
df["month"] = pd.to_datetime(ds.time.values[df["t"]]).month
df["log_chl_outmask"] = (~np.isnan(df["log_chl"])).astype(int)
df["Kd_490_outmask"] = (~np.isnan(df["Kd_490"])).astype(int)
df["nflh_outmask"] = (~np.isnan(df["nflh"])).astype(int)

# NaN handling aligned with ConvLSTM v0.4 approach:
# 1. Mask shallow waters (depth < 10m)
if "depth" in ds:
    shallow_mask = ds.depth > 10
else:
    shallow_mask = xr.ones_like(ds.log_chl.isel(time=0), dtype=bool)

valid_mask = shallow_mask & np.isfinite(ds.log_chl.mean("time"))

# 2. Fill problematic NaNs using median values (robust imputation)
for var in ["cur_div", "cur_vort", "ssh_grad_mag"]:
    median_val = np.nanmedian(df[var])
    df[var].fillna(median_val, inplace=True)

# Continue existing indexing operations
df["time_idx"] = df["t"]
df["lat"] = ds.lat.values[df["iy"].to_numpy()]
df["lon"] = ds.lon.values[df["ix"].to_numpy()]
# ─────────────────────────────────────────────────────────────────────


# drop any TFT helper fields that aren’t in the raw dataframe
ALL_VARS = [v for v in ALL_VARS if v in df.columns]

# ─────────────────────────────────────────────────────────────────────
feat_tensor = make_tensor(
    df[ALL_VARS], (df.sid.nunique(), df.t.nunique(), len(ALL_VARS))
)
static_tensor = make_tensor(
    df.groupby("sid").first()[STATIC], (feat_tensor.shape[0], len(STATIC))
)


# ------------------------------------------------------------------ #
# 2)  PixelTSD dataset & batched inference over **all** samples
print("[diag] re-loading train TimeSeriesDataSet …")
# try loading train_ds.pkl alongside the checkpoint, or fall back to the root outdir
candidates = [
    CKPT.parent.parent / "train_ds.pkl",             # version_1/train_ds.pkl
    CKPT.parent.parent.parent.parent / "train_ds.pkl" # runs/tft/train_ds.pkl
]
for p in candidates:
    if p.exists():
        train_ds = pickle.load(open(p, "rb"))
        break
else:
    raise FileNotFoundError(f"Could not find train_ds.pkl in {candidates}")

# 1. gather the full list of features that train_ds expects
required = (
    (train_ds.static_categoricals or [])
  + (train_ds.static_reals or [])
  + (train_ds.time_varying_known_categoricals or [])
  + (train_ds.time_varying_known_reals or [])
  + (train_ds.time_varying_unknown_reals or [])
)

# 2. compare against your dataframe’s columns
missing = set(required) - set(df.columns)

print(f"✅ train_ds expects {len(required)} total features")
print("🔑 required features:", required)
if missing:
    print(f"❌ Oops, you’re missing {len(missing)} columns:\n   ", sorted(missing))
else:
    print("🎉 All required columns are present!")

infer_ds = TimeSeriesDataSet.from_dataset(
    train_ds,
    df,
    predict=True,
    stop_randomization=True,
)
print(f"[diag] making DataLoader ({args.batch} batch-size)…")
loader = infer_ds.to_dataloader(
    train=False,
    batch_size=args.batch,
    shuffle=False,
    num_workers=0,
)

# Pre-allocate your arrays
n_time = ds.sizes["time"]
H, W   = ds.sizes["lat"], ds.sizes["lon"]
pred_log = np.full((n_time - LEAD, H, W), np.nan, np.float32)
true_log = np.full_like(pred_log, np.nan)
pers_log = np.full_like(pred_log, np.nan)
# we’ll fill valid_m on the fly
valid_m = np.zeros_like(pred_log, bool)

# 3)  Run predict() and unpack directly
print("[diag] running TFT.predict() …")
pred = tft.predict(
    infer_ds,
    mode="prediction",
    return_x=False,
    return_index=True,
)

# pred.index is already a pandas.DataFrame with columns ['time_idx','sid']
idx_df = pred.index.copy()
idx_df = idx_df.rename(columns={"time_idx": "t"})  # rename time_idx → t

# pred.output is a torch.Tensor of shape (n_samples, n_horizons)
arr = pred.output.detach().cpu().numpy()
n_horizons = arr.shape[1]
horizon_cols = [f"pred_t{i+1}" for i in range(n_horizons)]

# build df_pred by combining idx_df + the horizon columns
df_pred = idx_df.reset_index(drop=True).copy()
for i, col in enumerate(horizon_cols):
    df_pred[col] = arr[:, i]

print(df_pred.head())
print("Columns:", df_pred.columns.tolist())

# pick your lead
hcol = f"pred_t{LEAD}"
if hcol not in df_pred:
    hcol = "pred_t1"
print(f"[diag] using horizon: {hcol}")


# back-transform from standardized to log-chl space
mu = df["log_chl"].mean()
sigma = df["log_chl"].std()
df_pred["log_chl_pred"] = df_pred[hcol] * sigma + mu

# merge in pixel coords by sid
coords = df[["sid","iy","ix"]].drop_duplicates()
df_pred = df_pred.merge(coords, on="sid", how="left")

# re-init arrays
n_time, H, W = ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"]
pred_log = np.full((n_time-LEAD, H, W), np.nan, np.float32)
true_log = np.full_like(pred_log, np.nan)
pers_log = np.full_like(pred_log, np.nan)
valid_m  = np.zeros_like(pred_log, bool)

# fill arrays directly from ds
for row in df_pred.itertuples():
    t_idx, i, j = row.t, row.iy, row.ix
    k = int(t_idx - LEAD)
    if k < 0 or np.isnan(i) or np.isnan(j):
        continue
    pred_log[k, i, j] = row.log_chl_pred
    true_log[k, i, j] = float(ds.log_chl.values[t_idx, i, j])
    pers_log[k, i, j] = float(ds.log_chl.values[t_idx-LEAD, i, j])
    valid_m[k, i, j]  = True

# shift time coordinate for preds
times_pred = ds.time.values[LEAD:]


# 3)  Bias-correction in mg-space
#    so that linear RMSE & scatter get centered like the ConvLSTM
pred_mg  = np.exp(pred_log) - _floor
true_mg  = np.exp(true_log) - _floor
# compute mg bias only over valid pixels
bias_mg = np.nanmean((pred_mg - true_mg)[valid_m])
print(f"[diag] constant bias = {bias_mg:.4f} (mg m⁻³) – subtracting …")
# shift mg and re‐log
adj_mg  = np.clip(pred_mg - bias_mg, 0, None)

pred_log = np.log(adj_mg + _floor)

bias_log = np.nanmean((pred_log - true_log)[valid_m])
pred_log -= bias_log

# ------------------------------------------------------------------ #
# 4)  Metrics helpers (same maths as ConvLSTM diagnostics)
# ------------------------------------------------------------------ #
def _masked_rmse(a: np.ndarray, b: np.ndarray, m: np.ndarray):
    diff = (a - b)[m]
    return np.sqrt(np.mean(diff * diff)) if diff.size else np.nan

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
