#!/usr/bin/env python3
"""
tft_hab_model.py  —  Temporal Fusion Transformer baseline v0.1
==============================================================
Forecasts 8-day-ahead natural-log chlorophyll-a (log_chl) for every
valid 4 km coastal pixel in the analysis-ready freeze
`HAB_convLSTM_core_v1_clean.nc`.

Key parallels with the ConvLSTM baseline
----------------------------------------
* Predictor set: the exact same 30 variables (satellite, meteo, ocean, derived)
  plus 3 static covariates.
* History window  : SEQ = 6  →  48 days of context
* Lead time       : LEAD = 1 →  +8 day forecast
* Train/Val/Test  : <2016-01-01 / 2016-01-01–2018-12-31 / >2018-12-31
* Target          : log_chl (already log-transformed with detection floor)
* Metrics         : RMSE_log, Quantile loss; persistence RMSE for skill

Dependencies
------------
    pip install torch pytorch-lightning==2.2.2 pytorch-forecasting==1.0.0
    pip install xarray netCDF4 pandas numpy tqdm

Running
------------

python tft/tft_hab_model.py \
  --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --epochs 60 --batch 256 --seq 6 --lead 1
"""

'''

Sanity check to ensure all dependencies are installed:

python - <<'PY'
import torch, lightning, pytorch_forecasting, sys
print("Torch:", torch.__version__)
print("Lightning:", lightning.__version__)
print("PForecast:", pytorch_forecasting.__version__)
PY

'''

import argparse
import json
import math
import pathlib
from itertools import product  # (kept in case you add grid-search)
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)
from pytorch_forecasting.metrics import QuantileLoss

import torch.multiprocessing as mp
import os

mp.set_start_method("spawn", force=True)
# Disable the NumPy-arrow fast path that seg-faults on macOS/py3.10
os.environ["DATASET_PROCESSING_NUMPY"] = "0"

# ──────────────────────────────────────────────────────────────────────
# CONFIG (override at CLI)
# ──────────────────────────────────────────────────────────────────────
FREEZE   = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/"
    "HAB_convLSTM_core_v1_clean.nc"
)
SEQ       = 6        # 48-day encoder length
LEAD_IDX  = 1        # 8-day prediction horizon (1 step)
BATCH     = 256      # ≈ GPU-dependent; 256 fits 12 GB for this dataset
EPOCHS    = 60
SEED      = 42
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR   = pathlib.Path.home() / "HAB_Models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Predictor lists (identical to ConvLSTM)
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = [
    "u10", "v10", "wind_speed", "tau_mag",
    "avg_sdswrf", "tp", "t2m", "d2m",
]
OCEAN     = [
    "uo", "vo", "cur_speed", "cur_div", "cur_vort",
    "zos", "ssh_grad_mag", "so", "thetao",
]
DERIVED   = [
    "chl_anom_monthly",
    "chl_roll24d_mean", "chl_roll24d_std",
    "chl_roll40d_mean", "chl_roll40d_std",
]
STATIC    = ["river_rank", "dist_river_km", "ocean_mask_static"]

ALL_VARS      = SATELLITE + METEO + OCEAN + DERIVED + STATIC
STATIC_VARS   = STATIC
_FLOOR        = 0.056616  # detection floor used in log transform

torch.manual_seed(SEED)
np.random.seed(SEED)
pl.seed_everything(SEED, workers=True)

# ──────────────────────────────────────────────────────────────────────
# 1. Helper — melt (time,lat,lon) cube → long dataframe
# ──────────────────────────────────────────────────────────────────────
def melt_dataset_to_long(
    ds: xr.Dataset, pixel_mask: xr.DataArray
) -> pd.DataFrame:
    """
    Each valid (lat,lon) pixel becomes a unique series_id.
    Returns long-format DataFrame with columns:
        series_id | time_idx | date | dynamic vars … | target | static vars …
    """
    lat_idx, lon_idx = np.where(pixel_mask.values)
    series_ids = np.arange(len(lat_idx))
    time_idx = np.arange(ds.sizes["time"])
    dates = pd.to_datetime(ds.time.values)

    frames = []
    for sid, iy, ix in tqdm(
        zip(series_ids, lat_idx, lon_idx),
        total=len(series_ids),
        desc="Melting pixels → long format",
    ):
        dct = {
            "series_id": sid,
            "time_idx": time_idx,
            "date": dates,
        }
        # Dynamic variables (time-varying)
        for v in ALL_VARS:
            if v not in STATIC_VARS:
                dct[v] = ds[v][:, iy, ix].values
        # Target (already in dynamic set)
        dct["log_chl"] = ds["log_chl"][:, iy, ix].values
        df = pd.DataFrame(dct)

        # Static vars (broadcast)
        for v in STATIC_VARS:
            if v in ds.data_vars:
                if "time" in ds[v].dims:
                    val = float(ds[v][0, iy, ix].values)
                else:
                    val = float(ds[v][iy, ix].values)
            else:
                val = np.nan
            df[v] = val

        # Drop rows with NaN target
        df = df[np.isfinite(df["log_chl"])]
        frames.append(df)

    return pd.concat(frames, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────
# 2. Helper — build TimeSeriesDataSet & dataloaders
# ──────────────────────────────────────────────────────────────────────
def create_dataloaders(
    df: pd.DataFrame,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    # Date-based split indices (same thresholds as ConvLSTM)
    dates = df.groupby("time_idx").first()["date"].sort_index()
    train_cut = dates.searchsorted(np.datetime64("2016-01-01")) - 1
    val_start = dates.searchsorted(np.datetime64("2016-01-01"))
    val_end   = dates.searchsorted(np.datetime64("2018-12-31"))
    test_start = val_end + 1

    max_encoder_length = SEQ
    max_prediction_length = LEAD_IDX

    training = TimeSeriesDataSet(
        df[df.time_idx <= train_cut],
        time_idx="time_idx",
        target="log_chl",
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=STATIC_VARS,
        time_varying_known_reals=[],    # none for +8 d horizon
        time_varying_unknown_reals=[
            v for v in ALL_VARS if v not in STATIC_VARS
        ],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df[(df.time_idx >= val_start) & (df.time_idx <= val_end)],
        stop_randomization=True,
    )
    testing = TimeSeriesDataSet.from_dataset(
        training,
        df[df.time_idx >= test_start],
        stop_randomization=True,
    )

    train_dl = training.to_dataloader(
    train=True, batch_size=BATCH, num_workers=0, shuffle=True
    )
    val_dl = validation.to_dataloader(
        train=False, batch_size=BATCH, num_workers=0
    )
    test_dl = testing.to_dataloader(
        train=False, batch_size=BATCH, num_workers=0
    )
    return train_dl, val_dl, test_dl, training

# ──────────────────────────────────────────────────────────────────────
# 3. Helper — RMSE of simple persistence baseline
# ──────────────────────────────────────────────────────────────────────
def rmse_persistence(loader):
    errs = []
    for batch in loader:
        y_true = batch[0]["target"][..., -1]          # (B, 1)
        last_enc = batch[0]["encoder_target"][..., -1]
        mask = torch.isfinite(y_true)
        errs.append(((y_true - last_enc)[mask] ** 2).cpu().numpy())
    return math.sqrt(np.concatenate(errs).mean())

# ──────────────────────────────────────────────────────────────────────
# 4. Training routine
# ──────────────────────────────────────────────────────────────────────
def train_tft(df: pd.DataFrame):
    train_dl, val_dl, test_dl, training = create_dataloaders(df)

    model = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=64,
        lstm_layers=2,
        attention_head_size=4,
        dropout=0.10,
        hidden_continuous_size=32,
        learning_rate=3e-4,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=3,
        log_interval=10,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=OUT_DIR,
        filename="tft_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    early_cb = EarlyStopping(monitor="val_loss", patience=8, mode="min")
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="cpu",      # <-- force CPU
        devices=1,
        precision=32,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb, early_cb, lr_cb],
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dl, val_dl)

    best = TemporalFusionTransformer.load_from_checkpoint(
        ckpt_cb.best_model_path
    )

    val_metrics  = best.test(val_dl, verbose=False)[0]
    test_metrics = best.test(test_dl, verbose=False)[0]

    # Persistence reference
    val_rmse_p  = rmse_persistence(val_dl)
    test_rmse_p = rmse_persistence(test_dl)

    metrics = {
        "val":  {**val_metrics,  "rmse_persistence": val_rmse_p},
        "test": {**test_metrics, "rmse_persistence": test_rmse_p},
    }
    with open(OUT_DIR / "tft_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n✓ TFT evaluation complete")
    print(json.dumps(metrics, indent=2))
    print("Best checkpoint:", ckpt_cb.best_model_path)

# ──────────────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────────────
def main():

    global FREEZE, EPOCHS, BATCH, SEQ, LEAD_IDX

    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=str, default=str(FREEZE))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch",  type=int, default=BATCH)
    ap.add_argument("--seq",    type=int, default=SEQ)
    ap.add_argument("--lead",   type=int, default=LEAD_IDX)
    args = ap.parse_args()

    FREEZE   = pathlib.Path(args.freeze).expanduser()
    EPOCHS   = args.epochs
    BATCH    = args.batch
    SEQ      = args.seq
    LEAD_IDX = args.lead

    # Load NetCDF (into RAM once; small at 851 × 240 × 240)
    ds = xr.open_dataset(FREEZE, chunks={"time": -1})
    ds.load()

    # Build/derive pixel_ok mask (≥20 % finite log_chl)
    if "pixel_ok" in ds:
        pixel_mask = ds.pixel_ok.astype(bool)
    else:
        frac = np.isfinite(ds.log_chl).sum("time") / ds.sizes["time"]
        pixel_mask = frac >= 0.20

    df = melt_dataset_to_long(ds, pixel_mask)
    train_tft(df)

if __name__ == "__main__":
    main()
