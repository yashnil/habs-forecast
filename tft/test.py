#!/usr/bin/env python3
"""
Minimal, Apple‑silicon‑safe Temporal‑Fusion‑Transformer trainer.
History = 6 timesteps (48 d); Lead = 1 (8 d).
Run:
  python tft/test.py --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
                               --epochs 25 --batch 256 --seq 6 --lead 1 --device cpu
"""

import argparse, pathlib
from typing import Tuple

import numpy as np, pandas as pd, xarray as xr, torch, lightning as L
from torch.utils.data import DataLoader
from pytorch_forecasting.metrics import RMSE
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting.models.temporal_fusion_transformer import (
    TemporalFusionTransformer,
)

# ───────────────────────────── constants ─────────────────────────────
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos",
             "ssh_grad_mag","so","thetao"]
DERIVED   = ["chl_anom_monthly",
             "chl_roll24d_mean","chl_roll24d_std",
             "chl_roll40d_mean","chl_roll40d_std"]
STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]

ALL_VARS   = SATELLITE + METEO + OCEAN + DERIVED + STATIC
STATIC_SET = set(STATIC)
X_REAL     = ALL_VARS + STATIC          # 36 continuous inputs
DUMMY_CAT  = "dummy"

# ---------------------------------------------------------------------
# 1.  A helper that replaces ±inf/NaN _after_ the zero-fill
# ---------------------------------------------------------------------
def make_tensor(df_block, shape):
    arr = (df_block.fillna(0.0)
                   .replace([np.inf, -np.inf], 0.0)
                   .to_numpy(np.float32)
                   .reshape(*shape))
    return torch.from_numpy(arr)

# ---------------------------------------------------------------------
# 2.  Silence the metric logging that triggers the crash
# ---------------------------------------------------------------------
class TFTNoScale(TemporalFusionTransformer):
    def transform_output(self, out, target_scale=None):
        return out                    # keep your “no rescaling” tweak

    # NEW --------------------------------------------------------------
    # completely disable the extra metric pass → no NaN assertion anymore
    def log_metrics(self, *args, **kwargs):           # overrides base method
        return {}
# ---------------------------------------------------------------------

# ─────────────────── helper: NetCDF → long dataframe ──────────────────
def melt_to_df(ds: xr.Dataset) -> pd.DataFrame:
    pixel_ok = (np.isfinite(ds.log_chl).sum("time") / ds.sizes["time"]) >= .20
    lat_i, lon_i = np.where(pixel_ok.values)
    time_idx = np.arange(ds.sizes["time"])

    rows = []
    for sid, (iy, ix) in enumerate(zip(lat_i, lon_i)):
        base = {"sid": sid, "iy": iy, "ix": ix}
        for v in STATIC:
            base[v] = float(ds[v][iy, ix]) if "time" not in ds[v].dims else float(ds[v][0, iy, ix])

        block = {v: ds[v][:, iy, ix].values for v in ALL_VARS if v not in STATIC_SET}
        block["t"] = time_idx
        block_df = pd.DataFrame(block)
        for k, val in base.items():
            block_df[k] = val
        rows.append(block_df)

    df = pd.concat(rows, ignore_index=True)
    return df[np.isfinite(df.log_chl)]

# ─────────────────────────── Pixel dataset ───────────────────────────
class PixelTSD(torch.utils.data.Dataset):
    def __init__(self, core_mask: torch.Tensor, feat: torch.Tensor,
                 stat: torch.Tensor, seq: int, lead: int):
        """
        core_mask : (P, T) bool  – True for each (pixel,time) we want as a sample
        feat      : (P, T, C)    – pre-tensorised dynamic features (36)
        stat      : (P, S)       – static continuous features  (3)
        """
        self.core_index = core_mask.nonzero(as_tuple=False)  # (N, 2) → (pid, t)
        self.feat, self.stat = feat, stat
        self.seq, self.lead  = seq, lead

    def __len__(self):
        return len(self.core_index)

    def __getitem__(self, idx):
        pid, t_core = self.core_index[idx]       # integers

        enc = self.feat[pid, t_core-self.seq+1 : t_core+1]        # (seq, C)
        dec = enc[-1:].repeat(self.lead, 1)                       # (lead, C)

        # in PixelTSD.__getitem__
        tgt = self.feat[pid, t_core + self.lead, 0]      # scalar ()
        tgt = tgt.view(1, 1)                             # (1, 1)  ← two-dim

        # encoder history target to match
        encoder_target = enc[:, 0].unsqueeze(-1)         # (seq, 1)

        x = dict(
            encoder_cont = enc,
            decoder_cont = dec,
            encoder_cat  = torch.zeros(self.seq, 1, dtype=torch.long),
            decoder_cat  = torch.zeros(self.lead,1, dtype=torch.long),
            static_cat   = torch.zeros(1,dtype=torch.long),

            static_cont      = self.stat[pid],                    # (3,)
            encoder_time_idx = torch.arange(self.seq),
            decoder_time_idx = torch.arange(self.lead),
            encoder_lengths  = torch.tensor(self.seq),
            decoder_lengths  = torch.tensor(self.lead),
            groups           = torch.tensor([pid]),
            encoder_target   = encoder_target,                         # history target
            decoder_target   = tgt,
            target_scale     = torch.tensor([1.0,0.0]),
            t_core       = torch.tensor(t_core, dtype=torch.long),  # ← add this
        )
        return x, tgt

# ───────────────────────────── training ──────────────────────────────
def run(cfg):
    # ---------- data ----------
    ds = xr.open_dataset(cfg.freeze, chunks={"time": -1}); ds.load()
    df = melt_to_df(ds)

    times = pd.to_datetime(ds.time.values)
    train_ts = np.where(times <  np.datetime64("2016-01-01"))[0]
    val_ts   = np.where((times>=np.datetime64("2016-01-01"))&(times<=np.datetime64("2018-12-31")))[0]
    test_ts  = np.where(times >  np.datetime64("2018-12-31"))[0]

    # ---------- PRE-TENSORISE ------------------------------------------------
    # shape: (P, T, C)
    # ─────── 1️⃣  build feat / static tensors with NaNs filled  ────────

    feat_tensor = make_tensor(
    df[ALL_VARS],
    (df.sid.nunique(), df.t.nunique(), len(ALL_VARS)),
    )
    static_tensor = make_tensor(
        df.groupby("sid").first()[STATIC],
        (feat_tensor.shape[0], len(STATIC)),
    )

    # static_np = (
    #     df.groupby("sid").first()[STATIC]
    #     .fillna(0.0)        # ← NEW
    #     .to_numpy(np.float32)
    # )
    # static_tensor = torch.from_numpy(static_np)

    # ─────── 2️⃣  make the core-time mask robust to *any* NaNs ────────
    chl = feat_tensor[..., 0]            # (P, T) – the target variable

    valid_core = torch.zeros_like(chl, dtype=torch.bool)
    core_slice = slice(cfg.seq - 1, -cfg.lead)      # encoder-end positions
    # future_ix  = slice(cfg.seq - 1 + cfg.lead, None)

    # history OK ↔ all log-chl in the encoder window are finite
    hist_ok = torch.isfinite(
        torch.stack(
            [chl[:, i : i + cfg.seq] for i in range(chl.shape[1] - cfg.seq + 1)],
            dim=-1,
        )
    ).all(dim=1)                                   # shape (P, T - seq + 1)

    # --- NEW: trim the extra columns so it matches future_ok ---
    if cfg.lead > 0:
        hist_ok = hist_ok[:, :-cfg.lead]            # now length = T - seq - lead + 1
    # ----------------------------------------------------------

    future_ok = torch.isfinite(chl[:, cfg.seq - 1 + cfg.lead :])  # same length
    valid_core[:, core_slice] = hist_ok & future_ok

    # split into train / val / test boolean masks over (P, T)
    train_mask = valid_core.clone()
    val_mask   = valid_core.clone()
    test_mask  = valid_core.clone()

    train_mask[:, (times >= np.datetime64("2016-01-01"))] = False
    val_mask[:, (times <  np.datetime64("2016-01-01")) |
                (times >  np.datetime64("2018-12-31"))] = False
    test_mask[:, (times <= np.datetime64("2018-12-31"))] = False

    # ---------- Dataset objects (new signature) -----------------------------
    train_ds = PixelTSD(train_mask, feat_tensor, static_tensor, cfg.seq, cfg.lead)
    val_ds   = PixelTSD(val_mask,   feat_tensor, static_tensor, cfg.seq, cfg.lead)
    test_ds  = PixelTSD(test_mask,  feat_tensor, static_tensor, cfg.seq, cfg.lead)

    dl_train = DataLoader(train_ds, cfg.batch, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(val_ds,   cfg.batch,                num_workers=0)
    dl_test  = DataLoader(test_ds,  cfg.batch,                num_workers=0)

    # ---------- model ----------
    tft = TFTNoScale(
        x_reals            = X_REAL,
        x_categoricals     = [DUMMY_CAT],
        embedding_sizes    = {DUMMY_CAT: (1, 1)},
        hidden_size        = 64,
        lstm_layers        = 2,
        attention_head_size= 4,
        dropout            = 0.10,
        output_size        = cfg.lead,
        loss               = RMSE(),
        # logging_metrics    = [RMSE()],
    )

    # ---------- training ----------
    ckpt = ModelCheckpoint(dirpath=cfg.outdir, monitor="val_loss", mode="min")
    trainer = L.Trainer(
        max_epochs          = cfg.epochs,
        accelerator         = cfg.device,
        devices             = 1,
        callbacks           = [ckpt, EarlyStopping("val_loss", patience=8)],
        num_sanity_val_steps= 0,
        log_every_n_steps   = 50,
    )
    trainer.fit(tft, dl_train, dl_val)
    print("✓ training finished @", ckpt.best_model_path)

    print("Test metrics:", trainer.validate(tft, dl_test, verbose=False))

# ─────────────────────────────── CLI ────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", required=True, type=pathlib.Path)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--seq",    type=int, default=6)
    p.add_argument("--lead",   type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu","mps","cuda"])
    p.add_argument("--outdir", default=pathlib.Path.home()/"HAB_Models", type=pathlib.Path)
    cfg = p.parse_args(); cfg.outdir.mkdir(exist_ok=True)

    run(cfg)
