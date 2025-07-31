#!/usr/bin/env python3
"""
Temporal‑Fusion‑Transformer (TFT) trainer · California HABs
===========================================================
A single, self‑contained script that trains a Temporal‑Fusion‑Transformer
on the analysis‑ready coastal‑strip NetCDF:
    HAB_convLSTM_core_v1_clean.nc

Highlights
----------
* **Shape‑safe** — `transform_output()` copes with (B,2), (B,1,2) _or_ (B,L,1)
  `target_scale` tensors Lightning sometimes emits.
* **Device‑flexible** — choose `--device cpu|mps|cuda`; falls back gracefully.
* **Fast DataLoader** — pinned memory, `num_workers=min(8, nCPU)` by default.
* **Quiet logging** — no metric spam; progress every 50 steps.

Example (CUDA)
--------------
```bash
python tft/test.py \
  --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --epochs 25 --batch 256 --seq 6 --lead 1 --device cpu
```
Swap `cuda` for `mps` or `cpu` as needed.
"""

import argparse, pathlib, os, warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import xarray as xr
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback        # ← NEW
from torch.utils.data import DataLoader, default_collate
from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchPlateau
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,   # ← add this
)
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

# ───────────────────────────── constants ─────────────────────────────
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = ["u10", "v10", "wind_speed", "tau_mag", "avg_sdswrf", "tp", "t2m", "d2m"]
OCEAN     = ["uo", "vo", "cur_speed", "cur_div", "cur_vort", "zos", "ssh_grad_mag", "so", "thetao"]
DERIVED   = [
    "chl_anom_monthly",
    "chl_roll24d_mean", "chl_roll24d_std",
    "chl_roll40d_mean", "chl_roll40d_std",
]
STATIC    = ["river_rank", "dist_river_km", "ocean_mask_static"]

ALL_VARS   = SATELLITE + METEO + OCEAN + DERIVED + STATIC
STATIC_SET = set(STATIC)
X_REAL     = ALL_VARS + STATIC            # 36 continuous inputs
dummy_cat  = "dummy"

# --- add this helper near the top of the file --------------------
class PrintLossCallback(Callback):    
    def on_validation_epoch_end(self, trainer, pl_module):
        logs = trainer.callback_metrics     # ← dict with everything Lightning logged
        epoch = trainer.current_epoch
        trn  = logs.get("train_loss_epoch")
        val  = logs.get("val_loss")
        if trn is not None and val is not None:
            print(f"\n\x1b[36mepoch={epoch:02d}  train={trn:.3f}  val={val:.3f}\x1b[0m")
# -----------------------------------------------------------------

# ───────────────────────── helper functions ─────────────────────────

def make_tensor(df_block: pd.DataFrame, shape: Tuple[int, ...]) -> torch.Tensor:
    """Fill NaNs/inf ➜ 0, cast float32, reshape → tensor."""
    return torch.from_numpy(
        df_block.fillna(0.0)
                .replace([np.inf, -np.inf], 0.0)
                .to_numpy(np.float32)
                .reshape(*shape)
    )

def melt_to_df(ds: xr.Dataset) -> pd.DataFrame:
    """Flatten (time, lat, lon) cube into long pixel DataFrame."""
    pixel_ok = (np.isfinite(ds.log_chl).sum("time") / ds.sizes["time"]) >= 0.20
    lat_i, lon_i = np.where(pixel_ok.values)
    time_idx = np.arange(ds.sizes["time"], dtype=np.int32)

    rows: List[pd.DataFrame] = []
    for sid, (iy, ix) in enumerate(zip(lat_i, lon_i)):
        base = {"sid": sid, "iy": iy, "ix": ix}
        for v in STATIC:
            base[v] = float(ds[v][iy, ix]) if "time" not in ds[v].dims else float(ds[v][0, iy, ix])
        block = {v: ds[v][:, iy, ix].values for v in ALL_VARS if v not in STATIC_SET}
        block["t"] = time_idx
        df_block = pd.DataFrame(block)
        for k, val in base.items():
            df_block[k] = val
        rows.append(df_block)

    df = pd.concat(rows, ignore_index=True)
    return df[np.isfinite(df.log_chl)]

# ───────────────────────── custom Dataset ───────────────────────────
class PixelTSD(torch.utils.data.Dataset):
    def __init__(self, core_mask: torch.Tensor, feat: torch.Tensor, stat: torch.Tensor, seq: int, lead: int):
        self.core_index = core_mask.nonzero(as_tuple=False)  # (N,2)
        self.feat, self.stat = feat, stat
        self.seq, self.lead = seq, lead
        self.T = feat.shape[1]

    def __len__(self):
        return len(self.core_index)

    def __getitem__(self, idx):
        pid, t_core = self.core_index[idx].tolist()
        enc = self.feat[pid, t_core - self.seq + 1 : t_core + 1]  # (seq,C)
        dec = enc[-1:].repeat(self.lead, 1)                       # (lead,C)
        tgt         = self.feat[pid, t_core + self.lead, 0].view(1,1)
        hist_target = enc[:, 0].unsqueeze(-1)
        target_scale = torch.stack([
            hist_target.mean(),
            hist_target.std().clamp_min(1e-4)
        ])
        return dict(
            encoder_cont=enc,
            decoder_cont=dec,
            encoder_cat=torch.zeros(self.seq,1,dtype=torch.long),
            decoder_cat=torch.zeros(self.lead,1,dtype=torch.long),
            static_cat=torch.zeros(1,dtype=torch.long),
            static_cont=self.stat[pid],
            encoder_time_idx=torch.arange(self.seq,dtype=torch.long),
            decoder_time_idx=torch.arange(1, self.lead+1, dtype=torch.long),
            encoder_lengths=torch.tensor(self.seq),
            decoder_lengths=torch.tensor(self.lead),
            groups=torch.tensor([pid]),
            encoder_target=hist_target,
            decoder_target=tgt,
            target_scale=target_scale,
        ), tgt

def collate_batch(batch):
    xs, ys = zip(*batch)
    return default_collate(xs), default_collate(ys)

# ────────────────────── TFT wrapper ─────────────────────
class TFTBare(TemporalFusionTransformer):
    """TFT with a rock‑solid transform_output that handles both dict- and tensor‑style outputs."""

    @staticmethod
    def _unwrap(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    @staticmethod
    def _two_cols(ts: torch.Tensor) -> torch.Tensor:
        """Ensure the last dimension has μ and σ (len==2). If σ missing → 1e‑4."""
        if ts.size(-1) == 1:
            ts = torch.cat([ts, ts.new_full(ts.shape[:-1] + (1,), 1e-4)], dim=-1)
        return ts

    def transform_output(self, out, target_scale=None):
        # unwrap the sometimes‑tuple inputs
        target_scale = self._unwrap(target_scale)
        if target_scale is None:
            return out

        # Normalise target_scale shape to (B, L, 2)
        if target_scale.dim() == 1:            # (2,) or (1,)
            target_scale = target_scale.unsqueeze(0).unsqueeze(0)     # (1,1,*)
        elif target_scale.dim() == 2:          # (B,2) or (B,1)
            target_scale = target_scale.unsqueeze(1)                  # (B,1,*)
        target_scale = self._two_cols(target_scale)                   # ensure μ,σ

        mu, sigma = target_scale[..., 0], target_scale[..., 1]

        # --- branch on output type ---------------------------------------
        if isinstance(out, dict):
            pred = out["prediction"]
            while mu.dim() < pred.dim():
                mu, sigma = mu.unsqueeze(-1), sigma.unsqueeze(-1)
            out["prediction"] = pred * sigma + mu
            # out["prediction"] = out["prediction"] * SCALE     # ← add
            return out
        else:  # raw tensor
            pred = out
            while mu.dim() < pred.dim():
                mu, sigma = mu.unsqueeze(-1), sigma.unsqueeze(-1)
            return pred * sigma + mu
            # return (pred * sigma + mu) * SCALE                # ← add

    def log_metrics(self, *_, **__):
        return {}

# ───────────────────────── training loop ─────────────────────────

def build_masks(chl: torch.Tensor, seq:int, lead:int, times: np.ndarray):
    P,T = chl.shape
    valid_core = torch.zeros_like(chl,dtype=torch.bool)
    core_slice = slice(seq-1, -lead)
    hist_ok = torch.isfinite(torch.stack([chl[:,i:i+seq] for i in range(T-seq+1)],-1)).all(1)
    if lead:
        hist_ok = hist_ok[:,:-lead]
    future_ok = torch.isfinite(chl[:,seq-1+lead:])
    valid_core[:,core_slice] = hist_ok & future_ok

    train,val,test=[valid_core.clone() for _ in range(3)]
    train[:, times>=np.datetime64("2016-01-01")]=False
    val[:, (times<np.datetime64("2016-01-01")) | (times>np.datetime64("2018-12-31"))]=False
    test[:, times<=np.datetime64("2018-12-31")]=False
    return train,val,test

def run(cfg):
    ds = xr.open_dataset(cfg.freeze, chunks={"time":-1}); ds.load()
    print(f"Dataset: {ds.sizes['time']} steps · {ds.sizes['lat']}×{ds.sizes['lon']} grid")
    df = melt_to_df(ds)
    print(f"DataFrame: {len(df):,} rows, {df['sid'].nunique():,} pixels")

    times = pd.to_datetime(ds.time.values)
    feat_tensor = make_tensor(df[ALL_VARS], (df.sid.nunique(), df.t.nunique(), len(ALL_VARS)))
    static_tensor = make_tensor(df.groupby("sid").first()[STATIC], (feat_tensor.shape[0], len(STATIC)))

    train_mask,val_mask,test_mask = build_masks(feat_tensor[...,0], cfg.seq, cfg.lead, times)

    nw = cfg.num_workers or min(8, os.cpu_count())
    dl_train = DataLoader(
        PixelTSD(train_mask, feat_tensor, static_tensor, cfg.seq, cfg.lead),
        cfg.batch,
        shuffle=True,
        num_workers=nw,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    dl_val = DataLoader(
        PixelTSD(val_mask, feat_tensor, static_tensor, cfg.seq, cfg.lead),
        cfg.batch,
        num_workers=nw,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    dl_test = DataLoader(
        PixelTSD(test_mask, feat_tensor, static_tensor, cfg.seq, cfg.lead),
        cfg.batch,
        num_workers=nw,
        collate_fn=collate_batch,
        pin_memory=True,
    )

    # ─────────────────────────── model ────────────────────────────
    tft = TFTBare(
        x_reals=X_REAL,
        x_categoricals=[dummy_cat],
        embedding_sizes={dummy_cat: (1, 1)},
        hidden_size=256,
        lstm_layers=2,
        attention_head_size=8,
        dropout=0.10,
        output_size=cfg.lead,
        loss=QuantileLoss([0.5]),
    )

    # ---------- optimiser & scheduler ----------
    optimizer  = torch.optim.Adam(tft.parameters(), lr=1e-3)
    scheduler = TorchPlateau(optimizer, factor=0.2, patience=2)

    tft.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
    }

    # ──────────────────── checkpoint & trainer ────────────────────
    ckpt = ModelCheckpoint(
        dirpath=cfg.outdir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
    )

    accelerator = cfg.device
    if accelerator == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA unavailable – switching to CPU")
        accelerator = "cpu"
    if accelerator == "mps" and not torch.backends.mps.is_available():
        warnings.warn("MPS unavailable – switching to CPU")
        accelerator = "cpu"

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        enable_model_summary=False,
        devices=1,
        callbacks=[
            ckpt,
            EarlyStopping("val_loss", patience=6),
            PrintLossCallback(),
            LearningRateMonitor(logging_interval="epoch"),   # ← change this line
        ],
        log_every_n_steps=50,
        num_sanity_val_steps=0,
    )

    # ───────────────────── train & evaluate ──────────────────────
    trainer.fit(tft, dl_train, dl_val)
    print("✓ training complete — best checkpoint:", ckpt.best_model_path)

    test_metrics = trainer.validate(tft, dl_test, verbose=False)[0]
    print("Test metrics (log-space):", {k: f"{v:.4f}" for k, v in test_metrics.items()})


# ───────────────────────────── CLI ─────────────────────────────
if __name__ == "__main__":
    print(">>> California HABs TFT trainer — v3.2 <<<")
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True, type=pathlib.Path)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seq", type=int, default=6)
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument(
        "--outdir",
        default=pathlib.Path.home() / "HAB_Models",
        type=pathlib.Path,
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="dataloader workers (default=min(8, nCPU))",
    )
    cfg = ap.parse_args()
    cfg.outdir.mkdir(exist_ok=True)

    run(cfg)
    