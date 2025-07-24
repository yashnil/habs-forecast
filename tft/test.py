#!/usr/bin/env python3
"""
Minimal, Apple‑silicon‑safe Temporal‑Fusion‑Transformer trainer.
History = 6 timesteps (48 d); Lead = 1 (8 d).
Run:
  python tft/test.py --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
                               --epochs 60 --batch 256 --seq 6 --lead 1 --device cpu
"""

import argparse, pathlib
from typing import Tuple

import numpy as np, pandas as pd, xarray as xr, torch, lightning as L
from torch.utils.data import DataLoader
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

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
    def __init__(self, df: pd.DataFrame, core_ts: np.ndarray, seq: int, lead: int):
        self.df, self.core_ts, self.seq, self.lead = df, core_ts, seq, lead
        self.idx = df[df.t.isin(core_ts)].reset_index(drop=True)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        row  = self.idx.iloc[i]
        sid, t_core = int(row.sid), int(row.t)

        # ---------- encoder & decoder ----------
        enc = torch.tensor([
            self.df[(self.df.sid == sid) & (self.df.t == t_core + dt)].iloc[0][ALL_VARS].values
            for dt in range(-self.seq + 1, 1)
        ], dtype=torch.float32)
        dec = enc[-1:].repeat(self.lead, 1)

        # ---------- static & target ----------
        static_cont = torch.tensor(row[STATIC].values, dtype=torch.float32)
        tgt_val = self.df[(self.df.sid == sid) & (self.df.t == t_core + self.lead)].log_chl.values[0]
        tgt = torch.tensor([tgt_val], dtype=torch.float32)            # shape (lead,)

        x = dict(
            encoder_cont     = enc,
            decoder_cont     = dec,
            encoder_cat      = torch.zeros(self.seq, 1, dtype=torch.long),
            decoder_cat      = torch.zeros(self.lead, 1, dtype=torch.long),
            static_cat       = torch.zeros(1, dtype=torch.long),
            encoder_time_idx = torch.arange(self.seq),
            decoder_time_idx = torch.arange(self.lead),
            encoder_lengths  = torch.tensor(self.seq,  dtype=torch.long),
            decoder_lengths  = torch.tensor(self.lead, dtype=torch.long),
            static_cont      = static_cont,
            groups           = torch.tensor([sid]),
            decoder_target   = tgt,              # ★ required by Pytorch‑Forecasting internals
            target_scale     = torch.tensor([1.0, 0.0]),  # placeholder, unused
        )
        return x, tgt

# ────────────────── tiny patch: TFT without rescaling ─────────────────
class TFTNoScale(TemporalFusionTransformer):
    def transform_output(self, out, target_scale=None):  # override one line → skip scale logic
        return out

# ───────────────────────────── training ──────────────────────────────
def run(cfg):
    # ---------- data ----------
    ds = xr.open_dataset(cfg.freeze, chunks={"time": -1}); ds.load()
    df = melt_to_df(ds)

    times = pd.to_datetime(ds.time.values)
    train_ts = np.where(times <  np.datetime64("2016-01-01"))[0]
    val_ts   = np.where((times>=np.datetime64("2016-01-01"))&(times<=np.datetime64("2018-12-31")))[0]
    test_ts  = np.where(times >  np.datetime64("2018-12-31"))[0]

    train = PixelTSD(df, train_ts[train_ts>=cfg.seq-1], cfg.seq, cfg.lead)
    val   = PixelTSD(df, val_ts  [val_ts  >=cfg.seq-1], cfg.seq, cfg.lead)
    test  = PixelTSD(df, test_ts [test_ts >=cfg.seq-1], cfg.seq, cfg.lead)

    dl_train = DataLoader(train, cfg.batch, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(val,   cfg.batch,                num_workers=0)
    dl_test  = DataLoader(test,  cfg.batch,                num_workers=0)

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
        logging_metrics    = [RMSE()],
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
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch",  type=int, default=256)
    p.add_argument("--seq",    type=int, default=6)
    p.add_argument("--lead",   type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu","mps","cuda"])
    p.add_argument("--outdir", default=pathlib.Path.home()/"HAB_Models", type=pathlib.Path)
    cfg = p.parse_args(); cfg.outdir.mkdir(exist_ok=True)

    run(cfg)
