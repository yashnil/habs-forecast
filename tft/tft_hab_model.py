# tft_hab_model.py ---------------------------------------------------

'''
python tft/tft_hab_model.py \
  --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --epochs 15 \
  --seq 24 \
  --lead 7 \
  --device cpu
'''
import argparse, warnings
import pandas as pd, numpy as np, xarray as xr, torch
import lightning as L
from torch.utils.data import DataLoader
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import pickle, pathlib
import random

# ---------- 1. cube  -> tidy dataframe ---------- #
def cube_to_long(nc_path: pathlib.Path) -> pd.DataFrame:
    ds = xr.open_dataset(nc_path)
    ds = ds.copy()  # don’t overwrite the on-disk file

    # spatially fill missing cur_div and cur_vort from nearest neighbor
    for var in ["cur_div", "cur_vort", "ssh_grad_mag"]:
        ds[var] = (
            ds[var]
            .ffill(dim="lat")   # fill down along latitude
            .bfill(dim="lat")   # then back up
            .ffill(dim="lon")   # and same along longitude
            .bfill(dim="lon")
        )

    # now mask & flatten exactly as before
    keep = (np.isfinite(ds.log_chl).sum("time") / ds.sizes["time"]) >= .2
    ys, xs = np.where(keep)
    rows = []
    for sid, (iy, ix) in enumerate(zip(ys, xs)):
        tmp = ds.isel(lat=iy, lon=ix).to_dataframe().reset_index()
        tmp["sid"] = sid
        rows.append(tmp)
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=["log_chl"])
    df["sid"] = df["sid"].astype(str)
    return df

# ---------- 2. make TimeSeriesDataSet ---------- #
def make_tsd(df: pd.DataFrame, seq: int, lead: int, outdir: pathlib.Path):
    """
    Build train / validation TimeSeriesDataSet objects for the TFT.

    Parameters
    ----------
    df   : tidy dataframe returned by `cube_to_long`
    seq  : encoder length  (max history length, in days)
    lead : prediction length (forecast horizon, in days)

    Returns
    -------
    train_ds : TimeSeriesDataSet for training
    val_ds   : TimeSeriesDataSet for validation
    """
    # 1) housekeeping -------------------------------------------------
    df = df.copy()
    df["sid"] = df["sid"].astype(str)                       # categorical!
    df["time_idx"] = (df["time"] - df["time"].min()).dt.days

    # 2) calendar‐based split exactly like ConvLSTM:
    #    train = dates < 2016-01-01
    #    val   = 2016-01-01 … 2018-12-31
    #    test  = dates > 2018-12-31
    split_train_end = pd.Timestamp("2016-01-01")
    split_val_end   = pd.Timestamp("2019-01-01")  # one day past Dec 31, 2018

    # convert dates → indices
    df_min = df["time"].min()
    train_cutoff = (pd.Timestamp("2016-01-01") - df_min).days
    val_cutoff   = (pd.Timestamp("2019-01-01") - df_min).days

    # df.time_idx <  train_cutoff     # training  
    # (df.time_idx >= train_cutoff) & (df.time_idx < val_cutoff)  # validation

    # now build your TFT datasets
    unknown_reals = [c for c in df.columns
                 if c not in {"sid","time","time_idx","log_chl"}]
    
    train_ds = TimeSeriesDataSet(
        df[df.time_idx <  train_cutoff],
        time_idx                   = "time_idx",
        group_ids                  = ["sid"],
        target                     = "log_chl",
        max_encoder_length         = seq,
        max_prediction_length      = lead,
        static_categoricals        = ["sid"],
        static_reals        = ["river_rank","dist_river_km","ocean_mask_static"],
        time_varying_unknown_reals = [
            v for v in unknown_reals
            if v not in {"river_rank","dist_river_km","ocean_mask_static"}
        ],
        time_varying_known_reals=["time_idx"],
        target_normalizer          = GroupNormalizer(
                                        groups=["sid"], transformation=None
                                    ),
        add_relative_time_idx      = True,
        add_target_scales          = True,
        add_encoder_length         = True,
        allow_missing_timesteps    = True,
    )

    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds,
        df[(df.time_idx >= train_cutoff) & (df.time_idx < val_cutoff)],
        stop_randomization=True
    )

    dest = outdir / "lightning_logs" / "version_1"
    dest.mkdir(parents=True, exist_ok=True)
    fname = dest / "train_ds.pkl"
    with open(fname, "wb") as f:
        pickle.dump(train_ds, f)
    print("✅ Saved train_ds.pkl →", fname)
    # print("✅ Saved train_ds.pkl →", outdir/"train_ds.pkl")
    return train_ds, val_ds

# ---------- 3. train ---------- #
def run(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    df              = cube_to_long(cfg.freeze)
    train, val = make_tsd(df, cfg.seq, cfg.lead, cfg.outdir)
    batch_size      = min(256, 8*torch.cuda.device_count()+8)
    # --- build loaders -------------------------------------------------
    train_loader = train.to_dataloader(
        train=True,  batch_size=batch_size, num_workers=4
    )
    val_loader   = val.to_dataloader(
        train=False, batch_size=batch_size, num_workers=4
    )

    # quick sanity check – iterate one batch to prove it works
    x, y = next(iter(train_loader))
    y_data, y_weight = y            # y_weight == None in this case

    print("✅ first batch fetched – keys:", x.keys(),
        "| target shape:", y_data.shape,
        "| weight:", "None" if y_weight is None else y_weight.shape)

    tft = TemporalFusionTransformer.from_dataset(
        train,
        learning_rate = 2e-3,
        hidden_size   = 64,
        attention_head_size = 4,
        dropout = 0.1,
        loss    = QuantileLoss([0.1, 0.5, 0.9]),
    )

    trainer = L.Trainer(
        max_epochs          = cfg.epochs,
        accelerator         = cfg.device,
        devices             = 1,
        gradient_clip_val   = 1.0,
        default_root_dir    = cfg.outdir,
        log_every_n_steps   = 50,
    )
    trainer.fit(tft, train_loader, val_loader)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", type=pathlib.Path, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seq",   type=int, default=24)   # 24 days encoder
    ap.add_argument("--lead",  type=int, default=7)    # 7-day forecast
    ap.add_argument("--device", default="cpu", choices=["cpu","mps","cuda"])
    ap.add_argument("--outdir", type=pathlib.Path, default="runs/tft")
    ap.add_argument("--seed", type=int, default=42)
    cfg = ap.parse_args()
    run(cfg)

'''
Sanity Check in Terminal:

# quick, metrics-only smoke test (no heavy Cartopy plots)
python tft/diagnostics.py \
  --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --ckpt   runs/tft/lightning_logs/version_1/checkpoints/epoch=14-step=48780.ckpt \
  --out    Diagnostics_TFT_v0p1 \
  --seq    24 \
  --lead   7 \
  --batch  4096 \
  --device mps \
  --no_plots

'''