#!/usr/bin/env python3
"""
Quick sanity-check for a saved Temporal-Fusion-Transformer checkpoint.

It

1. loads the checkpoint onto CPU
2. rebuilds feature & static tensors exactly as in training
3. assembles one batch (256 samples) from PixelTSD
4. pushes the batch through the network and prints the output shape
"""

from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset
import xarray as xr

# ─────────────────────── user-specific paths ────────────────────────
CKPT_PATH = Path("/Users/yashnilmohanty/HAB_Models/epoch=1-step=7534.ckpt")
FREEZE    = Path(
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/"
    "HAB_convLSTM_core_v1_clean.nc"
)

# ─────────────────── import helpers from training ───────────────────
ROOT = Path(__file__).resolve().parents[1]          # …/habs-forecast/
sys.path.append(str(ROOT / "tft"))                  # so `import test` works

from test import (                                 # noqa:  E402
    melt_to_df,
    make_tensor,
    PixelTSD,
    TFTNoScale,           # ← the subclass you trained
    ALL_VARS,
    STATIC,
)

SEQ, LEAD = 6, 1      # must match training
BATCH     = 256

# ───────────── 1 ▸ load the trained model (CPU) ─────────────
model = (
    TFTNoScale
    .load_from_checkpoint(CKPT_PATH, map_location="cpu")
    .eval()
)

# ───────────── 2 ▸ rebuild tensors exactly like training ────
ds = xr.open_dataset(FREEZE, chunks={"time": -1}); ds.load()
df = melt_to_df(ds)

feat_tensor = make_tensor(
    df[ALL_VARS], (df.sid.nunique(), df.t.nunique(), len(ALL_VARS))
)
static_tensor = make_tensor(
    df.groupby("sid").first()[STATIC],
    (feat_tensor.shape[0], len(STATIC)),
)

# ───────────── 3 ▸ permissive core-mask, then one batch ─────
core_mask = torch.zeros_like(feat_tensor[..., 0], dtype=torch.bool)
core_mask[:, SEQ - 1 : -LEAD] = True          # any slot with full hist+future

dataset = PixelTSD(core_mask, feat_tensor, static_tensor, SEQ, LEAD)
subset  = Subset(dataset, range(BATCH))       # first 256 samples
loader  = DataLoader(subset, batch_size=BATCH, shuffle=False, num_workers=0)

batch_x, _ = next(iter(loader))               # dict in the expected format

# ───────────── 4 ▸ forward pass & shape check ───────────────
with torch.no_grad():
    out = model(batch_x)

print("Prediction tensor shape :", out.prediction.shape)  # expect (256, 1)