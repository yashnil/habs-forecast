# tft/pixel_ds.py
"""
Light-weight helpers for pixel-level TFT training / inference
-------------------------------------------------------------
• make_tensor  – long-df → (P, T, C) float32 tensor
• melt_to_df   – NetCDF cube → tidy long dataframe
• PixelTSD     – Dataset that returns the exact dict layout the
                 TemporalFusionTransformer learned on.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import Dataset, default_collate


# ------------ 1)  DataFrame → tensor --------------------------- #
def make_tensor(df_block: pd.DataFrame, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Replace NaN / ±∞ with 0, cast to float32 and reshape into a
    PyTorch tensor of shape *shape*.
    """
    return torch.from_numpy(
        df_block.fillna(0.0)
                .replace([np.inf, -np.inf], 0.0)
                .to_numpy(np.float32)
                .reshape(*shape)
    )


# ------------ 2)  NetCDF cube → long DF ------------------------ #
def melt_to_df(ds: xr.Dataset) -> pd.DataFrame:
    """
    Flatten (time, lat, lon) cube into a long pixel × time dataframe.

    Keeps only pixels with ≥20 % finite `log_chl` coverage.
    """
    SATELLITE = ["log_chl", "Kd_490", "nflh"]
    METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
    OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos",
                 "ssh_grad_mag","so","thetao"]
    DERIVED   = ["chl_anom_monthly",
                 "chl_roll24d_mean","chl_roll24d_std",
                 "chl_roll40d_mean","chl_roll40d_std"]
    STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]
    ALL_VARS  = SATELLITE + METEO + OCEAN + DERIVED + STATIC
    STATIC_SET = set(STATIC)

    pixel_ok = (np.isfinite(ds.log_chl).sum("time") / ds.sizes["time"]) >= 0.20
    ys, xs   = np.where(pixel_ok.values)
    time_idx = np.arange(ds.sizes["time"], dtype=np.int32)

    rows: List[pd.DataFrame] = []
    for sid, (iy, ix) in enumerate(zip(ys, xs)):
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


# ------------ 3)  Pixel-level Dataset -------------------------- #
class PixelTSD(Dataset):
    """
    Yields samples compatible with `TemporalFusionTransformer`:

        x_dict, target_tensor
    """

    def __init__(
        self,
        core_mask: torch.Tensor,
        feat: torch.Tensor,
        stat: torch.Tensor,
        seq: int,
        lead: int,
        x_reals: list        # <── new
    ):
        """
        core_mask : (P, T) bool  – locations where a seq-length history
                                   *and* a lead-ahead future value exist.
        feat      : (P, T, C) float tensor   – full feature cube
        stat      : (P, C_stat) float tensor – static predictors
        seq       : encoder length
        lead      : prediction horizon
        """
        self.core_index = core_mask.nonzero(as_tuple=False)   # (N, 2)
        self.feat       = feat
        self.stat       = stat
        self.seq        = seq
        self.lead       = lead
        self.x_reals    = x_reals      # <─ **add this line**

    def __len__(self):
        return len(self.core_index)

    def __getitem__(self, idx):
        pid, t_core = self.core_index[idx].tolist()

        # --- encoder / decoder continuous tensors ---------------------
        enc = self.feat[pid, t_core - self.seq + 1 : t_core + 1]    # (seq, C)
        dec = enc[-1:].repeat(self.lead, 1)                         # (lead, C)

        # --------------------------------------------------
        # ensure encoder/decoder have the full length the model expects
        n_reals = len(self.x_reals)          # <- we’ll pass this attribute when we build the dataset
        need_pad = n_reals - enc.shape[1]
        if need_pad > 0:                     # pad zeros on the right
            z_enc = enc.new_zeros(self.seq,  need_pad)
            z_dec = dec.new_zeros(self.lead, need_pad)
            enc = torch.cat([enc, z_enc], 1)
            dec = torch.cat([dec, z_dec], 1)
        # --------------------------------------------------

        # --- targets --------------------------------------------------
        tgt         = self.feat[pid, t_core + self.lead, 0].view(1, 1)
        hist_target = enc[:, 0].unsqueeze(-1)

        target_scale = torch.stack(
            [hist_target.mean(), hist_target.std().clamp_min(1e-4)]
        )

        x = dict(
            encoder_cont       = enc,
            decoder_cont       = dec,
            encoder_cat        = torch.zeros(self.seq, 1, dtype=torch.long),
            decoder_cat        = torch.zeros(self.lead, 1, dtype=torch.long),
            static_cat         = torch.zeros(1, dtype=torch.long),
            static_cont        = self.stat[pid],
            encoder_time_idx   = torch.arange(self.seq, dtype=torch.long),
            decoder_time_idx   = torch.arange(1, self.lead + 1, dtype=torch.long),
            encoder_lengths    = torch.tensor(self.seq),
            decoder_lengths    = torch.tensor(self.lead),
            groups             = torch.tensor([pid]),
            encoder_target     = hist_target,
            decoder_target     = tgt,
            target_scale       = target_scale,
            t_core             = torch.tensor(t_core),    # ← diagnostics needs this
        )
        return x, tgt


# convenience collate identical to default_collate
def collate_batch(batch):
    xs, ys = zip(*batch)
    return default_collate(xs), default_collate(ys)
