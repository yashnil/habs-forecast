#!/usr/bin/env python3
"""
Mini-cube generator for ConvLSTM (zero-mean, unit-var, NaN-free).

X  : (SEQ=4, C=18, 64, 64)
y  : scalar log10-chlor_a at t+1 (tile centre)

Channels
0-11 : core predictors (sst … zos)
12-13: sin_doy , cos_doy
14   : log_chl   (t-0)
15-17: curl_uv , dist_river_km , log1p_dist_river
"""
from __future__ import annotations
import pathlib, yaml, xarray as xr, numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List

# ───────────────────────── paths & constants ─────────────────────
root        = pathlib.Path(__file__).resolve().parents[1]
cfg         = yaml.safe_load(open(root / "config.yaml"))
cube_path   = pathlib.Path(cfg["data_root"]) / "HAB_cube_std0_2016_2021.nc"

SEQ, PATCH  = 4, 64
LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0

core_vars   = [
    "sst","t2m","u10","v10","avg_sdswrf",
    "Kd_490","nflh","so","thetao","uo","vo","zos",
    "sin_doy","cos_doy","log_chl"
]
deriv_vars  = ["curl_uv","dist_river_km","log1p_dist_river"]
ALL_VARS    = core_vars + deriv_vars          # expected 18
CHANS       = len(ALL_VARS)

# ───────────────────── 1. load cube lazily  ──────────────────────
root_ds = (
    xr.open_dataset(cube_path)
      .sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
)[core_vars]

try:
    deriv_ds = (
        xr.open_dataset(cube_path, group="derivatives")
          .sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    )[deriv_vars]
except (FileNotFoundError, KeyError):
    shp = (root_ds.sizes["time"],
           root_ds.sizes["lat"],
           root_ds.sizes["lon"])
    deriv_ds = xr.Dataset({
        v: (("time","lat","lon"), np.zeros(shp, np.float32))
        for v in deriv_vars
    }, coords=root_ds.coords)

cube = xr.merge([root_ds, deriv_ds]).to_array("var").sel(var=ALL_VARS)
# cube  shape → (var, time, lat, lon), already µ=0, σ=1, NaN-free

# ───────────────────── 2. Dataset / DataLoader  ──────────────────
class HABMiniCube(Dataset):
    def __init__(self, years: List[int], seq: int = SEQ, patch: int = PATCH):
        mask_time = cube["time"].dt.year.isin(years)
        self.da   = cube.sel(time=mask_time)
        self.seq  = seq
        self.patch= patch

        self.valid: list[Tuple[int,int,int]] = []
        for ti in range(seq, self.da.sizes["time"]):
            for yi in range(0, self.da.sizes["lat"] - patch, patch):
                for xi in range(0, self.da.sizes["lon"] - patch, patch):
                    self.valid.append((ti, yi, xi))

    def __len__(self) -> int:
        return len(self.valid)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ti, yi, xi = self.valid[idx]
        tile = (
            self.da.isel(time=slice(ti-self.seq, ti),
                         lat =slice(yi, yi+PATCH),
                         lon =slice(xi, xi+PATCH))
              .transpose("time","var","lat","lon")
              .values.astype("float32")
        )                               # (SEQ, C, H, W)

        yc, xc = yi + PATCH//2, xi + PATCH//2
        target = self.da.sel(var="log_chl").isel(time=ti, lat=yc, lon=xc
                   ).values.astype("float32")
        return torch.from_numpy(tile), torch.tensor(target)

def make_loaders(batch:int=32, workers:int=4):
    pin = torch.cuda.is_available()
    tr = DataLoader(HABMiniCube([2016,17,18,19]), batch_size=batch,
                    shuffle=True,  num_workers=workers, pin_memory=pin,
                    drop_last=True)
    va = DataLoader(HABMiniCube([2020]),          batch_size=batch,
                    shuffle=False, num_workers=workers, pin_memory=pin,
                    drop_last=True)
    te = DataLoader(HABMiniCube([2021]),          batch_size=batch,
                    shuffle=False, num_workers=workers, pin_memory=pin,
                    drop_last=True)
    return tr, va, te

# ───────────────────── 3. quick smoke-test  ──────────────────────
if __name__ == "__main__":
    tr, _, _ = make_loaders(batch=2, workers=0)
    X, y = next(iter(tr))
    print("✓ loader OK  ––  X", tuple(X.shape), "y", tuple(y.shape),
          f"(channels = {CHANS})")
