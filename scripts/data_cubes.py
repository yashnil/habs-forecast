#!/usr/bin/env python3
"""
Mini-cube generator for ConvLSTM (zero-mean, unit-var, NaN-aware).

X  : (SEQ=4, C=18, 64, 64)       – predictors, standardised
y  : scalar log10-chlor_a at t+1 – centred in tile
m  : (64, 64) float mask (1 = ocean, 0 = land/invalid)  ← NEW

Channels
0-11 : core predictors   (sst … zos)
12-13: seasonality       (sin_doy , cos_doy)
14    log_chl            (t-0)
15-17: derivatives       (curl_uv , dist_river_km , log1p_dist_river)
"""
from __future__ import annotations
import pathlib, yaml, xarray as xr, numpy as np, torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Tuple, List

# ────────────── paths & constants ────────────────────────────────
root        = pathlib.Path(__file__).resolve().parents[1]
cfg         = yaml.safe_load(open(root / "config.yaml"))
cube_path   = pathlib.Path(cfg["data_root"]) / "HAB_cube_std0_2016_2021.nc"

SEQ, PATCH  = 4, 64
LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0

core_vars  = [
    "sst","t2m","u10","v10","avg_sdswrf",
    "Kd_490","nflh","so","thetao","uo","vo","zos",
    "sin_doy","cos_doy","log_chl"
]
deriv_vars = ["curl_uv","dist_river_km","log1p_dist_river"]
ALL_VARS   = core_vars + deriv_vars
CHANS      = len(ALL_VARS)          # 18

# ───────── 1. load cube lazily (root + derivatives) ──────────────
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
    shp = (root_ds.sizes["time"], root_ds.sizes["lat"], root_ds.sizes["lon"])
    deriv_ds = xr.Dataset(
        {v: (("time","lat","lon"), np.zeros(shp, np.float32))
         for v in deriv_vars},
        coords=root_ds.coords
    )

cube_da = xr.merge([root_ds, deriv_ds]).to_array("var").sel(var=ALL_VARS)
# → dims (var, time, lat, lon)   already μ≈0, σ≈1

# ───────── 2. load water-mask (lat,lon)  ─────────────────────────
water_mask = xr.open_dataset(cube_path, group="masks")["water_mask"].values

# ───────── 3. Dataset / DataLoader  ──────────────────────────────
class HABMiniCube(Dataset):
    def __init__(self, years: list[int],
                 seq: int = SEQ, patch: int = PATCH):
        self.seq   = seq
        self.patch = patch

        # restrict to requested years
        time_sel   = cube_da["time"].dt.year.isin(years)
        self.da    = cube_da.sel(time=time_sel).load()   # bring coords into RAM

        # -------- pre-compute all valid (t,y,x) triplets ------------
        tgt = self.da.sel(var="log_chl")                 # (time,lat,lon)
        water = xr.open_dataset(cube_path, group="masks")["water_mask"]

        ok = (~tgt.isnull()) & water                    # finite & water

        self.valid: list[tuple[int,int,int]] = []
        for ti in range(seq, ok.sizes["time"]):
            # centre pixel indices
            ok_centres = ok.isel(time=ti, lat=slice(patch//2, None, patch),
                                  lon=slice(patch//2, None, patch))
            ys, xs = np.where(ok_centres.values)
            for y_off, x_off in zip(ys, xs):
                yi = y_off*patch
                xi = x_off*patch
                self.valid.append((ti, yi, xi))

    def __len__(self): return len(self.valid)

    def __getitem__(self, idx):
        ti, yi, xi = self.valid[idx]
        tile = ( self.da.isel(time=slice(ti-self.seq, ti),
                              lat=slice(yi, yi+PATCH),
                              lon=slice(xi, xi+PATCH))
                   .transpose("time","var","lat","lon")
                   .values.astype("float32") )
        tile = np.nan_to_num(tile, nan=0.0)

        yc, xc = yi + PATCH//2, xi + PATCH//2
        target = self.da.sel(var="log_chl").isel(time=ti, lat=yc, lon=xc
                     ).values.astype("float32")

        mask = water_mask[yi:yi+PATCH, xi:xi+PATCH].astype("float32")
        return torch.from_numpy(tile), torch.tensor(target), torch.from_numpy(mask)

def make_loaders(batch:int=32, workers:int=4):
    pin = torch.cuda.is_available()

    # --- build train sampler stratified by log_chl quintile ---
    train_ds = HABMiniCube([2016,17,18,19])
    # grab every target (log_chl at center)
    all_targets = np.array([train_ds[i][1].item()
                             for i in range(len(train_ds))])
    # quintile edges
    edges = np.percentile(all_targets, np.linspace(0, 100, 6))
    # bin into 0–4
    quint = np.digitize(all_targets, edges[1:-1], right=True)
    # inverse‐frequency weights
    counts = np.bincount(quint, minlength=5)
    weights = 1.0 / counts[quint]
    sampler = WeightedRandomSampler(weights,
                                    num_samples=len(weights),
                                    replacement=True)

    tr = DataLoader(train_ds, batch_size=batch,
                    sampler=sampler,
                    num_workers=workers, pin_memory=pin,
                    drop_last=True)

    va = DataLoader(HABMiniCube([2020]),          batch_size=batch,
                    shuffle=False, num_workers=workers,
                    pin_memory=pin, drop_last=True)
    te = DataLoader(HABMiniCube([2021]),          batch_size=batch,
                    shuffle=False, num_workers=workers,
                    pin_memory=pin, drop_last=True)
    return tr, va, te

# ───────── 4. smoke-test  ────────────────────────────────────────
if __name__ == "__main__":
    tr, _, _ = make_loaders(batch=2, workers=0)
    X, y, m = next(iter(tr))
    print("✓ loader OK  – X", tuple(X.shape),
          "y", tuple(y.shape), "mask", tuple(m.shape),
          f"(channels = {CHANS})")
