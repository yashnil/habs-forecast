#!/usr/bin/env python3
"""
Mini-cube generator: for each water pixel, extract a 4-frame 64×64 tile of
20 features, return also the true next-frame log_chl patch and water mask.

Inputs are zero-mean/unit-var standardized **over ocean only**.
All land pixels are set to 0 (and ignored during training via the mask).
"""
from __future__ import annotations
import pathlib, yaml, xarray as xr, numpy as np, torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Tuple

# ────────────── paths & constants ────────────────────────────────
root      = pathlib.Path(__file__).resolve().parents[1]
cfg       = yaml.safe_load(open(root / "config.yaml"))
cube_path = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021_varspec_nostripes.nc"

SEQ, PATCH  = 4, 64
LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0

# 20 predictor channels + 3 derivatives
core_vars  = [
    "sst","tp","avg_sdswrf","t2m","d2m","u10","v10",
    "Kd_490","nflh","so","thetao","uo","vo","zos",
    "sin_doy","cos_doy","log_chl"
]
deriv_vars = ["curl_uv","dist_river_km","log1p_dist_river"]
ALL_VARS   = core_vars + deriv_vars
CHANS      = len(ALL_VARS)
LOGCHL_IDX = ALL_VARS.index("log_chl")

# ─────────── load cube & mask ───────────────────────────────────
ds      = xr.open_dataset(cube_path)
cube_da = (
    ds
    .sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    .to_array("var")
    .sel(var=ALL_VARS)
)  # dims: (var, time, lat, lon)

# keep the mask as an xarray (for .where)…
wm_da = ds["water_mask"].sel(
    lat=slice(LAT_MIN, LAT_MAX),
    lon=slice(LON_MIN, LON_MAX),
)
# …and also pull out a numpy bool array for slicing patches
wm    = wm_da.values.astype(bool)

# ─────────── compute ocean-only μ/σ for each var ────────────────
stats: dict[str, tuple[float,float]] = {}
for v in ALL_VARS:
    da   = cube_da.sel(var=v)
    da_w = da.where(wm_da)            # NaNs on land
    μ    = float(da_w.mean(skipna=True).item())
    σ    = float(da_w.std (skipna=True).item())
    if σ <= 0: σ = 1.0
    stats[v] = (μ, σ)

# ─────────── Dataset / DataLoader ───────────────────────────────
class HABMiniCube(Dataset):
    def __init__(self, years: list[int], seq: int = SEQ, patch: int = PATCH):
        self.seq   = seq
        self.patch = patch

        time_sel = cube_da["time"].dt.year.isin(years)
        self.da  = cube_da.sel(time=time_sel).load()  # bring coords into RAM

        # only center locations where log_chl(t) is observed
        ok = ~self.da.sel(var="log_chl").isnull()

        self.valid: list[tuple[int,int,int]] = []
        for ti in range(seq, ok.sizes["time"]-1):  # need t+1 for label
            centres = ok.isel(
                time=ti,
                lat = slice(patch//2, None, patch),
                lon = slice(patch//2, None, patch),
            )
            ys, xs = np.where(centres.values)
            for y_off, x_off in zip(ys, xs):
                yi = y_off * patch
                xi = x_off * patch
                if yi+patch <= ok.sizes["lat"] and xi+patch <= ok.sizes["lon"]:
                    self.valid.append((ti, yi, xi))

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        ti, yi, xi = self.valid[idx]

        # 1) raw tile: (SEQ, VAR, PATCH, PATCH)
        tile = (
            self.da
              .isel(
                  time=slice(ti-self.seq, ti),
                  lat = slice(yi, yi+self.patch),
                  lon = slice(xi, xi+self.patch),
              )
              .transpose("time","var","lat","lon")
              .values
              .astype("float32")
        )

        # 2) standardize per-variable & zero out land
        for ci, v in enumerate(ALL_VARS):
            μ, σ = stats[v]
            tile[:,ci] = (tile[:,ci] - μ) / σ
        tile = np.nan_to_num(tile, nan=0.0)

        # 3) next-step true log_chl patch (raw units)
        y_patch = (
            self.da.sel(var="log_chl")
                   .isel(time=ti+1,
                         lat = slice(yi, yi+self.patch),
                         lon = slice(xi, xi+self.patch))
                   .values
                   .astype("float32")
        )

        # 4) corresponding water mask patch
        wm_patch = wm[yi:yi+self.patch, xi:xi+self.patch]

        return (
            torch.from_numpy(tile),        # (SEQ, CHANS, PATCH, PATCH)
            torch.from_numpy(y_patch),     # (PATCH, PATCH)
            torch.from_numpy(wm_patch).bool()  # (PATCH, PATCH)
        )

def make_loaders(batch:int=32, workers:int=4) -> Tuple[DataLoader,DataLoader,DataLoader]:
    pin = torch.cuda.is_available()

    # stratify train by log_chl at center
    train_ds = HABMiniCube([2016,2017,2018,2019])
    centres  = [train_ds[i][1][PATCH//2, PATCH//2].item() for i in range(len(train_ds))]
    edges    = np.percentile(centres, np.linspace(0,100,6))
    quint    = np.digitize(centres, edges[1:-1], right=True)
    counts   = np.bincount(quint, minlength=5)
    weights  = 1.0 / counts[quint]
    sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    tr = DataLoader(train_ds, batch_size=batch,
                    sampler=sampler, num_workers=workers,
                    pin_memory=pin, drop_last=True)
    va = DataLoader(HABMiniCube([2020]), batch_size=batch,
                    shuffle=False, num_workers=workers,
                    pin_memory=pin, drop_last=True)
    te = DataLoader(HABMiniCube([2021]), batch_size=batch,
                    shuffle=False, num_workers=workers,
                    pin_memory=pin, drop_last=True)
    return tr, va, te

# ────────── smoke-test ───────────────────────────────────────────
if __name__ == "__main__":
    tr, va, te = make_loaders(batch=2, workers=0)
    X, y, m = next(iter(tr))
    print(f"✓ loader OK – X{tuple(X.shape)}, y{tuple(y.shape)}, mask{tuple(m.shape)} (channels={CHANS})")

'''
✓ loader OK – X(2, 4, 20, 64, 64), y(2, 64, 64), mask(2, 64, 64) (channels=20)
'''