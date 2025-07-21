#!/usr/bin/env python3
"""
ConvLSTM baseline **v0.4** – California coastal HAB forecast
-----------------------------------------------------------
Major upgrades relative to v0.3:
• **Longer history**: SEQ = 6 (48 d) to improve spring‑upwelling skill.
• **Stratified patch sampler** with steeper weights (freq^-1.5).
• **Class‑weighted Huber loss** – emphasises Bloom/Extreme pixels.
• **Bay / shallow‑water mask** (<10 m depth) excluded from loss.
The rest of the training loop and I/O remain identical so you can
reuse diagnostics with a one‑line variable‑list update.
"""
from __future__ import annotations
import math, random, json, pathlib, numpy as np, xarray as xr
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #
# FREEZE   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")

FREEZE   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
SEQ       = 6        # ← 48 day history (was 4)
LEAD_IDX  = 1        # forecast +8 d
PATCH     = 64
BATCH     = 32       # fits on 12 GB GPU for 30 channels × SEQ 6
EPOCHS    = 40       # extra epochs; early‑stop still active
SEED      = 42
STRATIFY  = True
WEIGHT_EXP= 1.5      # strat weight exponent (freq^-exp)
HUBER_DELTA= 1.0     # Huber delta in log‑space
MIXED_PREC= torch.cuda.is_available()
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
SCALER    = GradScaler(enabled=MIXED_PREC)
OUT_DIR = pathlib.Path.home() / "HAB_Models"       # ~/HAB_Models
OUT_DIR.mkdir(parents=True, exist_ok=True)
_FLOOR = 0.056616   # detection floor used when log_chl was created

# ------------------------------------------------------------------ #
# Predictor lists  (identical to freezer)
# ------------------------------------------------------------------ #
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos","ssh_grad_mag","so","thetao"]
DERIVED   = ["chl_anom_monthly",
             "chl_roll24d_mean","chl_roll24d_std",
             "chl_roll40d_mean","chl_roll40d_std"]
STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]
ALL_VARS  = SATELLITE + METEO + OCEAN + DERIVED + STATIC
LOGCHL_IDX= SATELLITE.index("log_chl")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ------------------------------------------------------------------ #
# Normalisation helpers
# ------------------------------------------------------------------ #
def norm_stats(ds, train_idx):
    out = {}
    for v in ALL_VARS:
        if v not in ds: continue
        da = ds[v] if "time" not in ds[v].dims else ds[v].isel(time=train_idx)
        mu, sd = float(da.mean(skipna=True)), float(da.std(skipna=True)) or 1.0
        out[v] = (mu, sd)
    return out

def z(arr, mu_sd):
    mu, sd = mu_sd; return (arr - mu) / sd

# ------------------------------------------------------------------ #
# Dataset
# ------------------------------------------------------------------ #
class PatchDS(Dataset):
    """Random coastal 64×64 crops with on‑the‑fly standardisation."""
    def __init__(self, ds, tids, stats, mask):
        self.ds, self.tids, self.stats, self.mask = ds, tids, stats, mask
        self.latL = np.arange(0, ds.sizes["lat"] - PATCH + 1)
        self.lonL = np.arange(0, ds.sizes["lon"] - PATCH + 1)
        self.rng  = np.random.default_rng(SEED + len(tids))

    def __len__(self): return len(self.tids)

    def _corner(self):
        for _ in range(20):
            y0 = int(self.rng.choice(self.latL)); x0 = int(self.rng.choice(self.lonL))
            if self.mask.isel(lat=slice(y0,y0+PATCH), lon=slice(x0,x0+PATCH)).any():
                return y0, x0
        return 0, 0

    def __getitem__(self, k):
        t = int(self.tids[k]); y0, x0 = self._corner()
        frames = []
        for dt in range(SEQ):
            bands = []
            for v in ALL_VARS:
                if v not in self.ds: continue
                da = self.ds[v].isel(time=t-SEQ+1+dt, lat=slice(y0,y0+PATCH), lon=slice(x0,x0+PATCH)) if "time" in self.ds[v].dims else \
                     self.ds[v].isel(lat=slice(y0,y0+PATCH), lon=slice(x0,x0+PATCH))
                bands.append(np.nan_to_num(z(da.values, self.stats[v]), nan=0.0))
            frames.append(np.stack(bands, 0))
        X = torch.from_numpy(np.stack(frames, 0).astype(np.float32))

        tgt = self.ds["log_chl"].isel(time=t+LEAD_IDX, lat=slice(y0,y0+PATCH), lon=slice(x0,x0+PATCH)).values
        valid = self.mask.isel(lat=slice(y0,y0+PATCH), lon=slice(x0,x0+PATCH)).values & np.isfinite(tgt)
        return X, torch.from_numpy(tgt.astype(np.float32)), torch.from_numpy(valid)

# ------------------------------------------------------------------ #
# Data loaders
# ------------------------------------------------------------------ #
DEF_MASK_DEPTH = 10.0  # metres; mask bays & very shallow cells

def make_loaders():
    ds = xr.open_dataset(FREEZE, chunks={"time": 1})

    # ➡️ NEW: synthesise linear-space chl if missing
    if "chl_lin" not in ds:
        ds["chl_lin"] = np.exp(ds["log_chl"]) - _FLOOR
        # keep attrs tidy if you like
        ds["chl_lin"].attrs.update({"units": "mg m-3", "long_name": "chlorophyll-a"})

    # build pixel_ok if missing
    if "pixel_ok" not in ds:
        frac = np.isfinite(ds.log_chl).sum("time") / ds.sizes["time"]
        ds["pixel_ok"] = (frac >= .2).astype("uint8")

    pixel_ok = ds.pixel_ok.astype(bool)

    # mask bays / shallow water (if variable present)
    if "depth" in ds:
        pixel_ok = pixel_ok & (ds.depth > DEF_MASK_DEPTH)

    t_all = np.arange(ds.sizes["time"])
    times = ds.time.values
    tr = t_all[times <  np.datetime64("2016-01-01")]
    va = t_all[(times >= np.datetime64("2016-01-01")) & (times <= np.datetime64("2018-12-31"))]
    te = t_all[times >  np.datetime64("2018-12-31")]

    stats = norm_stats(ds, tr)
    tr_i, va_i, te_i = tr[SEQ-1:-LEAD_IDX], va[SEQ-1:-LEAD_IDX], te[SEQ-1:-LEAD_IDX]
    tr_ds, va_ds, te_ds = (PatchDS(ds, idx, stats, pixel_ok) for idx in (tr_i, va_i, te_i))

    # stratified sampler (steeper exponent)
    sampler = None
    if STRATIFY:
        chl = (np.exp(ds.log_chl) - _FLOOR).isel(time=tr_i).where(pixel_ok).median(("lat", "lon")).values
        q = np.nanquantile(chl, [.25, .5, .75]); bin_ = np.digitize(chl, q)
        freq = np.maximum(np.bincount(bin_, minlength=4), 1)
        w = (1 / freq) ** WEIGHT_EXP
        sampler = WeightedRandomSampler(torch.as_tensor(w[bin_]), len(bin_), replacement=True)

    tr_dl = DataLoader(tr_ds, BATCH, sampler=sampler, shuffle=sampler is None, num_workers=4, pin_memory=True, drop_last=True)
    va_dl = DataLoader(va_ds, BATCH, shuffle=False, num_workers=2, pin_memory=True)
    te_dl = DataLoader(te_ds, BATCH, shuffle=False, num_workers=2, pin_memory=True)
    return tr_dl, va_dl, te_dl, q

# ------------------------------------------------------------------ #
# Model (unchanged)
# ------------------------------------------------------------------ #
class PxLSTM(nn.Module):
    def __init__(self, ci, co):
        super().__init__(); self.cell = nn.LSTMCell(ci, co); self.conv = nn.Conv2d(co, co, 1)
    def forward(self, x, hc=None):
        B, C, H, W = x.shape; flat = x.permute(0,2,3,1).reshape(B*H*W, C)
        if hc is None:
            h = torch.zeros(flat.size(0), self.cell.hidden_size, dtype=flat.dtype, device=flat.device)
            hc = (h, torch.zeros_like(h))
        h, c = self.cell(flat, hc); h_map = h.view(B,H,W,-1).permute(0,3,1,2)
        return self.conv(h_map), (h, c)

class ConvLSTM(nn.Module):
    def __init__(self, Cin):
        super().__init__()
        self.reduce = nn.Conv2d(Cin, 24, 1)
        self.l1, self.l2 = PxLSTM(24,48), PxLSTM(48,64)
        self.skip       = nn.Conv2d(Cin, 1, 1)  # simple residual head
        self.head       = nn.Conv2d(64, 1, 1)
    def forward(self, x):                 # x: (B,L,C,H,W)
        h1 = h2 = None; last_in = None
        for t in range(x.size(1)):
            f = self.reduce(x[:,t]); o1,h1 = self.l1(f,h1); o2,h2 = self.l2(o1,h2); last_in = x[:,t]
        return self.head(o2).squeeze(1) + self.skip(last_in).squeeze(1)

# ------------------------------------------------------------------ #
# Train / eval helpers
# ------------------------------------------------------------------ #
@torch.no_grad()
def rmse(dl, net):
    net.eval(); se = n = 0
    for X, y, m in dl:
        X, y, m = [t.to(DEVICE) for t in (X, y, m)]
        with autocast(enabled=MIXED_PREC):
            p = X[:,-1,LOGCHL_IDX] + net(X)
        err = (p - y).masked_fill_(~m, 0)
        se += (err**2).sum().item(); n += m.sum().item()
    return math.sqrt(se / n)

def train_one(dl, net, opt, qthr):
    net.train()
    for X, y, m in dl:
        X, y, m = [t.to(DEVICE) for t in (X, y, m)]
        with autocast(enabled=MIXED_PREC):
            tgt = y - X[:,-1,LOGCHL_IDX]
            err = net(X) - tgt
            err = err.masked_fill_(~m, 0)
            # class weight based on linear chl quantiles (precomputed)
            chl_lin = torch.exp(y)
            w = torch.where(chl_lin < qthr[0], 1.0,
                    torch.where(chl_lin < qthr[1], 1.5,
                        torch.where(chl_lin < qthr[2], 2.5, 4.0)))
            w = w * m
            # Huber loss
            abs_err = err.abs(); quadratic = torch.minimum(abs_err, torch.tensor(HUBER_DELTA, device=abs_err.device))
            linear   = abs_err - quadratic
            huber    = 0.5 * quadratic**2 + HUBER_DELTA * linear
            loss = (w * huber).sum() / w.sum()
        SCALER.scale(loss).backward()
        SCALER.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.)
        SCALER.step(opt); SCALER.update(); opt.zero_grad()

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    tr_dl, va_dl, te_dl, qthr = make_loaders()
    Cin = len([v for v in ALL_VARS if v in xr.open_dataset(FREEZE).data_vars])
    net=ConvLSTM(Cin).to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),3e-4,weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, 'min', patience=3, factor=.5)
    OUT_DIR.mkdir(exist_ok=True)

    best=1e9; bad=0
    for ep in range(1,EPOCHS+1):
        train_one(tr_dl,net,opt,qthr)
        val_rm=rmse(va_dl,net); sched.step(val_rm)
        print(f"E{ep:02d}  val RMSE_log={val_rm:.3f}  lr={opt.param_groups[0]['lr']:.1e}")
        if val_rm < best-1e-3:
            best=val_rm; bad=0; torch.save(net.state_dict(),OUT_DIR/'convLSTM_best.pt')
        else:
            bad+=1
            if bad==6: break

    net.load_state_dict(torch.load(OUT_DIR/'convLSTM_best.pt'))
    metrics = {"train":rmse(tr_dl,net),"val":rmse(va_dl,net),"test":rmse(te_dl,net)}
    print("\nFINAL RMSE_log:",metrics)
    json.dump(metrics, open(OUT_DIR/'convLSTM_metrics.json','w'), indent=2)

if __name__=="__main__":
    main()

'''
E01  val RMSE_log=0.840  lr=3.0e-04
E02  val RMSE_log=0.834  lr=3.0e-04
E03  val RMSE_log=0.826  lr=3.0e-04
E04  val RMSE_log=0.820  lr=3.0e-04
E05  val RMSE_log=0.817  lr=3.0e-04
E06  val RMSE_log=0.807  lr=3.0e-04
E07  val RMSE_log=0.796  lr=3.0e-04
E08  val RMSE_log=0.776  lr=3.0e-04
E09  val RMSE_log=0.761  lr=3.0e-04
E10  val RMSE_log=0.760  lr=3.0e-04
E11  val RMSE_log=0.750  lr=3.0e-04
E12  val RMSE_log=0.748  lr=3.0e-04
E13  val RMSE_log=0.752  lr=3.0e-04
E14  val RMSE_log=0.743  lr=3.0e-04
E15  val RMSE_log=0.744  lr=3.0e-04
E16  val RMSE_log=0.743  lr=3.0e-04
E17  val RMSE_log=0.751  lr=3.0e-04
Epoch 00018: reducing learning rate of group 0 to 1.5000e-04.
E18  val RMSE_log=0.744  lr=1.5e-04
E19  val RMSE_log=0.739  lr=1.5e-04
E20  val RMSE_log=0.740  lr=1.5e-04
E21  val RMSE_log=0.737  lr=1.5e-04
E22  val RMSE_log=0.736  lr=1.5e-04
E23  val RMSE_log=0.737  lr=1.5e-04
E24  val RMSE_log=0.737  lr=1.5e-04
E25  val RMSE_log=0.735  lr=1.5e-04
E26  val RMSE_log=0.734  lr=1.5e-04
E27  val RMSE_log=0.735  lr=1.5e-04
E28  val RMSE_log=0.736  lr=1.5e-04
E29  val RMSE_log=0.738  lr=1.5e-04
Epoch 00030: reducing learning rate of group 0 to 7.5000e-05.
E30  val RMSE_log=0.734  lr=7.5e-05

FINAL RMSE_log: {'train': 0.7637301112146647, 'val': 0.7341518534766268, 'test': 0.7663904919928541}

'''