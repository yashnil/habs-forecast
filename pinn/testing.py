#!/usr/bin/env python3
"""

[baseline ConvLSTM model that we used]

ConvLSTM baseline **v0.4** – California coastal HAB forecast
-----------------------------------------------------------
Major upgrades relative to v0.3:
• **Longer history**: SEQ = 6 (48 d) to improve spring‑upwelling skill.
• **Stratified patch sampler** with steeper weights (freq^-1.5).
• **Class‑weighted Huber loss** – emphasises Bloom/Extreme pixels.
• **Bay / shallow‑water mask** (<10 m depth) excluded from loss.
The rest of the training loop and I/O remain identical so you can
reuse diagnostics with a one‑line variable‑list update.

run instructions:

python pinn/testing.py --epochs 40 --batch 16
"""
from __future__ import annotations
import math, random, json, pathlib, numpy as np, xarray as xr
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

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

def physics_residual(logCp, logC, u, v, κ, dx=4000.0, dt=8*86400.0):
    device, dtype = logCp.device, logCp.dtype
    LapK = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]], dtype=dtype, device=device)/(dx*dx)
    GxK  = torch.tensor([[[[-0.5,0,0.5]]]],             dtype=dtype, device=device)/dx
    GyK  = torch.tensor([[[[-0.5],[0],[0.5]]]],          dtype=dtype, device=device)/dx

    lap = lambda f: F.conv2d(f, LapK, padding=1)
    gx  = lambda f: F.conv2d(f, GxK, padding=(0,1))
    gy  = lambda f: F.conv2d(f, GyK, padding=(1,0))

    return (logCp - logC)/dt + u*gx(logCp) + v*gy(logCp) - κ*lap(logCp)

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
    ds = xr.open_dataset(FREEZE).load()

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
    return tr_dl, va_dl, te_dl, q, stats

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
        # trainable physics coeff
        self.kappa  = nn.Parameter(torch.tensor(25, dtype=torch.float32))
        self.l1, self.l2 = PxLSTM(24,48), PxLSTM(48,64)
        # self.skip       = nn.Conv2d(Cin, 1, 1)  # simple residual head
        self.dropout = nn.Dropout2d(0.1)
        self.head    = nn.Conv2d(64, 1, 1)
    def forward(self, x):                 # x: (B,L,C,H,W)
        h1 = h2 = None; last_in = None
        for t in range(x.size(1)):
            f = self.reduce(x[:,t]); o1,h1 = self.l1(f,h1); o2,h2 = self.l2(o1,h2); last_in = x[:,t]
        o = self.head(o2)
        o = self.dropout(o)
        return o.squeeze(1)

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

def train_one(dl, net, opt, qthr, stats, epoch, use_phys):
    net.train()
    for X, y, m in dl:
        X, y, m = [t.to(DEVICE) for t in (X, y, m)]

        with autocast(enabled=MIXED_PREC):
            # 1) supervised residual in log‐space
            last_log = X[:, -1, LOGCHL_IDX]                  # (B,H,W)
            delta    = net(X)                                # (B,H,W)
            tgt      = y - last_log
            sup_err  = (delta - tgt).masked_fill_(~m, 0)     # (B,H,W)

            # class‐weighted Huber
            #err = err.masked_fill_(~m, 0)
            
            err = sup_err

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

            sup_loss = (w * huber).sum() / w.sum()

            # 2) physics residual on normalized log‐chl
            # lift to channel for conv2d
            logC_  = last_log .unsqueeze(1)  # (B,1,H,W)
            logCp_ = (last_log + delta).unsqueeze(1)
            u_     = X[:, -1, ALL_VARS.index("uo")].unsqueeze(1)
            v_     = X[:, -1, ALL_VARS.index("vo")].unsqueeze(1)
            res = physics_residual(logCp_, logC_, u_, v_, net.kappa)
            # sum‐then‐divide by number of valid pixels → stronger PDE signal
            sq = (res[:,0][m]).pow(2)
            phys_loss = sq.sum() / (m.sum().clamp(min=1))

            # combine with a small weight (warm‐up for first 3 epochs)
            # after computing sup_loss, phys_loss …

            # 1) un-normalize the delta so phys_loss is in log-units  
            mu, sd = stats["log_chl"]                  # your stored mean/std
            # phys_loss currently on normalized logC; scale back:
            phys_loss_phys = phys_loss * (sd**2)       # because mse on z-scores * sd² → mse on raw

            # get λ so phys and sup are balanced, but allow larger weight
            λ_raw = sup_loss.detach() / (phys_loss_phys.detach() + 1e-8)
            λ_phys = λ_raw.clamp(min=0.1, max=1000.0)

            # ramp λ over first 10 PINN epochs (i.e. epochs 21–30)
            if use_phys:
                # by epoch 25 (5 epochs into stage-2) we’re at full weight
                ramp = min(max((epoch - 20) / 5,  0.0), 1.0)
                λ_phys = λ_phys * ramp
            else:
                λ_phys = 0.0

            # final loss: supervised + (if in stage 2) PINN term
            loss = sup_loss + λ_phys * phys_loss_phys



        SCALER.scale(loss).backward()
        SCALER.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.)
        SCALER.step(opt); SCALER.update(); opt.zero_grad()
        with torch.no_grad():
            net.kappa.clamp_(1e-2, 1e3)

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    tr_dl, va_dl, te_dl, qthr, stats = make_loaders()
    Cin = len([v for v in ALL_VARS if v in xr.open_dataset(FREEZE).data_vars])
    net=ConvLSTM(Cin).to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),3e-4,weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, 'min', patience=3, factor=.5)
    OUT_DIR.mkdir(exist_ok=True)

    best=1e9; bad=0
    for ep in range(1, EPOCHS+1):
        # two‐stage: first 20 epochs only supervised, then PINN
        use_phys = (ep > 20)
        train_one(tr_dl, net, opt, qthr, stats, ep, use_phys)
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

# ------------------------------------------------------------------ #
# Optional CLI overrides for quick smoke-tests
# ------------------------------------------------------------------ #
import argparse
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",    type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch",  type=int)
    args = ap.parse_args()

    if args.seq:    SEQ    = args.seq
    if args.epochs: EPOCHS = args.epochs
    if args.batch:  BATCH  = args.batch

    main()

'''
Vanilla + PINN Results (testing.py):

Version 1 
FINAL RMSE_log: {'train': 0.7634156236815086, 'val': 0.6924955483312915, 'test': 0.7961275815969414}

Version 2 (pushed to Github)
FINAL RMSE_log: {'train': 0.7634156029152483, 'val': 0.6924954863666604, 'test': 0.7961274946011418}

Version 3 (recent)
FINAL RMSE_log: {'train': 0.718884191753868, 'val': 0.6858681399130192, 'test': 0.7868559538957354}
'''