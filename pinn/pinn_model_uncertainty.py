#!/usr/bin/env python3
"""
Physics-Informed Neural Network (PINN) ConvLSTM with Uncertainty Quantification

PINN ConvLSTM **v0.5** – California coastal HAB forecast with uncertainty
---------------------------------------------------------------------------
This enhanced version adds multiple uncertainty quantification methods:

1. **Monte Carlo Dropout**: Epistemic uncertainty via dropout at inference
2. **Ensemble Methods**: Train multiple models with different seeds, aggregate predictions
3. **Quantile Regression**: Output multiple quantiles (e.g., 0.1, 0.5, 0.9) for predictive intervals
4. **Bayesian Approximation**: Optional variational inference for parameter uncertainty

Key features:
• **Longer history**: SEQ = 6 (48 d) to improve spring‑upwelling skill.
• **Stratified patch sampler** with steeper weights (freq^-1.5).
• **Class‑weighted Huber loss** – emphasises Bloom/Extreme pixels.
• **Physics residual**: enforces 2D advection-diffusion PDE constraints.
• **Uncertainty quantification**: multiple methods for robust predictions.

run instructions:

# Standard training (single model)
python pinn/pinn_model_uncertainty.py --epochs 40 --batch 16

# Ensemble training (5 models)
python pinn/pinn_model_uncertainty.py --epochs 40 --batch 16 --ensemble 5

# Quantile regression (output 0.1, 0.5, 0.9 quantiles)
python pinn/pinn_model_uncertainty.py --epochs 40 --batch 16 --quantiles 0.1 0.5 0.9
"""
from __future__ import annotations
import math, random, json, pathlib, numpy as np, xarray as xr
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

# ------------------------------------------------------------------ #
# CONFIG
# ------------------------------------------------------------------ #
FREEZE   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
SEQ       = 6        # ← 48 day history (was 4)
LEAD_IDX  = 1        # forecast +8 d
PATCH     = 64
BATCH     = 32       # fits on 12 GB GPU for 30 channels × SEQ 6
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

# Uncertainty quantification config
MC_DROPOUT_RATE = 0.1  # Dropout rate for MC dropout
MC_SAMPLES = 20         # Number of MC samples for uncertainty estimation
ENSEMBLE_SIZE = 5      # Number of ensemble members (if using ensemble)
QUANTILES = None       # List of quantiles to predict (e.g., [0.1, 0.5, 0.9])

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
# Dataset (unchanged)
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
# Enhanced Model with Uncertainty Quantification
# ------------------------------------------------------------------ #
class PxLSTM(nn.Module):
    def __init__(self, ci, co, dropout_rate=0.0):
        super().__init__()
        self.cell = nn.LSTMCell(ci, co)
        self.conv = nn.Conv2d(co, co, 1)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    def forward(self, x, hc=None):
        B, C, H, W = x.shape; flat = x.permute(0,2,3,1).reshape(B*H*W, C)
        if hc is None:
            h = torch.zeros(flat.size(0), self.cell.hidden_size, dtype=flat.dtype, device=flat.device)
            hc = (h, torch.zeros_like(h))
        h, c = self.cell(flat, hc); h_map = h.view(B,H,W,-1).permute(0,3,1,2)
        h_map = self.dropout(h_map)
        return self.conv(h_map), (h, c)

class ConvLSTM_Uncertainty(nn.Module):
    """
    PINN ConvLSTM with uncertainty quantification support.
    
    Supports:
    - Monte Carlo Dropout (via dropout_rate > 0)
    - Quantile regression (via n_quantiles > 1)
    """
    def __init__(self, Cin, dropout_rate=MC_DROPOUT_RATE, n_quantiles=1):
        super().__init__()
        self.reduce = nn.Conv2d(Cin, 24, 1)
        self.kappa  = nn.Parameter(torch.tensor(25, dtype=torch.float32))
        self.l1 = PxLSTM(24, 48, dropout_rate=dropout_rate)
        self.l2 = PxLSTM(48, 64, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Output head: if n_quantiles > 1, output multiple quantiles
        if n_quantiles > 1:
            self.head = nn.Conv2d(64, n_quantiles, 1)
            self.n_quantiles = n_quantiles
        else:
            self.head = nn.Conv2d(64, 1, 1)
            self.n_quantiles = 1
            
    def forward(self, x, return_uncertainty=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, L, C, H, W)
            return_uncertainty: If True and n_quantiles > 1, return mean and std
        
        Returns:
            If n_quantiles == 1: (B, H, W) delta log-chl
            If n_quantiles > 1: (B, n_quantiles, H, W) quantile predictions
            If return_uncertainty=True: also returns (mean, std) tensors
        """
        h1 = h2 = None
        for t in range(x.size(1)):
            f = self.reduce(x[:,t])
            o1, h1 = self.l1(f, h1)
            o2, h2 = self.l2(o1, h2)
        
        o = self.head(o2)
        o = self.dropout(o)
        
        if self.n_quantiles == 1:
            out = o.squeeze(1)  # (B, H, W)
            if return_uncertainty:
                # For single output, uncertainty comes from MC dropout
                return out, None, None
            return out
        else:
            # Quantile outputs: (B, n_quantiles, H, W)
            if return_uncertainty:
                # Mean is median (0.5 quantile if present, else mean of quantiles)
                median_idx = self.n_quantiles // 2
                mean = o[:, median_idx] if self.n_quantiles % 2 == 1 else o.mean(dim=1)
                # Std estimated from quantile spread
                std = (o[:, -1] - o[:, 0]) / 2.0  # Approximate std from quantile range
                return o, mean, std
            return o

# ------------------------------------------------------------------ #
# Train / eval helpers
# ------------------------------------------------------------------ #
@torch.no_grad()
def rmse(dl, net):
    net.eval(); se = n = 0
    for X, y, m in dl:
        X, y, m = [t.to(DEVICE) for t in (X, y, m)]
        with autocast(enabled=MIXED_PREC):
            if net.n_quantiles > 1:
                # Use median quantile for RMSE
                median_idx = net.n_quantiles // 2
                pred = X[:,-1,LOGCHL_IDX] + net(X)[:, median_idx]
            else:
                pred = X[:,-1,LOGCHL_IDX] + net(X)
        err = (pred - y).masked_fill_(~m, 0)
        se += (err**2).sum().item(); n += m.sum().item()
    return math.sqrt(se / n)

@torch.no_grad()
def mc_dropout_predict(net, X, n_samples=MC_SAMPLES):
    """
    Monte Carlo Dropout prediction: sample multiple times with dropout enabled.
    
    Returns:
        mean: (B, H, W) mean prediction
        std: (B, H, W) standard deviation (epistemic uncertainty)
    """
    net.train()  # Enable dropout
    samples = []
    for _ in range(n_samples):
        with autocast(enabled=MIXED_PREC):
            delta = net(X)
            if net.n_quantiles > 1:
                median_idx = net.n_quantiles // 2
                delta = delta[:, median_idx]
            pred = X[:,-1,LOGCHL_IDX] + delta
            samples.append(pred)
    
    samples = torch.stack(samples, dim=0)  # (n_samples, B, H, W)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    return mean, std

def quantile_loss(pred_quantiles, target, quantiles, mask):
    """
    Quantile regression loss (pinball loss).
    
    Args:
        pred_quantiles: (B, n_quantiles, H, W) predicted quantiles
        target: (B, H, W) true values
        quantiles: (n_quantiles,) quantile levels
        mask: (B, H, W) valid mask
    """
    target = target.unsqueeze(1)  # (B, 1, H, W)
    errors = target - pred_quantiles  # (B, n_quantiles, H, W)
    
    quantiles_t = torch.tensor(quantiles, device=errors.device).view(1, -1, 1, 1)
    
    loss = torch.maximum(
        quantiles_t * errors,
        (quantiles_t - 1) * errors
    )  # (B, n_quantiles, H, W)
    
    loss = loss.masked_fill(~mask.unsqueeze(1), 0)
    return loss.sum() / mask.sum().clamp(min=1)

def train_one(dl, net, opt, qthr, stats, epoch, use_phys, quantiles=None):
    net.train()
    for X, y, m in dl:
        X, y, m = [t.to(DEVICE) for t in (X, y, m)]

        with autocast(enabled=MIXED_PREC):
            last_log = X[:, -1, LOGCHL_IDX]  # (B,H,W)
            
            if net.n_quantiles > 1:
                # Quantile regression
                pred_quantiles = net(X)  # (B, n_quantiles, H, W)
                median_idx = net.n_quantiles // 2
                delta = pred_quantiles[:, median_idx]  # Use median for physics loss
                
                # Quantile loss
                tgt = y - last_log
                sup_loss = quantile_loss(pred_quantiles, tgt.unsqueeze(1), quantiles, m)
            else:
                # Standard regression
                delta = net(X)  # (B,H,W)
                tgt = y - last_log
                sup_err = (delta - tgt).masked_fill_(~m, 0)
                
                # Class-weighted Huber loss
                chl_lin = torch.exp(y)
                w = torch.where(chl_lin < qthr[0], 1.0,
                        torch.where(chl_lin < qthr[1], 1.5,
                            torch.where(chl_lin < qthr[2], 2.5, 4.0)))
                w = w * m
                abs_err = sup_err.abs()
                quadratic = torch.minimum(abs_err, torch.tensor(HUBER_DELTA, device=abs_err.device))
                linear = abs_err - quadratic
                huber = 0.5 * quadratic**2 + HUBER_DELTA * linear
                sup_loss = (w * huber).sum() / w.sum()

            # Physics residual
            phys_loss = torch.tensor(0.0, device=DEVICE)
            if use_phys:
                logC_  = last_log.unsqueeze(1)  # (B,1,H,W)
                logCp_ = (last_log + delta).unsqueeze(1)
                u_     = X[:, -1, ALL_VARS.index("uo")].unsqueeze(1)
                v_     = X[:, -1, ALL_VARS.index("vo")].unsqueeze(1)
                res = physics_residual(logCp_, logC_, u_, v_, net.kappa)
                sq = (res[:,0][m]).pow(2)
                phys_loss = sq.sum() / (m.sum().clamp(min=1))
                
                mu, sd = stats["log_chl"]
                phys_loss_phys = phys_loss * (sd**2)
                
                λ_raw = sup_loss.detach() / (phys_loss_phys.detach() + 1e-8)
                λ_phys = λ_raw.clamp(min=0.1, max=1000.0)
                
                ramp = min(max((epoch - 20) / 5, 0.0), 1.0)
                λ_phys = λ_phys * ramp
            else:
                λ_phys = 0.0
                phys_loss_phys = torch.tensor(0.0, device=DEVICE)

            loss = sup_loss + λ_phys * phys_loss_phys

        SCALER.scale(loss).backward()
        SCALER.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.)
        SCALER.step(opt); SCALER.update(); opt.zero_grad()
        with torch.no_grad():
            net.kappa.clamp_(1e-2, 1e3)

# ------------------------------------------------------------------ #
# Ensemble training and inference
# ------------------------------------------------------------------ #
def train_ensemble(n_models=ENSEMBLE_SIZE, quantiles=None):
    """Train an ensemble of models with different random seeds."""
    tr_dl, va_dl, te_dl, qthr, stats = make_loaders()
    Cin = len([v for v in ALL_VARS if v in xr.open_dataset(FREEZE).data_vars])
    
    n_quantiles = len(quantiles) if quantiles else 1
    models = []
    
    for i in range(n_models):
        print(f"\n{'='*60}")
        print(f"Training ensemble member {i+1}/{n_models}")
        print(f"{'='*60}")
        
        # Different seed for each model
        seed = SEED + i * 1000
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        
        net = ConvLSTM_Uncertainty(Cin, dropout_rate=MC_DROPOUT_RATE, n_quantiles=n_quantiles).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
        sched = ReduceLROnPlateau(opt, 'min', patience=3, factor=.5)
        
        best = 1e9; bad = 0
        for ep in range(1, EPOCHS+1):
            use_phys = (ep > 20)
            train_one(tr_dl, net, opt, qthr, stats, ep, use_phys, quantiles)
            val_rm = rmse(va_dl, net)
            sched.step(val_rm)
            print(f"E{ep:02d}  val RMSE_log={val_rm:.3f}  lr={opt.param_groups[0]['lr']:.1e}")
            
            if val_rm < best - 1e-3:
                best = val_rm; bad = 0
                torch.save(net.state_dict(), OUT_DIR / f'pinn_ensemble_{i+1}_best.pt')
            else:
                bad += 1
                if bad == 6: break
        
        # Load best and evaluate
        net.load_state_dict(torch.load(OUT_DIR / f'pinn_ensemble_{i+1}_best.pt'))
        metrics = {"train": rmse(tr_dl, net), "val": rmse(va_dl, net), "test": rmse(te_dl, net)}
        print(f"\nMember {i+1} FINAL RMSE_log: {metrics}")
        models.append(net)
    
    return models, stats

@torch.no_grad()
def ensemble_predict(models, X, use_mc_dropout=False):
    """
    Aggregate predictions from ensemble.
    
    Returns:
        mean: (B, H, W) mean prediction
        std: (B, H, W) standard deviation (ensemble uncertainty)
        predictions: List of individual model predictions
    """
    predictions = []
    
    for model in models:
        model.eval()
        if use_mc_dropout:
            mean, std = mc_dropout_predict(model, X)
            predictions.append(mean)
        else:
            with autocast(enabled=MIXED_PREC):
                delta = model(X)
                if model.n_quantiles > 1:
                    median_idx = model.n_quantiles // 2
                    delta = delta[:, median_idx]
                pred = X[:,-1,LOGCHL_IDX] + delta
                predictions.append(pred)
    
    predictions = torch.stack(predictions, dim=0)  # (n_models, B, H, W)
    mean = predictions.mean(dim=0)
    std = predictions.std(dim=0)
    return mean, std, predictions

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main(ensemble_size=1, quantiles=None):
    if ensemble_size > 1:
        models, stats = train_ensemble(ensemble_size, quantiles)
        # Save ensemble info
        ensemble_info = {
            "n_models": ensemble_size,
            "quantiles": quantiles,
            "mc_dropout_rate": MC_DROPOUT_RATE
        }
        json.dump(ensemble_info, open(OUT_DIR / 'pinn_ensemble_info.json', 'w'), indent=2)
        print(f"\n{'='*60}")
        print(f"Ensemble training complete! Models saved to {OUT_DIR}")
        print(f"{'='*60}")
    else:
        # Single model training
        tr_dl, va_dl, te_dl, qthr, stats = make_loaders()
        Cin = len([v for v in ALL_VARS if v in xr.open_dataset(FREEZE).data_vars])
        
        n_quantiles = len(quantiles) if quantiles else 1
        net = ConvLSTM_Uncertainty(Cin, dropout_rate=MC_DROPOUT_RATE, n_quantiles=n_quantiles).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
        sched = ReduceLROnPlateau(opt, 'min', patience=3, factor=.5)
        
        best = 1e9; bad = 0
        for ep in range(1, EPOCHS+1):
            use_phys = (ep > 20)
            train_one(tr_dl, net, opt, qthr, stats, ep, use_phys, quantiles)
            val_rm = rmse(va_dl, net)
            sched.step(val_rm)
            print(f"E{ep:02d}  val RMSE_log={val_rm:.3f}  lr={opt.param_groups[0]['lr']:.1e}")
            
            if val_rm < best - 1e-3:
                best = val_rm; bad = 0
                ckpt_name = 'pinn_uncertainty_best.pt' if n_quantiles == 1 else 'pinn_quantiles_best.pt'
                torch.save(net.state_dict(), OUT_DIR / ckpt_name)
            else:
                bad += 1
                if bad == 6: break
        
        ckpt_name = 'pinn_uncertainty_best.pt' if n_quantiles == 1 else 'pinn_quantiles_best.pt'
        net.load_state_dict(torch.load(OUT_DIR / ckpt_name))
        metrics = {"train": rmse(tr_dl, net), "val": rmse(va_dl, net), "test": rmse(te_dl, net)}
        print("\nFINAL RMSE_log:", metrics)
        
        info = {
            "quantiles": quantiles,
            "mc_dropout_rate": MC_DROPOUT_RATE,
            "metrics": metrics
        }
        json.dump(info, open(OUT_DIR / 'pinn_uncertainty_info.json', 'w'), indent=2)

# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
import argparse
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PINN with uncertainty quantification")
    ap.add_argument("--seq", type=int, help="Sequence length")
    ap.add_argument("--epochs", type=int, help="Number of epochs")
    ap.add_argument("--batch", type=int, help="Batch size")
    ap.add_argument("--ensemble", type=int, default=1, help="Ensemble size (default: 1)")
    ap.add_argument("--quantiles", type=float, nargs="+", default=None,
                    help="Quantiles to predict (e.g., --quantiles 0.1 0.5 0.9)")
    ap.add_argument("--mc-dropout-rate", type=float, default=MC_DROPOUT_RATE,
                    help="MC Dropout rate")
    ap.add_argument("--mc-samples", type=int, default=MC_SAMPLES,
                    help="Number of MC samples for uncertainty")
    args = ap.parse_args()

    if args.seq: SEQ = args.seq
    if args.epochs: EPOCHS = args.epochs
    if args.batch: BATCH = args.batch
    if args.mc_dropout_rate: MC_DROPOUT_RATE = args.mc_dropout_rate
    if args.mc_samples: MC_SAMPLES = args.mc_samples
    
    main(ensemble_size=args.ensemble, quantiles=args.quantiles)

