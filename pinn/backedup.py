"""
PINN‑ConvLSTM (minimal, self‑contained)
=====================================
Forecast +8‑day log‑chlorophyll composites for the California coastal strip.
The model learns the *innovation* (Δ to persistence) and is regularised by a
2‑D advection–diffusion residual.  Designed to avoid the persistence‑collapse
failure mode.

Usage (single‑GPU, 40 epochs):
    python pinn/testing.py \
        --freeze /Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc \
        --epochs 40 --seq 6 --lead 1 --batch 64 --device cpu

Freeze file requirements
-----------------------
* 8‑day composites already generated.
* Variables listed in VARS (below) present & z‑score statistics pre‑computed.
* land / deep‑ocean pixels = NaN.
* time dimension regular 8‑day cadence.

Key differences vs earlier template
-----------------------------------
* Predicts Δ = y − last_frame; persistence ⇒ 0, so any skill must be <0 error.
* Physics residual masked to ocean pixels *only*.
* λ auto‑scaled on first batch:  λ = L_sup / L_phys  ⇒ comparable magnitude.
* Warm‑up: first 3 epochs λ = 0.
* Loss: Huber on Δ, MSE on PDE residual.
* Simple ConvLSTM‑encoder → 2‑layer CNN head.

Created 2025‑08‑04
"""
import argparse, pathlib, math, random
from functools import partial

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

################################################################################
# 1. Hyper‑parameters & CLI
################################################################################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--freeze", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seq", type=int, default=6)
    p.add_argument("--lead", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save", default="runs/pinn_simple.pt")
    return p.parse_args()

# variable order – keep currents first for PDE.
VARS = [
    # currents (advective)
    "uo", "vo",
    # core colour vars
    "log_chl", "Kd_490", "nflh",
    # met
    "u10", "v10", "wind_speed", "avg_sdswrf",
    # hydro / physics
    "thetao", "so", "ssh_grad_mag",
]
CUR_U_IDX = VARS.index("uo")
CUR_V_IDX = VARS.index("vo")
LOG_IDX   = VARS.index("log_chl")

################################################################################
# 2. Dataset
################################################################################
class HABSequence(Dataset):
    def __init__(self, nc_path:str, split:str, seq=6, lead=1, crop=64):
        ds = xr.open_dataset(nc_path, chunks={})  # load in RAM once
        tsplit = {
            "train": slice(None, "2015-12-31"),
            "val":   slice("2016-01-01", "2018-12-31"),
            "test":  slice("2019-01-01", None)
        }[split]
        self.seq, self.lead, self.crop = seq, lead, crop
        self.data = ds[VARS].sel(time=tsplit).to_array().transpose("time","variable","lat","lon")
        self.mask = (~np.isnan(ds["log_chl"]).sel(time=tsplit))
        self._prep_stats()
        self.T, self.C = self.data.shape[1], len(VARS)

    def _prep_stats(self):
        mu = self.data.mean(dim=["time","lat","lon"], skipna=True)
        sig = self.data.std (dim=["time","lat","lon"], skipna=True)
        self.data = (self.data - mu)/sig
        # NaNs to zero (only land / deep ocean)
        self.data = self.data.fillna(0.)

    def __len__(self):
        return self.data.shape[0] - (self.seq + self.lead)

    def __getitem__(self, idx):
        t0 = idx
        X  = self.data.isel(time=slice(t0, t0+self.seq)).values.astype(np.float32)
        y  = self.data.isel(time=t0+self.seq+self.lead-1).sel(variable="log_chl").values.astype(np.float32)
        m  = self.mask.isel(time=t0+self.seq+self.lead-1).values.astype(np.bool_)
        # random crop (lat, lon dims)
        H,W = y.shape
        if H > self.crop:
            i = random.randrange(0, H-self.crop)
            j = random.randrange(0, W-self.crop)
            X = X[:, :, i:i+self.crop, j:j+self.crop]
            y = y[ i:i+self.crop, j:j+self.crop]
            m = m[ i:i+self.crop, j:j+self.crop]
        return torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(m)

################################################################################
# 3. Model
################################################################################
class ConvLSTMCell(nn.Module):
    def __init__(self, cin, cout, ks=3):
        super().__init__()
        pad = ks//2
        self.conv = nn.Conv2d(cin+cout, 4*cout, ks, padding=pad)
        self.cout = cout
    def forward(self, x, h):
        # x: B,C,H,W ; h=(h,c)
        h_prev, c_prev = h
        out = self.conv(torch.cat([x, h_prev], 1))
        i,f,g,o = torch.chunk(out,4,1)
        i,f,o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f*c_prev + i*g
        h = o*torch.tanh(c)
        return h,(h,c)

def init_hidden(B,C,H,W,device):
    h = torch.zeros(B,C,H,W,device=device)
    return (h.clone(), h.clone())

class ConvLSTM(nn.Module):
    def __init__(self, cin, hidden=32, layers=2):
        super().__init__()
        hs = [hidden]*layers
        cells=[]
        for i in range(layers):
            cells.append(ConvLSTMCell(cin if i==0 else hs[i-1], hs[i]))
        self.cells = nn.ModuleList(cells)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 1, 1)  # Δ log_chl
        )
    def forward(self, x):
        # x: B, S, C, H, W
        B,S,C,H,W = x.shape
        h = [init_hidden(B,cell.cout,H,W,x.device) for cell in self.cells]
        for t in range(S):
            z = x[:,t]
            for k,cell in enumerate(self.cells):
                h_k, c_k = h[k]
                h_k,(h_k,c_k) = cell(z,h[k])
                h[k] = (h_k,c_k)
                z = h_k
        d = self.head(z)[:,0]  # B,H,W
        return d

################################################################################
# 4. Physics residual helper
################################################################################
# ---------- full-size derivative kernels (no grads) ----------
_GRAD_KX = torch.tensor([[[[-0.5, 0.0, 0.5]]]],  dtype=torch.float32)
_GRAD_KY = torch.tensor([[[[-0.5], [0.0], [0.5]]]], dtype=torch.float32)
_LAPL_K  = torch.tensor([[[[0, 1, 0],
                           [1,-4, 1],
                           [0, 1, 0]]]],          dtype=torch.float32)

def _spatial_grads(field):
    kx = _GRAD_KX.to(field.device, dtype=field.dtype)
    ky = _GRAD_KY.to(field.device, dtype=field.dtype)
    gx = F.conv2d(field, kx, padding=(0, 1))
    gy = F.conv2d(field, ky, padding=(1, 0))
    return gx, gy

def _laplacian(field):
    k = _LAPL_K.to(field.device, dtype=field.dtype)
    return F.conv2d(field, k, padding=1)
# ----------------------------------------------------------------

################################################################################
# 5. Training utilities
################################################################################
HUBER_DELTA = 0.15

def train_epoch(model, dl, opt, epoch, λ, warm):
    model.train(); tot, n = 0,0
    for b,(X,y,m) in enumerate(dl):
        X,y,m = X.to(device),y.to(device),m.to(device)
        opt.zero_grad()
        d = model(X)            # Δ prediction
        last = X[:,-1,LOG_IDX]
        tgt = y - last          # Δ target
        sup = F.huber_loss(d[m], tgt[m], delta=HUBER_DELTA)
        # ---------- physics residual (full 64×64 grid) ----------
        u = X[:,-1,CUR_U_IDX]          # (B,H,W)
        v = X[:,-1,CUR_V_IDX]

        pred  = last + d               # absolute forecast  (B,H,W)
        pred1 = pred.unsqueeze(1)      # add channel dim   (B,1,H,W)

        gx, gy = _spatial_grads(pred1) # derivatives (B,1,H,W)
        lap    = _laplacian(pred1)

        gx, gy, lap = gx[:,0], gy[:,0], lap[:,0]   # drop channel dim
        dt   = 1.0                     # 1 time-step = 8 days  (constant factor)

        phys = ((pred - last)/dt + u*gx + v*gy - 1e-2*lap).pow(2)
        phys = phys[m].mean()
        # --------------------------------------------------------
        if warm:                    # λ = 0 during warm‑up
            loss = sup
        else:
            loss = sup + λ*phys
        loss.backward()
        opt.step()
        tot += loss.item()*m.sum().item(); n += m.sum().item()
        if b==0 and epoch==0:   # auto‑scale λ on first batch
            λ.data = sup.detach()/phys.detach().clamp_min(1e-8)
    return math.sqrt(tot/n)

def eval_epoch(model, dl):
    model.eval(); tot,n=0,0
    with torch.no_grad():
        for X,y,m in dl:
            X,y,m = X.to(device),y.to(device),m.to(device)
            d = model(X)
            last = X[:,-1,LOG_IDX]
            tgt = y - last
            sup = F.huber_loss(d[m], tgt[m], delta=HUBER_DELTA, reduction='sum')
            tot += sup.item(); n += m.sum().item()
    return math.sqrt(tot/n)

################################################################################
# 6. Main script
################################################################################
if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    ds_train = HABSequence(args.freeze, "train", args.seq, args.lead)
    ds_val   = HABSequence(args.freeze, "val",   args.seq, args.lead)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False)

    model = ConvLSTM(cin=len(VARS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    λ = torch.tensor(0.0, device=device, requires_grad=False)

    best = float('inf')
    for ep in range(args.epochs):
        warm = ep < 3
        rmse_tr = train_epoch(model, dl_train, opt, ep, λ, warm)
        rmse_v  = eval_epoch (model, dl_val)
        print(f"E{ep:02d} train {rmse_tr:.4f}  val {rmse_v:.4f}  λ {λ.item():.3e}")
        if rmse_v < best:
            best = rmse_v
            torch.save({"epoch":ep,"model":model.state_dict()}, args.save)
            print("  ✔ best so far, checkpoint saved")

    print("Training complete. Best val RMSE:", best)
