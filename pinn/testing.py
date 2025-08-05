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
import argparse, pathlib, math, random, json
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

################################################################################
# 1. CLI & Constants
################################################################################
FREEZE = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
SEQ, LEAD_IDX, PATCH = 6, 1, 64
BATCH, EPOCHS, SEED = 32, 40, 42
INIT_LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_PATH = pathlib.Path("runs/pinn_best.pt")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)   # ← make sure “runs/” exists

SAT = ["log_chl","Kd_490","nflh"]
MET = ["u10","v10","wind_speed","avg_sdswrf"]
OCE = ["uo","vo","thetao","so","ssh_grad_mag"]
ALL = [*OCE, *SAT, *MET]
IDX_LOG = ALL.index("log_chl")
IDX_U = ALL.index("uo")
IDX_V = ALL.index("vo")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

################################################################################
# 2. Dataset (from working v1.3 script)
################################################################################
def z(x, mu_sd): mu, sd = mu_sd; return (x - mu) / sd

def stats(ds, tids):
    s = {}
    for v in ALL:
        if v not in ds: continue
        da = ds[v] if "time" not in ds[v].dims else ds[v].isel(time=tids)
        mu, sd = float(da.mean(skipna=True)), float(da.std(skipna=True)) or 1.
        s[v] = (mu, sd)
    return s

class PatchDS(Dataset):
    def __init__(self, ds, tids, st, mask):
        self.ds, self.tids, self.st, self.mask = ds, tids, st, mask
        self.latL = np.arange(0, ds.sizes["lat"] - PATCH + 1)
        self.lonL = np.arange(0, ds.sizes["lon"] - PATCH + 1)
        self.rng = np.random.default_rng(SEED + len(tids))
    def __len__(self): return len(self.tids)
    def _corner(self):
        for _ in range(20):
            y = int(self.rng.choice(self.latL))
            x = int(self.rng.choice(self.lonL))
            if self.mask.isel(lat=slice(y,y+PATCH),lon=slice(x,x+PATCH)).any():
                return y,x
        return 0,0
    def __getitem__(self, k):
        t = int(self.tids[k]); y,x = self._corner(); frames = []
        for dt in range(SEQ):
            bands = []
            for v in ALL:
                if v not in self.ds: continue
                sel = dict(lat=slice(y,y+PATCH), lon=slice(x,x+PATCH))
                if "time" in self.ds[v].dims:
                    sel["time"] = t - SEQ + 1 + dt
                da = self.ds[v].isel(**sel)
                bands.append(np.nan_to_num(z(da.values, self.st[v]), nan=0.0))
            frames.append(np.stack(bands, 0))
        X = torch.from_numpy(np.stack(frames, 0).astype(np.float32))
        y_t = self.ds["log_chl"].isel(time=t+LEAD_IDX, lat=slice(y,y+PATCH), lon=slice(x,x+PATCH)).values
        m = self.mask.isel(lat=slice(y,y+PATCH), lon=slice(x,x+PATCH)).values & np.isfinite(y_t)
        return X, torch.from_numpy(y_t.astype(np.float32)), torch.from_numpy(m)

def make_loaders():
    ds = xr.open_dataset(FREEZE); ds.load()
    ok = (np.isfinite(ds.log_chl).sum("time") / ds.sizes["time"]) >= 0.2
    ds["pixel_ok"] = ok.astype("uint8")
    mask = ds.pixel_ok.astype(bool)
    times = ds.time.values
    t_all = np.arange(ds.sizes["time"])
    tr = t_all[times < np.datetime64("2016-01-01")]
    va = t_all[(times >= np.datetime64("2016-01-01")) & (times <= np.datetime64("2018-12-31"))]
    st = stats(ds, tr)
    tr_ds = PatchDS(ds, tr[SEQ-1:-LEAD_IDX], st, mask)
    va_ds = PatchDS(ds, va[SEQ-1:-LEAD_IDX], st, mask)
    return DataLoader(tr_ds, BATCH, shuffle=True, drop_last=True), DataLoader(va_ds, BATCH)

################################################################################
# 3. Model
################################################################################
def init_hidden(B,C,H,W,device):
    h = torch.zeros(B,C,H,W,device=device)
    return (h.clone(), h.clone())

class ConvLSTMCell(nn.Module):
    def __init__(self, cin, cout, ks=3):
        super().__init__()
        pad = ks//2
        self.conv = nn.Conv2d(cin+cout, 4*cout, ks, padding=pad)
        self.cout = cout
    def forward(self, x, h):
        h_prev, c_prev = h
        out = self.conv(torch.cat([x, h_prev], 1))
        i,f,g,o = torch.chunk(out, 4, 1)
        i,f,o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f*c_prev + i*g
        h = o*torch.tanh(c)
        return h, (h,c)

class ConvLSTM(nn.Module):
    def __init__(self, cin, hidden=32, layers=2):
        super().__init__()
        hs = [hidden]*layers
        self.cells = nn.ModuleList([ConvLSTMCell(cin if i==0 else hs[i-1], hs[i]) for i in range(layers)])
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 1, 1)
        )
    def forward(self, x):
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
# 4. Physics kernels
################################################################################
_GRAD_KX = torch.tensor([[[[-0.5, 0.0, 0.5]]]], dtype=torch.float32)
_GRAD_KY = torch.tensor([[[[-0.5], [0.0], [0.5]]]], dtype=torch.float32)
_LAPL_K  = torch.tensor([[[[0, 1, 0],[1,-4, 1],[0, 1, 0]]]], dtype=torch.float32)

def _spatial_grads(field):
    kx = _GRAD_KX.to(field.device, dtype=field.dtype)
    ky = _GRAD_KY.to(field.device, dtype=field.dtype)
    gx = F.conv2d(field, kx, padding=(0,1))
    gy = F.conv2d(field, ky, padding=(1,0))
    return gx, gy

def _laplacian(field):
    k = _LAPL_K.to(field.device, dtype=field.dtype)
    return F.conv2d(field, k, padding=1)

################################################################################
# 5. Training
################################################################################
HUBER_DELTA = 0.15
WARM_EPOCHS = 3
λ = torch.tensor(0.0, device=DEVICE, requires_grad=False)

def train_epoch(model, dl, opt, epoch):
    model.train(); tot, n = 0, 0
    warm = epoch < WARM_EPOCHS
    for b,(X,y,m) in enumerate(dl):
        X,y,m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
        opt.zero_grad()
        d = model(X)
        last = X[:,-1,IDX_LOG]
        tgt  = y - last
        sup = F.huber_loss(d[m], tgt[m], delta=HUBER_DELTA)

        pred = last + d
        u = X[:,-1,IDX_U]
        v = X[:,-1,IDX_V]
        pred1 = pred.unsqueeze(1)
        gx, gy = _spatial_grads(pred1)
        lap = _laplacian(pred1)
        gx, gy, lap = gx[:,0], gy[:,0], lap[:,0]

        phys = ((pred - last) + u*gx + v*gy - 1e-2*lap).pow(2)
        phys = phys[m].mean()
        loss = sup if warm else sup + λ*phys
        loss.backward(); opt.step()

        tot += loss.item() * m.sum().item(); n += m.sum().item()
        if b==0 and epoch==0: λ.data = sup.detach() / phys.detach().clamp_min(1e-8)
    return math.sqrt(tot / n)

def eval_epoch(model, dl):
    model.eval(); tot, n = 0, 0
    with torch.no_grad():
        for X,y,m in dl:
            X,y,m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            d = model(X)
            last = X[:,-1,IDX_LOG]
            tgt  = y - last
            sup = F.huber_loss(d[m], tgt[m], delta=HUBER_DELTA, reduction='sum')
            tot += sup.item(); n += m.sum().item()
    return math.sqrt(tot / n)

################################################################################
# 6. Run training
################################################################################
if __name__ == "__main__":
    tr_dl, va_dl = make_loaders()
    model = ConvLSTM(cin=len(ALL)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-2)

    best = float('inf')
    for ep in range(EPOCHS):
        rmse_tr = train_epoch(model, tr_dl, opt, ep)
        rmse_va = eval_epoch(model, va_dl)
        print(f"E{ep:02d}  train={rmse_tr:.4f}  val={rmse_va:.4f}  λ={λ.item():.2e}")
        if rmse_va < best:
            best = rmse_va
            torch.save(model.state_dict(), "runs/pinn_best.pt")
            print("  ✔ checkpoint saved")

    print("Training complete. Best val RMSE:", best)
