#!/usr/bin/env python3
"""
ConvLSTM-PINN **v1.0** – California coastal HAB forecast
--------------------------------------------------------
Adds a physics-informed residual loss for the 2-D surface
advection-diffusion equation:

  (Ĉ − C₀)/Δt + u ∂Ĉ/∂x + v ∂Ĉ/∂y − κ ∇²Ĉ = 0

where
  • Ĉ  = forecast chlorophyll (linear space, mg m⁻³)
  • C₀  = last input frame chlorophyll at t₀
  • u,v = surface currents (`uo`, `vo`, m s⁻¹)
  • κ   = 25 m² s⁻¹ (literature default)

All other training logic is identical to v0.4.

Running instruction:

python pinn/baseline_model.py \
     --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
     --epochs 40 --seq 6 --lead 1 --batch 32

"""
from __future__ import annotations
import math, random, json, argparse, pathlib, numpy as np, xarray as xr, torch
from torch import median
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ──────────────────────────────────────────────────────────────
# CONFIG – defaults can be overridden by CLI
# ──────────────────────────────────────────────────────────────
FREEZE   = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
SEQ       = 12          # 48 d history
LEAD_IDX  = 1          # +8 d forecast
PATCH     = 64
BATCH     = 32
EPOCHS    = 40
SEED      = 42
STRATIFY  = True
WEIGHT_EXP=2.0
HUBER_DELTA = 1.0
# PHASE2_LAMBDA = 0.0
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# use autocast only on CUDA; MPS/CPU fall back to fp32
MIXED_PREC  = (DEVICE.type == "cuda")
SCALER      = GradScaler(enabled=MIXED_PREC)
OUT_DIR  = pathlib.Path.home()/ "HAB_Models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# physics constants
DX_METERS = 4_000.0               # grid spacing
DT_SEC    = LEAD_IDX * 8 * 86_400 # forecast horizon
KAPPA     = 25.0                  # m² s⁻¹  (horizontal diffusivity)
_FLOOR = 0.056616                 # mg m⁻³ detection floor used in log
EPS   = 1e-9

# ──────────────────────────────────────────────────────────────
# Predictor lists
# ──────────────────────────────────────────────────────────────
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos","ssh_grad_mag","so","thetao"]
DERIVED   = ["chl_anom_monthly","chl_roll24d_mean","chl_roll24d_std",
             "chl_roll40d_mean","chl_roll40d_std"]
STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]
ALL_VARS  = SATELLITE + METEO + OCEAN + DERIVED + STATIC

LOGCHL_IDX = SATELLITE.index("log_chl")
U_IDX      = ALL_VARS.index("uo")
V_IDX      = ALL_VARS.index("vo")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


'''
def _LAMBDA_SCHEDULE(ep: int) -> float:
    """≤12 % phys-loss until data fit plateaus."""
    if ep < 8:          # 0–7  data-only
        return 0.0
    elif ep < 20:       # 8–19 λ = 0.01 → 0.12
        return 0.01 * (ep - 7)
    else:               # ≥20  hold at 0.12
        return 0.12
'''
def _LAMBDA_SCHEDULE(ep: int) -> float:
    # 0-9  : data only
    # 10-19: 0.005 → 0.05
    # 20-∞ : 0.05
    return 0.0            if ep < 10 else \
           0.005*(ep-9)   if ep < 20 else \
           0.05

# ──────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────
def norm_stats(ds, train_idx):
    st = {}
    for v in ALL_VARS:
        if v not in ds: continue
        da = ds[v] if "time" not in ds[v].dims else ds[v].isel(time=train_idx)
        mu, sd = float(da.mean(skipna=True)), float(da.std(skipna=True)) or 1.0
        st[v] = (mu, sd)
    return st

def z(arr, mu_sd):  mu, sd = mu_sd; return (arr - mu) / sd

def un_z(t, mu_sd): mu, sd = mu_sd; return t * sd + mu

# ──────────────────────────────────────────────────────────────
# Dataset – identical to v0.4, but we save μ,σ for u,v
# ──────────────────────────────────────────────────────────────
class PatchDS(Dataset):
    def __init__(self, ds, tids, stats, mask):
        self.ds, self.tids, self.stats, self.mask = ds, tids, stats, mask
        self.latL = np.arange(0, ds.sizes["lat"] - PATCH + 1)
        self.lonL = np.arange(0, ds.sizes["lon"] - PATCH + 1)
        self.rng  = np.random.default_rng(SEED + len(tids))
    def __len__(self): return len(self.tids)
    def _corner(self):
        for _ in range(20):
            y0=int(self.rng.choice(self.latL)); x0=int(self.rng.choice(self.lonL))
            if self.mask.isel(lat=slice(y0,y0+PATCH), lon=slice(x0,x0+PATCH)).any():
                return y0,x0
        return 0,0
    def __getitem__(self,k):
        t = int(self.tids[k])
        # 1 out of every 8 mini-batches → use full map instead of 64×64 patch
        # always sample a 64×64 patch
        y0, x0 = self._corner()
        PATCH_ = PATCH

        frames=[]
        for dt in range(SEQ):
            bands=[]
            for v in ALL_VARS:
                if v not in self.ds: continue
                da = self.ds[v].isel(time=t-SEQ+1+dt,
                        lat=slice(y0, y0+PATCH_), lon=slice(x0, x0+PATCH_)) \
                        if "time" in self.ds[v].dims else \
                     self.ds[v].isel(lat=slice(y0, y0+PATCH_), lon=slice(x0, x0+PATCH_))
                bands.append(np.nan_to_num(z(da.values, self.stats[v]), nan=0.0))
            frames.append(np.stack(bands,0))
        X = torch.from_numpy(np.stack(frames,0).astype(np.float32))

        tgt = self.ds["log_chl"].isel(time=t+LEAD_IDX,
                    lat=slice(y0, y0+PATCH_), lon=slice(x0, x0+PATCH_)).values
        valid = self.mask.isel(lat=slice(y0, y0+PATCH_), lon=slice(x0, x0+PATCH_)).values \
                & np.isfinite(tgt)
        return X, torch.from_numpy(tgt.astype(np.float32)), torch.from_numpy(valid)

# ──────────────────────────────────────────────────────────────
# Data loaders – unchanged
# ──────────────────────────────────────────────────────────────
def make_loaders():
    ds = xr.open_dataset(FREEZE)  # load whole file
    ds.load()                     # bring into RAM now
    if "chl_lin" not in ds:
        ds["chl_lin"] = np.exp(ds["log_chl"]) - _FLOOR
    if "pixel_ok" not in ds:
        frac = np.isfinite(ds.log_chl).sum("time")/ds.sizes["time"]
        ds["pixel_ok"] = (frac>=.2).astype("uint8")
    pixel_ok = ds.pixel_ok.astype(bool)

    t_all = np.arange(ds.sizes["time"])
    times=ds.time.values
    tr=t_all[times< np.datetime64("2016-01-01")]
    va=t_all[(times>=np.datetime64("2016-01-01"))&(times<=np.datetime64("2018-12-31"))]
    te=t_all[times> np.datetime64("2018-12-31")]

    stats=norm_stats(ds,tr)
    tr_i,va_i,te_i = tr[SEQ-1:-LEAD_IDX], va[SEQ-1:-LEAD_IDX], te[SEQ-1:-LEAD_IDX]
    tr_ds,va_ds,te_ds = (PatchDS(ds,i,stats,pixel_ok) for i in (tr_i,va_i,te_i))

    sampler=None
    if STRATIFY:
        chl=(np.exp(ds.log_chl)-_FLOOR).isel(time=tr_i).where(pixel_ok).median(("lat","lon")).values
        q=np.nanquantile(chl,[.25,.5,.75]); bin_=np.digitize(chl,q)
        freq=np.maximum(np.bincount(bin_,minlength=4),1)
        w=(1/freq)**WEIGHT_EXP
        sampler=WeightedRandomSampler(torch.as_tensor(w[bin_]), len(bin_), replacement=True)

    tr_dl = DataLoader(tr_ds, BATCH, sampler=sampler,
                       shuffle=sampler is None,
                       num_workers=0, pin_memory=False, drop_last=True)
    va_dl = DataLoader(va_ds, BATCH, shuffle=False,
                       num_workers=0, pin_memory=False)
    te_dl = DataLoader(te_ds, BATCH, shuffle=False,
                       num_workers=0, pin_memory=False)
    return tr_dl,va_dl,te_dl,stats

# ──────────────────────────────────────────────────────────────
# Model – identical to baseline
# ──────────────────────────────────────────────────────────────
HID1 = 64   # new constants
HID2 = 128

class PxLSTM(nn.Module):
    def __init__(self,ci,co):
        super().__init__(); self.cell=nn.LSTMCell(ci,co); self.conv=nn.Conv2d(co,co,1)
    def forward(self,x,hc=None):
        B,C,H,W=x.shape
        flat = x.permute(0,2,3,1).reshape(B*H*W,C)
        if hc is None:
            h=torch.zeros(flat.size(0),self.cell.hidden_size,dtype=flat.dtype,device=flat.device)
            hc=(h,torch.zeros_like(h))
        h,c=self.cell(flat,hc); h_map=h.view(B,H,W,-1).permute(0,3,1,2)
        return self.conv(h_map),(h,c)

class ConvLSTM(nn.Module):
    def __init__(self, Cin, mu_u: float, sd_u: float, mu_v: float, sd_v: float):
        super().__init__()
        # self.kappa = nn.Parameter(torch.tensor(25.0))
        self.register_buffer("kappa", torch.tensor(25.0))
        self.reduce  = nn.Conv2d(Cin, 48, 1)         # 3 × HID1/4
        self.dropout = nn.Dropout2d(0.15)
        self.l1, self.l2 = PxLSTM(48, HID1), PxLSTM(HID1, HID2)
        self.head    = nn.Conv2d(HID2, 1, 1)

        # save current-field stats so we can “de-normalise” on the fly
        self.mu_u, self.sd_u = mu_u, sd_u
        self.mu_v, self.sd_v = mu_v, sd_v

        # indices of u and v inside the *channel* dimension after reduce()
        # -- they’re the same as in ALL_VARS, just shifted because reduce()
        # keeps channel order.
        # self.u_ch = U_IDX
        # self.v_ch = V_IDX

    def forward(self,x):  # x:(B,L,C,H,W)
        h1=h2=None
        for t in range(x.size(1)):
            # inp = self.reduce(x[:, t])

            frame = x[:, t].clone()           # (B, C, H, W)

            # ---- de-normalise currents BEFORE the 1×1 conv mixes channels
            frame[:, U_IDX] = frame[:, U_IDX] * self.sd_u + self.mu_u
            frame[:, V_IDX] = frame[:, V_IDX] * self.sd_v + self.mu_v

            inp = self.reduce(frame)          # (B, 32, H, W)

            inp = self.dropout(inp)
            o1, h1 = self.l1(inp, h1)
            o2, h2 = self.l2(o1, h2)
        return self.head(o2).squeeze(1)  # Δlog-chl

# ──────────────────────────────────────────────────────────────
# Physics helper kernels (3×3 central diff, replicate padding)
# ──────────────────────────────────────────────────────────────
_KDX = torch.tensor([[0,0,0],
                     [-1,0,1],
                     [0,0,0]],dtype=torch.float32)/(2*DX_METERS)
_KDY = torch.tensor([[0,-1,0],
                     [0, 0,0],
                     [0, 1,0]],dtype=torch.float32)/(2*DX_METERS)
_KLAP = torch.tensor([[0,1,0],
                      [1,-4,1],
                      [0,1,0]],dtype=torch.float32)/(DX_METERS**2)

def _kernel(k):  # (H,W) → (1,1,H,W) conv kernel
    return k.view(1,1,*k.shape)

KDX = _kernel(_KDX).to(DEVICE); KDY=_kernel(_KDY).to(DEVICE); KLAP=_kernel(_KLAP).to(DEVICE)

@torch.no_grad()
def _convolve(field, kernel):
    """
    Replicate-pads a (B,H,W) field, then applies the 3×3 kernel.

    We pad manually because torch-functional conv2d does NOT accept
    `padding_mode`.  (The argument exists only in nn.Conv2d modules.)
    """
    # field → (B,1,H,W) → replicate-pad 1 px on all sides
    padded = F.pad(field.unsqueeze(1), (1,1,1,1), mode="replicate")
    return F.conv2d(padded, kernel, padding=0).squeeze(1)   # → (B,H,W)

def physics_residual(C_pred_lin, C_last_lin, u_lin, v_lin, kappa):
    dCdt = (C_pred_lin - C_last_lin) / DT_SEC
    dCdx = _convolve(C_pred_lin, KDX)
    dCdy = _convolve(C_pred_lin, KDY)
    lap  = _convolve(C_pred_lin, KLAP)

    raw  = dCdt + u_lin*dCdx + v_lin*dCdy - kappa*lap

    finite = raw.isfinite()
    if finite.any():
        sigma = median(raw[finite].abs()) + 1e-9      # batch MAD
    else:                                            # whole batch blew up
        sigma = raw.new_tensor(3e-8)                 # safe fallback
    
    phys = torch.clamp(raw / sigma, -10., 10.)
    return phys

# ──────────────────────────────────────────────────────────────
# Train / eval helpers
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def rmse(dl,net,stats):
    net.eval(); se=n=0
    for X,y,m in dl:
        X,y,m=[t.to(DEVICE) for t in (X,y,m)]
        with autocast(device_type=DEVICE.type, enabled=MIXED_PREC):
            pred = X[:,-1,LOGCHL_IDX] + net(X)
        err=(pred-y).masked_fill_(~m,0)
        se+=(err**2).sum().item(); n+=m.sum().item()
    return math.sqrt(se/n)

def train_one(dl, net, opt, stats, sched, epoch, phase2_lambda):
    net.train()
    λ = max(_LAMBDA_SCHEDULE(epoch), phase2_lambda)
    mu_u,sd_u = stats["uo"]; mu_v,sd_v = stats["vo"]
    running_d = running_p = 0.0
    for X,y,m in dl:
        X,y,m=[t.to(DEVICE) for t in (X,y,m)]
        with autocast(device_type=DEVICE.type, enabled=MIXED_PREC):
            delta      = net(X)
            log_pred   = X[:,-1,LOGCHL_IDX] + delta
            err = log_pred - y
            err = err.masked_fill_(~m,0)

            # class-weighted Huber (unchanged)
            chl_lin = torch.exp(y)
            w = torch.where(chl_lin<0.5,1.0,
                torch.where(chl_lin<2.0,1.5,
                    torch.where(chl_lin<5.0,2.5,4.0)))
            w=w*m
            abs_err=err.abs(); quad=torch.minimum(abs_err,torch.tensor(HUBER_DELTA,device=err.device))
            lin=abs_err-quad; huber=0.5*quad**2 + HUBER_DELTA*lin
            data_loss=(w*huber).sum()/w.sum()

            # --- physics loss -----------------------------------------
            # un-scale currents & convert log→linear
            u_lin = un_z(X[:, -1, U_IDX], (mu_u, sd_u))   # <= put un_z back
            v_lin = un_z(X[:, -1, V_IDX], (mu_v, sd_v))
            C_last_lin = torch.exp(X[:,-1,LOGCHL_IDX]) - _FLOOR
            C_pred_lin = torch.exp(log_pred) - _FLOOR

            phys = physics_residual(C_pred_lin, C_last_lin, u_lin, v_lin,
                                    net.kappa)
            # Huber(δ=5σ) stabilises outliers in the residual
            phys_err  = phys[m].abs()
            quad      = torch.minimum(phys_err, torch.tensor(3.0, device=phys_err.device))
            lin       = phys_err - quad
            phys_loss = (0.5*quad**2 + 5.0*lin).mean()
            phys_loss = phys_loss * 1e-3
            
            # -------- safety check ----------
            if torch.isnan(phys_loss) or torch.isnan(data_loss):
                continue            # drop this batch and move on

            loss = data_loss + λ*phys_loss
            running_d += data_loss.item()
            running_p += phys_loss.item()

            if m.any() and (torch.rand(1).item() < 0.01):   # 1-line throttle
                print(f"   data={data_loss.item():.3e}  phys={phys_loss.item():.3e}  λ={λ:.2f}")
        SCALER.scale(loss).backward()
        SCALER.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(),.8)
        SCALER.step(opt); SCALER.update(); opt.zero_grad()
        # sched.step()

    n_batches = len(dl)
    print(f"   data={running_d/n_batches:.2e}  phys={running_p/n_batches:.2e}  λ={λ:.2f}")

# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    print("➜ 1) Opening dataset …", flush=True)
    tr_dl, va_dl, te_dl, stats = make_loaders()
    print("   ✔ Dataset & loaders ready.", flush=True)

    Cin = len([v for v in ALL_VARS if v in xr.open_dataset(FREEZE).data_vars])
    mu_u, sd_u = stats["uo"]
    mu_v, sd_v = stats["vo"]
    net = ConvLSTM(Cin, mu_u, sd_u, mu_v, sd_v).to(DEVICE)

    INIT_LR = 5e-5                         # lower start
    opt   = torch.optim.AdamW(net.parameters(), lr=INIT_LR, weight_decay=1e-4)
    EFFECTIVE_EPOCHS = EPOCHS * len(tr_dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=EFFECTIVE_EPOCHS, eta_min=5e-6)

    '''
    opt   = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    # one-cycle from 1e-6 → 3e-4 → 1e-6 over EPOCHS
    sched = ReduceLROnPlateau(opt, mode="min",
                              factor=0.5, patience=2, min_lr=1e-6)
    '''
    '''
    sched = OneCycleLR(
            opt,
            max_lr=3e-4,
            total_steps=EPOCHS * len(tr_dl),
            pct_start=0.1,          # 10 % warm-up
            anneal_strategy="cos",
            cycle_momentum=False)
    '''

    print("➜ 2) Starting training …", flush=True)
    best = 1e9; bad = 0
    phase2_lambda = 0.0

    for ep in range(1, EPOCHS + 1):
        # --- PINN scheduling tweaks —
        '''
        if ep == 11:             # first time we turn phys-loss on hard
            for g in opt.param_groups:
                g['lr'] *= 0.5        # optional LR drop
            phase2_lambda = 0.60
        elif ep == 21:           # second λ bump
            phase2_lambda *= 2.0   # now 1.20
            for g in opt.param_groups:
                g['lr'] = 1e-4 
        '''
        print(f"→ Epoch {ep:02d}/{EPOCHS}", flush=True)       # NEW
        train_one(tr_dl, net, opt, stats, sched, ep, phase2_lambda=0.0)
        val_rm = rmse(va_dl, net, stats)
        sched.step(val_rm)

        eff_lambda = max(_LAMBDA_SCHEDULE(ep), phase2_lambda)
        print(f"E{ep:02d}  val RMSE_log={val_rm:.3f}  λ={eff_lambda:.2f}  lr={opt.param_groups[0]['lr']:.1e}")
        
        if val_rm<best-1e-3:
            best,val_rm,bad = val_rm,val_rm,0
            torch.save(net.state_dict(),OUT_DIR/'convLSTM_PINN_best.pt')
        else:
            bad+=1
            if bad==10: break

    net.load_state_dict(torch.load(OUT_DIR/'convLSTM_PINN_best.pt'))
    metrics={"train":rmse(tr_dl,net,stats),"val":rmse(va_dl,net,stats),"test":rmse(te_dl,net,stats)}
    print("\nFINAL RMSE_log:",metrics)
    json.dump(metrics, open(OUT_DIR/'convLSTM_PINN_metrics.json','w'), indent=2)

# ──────────────────────────────────────────────────────────────
if __name__=="__main__":

    ap=argparse.ArgumentParser()
    ap.add_argument("--freeze"); ap.add_argument("--epochs",type=int)
    ap.add_argument("--seq",type=int); ap.add_argument("--lead",type=int)
    ap.add_argument("--batch",type=int); ap.add_argument("--lambda_phys",type=float)
    args = ap.parse_args()

    if args.freeze: FREEZE = pathlib.Path(args.freeze)
    if args.epochs: EPOCHS = args.epochs
    if args.seq:    SEQ    = args.seq
    if args.lead:
        LEAD_IDX = args.lead
        DT_SEC   = LEAD_IDX * 8 * 86_400
    if args.batch: BATCH = args.batch

    # ⇣⇣  FIX – just re-assign, no `global` needed
    if args.lambda_phys is not None:
        _LAMBDA_SCHEDULE = lambda _: args.lambda_phys   # constant λ

    # >>>>>>  σ_phys sanity-check (do this once)  <<<<<<
    ds_tmp = xr.open_dataset(FREEZE)
    ds_tmp.load()
    chl_lin = (np.exp(ds_tmp.log_chl) - _FLOOR).isel(time=slice(None, -LEAD_IDX))
    dCdt_tmp = chl_lin.diff("time") / DT_SEC
    print("median |∂C/∂t| =", np.nanmedian(np.abs(dCdt_tmp)).item())
    # ------------------------------------------------------------------------

    main()

'''
Results (V1):
FINAL RMSE_log: {'train': 0.8271754853477319, 'val': 0.7977109217601834, 'test': 0.8156437664067274}

Results (V2):
FINAL RMSE_log: {'train': 0.8784411178414903, 'val': 0.804262289162801, 'test': 0.8186307758385508}

Results (V3):
FINAL RMSE_log: {'train': 0.8469711010102171, 'val': 0.8520329107221408, 'test': 0.8345486178811988}

'''