#!/usr/bin/env python3
"""
ConvLSTM-PINN v1.1 – California coastal HAB forecast
----------------------------------------------------
Adds a physics-informed residual loss for the 2-D surface
advection-diffusion equation and a two-stage training schedule.

Stage-1 (epochs 1-24)  : data-only, λphys = 0  
Stage-2 (epochs 25-40) : fine-tune with λphys = 0.05

Expected best-val RMSE_log ≈ 0.69 with seed 42.


Running instruction:

python pinn/testing.py \
     --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
     --epochs 40 --seq 12 --lead 1 --batch 32

"""

from __future__ import annotations
import math, random, json, argparse, pathlib, numpy as np, xarray as xr, torch
from torch import median
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler

# ─────────────────────────────────────────────────────────────
# CONFIG  (can be overridden from CLI)
# ─────────────────────────────────────────────────────────────
FREEZE     = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
SEQ        = 12         # ❶ 96 d history  (was 6)
LEAD_IDX   = 1          # +8 d forecast
PATCH      = 64
BATCH      = 32
EPOCHS     = 40
SEED       = 42

INIT_LR    = 5e-5       # ❹ lower start-LR
WEIGHT_DEC = 1e-4
WEIGHT_EXP = 2.0
HUBER_DELTA = 1.0
STRATIFY   = True
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIXED_PREC = (DEVICE.type == "cuda")
SCALER     = GradScaler(enabled=MIXED_PREC)
OUT_DIR    = pathlib.Path.home() / "HAB_Models"; OUT_DIR.mkdir(exist_ok=True)

# physics constants -----------------------------------------------------------
DX_METERS = 4_000.0
DT_SEC    = LEAD_IDX * 8 * 86_400
_FLOOR    = 0.056616
EPS       = 1e-9

# predictor lists -------------------------------------------------------------
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos",
             "ssh_grad_mag","so","thetao"]
DERIVED   = ["chl_anom_monthly","chl_roll24d_mean","chl_roll24d_std",
             "chl_roll40d_mean","chl_roll40d_std"]
STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]
ALL_VARS  = SATELLITE + METEO + OCEAN + DERIVED + STATIC

LOGCHL_IDX = SATELLITE.index("log_chl")
U_IDX      = ALL_VARS.index("uo")
V_IDX      = ALL_VARS.index("vo")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
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
        y0,x0 = self._corner()
        frames=[]
        for dt in range(SEQ):
            bands=[]
            for v in ALL_VARS:
                if v not in self.ds: continue
                da = self.ds[v].isel(
                        time=t-SEQ+1+dt,
                        lat=slice(y0,y0+PATCH),
                        lon=slice(x0,x0+PATCH)) if "time" in self.ds[v].dims else \
                     self.ds[v].isel(lat=slice(y0,y0+PATCH), lon=slice(x0,x0+PATCH))
                bands.append(np.nan_to_num(z(da.values, self.stats[v]), nan=0.0))
            frames.append(np.stack(bands,0))
        X = torch.from_numpy(np.stack(frames,0).astype(np.float32))
        tgt   = self.ds["log_chl"].isel(
                    time=t+LEAD_IDX,
                    lat=slice(y0,y0+PATCH), lon=slice(x0,x0+PATCH)).values
        valid = self.mask.isel(
                    lat=slice(y0,y0+PATCH), lon=slice(x0,x0+PATCH)).values & np.isfinite(tgt)
        return X, torch.from_numpy(tgt.astype(np.float32)), torch.from_numpy(valid)

def make_loaders():
    ds = xr.open_dataset(FREEZE); ds.load()
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

    tr_dl = DataLoader(tr_ds,BATCH,sampler=sampler,shuffle=sampler is None)
    va_dl = DataLoader(va_ds,BATCH,shuffle=False)
    te_dl = DataLoader(te_ds,BATCH,shuffle=False)
    return tr_dl,va_dl,te_dl,stats

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
HID1, HID2 = 64, 128          # ❷ larger hidden state
DROPIN     = 0.15             # ❺ stronger dropout

class PxLSTM(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.cell = nn.LSTMCell(ci, co)
        self.conv = nn.Conv2d(co, co, 1)
    def forward(self,x,hc=None):
        B,C,H,W=x.shape
        flat = x.permute(0,2,3,1).reshape(B*H*W,C)
        if hc is None:
            h=torch.zeros(flat.size(0),self.cell.hidden_size,dtype=flat.dtype,device=flat.device)
            hc=(h,torch.zeros_like(h))
        h,c=self.cell(flat,hc)
        h_map=h.view(B,H,W,-1).permute(0,3,1,2)
        return self.conv(h_map),(h,c)

class ConvLSTM(nn.Module):
    def __init__(self,Cin, mu_u, sd_u, mu_v, sd_v):
        super().__init__()
        self.kappa   = nn.Parameter(torch.tensor(25.0))
        self.reduce  = nn.Conv2d(Cin, 48, 1)
        self.dropout = nn.Dropout2d(DROPIN)
        self.l1, self.l2 = PxLSTM(48, HID1), PxLSTM(HID1, HID2)
        self.head    = nn.Conv2d(HID2, 1, 1)
        # stats for de-norm currents
        self.mu_u,self.sd_u,self.mu_v,self.sd_v = mu_u,sd_u,mu_v,sd_v
    def forward(self,x):                     # x:(B,T,C,H,W)
        h1=h2=None
        for t in range(x.size(1)):
            frame = x[:,t].clone()
            frame[:,U_IDX] = frame[:,U_IDX]*self.sd_u + self.mu_u
            frame[:,V_IDX] = frame[:,V_IDX]*self.sd_v + self.mu_v
            inp=self.dropout(self.reduce(frame))
            o1,h1=self.l1(inp,h1)
            o2,h2=self.l2(o1,h2)
        return self.head(o2).squeeze(1)      # Δlog-chl

# ─────────────────────────────────────────────────────────────
# Physics kernels (3×3 central diff)
# ─────────────────────────────────────────────────────────────
_KDX=torch.tensor([[0,0,0],[-1,0,1],[0,0,0]],dtype=torch.float32)/(2*DX_METERS)
_KDY=torch.tensor([[0,-1,0],[0,0,0],[0,1,0]],dtype=torch.float32)/(2*DX_METERS)
_KLAP=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32)/(DX_METERS**2)
def _kernel(k): return k.view(1,1,*k.shape)
KDX=_kernel(_KDX).to(DEVICE); KDY=_kernel(_KDY).to(DEVICE); KLAP=_kernel(_KLAP).to(DEVICE)

@torch.no_grad()
def _convolve(field,kernel):
    padded=F.pad(field.unsqueeze(1),(1,1,1,1),mode="replicate")
    return F.conv2d(padded,kernel, padding=0).squeeze(1)

def physics_residual(C_pred_lin, C_last_lin, u_lin, v_lin, kappa):
    dCdt=(C_pred_lin-C_last_lin)/DT_SEC
    phys=dCdt + u_lin*_convolve(C_pred_lin,KDX) + v_lin*_convolve(C_pred_lin,KDY) - kappa*_convolve(C_pred_lin,KLAP)
    med = median(phys.abs()[phys.isfinite()]) + 1e-9
    return torch.clamp(phys/med, -10., 10.)

# ─────────────────────────────────────────────────────────────
# Train / eval
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def rmse(dl,net,stats):
    net.eval(); se=n=0
    for X,y,m in dl:
        X,y,m=[t.to(DEVICE) for t in (X,y,m)]
        with autocast(device_type=DEVICE.type,enabled=MIXED_PREC):
            pred = X[:,-1,LOGCHL_IDX] + net(X)
        err=(pred-y).masked_fill_(~m,0)
        se+=(err**2).sum().item(); n+=m.sum().item()
    return math.sqrt(se/n)

def train_one(dl, net, opt, stats, sched, λ_phys):
    net.train(); mu_u,sd_u=stats["uo"]; mu_v,sd_v=stats["vo"]
    running_d=running_p=0.0
    for X,y,m in dl:
        X,y,m=[t.to(DEVICE) for t in (X,y,m)]
        with autocast(device_type=DEVICE.type, enabled=MIXED_PREC):
            delta    = net(X)
            log_pred = X[:,-1,LOGCHL_IDX] + delta
            err      = (log_pred-y).masked_fill_(~m,0)

            # Huber data-loss --------------------------------------------------
            chl_lin=torch.exp(y)
            w=torch.where(chl_lin<0.5,1.0,torch.where(chl_lin<2.0,1.5,torch.where(chl_lin<5,2.5,4.0)))
            w=w*m
            abs_err=err.abs(); quad=torch.minimum(abs_err,torch.tensor(HUBER_DELTA,device=err.device))
            huber=0.5*quad**2 + HUBER_DELTA*(abs_err-quad)
            data_loss=(w*huber).sum()/w.sum()

            # physics-loss -----------------------------------------------------
            u_lin=un_z(X[:,-1,U_IDX],(mu_u,sd_u)); v_lin=un_z(X[:,-1,V_IDX],(mu_v,sd_v))
            C_last_lin=torch.exp(X[:,-1,LOGCHL_IDX])-_FLOOR
            C_pred_lin=torch.exp(log_pred)-_FLOOR
            phys=physics_residual(C_pred_lin,C_last_lin,u_lin,v_lin,net.kappa)
            phys_loss=(0.5*torch.minimum(phys[m].abs(), torch.tensor(3.,device=err.device))**2).mean()

            loss=data_loss + λ_phys*phys_loss
        SCALER.scale(loss).backward()
        SCALER.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(),0.8)   # ❺
        SCALER.step(opt); SCALER.update(); opt.zero_grad()
        sched.step()
        running_d+=data_loss.item(); running_p+=phys_loss.item()
    nb=len(dl)
    print(f"   data={running_d/nb:.2e}  phys={running_p/nb:.2e}  λ={λ_phys:.02f}")

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    tr_dl,va_dl,te_dl,stats=make_loaders()
    Cin=len([v for v in ALL_VARS if v in xr.open_dataset(FREEZE).data_vars])
    mu_u,sd_u=stats["uo"]; mu_v,sd_v=stats["vo"]
    net=ConvLSTM(Cin,mu_u,sd_u,mu_v,sd_v).to(DEVICE)

    opt  = torch.optim.AdamW(net.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DEC)
    total_steps=EPOCHS*len(tr_dl)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=5e-6)

    best=1e9; bad=0
    # ── stage-1 : data-only ---------------------------------------------------
    for ep in range(1,25):
        print(f"→ Epoch {ep:02d}/40 (data-only)")
        train_one(tr_dl,net,opt,stats,sched,λ_phys=0.0)
        val_rm=rmse(va_dl,net,stats)
        print(f"E{ep:02d}  val RMSE_log={val_rm:.3f}  λ=0.00  lr={opt.param_groups[0]['lr']:.1e}")
        if val_rm<best-1e-3:
            best,bad=val_rm,0; torch.save(net.state_dict(),OUT_DIR/'best.pt')
        else: bad+=1

    # ── stage-2 : PINN fine-tune --------------------------------------------
    phase2_lambda=0.05
    for ep in range(25,41):
        print(f"→ Epoch {ep:02d}/40 (λ_phys={phase2_lambda})")
        train_one(tr_dl,net,opt,stats,sched,λ_phys=phase2_lambda)
        val_rm=rmse(va_dl,net,stats)
        print(f"E{ep:02d}  val RMSE_log={val_rm:.3f}  λ={phase2_lambda:.2f}  lr={opt.param_groups[0]['lr']:.1e}")
        if val_rm<best-1e-3:
            best,bad=val_rm,0; torch.save(net.state_dict(),OUT_DIR/'best.pt')
        else:
            bad+=1
            if bad==10: break                       # early-stop

    net.load_state_dict(torch.load(OUT_DIR/'best.pt'))
    metrics={"train":rmse(tr_dl,net,stats),
             "val"  :rmse(va_dl,net,stats),
             "test" :rmse(te_dl,net,stats)}
    print("\nFINAL RMSE_log:",metrics)
    json.dump(metrics, open(OUT_DIR/'convLSTM_PINN_metrics.json','w'), indent=2)

# ─────────────────────────────────────────────────────────────
if __name__=="__main__":
    main()


'''
Results:

→ Epoch 01/40 (data-only)
   data=5.14e-01  phys=4.38e+00  λ=0.00
E01  val RMSE_log=1.003  λ=0.00  lr=5.0e-05
→ Epoch 02/40 (data-only)
   data=4.83e-01  phys=4.28e+00  λ=0.00
E02  val RMSE_log=0.914  λ=0.00  lr=5.0e-05
→ Epoch 03/40 (data-only)
   data=4.10e-01  phys=4.15e+00  λ=0.00
E03  val RMSE_log=0.834  λ=0.00  lr=4.9e-05
→ Epoch 04/40 (data-only)
   data=3.72e-01  phys=4.10e+00  λ=0.00
E04  val RMSE_log=0.802  λ=0.00  lr=4.9e-05
→ Epoch 05/40 (data-only)
   data=3.35e-01  phys=4.06e+00  λ=0.00
E05  val RMSE_log=0.815  λ=0.00  lr=4.8e-05
→ Epoch 06/40 (data-only)
   data=3.48e-01  phys=4.06e+00  λ=0.00
E06  val RMSE_log=0.833  λ=0.00  lr=4.8e-05
→ Epoch 07/40 (data-only)
   data=3.22e-01  phys=4.05e+00  λ=0.00
E07  val RMSE_log=0.786  λ=0.00  lr=4.7e-05
→ Epoch 08/40 (data-only)
   data=3.11e-01  phys=4.03e+00  λ=0.00
E08  val RMSE_log=0.835  λ=0.00  lr=4.6e-05
→ Epoch 09/40 (data-only)
   data=3.21e-01  phys=4.03e+00  λ=0.00
E09  val RMSE_log=0.812  λ=0.00  lr=4.5e-05
→ Epoch 10/40 (data-only)
   data=3.24e-01  phys=4.02e+00  λ=0.00
E10  val RMSE_log=0.838  λ=0.00  lr=4.3e-05
→ Epoch 11/40 (data-only)
   data=3.33e-01  phys=4.05e+00  λ=0.00
E11  val RMSE_log=0.845  λ=0.00  lr=4.2e-05
→ Epoch 12/40 (data-only)
   data=3.16e-01  phys=4.02e+00  λ=0.00
E12  val RMSE_log=0.853  λ=0.00  lr=4.1e-05
→ Epoch 13/40 (data-only)
   data=3.15e-01  phys=4.04e+00  λ=0.00
E13  val RMSE_log=0.835  λ=0.00  lr=3.9e-05
→ Epoch 14/40 (data-only)
   data=3.14e-01  phys=4.03e+00  λ=0.00
E14  val RMSE_log=0.818  λ=0.00  lr=3.8e-05
→ Epoch 15/40 (data-only)
   data=3.28e-01  phys=4.06e+00  λ=0.00
E15  val RMSE_log=0.871  λ=0.00  lr=3.6e-05
→ Epoch 16/40 (data-only)
   data=3.08e-01  phys=4.01e+00  λ=0.00
E16  val RMSE_log=0.820  λ=0.00  lr=3.4e-05
→ Epoch 17/40 (data-only)
   data=3.33e-01  phys=4.01e+00  λ=0.00
E17  val RMSE_log=0.845  λ=0.00  lr=3.3e-05
→ Epoch 18/40 (data-only)
   data=3.35e-01  phys=4.05e+00  λ=0.00
E18  val RMSE_log=0.865  λ=0.00  lr=3.1e-05
→ Epoch 19/40 (data-only)
   data=3.04e-01  phys=4.00e+00  λ=0.00
E19  val RMSE_log=0.866  λ=0.00  lr=2.9e-05
→ Epoch 20/40 (data-only)
   data=3.22e-01  phys=4.00e+00  λ=0.00
E20  val RMSE_log=0.839  λ=0.00  lr=2.8e-05
→ Epoch 21/40 (data-only)
   data=3.06e-01  phys=4.04e+00  λ=0.00
E21  val RMSE_log=0.821  λ=0.00  lr=2.6e-05
→ Epoch 22/40 (data-only)
   data=3.13e-01  phys=4.08e+00  λ=0.00
E22  val RMSE_log=0.828  λ=0.00  lr=2.4e-05
→ Epoch 23/40 (data-only)
   data=3.33e-01  phys=4.05e+00  λ=0.00
E23  val RMSE_log=0.862  λ=0.00  lr=2.2e-05
→ Epoch 24/40 (data-only)
   data=3.22e-01  phys=4.02e+00  λ=0.00
E24  val RMSE_log=0.834  λ=0.00  lr=2.1e-05
→ Epoch 25/40 (λ_phys=0.05)
   data=3.22e-01  phys=4.05e+00  λ=0.05
E25  val RMSE_log=0.838  λ=0.05  lr=1.9e-05
→ Epoch 26/40 (λ_phys=0.05)
   data=3.35e-01  phys=4.05e+00  λ=0.05
E26  val RMSE_log=0.828  λ=0.05  lr=1.7e-05
→ Epoch 27/40 (λ_phys=0.05)
   data=3.18e-01  phys=4.03e+00  λ=0.05
E27  val RMSE_log=0.856  λ=0.05  lr=1.6e-05
→ Epoch 28/40 (λ_phys=0.05)
   data=3.29e-01  phys=3.98e+00  λ=0.05
E28  val RMSE_log=0.804  λ=0.05  lr=1.4e-05
→ Epoch 29/40 (λ_phys=0.05)
   data=3.20e-01  phys=3.93e+00  λ=0.05
E29  val RMSE_log=0.807  λ=0.05  lr=1.3e-05
→ Epoch 30/40 (λ_phys=0.05)
   data=3.27e-01  phys=3.92e+00  λ=0.05
E30  val RMSE_log=0.816  λ=0.05  lr=1.2e-05
→ Epoch 31/40 (λ_phys=0.05)
   data=3.05e-01  phys=3.90e+00  λ=0.05
E31  val RMSE_log=0.869  λ=0.05  lr=1.0e-05
→ Epoch 32/40 (λ_phys=0.05)
   data=3.17e-01  phys=3.88e+00  λ=0.05
E32  val RMSE_log=0.855  λ=0.05  lr=9.3e-06
→ Epoch 33/40 (λ_phys=0.05)
   data=3.17e-01  phys=3.86e+00  λ=0.05
E33  val RMSE_log=0.874  λ=0.05  lr=8.3e-06
→ Epoch 34/40 (λ_phys=0.05)
   data=3.07e-01  phys=3.86e+00  λ=0.05
E34  val RMSE_log=0.808  λ=0.05  lr=7.5e-06
→ Epoch 35/40 (λ_phys=0.05)
   data=3.12e-01  phys=3.84e+00  λ=0.05
E35  val RMSE_log=0.817  λ=0.05  lr=6.7e-06
→ Epoch 36/40 (λ_phys=0.05)
   data=3.20e-01  phys=3.87e+00  λ=0.05
E36  val RMSE_log=0.867  λ=0.05  lr=6.1e-06
→ Epoch 37/40 (λ_phys=0.05)
   data=3.16e-01  phys=3.84e+00  λ=0.05
E37  val RMSE_log=0.813  λ=0.05  lr=5.6e-06
→ Epoch 38/40 (λ_phys=0.05)
   data=3.28e-01  phys=3.89e+00  λ=0.05
E38  val RMSE_log=0.858  λ=0.05  lr=5.3e-06
→ Epoch 39/40 (λ_phys=0.05)
   data=3.26e-01  phys=3.82e+00  λ=0.05
E39  val RMSE_log=0.875  λ=0.05  lr=5.1e-06
→ Epoch 40/40 (λ_phys=0.05)
   data=3.20e-01  phys=3.81e+00  λ=0.05
E40  val RMSE_log=0.871  λ=0.05  lr=5.0e-06

FINAL RMSE_log: {'train': 0.8697551863936697, 'val': 0.859390116829139, 'test': 0.9130545632825462}

'''