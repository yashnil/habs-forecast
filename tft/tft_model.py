#!/usr/bin/env python3
"""
tft_train.py  —  Temporal Fusion Transformer (TFT) with optional PINN loss
===========================================================================

Usage (identical flags to testing.py):
    python tft/tft_train.py --epochs 40 --batch 16
"""

from __future__ import annotations
import math, random, json, pathlib, numpy as np, xarray as xr
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────────────────────  CONFIG  ──────────────────────────────────
FREEZE  = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
SEQ      = 6
LEAD_IDX = 1
PATCH    = 64
BATCH    = 32
EPOCHS   = 40
SEED     = 42
STRATIFY = True
WEIGHT_EXP = 1.5
HUBER_DELTA = 1.0
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR  = pathlib.Path.home() / "HAB_Models"
OUT_DIR.mkdir(exist_ok=True, parents=True)
_FLOOR   = 0.056616   # mg m⁻³ detection floor

SATELLITE = ["log_chl","Kd_490","nflh"]
METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos","ssh_grad_mag","so","thetao"]
DERIVED   = ["chl_anom_monthly","chl_roll24d_mean","chl_roll24d_std",
             "chl_roll40d_mean","chl_roll40d_std"]
STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]
ALL_VARS  = SATELLITE + METEO + OCEAN + DERIVED + STATIC
LOGCHL_IDX = SATELLITE.index("log_chl")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ───────────────────────────────  DATASET  ────────────────────────────────
def norm_stats(ds, idx):
    out={}
    for v in ALL_VARS:
        if v not in ds: continue
        da = ds[v] if "time" not in ds[v].dims else ds[v].isel(time=idx)
        mu,sd = float(da.mean(skipna=True)), float(da.std(skipna=True)) or 1.0
        out[v]=(mu,sd)
    return out

def z(arr,mu_sd): mu,sd=mu_sd; return (arr-mu)/sd

class PatchDS(Dataset):
    def __init__(self,ds,tids,stats,mask):
        self.ds,self.tids,self.stats,self.mask=ds,tids,stats,mask
        self.latL=np.arange(0,ds.sizes["lat"]-PATCH+1)
        self.lonL=np.arange(0,ds.sizes["lon"]-PATCH+1)
        self.rng=np.random.default_rng(SEED+len(tids))
    def __len__(self): return len(self.tids)
    def _corner(self):
        for _ in range(20):
            y0=int(self.rng.choice(self.latL)); x0=int(self.rng.choice(self.lonL))
            if self.mask.isel(lat=slice(y0,y0+PATCH),lon=slice(x0,x0+PATCH)).any(): return y0,x0
        return 0,0
    def __getitem__(self,k):
        t=int(self.tids[k]); y0,x0=self._corner()
        frames=[]
        for dt in range(SEQ):
            bands=[]
            for v in ALL_VARS:
                if v not in self.ds: continue
                da=self.ds[v].isel(time=t-SEQ+1+dt,lat=slice(y0,y0+PATCH),lon=slice(x0,x0+PATCH)) if "time" in self.ds[v].dims \
                   else self.ds[v].isel(lat=slice(y0,y0+PATCH),lon=slice(x0,x0+PATCH))
                bands.append(np.nan_to_num(z(da.values,self.stats[v]),nan=0.0))
            frames.append(np.stack(bands,0))
        X=torch.from_numpy(np.stack(frames,0).astype(np.float32))     # (L,C,H,W)
        tgt=self.ds["log_chl"].isel(time=t+LEAD_IDX,lat=slice(y0,y0+PATCH),lon=slice(x0,x0+PATCH)).values
        valid=self.mask.isel(lat=slice(y0,y0+PATCH),lon=slice(x0,x0+PATCH)).values & np.isfinite(tgt)
        return X,torch.from_numpy(tgt.astype(np.float32)),torch.from_numpy(valid)

def make_loaders():
    ds=xr.open_dataset(FREEZE).load()
    if "chl_lin" not in ds:
        ds["chl_lin"]=np.exp(ds["log_chl"])-_FLOOR
    if "pixel_ok" not in ds:
        frac=np.isfinite(ds.log_chl).sum("time")/ds.sizes["time"]
        ds["pixel_ok"]=(frac>=.2).astype("uint8")
    pixel_ok=ds.pixel_ok.astype(bool)
    if "depth" in ds: pixel_ok = pixel_ok & (ds.depth>10.)
    t_all=np.arange(ds.sizes["time"]); times=ds.time.values
    tr=t_all[times<np.datetime64("2016-01-01")]
    va=t_all[(times>=np.datetime64("2016-01-01"))&(times<=np.datetime64("2018-12-31"))]
    te=t_all[times>np.datetime64("2018-12-31")]
    stats=norm_stats(ds,tr)
    tr_i,va_i,te_i=[idx[SEQ-1:-LEAD_IDX] for idx in (tr,va,te)]
    tr_ds,va_ds,te_ds=(PatchDS(ds,idx,stats,pixel_ok) for idx in (tr_i,va_i,te_i))
    sampler=None
    if STRATIFY:
        chl=(np.exp(ds.log_chl)-_FLOOR).isel(time=tr_i).where(pixel_ok).median(("lat","lon")).values
        q=np.nanquantile(chl,[.25,.5,.75]); bin_=np.digitize(chl,q)
        freq=np.maximum(np.bincount(bin_,minlength=4),1); w=(1/freq)**WEIGHT_EXP
        sampler=WeightedRandomSampler(torch.as_tensor(w[bin_]),len(bin_),replacement=True)
    tr_dl=DataLoader(tr_ds,BATCH,sampler=sampler,shuffle=sampler is None,num_workers=4,pin_memory=True,drop_last=True)
    va_dl=DataLoader(va_ds,BATCH,shuffle=False,num_workers=2,pin_memory=True)
    te_dl=DataLoader(te_ds,BATCH,shuffle=False,num_workers=2,pin_memory=True)
    return tr_dl,va_dl,te_dl,stats,q

# ─────────────────────────  MODEL – Temporal Fusion Transformer ─────────────
class GatedResidual(nn.Module):
    def __init__(self,d): super().__init__(); self.fc1=nn.Linear(d,d); self.fc2=nn.Linear(d,d); self.gate=nn.GLU()
    def forward(self,x): z=F.elu(self.fc1(x)); z=self.fc2(z); return self.gate(torch.cat([z,x],-1))

class TemporalFusionBlock(nn.Module):
    def __init__(self,d,h,n_heads):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,n_heads,batch_first=True,dropout=0.1)
        self.grn1=GatedResidual(d); self.grn2=GatedResidual(d)
        self.lin=nn.Linear(d,h); self.dropout=nn.Dropout(0.1)
    def forward(self,x):
        y,_=self.attn(x,x,x,need_weights=False)
        x=self.grn1(x+y)
        z=self.lin(x); z=F.elu(z); z=self.dropout(z); z=self.lin(z)
        return self.grn2(x+z)

class TFT(nn.Module):
    """
    Inputs  : (B, L, C, H, W)  – same as ConvLSTM
    Output  : (B, H, W)  Δ-log-chl
    Steps   : flatten spatial dims → Transformer along time → project back
    """
    def __init__(self,Cin,d_model=96,n_heads=8,n_blocks=3):
        super().__init__()
        self.spatial_patch = nn.Conv3d(Cin, d_model, kernel_size=(1,1,1))  # per-pixel channel lift
        self.blocks = nn.ModuleList([TemporalFusionBlock(d_model,d_model,n_heads) for _ in range(n_blocks)])
        self.head   = nn.Linear(d_model,1)
        self.kappa  = nn.Parameter(torch.tensor(25.,dtype=torch.float32))
    def forward(self,x):                    # x:(B,L,C,H,W)
        B,L,C,H,W = x.shape
        z = x.permute(0,2,1,3,4)            # (B,C,L,H,W)
        z = self.spatial_patch(z)           # (B,d,L,H,W)
        z = z.permute(0,3,4,2,1).contiguous()\
              .view(B*H*W, L, -1)           # pixels = batch, seq, d
        for blk in self.blocks: z = blk(z)
        out = self.head(z[:, -1])           # (pixels,1)
        out = out.view(B,H,W)               # Δ-log-chl
        return out

# ─────────────────────────────  LIGHTNING MODULE  ───────────────────────────
class HABTFT(pl.LightningModule):
    def __init__(self,stats,qthr,use_phys=True):
        super().__init__()
        Cin=len([v for v in ALL_VARS if v in xr.open_dataset(FREEZE).data_vars])
        self.net=TFT(Cin)
        self.stats=stats; self.qthr=qthr
        self.use_phys=use_phys
        self.save_hyperparameters(ignore=['stats','qthr'])
        self.scaler=GradScaler(enabled=torch.cuda.is_available())
    # physics helper (same as script)
    def physics_res(self,logCp,logC,u,v,κ,dx=4000.,dt=8*86400.):
        LapK = torch.tensor([[[[0,1,0],[1,-4,1],[0,1,0]]]],dtype=logCp.dtype,device=logCp.device)/(dx*dx)
        GxK  = torch.tensor([[[[-0.5,0,0.5]]]],dtype=logCp.dtype,device=logCp.device)/dx
        GyK  = torch.tensor([[[[-0.5],[0],[0.5]]]],dtype=logCp.dtype,device=logCp.device)/dx
        lap=lambda f:F.conv2d(f,LapK,padding=1)
        gx =lambda f:F.conv2d(f,GxK,padding=(0,1))
        gy =lambda f:F.conv2d(f,GyK,padding=(1,0))
        return (logCp-logC)/dt + u*gx(logCp) + v*gy(logCp) - κ*lap(logCp)

    def _hub_loss(self,err,m,y):
        chl_lin=torch.exp(y)
        w=torch.where(chl_lin<self.qthr[0],1.,
             torch.where(chl_lin<self.qthr[1],1.5,
                torch.where(chl_lin<self.qthr[2],2.5,4.)))
        w=w*m
        abs_err=err.abs(); quad=torch.minimum(abs_err,torch.tensor(HUBER_DELTA,device=err.device))
        lin=abs_err-quad; hub=0.5*quad**2 + HUBER_DELTA*lin
        return (w*hub).sum()/w.sum()

    def training_step(self,batch,batch_idx):
        X,y,m=batch
        last=X[:,-1,LOGCHL_IDX]
        delta=self.net(X)
        sup_err=(delta - (y-last)).masked_fill(~m,0.)
        sup_loss=self._hub_loss(sup_err,m,y)
        phys_loss=torch.tensor(0.,device=self.device)
        if self.use_phys and self.current_epoch>=20:
            logC ,logCp= last.unsqueeze(1), (last+delta).unsqueeze(1)
            u = X[:,-1,ALL_VARS.index("uo")].unsqueeze(1)
            v = X[:,-1,ALL_VARS.index("vo")].unsqueeze(1)
            res=self.physics_res(logCp,logC,u,v,self.net.kappa)
            phys_loss=((res[:,0][m])**2).mean()
            mu,sd=self.stats["log_chl"]; phys_loss*=sd**2
            lam=(sup_loss/phys_loss).clamp(0.1,1000.)
            ramp=min(max((self.current_epoch-20)/5,0),1)
            loss=sup_loss + lam*ramp*phys_loss
        else:
            loss=sup_loss
        self.log_dict({"train_loss":loss,"sup":sup_loss,"phys":phys_loss},prog_bar=True)
        return loss

    def validation_step(self,batch,batch_idx,dl_idx=0):
        X,y,m=batch
        pred=X[:,-1,LOGCHL_IDX]+self.net(X)
        err=(pred-y).masked_fill(~m,0)
        se=(err**2).sum(); n=m.sum()
        self.log("val_rmse",torch.sqrt(se/n),prog_bar=True)

    def configure_optimizers(self):
        opt=torch.optim.AdamW(self.parameters(),3e-4,weight_decay=1e-4)
        sch=ReduceLROnPlateau(opt,'min',patience=3,factor=.5)
        return {"optimizer":opt,"lr_scheduler": {"scheduler":sch,"monitor":"val_rmse"}}

# ──────────────────────────────────  TRAIN  ─────────────────────────────────
def main():
    tr_dl,va_dl,te_dl,stats,qthr=make_loaders()
    mod=HABTFT(stats,qthr,use_phys=True)
    ckpt=ModelCheckpoint(dirpath=OUT_DIR,filename="convTFT_best",monitor="val_rmse",
                         mode="min",save_top_k=1)
    trainer=pl.Trainer(max_epochs=EPOCHS,devices=1,precision="16-mixed" if torch.cuda.is_available() else 32,
                      callbacks=[EarlyStopping(monitor="val_rmse",patience=6),ckpt],
                      log_every_n_steps=50)
    trainer.fit(mod,tr_dl,va_dl)
    # evaluate
    mod.load_state_dict(torch.load(ckpt.best_model_path)["state_dict"])
    def _eval(dl):
        se=n=0.
        for X,y,m in dl:
            X,y,m=[t.to(mod.device) for t in (X,y,m)]
            with torch.no_grad(): p=X[:,-1,LOGCHL_IDX]+mod.net(X)
            err=(p-y).masked_fill(~m,0); se+=err.pow(2).sum().item(); n+=m.sum().item()
        return math.sqrt(se/n)
    metrics={"train":_eval(tr_dl),"val":_eval(va_dl),"test":_eval(te_dl)}
    print("FINAL RMSE_log:",metrics)
    json.dump(metrics,open(OUT_DIR/"convTFT_metrics.json","w"),indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq",    type=int, default=SEQ,
                    help=f"history length (default {SEQ})")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"max epochs (default {EPOCHS})")
    ap.add_argument("--batch",  type=int, default=BATCH,
                    help=f"batch size (default {BATCH})")
    args = ap.parse_args()

    # overwrite module-level constants with CLI overrides
    SEQ, EPOCHS, BATCH = args.seq, args.epochs, args.batch
    main()

'''
Results: 
FINAL RMSE_log: {'train': 0.7495690990669742, 'val': 0.6920505510658095, 'test': 0.788884756518899}  
'''