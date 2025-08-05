#!/usr/bin/env python3
"""
(FINAL CODE WE USED WITH FINAL COMPOSITES)

ConvLSTM‑PINN **v1.3**  – stable recipe (low‑0.7 RMSE)
-----------------------------------------------------
This version reunites the hyper‑parameters that delivered
0.79 RMSE (v1.0) with the cleaner v1.2 architecture, plus a
simple fixed physics weight.  It routinely reaches 0.74 – 0.78
validation RMSE on the 2019‑2021 split with seed 42.

Key choices
===========
* SEQ = 6  (shorter history = lower variance)
* Batch = 32, **no stratified sampling** (kept the exact TTL
  distribution that worked before).
* optimiser: AdamW  –  LR 1e‑4  → ReduceLROnPlateau(patience = 2)
* λ_phys schedule:
      0 for epochs 1‑4,
      0.02 for epochs 5‑∞.
* Single‑stage training  (no stage‑2 hand‑off).

You can still override any hyper‑param from the CLI:
    python pinn/pinn_convLSTM_v1p3.py --seq 6 --epochs 40 --seed 17

The script auto‑saves the best checkpoint to
    ~/HAB_Models/best_pinn_v1p3.pt

# activate your HABs conda/venv first, then:
python pinn/pinn_convLSTM_v1p2.py \
  --freeze "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --epochs 40 \
  --seq 6 \
  --lead 1 \
  --batch 32 \
  --seed 42

After run, freeze:

# in repo root
mkdir -p archive/models
cp ~/HAB_Models/best_pinn_v1p3.pt archive/models/
cp /Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc archive/
git add archive && git commit -m "freeze v1.3 assets"
git tag v1.3-freeze

"""
from __future__ import annotations
import math, random, json, argparse, pathlib, numpy as np, xarray as xr, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast

# ───────────────────────── default hyper‑params ───────────────────────────
FREEZE = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
SEQ, LEAD_IDX, PATCH = 6, 1, 64           # ❶ shorter history, same horizon
BATCH, EPOCHS, SEED = 32, 40, 42
INIT_LR, MIN_LR = 1e-4, 1e-6              # ❷ stable LR range
STRATIFY = False                          # ❸ keep natural class freq.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIXED  = DEVICE.type == "cuda"
SCALER = GradScaler(enabled=MIXED)
OUT_DIR = pathlib.Path.home()/"HAB_Models"; OUT_DIR.mkdir(exist_ok=True)

DX = 4000.0; DT = LEAD_IDX*8*86_400; FLOOR = 0.056616

# ───────────────────────── variable lists (unchanged) ─────────────────────
SAT = ["log_chl","Kd_490","nflh"]
MET  = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCE  = ["uo","vo","cur_speed","cur_div","cur_vort","zos","ssh_grad_mag","so","thetao"]
DER  = ["chl_anom_monthly","chl_roll24d_mean","chl_roll24d_std","chl_roll40d_mean","chl_roll40d_std"]
STAT = ["river_rank","dist_river_km","ocean_mask_static"]
ALL  = SAT+MET+OCE+DER+STAT
IDX_LOG, IDX_U, IDX_V = SAT.index("log_chl"), ALL.index("uo"), ALL.index("vo")
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ------------------------------------------------------------------ #
# Predictor (INPUT) list – notice log_chl is *excluded*
# ------------------------------------------------------------------ #
ALL_INPUT = [v for v in ALL if v != "log_chl"]      # 29 channels
IDX_U_IN = ALL_INPUT.index("uo")   # u-current in X[:, :, IDX_U_IN, ...]
IDX_V_IN = ALL_INPUT.index("vo")   # v-current in X[:, :, IDX_V_IN, ...]

λ_FIXED = 0.02          # physics weight after warm‑up
WARM_EPOCHS = 4

# ───────────────────────── normalisation helpers ──────────────────────────
def stats(ds, tids):
    s={}
    for v in ALL:
        if v not in ds: continue
        da = ds[v] if "time" not in ds[v].dims else ds[v].isel(time=tids)
        mu,sd=float(da.mean(skipna=True)), float(da.std(skipna=True)) or 1.
        s[v]=(mu,sd)
    return s

def z(x,mu_sd): mu,sd=mu_sd; return (x-mu)/sd

def un_z(t,mu_sd): mu,sd=mu_sd; return t*sd+mu

# ───────────────────────── PatchDS ──────────────────────────
class PatchDS(Dataset):
    """
    • Training  → random patches      (randomise=True,  ys/xs=None)
    • Val/Test  → deterministic grid  (randomise=False, ys/xs pre-computed)
    """
    def __init__(
        self,
        ds: xr.Dataset,
        tids: np.ndarray,
        st: dict[str, tuple[float, float]],
        mask2d: xr.DataArray | np.ndarray,
        *,
        ys: np.ndarray | None = None,
        xs: np.ndarray | None = None,
        randomise: bool = True,
    ):
        self.ds      = ds
        self.tids    = np.asarray(tids)
        self.st      = st
        self.mask2d  = mask2d        # (lat, lon) boolean
        self.patch   = PATCH
        self.random  = randomise or (ys is None)

        # deterministic corner lists (val / test)
        self.yc = ys
        self.xc = xs

        # lists used only for random sampling (train)
        if self.random:
            self.latL = np.arange(0, ds.sizes["lat"] - self.patch + 1)
            self.lonL = np.arange(0, ds.sizes["lon"] - self.patch + 1)
            self.rng  = np.random.default_rng(SEED + len(tids))

    # --------------------------------------------------------
    def __len__(self) -> int:
        if self.yc is not None:                # grid mode
            return len(self.tids) * len(self.yc)
        return len(self.tids)

    # --------------------------------------------------------
    def _corner_random(self) -> tuple[int, int]:
        """Randomly pick a patch corner that contains ≥ 1 valid pixel."""
        for _ in range(50):                    # try 50 random spots first
            y = int(self.rng.choice(self.latL))
            x = int(self.rng.choice(self.lonL))
            if self.mask2d.isel(lat=slice(y, y+self.patch),
                                lon=slice(x, x+self.patch)).any():
                return y, x
        # fallback exhaustive scan (rare)
        for y in self.latL:
            for x in self.lonL:
                if self.mask2d.isel(lat=slice(y, y+self.patch),
                                    lon=slice(x, x+self.patch)).any():
                    return y, x
        raise RuntimeError("No valid patch found – mask too strict?")

    # --------------------------------------------------------
    def __getitem__(self, idx: int):
        # --- choose (t, y, x) --------------------------------------------
        if self.yc is not None:                # deterministic grid
            n_corners = len(self.yc)
            t_idx = idx // n_corners
            c_idx = idx %  n_corners
            t = int(self.tids[t_idx])
            y = int(self.yc[c_idx])
            x = int(self.xc[c_idx])
        else:                                  # random patch (train)
            t = int(self.tids[idx])
            y, x = self._corner_random()

        # --- build input tensor (SEQ, C, H, W) ----------------------------
        frames = []
        for dt in range(SEQ):
            bands = []
            ti = t - SEQ + 1 + dt
            for v in ALL_INPUT:                # log_chl is *excluded*
                if v not in self.ds:
                    continue
                da = self.ds[v]
                if "time" in da.dims:
                    da = da.isel(time=ti,
                                 lat=slice(y, y+self.patch),
                                 lon=slice(x, x+self.patch))
                else:
                    da = da.isel(lat=slice(y, y+self.patch),
                                 lon=slice(x, x+self.patch))

                # ── CLEAN & CLIP RAW VALUES ────────────────────────────
                arr = da.values.astype(np.float32)
                arr[~np.isfinite(arr)] = np.nan                # kill ±∞
                z_arr = z(arr, self.st[v])                     # z-score
                z_arr = np.clip(z_arr, -10.0, 10.0)            # tame outliers
                bands.append(np.nan_to_num(z_arr, nan=0.0))    # final fill

            frames.append(np.stack(bands, 0))
        X = torch.from_numpy(np.stack(frames, 0).astype(np.float32))
        X = torch.nan_to_num(X, 0.0, 0.0, 0.0)                  # extra belt

        # --- targets & mask ----------------------------------------------
        y_t = self.ds["log_chl"].isel(time=t+LEAD_IDX,
                                      lat=slice(y, y+self.patch),
                                      lon=slice(x, x+self.patch)).values
        logC_prev = self.ds["log_chl"].isel(time=t,
                                            lat=slice(y, y+self.patch),
                                            lon=slice(x, x+self.patch)).values
        m = (self.mask2d.isel(lat=slice(y, y+self.patch),
                              lon=slice(x, x+self.patch)).values
             & np.isfinite(y_t))

        return (
            X,
            torch.from_numpy(y_t.astype(np.float32)),
            torch.from_numpy(m),
            torch.from_numpy(logC_prev.astype(np.float32)),
        )
# ───────────────────────── end PatchDS ──────────────────────────

# Helper to pre-compute valid grid corners (unchanged)
def valid_corners(mask2d: np.ndarray, patch: int):
    ys, xs = [], []
    for y in range(0, mask2d.shape[0] - patch + 1, patch):
        for x in range(0, mask2d.shape[1] - patch + 1, patch):
            if mask2d[y:y+patch, x:x+patch].any():
                ys.append(y); xs.append(x)
    return np.array(ys), np.array(xs)

def make_loaders():
    ds=xr.open_dataset(FREEZE); ds.load()
    if "pixel_ok" not in ds:
        ok=(np.isfinite(ds.log_chl).sum("time")/ds.sizes["time"])>=0.2
        ds["pixel_ok"]=ok.astype("uint8")
    mask = ds.pixel_ok.astype(bool)
    ys, xs = valid_corners(mask.values, PATCH)

    t_all=np.arange(ds.sizes["time"]); times=ds.time.values
    tr=t_all[times<np.datetime64("2016-01-01")]
    va=t_all[(times>=np.datetime64("2016-01-01"))&(times<=np.datetime64("2018-12-31"))]
    te=t_all[times>np.datetime64("2018-12-31")]

    st=stats(ds,tr)

    def _ds(idx, randomise):
        return PatchDS(
            ds, idx[SEQ-1:-LEAD_IDX], st, mask,
            ys=ys, xs=xs,  randomise=randomise
        )

    tr_ds = PatchDS(ds, tr[SEQ-1:-LEAD_IDX], st, mask,
                    randomise=True)              # random patches

    va_ds = PatchDS(ds, va[SEQ-1:-LEAD_IDX], st, mask,
                    ys=ys, xs=xs, randomise=False)

    te_ds = PatchDS(ds, te[SEQ-1:-LEAD_IDX], st, mask,
                    ys=ys, xs=xs, randomise=False)

    tr_dl=DataLoader(tr_ds,BATCH,shuffle=True,num_workers=0,drop_last=True)
    va_dl=DataLoader(va_ds,BATCH,shuffle=False)
    te_dl=DataLoader(te_ds,BATCH,shuffle=False)
    return tr_dl,va_dl,te_dl,st

# ───────────────────────── model ──────────────────────────────────────────
class PxLSTM(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.cell = nn.LSTMCell(ci, co)
        self.conv = nn.Conv2d(co, co, 1)

    def forward(self, x, hc=None):
        B, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        if hc is None:                      # ← FIX begins
            h = torch.zeros(
                flat.size(0),               # B·H·W
                self.cell.hidden_size,      # 48
                device=flat.device,
                dtype=flat.dtype
            )
            hc = (h, h.clone())             # (hₜ₋₁, cₜ₋₁)
        # ← FIX ends

        h, c  = self.cell(flat, hc)
        h_map = h.view(B, H, W, -1).permute(0, 3, 1, 2)
        return self.conv(h_map), (h, c)


_KDX=torch.tensor([[0,0,0],[-1,0,1],[0,0,0]],dtype=torch.float32)/(2*DX)
_KDY=torch.tensor([[0,-1,0],[0,0,0],[0,1,0]],dtype=torch.float32)/(2*DX)
_KLAP=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32)/(DX**2)
KDX=_KDX.view(1,1,3,3).to(DEVICE); KDY=_KDY.view(1,1,3,3).to(DEVICE); KLAP=_KLAP.view(1,1,3,3).to(DEVICE)

def _conv(f,k):
    pad=F.pad(f.unsqueeze(1),(1,1,1,1),mode="replicate")
    return F.conv2d(pad,k).squeeze(1)

def phys_res(logCp,logC,u,v,kappa):
    return (logCp-logC)/DT + u*_conv(logCp,KDX) + v*_conv(logCp,KDY) - kappa*_conv(logCp,KLAP)

class ConvLSTM(nn.Module):
    def __init__(self,Cin):
        super().__init__(); self.kappa=nn.Parameter(torch.tensor(25.))
        self.reduce=nn.Conv2d(Cin,24,1); self.l1,self.l2=PxLSTM(24,48),PxLSTM(48,64)
        self.delta=nn.Conv2d(64,1,1); self.src=nn.Conv2d(64,1,1,bias=False)
    def forward(self,x):
        h1=h2=None
        for t in range(x.size(1)):
            o,h1=self.l1(self.reduce(x[:,t]),h1)
            o,h2=self.l2(o,h2)
        return self.delta(o).squeeze(1), self.src(o).squeeze(1)

# ───────────────────────── loss / metric ─────────────────────────────────
@torch.no_grad()
def rmse(dl,net):
    net.eval(); se=n=0
    for X,y,m,logC_prev in dl:
        X,y,m,logC_prev=[t.to(DEVICE) for t in (X,y,m,logC_prev)]
        with autocast(device_type=DEVICE.type, enabled=MIXED):
            d,_=net(X); pred = logC_prev + d 

        valid_pix = m.sum().item()
        if valid_pix == 0:
            continue
        err = (pred-y).masked_fill_(~m,0)
        se += (err**2).sum().item()
        n  += valid_pix

    return math.sqrt(se/n)

# ───────────────────────── training step ────────────────────────────────
HUBER_DELTA=1.0

def huber(e,δ):
    q=torch.minimum(e.abs(),torch.tensor(δ,device=e.device)); return 0.5*q**2+δ*(e.abs()-q)

def train_one(dl, net, opt, epoch, mu_u, sd_u, mu_v, sd_v):
    λ=0.0 if epoch<=WARM_EPOCHS else λ_FIXED
    net.train(); run_d=run_p=0.
    for X,y,m, logC_prev in dl:
        X,y,m,logC_prev=[t.to(DEVICE) for t in (X,y,m,logC_prev)]
        with autocast(device_type=DEVICE.type, enabled=MIXED):
            d,S=net(X) # logC=X[:,-1,IDX_LOG]
            logCp  = logC_prev + d

            # ── guard: skip batches with zero valid pixels ─────────────────
            valid_pix = m.sum()
            if valid_pix == 0:
                continue                    # drop this batch safely

            data_loss = huber((logCp-y).masked_fill_(~m,0), HUBER_DELTA).sum() / valid_pix

            u, v = X[:,-1, IDX_U_IN], X[:,-1, IDX_V_IN] # ← uses correct channels
            u   = un_z(u, (mu_u, sd_u))
            v   = un_z(v, (mu_v, sd_v))
            phys = huber((phys_res(logCp, logC_prev, u, v, net.kappa)-S).abs()*m, 3.0).mean()
            loss=data_loss+λ*phys
        SCALER.scale(loss).backward(); SCALER.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(),0.8)
        SCALER.step(opt); SCALER.update(); opt.zero_grad()
        run_d+=data_loss.item(); run_p+=phys.item()
    return run_d/len(dl), run_p/len(dl), λ

# ───────────────────────── main ─────────────────────────────────────────

def main(argv=None):
    global FREEZE,SEQ,LEAD_IDX,BATCH,SEED
    ap=argparse.ArgumentParser()
    ap.add_argument("--freeze",type=pathlib.Path,default=FREEZE)
    ap.add_argument("--epochs",type=int,default=EPOCHS)
    ap.add_argument("--seq",type=int,default=SEQ)
    ap.add_argument("--lead",type=int,default=LEAD_IDX)
    ap.add_argument("--batch",type=int,default=BATCH)
    ap.add_argument("--seed",type=int,default=SEED)
    args=ap.parse_args(argv)
    # -- inside main() after arg parsing --------------------------
    FREEZE = args.freeze
    SEQ    = args.seq
    LEAD_IDX = args.lead
    BATCH  = args.batch
    SEED   = args.seed
    global DT          # add this
    DT     = LEAD_IDX * 8 * 86_400   # update horizon

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    tr_dl, va_dl, te_dl, st = make_loaders()
    mu_u, sd_u = st["uo"];  mu_v, sd_v = st["vo"]
    # Cin = len([v for v in ALL if v in xr.open_dataset(FREEZE).data_vars])
    Cin = len(ALL_INPUT)
    net = ConvLSTM(Cin).to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),lr=INIT_LR,weight_decay=1e-4)
    plate = ReduceLROnPlateau(opt, mode="min", factor=0.5,
                            patience=2, min_lr=MIN_LR, threshold=1e-3)

    print("Training …", flush=True)
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        d_loss, p_loss, lam = train_one(tr_dl, net, opt, ep, mu_u, sd_u, mu_v, sd_v)
        val_rm = rmse(va_dl, net)
        plate.step(val_rm)
        print(f"E{ep:02d}  val RMSE_log={val_rm:.3f}  λ={lam:.2f}  "
            f"lr={opt.param_groups[0]['lr']:.1e}")

        if val_rm < best - 1e-3:
            best = val_rm
            torch.save(net.state_dict(), OUT_DIR / "best_pinn_v1p3.pt")

    # ─────────────────────── evaluation & export ────────────────────────────
    net.load_state_dict(torch.load(OUT_DIR / "best_pinn_v1p3.pt"))
    metrics = {
        "train": rmse(tr_dl, net),
        "val":   rmse(va_dl, net),
        "test":  rmse(te_dl, net),
    }
    print("FINAL RMSE_log:", metrics)
    json.dump(metrics, open(OUT_DIR / "pinn_v1p3_metrics.json", "w"), indent=2)

if __name__ == "__main__":
    main()

