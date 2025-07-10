#!/usr/bin/env python3
"""
Compute baselines and diagnostics for HAB ConvLSTM forecasts **without** re-training.

1. Persistence baseline (t+1 = t)
2. Climatology baseline (multi‐year day‐of‐year mean)
3. Pred vs. obs scatter
4. Mean spatial residual map
5. Residual histogram & QQ‐plot
"""
from __future__ import annotations
import pathlib, yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.stats as st
import torch, torch.nn as nn

def main():
    # ───────── paths & config ─────────────────────────────────────────
    root     = pathlib.Path(__file__).resolve().parents[1]
    cfg      = yaml.safe_load(open(root/"config.yaml"))
    cube_fp  = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021_varspec_nostripes.nc"
    model_fp = root/"Models"/"convLSTM_best.pt"

    # ───────── load data & build test‐loader ─────────────────────────
    from data_cubes import make_loaders, LOGCHL_IDX, CHANS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds  = xr.open_dataset(cube_fp)
    chl = ds["log_chl"].transpose("time","lat","lon")

    # run baselines on the full grid
    test    = chl.sel(time=chl.time.dt.year==2021)
    truth   = test.isel(time=slice(1,None)).values
    persist = test.isel(time=slice(0,-1)).values
    mask_p  = np.isfinite(truth)&np.isfinite(persist)
    rmse_p  = np.sqrt(((persist-truth)[mask_p]**2).mean())
    print(f"Persistence RMSE = {rmse_p:.4f}")

    ds_all   = chl.sel(time=chl.time.dt.year<2021)
    doy_mean = ds_all.groupby("time.dayofyear").mean("time")
    doys     = test.time.dt.dayofyear.values
    clim_pred= np.stack([doy_mean.sel(dayofyear=int(d)).values for d in doys],0)
    clim_truth= test.values
    mask_c   = np.isfinite(clim_pred)&np.isfinite(clim_truth)
    rmse_c   = np.sqrt(((clim_pred-clim_truth)[mask_c]**2).mean())
    print(f"Climatology RMSE = {rmse_c:.4f}\n")

    # build test‐loader with no subprocesses
    _,_,te_dl = make_loaders(batch=8, workers=0)

    # define your model
    class PixelLSTM(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.cell = nn.LSTMCell(cin, cout)
            self.conv1= nn.Conv2d(cout, cout, 1)
        @staticmethod
        def _zeros(N,F,dtype,dev):
            z = torch.zeros(N,F,dtype=dtype,device=dev)
            return z.clone(), z.clone()
        def forward(self,x,hc=None):
            B,C,H,W = x.shape
            flat    = x.permute(0,2,3,1).reshape(B*H*W,C)
            if hc is None:
                hc = self._zeros(B*H*W, self.cell.hidden_size,
                                 flat.dtype, flat.device)
            h,c = self.cell(flat,hc)
            hmap= h.reshape(B,H,W,-1).permute(0,3,1,2)
            return self.conv1(hmap),(h,c)

    class ConvLSTMNet(nn.Module):
        def __init__(self,chans=CHANS):
            super().__init__()
            self.l1   = PixelLSTM(chans,32)
            self.l2   = PixelLSTM(32,64)
            self.head= nn.Conv2d(64,1,1)
        def forward(self,x):
            h1=h2=None
            for t in range(x.size(1)):
                out1,h1 = self.l1(x[:,t],h1)
                out2,h2 = self.l2(out1,h2)
            return self.head(out2).squeeze(1)

    # load model
    model = ConvLSTMNet().to(device)
    model.load_state_dict(torch.load(model_fp, map_location=device))
    model.eval()

    # run model over mini-cubes
    all_preds, all_truth, all_mask = [], [], []
    with torch.no_grad():
        for X,y,mask in te_dl:
            X = X.to(device)
            pers = X[:,-1,LOGCHL_IDX]
            dp   = model(X)
            preds= (pers+dp).cpu().numpy()
            all_preds.append(preds)
            all_truth.append(y.numpy())
            all_mask.append(mask.numpy()>0)

    mdl_pred = np.concatenate(all_preds,axis=0)
    truth    = np.concatenate(all_truth,axis=0)
    mask_m   = np.concatenate(all_mask,axis=0)
    resid    = mdl_pred - truth

    # ───────── Pred vs Obs scatter ─────────────────────────────────
    flat_t = truth[mask_m]
    flat_p = mdl_pred[mask_m]
    plt.figure(figsize=(6,6))
    plt.scatter(flat_t, flat_p, s=1, alpha=0.3)
    lims = [flat_t.min(), flat_t.max()]
    plt.plot(lims, lims, 'k--', lw=1)
    plt.xlabel("Observed log-chl"); plt.ylabel("Predicted log-chl")
    plt.title("Pred vs Obs (test year)")
    plt.savefig("pred_vs_obs_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ───────── 4. Mean spatial residual map (full-grid) ────────────────
    # wm is your water mask on the full grid
    wm = ds["water_mask"].values.astype(bool)    # shape (lat, lon)

    # compute full-grid truth & persistence over test year
    test = chl.sel(time=chl.time.dt.year == 2021)
    truth   = test.isel(time=slice(1, None)).values    # (T, lat, lon)
    persist = test.isel(time=slice(0, -1)).values      # (T, lat, lon)

    # full-grid mean bias (persist–truth)
    bias = np.nanmean(persist - truth, axis=0)         # (lat, lon)

    # mask out land
    bias_ocean = np.ma.masked_where(~wm, bias)

    # build lon/lat arrays
    Lon, Lat = np.meshgrid(ds["lon"], ds["lat"])      # both (lat, lon)

    plt.figure(figsize=(6,4))
    # ocean bias
    c = plt.pcolormesh(
        Lon, Lat, bias_ocean,
        cmap="RdBu_r", shading="auto",
        vmin=-np.nanmax(np.abs(bias_ocean)),
        vmax= np.nanmax(np.abs(bias_ocean)),
    )
    # overlay land as gray
    land = np.ma.masked_where(wm, wm)                 # land==True
    plt.pcolormesh(Lon, Lat, land, cmap="Greys", shading="auto", alpha=0.6)

    plt.gca().set_aspect("equal", "box")
    plt.colorbar(c, label="mean persistence bias (log-chl)")
    plt.title("Mean spatial persistence bias (test year)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.savefig("mean_persistence_bias_map.png", dpi=200, bbox_inches="tight")
    plt.close()


    # ───────── Residual hist & QQ ───────────────────────────────
    flat_res = resid[mask_m]
    plt.figure(); plt.hist(flat_res, bins=100, density=True)
    plt.title("Residual histogram"); plt.savefig("residual_hist.png", dpi=200, bbox_inches="tight"); plt.close()
    plt.figure(); st.probplot(flat_res, dist="norm", plot=plt)
    plt.title("QQ-plot"); plt.savefig("residual_qq.png", dpi=200, bbox_inches="tight"); plt.close()

    print("✅ Diagnostics complete.")

if __name__ == "__main__":
    main()
