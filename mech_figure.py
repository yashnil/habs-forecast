#!/usr/bin/env python3
"""
Domain-wide robustness figure (NO case study):

(A) Scale-dependent RMSE vs wavelength (ConvLSTM/TFT/PINN)
(B) Power-spectrum ratio P_model / P_obs vs wavelength
(C) RMSE vs distance-to-coast deciles
(D) Advection–diffusion residual distributions (all models)

Example:
chmod +x run_mech.sh
./run_mech.sh
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union, transform as shp_transform
from shapely.prepared import prep as shp_prep
from scipy.ndimage import distance_transform_edt, binary_erosion, gaussian_filter
from pyproj import Transformer
import argparse, os, warnings
warnings.filterwarnings("ignore")

# ---------------- prefs / assumptions ----------------
PREF_VARS_OBS  = ("log_chl",)
PREF_VARS_PRED = ("log_chl_pred","log_chl")
PREF_UV        = ("uo","vo")           # surface currents in obs file
PINN_KAPPA     = 25.0                  # m^2 s^-1 for residual calc
DT_SECONDS     = 8*24*3600             # 8-day cadence
COAST_MILES    = 10                    # mask within 10 miles of coastline

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})

# ---------------- utils ----------------
EPS_X = 1e-6     # for wavelengths / wavenumbers
EPS_Y = 1e-12    # for spectra / RMSE / ratios / residuals

def _finite_pos(x, y, epsx=EPS_X, epsy=EPS_Y):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size:
        x = np.clip(x, epsx, None)
    if y.size:
        y = np.clip(y, epsy, None)
    return x, y

def _try_log_axes(ax, x_has_pos, y_has_pos, x_fallback='linear', y_fallback='linear'):
    # Only set log if we truly have positive data after clipping
    ax.set_xscale('log' if x_has_pos else x_fallback)
    ax.set_yscale('log' if y_has_pos else y_fallback)

def _safe_logplot(ax, x, y, **kwargs):
    x, y = _finite_pos(x, y)
    x_has = (x.size > 0) and np.isfinite(x).any() and (np.nanmax(x) > 0)
    y_has = (y.size > 0) and np.isfinite(y).any() and (np.nanmax(y) > 0)
    _try_log_axes(ax, x_has, y_has)
    if x_has and y_has:
        ax.plot(x, y, **kwargs)
    else:
        ax.plot([], [])  # keep artist list sane
        ax.text(0.5, 0.5, "no positive data", ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='0.4')

def choose_var(ds, prefer):
    for v in prefer:
        if v in ds.data_vars: return v
    # fallback: first float var with lat/lon
    for k,v in ds.data_vars.items():
        if np.issubdtype(v.dtype, np.floating) and {"lat","lon"}.issubset(v.dims):
            return k
    raise ValueError(f"No suitable var in {list(ds.data_vars)}")

def meters_per_degree(lat_deg):
    lat = np.deg2rad(lat_deg)
    m_per_deg_lat = 111_132.954 - 559.822*np.cos(2*lat) + 1.175*np.cos(4*lat)
    m_per_deg_lon = 111_132.954*np.cos(lat)
    return float(m_per_deg_lon), float(m_per_deg_lat)

def grid_metrics(lat, lon):
    lat0 = 0.5*(lat.min()+lat.max())
    mx, my = meters_per_degree(lat0)
    dx = mx * float(abs(lon[1]-lon[0]))
    dy = my * float(abs(lat[1]-lat[0]))
    return dx, dy

def grad_centered(F, dx, dy):
    dFdx = np.zeros_like(F); dFdy = np.zeros_like(F)
    dFdx[:,1:-1] = (F[:,2:]-F[:,:-2])/(2*dx)
    dFdy[1:-1,:] = (F[2:,:]-F[:-2,:])/(2*dy)
    dFdx[:,0]  = (F[:,1]-F[:,0])/dx
    dFdx[:,-1] = (F[:,-1]-F[:,-2])/dx
    dFdy[0,:]  = (F[1,:]-F[0,:])/dy
    dFdy[-1,:] = (F[-1,:]-F[-2,:])/dy
    return dFdx, dFdy

def laplacian(F, dx, dy):
    L = np.zeros_like(F)
    L[1:-1,1:-1] = ((F[1:-1,2:]-2*F[1:-1,1:-1]+F[1:-1,:-2])/(dx*dx) +
                    (F[2:,1:-1]-2*F[1:-1,1:-1]+F[:-2,1:-1])/(dy*dy))
    L[:,0]=L[:,1]; L[:,-1]=L[:,-2]; L[0,:]=L[1,:]; L[-1,:]=L[-2,:]
    return L

def ade_residual(C_prev, C_next, u, v, dx, dy, dt, kappa=PINN_KAPPA, smooth_sigma=0.6):
    Cn = gaussian_filter(C_next, sigma=smooth_sigma) if smooth_sigma else C_next
    dCdt = (Cn - C_prev)/dt
    Cx, Cy = grad_centered(Cn, dx, dy)
    adv = u*Cx + v*Cy
    diff = -kappa*laplacian(Cn, dx, dy)
    res = np.abs(dCdt + adv + diff)
    res[~np.isfinite(res)] = np.nan
    return np.clip(res, EPS_Y, None)   # <= guarantees strictly positive values

def inpaint_nearest(arr, valid_mask):
    out = np.array(arr, float, copy=True)
    miss = np.isnan(out) & valid_mask
    if not miss.any(): return out
    _, (iy,ix) = distance_transform_edt(~(np.isfinite(out)&valid_mask), return_indices=True)
    out[miss] = out[iy[miss], ix[miss]]
    return out

def inpaint_any(arr):
    """Fill NaNs everywhere (not just within a mask) using nearest neighbor."""
    arr = np.array(arr, float, copy=True)
    valid = np.isfinite(arr)
    if not valid.any():
        return np.zeros_like(arr)
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    arr[~valid] = arr[iy[~valid], ix[~valid]]
    return arr

def fill_for_fft(field, valid_mask):
    """Fill outside-coast NaNs with nearest valid coastal value (good for FFT)."""
    F = np.array(field, float, copy=True)
    valid = np.isfinite(F) & valid_mask
    if not valid.any():
        return np.nan_to_num(F, nan=0.0)
    # nearest-neighbor inpaint to the *entire* grid
    _, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    F[~valid] = F[iy[~valid], ix[~valid]]
    return F

def radial_psd(field, dx, dy, detrend=True, window=True, nbins=30):
    F = np.array(field, float)
    if detrend: F -= np.nanmean(F)
    if window:
        wx = np.hanning(F.shape[1]); wy = np.hanning(F.shape[0])
        F = (F * wy[:,None]) * wx[None,:]
    ny, nx = F.shape
    kx = np.fft.fftfreq(nx, d=dx)*2*np.pi
    ky = np.fft.fftfreq(ny, d=dy)*2*np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kk = np.sqrt(KX**2 + KY**2)
    S = (np.abs(np.fft.fft2(F))**2) * dx*dy / (nx*ny)
    kmax = np.nanmax(kk); bins = np.linspace(0, kmax, nbins+1)
    kr = 0.5*(bins[:-1]+bins[1:])
    psd = np.full_like(kr, np.nan, dtype=float)
    for i in range(nbins):
        m = (kk>=bins[i]) & (kk<bins[i+1])
        if m.any(): psd[i] = np.nanmean(S[m])
    wl = np.where(kr>0, 2*np.pi/kr, np.nan)/1000.0  # km

    psd = np.maximum(psd, EPS_Y)
    return wl, psd

def reliable_lambda_min_km(dx, dy, factor=4.0):
    # rule of thumb: don’t trust wavelengths smaller than ~4 grid cells
    return factor * min(dx, dy) / 1000.0

def log_slope(x_wl_km, y, lam_min, lam_max):
    m = (x_wl_km >= lam_min) & (x_wl_km <= lam_max) & np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return np.nan
    X = np.log10(x_wl_km[m]); Y = np.log10(y[m])
    return np.polyfit(X, Y, 1)[0]  # slope in log10–log10


def build_coast_mask_and_distance(lon, lat, coast_miles=10, erode_px=1):
    """Return mask (within ≤coast_miles) and distance-to-coast (meters) for ocean cells."""
    land = unary_union(list(shpreader.Reader(
        shpreader.natural_earth("10m","physical","land")
    ).geometries()))
    land_p = shp_prep(land)
    to_m = Transformer.from_crs("EPSG:4326","EPSG:32610", always_xy=True).transform
    land_m = shp_transform(to_m, land)
    coast_m = land_m.boundary
    thresh_m = coast_miles*1609.344

    LAT, LON = np.meshgrid(lat, lon, indexing="ij") if (lat.ndim==1 and lon.ndim==1) else (lat, lon)
    H, W = LAT.shape
    within = np.zeros((H,W), bool)
    dist_to_coast = np.full((H,W), np.nan, float)

    for i in range(H):
        for j in range(W):
            lo, la = float(LON[i,j]), float(LAT[i,j])
            if land_p.contains(Point(lo,la)):
                continue
            x,y = to_m(lo,la)
            d = Point(x,y).distance(coast_m)
            dist_to_coast[i,j] = d
            if d <= thresh_m:
                within[i,j] = True

    if erode_px>0:
        within = binary_erosion(within, iterations=int(erode_px))
    return within, dist_to_coast

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--pred", action="append", required=True, help="file:model (e.g., /p/pinn.nc:PINN)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="fig_robustness.png")
    args = ap.parse_args()

    ds_obs = xr.open_dataset(args.obs)
    if PREF_UV[0] not in ds_obs or PREF_UV[1] not in ds_obs:
        raise ValueError("Obs file must contain surface currents 'uo','vo'.")

    var_obs = choose_var(ds_obs, PREF_VARS_OBS)
    da_obs  = ds_obs[var_obs].sel(time=slice(args.start, args.end))
    da_u    = ds_obs[PREF_UV[0]].sel(time=slice(args.start, args.end))
    da_v    = ds_obs[PREF_UV[1]].sel(time=slice(args.start, args.end))

    lat = da_obs["lat"].values if "lat" in da_obs.dims else da_obs["latitude"].values
    lon = da_obs["lon"].values if "lon" in da_obs.dims else da_obs["longitude"].values
    dx, dy = grid_metrics(lat, lon)

    # Coast mask + distance
    print("Building coastal mask (≤10 miles) and distance-to-coast…")
    coast_mask, dist2coast_m = build_coast_mask_and_distance(lon, lat, coast_miles=COAST_MILES, erode_px=1)

    # Load models aligned to obs times
    models = {}
    for item in args.pred:
        path, name = item.split(":")
        ds = xr.open_dataset(path)
        var = choose_var(ds, PREF_VARS_PRED)
        da = ds[var].sel(time=slice(args.start, args.end))
        # align exact intersection of timestamps
        common_t = np.array(sorted(set(da_obs.time.values) & set(da.time.values)))
        if common_t.size == 0:
            raise ValueError(f"No overlapping times with {name}")
        models[name] = da.sel(time=common_t).sortby("time")
    # align obs/currents to common times of ALL models
    common_all = set(da_obs.time.values)
    for m in models.values():
        common_all &= set(m.time.values)
    common_all = np.array(sorted(common_all))
    da_obs = da_obs.sel(time=common_all)
    da_u   = da_u.sel(time=common_all)
    da_v   = da_v.sel(time=common_all)
    for k in list(models.keys()):
        models[k] = models[k].sel(time=common_all)

    times = da_obs.time.values
    if times.size < 3:
        raise ValueError("Need >=3 time steps in the chosen window.")
    
    print(f"Coastal mask cells: {int(coast_mask.sum())} / {coast_mask.size}")
    print(f"Common timesteps: {len(common_all)}")

    # -------- accumulate stats over time --------
    # (A) scale-dependent RMSE: accumulate PSD of error each time, average, then take sqrt
    nbins = 30
    spectra_err = {k: {"ps_sum": np.zeros(nbins), "n":0, "wl":None} for k in models}

    # For panel A normalization
    spectra_obs = {"ps_sum": np.zeros(nbins), "n": 0, "wl": None}

    # For panel D “physics floor”
    obs_resid_chunks = []

    # (B) power-spectrum ratio: compute later from time-mean fields
    # (C) RMSE vs distance-to-coast deciles
    ocean_vals = dist2coast_m[coast_mask]
    deciles = np.nanpercentile(ocean_vals, [10,20,30,40,50,60,70,80,90])
    def decile_index(d):
        # returns bin idx 0..9 (0=nearest coast)
        return np.digitize(d, deciles, right=True)

    rmse_bins = {k: {"se":np.zeros(10), "n":np.zeros(10)} for k in models}

    # (D) ADE residual distributions per model
    resid_samples = {k: [] for k in models}

    for ti in range(1, len(times)):  # need t-1
        t  = times[ti]
        tm = times[ti-1]
        obs_t  = np.where(coast_mask, da_obs.sel(time=t).values,  np.nan)
        obs_tm = np.where(coast_mask, da_obs.sel(time=tm).values, np.nan)
        u_t    = np.where(coast_mask, da_u.sel(time=t).values,    np.nan)
        v_t    = np.where(coast_mask, da_v.sel(time=t).values,    np.nan)

        # inpaint for derivatives/FFT stability
        obs_t = inpaint_nearest(obs_t, coast_mask)

        # currents: try mask-aware inpaint, then global fallback, then zeros if still bad
        u_t = inpaint_nearest(u_t, coast_mask)
        v_t = inpaint_nearest(v_t, coast_mask)
        if not np.isfinite(u_t).any():
            u_t = inpaint_any(u_t)      # fill from anywhere (including outside mask)
        if not np.isfinite(v_t).any():
            v_t = inpaint_any(v_t)
        if not np.isfinite(u_t).any():
            u_t = np.zeros_like(u_t)    # hard fallback
        if not np.isfinite(v_t).any():
            v_t = np.zeros_like(v_t)


        # --- Panel A normalization: observed PSD at time t
        obs_fft = fill_for_fft(obs_t, coast_mask)
        wl_o_t, ps_o_t = radial_psd(obs_fft, dx, dy, detrend=True, window=True, nbins=nbins)
        if spectra_obs["wl"] is None:
            spectra_obs["wl"] = wl_o_t
        spectra_obs["ps_sum"] += np.nan_to_num(ps_o_t)
        spectra_obs["n"]      += 1


        for name, mda in models.items():
            mdl_t  = np.where(coast_mask, mda.sel(time=t).values,  np.nan)
            mdl_tm = np.where(coast_mask, mda.sel(time=tm).values, np.nan)
            mdl_t  = inpaint_nearest(mdl_t,  coast_mask)
            mdl_tm = inpaint_nearest(mdl_tm, coast_mask)

            # (A) error field spectrum
            err = mdl_t - obs_t
            err_fft = fill_for_fft(err, coast_mask)
            wl, ps = radial_psd(err_fft, dx, dy, detrend=True, window=True, nbins=nbins)
            if spectra_err[name]["wl"] is None: spectra_err[name]["wl"] = wl
            spectra_err[name]["ps_sum"] += np.nan_to_num(ps)
            spectra_err[name]["n"]      += 1

            # (C) RMSE vs distance-to-coast deciles
            se = (err**2)
            bidx = decile_index(dist2coast_m)
            for b in range(10):
                mask_b = coast_mask & (bidx==b) & np.isfinite(se)
                if mask_b.any():
                    rmse_bins[name]["se"][b] += np.nansum(se[mask_b])
                    rmse_bins[name]["n"][b]  += np.sum(mask_b)

            # (D) ADE residual using model C (physics consistency)
            # make all inputs globally finite so derivatives don’t see NaNs at the mask edge
            Cprev = fill_for_fft(mdl_tm, coast_mask)
            Cnext = fill_for_fft(mdl_t,  coast_mask)
            U     = fill_for_fft(u_t,    coast_mask)
            V     = fill_for_fft(v_t,    coast_mask)

            R = ade_residual(Cprev, Cnext, U, V, dx, dy, DT_SECONDS,
                            kappa=PINN_KAPPA, smooth_sigma=0.6)

            print(f"[DBG] {name} residual min/max (all):", np.nanmin(R), np.nanmax(R))
            rvals = R[coast_mask].ravel()
            rvals = rvals[np.isfinite(rvals)]
            if rvals.size:
                rvals = np.clip(rvals, EPS_Y, None)  # keep strictly positive for log scale
                k = min(5000, rvals.size)
                resid_samples[name].append(np.random.choice(rvals, size=k, replace=False))
            
        # --- Panel D: obs “physics floor” residual (model = obs)
        R_obs = ade_residual(obs_tm, obs_t, u_t, v_t, dx, dy, DT_SECONDS,
                            kappa=PINN_KAPPA, smooth_sigma=0.6)
        r_obs = R_obs[coast_mask].ravel()
        r_obs = r_obs[np.isfinite(r_obs)]
        if r_obs.size:
            r_obs = np.clip(r_obs, EPS_Y, None)
            k = min(5000, r_obs.size)
            obs_resid_chunks.append(np.random.choice(r_obs, size=k, replace=False))

    # finalize (A): relative scale error = sqrt(<PS_err> / <PS_obs>)
    ps_obs_mean = spectra_obs["ps_sum"] / max(spectra_obs["n"], 1)
    ps_obs_mean = np.maximum(ps_obs_mean, EPS_Y)

    for name in models:
        ps_mean = spectra_err[name]["ps_sum"] / max(spectra_err[name]["n"], 1)
        ps_mean = np.maximum(ps_mean, EPS_Y)
        spectra_err[name]["amp_rel"] = np.sqrt(ps_mean / ps_obs_mean)
        spectra_err[name]["wl"] = spectra_obs["wl"]  # ensure same x

    # (B) spectral ratio from time-mean fields
    obs_mean = np.nanmean(np.where(coast_mask[None,:,:], da_obs.values, np.nan), axis=0)
    obs_mean = fill_for_fft(obs_mean, coast_mask)

    spectra_ratio = {}
    for name, mda in models.items():
        mdl_mean = np.nanmean(np.where(coast_mask[None, :, :], mda.values, np.nan), axis=0)
        mdl_mean = inpaint_any(mdl_mean)  # <<< fully finite

        wl_m, ps_m = radial_psd(mdl_mean, dx, dy, detrend=True, window=True, nbins=nbins)
        wl_o, ps_o = radial_psd(obs_mean, dx, dy, detrend=True, window=True, nbins=nbins)
        ps_m = np.maximum(ps_m, EPS_Y)
        ps_o = np.maximum(ps_o, EPS_Y)
        ratio = np.maximum(ps_m / ps_o, EPS_Y)  # strictly positive for log plotting
        spectra_ratio[name] = (wl_m, ratio)
    
    # Scale Bias Index over trustworthy scales (lower is better)
    lam_min = reliable_lambda_min_km(dx, dy, factor=4.0)
    lam_max = 150.0  # km
    sbi = {}
    for name, (wl, ratio) in spectra_ratio.items():
        m = (wl >= lam_min) & (wl <= lam_max) & np.isfinite(ratio) & (ratio > 0)
        if m.sum() >= 3:
            x = np.log10(wl[m]); y = np.abs(np.log10(ratio[m]))
            sbi[name] = np.trapz(y, x) / (x[-1] - x[0])
        else:
            sbi[name] = np.nan
        print(f"[SBI] {name}: {sbi[name]:.3f}")

    # (C) RMSE per decile
    rmse_by_dec = {}
    for name in models:
        se = rmse_bins[name]["se"]; n = np.maximum(rmse_bins[name]["n"], 1.0)
        rmse_by_dec[name] = np.sqrt(se / n)

    # (D) concatenate residual samples
    resid_concat = {name: (np.concatenate(v) if len(v)>0 else np.array([])) for name,v in resid_samples.items()}
    for name, chunks in resid_samples.items():
        tot = int(np.sum([a.size for a in chunks])) if chunks else 0
        print(f"[DEBUG] residual samples collected — {name}: {tot}")

    # Obs residual “physics floor”
    obs_resid_all = np.concatenate(obs_resid_chunks) if obs_resid_chunks else np.array([])
    obs_floor = None
    if obs_resid_all.size:
        obs_floor = {
            "median": np.median(obs_resid_all),
            "q1":     np.quantile(obs_resid_all, 0.25),
            "q3":     np.quantile(obs_resid_all, 0.75),
        }
        print("[OBS floor] median={:.3e}  IQR=[{:.3e}, {:.3e}]"
            .format(obs_floor["median"], obs_floor["q1"], obs_floor["q3"]))

    # ---------------- plotting (2×2) ----------------
    fig = plt.figure(figsize=(12,9))

    # (A) scale-dependent RMSE
    axA = plt.subplot(2, 2, 1)
    lam_min = reliable_lambda_min_km(dx, dy, factor=4.0)
    lam_max = 150.0

    for name in models:
        wl = spectra_err[name]["wl"]
        amp = spectra_err[name]["amp_rel"]  # relative (RMSE/obs)
        _safe_logplot(axA, wl, amp, lw=2, label=name)

    # shade unreliable wavelengths
    # compute tight x-limits from data so the left grey slab disappears
    wl_all = np.hstack([np.asarray(spectra_err[n]["wl"]) for n in models])
    wl_all = wl_all[np.isfinite(wl_all) & (wl_all > 0)]
    xlo = max(lam_min, wl_all.min())
    xhi = min(lam_max, wl_all.max())
    # Panel A: restrict to resolved wavelengths
    lam_min = reliable_lambda_min_km(dx, dy, factor=4.0)  # e.g., ~4 grid cells
    lam_max = np.nanmax([np.nanmax(spectra_err[k]["wl"]) for k in models])
    axA.set_xlim(lam_min, lam_max)  # e.g., ~10–150 km for your grid

    # optional: show the resolution limit as a thin guide instead of a shaded block
    axA.axvline(lam_min, color="0.8", lw=1, ls="--")

    axA.set_xlabel("Wavelength (km)")
    axA.set_ylabel("Relative error amplitude (RMSE/obs)")
    axA.set_title("(A) Relative scale error (domain-avg)")
    axA.grid(True, which="both", alpha=0.3); axA.legend(frameon=False)

    # (B) power spectrum ratio
    axB = plt.subplot(2, 2, 2)
    for name, (wl, ratio) in spectra_ratio.items():
        _safe_logplot(axB, wl, ratio, lw=2, label=name)  # ← use safe log plotter
    # draw 1.0 guide if y-axis is log-capable; if not, it will still show fine
    axB.axhline(1.0, color="k", lw=1, ls="--", alpha=0.6)
    axB.set_xlabel("Wavelength (km)"); axB.set_ylabel(r"$P_\mathrm{model}/P_\mathrm{obs}$")
    axB.set_title("(B) Spectral energy ratio (1 = perfect)")
    axB.grid(True, which="both", alpha=0.3); axB.legend(frameon=False)

    axB.text(0.02, 0.02,
         "SBI  " + "  ".join([f"{k}={sbi[k]:.2f}" for k in models.keys()]),
         transform=axB.transAxes, fontsize=9, va="bottom")

    # (C) RMSE vs distance-to-coast deciles
    axC = plt.subplot(2,2,3)
    x = np.arange(1,11)
    for name, y in rmse_by_dec.items():
        axC.plot(x, y, marker="o", label=name)
    axC.set_xlabel("Distance-to-coast decile (1=nearest)"); axC.set_ylabel("RMSE (ln mg m$^{-3}$)")
    axC.set_title("(C) RMSE across coastal gradient")
    axC.grid(True, alpha=0.3); axC.legend(frameon=False)

    # (D) ADE residual distributions
    axD = plt.subplot(2,2,4)

    # Observational residual floor band
    if 'obs_floor' in locals() and obs_floor is not None:
        axD.axhspan(obs_floor["q1"], obs_floor["q3"], color="0.88", zorder=0,
                    label="Obs residual IQR")
        axD.axhline(obs_floor["median"], color="0.5", lw=1, ls="--",
                    label="Obs residual median")

    # Build clean arrays per model (finite + positive only)
    data, labels = [], []
    for name in models:
        arr = resid_concat[name]
        if arr.size:
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size:
                arr = np.clip(arr, EPS_Y, None)
                data.append(arr)
                labels.append(name)

    if data:
        axD.boxplot(data, labels=labels, showfliers=False)
        axD.set_yscale("log")
    else:
        axD.text(0.5, 0.5, "no finite residuals", ha='center', va='center',
                transform=axD.transAxes, fontsize=10, color='0.4')

    axD.set_ylabel(r"|Residual|  (ln mg m$^{-3}$ s$^{-1}$)")
    axD.set_title("(D) Physics residual distributions")
    axD.grid(True, which="both", alpha=0.3)

    axD.legend(frameon=False, loc="lower left")


    fig.suptitle("Domain-wide robustness: scale, regime, and physics fidelity", y=0.98, fontsize=14, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(args.out, dpi=300)
    print(f"✓ Saved: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
