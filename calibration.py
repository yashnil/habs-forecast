#!/usr/bin/env python3
# calibration.py
import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "regular",
})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import brier_score_loss, precision_recall_curve, average_precision_score
from scipy.special import expit

# coastline + masking utils
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point, box as shp_box
from shapely.ops import unary_union, transform as shp_transform
from shapely.prepared import prep as shp_prep
from pyproj import Transformer
from scipy.ndimage import binary_erosion

# ----------------------------- helpers -----------------------------
PREF_VARS_PRED = ("log_chl_pred", "chl_log_pred")
PREF_VARS_OBS  = ("log_chl_true", "log_chl", "chl_log")
PREF_DIMS_LAT  = ("lat", "latitude", "y")
PREF_DIMS_LON  = ("lon", "longitude", "x")

MILES_TO_METERS = 1609.344

def choose(mapping, names, what="item"):
    for n in names:
        if n in mapping: return n
    raise KeyError(f"Could not find any {what} named {names} in {list(mapping)}")

def get_lat_lon(ds):
    dlat = choose(ds.dims, PREF_DIMS_LAT, "lat dim")
    dlon = choose(ds.dims, PREF_DIMS_LON, "lon dim")
    lat = ds[dlat].values
    lon = ds[dlon].values
    if lat.ndim == 1 and lon.ndim == 1:
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    else:
        LAT, LON = lat, lon
    return dlat, dlon, LAT, LON

def build_coast_mask(lon2d, lat2d, miles=10.0, erode_px=1):
    """Bool mask of ocean cells within ≤ miles of coastline (Natural Earth)."""
    land = unary_union(list(shpreader.Reader(
        shpreader.natural_earth("10m","physical","land")).geometries()))
    land_prep = shp_prep(land)
    to_m = Transformer.from_crs("EPSG:4326","EPSG:32610", always_xy=True).transform
    land_m = shp_transform(to_m, land)
    coast_m = land_m.boundary
    thresh_m = miles * MILES_TO_METERS

    H, W = lat2d.shape
    mask = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            lo, la = float(lon2d[i,j]), float(lat2d[i,j])
            if land_prep.contains(Point(lo, la)):  # exclude land
                continue
            x, y = to_m(lo, la)
            if Point(x, y).distance(coast_m) <= thresh_m:
                mask[i, j] = True
    if erode_px > 0:
        mask = binary_erosion(mask, iterations=int(erode_px))
    return mask

def to_linear_from_log(log_chl, floor=0.0):
    lin = np.exp(log_chl) - float(floor)
    return np.clip(lin, 0.0, None)

def logistic_prob_from_conc(chl_mg, thresh=5.0, scale=2.5):
    """Monotone proxy probability P(chl ≥ thresh) from a continuous prediction."""
    return expit((chl_mg - float(thresh))/float(scale))

def reliability_points(y_true01, p, n_bins=10, min_count=50):
    bins = np.linspace(0, 1, n_bins+1)
    mids = 0.5*(bins[:-1]+bins[1:])
    out = []
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1])
        if m.sum() >= min_count:
            out.append((mids[i], y_true01[m].mean(), m.sum()))
    if not out:
        return np.array([]), np.array([]), np.array([])
    mids, obs_freq, counts = map(np.array, zip(*out))
    return mids, obs_freq, counts

def coverage_hovmoller(da_obs, coast_mask, dlat, dlon):
    """Return (LAT1D, times, coverage[%] array (lat x time)) within coastal strip."""
    # mean across longitude within coastal rows → % valid per time & lat
    if da_obs.sizes[dlat] != coast_mask.shape[0] or da_obs.sizes[dlon] != coast_mask.shape[1]:
        raise ValueError("Mask shape does not match data grid.")
    mask_da = xr.DataArray(coast_mask, dims=(dlat, dlon))
    valid = xr.apply_ufunc(np.isfinite, da_obs)
    valid = valid.where(mask_da)
    # counts per lat row
    denom = mask_da.sum(dim=dlon)
    # avoid division by zero
    denom = xr.where(denom > 0, denom, np.nan)
    # % valid per (time, lat)
    cov = 100.0 * (valid.sum(dim=dlon) / denom)
    # final 2D: lat x time (more “Hovmöller-ish”)
    cov2d = cov.transpose(dlat, "time").values
    lat1d = da_obs[dlat].values if da_obs[dlat].ndim==1 else np.nanmean(da_obs[dlat].values, axis=1)
    times = da_obs["time"].values
    return lat1d, times, cov2d

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Calibration & PR + Coverage Hovmöller")
    ap.add_argument("--pred", action="append", required=True,
                    help="file.nc:Label (repeat for multiple models)")
    ap.add_argument("--obs", default=None,
                    help="Optional NetCDF with observed log_chl; if omitted, uses log_chl_true in first --pred.")
    ap.add_argument("--out_calib", default="calibration_pr.png",
                    help="Output PNG for calibration (reliability + PR).")
    ap.add_argument("--out_hov", default="coverage_hovmoller.png",
                    help="Output PNG for coverage Hovmöller.")
    ap.add_argument("--thresh", type=float, default=5.0, help="HAB threshold in mg m^-3.")
    ap.add_argument("--scale", type=float, default=2.5, help="Logistic scale for probability mapping.")
    ap.add_argument("--floor", type=float, default=0.0, help="Linear-space floor to subtract after exp(log_chl).")
    ap.add_argument("--coast_miles", type=float, default=10.0, help="Coastal strip width (miles) for Hovmöller.")
    ap.add_argument("--erode_px", type=int, default=1, help="Mask erosion (pixels).")
    ap.add_argument("--start", default=None, help="Optional start date (YYYY-MM-DD).")
    ap.add_argument("--end",   default=None, help="Optional end date (YYYY-MM-DD).")
    args = ap.parse_args()

    # ---------- Load first dataset to discover grid / obs ----------
    pred_specs = []
    for item in args.pred:
        if ":" not in item:
            raise ValueError("Each --pred must be 'path.nc:Label'")
        path, label = item.split(":", 1)
        pred_specs.append((path, label))

    # For coverage Hovmöller, prefer user-provided --obs; else use first pred file's obs
    if args.obs:
        ds_obs = xr.open_dataset(args.obs)
        vobs = choose(ds_obs.data_vars, PREF_VARS_OBS, "obs var")
        if args.start or args.end:
            ds_obs = ds_obs.sel(time=slice(args.start, args.end))
        obs_da = ds_obs[vobs]
        dlat, dlon, LAT, LON = get_lat_lon(ds_obs)
    else:
        path0, _ = pred_specs[0]
        d0 = xr.open_dataset(path0)
        vobs = choose(d0.data_vars, PREF_VARS_OBS, "obs var in pred file")
        if args.start or args.end:
            d0 = d0.sel(time=slice(args.start, args.end))
        obs_da = d0[vobs]
        dlat, dlon, LAT, LON = get_lat_lon(d0)

    # ---------- Build coastal mask once (for Hovmöller) ----------
    print("Building ≤10-mile coastal mask …")
    coast_mask = build_coast_mask(LON, LAT, miles=args.coast_miles, erode_px=args.erode_px)

    # ---------- Data-coverage Hovmöller ----------
    if "time" in obs_da.dims:
        print("Computing coverage Hovmöller …")
        lat1d, times, cov2d = coverage_hovmoller(obs_da, coast_mask, dlat, dlon)
        fig, ax = plt.subplots(figsize=(11, 4))
        # imshow with datetime axis
        tnum0 = mdates.date2num(np.datetime64(times[0]).astype('datetime64[ns]').astype(object))
        tnum1 = mdates.date2num(np.datetime64(times[-1]).astype('datetime64[ns]').astype(object))
        im = ax.imshow(cov2d,
                       extent=[mdates.date2num(times[0].astype('datetime64[ns]').astype(object)),
                               mdates.date2num(times[-1].astype('datetime64[ns]').astype(object)),
                               np.nanmin(lat1d), np.nanmax(lat1d)],
                       aspect="auto", origin="lower", vmin=0, vmax=100, cmap="viridis")
        ax.set_ylabel("Latitude (°N)")
        ax.set_xlabel("Time")
        ax.set_title("Data coverage within ≤10-mile coastal strip (% valid per latitude × time)")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Coverage (%)")
        fig.tight_layout(); fig.savefig(args.out_hov, dpi=250)
        print(f"✓ Saved Hovmöller: {os.path.abspath(args.out_hov)}")
    else:
        print("Obs has no time dimension; skipping Hovmöller.")

    # ---------- Calibration & PR (multi-model) ----------
    print("Computing calibration + precision–recall …")
    fig, axs = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax_rel, ax_pr = axs

    prevalence_ref = None

    for path, label in pred_specs:
        ds = xr.open_dataset(path)
        if args.start or args.end:
            ds = ds.sel(time=slice(args.start, args.end))
        vp = choose(ds.data_vars, PREF_VARS_PRED, "pred var")
        # prefer obs in this file; else fall back to global obs_da aligned
        if any(v in ds.data_vars for v in PREF_VARS_OBS):
            vo = choose(ds.data_vars, PREF_VARS_OBS, "obs var")
            obs_here = ds[vo]
        else:
            # align to ds time if possible
            obs_here = obs_da
            if "time" in ds[vp].dims and "time" in obs_da.dims:
                common_t = np.array(sorted(set(ds[vp].time.values) & set(obs_da.time.values)))
                if common_t.size:
                    ds = ds.sel(time=common_t); obs_here = obs_da.sel(time=common_t)

        # grab arrays & mask
        pred_log = ds[vp].values
        obs_log  = obs_here.values
        valid = np.isfinite(pred_log) & np.isfinite(obs_log)

        y_obs_lin  = to_linear_from_log(obs_log, floor=args.floor)
        y_pred_lin = to_linear_from_log(pred_log, floor=args.floor)

        # event indicator & pseudo-prob
        y_true = (y_obs_lin >= args.thresh).astype(int)
        p_hat  = logistic_prob_from_conc(y_pred_lin, thresh=args.thresh, scale=args.scale)

        y_true_f = y_true[valid].ravel()
        p_hat_f  = p_hat [valid].ravel()

        # prevalence (for PR baseline)
        prev = y_true_f.mean() if y_true_f.size else np.nan
        if prevalence_ref is None: prevalence_ref = prev

        # reliability
        mids, obs_freq, counts = reliability_points(y_true_f, p_hat_f, n_bins=10, min_count=50)
        if mids.size:
            ax_rel.plot(mids, obs_freq, "-o", lw=1.8, ms=4, label=f"{label}")
        try:
            bs = brier_score_loss(y_true_f, p_hat_f)
            ax_rel.text(0.02, 0.96 - 0.08*len(ax_rel.lines), f"{label} BS={bs:.3f}",
                        transform=ax_rel.transAxes, fontsize=9, va="top")
        except Exception:
            pass

        # precision–recall
        try:
            prec, rec, thr = precision_recall_curve(y_true_f, p_hat_f)
            auc_pr = average_precision_score(y_true_f, p_hat_f)
            ax_pr.plot(rec, prec, lw=2, label=f"{label} (AUPRC={auc_pr:.2f})")
        except Exception:
            pass

    # finalize reliability
    ax_rel.plot([0,1],[0,1],"k--",lw=1,alpha=0.6)
    ax_rel.set_xlim(0,1); ax_rel.set_ylim(0,1)
    ax_rel.set_xlabel("Forecast probability")
    ax_rel.set_ylabel("Observed frequency")
    ax_rel.set_title("Reliability diagram (chl ≥ {:.1f} mg m$^{{-3}}$)".format(args.thresh))
    ax_rel.grid(True, alpha=0.3); ax_rel.legend(frameon=False, loc="lower right")

    # finalize PR
    if prevalence_ref is not None and np.isfinite(prevalence_ref):
        ax_pr.hlines(prevalence_ref, 0, 1, colors="k", linestyles="--", lw=1, alpha=0.5,
                     label=f"Baseline (prevalence={prevalence_ref:.2f})")
    ax_pr.set_xlim(0,1); ax_pr.set_ylim(0,1)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall at HAB threshold")
    ax_pr.grid(True, alpha=0.3); ax_pr.legend(frameon=False, loc="lower left")

    fig.suptitle("Calibration & Detection Skill", y=0.98, fontsize=14, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(args.out_calib, dpi=250)
    print(f"✓ Saved calibration/PR: {os.path.abspath(args.out_calib)}")

if __name__ == "__main__":
    main()

'''
Run file:

python calibration.py --pred "/Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_PINN/predicted_fields.nc:PINN" --pred "/Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_ConvLSTM/predicted_fields.nc:ConvLSTM" --pred "/Users/yashnilmohanty/Desktop/habs-forecast/Diagnostics_TFT/predicted_fields.nc:TFT" --obs "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" --thresh 5.0 --scale 2.5 --floor 0.056616 --out_calib calib_pr.png --out_hov coverage_hov.png

'''