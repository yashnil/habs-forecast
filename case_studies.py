#!/usr/bin/env python3
"""
case_studies.py — Publication-quality HAB case study figures

This script builds a multi-panel, aesthetically rich figure for a chosen coastal
event window and region (e.g., Monterey Bay), comparing Observed chlorophyll-a
(derived from log_chl) with model predictions exported by export_preds.py.
It also includes a Hovmöller “spread” view and an area-above-threshold time series.

Highlights
- Regional context + zoom-in side-by-side (Obs vs ConvLSTM vs TFT vs PINN)
- Optional coastline overlay (via Cartopy if available; falls back cleanly otherwise)
- Super-resolution *visualization* (bicubic upsampling + light smoothing) so the map
  looks “alive” at ~250–500 m — purely for aesthetics; underlying 4 km data are unchanged
- Adaptive color scale and bloom thresholds (95th percentile in window unless overridden)
- Hovmöller along-coast band to show “spread” through time
- Bloom polygon outlines (Obs) overlaid on all zoom panels for easy comparison
- Minimal heavy dependencies: xarray, numpy, matplotlib, scipy (cartopy optional)

Usage
------
python case_studies.py \
  --obs "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --pred "/Users/yashnilmohanty/HAB_Models/exports/convlstm__vanilla_best.nc:ConvLSTM" \
  --pred "/Users/yashnilmohanty/HAB_Models/exports/tft__convTFT_best.nc:TFT" \
  --pred "/Users/yashnilmohanty/HAB_Models/exports/pinn__convLSTM_best.nc:PINN" \
  --region monterey \
  --date 2020-06-01 \
  --window-days 56 \
  --upsample 8 \
  --smooth-sigma 0.6 \
  --out "fig_case_study_monterey_20200601.png"

If --date is omitted, the script will auto-pick the peak observed day (max regional mean chl)
within the window (default last 56 valid days).

Notes
- The script expects log-space in both OBS (“log_chl”) and predictions.
  It back-transforms to linear mg m⁻³ for plotting.
- If a prediction file stores a DataArray, it will be read directly; if it
  stores a Dataset, it will try common variable names:
  ["log_chl", "log_chl_pred", "pred", "yhat", "prediction"].
- “Super-resolution” here is purely a plotting trick (bicubic interpolation).
  Please do not interpret sub-km textures as new information.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import patheffects as pe

from scipy.ndimage import zoom as ndi_zoom, gaussian_filter

# Optional coastline
_HAS_CARTOPY = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False

# -------------------------------------------------------------
# Region presets (approx boxes). You can add your own.
# lon_min, lon_max, lat_min, lat_max
REGIONS = {
    "monterey": (-122.5, -121.3, 36.4, 37.1),
    "sfbay":    (-123.2, -122.0, 37.3, 38.3),
    "socal":    (-121.0, -117.5, 32.5, 34.5),
    "custom":   None,  # use --bbox
}

COMMON_PRED_VAR_NAMES = ["log_chl", "log_chl_pred", "pred", "yhat", "prediction"]

def _log_to_linear(da_log: xr.DataArray) -> xr.DataArray:
    """exp back-transform with safe nan handling; returns mg m^-3"""
    out = np.exp(da_log)
    out.name = "chl_mg_m3"
    out.attrs["units"] = "mg m-3"
    return out

def _open_obs(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path)
    # Heuristic: prefer 'log_chl'; otherwise allow 'chlor_a' and convert (protect zeros)
    if "log_chl" not in ds:
        if "chlor_a" in ds:
            # Clamp minimum to avoid log(0)
            chl = ds["chlor_a"].clip(min=1e-3)
            ds["log_chl"] = np.log(chl)
        else:
            # Try any var that looks like log_chl
            for cand in ds.data_vars:
                if "log" in cand and "chl" in cand:
                    ds = ds.rename({cand: "log_chl"})
                    break
    if "log_chl" not in ds:
        raise ValueError("Could not find 'log_chl' or convertible chlorophyll in OBS file.")
    return ds

def _open_pred(path: str) -> xr.DataArray:
    """Open prediction file as DataArray in log space (name 'log_chl')."""
    try:
        da = xr.open_dataarray(path)
        if da.name is None:
            da.name = "log_chl"
        return da
    except Exception:
        ds = xr.open_dataset(path)
        for v in COMMON_PRED_VAR_NAMES:
            if v in ds:
                da = ds[v]
                break
        else:
            # If only one var, take it
            if len(ds.data_vars) == 1:
                da = list(ds.data_vars.values())[0]
            else:
                raise ValueError(f"Could not find prediction variable in {path}. "
                                 f"Tried {COMMON_PRED_VAR_NAMES}.")
        # Standardize name
        if da.name != "log_chl":
            da = da.rename("log_chl")
        return da

def _coord_names(da):
    """Return (lon_name, lat_name) matching common conventions."""
    lon_name = "lon" if "lon" in da.coords else ("longitude" if "longitude" in da.coords else list(da.coords)[-1])
    lat_name = "lat" if "lat" in da.coords else ("latitude" if "latitude" in da.coords else list(da.coords)[-2])
    return lon_name, lat_name

def _crop_bbox(da, bbox, *, min_cells=4, pad_deg=0.05):
    """Robust spatial crop that works for ascending or descending coords.

    - Clips requested bbox to data extents
    - Handles ascending/descending latitude/longitude
    - If the crop would be empty, gradually expands the bbox outward until it
      contains at least `min_cells` along each spatial dimension (or hits edges).
    """
    import numpy as np
    lon_name, lat_name = _coord_names(da)

    def _ordered_slice(a, lo, hi):
        asc = bool(np.all(np.diff(a) > 0))
        if asc:
            return slice(min(lo, hi), max(lo, hi))
        else:
            return slice(max(lo, hi), min(lo, hi))

    lon0, lon1, lat0, lat1 = bbox
    # Clip to data range
    lons = np.asarray(da[lon_name].values)
    lats = np.asarray(da[lat_name].values)
    lon0 = float(np.clip(lon0, np.nanmin(lons), np.nanmax(lons)))
    lon1 = float(np.clip(lon1, np.nanmin(lons), np.nanmax(lons)))
    lat0 = float(np.clip(lat0, np.nanmin(lats), np.nanmax(lats)))
    lat1 = float(np.clip(lat1, np.nanmin(lats), np.nanmax(lats)))

    out = da.sel({lon_name: _ordered_slice(lons, lon0, lon1),
                  lat_name: _ordered_slice(lats, lat0, lat1)})

    # Fallback: expand gently if empty
    tries = 0
    while ((out.sizes.get(lon_name, 0) == 0 or out.sizes.get(lat_name, 0) == 0) and tries < 10):
        tries += 1
        pad = pad_deg * tries
        out = da.sel({lon_name: _ordered_slice(lons, lon0 - pad, lon1 + pad),
                      lat_name: _ordered_slice(lats, lat0 - pad, lat1 + pad)})
        if out.sizes.get(lon_name, 0) >= min_cells and out.sizes.get(lat_name, 0) >= min_cells:
            break

    # Final fallback: boolean mask drop (handles non-monotonic edge cases)
    if out.sizes.get(lon_name, 0) == 0 or out.sizes.get(lat_name, 0) == 0:
        out = da.where(
            (da[lon_name] >= min(lon0, lon1)) & (da[lon_name] <= max(lon0, lon1)) &
            (da[lat_name]  >= min(lat0, lat1)) & (da[lat_name]  <= max(lat0, lat1)),
            drop=True
        )
    return out

def _align_times(das: List[xr.DataArray]) -> List[xr.DataArray]:
    """Align on the intersection of times; cast to same calendar if needed."""
    times = set(das[0]["time"].values)
    for d in das[1:] :
        times = times & set(d["time"].values)
    times = sorted(list(times))
    return [d.sel(time=times) for d in das]

def _upsample_for_plot(da_lin, factor=8, smooth_sigma=0.6):
    """Upsample 2D field for prettier plotting, safely handling NaNs/empties.

    Returns: hi_res_array, lons_hr, lats_hr
    """
    import numpy as np
    from scipy.ndimage import zoom as ndi_zoom, gaussian_filter

    if da_lin.ndim != 2:
        # If a DataArray with time or other dims slipped through, squeeze them.
        da_lin = da_lin.squeeze()

    # Guard: empty selection
    if da_lin.size == 0 or 0 in da_lin.shape:
        # Return minimal NaN canvas to avoid crashes; caller can annotate "no data".
        hi = np.full((2, 2), np.nan)
        lat_name, lon_name = ("lat" if "lat" in da_lin.coords else "latitude", 
                              "lon" if "lon" in da_lin.coords else "longitude")
        lats = np.linspace(-1, 1, 2)
        lons = np.linspace(-1, 1, 2)
        return hi, lons, lats

    arr = np.asarray(da_lin.values, dtype=float)
    finite = np.isfinite(arr)

    if not finite.any():
        # All-NaN: make a blank canvas to keep the pipeline alive.
        hi = np.full((max(2, arr.shape[0]*factor), max(2, arr.shape[1]*factor)), np.nan)
        lats = np.linspace(float(da_lin[da_lin.dims[0]].values.min()), float(da_lin[da_lin.dims[0]].values.max()), hi.shape[0])
        lons = np.linspace(float(da_lin[da_lin.dims[1]].values.min()), float(da_lin[da_lin.dims[1]].values.max()), hi.shape[1])
        return hi, lons, lats

    # Fill NaNs with local median (global median fallback) to avoid zoom ringing
    med = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    arr_filled = np.where(np.isfinite(arr), arr, med)

    # Upsample with cubic interpolation
    hi = ndi_zoom(arr_filled, zoom=factor, order=3, mode="nearest")

    # Gentle Gaussian to reduce pixelation
    if smooth_sigma and smooth_sigma > 0:
        hi = gaussian_filter(hi, sigma=smooth_sigma, mode="nearest")

    # High-res coord vectors assuming quasi-regular grid
    lat_name, lon_name = ("lat" if "lat" in da_lin.coords else "latitude",
                          "lon" if "lon" in da_lin.coords else "longitude")
    lats = np.linspace(float(da_lin[lat_name].values.min()), float(da_lin[lat_name].values.max()), hi.shape[0])
    lons = np.linspace(float(da_lin[lon_name].values.min()), float(da_lin[lon_name].values.max()), hi.shape[1])

    return hi, lons, lats

def _auto_event_date(obs_lin: xr.DataArray, window_days: int = 56) -> pd.Timestamp:
    # Pick last N valid timesteps, choose date with max regional-mean chl
    sub = obs_lin.dropna("time", how="all").isel(time=slice(-window_days, None))
    means = sub.mean(dim=("lat", "lon"), skipna=True)
    idx = int(means.argmax())
    return pd.to_datetime(sub["time"].values[idx])

def _compute_threshold(obs_lin: xr.DataArray, q: float = 0.95) -> float:
    vals = obs_lin.values
    thresh = np.nanquantile(vals, q)
    # guard against pathological NaN or zero spread
    if not np.isfinite(thresh) or thresh <= 0:
        thresh = float(np.nanmedian(vals) * 1.5)
    return float(thresh)

def _contour_from_mask(ax, img_extent, mask_hr: np.ndarray, label: str, lw: float = 1.2):
    """Plot a thin outline of the high-chl mask on top of an imshow background."""
    from skimage import measure
    contours = measure.find_contours(mask_hr.astype(float), 0.5)
    # Transform contour pixel coords to data coords via extent
    x0, x1, y0, y1 = img_extent
    h, w = mask_hr.shape
    for c in contours:
        # c[:,0]=row(y), c[:,1]=col(x)
        xs = x0 + (x1 - x0) * (c[:,1] / (w - 1))
        ys = y0 + (y1 - y0) * (1 - c[:,0] / (h - 1))  # imshow's origin='upper'
        ax.plot(xs, ys, lw=lw, color="black",
                path_effects=[pe.SimpleLineShadow(offset=(0.8,-0.8), alpha=0.6),
                              pe.Normal()],
                label=label)

def _add_coast(ax, bbox, satellite_bg=False):
    if not _HAS_CARTOPY:
        return
    ax.coastlines(resolution="10m", linewidth=0.6, color="k")
    land = cfeature.NaturalEarthFeature("physical", "land", "10m",
                                        edgecolor="face", facecolor="#f2efe6", zorder=0)
    ax.add_feature(land, zorder=0)
    ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=ccrs.PlateCarree())
    if satellite_bg and _HAS_CARTOPY:
        ax.stock_img()  # simple “satellite style” background
    if satellite_bg and _HAS_CARTOPY:
        try:
            ax.stock_img()
        except Exception:
            pass

def _hovmöller_band(da_lin: xr.DataArray, bbox: Tuple[float, float, float, float], width_km: float = 20.0) -> xr.DataArray:
    """A simple along-coast Hovmöller surrogate: mean across a narrow offshore band within the box,
    aggregating across longitudes to produce lat x time (or across latitudes to produce lon x time
    if the coastline is E-W oriented)."""
    lon = "lon" if "lon" in da_lin.coords else "longitude"
    lat = "lat" if "lat" in da_lin.coords else "latitude"
    sub = _crop_bbox(da_lin, bbox)
    # Decide orientation by box aspect ratio
    dlon = bbox[1] - bbox[0]
    dlat = bbox[3] - bbox[2]
    if dlat >= dlon:  # N-S coastline: average over a small lon band to get lat x time
        hv = sub.mean(dim=lon, skipna=True)
        hv = hv.rename({lat: "y"})
    else:
        hv = sub.mean(dim=lat, skipna=True)
        hv = hv.rename({lon: "x"})
    return hv  # dims: time, y|x

def build_case_figure(
    obs_path: str,
    pred_specs: List[str],
    region: str = "monterey",
    bbox: Optional[Tuple[float, float, float, float]] = None,
    date: Optional[str] = None,
    window_days: int = 56,
    upsample: int = 8,
    smooth_sigma: float = 0.6,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    out_path: str = "fig_case_study.png",
    cmap: str = "turbo",
    satellite_bg: bool = False,
) -> None:
    """
    pred_specs: list of strings like "/path/to/file.nc:Label"
    """
    # ---------------- Load data
    obs_ds = _open_obs(obs_path)
    obs_da = obs_ds["log_chl"]
    # back-transform to mg m^-3
    obs_lin = _log_to_linear(obs_da)

    preds: Dict[str, xr.DataArray] = {}
    for spec in pred_specs:
        if ":" in spec:
            p, label = spec.split(":", 1)
        else:
            p, label = spec, Path(spec).stem
        da = _open_pred(p)  # log space
        preds[label] = _log_to_linear(da)

    # ---------------- Region bbox
    if region.lower() != "custom":
        bbox = REGIONS.get(region.lower(), None)
    if bbox is None:
        raise ValueError("Provide either a known --region or a --bbox lon_min lon_max lat_min lat_max.")
    # Crop
    obs_lin_box = _crop_bbox(obs_lin, bbox)
    pred_lin_box = {k: _crop_bbox(v, bbox) for k, v in preds.items()}

    # ---------------- Align times
    series = [obs_lin_box] + list(pred_lin_box.values())
    series = _align_times(series)
    obs_lin_box = series[0]
    labels = list(pred_lin_box.keys())
    for i, lbl in enumerate(labels):
        pred_lin_box[lbl] = series[i + 1]

    if obs_lin_box.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping times found within region. Check inputs.")

    # ---------------- Pick event date
    if date:
        t0 = pd.to_datetime(date)
        if t0 not in pd.to_datetime(obs_lin_box["time"].values):
            # snap to nearest available time
            t0 = pd.to_datetime(obs_lin_box["time"].sel(time=t0, method="nearest").values)
    else:
        t0 = _auto_event_date(obs_lin_box, window_days=window_days)

    # Determine window bounds for secondary plots
    t_all = pd.to_datetime(obs_lin_box["time"].values)
    tmin = max(t_all.min(), t0 - pd.Timedelta(days=window_days))
    tmax = min(t_all.max(), t0 + pd.Timedelta(days=window_days))

    # ---------------- Color scale & threshold
    if vmin is None or vmax is None:
        # robust range from OBS over the window
        obs_win = obs_lin_box.sel(time=slice(tmin, tmax))
        lo = float(np.nanquantile(obs_win.values, 0.05))
        hi = float(np.nanquantile(obs_win.values, 0.995))
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    thresh = _compute_threshold(obs_lin_box.sel(time=t0), q=0.95)

    # ---------------- Prepare zoom imagery (obs + preds at t0)
        # ---------------- Prepare zoom imagery (obs + preds at t0)
    def _prep_img(da_lin_t: xr.DataArray):
        """Return (hi, extent, mask_hr) for plotting. Never raises on empty."""
        # Safe upsample; returns small NaN canvas if empty
        hi, lons_hr, lats_hr = _upsample_for_plot(da_lin_t, factor=upsample, smooth_sigma=smooth_sigma)
        extent = (np.nanmin(lons_hr), np.nanmax(lons_hr), np.nanmin(lats_hr), np.nanmax(lats_hr))
        # Bloom mask against threshold; NaNs become False
        mask_hr = np.where(np.isfinite(hi), hi >= thresh, False)
        return hi, extent, mask_hr

    obs_t = obs_lin_box.sel(time=t0)
    imgs = {"Observed": _prep_img(obs_t)}
    for lbl, da in pred_lin_box.items():
        imgs[lbl] = _prep_img(da.sel(time=t0))

    # ---------------- Hovmöller (obs vs models, stacked)
    hov_obs = _hovmöller_band(obs_lin_box.sel(time=slice(tmin, tmax)), bbox)
    hov_preds = {lbl: _hovmöller_band(da.sel(time=slice(tmin, tmax)), bbox) for lbl, da in pred_lin_box.items()}

    # ---------------- Area above threshold
    def _area_ts(da_lin: xr.DataArray) -> pd.Series:
        mask = (da_lin >= thresh)
        # approximate km^2 per cell from mean lat (assume square 4 km — rough); scale after upsample?
        # Use grid spacing from coords if regular lat/lon
        lat = da_lin["lat"].values if "lat" in da_lin.coords else da_lin["latitude"].values
        lon = da_lin["lon"].values if "lon" in da_lin.coords else da_lin["longitude"].values
        # crude cell area using mean lat
        R = 6371.0  # km
        dlat = np.deg2rad(np.abs(lat[1] - lat[0])) if len(lat) > 1 else 0.036  # ~4 km / R (fallback)
        dlon = np.deg2rad(np.abs(lon[1] - lon[0])) if len(lon) > 1 else 0.036
        mean_lat = np.deg2rad(np.nanmean(lat))
        # area per cell (km^2) on sphere approx
        A = (R**2) * dlat * dlon * np.cos(mean_lat)
        areas = mask.sum(dim=("lat", "lon"), skipna=True).to_pandas() * A
        return areas

    area_obs = _area_ts(obs_lin_box.sel(time=slice(tmin, tmax)))
    area_preds = {lbl: _area_ts(da.sel(time=slice(tmin, tmax))) for lbl, da in pred_lin_box.items()}

    # ---------------- Figure
    # Layout: 2 rows
    # Row 1: [Regional OBS map]  [Hovmöller (stacked small)]  [Area timeseries]
    # Row 2: [Zoom OBS] [Zoom ConvLSTM] [Zoom TFT] [Zoom PINN]
    plt.close("all")
    figsize = (15, 9.5)
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    # Gridspec
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1, 1.05], width_ratios=[1.2, 1.1, 0.9, 1.0, 1.0, 1.0])

    # --- Row 1, col 0-1: regional OBS
    if _HAS_CARTOPY:
        ax0 = fig.add_subplot(gs[0, 0:2], projection=ccrs.PlateCarree())
    else:
        ax0 = fig.add_subplot(gs[0, 0:2])
    im0 = ax0.imshow(obs_t.values, origin="upper",
                     extent=(bbox[0], bbox[1], bbox[2], bbox[3]),
                     interpolation="none",
                     vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    if _HAS_CARTOPY:
        _add_coast(ax0, bbox, satellite_bg=args.satellite_bg)
    ax0.set_title(f"Regional context — Observed (mg m$^{{-3}}$) @ {pd.to_datetime(t0).date()}")
    ax0.grid(ls=":", lw=0.3, color="k", alpha=0.2)

    cax0 = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im0, cax=cax0, orientation="vertical")
    cb.set_label("Chlorophyll-a (mg m$^{-3}$)")

    # --- Row 1, col 3: Hovmöller OBS + a light marker at t0
    ax1 = fig.add_subplot(gs[0, 3])
    # decide orientation
    if "y" in hov_obs.dims:
        im1 = ax1.pcolormesh(pd.to_datetime(hov_obs["time"].values),
                             hov_obs["y"].values, hov_obs.T,
                             shading="nearest", rasterized=True, vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.set_ylabel("Latitude (°)")
    else:
        im1 = ax1.pcolormesh(pd.to_datetime(hov_obs["time"].values),
                             hov_obs["x"].values, hov_obs.T,
                             shading="nearest", rasterized=True, vmin=vmin, vmax=vmax, cmap=cmap)
        ax1.set_ylabel("Longitude (°)")
    ax1.axvline(pd.to_datetime(t0), color="k", lw=1.0, ls="--", alpha=0.7)
    ax1.set_title("Obs Hovmöller — along-coast band")
    ax1.set_xlabel("Time")

    # --- Row 1, col 4-5: Area above threshold time series
    ax2 = fig.add_subplot(gs[0, 4:6])
    ax2.plot(area_obs.index, area_obs.values, lw=2.0, label="Observed")
    for lbl, s in area_preds.items():
        ax2.plot(s.index, s.values, lw=1.5, alpha=0.9, label=lbl)
    ax2.axvline(pd.to_datetime(t0), color="k", lw=1.0, ls="--", alpha=0.7)
    ax2.set_title(f"Bloom area ≥ {thresh:.2f} mg m$^{{-3}}$")
    ax2.set_ylabel("km$^2$ within box")
    ax2.set_xlabel("Time")
    ax2.grid(ls=":", lw=0.3)
    ax2.legend(ncols=2, fontsize=9, frameon=True)

    # --- Row 2: Zoom panels with super-resolution + observed outline overlay
    panel_order = ["Observed"] + labels
    for j, lbl in enumerate(panel_order):
        hi, extent, mask_hr = imgs[lbl]
        if _HAS_CARTOPY:
            axi = fig.add_subplot(gs[1, j], projection=ccrs.PlateCarree())
            img = axi.imshow(hi, origin="upper", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
            _add_coast(ax0, bbox, satellite_bg=args.satellite_bg)
        else:
            axi = fig.add_subplot(gs[1, j])
            img = axi.imshow(hi, origin="upper", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        # Observed mask contour plotted on all panels for reference
        if lbl != "Observed":
            obs_mask_hr = imgs["Observed"][2]
            _contour_from_mask(axi, extent, obs_mask_hr, label="Obs ≥ thr", lw=1.0)
        else:
            _contour_from_mask(axi, extent, mask_hr, label="Obs ≥ thr", lw=1.2)
        axi.set_title(lbl)
        axi.set_xticks([]); axi.set_yticks([])

    # Suptitle and small caption
    fig.suptitle(f"Case study — {region.title()} @ {pd.to_datetime(t0).date()}  "
                 f"(upsample×{upsample}, smooth σ={smooth_sigma})", y=0.995, fontsize=14)
    fig.text(0.01, 0.01,
             "Notes: Upsampling is for visualization only. Threshold is 95th percentile of observed chl "
             "within the selected region/day. Hovmöller built from a narrow along-coast band.\n"
             "Files: " + ", ".join([Path(obs_path).name] + [Path(spec.split(':',1)[0]).name for spec in pred_specs]),
             fontsize=8, ha="left", va="bottom")

    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Build publication-quality HAB case study figures.")
    p.add_argument("--obs", required=True, help="Path to OBS NetCDF with 'log_chl' or 'chlor_a'.")
    p.add_argument("--pred", action="append", default=[], help="Prediction file spec 'path:Label'. Repeatable.")
    p.add_argument("--region", default="monterey", choices=list(REGIONS.keys()), help="Named region preset.")
    p.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("LON_MIN","LON_MAX","LAT_MIN","LAT_MAX"),
                   help="Custom bbox if region=='custom'.")
    p.add_argument("--date", default=None, help="YYYY-MM-DD. If omitted, auto-peak within window.")
    p.add_argument("--window-days", type=int, default=56, help="Half-window around date for context stats.")
    p.add_argument("--upsample", type=int, default=8, help="Super-resolution visualization factor (e.g., 8 => ~500 m).")
    p.add_argument("--smooth-sigma", type=float, default=0.6, help="Gaussian blur (hi-res pixels) to reduce blockiness.")
    p.add_argument("--vmin", type=float, default=None, help="Color scale min (mg m⁻3). Defaults to robust 5th pct in window.")
    p.add_argument("--vmax", type=float, default=None, help="Color scale max (mg m⁻3). Defaults to robust 99.5th pct in window.")
    p.add_argument("--out", default="fig_case_study.png", help="Output PNG path.")
    p.add_argument("--cmap", default="turbo", help="Matplotlib colormap. Try 'viridis' or 'magma' if you prefer.")
    p.add_argument(
        "--satellite-bg",
        action="store_true",
        help="Use satellite-like basemap background under the chlorophyll map."
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_case_figure(
        obs_path=args.obs,
        pred_specs=args.pred,
        region=args.region,
        bbox=tuple(args.bbox) if args.bbox is not None else None,
        date=args.date,
        window_days=args.window_days,
        upsample=args.upsample,
        smooth_sigma=args.smooth_sigma,
        vmin=args.vmin, vmax=args.vmax,
        out_path=args.out,
        cmap=args.cmap,
        satellite_bg=args.satellite_bg
    )
