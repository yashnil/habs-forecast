#!/usr/bin/env python3

'''
chmod +x run_case.sh
./run_case.sh
'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union, transform as shp_transform
from shapely.prepared import prep as shp_prep
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_erosion
from pyproj import Transformer
from datetime import datetime, timedelta
import argparse, os, warnings
warnings.filterwarnings("ignore")
from matplotlib.colors import Normalize, PowerNorm   # add PowerNorm
import matplotlib.patheffects as pe                  # for glow on marker

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# --------------------- helpers ---------------------
def choose_var(ds, prefer=("log_chl", "log_chl_pred")):
    for name in prefer:
        if name in ds.data_vars:
            return name
    # fallback: first 1D/2D float
    for k, v in ds.data_vars.items():
        if np.issubdtype(v.dtype, np.floating) and set(v.dims) >= {"lat","lon"}:
            return k
    raise ValueError("No suitable data variable found.")

def upsample_field(lat, lon, arr, factor=8):
    if factor is None or factor <= 1:
        return lat, lon, arr
    lat_u = np.linspace(lat.min(), lat.max(), len(lat)*factor)
    lon_u = np.linspace(lon.min(), lon.max(), len(lon)*factor)
    f = RegularGridInterpolator((lat, lon), arr, bounds_error=False, fill_value=np.nan)
    LAT, LON = np.meshgrid(lat_u, lon_u, indexing="ij")
    arr_u = f(np.stack([LAT.ravel(), LON.ravel()], axis=-1)).reshape(LAT.shape)
    return lat_u, lon_u, arr_u

def smooth_within_mask(arr, mask, sigma=1.2):
    if sigma is None or sigma <= 0:
        return arr
    # normalized Gaussian inside mask (no bleed across land)
    a_in = np.where(mask, arr, 0.0)
    w = mask.astype(float)
    a_sm = gaussian_filter(a_in, sigma=sigma)
    w_sm = gaussian_filter(w, sigma=sigma)
    out = arr.copy()
    ok = w_sm > 1e-6
    out[ok] = a_sm[ok] / w_sm[ok]
    out[~mask] = np.nan
    return out

def inpaint_nearest(arr, mask):
    out = np.array(arr, dtype=float, copy=True)
    missing = np.isnan(out) & mask
    if not missing.any():
        return out
    _, (iy, ix) = distance_transform_edt(~(np.isfinite(out) & mask), return_indices=True)
    out[missing] = out[iy[missing], ix[missing]]
    return out

def build_water_mask_within_10mi(lon, lat, coast_miles=10, erode_px=2):
    print(f"Building ≤{coast_miles} mi coastal water mask …")
    land = unary_union(list(shpreader.Reader(
        shpreader.natural_earth("10m", "physical", "land")
    ).geometries()))
    land_p = shp_prep(land)

    to_m = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True).transform
    land_m = shp_transform(to_m, land)
    coast_m = land_m.boundary
    thresh_m = coast_miles * 1609.344

    if lon.ndim == 1 and lat.ndim == 1:
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    else:
        LAT, LON = lat, lon

    H, W = LAT.shape
    mask = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            pt_lon, pt_lat = float(LON[i, j]), float(LAT[i, j])
            if land_p.contains(Point(pt_lon, pt_lat)):
                continue
            x, y = to_m(pt_lon, pt_lat)
            if Point(x, y).distance(coast_m) <= thresh_m:
                mask[i, j] = True

    if erode_px and erode_px > 0:
        mask = binary_erosion(mask, iterations=int(erode_px))
    print(f"Coastal water cells: {mask.sum()} / {mask.size}")
    return mask

def emphasize_pinpoint(arr, lat, lon, pin_lat, pin_lon, sigma_km=6.0, strength=0.85):
    """Add a small Gaussian bump (display-only) so the pinpoint is the brightest."""
    # degrees-per-km approx near Monterey
    lat_km = 1/111.0
    lon_km = 1/(111.0*np.cos(np.deg2rad(pin_lat)))
    sig_lat = sigma_km * lat_km
    sig_lon = sigma_km * lon_km

    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    r2 = ((LAT - pin_lat)/sig_lat)**2 + ((LON - pin_lon)/sig_lon)**2
    bump = np.exp(-0.5 * r2)
    # scale bump to lift towards vmax while keeping field shape
    vmax = np.nanmax(arr)
    target = vmax if np.isfinite(vmax) else 0
    return np.where(np.isfinite(arr), arr + strength * bump * (target - arr), arr)

def subset_xy(da, lat_bounds, lon_bounds):
    lat_dim = "lat" if "lat" in da.dims else "latitude"
    lon_dim = "lon" if "lon" in da.dims else "longitude"
    sub = da.sel(**{
        lat_dim: slice(lat_bounds[1], lat_bounds[0]),
        lon_dim: slice(lon_bounds[0], lon_bounds[1])
    })
    return sub, lat_dim, lon_dim

def prep_for_map(da2d, lat_dim, lon_dim, upsample=8, smooth_sigma=1.2,
                 pin=None, emphasize=True):
    lat = da2d[lat_dim].values
    lon = da2d[lon_dim].values
    arr = np.asarray(da2d.values, dtype=float)

    # upsample → mask(≤10mi) → inpaint → smooth → (optional) emphasize pinpoint → mask outside
    lat_u, lon_u, arr_u = upsample_field(lat, lon, arr, factor=upsample)
    coast = build_water_mask_within_10mi(lon_u, lat_u, coast_miles=10, erode_px=2)
    masked = np.where(coast, arr_u, np.nan)
    filled = inpaint_nearest(masked, coast)
    smoothed = smooth_within_mask(filled, coast, sigma=smooth_sigma)
    if pin and emphasize:
        smoothed = emphasize_pinpoint(smoothed, lat_u, lon_u, pin[0], pin[1],
                                      sigma_km=6.0, strength=0.85)
    smoothed[~coast] = np.nan
    return lon_u, lat_u, smoothed, coast

def _draw_pin(ax, pin):
    # red outer dot with white glow + small inner dot
    (plat, plon) = pin
    h = ax.plot(plon, plat, 'o', ms=13, mfc='red', mec='white', mew=2.5,
                transform=ccrs.PlateCarree(), zorder=5)
    # subtle glow
    h[0].set_path_effects([pe.withStroke(linewidth=5, foreground="white", alpha=0.7)])

def plot_panel(ax, lon, lat, arr, title, pin, vlim=None):
    proj = ccrs.PlateCarree()
    ax.set_extent([-122.3, -121.7, 36.4, 37.0], crs=proj)
    ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=1)
    ax.coastlines(resolution="10m", linewidth=0.8, color="k", zorder=3)

    # limits + mildly contrast-boosted norm to help bloom stand out
    if vlim is None:
        finite = np.isfinite(arr); vmin = np.nanpercentile(arr[finite], 2) if finite.any() else 0
        vmax = np.nanpercentile(arr[finite], 98) if finite.any() else 1
    else:
        vmin, vmax = vlim
    norm = PowerNorm(gamma=0.85, vmin=vlim[0], vmax=vlim[1])

    im = ax.pcolormesh(
        lon, lat, np.ma.masked_invalid(arr),
        cmap="viridis", norm=norm, transform=proj,
        shading="gouraud",       # smooth color transitions
        edgecolors="none", antialiased=False,
        zorder=2
    )

    _draw_pin(ax, pin)
    ax.set_title(title, fontsize=12, weight="bold")
    return im

def select_pre_post_frames(da, event_dt):
    """Return the 8-day frame at/just before event_dt, and the next frame after."""
    t = da["time"].values
    ev = np.datetime64(event_dt)
    i_post = np.searchsorted(t, ev, side="right")     # first strictly AFTER the event
    i_pre  = max(i_post - 1, 0)                       # frame at/just before the event
    i_post = min(i_post, t.size - 1)
    return da.isel(time=i_pre), da.isel(time=i_post)

# --------------------- main ---------------------
def main():
    p = argparse.ArgumentParser(description="Monterey HAB case-study panels (pre/post 8-day)")
    p.add_argument("--obs", required=True)
    p.add_argument("--pred", action="append", required=True, help="file:model_name")
    p.add_argument("--region", default="monterey")
    p.add_argument("--event-date", default="2021-05-25")
    p.add_argument("--window-days", type=int, default=56)
    p.add_argument("--upsample", type=int, default=8)
    p.add_argument("--smooth-sigma", type=float, default=1.2)
    p.add_argument("--bloom-lat", type=float, default=36.609)
    p.add_argument("--bloom-lon", type=float, default=-121.890)
    p.add_argument("--output-dir", default="bloom_panels")
    p.add_argument("--no-emphasize", action="store_true", help="disable visual bump at pinpoint")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    pin = (args.bloom_lat, args.bloom_lon)
    event_dt = datetime.strptime(args.event_date, "%Y-%m-%d")
    pre_start, pre_end = event_dt - timedelta(days=7), event_dt
    post_start, post_end = event_dt, event_dt + timedelta(days=7)

    # region bounds (your current box)
    bounds = dict(lat=[36.4, 37.0], lon=[-122.3, -121.7])

    # load observed
    obs = xr.open_dataset(args.obs)
    obs_var = choose_var(obs, ("log_chl",))
    obs = obs[obs_var]

    # load predictions
    models = {}
    for pair in args.pred:
        path, name = pair.split(":")
        ds = xr.open_dataset(path)
        var = choose_var(ds, ("log_chl_pred","log_chl"))
        models[name] = ds[var]

    # build dict in display order
    layers = {"Observed": obs}
    layers.update(models)  # ConvLSTM, TFT, PINN (order by your args)

    # compute composites and prep arrays once to share color limits
    pre_maps, post_maps = {}, {}
    pre_vals, post_vals = [], []

    for name, da in layers.items():
        sub, lat_dim, lon_dim = subset_xy(da, bounds["lat"], bounds["lon"])

        pre, post = select_pre_post_frames(sub, event_dt)

        lon_u, lat_u, arr_pre, _ = prep_for_map(pre, lat_dim, lon_dim,
                                upsample=args.upsample, smooth_sigma=args.smooth_sigma,
                                pin=pin, emphasize=not args.no_emphasize)
        lon_u2, lat_u2, arr_post, _ = prep_for_map(post, lat_dim, lon_dim,
                                upsample=args.upsample, smooth_sigma=args.smooth_sigma,
                                pin=pin, emphasize=not args.no_emphasize)

        pre_maps[name] = (lon_u, lat_u, arr_pre)
        post_maps[name] = (lon_u2, lat_u2, arr_post)

        pre_vals.append(arr_pre[np.isfinite(arr_pre)])
        post_vals.append(arr_post[np.isfinite(arr_post)])

    # get observed arrays for anchoring vlims
    obs_pre_arr  = pre_maps["Observed"][2]
    obs_post_arr = post_maps["Observed"][2]

    def _finite(x): 
        return x[np.isfinite(x)] if x is not None else np.array([])

    obs_both = np.concatenate([_finite(obs_pre_arr), _finite(obs_post_arr)])
    if obs_both.size:
        vmin = np.nanpercentile(obs_both, 2)
        vmax = np.nanpercentile(obs_both, 98)
    else:
        # fallback to union of all layers
        pre_all  = np.concatenate([_finite(v[2]) for v in pre_maps.values()])
        post_all = np.concatenate([_finite(v[2]) for v in post_maps.values()])
        both     = np.concatenate([pre_all, post_all]) if pre_all.size and post_all.size else pre_all
        vmin = np.nanpercentile(both, 2) if both.size else 0.0
        vmax = np.nanpercentile(both, 98) if both.size else 1.0

    if vmax <= vmin:
        vmax = vmin + 1e-6
    vlim = (vmin, vmax)

    obs_max = np.nanmax(obs_both) if obs_both.size else vlim[1]
    vlim = (vlim[0], max(vlim[1], obs_max))

    # -------- PRE panel --------
    fig = plt.figure(figsize=(12, 10))
    titles = list(pre_maps.keys())
    for i, name in enumerate(titles):
        ax = plt.subplot(2, 2, i+1, projection=ccrs.PlateCarree())
        lon, lat, arr = pre_maps[name]
        im = plot_panel(ax, lon, lat, arr, name, pin, vlim=vlim)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = PowerNorm(gamma=0.85, vmin=vlim[0], vmax=vlim[1])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                  cax=cax, orientation="vertical")
    cb.set_label("log Chlorophyll-a (ln mg m⁻³)", fontsize=12, weight="bold")
    fig.suptitle(f"Monterey Bay HAB — Pre composite (center: {args.event_date})", fontsize=14, weight="bold", y=0.98)
    fig.tight_layout(rect=[0,0,0.9,0.96])
    pre_path = os.path.join(args.output_dir, "pre_panel.png")
    fig.savefig(pre_path, dpi=300)
    plt.close(fig)

    # -------- POST panel --------
    fig = plt.figure(figsize=(12, 10))
    for i, name in enumerate(titles):
        ax = plt.subplot(2, 2, i+1, projection=ccrs.PlateCarree())
        lon, lat, arr = post_maps[name]
        im = plot_panel(ax, lon, lat, arr, name, pin, vlim=vlim)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = Normalize(vmin=vlim[0], vmax=vlim[1])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                      cax=cax, orientation="vertical")
    cb.set_label("log Chlorophyll-a (ln mg m⁻³)", fontsize=12, weight="bold")
    fig.suptitle(f"Monterey Bay HAB — Post composite (center: {args.event_date})", fontsize=14, weight="bold", y=0.98)
    fig.tight_layout(rect=[0,0,0.9,0.96])
    post_path = os.path.join(args.output_dir, "post_panel.png")
    fig.savefig(post_path, dpi=300)
    plt.close(fig)

    print(f"✓ Saved:\n  {pre_path}\n  {post_path}")

if __name__ == "__main__":
    main()
