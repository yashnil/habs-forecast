#!/usr/bin/env python3
"""
Figure 1 — Study domain & alongshore context (2 panels)

(A) Study domain with satellite/terrain basemap, bright 10-mile coastal strip,
    latitudinal bands, and a *short curated* set of coastal landmarks (incl. key rivers).
(B) Alongshore (latitude) coastal climatology of log-chl (median ± IQR) over the coastal strip.

Usage:
python study_domain.py \
  --obs "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --out study_domain.png

Optional flags:
  --no_tiles            (disable basemap tiles)
"""
import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.io.shapereader as shpreader

from shapely.geometry import Point
from shapely.ops import unary_union, transform as shp_transform
from shapely.prepared import prep as shp_prep
from pyproj import Transformer
from scipy.ndimage import binary_erosion

# ----------------------------- preferences -----------------------------
PREF_VARS_CHL = ("log_chl", "chl_log", "chl_ln", "chlor_a_log")
PREF_DIMS_LAT = ("lat", "latitude", "y")
PREF_DIMS_LON = ("lon", "longitude", "x")

MILES_TO_METERS = 1609.344
COAST_MILES = 10.0           # ≤ 10 miles mask
ERODE_PX = 1                 # erode mask slightly to avoid land bleed
TILE_ZOOM = 7                # basemap zoom (6–8 looks good for CA)

# Aesthetics
plt.rcParams.update({"font.family": "DejaVu Sans"})
BAND_COLORS = ["#fdd49e", "#fee8c8", "#f6e8c3", "#ccebc5", "#b3cde3"]

# Bright coastal strip
COAST_FILL_COLOR = "#f1c40f"   # vivid yellow
COAST_FILL_ALPHA = 0.50
COAST_EDGE_COLOR = "#8a6d00"
COAST_EDGE_LW = 1.8

# Curated *coastal* landmarks only (lon, lat, label)
LANDMARKS = [
    (-124.067, 41.546, "Klamath River"),
    (-124.356, 40.718, "Eel River"),
    (-123.743, 39.128, "Navarro River"),
    (-123.740, 38.950, "Point Arena"),
    (-122.974, 38.020, "Point Reyes"),
    (-122.419, 37.775, "San Francisco"),
    (-122.030, 36.974, "Santa Cruz"),
    (-121.894, 36.600, "Monterey"),
    (-120.855, 35.365, "Morro Bay"),
    (-120.471, 34.449, "Point Conception"),
]

# ----------------------------- helpers -----------------------------
def choose(mapping, names):
    for n in names:
        if n in mapping:
            return n
    raise KeyError(f"None of {names} found in {list(mapping)}")

def meters_per_degree(lat_deg):
    lat = np.deg2rad(lat_deg)
    m_per_deg_lat = 111_132.954 - 559.822*np.cos(2*lat) + 1.175*np.cos(4*lat)
    m_per_deg_lon = 111_132.954*np.cos(lat)
    return float(m_per_deg_lon), float(m_per_deg_lat)

def build_coast_mask_and_distance(lon2d, lat2d, miles=10.0, erode_px=1):
    """Return mask (ocean cells within ≤ miles of coastline) and distance (m)."""
    land = unary_union(list(
        shpreader.Reader(shpreader.natural_earth("10m", "physical", "land")).geometries()
    ))
    land_prep = shp_prep(land)

    to_m = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True).transform
    land_m = shp_transform(to_m, land)
    coast_m = land_m.boundary
    thresh_m = miles * MILES_TO_METERS

    H, W = lat2d.shape
    mask = np.zeros((H, W), dtype=bool)
    dist_m = np.full((H, W), np.nan, dtype=float)

    for i in range(H):
        for j in range(W):
            lo, la = float(lon2d[i, j]), float(lat2d[i, j])
            if land_prep.contains(Point(lo, la)):   # exclude land
                continue
            x, y = to_m(lo, la)
            d = Point(x, y).distance(coast_m)
            dist_m[i, j] = d
            if d <= thresh_m:
                mask[i, j] = True

    if erode_px > 0:
        mask = binary_erosion(mask, iterations=int(erode_px))
    return mask, dist_m

def compute_bands(lat2d, n=5, coast_mask=None):
    lats = lat2d[coast_mask] if coast_mask is not None else lat2d.ravel()
    qs = np.nanquantile(lats, np.linspace(0, 1, n+1))
    return list(zip(qs[:-1], qs[1:]))

def add_tiles(ax):
    """Add ESRI imagery if available; fallback to Stamen Terrain if not."""
    try:
        class Esri(cimgt.GoogleWTS):
            def _image_url(self, tile):
                x, y, z = tile
                return (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                        f"World_Imagery/MapServer/tile/{z}/{y}/{x}")
        ax.add_image(Esri(), TILE_ZOOM, interpolation="bilinear")
    except Exception:
        ax.add_image(cimgt.Stamen("terrain-background"), TILE_ZOOM-1, interpolation="bilinear")

def pretty_gridlines(ax):
    gl = ax.gridlines(draw_labels=True, xlocs=np.arange(-126, -116, 2),
                      ylocs=np.arange(32, 43, 1), color="0.7", lw=0.5, alpha=0.7)
    gl.right_labels = False
    gl.top_labels = False

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="NetCDF with log chlorophyll + lat/lon/time")
    ap.add_argument("--out", default="study_domain.png")
    ap.add_argument("--no_tiles", action="store_true", help="Disable background tiles")
    args = ap.parse_args()

    ds = xr.open_dataset(args.obs)
    vchl = choose(ds.data_vars, PREF_VARS_CHL)
    dlat = choose(ds.dims,      PREF_DIMS_LAT)
    dlon = choose(ds.dims,      PREF_DIMS_LON)
    da = ds[vchl]

    # Lat/Lon 2D
    lat = ds[dlat].values
    lon = ds[dlon].values
    if lat.ndim == 1 and lon.ndim == 1:
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
        lat1d = lat
    else:
        LAT, LON = lat, lon
        lat1d = np.nanmean(LAT, axis=1)  # synthetic 1D for plotting

    # Coastal mask (≤10 miles)
    print("Building coastal mask…")
    coast_mask, _ = build_coast_mask_and_distance(LON, LAT, miles=COAST_MILES, erode_px=ERODE_PX)

    # Map extent (pad a bit around mask)
    mask_where = np.where(coast_mask)
    lat_min = float(np.nanmin(LAT[mask_where])) - 0.4
    lat_max = float(np.nanmax(LAT[mask_where])) + 0.4
    lon_min = float(np.nanmin(LON[mask_where])) - 0.6
    lon_max = float(np.nanmax(LON[mask_where])) + 0.6
    extent = (lon_min, lon_max, lat_min, lat_max)

    # Latitudinal bands
    bands = compute_bands(LAT, n=5, coast_mask=coast_mask)

    # ----------------- Figure layout -----------------
    fig = plt.figure(figsize=(12.2, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.12)

    proj = ccrs.PlateCarree()

    # Panel A — Study domain
    axA = fig.add_subplot(gs[0, 0], projection=proj)
    axA.set_extent(extent, crs=proj)
    if not args.no_tiles: add_tiles(axA)
    axA.coastlines(resolution="10m", linewidth=0.7, color="k", alpha=0.9)

    # bands (soft tint below)
    for (lat0, lat1), col in zip(bands, BAND_COLORS):
        axA.add_patch(plt.Rectangle((lon_min, lat0), lon_max-lon_min, lat1-lat0,
                                    facecolor=col, alpha=0.22, transform=proj,
                                    edgecolor="none", zorder=0))
        axA.text(lon_min+0.1, 0.5*(lat0+lat1), f"{lat0:.1f}–{lat1:.1f}°N",
                 va="center", ha="left", fontsize=8, color="white",
                 transform=proj, zorder=7,
                 path_effects=[pe.withStroke(linewidth=2.2, foreground="0.15")])

    # bright coastal strip fill + edge
    axA.contourf(LON, LAT, coast_mask.astype(int),
                 levels=[0.5, 1.5], colors=[COAST_FILL_COLOR],
                 alpha=COAST_FILL_ALPHA, transform=proj, zorder=2)
    axA.contour(LON, LAT, coast_mask.astype(float), levels=[0.5],
                colors=[COAST_EDGE_COLOR], linewidths=COAST_EDGE_LW,
                linestyles="solid", transform=proj, zorder=3)

    # curated coastal landmarks (white bold text with dark halo)
    if LANDMARKS:
        xl, yl, _ = zip(*LANDMARKS)
        axA.scatter(xl, yl, s=22, color="#222", edgecolor="white", linewidth=0.6,
                    transform=proj, zorder=5)
        for lo, la, nm in LANDMARKS:
            axA.text(lo+0.06, la+0.02, nm, fontsize=9, color="white", weight="bold",
                     transform=proj, zorder=6,
                     path_effects=[pe.withStroke(linewidth=2.5, foreground="0.15")])

    # niceties
    def add_scalebar(ax, length_km=100, location=(0.05, 0.05), linewidth=3):
        x0, y0 = location
        ax_xmin, ax_xmax = ax.get_xbound()
        ax_ymin, ax_ymax = ax.get_ybound()
        cx = ax_xmin + (ax_xmax-ax_xmin)*x0
        cy = ax_ymin + (ax_ymax-ax_ymin)*y0
        lat_c = np.clip(np.mean(ax.get_ylim()), -80, 80)
        m_per_deg_lon, _ = meters_per_degree(lat_c)
        dlon = (length_km*1000.0) / m_per_deg_lon
        ax.plot([cx, cx+dlon], [cy, cy], transform=proj,
                color="k", lw=linewidth, solid_capstyle="butt")
        ax.text(cx+dlon/2, cy + 0.01*(ax_ymax-ax_ymin), f"{int(length_km)} km",
                ha="center", va="bottom", fontsize=9, transform=proj,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    def add_north_arrow(ax, xy=(0.96, 0.08), size=0.04):
        x, y = xy
        ax.annotate("N", xy=xy, xytext=(x, y+size), xycoords="axes fraction",
                    textcoords="axes fraction", ha="center", va="bottom",
                    fontsize=10, color="white",
                    path_effects=[pe.withStroke(linewidth=2, foreground="0.15")])
        ax.annotate("", xy=(x, y+size*0.9), xytext=(x, y),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="white",
                                    shrinkA=0, shrinkB=0,
                                    mutation_scale=12,
                                    alpha=0.9))

    add_scalebar(axA, length_km=100)
    add_north_arrow(axA)
    pretty_gridlines(axA)
    axA.set_title("(A) Study domain",
                  fontsize=12, weight="bold", pad=6)

    # Panel B — Alongshore (latitude) coastal climatology (median ± IQR)
    axB = fig.add_subplot(gs[0, 1])
    if "time" in da.dims:
        coast_da = xr.DataArray(coast_mask, dims=(dlat, dlon))
        coastal = da.where(coast_da)
        row_ts = coastal.mean(dim=dlon, skipna=True)                # dims: time, lat
        med  = row_ts.median(dim="time", skipna=True)
        q25  = row_ts.quantile(0.25, dim="time", skipna=True)
        q75  = row_ts.quantile(0.75, dim="time", skipna=True)

        for (lat0, lat1), col in zip(bands, BAND_COLORS):
            axB.axvspan(lat0, lat1, color=col, alpha=0.12, lw=0)

        axB.fill_between(lat1d, q25.values, q75.values, alpha=0.30, lw=0, label="IQR")
        axB.plot(lat1d, med.values, lw=2.4, color="#1b9e77", label="Median")

        axB.set_xlim(min(lat1d), max(lat1d))
        axB.set_xlabel("Latitude (°N)")
        axB.set_ylabel("log chlorophyll (ln mg m$^{-3}$)")
        axB.grid(True, alpha=0.35)
        t0 = str(np.datetime_as_string(da.time.values[0], unit="D"))
        t1 = str(np.datetime_as_string(da.time.values[-1], unit="D"))
        axB.set_title(f"(B) Alongshore coastal climatology",
                      fontsize=12, weight="bold", pad=6)
        axB.legend(frameon=False, loc="best")
    else:
        axB.text(0.5, 0.5, "No time dimension in dataset\n(cannot build alongshore climatology)",
                 ha="center", va="center", fontsize=11)
        axB.set_axis_off()
        axB.set_title("(B) Alongshore coastal climatology", fontsize=12, weight="bold", pad=6)

    fig.suptitle("Study domain & alongshore context",
                 fontsize=14, weight="bold", y=0.98)
    fig.savefig(args.out, dpi=350, bbox_inches="tight")
    print(f"✓ Saved: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
