#!/usr/bin/env python3
"""
Region overviews with red bounding boxes (two separate images)

- Uses your dataset's lat/lon grid to set the same California snapshot extent.
- Draws a red rectangle around the requested subdomain (Monterey or Navarro).
- Exports two images: monterey_overview.png and navarro_overview.png

Usage:
python region_overviews.py \
  --obs "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc" \
  --outdir ./overviews

Optional:
  --no_tiles    (disable satellite/terrain tiles)
  --dpi 350
"""

import argparse, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

from shapely.geometry import box as shp_box

# ---------------- constants ----------------
# (lon_min, lon_max, lat_min, lat_max)
MONTEREY_BOX = (-122.30, -121.70, 36.40, 37.00)
NAVARRO_BOX  = (-124.30, -123.70, 38.90, 39.50)

# style
plt.rcParams.update({"font.family": "DejaVu Sans"})
BOX_COLOR = "#e41a1c"      # red
BOX_LW    = 2.2
LABEL_KW  = dict(color="white", weight="bold", fontsize=11,
                 path_effects=[pe.withStroke(linewidth=2.5, foreground="0.15")])

# ---------------- helpers ----------------
def choose(mapping, names):
    for n in names:
        if n in mapping: return n
    raise KeyError(f"None of {names} in {list(mapping)}")

def add_tiles(ax, zoom=7):
    """ESRI World Imagery; fallback to Stamen Terrain."""
    try:
        class Esri(cimgt.GoogleWTS):
            def _image_url(self, tile):
                x, y, z = tile
                return (f"https://server.arcgisonline.com/ArcGIS/rest/services/"
                        f"World_Imagery/MapServer/tile/{z}/{y}/{x}")
        ax.add_image(Esri(), zoom, interpolation="bilinear")
    except Exception:
        ax.add_image(cimgt.Stamen("terrain-background"), zoom-1, interpolation="bilinear")

def compute_extent_from_grid(LON, LAT, pad_lon=0.6, pad_lat=0.4):
    """Tight extent around grid with a bit of padding."""
    lon_min = float(np.nanmin(LON)); lon_max = float(np.nanmax(LON))
    lat_min = float(np.nanmin(LAT)); lat_max = float(np.nanmax(LAT))
    return (lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat)

def draw_overview(ax, proj, extent, box, label, show_tiles=True):
    ax.set_extent(extent, crs=proj)
    if show_tiles: add_tiles(ax)
    ax.coastlines(resolution="10m", linewidth=0.8, color="k", alpha=0.9)
    # bounding box
    x0, x1, y0, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                               ec=BOX_COLOR, lw=BOX_LW, transform=proj, zorder=5))
    ax.text(x0, y1 + 0.05*(extent[3]-extent[2]), label,
            transform=proj, ha="left", va="bottom", **LABEL_KW)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True, help="NetCDF with lat/lon grid (same as your model/obs)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--no_tiles", action="store_true")
    ap.add_argument("--dpi", type=int, default=350)
    # allow custom dim names (defaults cover your datasets)
    ap.add_argument("--lat_names", nargs="+", default=["lat", "latitude", "y"])
    ap.add_argument("--lon_names", nargs="+", default=["lon", "longitude", "x"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ds = xr.open_dataset(args.obs)
    dlat = choose(ds.dims, args.lat_names)
    dlon = choose(ds.dims, args.lon_names)

    lat = ds[dlat].values
    lon = ds[dlon].values
    # build 2D if needed
    if lat.ndim == 1 and lon.ndim == 1:
        LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    else:
        LAT, LON = lat, lon

    extent = compute_extent_from_grid(LON, LAT)

    proj = ccrs.PlateCarree()

    # ---- Monterey overview ----
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(1,1,1, projection=proj)
    draw_overview(ax, proj, extent, MONTEREY_BOX, "Monterey Bay",
                  show_tiles=(not args.no_tiles))
    ax.set_title("California overview — Monterey Bay box", fontsize=12, weight="bold", pad=6)
    fig.savefig(os.path.join(args.outdir, "monterey_overview.png"),
                dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # ---- Navarro overview ----
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(1,1,1, projection=proj)
    draw_overview(ax, proj, extent, NAVARRO_BOX, "Navarro River",
                  show_tiles=(not args.no_tiles))
    ax.set_title("California overview — Navarro River box", fontsize=12, weight="bold", pad=6)
    fig.savefig(os.path.join(args.outdir, "navarro_overview.png"),
                dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("✓ Saved:",
          os.path.abspath(os.path.join(args.outdir, "monterey_overview.png")))
    print("✓ Saved:",
          os.path.abspath(os.path.join(args.outdir, "navarro_overview.png")))

if __name__ == "__main__":
    main()
