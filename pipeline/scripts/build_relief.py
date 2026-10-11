# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.11", "pillow>=10.0", "contourpy>=1.2"]
# ///
"""Build the Ocean Map's relief basemap from NOAA NCEI ETOPO 2022 (15 arc-second).

Outputs, written to coastwatch-web/public/basemap/ and committed (they change only when
this script changes):

- relief-sea.webp: seafloor shading for depths below 10 m. A neutral tone that is darker
  than every data colour, lightened slightly over the shelf, with hillshade so the shelf
  break and the canyons read. It is drawn *under* every data layer, so it can never be
  mistaken for a value; gaps in data keep their no-data hatch on top of it.
- relief-land.webp: land hillshade on the basemap's land tone (drawn under the vector
  water, which clips it to the real coastline).
- isobaths.geojson: 200, 500, 1000, 2000 and 3000 m depth contours, simplified.
- relief.json: image corners, the source, the licence and the citation.

Both images are resampled onto pixels uniform in Web Mercator y, so placing them by their
four corners is exact (see coastwatch_pipeline/process/mercator.py).

Data: ETOPO 2022 v1, 15 arc-second, Ice Surface; NOAA National Centers for Environmental
Information, doi:10.25921/fd45-gt74. Free to use and redistribute; not for navigation.
Fetched from NOAA CoastWatch ERDDAP (dataset ETOPO_2022_v1_15s). Optional Monterey Bay inset:
NOAA NCEI U.S. Coastal Relief Model volumes 6 and 7 (3 arc-second, ~90 m), public domain,
read over OPeNDAP from https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol{6,7}.nc for
-122.75..-121.55 E, 36.2..37.35 N and saved as an npz (lat, lon, z).

usage: uv run --script pipeline/scripts/build_relief.py <etopo.nc> <out-dir> [monterey_crm3s.npz]
       (download: https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.nc?z[(31.9):1:(42.7)][(-126.8):1:(-116.2)])
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import contourpy
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.io import netcdf_file

R = 6378137.0
WEST, EAST, SOUTH, NORTH = -126.6, -116.4, 32.0, 42.6
# basemap tones (coastwatch-web/src/lib/basemap.ts): SEA is the deep-water colour, so the
# image's deep ocean equals the vector water and its edges are invisible
SEA = np.array([0x0B, 0x1D, 0x33], float)
SHELF = np.array([0x1A, 0x2B, 0x3E], float)  # neutral; darker than the lowest chlorophyll class (#102a4c is bluer, #14466e lighter)
LAND = np.array([0x2A, 0x36, 0x46], float)
ISOBATHS = [200, 500, 1000, 2000, 3000]


def merc_y(lat):
    return R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def lat_of(y):
    return np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)


def hillshade(z, dx, dy, exaggeration, azimuth=315.0, altitude=45.0):
    """Lambertian hillshade in [0, 1]; dx per row (ground metres), dy scalar."""
    gy, gx = np.gradient(z * exaggeration, dy, axis=0), np.gradient(z * exaggeration, axis=1) / dx[:, None]
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)  # rows run north to south
    az, alt = np.radians(azimuth), np.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def edge_fade(h, w, px=48):
    """1 inside, falling smoothly to 0 over the outer `px` pixels."""
    r = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    c = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    t = np.clip(np.minimum(r, c) / px, 0, 1)
    return t * t * (3 - 2 * t)


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification (iterative)."""
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = pts[b] - pts[a]
        n = np.hypot(*seg) or 1e-12
        v = pts[a + 1 : b] - pts[a]
        d = np.abs(seg[0] * v[:, 1] - seg[1] * v[:, 0]) / n
        i = int(np.argmax(d))
        if d[i] > eps:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return pts[keep]


def to_mercator(lat, lon, z, west, east, south, north):
    """Bilinear resample of a regular lat/lon grid onto pixels uniform in Mercator y."""
    step = lon[1] - lon[0]
    width = int(round((east - west) / step))
    px = np.radians(east - west) * R / width
    y_top, y_bot = merc_y(north), merc_y(south)
    height = int(round((y_top - y_bot) / px))
    rows_lat = lat_of(y_top - (np.arange(height) + 0.5) * px)
    cols_lon = west + (np.arange(width) + 0.5) * step
    ri = np.interp(rows_lat, lat, np.arange(len(lat)))
    ci = np.interp(cols_lon, lon, np.arange(len(lon)))
    RR, CC = np.meshgrid(ri, ci, indexing="ij")
    zm = ndimage.map_coordinates(z, [RR, CC], order=1, mode="nearest")
    return zm, px * np.cos(np.radians(rows_lat)), width, height


def sea_rgba(zm, dx, exaggeration, fade):
    depth = np.clip(-zm, 0, None)
    hs = hillshade(np.minimum(zm, 0), dx, float(np.mean(dx)), exaggeration=exaggeration)
    t = np.clip(np.log1p(depth / 40) / np.log1p(3000 / 40), 0, 1)
    base = SHELF[None, None, :] * (1 - t[..., None]) + SEA[None, None, :] * t[..., None]
    shade = (hs - np.cos(np.radians(45))) * 0.55
    rgb = base * (1 + shade[..., None] * fade[..., None])
    alpha = np.clip((depth - 6) / 8, 0, 1) * 255 * fade
    return np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8)


def crm_inset(npz: Path, out: Path) -> dict:
    """Monterey Bay seafloor from the NOAA NCEI Coastal Relief Model (3 arc-second, volumes 6
    and 7), drawn over the ETOPO relief with soft edges. Same tones as the ETOPO sea."""
    d = np.load(npz)
    lat, lon, z = d["lat"], d["lon"], d["z"].astype(float)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    W, E, S, N = -122.72, -121.58, 36.22, 37.33
    zm, dx, w, h = to_mercator(lat, lon, z, W, E, S, N)
    img = sea_rgba(zm, dx, 2.0, edge_fade(h, w, px=90))
    Image.fromarray(img, "RGBA").save(out / "relief-sea-monterey.webp", "WEBP", quality=82, method=6, alpha_quality=60)
    return {"corners_lnglat": [[W, N], [E, N], [E, S], [W, S]], "width": w, "height": h,
            "source": "NOAA NCEI U.S. Coastal Relief Model, volumes 6 and 7, 3 arc-second", "source_url": "https://www.ncei.noaa.gov/products/coastal-relief-model"}


def main(src: Path, out: Path, crm: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with netcdf_file(src, "r", mmap=False) as nc:
        lat = nc.variables["latitude"][:].astype(float)
        lon = nc.variables["longitude"][:].astype(float)
        z = nc.variables["z"][:].astype(float)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]

    # ---- Mercator target grid at the source's longitude resolution
    step = lon[1] - lon[0]
    width = int(round((EAST - WEST) / step))
    px = np.radians(EAST - WEST) * R / width  # Mercator metres per pixel
    y_top, y_bot = merc_y(NORTH), merc_y(SOUTH)
    height = int(round((y_top - y_bot) / px))
    rows_lat = lat_of(y_top - (np.arange(height) + 0.5) * px)
    cols_lon = WEST + (np.arange(width) + 0.5) * step
    ri = np.interp(rows_lat, lat, np.arange(len(lat)))
    ci = np.interp(cols_lon, lon, np.arange(len(lon)))
    RR, CC = np.meshgrid(ri, ci, indexing="ij")
    zm = ndimage.map_coordinates(z, [RR, CC], order=1, mode="nearest")
    print(f"mercator image {width}x{height}, depth range {zm.min():.0f}..{zm.max():.0f} m")

    # ground spacing per row (Mercator pixels shrink with cos(lat))
    dx = px * np.cos(np.radians(rows_lat))
    fade = edge_fade(height, width)

    # ---- seafloor
    depth = np.clip(-zm, 0, None)
    hs_sea = hillshade(np.minimum(zm, 0), dx, float(np.mean(dx)), exaggeration=4.0)
    # depth tone: shelf (≤ 200 m) lightest, easing to the deep-water colour by ~3000 m
    t = np.clip(np.log1p(depth / 40) / np.log1p(3000 / 40), 0, 1)
    base = SHELF[None, None, :] * (1 - t[..., None]) + SEA[None, None, :] * t[..., None]
    shade = (hs_sea - np.cos(np.radians(45))) * 0.55  # 0 on flat floor
    rgb = base * (1 + shade[..., None] * fade[..., None])
    alpha = np.clip((depth - 6) / 8, 0, 1) * 255  # nothing on land or in the surf zone
    sea = np.dstack([np.clip(rgb, 0, 255), alpha]).astype(np.uint8)
    Image.fromarray(sea, "RGBA").save(out / "relief-sea.webp", "WEBP", quality=82, method=6, alpha_quality=60)

    # ---- land
    hs_land = hillshade(np.maximum(zm, 0), dx, float(np.mean(dx)), exaggeration=1.6)
    lshade = (hs_land - np.cos(np.radians(45))) * 0.9
    lrgb = LAND[None, None, :] * (1 + lshade[..., None])
    land = np.dstack([np.clip(lrgb, 0, 255), fade * 255]).astype(np.uint8)
    Image.fromarray(land, "RGBA").save(out / "relief-land.webp", "WEBP", quality=80, method=6, alpha_quality=50)

    # ---- isobaths on the geographic grid (lightly smoothed so 15" stair steps don't show)
    zs = ndimage.gaussian_filter(z, 1.2)
    gen = contourpy.contour_generator(lon, lat, zs, line_type=contourpy.LineType.Separate)
    feats = []
    for d in ISOBATHS:
        for seg in gen.lines(-d):
            s = rdp(np.asarray(seg), 0.004)
            if len(s) < 4 or np.hypot(*(s.max(0) - s.min(0))) < 0.08:
                continue
            feats.append({
                "type": "Feature",
                "properties": {"depth_m": d, "label": f"{d:,} m"},
                "geometry": {"type": "LineString", "coordinates": np.round(s, 4).tolist()},
            })
    (out / "isobaths.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))

    meta = {
        "corners_lnglat": [[WEST, NORTH], [EAST, NORTH], [EAST, SOUTH], [WEST, SOUTH]],
        "width": width,
        "height": height,
        "isobaths_m": ISOBATHS,
        "source": "ETOPO 2022 v1, 15 arc-second (Ice Surface), NOAA National Centers for Environmental Information",
        "source_url": "https://www.ncei.noaa.gov/products/etopo-global-relief-model",
        "doi": "https://doi.org/10.25921/fd45-gt74",
        "license": "Free to use and redistribute; not intended for legal use or navigation (NOAA NCEI).",
        "rendering": "Seafloor and land hillshade (azimuth 315°, altitude 45°, vertical exaggeration 4x sea / 1.6x land); depth tone lightest on the shelf; isobaths from a 1.2-cell Gaussian-smoothed grid, simplified to about 400 m.",
    }
    if crm:
        meta["monterey_inset"] = crm_inset(crm, out)
    (out / "relief.json").write_text(json.dumps(meta, indent=2) + "\n")
    for f in sorted(out.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
