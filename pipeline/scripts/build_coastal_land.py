# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "pillow>=10.0", "shapely>=2.0", "mapbox-vector-tile>=2.0", "netCDF4>=1.6"]
# ///
"""Build the Ocean Map's coastal land cover: the land relief, opaque over land and clear over water,
in a band along the ocean coast (written to coastwatch-web/public/basemap/).

Why: data rasters (C-HARM at 0.03°, satellite products) are drawn as whole cells, so a cell that
straddles the shore paints its colour over land as a 3 km square. Drawing this layer *above* the
data rasters and *below* the coastline line clips every data layer to the water the basemap
shows, without changing any value: the part of a cell over water is drawn exactly as before, the
part over land is covered by the same land relief that lies underneath. Gaps in the data (cells
with no value) are untouched and keep their no-data marking.

The coastline is the basemap's own: water polygons from the OpenFreeMap vector tiles the map
draws (OpenMapTiles schema, © OpenStreetMap contributors, ODbL), at zoom 10, so the clip matches
the drawn shoreline. Coverage is computed at 4x4 sub-pixels per image pixel, so the edge is
anti-aliased rather than stair-stepped. Outside a ~10 km band along the ocean coast the layer is
fully transparent (no data raster reaches farther inland), so lakes, rivers and inland relief are
left to the layers beneath.

Outputs (pixel grid identical to relief-land.webp: same corners, Mercator-uniform rows):
- coastal-land.webp       relief-land colours, alpha = land coverage (relief on)
- coastal-land-flat.webp  the flat land tone, same alpha (relief off)
- coastal-land.json       sources, licence, zoom, band and build date

usage: uv run --script pipeline/scripts/build_coastal_land.py <ca15s.nc> <tile-cache-dir> <coastwatch-web/public/basemap>
       (ca15s.nc: the ETOPO 2022 15" extract used by build_relief.py; only used to find coastal tiles)
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import mapbox_vector_tile
import numpy as np
from netCDF4 import Dataset
from PIL import Image, ImageDraw
from shapely.geometry import MultiPolygon, Polygon, box, shape
from shapely.ops import transform, unary_union

Z = 10
TILEJSON = "https://tiles.openfreemap.org/planet"
BAND_DEG = 0.1  # how far inland the cover reaches from the ocean coast (~9-11 km)
SS = 4  # sub-pixels per side for coverage
LAND_RGB = (0x2A, 0x36, 0x46)  # basemap.ts LAND
R = 6378137.0
UA = {"User-Agent": "coastwatch-basemap-build/1 (one-off build of a static basemap asset; github.com/yashnil/habs-forecast)"}


def get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def merc_y(lat):
    return R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def tile_bounds(x, y, z):
    n = 2**z
    lon0 = x / n * 360 - 180
    lon1 = (x + 1) / n * 360 - 180
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon0, lat0, lon1, lat1


def tile_xy(lon, lat, z):
    n = 2**z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def coastal_tiles(nc_path, west, south, east, north):
    nc = Dataset(nc_path)
    names = list(nc.variables)
    zname = next(n for n in names if n.lower() in ("z", "elevation", "band1"))
    lat = np.asarray(nc.variables[next(n for n in names if n.lower().startswith("lat"))][:])
    lon = np.asarray(nc.variables[next(n for n in names if n.lower().startswith("lon"))][:])
    z = np.asarray(nc.variables[zname][:], dtype=np.float32)
    x0, y1 = tile_xy(west, south, Z)
    x1, y0 = tile_xy(east, north, Z)
    out = []
    pad = BAND_DEG + 0.05
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            lo0, la0, lo1, la1 = tile_bounds(x, y, Z)
            ri = (lat >= la0 - pad) & (lat <= la1 + pad)
            ci = (lon >= lo0 - pad) & (lon <= lo1 + pad)
            if not ri.any() or not ci.any():
                continue
            sub = z[np.ix_(ri, ci)]
            # ocean coast: real sea (< -5 m, so Death Valley and the Salton Sink do not count) next to land
            if (sub >= 0).any() and (sub < -5).any():
                out.append((x, y))
    return out


def fetch_water(tiles, cache: Path):
    tmpl = json.loads(get(TILEJSON))["tiles"][0]
    cache.mkdir(parents=True, exist_ok=True)
    water, ocean = [], []
    for i, (x, y) in enumerate(tiles):
        f = cache / f"{Z}-{x}-{y}.pbf"
        if not f.exists():
            for attempt in range(4):
                try:
                    f.write_bytes(get(tmpl.format(z=Z, x=x, y=y)))
                    break
                except (OSError, urllib.error.URLError):
                    time.sleep(2 * (attempt + 1))
            time.sleep(0.05)
        raw = f.read_bytes()
        if not raw:
            continue
        d = mapbox_vector_tile.decode(raw)
        if "water" not in d:
            continue
        lyr = d["water"]
        ext = lyr.get("extent", 4096)
        lo0, la0, lo1, la1 = tile_bounds(x, y, Z)
        my0, my1 = merc_y(la0), merc_y(la1)

        def to_ll(px, py, lo0=lo0, lo1=lo1, my0=my0, my1=my1, ext=ext):
            lonv = lo0 + np.asarray(px) / ext * (lo1 - lo0)
            # mapbox_vector_tile returns y up (0 at the bottom) by default
            myv = my0 + np.asarray(py) / ext * (my1 - my0)
            latv = np.degrees(2 * np.arctan(np.exp(myv / R)) - np.pi / 2)
            return lonv, latv

        tb = box(lo0, la0, lo1, la1)
        for feat in lyr["features"]:
            g = shape(feat["geometry"])
            if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            g = transform(to_ll, g).buffer(0).intersection(tb)
            if g.is_empty:
                continue
            water.append(g)
            if feat["properties"].get("class") == "ocean":
                ocean.append(g)
        if i % 50 == 0:
            print(f"  tiles {i + 1}/{len(tiles)}", flush=True)
    return unary_union(water), unary_union(ocean)


def polys(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def main(nc_path: str, cache: str, basemap: str) -> None:
    bm = Path(basemap)
    meta = json.loads((bm / "relief.json").read_text())
    (west, north), (east, _), (_, south), _ = meta["corners_lnglat"]
    relief = Image.open(bm / "relief-land.webp").convert("RGB")
    W, H = relief.size
    tiles = coastal_tiles(nc_path, west, south, east, north)
    print(f"{len(tiles)} coastal z{Z} tiles")
    water, ocean = fetch_water(tiles, Path(cache))
    # only where the water is known: a tile that was not fetched must never read as land
    fetched = unary_union([box(*tile_bounds(x, y, Z)) for x, y in tiles])
    band = ocean.buffer(BAND_DEG, resolution=8).intersection(fetched).intersection(box(west, south, east, north))
    land = band.difference(water)
    print("land band area (deg²)", round(land.area, 3))

    # rasterize land at SS x SS sub-pixels on the relief grid (Mercator-uniform rows)
    yt, yb = merc_y(north), merc_y(south)
    sw, sh = W * SS, H * SS

    def px(lon, lat):
        return (np.asarray(lon) - west) / (east - west) * sw, (yt - merc_y(np.asarray(lat))) / (yt - yb) * sh

    mask = Image.new("L", (sw, sh), 0)
    dr = ImageDraw.Draw(mask)
    for p in polys(land):
        xs, ys = px(*p.exterior.xy)
        dr.polygon(list(zip(xs.tolist(), ys.tolist())), fill=255)
        for hole in p.interiors:
            hx, hy = px(*hole.xy)
            dr.polygon(list(zip(hx.tolist(), hy.tolist())), fill=0)
    alpha = mask.resize((W, H), Image.BOX)
    a = np.asarray(alpha)
    print("pixels with any land cover", int((a > 0).sum()), "partial", int(((a > 0) & (a < 255)).sum()))

    rgba = relief.copy()
    rgba.putalpha(alpha)
    rgba.save(bm / "coastal-land.webp", "WEBP", quality=88, alpha_quality=100, method=6)
    flat = Image.new("RGB", (W, H), LAND_RGB)
    flat.putalpha(alpha)
    flat.save(bm / "coastal-land-flat.webp", "WEBP", lossless=True, method=6)
    (bm / "coastal-land.json").write_text(
        json.dumps(
            {
                "corners_lnglat": meta["corners_lnglat"],
                "width": W,
                "height": H,
                "coastline": f"OpenFreeMap vector tiles (OpenMapTiles schema), water layer, zoom {Z}, {len(tiles)} coastal tiles",
                "coastline_license": "© OpenStreetMap contributors, ODbL; OpenMapTiles; OpenFreeMap",
                "band_deg": BAND_DEG,
                "supersample": SS,
                "relief": "relief-land.webp (see relief.json)",
                "purpose": "Drawn above data rasters and below the coastline line, so whole-cell data rasters stop at the drawn shoreline. Values are unchanged; cells without a value keep their no-data marking.",
                "built_at": datetime.now(UTC).strftime("%Y-%m-%d"),
            },
            indent=1,
        )
        + "\n"
    )
    for n in ("coastal-land.webp", "coastal-land-flat.webp"):
        print(n, (bm / n).stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:4])
