# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "shapely>=2.0,<3", "mapbox-vector-tile>=2.1,<3"]
# ///
"""Build the Ocean Map's bundled water geometry, the fallback for the basemap's own water.

Data rasters (C-HARM cells, satellite tiles) are clipped to the basemap's water so that a cell
straddling the shore is drawn on water only (coastwatch-web/src/lib/waterMask.ts). The clip reads
the basemap's vector tiles; when they cannot be read (tile server down, offline, very slow), the
web app uses this file instead, so data never spill onto land. It is the *same* water: the
OpenMapTiles `water` layer served by OpenFreeMap, which the basemap draws.

Method:
- every zoom-10 tile over the domain is read: tiles all ocean become water, tiles with no ocean
  are land;
- tiles with both are read again at zoom 11 (finer) for their ocean;
- only the `ocean` class is kept (the sea, bays and estuaries up to the OSM coastline). Lakes,
  lagoons and rivers are left out: the basemap keeps or drops them by size from one zoom to the
  next, and C-HARM is an ocean model, so the fallback never draws its values on them;
- the ocean within one cell of a C-HARM ocean cell (data/reference/charm_ocean_mask.json) is
  simplified to NEAR_TOLERANCE_M, elsewhere (inner bays beyond the model, the map's margins) to
  FAR_TOLERANCE_M, which is enough for the sea drawn while the basemap is unavailable and for
  satellite pixels (750 m and coarser). Douglas-Peucker with topology kept;
- the result is cut along the zoom-SPLIT_Z tile grid (the domain is widened to whole tiles), so a
  data tile only ever reads the coastline near it; the cuts lie exactly on tile edges, where the
  web app leaves them out of the drawn coastline. Stored as Web Mercator integers (QUANT units
  per world width), ring by ring, delta encoded.

Output (committed; it changes only when this script or the domain changes):
coastwatch-web/public/basemap/water-fallback.json

usage: uv run --script pipeline/scripts/build_water_fallback.py <tile-cache-dir> <out.json>
"""

from __future__ import annotations

import gzip
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import mapbox_vector_tile
import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box, shape

TILEJSON = "https://tiles.openfreemap.org/planet"
# The Ocean Map's pannable area (CA_BOUNDS in coastwatch-web/src/lib/basemap.ts), which holds the
# C-HARM grid (-127.515..-116.985 E, 31.285..43.015 N) and the relief images.
DOMAIN = (-131.5, 29.5, -112.5, 44.8)
DETECT_Z = 10
DETAIL_Z = 11
SPLIT_Z = 8
NEAR_TOLERANCE_M = 8.0
FAR_TOLERANCE_M = 40.0
OCEAN_CELLS = Path(__file__).resolve().parents[2] / "data" / "reference" / "charm_ocean_mask.json"
QUANT = 2**26  # Web Mercator units per world width: ~0.6 m at the equator, ~0.45 m at 40 N
UA = {"User-Agent": "coastwatch-pipeline build_water_fallback (research; one-off build)"}


def merc(lon: float, lat: float) -> tuple[float, float]:
    """Normalised Web Mercator (0..1, y down)."""
    s = math.sin(math.radians(lat))
    return (lon + 180) / 360, 0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)


def tiles_over(z: int) -> list[tuple[int, int]]:
    w, s, e, n = DOMAIN
    (x0, y0), (x1, y1) = merc(w, n), merc(e, s)
    k = 2**z
    return [(x, y) for x in range(int(x0 * k), int(x1 * k) + 1) for y in range(int(y0 * k), int(y1 * k) + 1)]


def fetch(template: str, cache: Path, z: int, x: int, y: int) -> bytes:
    f = cache / f"{z}-{x}-{y}.pbf"
    if f.exists():
        return f.read_bytes()
    url = template.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                b = r.read()
            break
        except Exception:  # noqa: BLE001 - retried, then raised
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    f.write_bytes(b)
    return b


def water(data: bytes, ocean_only: bool = True) -> tuple[shapely.Geometry, int]:
    """The tile's water, cut to the tile, in tile units (y down), and the tile extent."""
    layers = mapbox_vector_tile.decode(data, default_options={"y_coord_down": True})
    lyr = layers.get("water")
    if not lyr:
        return Polygon(), 4096
    ext = lyr.get("extent", 4096)
    geoms = [
        shape(f["geometry"]).buffer(0)
        for f in lyr["features"]
        if f["geometry"]["type"] in ("Polygon", "MultiPolygon") and (not ocean_only or f["properties"].get("class") == "ocean")
    ]
    return polygonal(shapely.unary_union(geoms).intersection(box(0, 0, ext, ext))), ext


def polygonal(g: shapely.Geometry) -> shapely.Geometry:
    """Only the areas of a geometry (an intersection can add touching lines and points)."""
    return MultiPolygon([p for p in shapely.get_parts(g) if isinstance(p, Polygon) and not p.is_empty])


def to_world(g: shapely.Geometry, z: int, x: int, y: int, ext: int) -> shapely.Geometry:
    k = 1 / (ext * 2**z)
    return affinity.affine_transform(g, [k, 0, 0, k, x / 2**z, y / 2**z])


def near_forecast(grow: int = 1) -> shapely.Geometry:
    """C-HARM ocean cells, grown by `grow` cells, as one Web Mercator area."""
    ref = json.loads(OCEAN_CELLS.read_text())
    h, w = ref["height"], ref["width"]
    m = np.zeros((h + 2, w + 2), bool)
    for r, runs in enumerate(ref["rows_rle"]):
        for s, n in runs:
            m[r + 1, s + 1 : s + 1 + n] = True
    grown = np.zeros_like(m)
    for dr in range(-grow, grow + 1):
        for dc in range(-grow, grow + 1):
            grown |= np.roll(np.roll(m, dr, 0), dc, 1)
    grown = grown[1:-1, 1:-1]
    boxes = []
    for r in range(h):
        lat = ref["lat_first"] + r * ref["lat_step"]
        (_, ya), (_, yb) = merc(0, lat + abs(ref["lat_step"]) / 2), merc(0, lat - abs(ref["lat_step"]) / 2)
        row = np.flatnonzero(np.diff(np.concatenate([[0], grown[r].astype(np.int8), [0]])))
        for c0, c1 in zip(row[::2], row[1::2]):
            (xa, _), (xb, _) = merc(ref["lon_first"] + (c0 - 0.5) * ref["lon_step"], 0), merc(ref["lon_first"] + (c1 - 0.5) * ref["lon_step"], 0)
            boxes.append(box(xa, ya, xb, yb))
    return shapely.unary_union(boxes)


def main() -> None:
    cache, out = Path(sys.argv[1]), Path(sys.argv[2])
    cache.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(TILEJSON, headers=UA), timeout=30) as r:
        template = json.load(r)["tiles"][0]
    pool = ThreadPoolExecutor(6)

    detect = tiles_over(DETECT_Z)
    blobs = list(pool.map(lambda t: fetch(template, cache, DETECT_Z, *t), detect))
    ocean, mixed = [], []
    for (x, y), b in zip(detect, blobs):
        g, ext = water(b, ocean_only=True)
        frac = g.area / ext**2
        if frac >= 1 - 1e-9:
            ocean.append(to_world(box(0, 0, ext, ext), DETECT_Z, x, y, ext))
        elif frac > 0:
            mixed.append((x, y))
    n_ocean = len(ocean)
    children = [(2 * x + i, 2 * y + j) for x, y in mixed for i in (0, 1) for j in (0, 1)]
    blobs = list(pool.map(lambda t: fetch(template, cache, DETAIL_Z, *t), children))
    for (x, y), b in zip(children, blobs):
        ocean.append(to_world(water(b, ocean_only=True)[0], DETAIL_Z, x, y, 4096))

    # the domain, widened to whole zoom-SPLIT_Z tiles
    k = 2**SPLIT_Z
    (x0, y0), (x1, y1) = merc(DOMAIN[0], DOMAIN[3]), merc(DOMAIN[2], DOMAIN[1])
    tx0, ty0, tx1, ty1 = math.floor(x0 * k), math.floor(y0 * k), math.ceil(x1 * k), math.ceil(y1 * k)
    x0, y0, x1, y1 = tx0 / k, ty0 / k, tx1 / k, ty1 / k
    dom = box(x0, y0, x1, y1)
    near = near_forecast().intersection(dom)
    ocean_all = polygonal(shapely.unary_union(ocean).intersection(dom))
    # metres per normalised unit at the domain's southern edge (ground scale shrinks northward,
    # so the tolerance is at most the stated metres everywhere)
    m_per_unit = 40_075_016.686 * math.cos(math.radians(DOMAIN[1]))
    near_w = polygonal(ocean_all.intersection(near))
    far_w = polygonal(ocean_all.difference(near))
    full = polygonal(shapely.unary_union([near_w, far_w]))
    simple = polygonal(
        shapely.unary_union(
            [
                near_w.simplify(NEAR_TOLERANCE_M / m_per_unit, preserve_topology=True),
                far_w.simplify(FAR_TOLERANCE_M / m_per_unit, preserve_topology=True),
            ]
        )
    )
    polys = [
        p
        for tx in range(tx0, tx1)
        for ty in range(ty0, ty1)
        for p in polygonal(simple.intersection(box(tx / k, ty / k, (tx + 1) / k, (ty + 1) / k))).geoms
    ]

    def ring(coords) -> list[int]:
        pts = [(round(cx * QUANT), round(cy * QUANT)) for cx, cy in list(coords)[:-1]]
        flat = [pts[0][0], pts[0][1]]
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            flat += [bx - ax, by - ay]
        return flat

    polygons = [[ring(p.exterior.coords)] + [ring(h.coords) for h in p.interiors] for p in polys]
    n_vertices = sum(len(r) // 2 for p in polygons for r in p)
    near_diff = near_w.symmetric_difference(simple.intersection(near)).area / near_w.area
    doc = {
        "source": "OpenMapTiles water layer, served by OpenFreeMap (the Ocean Map basemap); © OpenMapTiles © OpenStreetMap contributors (ODbL)",
        "tiles": template,
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": (
            f"zoom-{DETECT_Z} tiles all ocean are water, tiles without ocean are land; tiles with both read at zoom {DETAIL_Z}. "
            f"Ocean class only. Within one cell of a C-HARM ocean cell simplified to {NEAR_TOLERANCE_M:g} m, "
            f"elsewhere to {FAR_TOLERANCE_M:g} m. Cut to the domain and along the zoom-{SPLIT_Z} tile grid."
        ),
        "domain_lnglat": list(DOMAIN),
        "domain_merc_q": [round(v * QUANT) for v in (x0, y0, x1, y1)],
        "split_zoom": SPLIT_Z,
        "quant": QUANT,
        "encoding": "polygons -> rings (outer ring, then holes) -> [x0, y0, dx1, dy1, ...] in Web Mercator units of 1/quant world width, y down; domain_merc_q is the covered area [x0, y0, x1, y1] in the same units (domain_lnglat widened to whole zoom-split_zoom tiles)",
        "stats": {
            "detect_tiles": len(detect),
            "ocean_tiles": n_ocean,
            "mixed_tiles": len(mixed),
            "detail_tiles": len(children),
            "polygons": len(polygons),
            "vertices": n_vertices,
            "near_area_change": round(near_diff, 7),
        },
        "polygons": polygons,
    }
    out.write_text(json.dumps(doc, separators=(",", ":")))
    print(json.dumps(doc["stats"]), out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
