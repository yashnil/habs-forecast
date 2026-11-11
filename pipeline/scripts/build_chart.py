# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.11", "pillow>=10.0", "contourpy>=1.2", "shapely>=2.0"]
# ///
"""Build the Neritic v12 chart (docs/coastwatch/v12/design-system/README.md, "the sheet") from real data only.

Outputs, written to coastwatch-web/public/chart/ and committed (they change only when this script or
coastwatch-web/src/nx/ds/tokens.css changes):

- depth.geojson        ENC-style depth areas: one polygon set per threshold ("shallower than d m"),
                       drawn deepest first, so the shoal water is the most saturated and the open ocean
                       the palest (S-52 logic). ETOPO 2022 15" statewide; NOAA CRM 3" inside the
                       Monterey inset only (T30).
- isobaths.geojson     Depth contours. Statewide 200, 500, 1000, 2000, 3000 m (ETOPO, smoothed and
                       simplified as in build_relief.py). Inside the Monterey inset the CRM's own
                       contours: index 200/500/1000/2000/3000 m plus minor lines (T30: finer lines only
                       where the 3" grid supports them).
- land.geojson         Land, the complement of the bundled OpenMapTiles ocean (public/basemap/
                       water-fallback.json, © OpenMapTiles © OpenStreetMap contributors).
- relief-land.webp     Land relief at 1x (no exaggeration), ETOPO, as a shade with alpha: a warm shadow
- relief-land-monterey.webp   and a light highlight laid over the land colour (opacity per chart mode).
- chart-{mode}-ca.webp, chart-{mode}-monterey.webp
                       The same chart baked to Web Mercator images for the static tier (no WebGL, a saved
                       copy, data saver): depth tints, land, relief, isobaths and coastline in each mode's
                       tokens. Labels and notes are drawn by the page over them, as on the live chart.
- opening-relief.png   (an intermediate in the system temp folder, not committed) The opening's source: Monterey Canyon lit from the north-west with the seafloor
                       exaggerated 4x (land 1x), plan view, Web Mercator. scripts/bake-opening.mjs dithers
                       it into public/chart/opening-canyon.webp. The page labels the exaggeration (T29).
- canyon-axis.json     The canyon's thalweg from the CRM, for the water name "Monterey Canyon".
- chart.json           Corners of every image, sources, licences and how each was drawn.

Relief is not for navigation (NOAA terms, T31). No soundings are drawn from these grids.

Data: ETOPO 2022 v1, 15 arc-second, doi:10.25921/fd45-gt74 (NOAA CoastWatch ERDDAP, ETOPO_2022_v1_15s);
NOAA NCEI U.S. Coastal Relief Model vols 6 and 7, 3 arc-second (npz: lat, lon, z), as in build_relief.py.

usage: uv run --script pipeline/scripts/build_chart.py <etopo.nc> <monterey_crm3s.npz> [out-dir]
       (download: https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.nc?z[(31.9):1:(42.7)][(-126.8):1:(-116.2)])
"""

from __future__ import annotations

import json
import math
import re
import sys
import tempfile
from pathlib import Path

import contourpy
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from scipy.io import netcdf_file
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "coastwatch-web"
TOKENS = WEB / "src" / "nx" / "ds" / "tokens.css"
WATER = WEB / "public" / "basemap" / "water-fallback.json"

R = 6378137.0
# the chart's statewide frame: the ETOPO subset's own extent (nothing is drawn from beyond the grid)
CA = (-126.8, 31.9, -116.2, 42.7)
# land and the map's pannable bounds: the bundled ocean's whole domain, so a wide screen at state scale never
# shows the frame's edge (beyond CA the land is plain buff and the open ocean the deepest tint)
LAND_BOX = (-131.4, 29.6, -112.6, 44.7)
# the Monterey inset, where the CRM's 3" grid replaces ETOPO (the same box as relief.json)
INSET = (-122.72, 36.22, -121.58, 37.33)
# ENC depth areas: a polygon set per threshold, coloured with the band just shallower than it
THRESHOLDS = [2000, 1000, 500, 200, 100, 50, 20, 10]
TINT_OF = {2000: 7, 1000: 6, 500: 5, 200: 4, 100: 3, 50: 2, 20: 1, 10: 0}
INDEX = [200, 500, 1000, 2000, 3000]
CRM_MINOR = [50, 100, 300, 400, 600, 700, 800, 900, 1250, 1500, 1750, 2250, 2500, 2750]
MODES = ("day", "dusk", "night")
# the opening's still: centred a little west of the canyon head, square in Mercator
STILL_CENTER = (-121.95, 36.78)
STILL_HALF_LON = 0.36
STILL_PX = 1600
STILL_EXAGGERATION = 4.0


# ---------------------------------------------------------------- tokens


def read_tokens() -> dict[str, dict[str, tuple]]:
    """Each chart mode's colours from tokens.css (the only place colours live)."""
    css = TOKENS.read_text()
    out: dict[str, dict[str, tuple]] = {}
    for mode in MODES:
        sel = r':root,\s*\[data-chart-mode="day"\]' if mode == "day" else rf'\[data-chart-mode="{mode}"\]'
        m = re.search(sel + r"\s*\{(.*?)\n\}", css, re.S)
        if not m:
            raise SystemExit(f"tokens.css: no block for {mode}")
        vals: dict[str, tuple] = {}
        for k, v in re.findall(r"--([\w-]+):\s*([^;]+);", m.group(1)):
            v = v.strip()
            if v.startswith("#"):
                vals[k] = tuple(int(v[i : i + 2], 16) for i in (1, 3, 5)) + (1.0,)
            elif v.startswith("rgba"):
                p = [float(x) for x in re.findall(r"[\d.]+", v)]
                vals[k] = (p[0], p[1], p[2], p[3])
            else:
                try:
                    vals[k] = (float(v),)
                except ValueError:
                    pass
        out[mode] = vals
    return out


def rgb(c) -> np.ndarray:
    return np.array(c[:3], float)


# ---------------------------------------------------------------- geometry helpers


def merc_y(lat):
    return R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def lat_of(y):
    return np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)


def mercator_grid(w, s, e, n, width):
    """Pixel centres of a Web Mercator image of `width` px over the box: (rows_lat, cols_lon, height, px)."""
    px = np.radians(e - w) * R / width
    y_top, y_bot = merc_y(n), merc_y(s)
    height = int(round((y_top - y_bot) / px))
    rows_lat = lat_of(y_top - (np.arange(height) + 0.5) * px)
    cols_lon = w + (np.arange(width) + 0.5) * (e - w) / width
    return rows_lat, cols_lon, height, px


def resample(lat, lon, z, rows_lat, cols_lon):
    ri = np.interp(rows_lat, lat, np.arange(len(lat)))
    ci = np.interp(cols_lon, lon, np.arange(len(lon)))
    RR, CC = np.meshgrid(ri, ci, indexing="ij")
    return ndimage.map_coordinates(z, [RR, CC], order=1, mode="nearest")


def hillshade(z, dx, dy, exaggeration=1.0, azimuth=315.0, altitude=45.0):
    gy = np.gradient(z * exaggeration, dy, axis=0)
    gx = np.gradient(z * exaggeration, axis=1) / dx[:, None]
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    return np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)


def filled(lon, lat, z, lower):
    """Polygons where z > lower (contourpy filled), as one shapely geometry."""
    gen = contourpy.contour_generator(lon, lat, z, fill_type=contourpy.FillType.OuterOffset)
    pts, offs = gen.filled(lower, 1e9)
    polys = []
    for p, o in zip(pts, offs):
        rings = [p[o[i] : o[i + 1]] for i in range(len(o) - 1)]
        if len(rings[0]) < 4:
            continue
        polys.append(Polygon(rings[0], [r for r in rings[1:] if len(r) >= 4]))
    g = unary_union([q.buffer(0) for q in polys])
    return g


def lines(lon, lat, z, level):
    gen = contourpy.contour_generator(lon, lat, z, line_type=contourpy.LineType.Separate)
    return [LineString(s) for s in gen.lines(level) if len(s) >= 2]


def polys_of(g):
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon) and not p.is_empty]


def lines_of(g):
    if g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    out = []
    for p in getattr(g, "geoms", []):
        out += lines_of(p)
    return out


def rounded(geom, nd):
    """GeoJSON geometry with coordinates rounded to `nd` decimals (and degenerate parts dropped)."""
    def r(c):
        return [round(c[0], nd), round(c[1], nd)]

    if isinstance(geom, Polygon):
        rings = [[r(c) for c in geom.exterior.coords]] + [[r(c) for c in i.coords] for i in geom.interiors]
        return {"type": "Polygon", "coordinates": rings}
    if isinstance(geom, LineString):
        return {"type": "LineString", "coordinates": [r(c) for c in geom.coords]}
    raise TypeError(type(geom))


def write_geojson(path: Path, feats: list[dict]):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))


# ---------------------------------------------------------------- sources


def load_etopo(src: Path):
    with netcdf_file(src, "r", mmap=False) as nc:
        lat = nc.variables["latitude"][:].astype(float)
        lon = nc.variables["longitude"][:].astype(float)
        z = nc.variables["z"][:].astype(float)
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    return lat, lon, z


def load_crm(npz: Path):
    d = np.load(npz)
    lat, lon, z = d["lat"].astype(float), d["lon"].astype(float), np.nan_to_num(d["z"].astype(float))
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1]
    return lat, lon, z


def load_water():
    """The bundled OpenMapTiles ocean, as lon/lat shapely polygons (unioned: the tile cuts disappear)."""
    f = json.loads(WATER.read_text())
    q = f["quant"]
    polys = []
    for poly in f["polygons"]:
        rings = []
        for enc in poly:
            xs = np.cumsum(enc[0::2]) / q
            ys = np.cumsum(enc[1::2]) / q
            lon = xs * 360 - 180
            lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * ys))))
            rings.append(np.column_stack([lon, lat]))
        if len(rings[0]) >= 4:
            polys.append(Polygon(rings[0], [r for r in rings[1:] if len(r) >= 4]).buffer(0))
    water = unary_union(polys)
    return water


# ---------------------------------------------------------------- vector chart


def depth_areas(lat, lon, z, clip, simplify):
    feats = []
    for d in THRESHOLDS:
        g = filled(lon, lat, z, -d).intersection(clip).simplify(simplify, preserve_topology=True)
        for p in polys_of(g):
            if p.area < simplify * simplify * 4:
                continue
            feats.append({"type": "Feature", "properties": {"d": d, "t": TINT_OF[d]}, "geometry": rounded(p, 4)})
    return feats


def isobath_feats(lat, lon, z, levels, clip, simplify, index_set, min_len):
    feats = []
    for d in levels:
        for ln in lines(lon, lat, z, -d):
            g = ln.simplify(simplify).intersection(clip)
            for part in lines_of(g):
                if part.length < min_len:
                    continue
                feats.append({"type": "Feature", "properties": {"d": d, "i": 1 if d in index_set else 0, "label": f"{d:,} m"}, "geometry": rounded(part, 4)})
    return feats


# ---------------------------------------------------------------- rasters


def rasterize(geoms, w, s, e, n, width, height, ss=2):
    """A 0..1 coverage mask of polygons on the Mercator image (supersampled for smooth edges)."""
    W, H = width * ss, height * ss
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    y_top, y_bot = merc_y(n), merc_y(s)

    def px(coords):
        c = np.asarray(coords)
        x = (c[:, 0] - w) / (e - w) * W
        y = (y_top - merc_y(c[:, 1])) / (y_top - y_bot) * H
        return list(map(tuple, np.column_stack([x, y])))

    for g in geoms:
        for p in polys_of(g):
            dr.polygon(px(p.exterior.coords), fill=255)
            for i in p.interiors:
                dr.polygon(px(i.coords), fill=0)
    img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, float) / 255


def stroke(w, s, e, n, width, height, widths, ss=3):
    """Anti-aliased line coverage per width class: {key: mask}."""
    W, H = width * ss, height * ss
    y_top, y_bot = merc_y(n), merc_y(s)
    out = {}
    for key, (feats, lw) in widths.items():
        img = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(img)
        for ln in feats:
            c = np.asarray(ln.coords)
            x = (c[:, 0] - w) / (e - w) * W
            y = (y_top - merc_y(c[:, 1])) / (y_top - y_bot) * H
            dr.line(list(map(tuple, np.column_stack([x, y]))), fill=255, width=max(1, int(round(lw * ss))), joint="curve")
        out[key] = np.asarray(img.resize((width, height), Image.LANCZOS), float) / 255
    return out


def edge_fade(rows_lat, cols_lon, box_, width_deg):
    """1 inside, easing (smoothstep) to 0 over `width_deg` at the box's east, north and south edges, where the
    land continues past the relief's grid. The west edge is open ocean and needs none."""
    w, s, e, n = box_
    lat = np.asarray(rows_lat)[:, None]
    lon = np.asarray(cols_lon)[None, :]
    def ease(d):
        t = np.clip(d / width_deg, 0, 1)
        return t * t * (3 - 2 * t)
    return ease(e - lon) * ease(n - lat) * ease(lat - s)


def relief_rgba(zm, dx, land_mask, tok):
    """Land relief at 1x (no exaggeration) as one warm shadow colour with alpha: slopes facing the light carry
    less shadow than the flat, slopes away from it more, so a single darkening layer reads both ways and
    compresses to a fraction of a two-tone image."""
    hs = 0.6 * hillshade(np.maximum(zm, 0), dx, float(np.mean(dx)), 1.0, 315, 42) + 0.4 * hillshade(np.maximum(zm, 0), dx, float(np.mean(dx)), 1.0, 350, 50)
    flat = 0.6 * math.sin(math.radians(42)) + 0.4 * math.sin(math.radians(50))
    t = ndimage.gaussian_filter(flat - hs, 0.8)
    land = rgb(tok["day"]["land"])
    ink = rgb(tok["day"]["ink"])
    dark = land * 0.42 + ink * 0.2  # a warm sepia, darker than the buff
    a = np.clip(0.1 + 1.5 * t, 0, 0.85) * land_mask
    out = np.zeros(zm.shape + (4,), float)
    out[..., :3] = dark
    out[..., 3] = a * 255
    return out.astype(np.uint8)


def bake_chart_image(zs, zraw, rows_lat, cols_lon, px, land_cov, coast_feats, iso_index, iso_minor, tok, mode, box_, relief_strength):
    """One mode's chart image: tints from the smoothed grid (analytic anti-aliasing), land, relief, lines."""
    w, s, e, n = box_
    height, width = zs.shape
    T = tok[mode]
    gy, gx = np.gradient(zs)
    grad = np.maximum(np.hypot(gx, gy), 1e-3)
    img = np.zeros((height, width, 3), float)
    img[:] = rgb(T["depth-8"])
    for d in THRESHOLDS:
        cov = np.clip(0.5 + (zs + d) / grad, 0, 1)[..., None]
        img = img * (1 - cov) + rgb(T[f"depth-{TINT_OF[d]}"]) * cov
    lines_ = stroke(w, s, e, n, width, height, {"index": (iso_index, 1.1), "minor": (iso_minor, 0.7), "coast": (coast_feats, 1.3)})
    iso = rgb(T["isobath"])
    img = img * (1 - 0.35 * lines_["minor"][..., None]) + iso * 0.35 * lines_["minor"][..., None]
    img = img * (1 - 0.8 * lines_["index"][..., None]) + iso * 0.8 * lines_["index"][..., None]
    # land and its relief
    land = rgb(T["land"])
    img = img * (1 - land_cov[..., None]) + land * land_cov[..., None]
    if relief_strength > 0:
        dx = px * np.cos(np.radians(rows_lat))
        rel = relief_rgba(zraw, dx, land_cov, tok).astype(float)
        a = rel[..., 3:4] / 255 * relief_strength
        img = img * (1 - a) + rel[..., :3] * a
    ink = rgb(T["ink"])
    img = img * (1 - lines_["coast"][..., None]) + ink * lines_["coast"][..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def inset_weight(rows_lat, cols_lon, band=0.06):
    """1 well inside the Monterey inset, easing to 0 at its edge over `band` degrees (smoothstep)."""
    w, s, e, n = INSET
    dx = np.minimum(cols_lon - w, e - cols_lon)[None, :]
    dy = np.minimum(rows_lat - s, n - rows_lat)[:, None]
    t = np.clip(np.minimum(dx, dy) / band, 0, 1)
    return t * t * (3 - 2 * t)


def thalweg(lat, lon, zs):
    """The canyon's axis: the deepest cell per longitude column, walking west from the head."""
    pts = []
    for lo0 in np.arange(-121.84, -122.40, -0.02):
        j = int(np.argmin(np.abs(lon - lo0)))
        col = zs[:, j]
        sel = (lat > 36.55) & (lat < 36.9)
        if pts:
            sel &= np.abs(lat - pts[-1][1]) < 0.05
        i = np.where(sel)[0]
        i = i[np.argmin(col[i])]
        pts.append((round(float(lon[j]), 4), round(float(lat[i]), 4), round(float(col[i]))))
    # a light smoothing of the latitude, so the name sits on a calm curve
    la = ndimage.gaussian_filter1d(np.array([p[1] for p in pts]), 1.2)
    return [[p[0], round(float(y), 4), p[2]] for p, y in zip(pts, la)]


# ---------------------------------------------------------------- main


def main(etopo: Path, crm: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tok = read_tokens()
    meta: dict = {"sources": {}, "images": {}, "notes": "Not for navigation (NOAA). Depth areas and isobaths from smoothed grids; no soundings are drawn from them."}

    print("reading ETOPO, CRM and the bundled ocean")
    elat, elon, ez = load_etopo(etopo)
    clat, clon, cz = load_crm(crm)
    water = load_water()
    ca_box = box(*CA)
    inset = box(*INSET)
    outside = ca_box.difference(inset)

    # ---- land (OpenMapTiles ocean, complemented over a frame wider than the map's bounds, so the frame's own
    # edges are never on screen); the coastline is this polygon's outline, drawn by the page
    land_box = box(*LAND_BOX)
    water_ca = water.intersection(land_box.buffer(0.5))
    land = land_box.difference(water_ca).simplify(0.00022, preserve_topology=True)
    land_feats = [{"type": "Feature", "properties": {}, "geometry": rounded(p, 5)} for p in polys_of(land) if p.area > 4e-7]
    write_geojson(out / "land.geojson", land_feats)
    inner = land_box.buffer(-1e-6)
    coast = []
    for p in polys_of(land):
        for ring in [p.exterior, *p.interiors]:
            for part in lines_of(LineString(ring.coords).intersection(inner)):
                if part.length > 0.002:
                    coast.append(part)

    # ---- depth areas and isobaths
    ezs = ndimage.gaussian_filter(ez, 1.2)
    czs = ndimage.gaussian_filter(cz, 1.4)
    depth = depth_areas(elat, elon, ezs, outside, 0.0035) + depth_areas(clat, clon, czs, inset, 0.00025)
    for f in depth:
        f["properties"]["src"] = "crm" if shape(f["geometry"]).within(inset.buffer(1e-6)) else "etopo"
    write_geojson(out / "depth.geojson", depth)
    iso = isobath_feats(elat, elon, ezs, INDEX, outside, 0.004, set(INDEX), 0.08)
    iso += isobath_feats(clat, clon, czs, sorted(INDEX + CRM_MINOR), inset, 0.0003, set(INDEX), 0.02)
    # minor lines in the shallows stay off the land (a smoothed grid can cross a harbour's edge)
    water_inset = water.intersection(inset)
    keep = []
    for f in iso:
        if f["properties"]["d"] < 200:
            g = shape(f["geometry"]).intersection(water_inset)
            for part in lines_of(g):
                if part.length > 0.01:
                    keep.append({**f, "geometry": rounded(part, 4)})
        else:
            keep.append(f)
    write_geojson(out / "isobaths.geojson", keep)
    iso_index = [shape(f["geometry"]) for f in keep if f["properties"]["i"] == 1]
    iso_minor = [shape(f["geometry"]) for f in keep if f["properties"]["i"] == 0]

    # ---- rasters: statewide (ETOPO) and the Monterey inset (CRM)
    regions = {
        "ca": dict(box=CA, lat=elat, lon=elon, z=ez, zs=ezs, width=2400, relief_width=1300),
        "monterey": dict(box=INSET, lat=clat, lon=clon, z=cz, zs=czs, width=1600, relief_width=1100),
    }
    for name, r in regions.items():
        w, s, e, n = r["box"]
        rows_lat, cols_lon, height, px = mercator_grid(w, s, e, n, r["width"])
        zraw = resample(r["lat"], r["lon"], r["z"], rows_lat, cols_lon)
        zsm = resample(r["lat"], r["lon"], r["zs"], rows_lat, cols_lon)
        land_cov = rasterize([land], w, s, e, n, r["width"], height)
        dx = px * np.cos(np.radians(rows_lat))
        rel = relief_rgba(zraw, dx, land_cov, tok)
        # the two reliefs cross-fade over the inset's edge, so neither shows a rectangle nor doubles the shading
        wgt = inset_weight(rows_lat, cols_lon)
        rel[..., 3] = (rel[..., 3] * ((1 - wgt) if name == "ca" else wgt)).astype(np.uint8)
        if name == "ca":
            # the land runs on past the ETOPO box (to the frame's bounds): the relief eases out over its last
            # 0.35 degrees on the east, north and south, so no edge shows where the shading stops
            rel[..., 3] = (rel[..., 3] * edge_fade(rows_lat, cols_lon, r["box"], 0.35)).astype(np.uint8)
        rel_name = "relief-land.webp" if name == "ca" else "relief-land-monterey.webp"
        rw = r["relief_width"]
        rh = round(height * rw / r["width"])
        Image.fromarray(rel, "RGBA").resize((rw, rh), Image.LANCZOS).save(out / rel_name, "WEBP", quality=55, alpha_quality=32, method=6)
        meta["images"][rel_name] = {"corners_lnglat": [[w, n], [e, n], [e, s], [w, s]], "width": rw, "height": rh}
        for mode in MODES:
            strength = tok[mode].get("land-relief-opacity", (0.5,))[0]
            img = bake_chart_image(zsm, zraw, rows_lat, cols_lon, px, land_cov, coast, iso_index, iso_minor, tok, mode, r["box"], strength)
            fn = f"chart-{mode}-{name}.webp"
            Image.fromarray(img, "RGB").save(out / fn, "WEBP", quality=74, method=6)
            meta["images"][fn] = {"corners_lnglat": [[w, n], [e, n], [e, s], [w, s]], "width": r["width"], "height": height}
        print(f"{name}: {r['width']}x{height}")

    # ---- the opening's lit canyon (CRM, plan view, sea x4, land x1)
    cx, cy = STILL_CENTER
    half_lat = (lat_of(merc_y(cy) + math.radians(STILL_HALF_LON) * R) - lat_of(merc_y(cy) - math.radians(STILL_HALF_LON) * R)) / 2
    sw, se = cx - STILL_HALF_LON, cx + STILL_HALF_LON
    yc = merc_y(cy)
    half_m = math.radians(STILL_HALF_LON) * R
    sn, ss_ = float(lat_of(yc + half_m)), float(lat_of(yc - half_m))
    rows_lat, cols_lon, height, px = mercator_grid(sw, ss_, se, sn, STILL_PX)
    zst = resample(clat, clon, ndimage.gaussian_filter(cz, 0.8), rows_lat, cols_lon)
    dx = px * np.cos(np.radians(rows_lat))
    sea = np.minimum(zst, 0)
    hs_sea = hillshade(sea, dx, float(np.mean(dx)), STILL_EXAGGERATION, 300, 32)
    hs_land = hillshade(np.maximum(zst, 0), dx, float(np.mean(dx)), 1.0, 300, 32)
    is_land = rasterize([land], sw, ss_, se, sn, STILL_PX, height)
    # light: shade, plus a gentle depth ramp so the canyon floor reads deeper than the shelf
    depth_t = np.clip(np.log1p(-sea / 30) / np.log1p(3000 / 30), 0, 1)
    sea_l = 0.18 + 0.82 * hs_sea * (1 - 0.35 * depth_t)
    land_l = 0.97 + 0.12 * (hs_land - math.sin(math.radians(32)))
    lum = sea_l * (1 - is_land) + np.clip(land_l, 0, 1) * is_land
    # an intermediate, not shipped: scripts/bake-opening.mjs dithers it into public/chart/opening-canyon.webp
    scratch = Path(tempfile.gettempdir()) / "neritic-chart"
    scratch.mkdir(exist_ok=True)
    Image.fromarray(np.clip(lum * 255, 0, 255).astype(np.uint8), "L").save(scratch / "opening-relief.png", optimize=True)
    print(f"opening relief (for scripts/bake-opening.mjs): {scratch / 'opening-relief.png'}")
    meta["images"]["opening-canyon.webp"] = {
        "corners_lnglat": [[sw, sn], [se, sn], [se, ss_], [sw, ss_]], "width": STILL_PX, "height": height,
        "rendering": f"NOAA CRM 3\", hillshade from 300° at 32°, seafloor exaggerated {STILL_EXAGGERATION:g}x, land 1x, dithered (Paper Shaders image-dithering, scripts/bake-opening.mjs)",
        "exaggeration": STILL_EXAGGERATION,
    }

    (out / "canyon-axis.json").write_text(json.dumps(thalweg(clat, clon, czs)))

    meta["sources"] = {
        "etopo": "ETOPO 2022 v1, 15 arc-second (Ice Surface), NOAA NCEI, doi:10.25921/fd45-gt74. Free to use; not for navigation.",
        "crm": "NOAA NCEI U.S. Coastal Relief Model, volumes 6 and 7, 3 arc-second; Monterey inset only. Public domain; not for navigation.",
        "coast": "OpenMapTiles water (public/basemap/water-fallback.json). © OpenMapTiles © OpenStreetMap contributors (ODbL).",
    }
    meta["frame"] = {"ca": CA, "inset": INSET, "bounds": LAND_BOX}
    meta["depth_areas"] = {"thresholds_m": THRESHOLDS, "tint_of": TINT_OF, "smoothing": "Gaussian 1.2 cells (ETOPO), 1.4 cells (CRM)"}
    meta["isobaths"] = {"index_m": INDEX, "crm_minor_m": CRM_MINOR}
    (out / "chart.json").write_text(json.dumps(meta, indent=2) + "\n")
    for f in sorted(out.iterdir()):
        print(f"{f.name}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(Path(a[0]), Path(a[1]), Path(a[2]) if len(a) > 2 else WEB / "public" / "chart")
