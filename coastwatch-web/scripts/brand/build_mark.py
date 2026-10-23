# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "scipy>=1.11", "contourpy>=1.2", "fonttools>=4.50"]
# ///
"""Build Neritic's contour mark from real bathymetry.

The mark is the head of Monterey Canyon, drawn by its own isobaths: the 100, 150, 200, 250 and
300 m depth contours of the NOAA NCEI U.S. Coastal Relief Model, smoothed the way a chart
generalises them, and turned so the canyon head (which points east, at Moss Landing) points up.
The nested arches read as contours first and as a lowercase n second. The 200 m isobath, the
edge of the continental shelf and so of the neritic zone the product is named after, is the
index contour: drawn heavier, in the signal colour.

Outputs (committed; rerun only when this script or its data change):
- src/v4/brand/mark.geometry.ts: the tile, three optical sizes of the symbol (micro <= 20 px,
  small <= 56 px, full), the outlined wordmark, and the launch screen's contour field.

Source data, committed beside this script (monterey-canyon-crm.csv): the CRM volume 6
(3 arc-second; public domain) for Monterey Bay, 36.40-37.00 N, 122.45-121.75 W, every fourth
cell (12 arc-second, about 300 x 370 m; 181 x 211), read once over OPeNDAP:
  https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol6.nc.ascii?z[5280:4:6000][4260:4:5100]
(`--fetch` downloads it again and rewrites the CSV).

Wordmark: Source Serif 4 (SIL Open Font License 1.1; outlining it in a logo is permitted),
Semibold at the 60 pt optical size, from google/fonts at a pinned commit (checked by SHA-256).

usage: uv run --script coastwatch-web/scripts/brand/build_mark.py [--fetch]
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path

import contourpy
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
WEB = HERE.parent.parent
DATA = HERE / "monterey-canyon-crm.csv"
OUT = WEB / "src/v4/brand/mark.geometry.ts"
CRM_URL = "https://www.ngdc.noaa.gov/thredds/dodsC/crm/crm_vol6.nc.ascii?z[5280:4:6000][4260:4:5100]"
LAT0, LAT1, LON0, LON1 = 36.40, 37.00, -122.45, -121.75
SHAPE = (181, 211)
FONT_URL = "https://raw.githubusercontent.com/google/fonts/7b203a635ebe80801c80f29633d4fc467cd1214e/ofl/sourceserif4/SourceSerif4%5Bopsz,wght%5D.ttf"
FONT_SHA256 = "97b2d4da6e3cb494b5a1e66ae176914d852ccabef49e0c02c0df25f3e39aca0b"
FONT_CACHE = Path.home() / ".cache/neritic-brand/SourceSerif4-opsz-wght.ttf"

# The drawing: a 48-unit tile showing a 9.6 km square of the canyon head (centre 0.6 km north and
# 1.8 km east of 36.79 N 121.90 W in the turned frame), turned 18 degrees further so the arches
# stand upright. Smoothing: a 1.6 km Gaussian on the grid (isotropic in km).
SMOOTH_KM = 1.6
WINDOW = dict(cx=0.6, cy=1.8, half=4.8, rot=-18.0)
INDEX = 200
TIERS = {
    # name: (depths in m, minor stroke, index stroke) in tile units
    "micro": ([100, 200], 3.6, 5.4),
    "small": ([100, 200, 300], 2.6, 4.2),
    "full": ([100, 150, 200, 250, 300], 1.7, 3.4),
}
# the launch screen's field: the same isobaths, wider (shallow every 50 m, deep every 100 m)
FIELD_DEPTHS = [0, 25, 50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1200, 1400, 1600, 2000]
FIELD_HALF = 150  # tile units either side of the tile's centre
TILE = "M11 0H37A11 11 0 0 1 48 11V37A11 11 0 0 1 37 48H11A11 11 0 0 1 0 37V11A11 11 0 0 1 11 0Z"


def fetch_crm() -> None:
    text = urllib.request.urlopen(CRM_URL, timeout=120).read().decode()
    body = text.split(f"z.z[{SHAPE[0]}][{SHAPE[1]}]")[1]
    rows = []
    for line in body.strip().splitlines():
        if not line.startswith("["):
            break
        rows.append([float(v) for v in line.split(",")[1:]])
    z = np.array(rows)
    assert z.shape == SHAPE, z.shape
    with DATA.open("w") as f:
        f.write(f"# NOAA NCEI U.S. Coastal Relief Model vol. 6 (3 arc-second), every 4th cell; rows south->north lat {LAT0:.6f}..{LAT1:.6f}, cols west->east lon {LON0:.6f}..{LON1:.6f}; elevation m (negative = depth); {CRM_URL}\n")
        for r in z:
            f.write(",".join(str(int(round(v))) for v in r) + "\n")


def load():
    z = np.loadtxt(DATA, delimiter=",", comments="#")
    lat = np.linspace(LAT0, LAT1, z.shape[0])
    lon = np.linspace(LON0, LON1, z.shape[1])
    k, c = 111.32, math.cos(math.radians(36.79))
    x = (lon + 121.9) * k * c  # km east of 121.90 W
    y = (lat - 36.79) * k  # km north of 36.79 N
    dy, dx = (lat[1] - lat[0]) * k, (lon[1] - lon[0]) * k * c
    zs = ndimage.gaussian_filter(z, (SMOOTH_KM / dy, SMOOTH_KM / dx), mode="nearest")
    return contourpy.contour_generator(x, y, zs)


def resample(p, step):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    n = max(2, int(d[-1] / step) + 1)
    t = np.linspace(0, d[-1], n)
    return np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])]


def simplify(pts, eps):
    """Douglas-Peucker."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.hypot(*ab) or 1e-9
    d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([simplify(pts[: i + 1], eps)[:-1], simplify(pts[i:], eps)])
    return np.vstack([a, b])


def num(v, prec):
    s = f"{v:.{prec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def to_path(pts, prec=2, grid=None):
    """Catmull-Rom through the points, as cubic Beziers. `grid`: snap to 1/grid units and write
    relative commands (the launch screen's field, where size matters more than the last 0.1 unit)."""
    p = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    if grid:
        r = lambda v: int(round(v * grid)) / grid
        f = lambda v: num(r(v), 2)
        s, cur = f"M{f(pts[0][0])} {f(pts[0][1])}", np.array([r(pts[0][0]), r(pts[0][1])])
        for i in range(1, len(p) - 2):
            c1 = p[i] + (p[i + 1] - p[i - 1]) / 6
            c2 = p[i + 1] - (p[i + 2] - p[i]) / 6
            e = np.array([r(p[i + 1][0]), r(p[i + 1][1])])
            vals = [r(c1[0]) - cur[0], r(c1[1]) - cur[1], r(c2[0]) - cur[0], r(c2[1]) - cur[1], e[0] - cur[0], e[1] - cur[1]]
            parts = [f(v) for v in vals]
            s += "c" + "".join(("" if j == 0 or v.startswith("-") else " ") + v for j, v in enumerate(parts))
            cur = e
        return s
    f = lambda v: num(v, prec)
    s = f"M{f(pts[0][0])} {f(pts[0][1])}"
    for i in range(1, len(p) - 2):
        c1 = p[i] + (p[i + 1] - p[i - 1]) / 6
        c2 = p[i + 1] - (p[i + 2] - p[i]) / 6
        s += f"C{f(c1[0])} {f(c1[1])} {f(c2[0])} {f(c2[1])} {f(p[i + 1][0])} {f(p[i + 1][1])}"
    return s


def isobaths(gen, depth, reach, smooth=2.0, eps=0.2, step=0.2):
    """The isobath at `depth` m in tile units (0-48 is the tile), kept within `reach` units of the tile."""
    th = math.radians(WINDOW["rot"])
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    runs = []
    for line in gen.lines(-depth):
        p = np.c_[-line[:, 1], line[:, 0]] @ rot.T  # east -> up, then upright
        q = np.c_[(p[:, 0] - WINDOW["cx"]) / WINDOW["half"] * 24 + 24, 24 - (p[:, 1] - WINDOW["cy"]) / WINDOW["half"] * 24]
        inside = (q[:, 0] > -reach) & (q[:, 0] < 48 + reach) & (q[:, 1] > -reach) & (q[:, 1] < 48 + reach)
        idx = np.where(inside)[0]
        for r in np.split(idx, np.where(np.diff(idx) != 1)[0] + 1):
            if len(r) < 4:
                continue
            s = resample(q[r], step)
            if smooth:
                k = smooth / step
                s = np.c_[ndimage.gaussian_filter1d(s[:, 0], k, mode="nearest"), ndimage.gaussian_filter1d(s[:, 1], k, mode="nearest")]
            if np.hypot(*(s[-1] - s[0])) < 1 and len(s) < 10:
                continue
            runs.append(simplify(s, eps))
    return runs


def split_at_apex(pts):
    """Two halves, each starting at the line's highest point (so a drawing grows from the apex outward)."""
    i = int(np.argmin(pts[:, 1]))
    if i < 2 or i > len(pts) - 3:
        return [pts if i < 2 else pts[::-1]]
    return [pts[i::-1], pts[i:]]


def wordmark():
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    if not FONT_CACHE.exists():
        FONT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        FONT_CACHE.write_bytes(urllib.request.urlopen(FONT_URL, timeout=120).read())
    blob = FONT_CACHE.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == FONT_SHA256, "font changed upstream"
    font = instantiateVariableFont(TTFont(FONT_CACHE), {"wght": 600, "opsz": 60})
    glyphs, cmap = font.getGlyphSet(), font.getBestCmap()
    upm = font["head"].unitsPerEm
    x_height = font["OS/2"].sxHeight
    # x-height 17 tile units: the word's x-height centred on the symbol's middle
    s = 17 / x_height
    track = -0.012 * upm  # a touch tighter than the text setting
    pen_x, d = 0.0, ""
    for ch in "neritic":
        g = cmap[ord(ch)]
        pen = SVGPathPen(glyphs, ntos=lambda v: num(v, 2))
        glyphs[g].draw(TransformPen(pen, (s, 0, 0, -s, pen_x * s, 0)))
        d += pen.getCommands()
        pen_x += glyphs[g].width + track
    width = (pen_x - track) * s
    return {"d": d, "width": round(width, 2), "baseline": round(24 + 17 / 2, 2), "xHeight": 17}


def main():
    if "--fetch" in sys.argv:
        fetch_crm()
    gen = load()
    tiers = {}
    for name, (depths, minor, index) in TIERS.items():
        lines = []
        for depth in depths:
            for run in isobaths(gen, depth, reach=3):
                lines.append({"depth": depth, "d": to_path(run)})
        tiers[name] = {"lines": lines, "minor": minor, "index": index}
    # the symbol's own lines, split at their apex, for the launch screen to draw outward
    hero = []
    for depth in TIERS["full"][0]:
        for run in isobaths(gen, depth, reach=3):
            hero += [{"depth": depth, "d": to_path(h)} for h in split_at_apex(run)]
    field = []
    for depth in FIELD_DEPTHS:
        for run in isobaths(gen, depth, reach=FIELD_HALF - 24, smooth=1.8, eps=0.6, step=0.4):
            field += [{"depth": depth, "d": to_path(h, grid=2)} for h in split_at_apex(run)]
    word = wordmark()
    source = {
        "data": "NOAA NCEI U.S. Coastal Relief Model, vol. 6 (3 arc-second), public domain",
        "area": "Monterey Bay, 36.40-37.00 N, 122.45-121.75 W (the symbol: the canyon head)",
        "smoothing_km": SMOOTH_KM,
        "index_depth_m": INDEX,
        "script": "coastwatch-web/scripts/brand/build_mark.py",
    }
    ts = [
        "// GENERATED by scripts/brand/build_mark.py from real bathymetry (scripts/brand/monterey-canyon-crm.csv).",
        "// Do not edit by hand: change the script and rerun `uv run --script coastwatch-web/scripts/brand/build_mark.py`.",
        f"export const MARK_SOURCE = {json.dumps(source)} as const;",
        f"export const MARK_TILE = {json.dumps(TILE)};",
        f"export const MARK_INDEX_DEPTH = {INDEX};",
        f"export const MARK_TIERS = {json.dumps(tiers)} as const;",
        f"export const MARK_HERO = {json.dumps(hero)} as const;",
        f"/** The launch screen's field in tile units: the tile is 0-48; the field reaches {FIELD_HALF} units from its centre. */",
        f"export const MARK_FIELD = {{ half: {FIELD_HALF}, lines: {json.dumps(field)} }} as const;",
        f"/** Wordmark: Source Serif 4 Semibold (OFL), outlined; drawn from x 0 on the baseline, in tile units. */",
        f"export const MARK_WORD = {json.dumps(word)} as const;",
        "",
    ]
    OUT.write_text("\n".join(ts))
    sizes = {k: sum(len(l["d"]) for l in v["lines"]) for k, v in tiers.items()}
    print(f"wrote {OUT.relative_to(WEB)}: tiers {sizes}, hero {sum(len(l['d']) for l in hero)}, field {len(field)} lines / {sum(len(l['d']) for l in field)} chars, word {len(word['d'])}")


if __name__ == "__main__":
    main()
