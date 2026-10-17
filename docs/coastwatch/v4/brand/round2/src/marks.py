"""Round 2 marks: vector geometry (48-unit grid), SVG builders and the pixel-master rasteriser.

Coordinates: 48 u square, y down, north up. 1 u = 1/3 px at 16 px, 1/2 px at 24 px.
The tile fills the whole square (round 1 left a 4 u margin; at 16 px that cost 2.7 px of drawing).
"""
import os, sys
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import wm  # noqa: E402  (round-1 outliner: Literata static instances in brand/src/fonts, git-ignored)
from masters import MASTERS  # noqa: E402

INK = "#0b1a24"
NAVY = "#12395a"
SEA = "#2e86ab"
SHELF = "#bfe4ec"
FAV_SHELF = "#a6d6e3"
FAV_NAVY = "#1b5583"  # favicon only: one step lighter than NAVY so the tile edge survives on dark tabs (#202124)  # a step deeper than SHELF so the pale part still separates from a light tab (#f1f3f4)
PAPER = "#f6f4ef"
FOG = "#e8eef2"

TILE = "M10 0H38A10 10 0 0 1 48 10V38A10 10 0 0 1 38 48H10A10 10 0 0 1 0 38V10A10 10 0 0 1 10 0Z"
SQUARE = "M0 0H48V48H0Z"
R1_SQ = "M15 4H33A11 11 0 0 1 44 15V33A11 11 0 0 1 33 44H15A11 11 0 0 1 4 33V15A11 11 0 0 1 15 4Z"

# ---------------------------------------------------------------- B geometry
GEO = {
    # round 1, unchanged (reference)
    "b0": dict(
        deep="M-1 -1H13C15 10 17.6 17.6 20.6 23.4C23.4 29 23.4 38 22.6 49H-1Z",
        land="M49 -1H19C20.6 8.4 23.2 14.6 26.6 18.8C30.6 23.6 31 27 30.2 30.2C29.4 34 30.6 41 34 49H49Z",
        tile=R1_SQ),
    # R1 Taper: band 2 px at the north edge, 4.5 px at the south edge; smooth deep edge, one bay and one point
    "r1": dict(
        deep="M-1 -1H16.4C17.4 10 19.4 18 21.2 25C22.8 31.6 22.8 40 21.6 49H-1Z",
        land="M23 -1H49V49H35C34.4 43 34 38 31.6 33.6C29.4 29.6 26.8 27.4 26.6 23.4C26.4 19.6 30.6 17.4 29.6 13C28.8 9.2 25 5 23 -1Z"),
    # R2 Bight: narrow northern shelf, pinched at the point, opening east into the Southern California Bight
    "r2": dict(
        deep="M-1 -1H19C20.4 12 22 22 21.6 30C21.2 38 18.6 44 16 49H-1Z",
        land="M25 -1H49V26C45 27.6 41.4 27 38.6 24.2C36.2 21.8 33.4 22 31 19.8C28.4 17.2 26.6 9.4 25 -1Z"),
    # R3 Bight, closed south: deep water returns along the south edge
    "r3": dict(
        deep="M-1 -1H19C20.4 12 22 22 21.6 30C21.4 34 21.8 37 24.6 39C28 41.2 36 41.6 49 41.4V49H-1Z",
        land="M25 -1H49V26C45 27.6 41.4 27 38.6 24.2C36.2 21.8 33.4 22 31 19.8C28.4 17.2 26.6 9.4 25 -1Z"),
    # R4 (tried, not shortlisted) Bight + Santa Cruz Island
    "r4": dict(
        deep="M-1 -1H19C20.4 12 22 22 21.6 30C21.2 38 18.6 44 16 49H-1Z",
        land="M25 -1H49V26C45 27.6 41.4 27 38.6 24.2C36.2 21.8 33.4 22 31 19.8C28.4 17.2 26.6 9.4 25 -1Z",
        island="M33.4 31.2C35.4 30 40.4 29.8 42.6 30.8C43.8 31.4 43.2 32.9 41.4 33.2C38.8 33.6 35.2 33.6 33.8 33C33 32.6 32.9 31.6 33.4 31.2Z"),
}


def svg(inner, vb="0 0 48 48", w=None, h=None, title=None):
    size = f' width="{w}" height="{h}"' if w else ""
    t = f"<title>{title}</title>" if title else ""
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}"{size}>{t}{inner}</svg>'


def b_inner(k, uid, solid=NAVY, band=FAV_SHELF, clip=None, mono=None):
    g = GEO[k]
    clip = clip or g.get("tile", TILE)
    shapes = f'<path d="{g["deep"]}"/><path d="{g["land"]}"/>' + (f'<path d="{g["island"]}"/>' if "island" in g else "")
    if mono:  # one colour: the tile with the shelf knocked out (the ground shows through)
        return (f'<defs><clipPath id="{uid}c"><path d="{clip}"/></clipPath></defs>'
                f'<g clip-path="url(#{uid}c)" fill="{mono}">{shapes}</g>')
    return (f'<defs><clipPath id="{uid}c"><path d="{clip}"/></clipPath></defs>'
            f'<g clip-path="url(#{uid}c)"><path d="{clip}" fill="{band}"/><g fill="{solid}">{shapes}</g></g>')


# ---------------------------------------------------------------- C geometry
def c_parts():
    d, w, b = wm.outline("i600", "n", 46)
    x0, y0, x1, y1 = b
    tx, ty = 24 - (x0 + x1) / 2, 24 - (y0 + y1) / 2 + 0.3
    gl, gr = x0 + tx, x1 + tx
    y = 24.6
    return d, tx, ty, gl, gr, y


def c_inner(uid, tile=NAVY, line=SHELF, glyph=FOG, clip=TILE, mono=None, sw=4.0):
    d, tx, ty, gl, gr, y = c_parts()
    left = f"M-1 {y + 1.4:.2f}H{gl - 2.6:.2f}"
    right = f"M{gr + 2.4:.2f} {y - 1.4:.2f}H49"
    if mono:
        return (f'<defs><mask id="{uid}k" maskUnits="userSpaceOnUse" x="0" y="0" width="48" height="48">'
                f'<path d="{clip}" fill="#fff"/><g stroke="#000" stroke-width="{sw}"><path d="{left}"/><path d="{right}"/></g>'
                f'<path transform="translate({tx:.2f} {ty:.2f})" d="{d}" fill="#000"/></mask></defs>'
                f'<path d="{clip}" fill="{mono}" mask="url(#{uid}k)"/>')
    return (f'<defs><clipPath id="{uid}c"><path d="{clip}"/></clipPath></defs>'
            f'<g clip-path="url(#{uid}c)"><path d="{clip}" fill="{tile}"/>'
            f'<g stroke="{line}" stroke-width="{sw}"><path d="{left}"/><path d="{right}"/></g>'
            f'<path transform="translate({tx:.2f} {ty:.2f})" d="{d}" fill="{glyph}"/></g>')


# ---------------------------------------------------------------- per-candidate builders
def fav_inner(k, uid):
    """the favicon drawing (used for 32 px and every 2x slot; 16/24 at 1x use the pixel masters)"""
    if k == "c2":
        return c_inner(uid, tile=FAV_NAVY)
    if k == "b0":
        return b_inner(k, uid, band=SHELF)
    return b_inner(k, uid, solid=FAV_NAVY)


def symbol_inner(k, uid, dark=False):
    """the symbol for lockups. Dark grounds: see DARK below."""
    if k == "c2":
        return c_inner(uid, tile=SEA, line=INK, glyph=PAPER) if dark else c_inner(uid, line=SHELF)
    if dark:
        return b_inner(k, uid, solid=DARK_SOLID, band=DARK_BAND)
    return b_inner(k, uid, band=SHELF)


# B on dark grounds: the tile stays a tile. Pale solids with a sea-blue shelf keep the outline of the square
# visible on near-black, so the two solids never float free (round 1's "brackets").
DARK_SOLID = FOG
DARK_BAND = SEA


def app_inner(k, uid):
    if k == "c2":
        return c_inner(uid, clip=SQUARE)
    return b_inner(k, uid, band=SHELF, clip=SQUARE)


def mono_inner(k, uid, color, rim=False):
    if k == "c2":
        return c_inner(uid, mono=color)
    inner = b_inner(k, uid, mono=color)
    if rim:  # keep a 1 px (at 16 px) rim of the tile so the shelf never opens the outline
        inner += (f'<defs><clipPath id="{uid}r"><path d="{TILE}"/></clipPath></defs>'
                  f'<path d="{TILE}" fill="none" stroke="{color}" stroke-width="6" clip-path="url(#{uid}r)"/>')
    return inner


# ---------------------------------------------------------------- pixel masters -> PNG
def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _mix(a, b, t):  # t = share of a
    a, b = _hex(a), _hex(b)
    return tuple(round(a[i] * t + b[i] * (1 - t)) for i in range(3))


def tile_alpha(s, r_frac=10 / 48, ss=8):
    r = s * r_frac
    out = []
    for y in range(s):
        row = []
        for x in range(s):
            hit = 0
            for j in range(ss):
                for i in range(ss):
                    px, py = x + (i + .5) / ss, y + (j + .5) / ss
                    cx = min(max(px, r), s - r)
                    cy = min(max(py, r), s - r)
                    if (px - cx) ** 2 + (py - cy) ** 2 <= r * r:
                        hit += 1
            row.append(hit / ss / ss)
        out.append(row)
    return out


def palette(k):
    if k == "c2":
        return {"#": _hex(FAV_NAVY), "F": _hex(FOG), "f": _mix(FOG, FAV_NAVY, .5), "L": _hex(SHELF), "l": _mix(SHELF, FAV_NAVY, .5)}
    return {"#": _hex(FAV_NAVY), "o": _mix(FAV_NAVY, FAV_SHELF, .65), "+": _mix(FAV_NAVY, FAV_SHELF, .35), ".": _hex(FAV_SHELF)}


MONO_A = {"#": 1, "o": .65, "+": .35, ".": 0, "F": 0, "f": .5, "L": 0, "l": .5}


def master_png(k, s, path, mono=None, rim=False):
    g = MASTERS[k][s]
    ta = tile_alpha(s)
    if rim:  # pixels on the tile's edge become solid
        def edge(x, y):
            return any(not (0 <= x + dx < s and 0 <= y + dy < s) or ta[y + dy][x + dx] < .5
                       for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        g = ["".join("#" if ta[y][x] > 0 and edge(x, y) else ch for x, ch in enumerate(row)) for y, row in enumerate(g)]
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    pal = palette(k)
    for y, row in enumerate(g):
        for x, ch in enumerate(row):
            if mono:
                a = MONO_A[ch] * ta[y][x]
                im.putpixel((x, y), _hex(mono) + (round(a * 255),))
            else:
                im.putpixel((x, y), pal[ch] + (round(ta[y][x] * 255),))
    im.save(path)


# ---------------------------------------------------------------- wordmarks and lockups
def wordmark(k):
    if k == "c2":
        return wm.outline("i600", "neritic", 32, track=0.008)
    return wm.outline("r600", "Neritic", 30, track=-0.015, adj={1: 0.004, 3: 0.006})


def lockup(k, sym, ink):
    d, w, b = wordmark(k)
    cap = -b[1]
    base = 24 + cap / 2 if k != "c2" else 24 + 0.511 * 32 / 2 + 1.2
    gap = 12 if k != "c2" else 13
    x = 48 + gap - b[0]
    W = x + b[2] + 2
    inner = f'<g>{sym}</g><path transform="translate({x:.2f} {base:.2f})" d="{d}" fill="{ink}"/>'
    return svg(inner, vb=f"0 0 {W:.1f} 48"), W / 48
