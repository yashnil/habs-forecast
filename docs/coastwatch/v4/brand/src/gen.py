"""Neritic brand concepts, v4. Writes concept-{a,b,c}*.svg and board.html into the parent folder.

All symbol geometry is hand-written on a 48-unit grid (1 u = 1/3 px at 16 px). Wordmarks are Literata
(OFL, Google Fonts) outlined with fontTools from the static instances in src/fonts/.
Run: python3 src/gen.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wm

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INK = "#0b1a24"     # dark ground
NAVY = "#12395a"    # deep water / primary ink on light
SEA = "#2e86ab"     # slope
SHELF = "#bfe4ec"   # shelf water
PAPER = "#f6f4ef"   # light ground
FOG = "#e8eef2"     # primary ink on dark

SQ = "M15 4H33A11 11 0 0 1 44 15V33A11 11 0 0 1 33 44H15A11 11 0 0 1 4 33V15A11 11 0 0 1 15 4Z"
SQ_FULL = "M0 0H48V48H0Z"


def svg(inner, vb="0 0 48 48", w=None, h=None, title=None):
    size = f' width="{w}" height="{h}"' if w else ""
    t = f"<title>{title}</title>" if title else ""
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}"{size}>{t}{inner}</svg>'


# ---------------------------------------------------------------- A. Canyon (reduced contours)
# Three depth bands, plan view, coast off the right edge: deep (left), slope, shelf (right). The two
# isobaths between them (about 200 m and 2,000 m) are each cut by one canyon reaching up-coast, the way
# Monterey Canyon cuts the shelf. Two contours, one event; the bands are fills, so nothing thins out.
A_SLOPE = ("M48 0V48H19.5C19 41 18.6 35.4 20 32.2C21.7 28.6 25.6 26.6 29.4 23.6C31.7 21.8 30.7 18.8 28 20.2"
           "C24.6 22 21.2 23.6 19.1 21.2C17.6 19.4 17.2 10 16.8 0Z")
A_SHELF = ("M48 0V48H33.6C33.2 41.4 32.8 35.8 34.2 32.8C35.5 30 38.4 28 40.6 25.4C42 23.6 40.4 21.4 38.4 22.8"
           "C36.4 24.2 34.2 24.6 33 22.4C31.9 20.4 31.6 10 32 0Z")


def a_symbol(uid="a", full=False, deep=NAVY, slope=SEA, shelf=SHELF):
    clip = SQ_FULL if full else SQ
    return (f'<defs><clipPath id="{uid}c"><path d="{clip}"/></clipPath></defs>'
            f'<g clip-path="url(#{uid}c)"><path d="{clip}" fill="{deep}"/>'
            f'<path d="{A_SLOPE}" fill="{slope}"/><path d="{A_SHELF}" fill="{shelf}"/></g>')


def a_favicon(uid="af"):
    # 16-32 px cut: straight isobaths; the canyon as one parallel-sided, round-ended tongue 8.4 u thick
    # (2.8 px at 16 px); the shelf line stays straight. Tapered, angled, doubled or echoed tongues read as
    # arrows, tick marks or a plus at this size;
    # shelf tint a step deeper so the tile edge holds on a light tab
    return (f'<defs><clipPath id="{uid}c"><path d="{SQ}"/></clipPath></defs><g clip-path="url(#{uid}c)">'
            f'<path d="{SQ_FULL}" fill="{NAVY}"/>'
            f'<path d="M16.6 0H48V48H17.4V33.6C17.4 32.2 18.4 31.4 19.8 31.4H25.4A4.2 4.2 0 0 0 25.4 23H19.4C18 23 17.1 22.2 17 20.8Z" fill="{SEA}"/>'
            f'<path d="M33 0H48V48H33.6Z" fill="#9fd4e2"/></g>')


def a_mono(uid="am", color="#000"):
    # one colour: deep and shelf solid, the slope band left open (the canyon still cuts both isobaths)
    return (f'<defs><mask id="{uid}k" maskUnits="userSpaceOnUse" x="0" y="0" width="48" height="48">'
            f'<path d="{SQ}" fill="#fff"/><path d="{A_SLOPE}" fill="#000"/><path d="{A_SHELF}" fill="#fff"/></mask></defs>'
            f'<path d="{SQ}" fill="{color}" mask="url(#{uid}k)"/>')


# ---------------------------------------------------------------- B. Shelf band (figure-ground)
# Plan view of the California margin. Two solids: the deep ocean (west) and the land (east). The shelf is
# never drawn: it is the gap between them, narrow in the north, swelling at a bay, pinched at a point and
# widening into the Southern California Bight, as the real 200 m line does.
B_DEEP = "M-1 -1H13C15 10 17.6 17.6 20.6 23.4C23.4 29 23.4 38 22.6 49H-1Z"
B_LAND = "M49 -1H19C20.6 8.4 23.2 14.6 26.6 18.8C30.6 23.6 31 27 30.2 30.2C29.4 34 30.6 41 34 49H49Z"
B_ISO = "M9.5 -1C12.5 11 16.5 18.5 20.5 24.5C24 30 25.4 38 26 49"


def b_symbol(uid="b", color=NAVY, band=None, full=False, bg=None):
    clip = SQ_FULL if full else SQ
    under = f'<path d="{clip}" fill="{band}"/>' if band else ""
    base = f'<path d="{clip}" fill="{bg}"/>' if bg else ""
    return (f'<defs><clipPath id="{uid}c"><path d="{clip}"/></clipPath></defs>{base}'
            f'<g clip-path="url(#{uid}c)">{under}<path d="{B_DEEP}" fill="{color}"/>'
            f'<path d="{B_LAND}" fill="{color}"/></g>')


# ---------------------------------------------------------------- C. Chart label (wordmark-led)
# On charts, water is lettered in sloped (italic) type and a contour breaks where its label sits.
# Wordmark: "neritic" in Literata SemiBold Italic, lowercase. Symbol: one isobath across the tile, broken by
# the italic n, as a depth label breaks a contour. Editorial use: spaced italic capitals, as a chart names a sea.
def c_glyph_n():
    d, w, b = wm.outline("i600", "n", 42, base=0)
    x0, y0, x1, y1 = b
    return d, x0, y0, x1, y1


def c_symbol(uid="c", tile=NAVY, line=SHELF, glyph=FOG, full=False, mono=None):
    d, x0, y0, x1, y1 = c_glyph_n()
    gw, gh = x1 - x0, y1 - y0
    # centre the n optically (italic: centre of its bounding box)
    tx = 24 - (x0 + gw / 2)
    ty = 24 - (y0 + gh / 2) + 0.2
    gl, gr = x0 + tx, x1 + tx
    y = 25.4  # isobath height: through the middle of the n's bowl
    left = f"M{0 if full else 2} {y + 1.2}C6 {y + 1.2} 8 {y} {gl - 2.6:.2f} {y}"
    right = f"M{gr + 2.6:.2f} {y}C38 {y} 41 {y - 1.6} {48 if full else 46} {y - 1.6}"
    clip = SQ_FULL if full else SQ
    if mono:
        return (f'<defs><mask id="{uid}k" maskUnits="userSpaceOnUse" x="0" y="0" width="48" height="48">'
                f'<path d="{clip}" fill="#fff"/><g fill="none" stroke="#000" stroke-width="3.4" stroke-linecap="round">'
                f'<path d="{left}"/><path d="{right}"/></g>'
                f'<path transform="translate({tx:.2f} {ty:.2f})" d="{d}" fill="#000"/></mask></defs>'
                f'<path d="{clip}" fill="{mono}" mask="url(#{uid}k)"/>')
    return (f'<defs><clipPath id="{uid}c"><path d="{clip}"/></clipPath></defs>'
            f'<g clip-path="url(#{uid}c)"><path d="{clip}" fill="{tile}"/>'
            f'<g fill="none" stroke="{line}" stroke-width="3.4" stroke-linecap="round">'
            f'<path d="{left}"/><path d="{right}"/></g>'
            f'<path transform="translate({tx:.2f} {ty:.2f})" d="{d}" fill="{glyph}"/></g>')


# ---------------------------------------------------------------- wordmarks
def wordmark(kind):
    """returns (d, width, cap_top, baseline) at a 30 u size, baseline y=0"""
    if kind == "c":
        d, w, b = wm.outline("i600", "neritic", 32, track=0.008)
    else:
        # upright Literata SemiBold, tightened 1.5 %; 'ri' and 'ti' opened a hair for small sizes
        d, w, b = wm.outline("r600", "Neritic", 30, track=-0.015, adj={1: 0.004, 3: 0.006})
    return d, w, b


def lockup(kind, symbol_inner, ink, uid):
    """symbol (48 u) + wordmark; returns svg string and aspect"""
    d, w, b = wordmark(kind)
    cap = -b[1]
    # A/B: centre the caps on the symbol; C (lowercase italic): centre the x-height, nudged for the ascenders
    base = 24 + cap / 2 if kind != "c" else 24 + 0.511 * 32 / 2 + 1.2
    gap = 13 if kind != "c" else 15
    x = 48 + gap - b[0]
    W = x + b[2] + 2
    inner = (f'<g>{symbol_inner}</g>'
             f'<path transform="translate({x:.2f} {base:.2f})" d="{d}" fill="{ink}"/>')
    return svg(inner, vb=f"0 0 {W:.1f} 48"), W / 48


def wordmark_only(kind, ink):
    d, w, b = wordmark(kind)
    x0, y0, x1, y1 = b
    pad = 2
    return svg(f'<path transform="translate({pad - x0:.2f} {pad - y0:.2f})" d="{d}" fill="{ink}"/>',
               vb=f"0 0 {x1 - x0 + 2 * pad:.1f} {y1 - y0 + 2 * pad:.1f}"), (x1 - x0 + 2 * pad) / (y1 - y0 + 2 * pad)


# ---------------------------------------------------------------- files
CONCEPTS = {
    "a": dict(
        name="A · Canyon",
        line="Reduced contours. Three depth bands with two isobaths, both cut by one canyon reaching for the coast.",
        sym=lambda u, dark=False: a_symbol(u),
        mono=lambda u, c: a_mono(u, c),
        app=lambda u: a_symbol(u, full=True),
        fav=lambda u: a_favicon(u),
        ink_light=NAVY, ink_dark=FOG,
    ),
    "b": dict(
        name="B · Shelf band",
        line="Figure-ground. Deep ocean and land as two solids; the shelf is the gap between them, never drawn.",
        sym=lambda u, dark=False: b_symbol(u, color=FOG if dark else NAVY),
        mono=lambda u, c: b_symbol(u, color=c),
        app=lambda u: b_symbol(u, color=NAVY, band=SHELF, full=True),
        fav=lambda u: b_symbol(u, color=NAVY, band=SHELF),
        ink_light=NAVY, ink_dark=FOG,
    ),
    "c": dict(
        name="C · Chart label",
        line="Wordmark-led. Lowercase Literata italic, the chart convention for water; the symbol is an isobath broken by its label, the n.",
        sym=lambda u, dark=False: c_symbol(u, tile=SEA if dark else NAVY, line=INK if dark else SHELF, glyph=PAPER if dark else FOG),
        mono=lambda u, c: c_symbol(u, mono=c),
        app=lambda u: c_symbol(u, full=True),
        fav=lambda u: c_symbol(u),
        ink_light=NAVY, ink_dark=FOG,
    ),
}


def write(name, content):
    with open(os.path.join(OUT, name), "w") as f:
        f.write(content + "\n")


def build_files():
    for k, c in CONCEPTS.items():
        write(f"concept-{k}.svg", svg(c["sym"](f"{k}s"), w=48, h=48, title=f"Neritic concept {k.upper()}"))
        write(f"concept-{k}-dark.svg", svg(c["sym"](f"{k}d", True), w=48, h=48, title=f"Neritic concept {k.upper()}, for dark grounds"))
        write(f"concept-{k}-mono.svg", svg(c["mono"](f"{k}m", "#000"), w=48, h=48, title=f"Neritic concept {k.upper()}, one colour"))
        write(f"concept-{k}-favicon.svg", svg(c["fav"](f"{k}f"), w=32, h=32, title="Neritic"))
        write(f"concept-{k}-app.svg", svg(c["app"](f"{k}a"), w=512, h=512, title=f"Neritic app icon, concept {k.upper()}"))
        s, _ = lockup(k, c["sym"](f"{k}l"), c["ink_light"], k)
        write(f"concept-{k}-lockup.svg", s)
        s, _ = lockup(k, c["sym"](f"{k}ld", True), c["ink_dark"], k)
        write(f"concept-{k}-lockup-dark.svg", s)


# ---------------------------------------------------------------- board
FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link href="https://fonts.googleapis.com/css2?family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,600;1,7..72,500&display=swap" rel="stylesheet">')


def inline(inner, w, h, vb="0 0 48 48", cls=""):
    return f'<svg class="{cls}" viewBox="{vb}" width="{w}" height="{h}" aria-hidden="true">{inner}</svg>'


def lock_inline(k, dark, h, uid):
    c = CONCEPTS[k]
    s, ar = lockup(k, c["sym"](uid, dark), c["ink_dark"] if dark else c["ink_light"], k)
    return s.replace("<svg ", f'<svg height="{h}" width="{h * ar:.1f}" ', 1)


def editorial(k):
    c = CONCEPTS[k]
    if k == "a":
        return (f'<div class="ed ed-a">{inline(a_symbol("eda", full=True), 220, 220, cls="edsym")}'
                f'<div><p class="kick">Neritic · Coastal intelligence</p><h3>Monterey Bay, this week</h3>'
                f'<p>Chlorophyll near the canyon head is above the 2015–2024 median for mid-October. Observation, not a toxin test.</p></div></div>')
    if k == "b":
        return (f'<div class="ed ed-b">{inline(b_symbol("edb", color=NAVY, band=SHELF, full=True), 220, 220, cls="edsym")}'
                f'<div><p class="kick">Neritic · Research preview</p><h3>The water over the shelf</h3>'
                f'<p>Everything Neritic watches happens in the narrow band between the coast and the 200&nbsp;m line.</p></div></div>')
    d, w, b = wm.outline("i500", "NERITIC", 30, track=0.16)
    s = svg(f'<path transform="translate({2 - b[0]:.2f} {2 - b[1]:.2f})" d="{d}" fill="{FOG}"/>', vb=f"0 0 {b[2] - b[0] + 4:.1f} {b[3] - b[1] + 4:.1f}")
    ar = (b[2] - b[0] + 4) / (b[3] - b[1] + 4)
    big = s.replace("<svg ", f'<svg height="34" width="{34 * ar:.0f}" ', 1)
    return (f'<div class="ed ed-c"><svg class="edline" viewBox="0 0 400 120" preserveAspectRatio="none" aria-hidden="true">'
            f'<path d="M0 70C60 70 90 64 120 64" stroke="{SEA}" stroke-width="2" fill="none"/>'
            f'<path d="M290 64C330 64 360 58 400 58" stroke="{SEA}" stroke-width="2" fill="none"/>'
            f'<path d="M0 96C80 96 140 90 220 90S340 84 400 84" stroke="{SEA}" stroke-opacity=".45" stroke-width="1.4" fill="none"/>'
            f'<path d="M0 30C90 30 160 36 240 34S350 28 400 28" stroke="{SEA}" stroke-opacity=".45" stroke-width="1.4" fill="none"/></svg>'
            f'<div class="edword">{big}</div><p class="edcap">Lettered as a chart names a sea; the isobath breaks for its label.</p></div>')


def phone(k):
    return (f'<div class="phone"><div class="pbar">{lock_inline(k, True, 26, f"{k}ph")}'
            f'<span class="pnav">My Coast</span><span class="burger"><i></i><i></i></span></div>'
            f'<div class="pbody"><p class="pk">Santa Cruz Harbor</p><p class="pt">Bloom watch: elevated</p></div></div>')


def desktop(k):
    return (f'<div class="desk"><div class="dbar">{lock_inline(k, False, 30, f"{k}dk")}'
            f'<nav><a>My Coast</a><a>Map</a><a>Blooms</a><a>Harbours</a><a>Method</a></nav>'
            f'<span class="tag">Research preview</span></div></div>')


def fav_strip(k):
    cells = []
    for ground, cls in (("light tab", "tl"), ("dark tab", "td")):
        imgs = "".join(f'<img src="concept-{k}-favicon.svg" width="{s}" height="{s}" alt="">' for s in (16, 24, 32))
        cells.append(f'<div class="tab {cls}">{imgs}<span>Neritic</span></div>')
    return f'<div class="favs" id="fav-{k}">{"".join(cells)}</div>'


def card(k):
    c = CONCEPTS[k]
    return f'''
<section class="card">
  <header><h2>{c["name"]}</h2><p>{c["line"]}</p></header>
  <div class="cell"><span class="lab">favicon 16 · 24 · 32, actual pixels</span>{fav_strip(k)}</div>
  <div class="cell light"><span class="lab">nav lockup, light</span>{lock_inline(k, False, 32, f"{k}n1")}</div>
  <div class="cell dark"><span class="lab">nav lockup, dark</span>{lock_inline(k, True, 32, f"{k}n2")}</div>
  <div class="cell light row"><span class="lab">monochrome</span>{inline(c["mono"](f"{k}m1", "#000"), 72, 72)}{inline(c["mono"](f"{k}m2", "#000"), 32, 32)}
     <div class="inv">{inline(c["mono"](f"{k}m3", "#fff"), 72, 72)}{inline(c["mono"](f"{k}m4", "#fff"), 32, 32)}</div></div>
  <div class="cell light"><span class="lab">app icon 512, shown at 160 (square, and with an OS circle mask)</span>
     <div class="apps"><img class="app" src="concept-{k}-app.svg" width="160" height="160" alt=""><img class="app circ" src="concept-{k}-app.svg" width="160" height="160" alt=""></div></div>
  <div class="cell"><span class="lab">editorial</span>{editorial(k)}</div>
  <div class="cell grey"><span class="lab">phone header, 390 px</span>{phone(k)}</div>
</section>'''


CSS = f"""
:root{{--ink:{INK};--navy:{NAVY};--sea:{SEA};--shelf:{SHELF};--paper:{PAPER};--fog:{FOG};--rule:#d9dde1;--muted:#5b6670}}
*{{box-sizing:border-box}}body{{margin:0;background:#eceff1;color:#17232c;font:14px/1.45 Literata,Georgia,serif}}
.wrap{{max-width:1440px;margin:0 auto;padding:36px 32px 48px}}
h1{{font:600 26px/1.2 Literata,serif;margin:0 0 4px}}.sub{{color:var(--muted);margin:0 0 24px;max-width:900px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}}
.card{{background:#fff;border:1px solid var(--rule);border-radius:6px;overflow:hidden}}
.card header{{padding:16px 18px;border-bottom:1px solid var(--rule)}}.card h2{{font:600 18px/1.2 Literata,serif;margin:0 0 4px}}.card header p{{margin:0;color:var(--muted);font-size:13px}}
.cell{{position:relative;padding:30px 18px 18px;border-bottom:1px solid var(--rule);min-height:96px}}
.cell.dark{{background:var(--ink)}}.cell.light{{background:var(--paper)}}.cell.grey{{background:#dfe4e8}}
.cell.row{{display:flex;gap:18px;align-items:center}}.inv{{display:flex;gap:18px;align-items:center;background:var(--ink);padding:10px 14px;border-radius:4px}}
.lab{{position:absolute;top:8px;left:18px;font:11px/1 ui-monospace,Menlo,monospace;color:#7a858e}}.dark .lab{{color:#8a9aa6}}
.favs{{display:flex;flex-direction:column;gap:8px}}
.tab{{display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:8px 8px 0 0;font:13px system-ui;width:max-content}}
.tab img{{display:block}}.tl{{background:#f1f3f4;color:#202124}}.td{{background:#35363a;color:#e8eaed}}
.apps{{display:flex;gap:20px}}.app{{border-radius:36px;box-shadow:0 1px 3px rgba(0,0,0,.18)}}.app.circ{{border-radius:50%}}
.ed{{border-radius:4px;overflow:hidden}}
.ed-a,.ed-b{{display:grid;grid-template-columns:150px 1fr;gap:16px;background:var(--paper);align-items:center}}
.ed-a .edsym,.ed-b .edsym{{width:150px;height:150px;display:block}}
.ed .kick{{font:500 italic 12px Literata,serif;color:var(--sea);margin:0 0 4px}}.ed h3{{font:600 20px/1.2 Literata,serif;margin:0 0 6px;color:var(--navy)}}.ed p{{margin:0;font-size:13px}}
.ed-c{{background:var(--ink);position:relative;height:150px}}.edline{{position:absolute;inset:0;width:100%;height:100%}}
.edword{{position:absolute;left:0;right:0;top:38%;transform:translateY(-50%);display:flex;justify-content:center}}
.edcap{{position:absolute;bottom:10px;left:14px;right:14px;color:#9fb2bf;font:italic 12px Literata,serif;margin:0}}
.phone{{width:390px;max-width:100%;margin:0 auto;background:#fff;border-radius:18px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.15)}}
.pbar{{height:56px;background:var(--ink);display:flex;align-items:center;padding:0 16px;gap:12px}}.pbar svg{{flex:none}}
.pnav{{margin-left:auto;color:#cfdbe3;font:500 14px system-ui}}.burger{{display:flex;flex-direction:column;gap:5px;width:20px}}.burger i{{display:block;height:2px;background:#cfdbe3}}
.pbody{{padding:16px}}.pk{{margin:0;color:var(--muted);font-size:12px}}.pt{{margin:2px 0 0;font:600 18px Literata,serif;color:var(--navy)}}
.desks{{margin-top:28px;display:flex;flex-direction:column;gap:12px}}.desks h2{{font:600 18px Literata,serif;margin:0}}
.desk{{background:#fff;border:1px solid var(--rule);border-radius:6px;overflow:hidden}}
.dbar{{height:64px;display:flex;align-items:center;gap:36px;padding:0 28px;background:var(--paper);border-bottom:1px solid var(--rule)}}
.dbar nav{{display:flex;gap:24px;font:500 14px system-ui;color:#2a3a46}}.tag{{margin-left:auto;font:500 italic 13px Literata,serif;color:var(--sea);border:1px solid #bcd3de;border-radius:99px;padding:3px 10px}}
.note{{margin-top:24px;color:var(--muted);font-size:13px;max-width:1000px}}
"""


def build_board():
    cards = "".join(card(k) for k in CONCEPTS)
    desks = "".join(f'<div><p class="lab2">{CONCEPTS[k]["name"]}</p>{desktop(k)}</div>' for k in CONCEPTS)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neritic brand concepts</title>{FONT_LINK}<style>{CSS}.lab2{{margin:0 0 6px;font:12px ui-monospace,Menlo,monospace;color:#7a858e}}</style></head><body><div class="wrap">
<h1>Neritic · brand concepts, v4 (bathymetric)</h1>
<p class="sub">Three directions. Wordmark face: <b>Literata</b> (TypeTogether for Google Fonts, SIL OFL 1.1): a text serif drawn for screens with an optical-size axis, so it holds at 16–20 px nav sizes, and with a true italic, which carries the chart convention that water is lettered in sloped type. A and B use upright SemiBold; C uses SemiBold Italic, with Medium Italic spaced capitals for editorial labels. Wordmarks in the SVG files are outlined paths.</p>
<div class="grid">{cards}</div>
<div class="desks"><h2>Desktop header</h2>{desks}</div>
<p class="note">Geometry: A and B are simplified from the bundled isobaths (Monterey Canyon notch in the 200/500/1,000 m lines; the narrow northern shelf widening into the Southern California Bight). Nothing here is installed in the app.</p>
</div></body></html>"""
    # self-contained: the favicon and app-icon files are embedded as data URIs (same bytes as the .svg files)
    import base64, re
    def emb(m):
        with open(os.path.join(OUT, m.group(1)), "rb") as f:
            return 'src="data:image/svg+xml;base64,' + base64.b64encode(f.read()).decode() + '"'
    html = re.sub(r'src="(concept-[a-z]-(?:favicon|app)\.svg)"', emb, html)
    write("board.html", html)


if __name__ == "__main__":
    build_files()
    build_board()
    print("ok")
