"""Neritic brand, round 2: refine B (Shelf band) against C (Chart label) and show the real pixels.

Writes into round2/:
  svg/        vector files per candidate (favicon, symbol, dark symbol, mono, app, lockups)
  icons/      the hand-drawn 16 px and 24 px PNG masters (colour and one-colour)
  render/     strips.html and favicons.html (sources for the screenshots)
  board.html  the review board (references shots/, which render.cjs fills)

Run from anywhere:  python3 round2/src/gen2.py && node round2/src/render.cjs
Fonts: Literata static instances in brand/src/fonts (git-ignored; see critique.md for the download).
"""
import os, html as H
from marks import (GEO, svg, fav_inner, symbol_inner, app_inner, mono_inner, master_png, lockup,
                   INK, NAVY, SEA, SHELF, FAV_SHELF, PAPER, FOG, DARK_SOLID, DARK_BAND, b_inner)

R2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("svg", "icons", "render", "shots"):
    os.makedirs(os.path.join(R2, d), exist_ok=True)

CANDS = {
    "b0": dict(name="B · round 1", short="B0", line="Round-1 Shelf band, unchanged, for reference.", masters=False),
    "r1": dict(name="R1 · Taper", short="R1",
               line="The conservative fix. Same top-to-bottom band, now 2 px at the north edge widening to 4.5 px in the south; the deep-water edge is smooth, the coast edge has one bay and one point.",
               masters=True),
    "r2": dict(name="R2 · Bight", short="R2",
               line="Real California geometry: a narrow northern shelf, pinched at Point Conception, opening east into the Southern California Bight. Two unequal solids: a large deep-water mass, a small land block.",
               masters=True),
    "r3": dict(name="R3 · Bight, closed south", short="R3",
               line="R2 with deep water returning along the south edge (the borderland beyond the Bight), so the tile keeps a navy base and corner on a light tab.",
               masters=True),
    "c2": dict(name="C2 · Chart label, refined", short="C2",
               line="The strongest alternative, given one fair pass: the tile fills the square, the n is larger, and hand-drawn 16/24 px masters put the n and the contour stubs on whole pixels.",
               masters=True),
}
DROPPED = {"r4": dict(name="R4 · Bight + island (tried, dropped)", short="R4", line="R2 with Santa Cruz Island.", masters=True)}
ALL = dict(CANDS, **DROPPED)


def write(rel, content, mode="w"):
    with open(os.path.join(R2, rel), mode) as f:
        f.write(content)


# ---------------------------------------------------------------- files
def build_files():
    for k, c in ALL.items():
        write(f"svg/{k}-favicon.svg", svg(fav_inner(k, f"{k}f"), w=32, h=32, title="Neritic") + "\n")
        if k == "r4":
            master_png(k, 16, os.path.join(R2, f"icons/{k}-16.png"))
            continue
        write(f"svg/{k}-symbol.svg", svg(symbol_inner(k, f"{k}s"), w=48, h=48, title=f"Neritic {c['short']}") + "\n")
        write(f"svg/{k}-symbol-dark.svg", svg(symbol_inner(k, f"{k}sd", dark=True), w=48, h=48, title=f"Neritic {c['short']}, dark grounds") + "\n")
        write(f"svg/{k}-mono.svg", svg(mono_inner(k, f"{k}m", "#000"), w=48, h=48, title=f"Neritic {c['short']}, one colour") + "\n")
        write(f"svg/{k}-app.svg", svg(app_inner(k, f"{k}a"), w=180, h=180, title=f"Neritic app icon {c['short']}") + "\n")
        s, _ = lockup(k, symbol_inner(k, f"{k}l"), NAVY)
        write(f"svg/{k}-lockup.svg", s + "\n")
        s, _ = lockup(k, symbol_inner(k, f"{k}ld", dark=True), FOG)
        write(f"svg/{k}-lockup-dark.svg", s + "\n")
        if c["masters"]:
            for sz in (16, 24):
                master_png(k, sz, os.path.join(R2, f"icons/{k}-{sz}.png"))
            master_png(k, 16, os.path.join(R2, f"icons/{k}-16-mono-black.png"), mono="#000000")
            master_png(k, 16, os.path.join(R2, f"icons/{k}-16-mono-white.png"), mono="#ffffff")
            if k in ("r2", "r3"):
                master_png(k, 16, os.path.join(R2, f"icons/{k}-16-mono-rim-black.png"), mono="#000000", rim=True)
                master_png(k, 16, os.path.join(R2, f"icons/{k}-16-mono-rim-white.png"), mono="#ffffff", rim=True)
                write(f"svg/{k}-mono-rim.svg", svg(mono_inner(k, f"{k}mr", "#000", rim=True), w=48, h=48, title=f"Neritic {c['short']}, one colour, rim kept") + "\n")


# ---------------------------------------------------------------- favicon <img> per slot
def fav_img(k, slot, auto=False):
    """What a browser shows in a 16/24/32 slot. 1x: the pixel master (16, 24) or the SVG (32);
    2x: always the SVG, drawn at twice the size (a favicon.ico would carry a 32/48 px render of the same SVG)."""
    vec = f"../svg/{k}-favicon.svg"
    if ALL[k]["masters"] and slot in (16, 24) and not auto:
        return f'<img src="../icons/{k}-{slot}.png" srcset="../icons/{k}-{slot}.png 1x, {vec} 2x" width="{slot}" height="{slot}" alt="">'
    return f'<img src="{vec}" width="{slot}" height="{slot}" alt="">'


# ---------------------------------------------------------------- neighbouring tab icons (generic, no brands)
NEIGH = [
    ("Inbox (4)", '<rect x="0" y="1" width="16" height="14" rx="3" fill="#d1493f"/><path d="M3.5 5h9v6.5h-9z" fill="none" stroke="#fff" stroke-width="1.4"/><path d="M3.5 5l4.5 3.6L12.5 5" fill="none" stroke="#fff" stroke-width="1.4"/>'),
    ("Tide tables · Harbor", '<circle cx="8" cy="8" r="8" fill="#1d4e89"/><path d="M2.5 7.2c1.4-1.2 2.4-1.2 3.7 0s2.4 1.2 3.7 0 2.4-1.2 3.6 0M2.5 10.6c1.4-1.2 2.4-1.2 3.7 0s2.4 1.2 3.7 0 2.4-1.2 3.6 0" fill="none" stroke="#fff" stroke-width="1.4"/>'),
    ("Field notes – Docs", '<path d="M2.5 0h7.5l3.5 3.5V16h-11z" fill="#3b7ddd"/><path d="M10 0v3.5h3.5" fill="#a9c7f2"/><path d="M5 7h6M5 9.5h6M5 12h4" stroke="#fff" stroke-width="1.2"/>'),
    ("Weather · 7-day", '<circle cx="8" cy="8" r="4.2" fill="#f2b42b"/><g stroke="#f2b42b" stroke-width="1.5" stroke-linecap="round"><path d="M8 .8v1.6M8 13.6v1.6M.8 8h1.6M13.6 8h1.6M2.9 2.9l1.1 1.1M12 12l1.1 1.1M2.9 13.1l1.1-1.1M12 4l1.1-1.1"/></g>'),
    ("The Coast Daily", '<rect width="16" height="16" rx="3" fill="#1f1f1f"/><text x="8" y="12.4" text-anchor="middle" font-family="Georgia,serif" font-weight="700" font-size="12.5" fill="#fff">a</text>'),
    ("Survey sheet", '<rect width="16" height="16" rx="2.5" fill="#1e8e3e"/><path d="M3 4h10v8H3zM3 8h10M8 4v8" fill="none" stroke="#fff" stroke-width="1.2"/>'),
    ("Launch ramps – Map", '<path d="M8 0.6C4.9 0.6 2.6 3 2.6 6c0 4 5.4 9.4 5.4 9.4s5.4-5.4 5.4-9.4C13.4 3 11.1.6 8 .6z" fill="#0f8f8b"/><circle cx="8" cy="6" r="2.1" fill="#fff"/>'),
    ("Surf report", '<rect width="16" height="16" rx="3.5" fill="#3aa6d6"/><path d="M1 11.5c2.5 0 3.5-6.5 8-6.5 2.6 0 4 1.6 4.4 3.2-1.6-1.1-3.6-.6-4.2 1.2-.6 2-2.3 3.1-4.8 3.1H1z" fill="#fff"/>'),
]


def neigh_icon(i):
    return f'<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">{NEIGH[i][1]}</svg>'


def tabstrip(k, theme, crowded=False, auto=False):
    order = [0, 1, 2, 3, "N", 4, 5, 6, 7]
    active = 2
    tabs = []
    for pos, it in enumerate(order):
        if it == "N":
            icon, title = fav_img(k, 16, auto=auto), "Neritic · My Coast"
        else:
            icon, title = neigh_icon(it), NEIGH[it][0]
        cls = "t" + (" act" if pos == active else "")
        tabs.append(f'<div class="{cls}">{icon}' + ("" if crowded else f'<span>{H.escape(title)}</span>') + '</div>')
    return f'<div class="strip {theme}{" crowd" if crowded else ""}">{"".join(tabs)}</div>'


STRIP_CSS = """
*{box-sizing:border-box}body{margin:0;padding:12px;background:#fff;font:12px/1 system-ui,-apple-system,"Segoe UI",sans-serif;width:max-content}
.shot{margin:0 0 14px;width:max-content}
.strip{display:flex;align-items:flex-end;padding:6px 8px 0;gap:2px;width:max-content}
.strip .t{display:flex;align-items:center;gap:8px;height:30px;padding:0 10px;width:134px;border-radius:8px 8px 0 0;white-space:nowrap;overflow:hidden}
.strip.crowd .t{width:34px;padding:0 9px;gap:0}
.strip .t img,.strip .t svg{flex:none;display:block}
.strip.light{background:#dee1e6;color:#3c4043}.strip.light .act{background:#fff;color:#202124}
.strip.dark{background:#202124;color:#9aa0a6}.strip.dark .act{background:#35363a;color:#e8eaed}
.strip .t span{overflow:hidden;text-overflow:ellipsis}
.tabs{display:flex;gap:10px;align-items:flex-end}
.tab1{display:flex;align-items:center;gap:8px;height:34px;padding:0 12px;width:140px;border-radius:9px 9px 0 0;font:12px system-ui,sans-serif}
.tab1 img{display:block;flex:none}.tab1.light{background:#fff;color:#202124;box-shadow:0 0 0 6px #dee1e6}.tab1.dark{background:#35363a;color:#e8eaed;box-shadow:0 0 0 6px #202124}
.chips{display:flex;gap:6px;align-items:center;padding:6px}
.chips.one{padding:2px}.chips.light{background:#f1f3f4}.chips.dark{background:#35363a}
.chips img{display:block}
.mono{display:flex;gap:8px;align-items:center;padding:6px}.mono.light{background:#f1f3f4}.mono.dark{background:#202124}.mono img{display:block}
"""


def build_strips():
    parts = []
    for k in ALL:
        # (a) true-size: one Chrome-like tab on light and on dark, then 24/32 chips
        parts.append(f'<div class="shot tabs" data-shot="tab-{k}"><div class="tab1 light">{fav_img(k, 16)}<span>Neritic</span></div>'
                     f'<div class="tab1 dark">{fav_img(k, 16)}<span>Neritic</span></div></div>')
        # (b) the icons alone for enlargement: 16 · 24 · 32 on light and dark tab colours
        for g in ("light", "dark"):
            parts.append(f'<div class="shot chips {g}" data-shot="chips-{k}-{g}">{fav_img(k, 16)}{fav_img(k, 24)}{fav_img(k, 32)}</div>')
            parts.append(f'<div class="shot chips one {g}" data-shot="one-{k}-{g}">{fav_img(k, 16)}</div>')
            if ALL[k]["masters"]:
                parts.append(f'<div class="shot chips one {g}" data-shot="auto-{k}-{g}">{fav_img(k, 16, auto=True)}</div>')
        if k == "r4":
            continue
        # (c) legibility among neighbours
        for theme in ("light", "dark"):
            parts.append(f'<div class="shot" data-shot="neigh-{k}-{theme}">{tabstrip(k, theme)}</div>')
            parts.append(f'<div class="shot" data-shot="crowd-{k}-{theme}">{tabstrip(k, theme, crowded=True)}</div>')
        # (d) one colour at 16 (pixel master) and 32 (vector)
        if ALL[k]["masters"]:
            parts.append(f'<div class="shot" style="display:flex" data-shot="mono-{k}">'
                         f'<div class="mono light"><img src="../icons/{k}-16-mono-black.png" width="16" height="16" alt=""><img src="../svg/{k}-mono.svg" width="32" height="32" alt=""></div>'
                         f'<div class="mono dark"><img src="../icons/{k}-16-mono-white.png" width="16" height="16" alt="">'
                         f'<svg width="32" height="32" viewBox="0 0 48 48">{mono_inner(k, f"{k}mw", "#fff")}</svg></div></div>')
            if k in ("r2", "r3"):
                parts.append(f'<div class="shot" style="display:flex" data-shot="monorim-{k}">'
                             f'<div class="mono light"><img src="../icons/{k}-16-mono-rim-black.png" width="16" height="16" alt=""><img src="../svg/{k}-mono-rim.svg" width="32" height="32" alt=""></div>'
                             f'<div class="mono dark"><img src="../icons/{k}-16-mono-rim-white.png" width="16" height="16" alt="">'
                             f'<svg width="32" height="32" viewBox="0 0 48 48">{mono_inner(k, f"{k}mrw", "#fff", rim=True)}</svg></div></div>')
    write("render/strips.html", f'<!doctype html><html><head><meta charset="utf-8"><title>strips</title><style>{STRIP_CSS}</style></head><body>{"".join(parts)}</body></html>\n')


# ---------------------------------------------------------------- board
FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link href="https://fonts.googleapis.com/css2?family=Literata:ital,opsz,wght@0,7..72,400;0,7..72,600;1,7..72,500&display=swap" rel="stylesheet">')


def lock_inline(k, dark, h, uid):
    s, ar = lockup(k, symbol_inner(k, uid, dark), FOG if dark else NAVY)
    return s.replace("<svg ", f'<svg height="{h}" width="{h * ar:.1f}" aria-hidden="true" ', 1)


def px(name, scale, label=None, cls=""):
    """an enlarged screenshot, nearest-neighbour. Width/height are set by JS from the natural size."""
    lab = f'<figcaption>{label}</figcaption>' if label else ""
    return f'<figure class="px {cls}"><img src="shots/{name}.png" data-scale="{scale}" alt="">{lab}</figure>'


VERDICT = {
    "b0": ("16 px: a pale stroke through a navy tile. Road, backslash or stylised N, as round 1 feared.",
           "Same reading, bigger. The bay and point appear only from 32 px."),
    "r1": ("16 px: still one pale stroke corner to corner; the bay notch makes it a zigzag or lightning bolt.",
           "Taper helps at 32 px and up; it does not change the 16 px genre. Not enough."),
    "r2": ("16 px: a navy tile with a slit at the top opening into a pale lower-right quarter. Not a road, not a letter.",
           "Map-like from 24 px: you can see a coast turn east. Risk: the pale corner can look like a folded or peeled corner."),
    "r3": ("16 px: an L or boot shape, which reads as a letter more than R2 does.",
           "Keeps the tile edge on light tabs, but the shelf now looks like a pipe or a sock."),
    "c2": ("16 px: an n with a dash each side. Instantly legible, and a letter tile; at 16 px the italic is gone, so it is a pixel n.",
           "The strongest-reading favicon here. Below 32 px the contour stubs read as dashes, not a contour."),
}


def cand_row(k):
    c = CANDS[k]
    v16, v24 = VERDICT[k]
    auto = ""
    return f'''
<section class="cand" id="cand-{k}">
  <div class="ctext"><h3>{c["name"]}</h3><p>{c["line"]}</p><p class="v"><b>Reads as:</b> {v16}</p><p class="v">{v24}</p></div>
  <div class="ctrue"><span class="lab">true size, 1x</span>
     <div class="truebox"><img src="shots/tab-{k}.png" data-scale="1" alt="">
     <div class="row2"><img src="shots/chips-{k}-light.png" data-scale="1" alt=""><img src="shots/chips-{k}-dark.png" data-scale="1" alt=""></div></div>
     <span class="lab small">16 · 24 · 32 px on light and dark tab colours</span></div>
  <div class="cbig"><span class="lab">16 px enlarged 8x (nearest-neighbour): light tab, dark tab, retina (2x, shown 4x)</span>
     <div class="bigrow">{px(f"one-{k}-light", 8, "1x light")}{px(f"one-{k}-dark", 8, "1x dark")}{px(f"one-{k}-light@2x", 4, "2x light")}{px(f"one-{k}-dark@2x", 4, "2x dark")}{auto}</div></div>
</section>'''


def neigh_block():
    rows = []
    for k in CANDS:
        rows.append(f'<div class="nb"><p class="nlab">{CANDS[k]["short"]}</p><div class="nstrips">'
                    f'<img src="shots/neigh-{k}-light.png" data-scale="1" alt=""><img src="shots/neigh-{k}-dark.png" data-scale="1" alt="">'
                    f'<div class="crowds"><img src="shots/crowd-{k}-light.png" data-scale="1" alt=""><img src="shots/crowd-{k}-dark.png" data-scale="1" alt=""></div>'
                    f'<div class="crowds"><img src="shots/crowd-{k}-light.png" data-scale="3" class="pxl" alt=""></div></div></div>')
    return "".join(rows)


MONO_NOTE = {
    "r1": "16 px: an R or a stylised N; the taper does not help here",
    "r2": "16 px: two loose shapes that read as a lowercase r",
    "r2rim": "one shape again; the shelf reads as a cut, the r is gone",
    "r3": "16 px: a reversed L",
    "r3rim": "one shape; reads as a tile with an L-shaped slot",
    "c2": "16 px: n with dashes; the strongest one-colour mark",
}


def mono_block():
    cells = []
    for k in CANDS:
        if not CANDS[k]["masters"]:
            continue
        cells.append(f'<figure class="mb"><img src="shots/mono-{k}.png" data-scale="1" alt=""><img src="shots/mono-{k}.png" data-scale="4" class="pxl" alt="">'
                     f'<figcaption>{CANDS[k]["short"]} · {MONO_NOTE[k]}</figcaption></figure>')
        if k in ("r2", "r3"):
            cells.append(f'<figure class="mb"><img src="shots/monorim-{k}.png" data-scale="1" alt=""><img src="shots/monorim-{k}.png" data-scale="4" class="pxl" alt="">'
                         f'<figcaption>{CANDS[k]["short"]}, rim kept · {MONO_NOTE[k + "rim"]}</figcaption></figure>')
    return "".join(cells)


def app_block():
    cells = []
    for k in CANDS:
        cells.append(f'<figure class="ab"><div class="home light"><img class="ios" src="svg/{k}-app.svg" width="180" height="180" alt="">'
                     f'<div class="small"><img class="ios s" src="svg/{k}-app.svg" width="60" height="60" alt=""><span>Neritic</span></div></div>'
                     f'<figcaption>{CANDS[k]["short"]} · 180 px, and 60 px as on a home screen</figcaption></figure>')
    return "".join(cells)


def lockup_block():
    rows = []
    for k in CANDS:
        rows.append(f'''<div class="lk"><p class="nlab">{CANDS[k]["short"]}</p>
  <div class="nav light">{lock_inline(k, False, 30, f"{k}nl")}<nav><a>My Coast</a><a>Map</a><a>Blooms</a><a>Method</a></nav><span class="tag">Research preview</span></div>
  <div class="nav dark">{lock_inline(k, True, 30, f"{k}nd")}<nav><a>My Coast</a><a>Map</a><a>Blooms</a></nav></div>
  <div class="nav light big">{lock_inline(k, False, 56, f"{k}nb")}</div></div>''')
    return "".join(rows)


def dark_options():
    """R2 on a near-black header: the three ways to keep a tile a tile."""
    opts = [
        ("navy tile, pale shelf (the light version)", b_inner("r2", "do1", solid=NAVY, band=SHELF)),
        ("pale solids, sea-blue shelf (chosen)", b_inner("r2", "do2", solid=DARK_SOLID, band=DARK_BAND)),
        ("round-1 style: two pale solids, no tile", b_inner("r2", "do3", solid=FOG, band=INK)),
    ]
    return "".join(f'<figure class="do"><div><svg width="48" height="48" viewBox="0 0 48 48">{inner}</svg>'
                   f'<svg width="24" height="24" viewBox="0 0 48 48">{inner.replace("do", "dq")}</svg></div><figcaption>{t}</figcaption></figure>'
                   for t, inner in opts)


BOARD_CSS = f"""
:root{{--ink:{INK};--navy:{NAVY};--sea:{SEA};--shelf:{SHELF};--paper:{PAPER};--fog:{FOG};--rule:#d9dde1;--muted:#5b6670}}
*{{box-sizing:border-box}}body{{margin:0;background:#eceff1;color:#17232c;font:14px/1.45 Literata,Georgia,serif}}
.wrap{{max-width:1440px;margin:0 auto;padding:36px 32px 48px}}
h1{{font:600 26px/1.2 Literata,serif;margin:0 0 6px}}h2{{font:600 19px/1.2 Literata,serif;margin:34px 0 4px}}
.sub{{color:var(--muted);margin:0 0 6px;max-width:1040px}}.hint{{color:var(--muted);margin:0 0 14px;font-size:13px;max-width:1040px}}
.argue{{background:#fff;border:1px solid var(--rule);border-left:4px solid var(--sea);border-radius:6px;padding:14px 18px;margin:18px 0 8px;max-width:1180px}}
.argue h3{{font:600 15px Literata,serif;margin:0 0 4px}}.argue p{{margin:0;font-size:13.5px}}
.lab{{display:block;font:11px/1.3 ui-monospace,Menlo,monospace;color:#7a858e;margin:0 0 8px}}.lab.small{{margin:8px 0 0}}
.cand{{display:grid;grid-template-columns:250px 320px 1fr;gap:20px;background:#fff;border:1px solid var(--rule);border-radius:6px;padding:16px 18px;margin:0 0 12px;align-items:start}}
.cand h3{{font:600 16px/1.2 Literata,serif;margin:0 0 4px}}.cand p{{margin:0 0 6px;font-size:12.5px;color:#33434f}}.cand .v{{color:#17232c}}
#cand-r2{{border:2px solid var(--sea)}}
.truebox{{display:flex;flex-direction:column;gap:12px;padding:14px 10px 10px;background:#f7f8f9;border-radius:4px;align-items:flex-start}}.row2{{display:flex;gap:10px}}
.bigrow{{display:flex;gap:10px;flex-wrap:nowrap;align-items:flex-start}}
.px{{margin:0}}.px img,.pxl{{image-rendering:pixelated;display:block}}.px figcaption,.mb figcaption,.ab figcaption,.do figcaption{{font:11px ui-monospace,Menlo,monospace;color:#7a858e;margin-top:4px}}
.px.auto{{padding-left:14px;border-left:1px dashed #c8d0d6}}
.panel{{background:#fff;border:1px solid var(--rule);border-radius:6px;padding:16px 18px}}
.nb{{display:grid;grid-template-columns:40px 1fr;gap:10px;align-items:start;padding:10px 0;border-bottom:1px solid #eef0f2}}.nb:last-child{{border:0}}
.nlab{{font:600 13px Literata,serif;margin:6px 0 0}}.nstrips{{display:flex;flex-direction:column;gap:6px}}.crowds{{display:flex;gap:10px;align-items:flex-end}}
.mbs{{display:grid;grid-template-columns:repeat(2,1fr);gap:22px 26px}}.mb{{margin:0;display:flex;flex-direction:column;gap:6px}}
.abs{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}}.ab{{margin:0}}
.home{{border-radius:6px;padding:16px;display:flex;flex-direction:column;align-items:center;gap:12px;background:linear-gradient(160deg,#c9d6df,#e9e1d3)}}
.ios{{border-radius:22.4%;display:block;box-shadow:0 1px 3px rgba(0,0,0,.2)}}.small{{display:flex;flex-direction:column;align-items:center;gap:4px;font:11px system-ui;color:#1d2a33}}
.lk{{display:grid;grid-template-columns:40px 1fr 360px 250px;gap:10px;align-items:center;padding:8px 0}}
.nav{{height:60px;display:flex;align-items:center;gap:28px;padding:0 20px;border-radius:4px}}.nav.light{{background:var(--paper);border:1px solid var(--rule)}}.nav.dark{{background:var(--ink)}}
.nav nav{{display:flex;gap:18px;font:500 14px system-ui;color:#2a3a46;white-space:nowrap}}.nav.dark nav{{color:#cfdbe3}}.nav.big{{height:88px;justify-content:center}}
.tag{{white-space:nowrap;margin-left:auto;font:500 italic 13px Literata,serif;color:var(--sea);border:1px solid #bcd3de;border-radius:99px;padding:3px 10px}}
.dos{{display:flex;gap:28px;background:var(--ink);padding:16px 20px;border-radius:6px}}.do{{margin:0}}.do div{{display:flex;gap:12px;align-items:center}}.do figcaption{{color:#9fb2bf}}
.rec{{background:#fff;border:1px solid var(--rule);border-radius:6px;padding:16px 20px;max-width:1180px}}.rec p{{margin:0 0 8px}}.rec ul{{margin:0 0 8px;padding-left:20px}}
"""

SCALE_JS = """<script>
// enlarge screenshots by an integer factor using their natural pixel size (nearest-neighbour via CSS)
for (const im of document.querySelectorAll('img[data-scale]')) {
  // the factor applies to screenshot pixels: a 1x shot at 8 and a 2x shot at 4 come out the same size
  const f = () => { const s = +im.dataset.scale; im.style.width = im.naturalWidth * s + 'px';
                    im.style.height = im.naturalHeight * s + 'px'; im.style.imageRendering = s > 1 ? 'pixelated' : 'auto'; };
  im.complete ? f() : im.addEventListener('load', f);
}
</script>"""

ARGUE = ("C (Chart label) is the strongest alternative, not A. The favicon is where B is being judged, and A loses there: "
         "after four passes its 16 px cut still reads as stripes with a plug, it needs three tints, and in one colour it turns into an H. "
         "A is the best illustration of the family, not a better identity. C is the opposite case. Its 16 px tile is the most legible thing on the board, "
         "it comes with a complete typographic system (italic for water names, spaced italic capitals for editorial use), and its "
         "contour-break logic is real chart practice. Its weakness is what makes it a fair test of B: at tab size C is a letter tile, "
         "the same genre as the rejected N tile. So the question for round 2 is whether B can become as legible as C at 16 px "
         "without turning into a letter, a road or a slash.")


def build_board():
    rows = "".join(cand_row(k) for k in CANDS)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neritic brand, round 2</title>{FONT_LINK}<style>{BOARD_CSS}</style></head><body><div class="wrap">
<h1>Neritic · brand round 2: Shelf band against Chart label</h1>
<p class="sub">Owner's brief: do not finalise B yet; refine it against the strongest alternative and show the actual 16/24/32 px results.
Every small image below is a real Chromium screenshot at device scale 1 (or 2 where marked). Enlargements are those screenshots scaled by whole numbers with nearest-neighbour, so each square is one screen pixel.</p>
<p class="hint">Three changes apply to every round-2 candidate, C2 included: the tile fills the favicon square (round 1 left 1.3 px of margin each side at 16 px); the favicon tile is one step lighter than brand navy (#1b5583), so its edge survives on a dark tab; and the 16 px and 24 px favicons are hand-drawn pixel masters. The SVG serves 32 px and every retina slot. To ship the masters, the site declares favicon.ico (16/24/32/48) and no SVG favicon, because Chrome and Firefox pick an SVG over the ICO when both are offered.</p>
<div class="argue"><h3>Strongest alternative: C · Chart label</h3><p>{ARGUE}</p></div>

<h2>1 · Favicons at actual size</h2>
<p class="hint">Left: what it reads as. Middle: true size in a Chrome-like tab and as 16/24/32 chips. Right: the 16 px favicon at 8x, then the retina version (32 device pixels) at 4x so it appears at the same size. favicons.png has every size at 8x, and the same SVGs rasterised without the hand-drawn masters.</p>
{rows}

<h2>2 · Among neighbouring tabs</h2>
<p class="hint">Eight generic neighbours (no real brands), chosen to be hard: a navy circle with waves, a blue surf swoosh, a teal pin and a black serif letter tile. Neritic is the fifth tab and not active, the way you scan for it. Bottom: crowded tabs (icon only), at 1x and 3x.</p>
<div class="panel">{neigh_block()}</div>

<h2>3 · One colour</h2>
<p class="hint">Black on a light tab and white on a dark one (pinned-tab masks, monochrome system icons, embossing): hand-drawn 16 px, then 32 px vector; true size, then 4x. Figure-ground marks have a catch in one colour: wherever the shelf touches the tile edge the outline opens, and what is left is two loose shapes.</p>
<div class="panel mbs">{mono_block()}</div>

<h2>4 · B on dark grounds</h2>
<p class="hint">Round 1's dark version (two pale shapes) read as brackets. In round 2 the mark is always a tile: pale solids with a sea-blue shelf (middle). Without the tile (right), R2's unequal solids no longer read as brackets, but they do read as a lowercase r, so the loose version is out.</p>
<div class="dos">{dark_options()}</div>

<h2>5 · App icon, 180 px</h2>
<p class="hint">The full-square drawing with an iOS-style corner mask at 180 px (apple-touch-icon), and at 60 px with its label.</p>
<div class="panel abs">{app_block()}</div>

<h2>6 · Nav lockups</h2>
<p class="hint">Upright Literata SemiBold for the B family, SemiBold Italic lowercase for C (outlined paths). Light header, dark header, and a larger size.</p>
<div class="panel">{lockup_block()}</div>

<h2>Recommendation</h2>
<div class="rec">{RECOMMEND}</div>
<p class="hint" style="margin-top:18px">Nothing here is installed in the app. Generated by round2/src/gen2.py; screenshots by round2/src/render.cjs (Playwright Chromium).</p>
</div>{SCALE_JS}</body></html>
"""
    write("board.html", html)


RECOMMEND = """<p><b>Carry R2 · Bight forward as the B direction; do not finalise it yet.</b> It is the only B variant that stops being a stroke at 16 px.
The band no longer runs corner to corner, the two solids are unequal, and in colour nothing about it is a letter, a road or a slash. Among wave, swoosh and
letter-tile neighbours it stays findable on both tab colours, and from 24 px up it reads as a coast turning east.</p>
<p><b>It does not beat C2 on raw 16 px legibility.</b> C2's n is the fastest thing to find in a crowded strip. But it is a letter tile, the genre the owner already
rejected, and below 32 px the italic and the contour are gone, leaving a pixel n between two dashes. R2 wins on meaning and on not being a letter. If 16 px legibility
must win outright, C2 is the honest pick.</p>
<p><b>Open issues for one more short round:</b></p><ul>
<li>One colour. Loose, R2's two solids read as a lowercase r. The one-colour version must keep the tile rim (shown in section 3), which turns the shelf into a cut.</li>
<li>The pale Bight touches the tile corner, which softens the outline on light tabs and can read as a folded corner.</li>
<li>A hand-cut 32 px master to sharpen the point (32 px is still the SVG).</li>
<li>Shipping: the masters only appear if the site serves favicon.ico and no SVG favicon.</li></ul>"""


# ---------------------------------------------------------------- favicons.png source (8x sheet)
def build_favsheet():
    rows = []
    for k, c in ALL.items():
        auto = ""
        if c["masters"]:
            auto = (f'<div class="g"><p>no hand drawing (SVG at 16, light / dark)</p><div class="pair">'
                    f'<img src="../shots/auto-{k}-light.png" data-scale="8"><img src="../shots/auto-{k}-dark.png" data-scale="8"></div></div>')
        retina = ""
        if k != "r4":
            retina = (f'<div class="g"><p>retina: device scale 2, shown 4x (same size as 8x)</p><div class="pair">'
                      f'<img src="../shots/chips-{k}-light@2x.png" data-scale="4"><img src="../shots/chips-{k}-dark@2x.png" data-scale="4"></div></div>')
        rows.append(f'''<section><h2>{c["name"]}</h2>
<div class="true"><span>true size:</span><img src="../shots/chips-{k}-light.png" data-scale="1"><img src="../shots/chips-{k}-dark.png" data-scale="1"></div>
<div class="g"><p>1x, enlarged 8x: 16 · 24 · 32 on light and dark tab colours</p><div class="pair">
<img src="../shots/chips-{k}-light.png" data-scale="8"><img src="../shots/chips-{k}-dark.png" data-scale="8"></div></div>
<div class="side">{retina}{auto}</div></section>''')
    css = """body{margin:0;padding:28px;background:#fff;font:14px system-ui,sans-serif;color:#17232c;width:1840px}
h1{font:600 24px Georgia,serif;margin:0 0 4px}.s{color:#5b6670;margin:0 0 18px}
section{border-top:1px solid #d9dde1;padding:16px 0 18px}h2{font:600 18px Georgia,serif;margin:0 0 8px}
.true{display:flex;gap:10px;align-items:center;margin:0 0 10px;font:12px ui-monospace,monospace;color:#7a858e}
.g p{font:12px ui-monospace,monospace;color:#7a858e;margin:0 0 6px}.pair{display:flex;gap:16px}.side{display:flex;flex-direction:column;gap:14px;margin-top:12px}
img{display:block;image-rendering:pixelated}"""
    html = (f'<!doctype html><html><head><meta charset="utf-8"><title>Neritic favicons, round 2</title><style>{css}</style></head><body>'
            f'<h1>Neritic round 2 · favicons, actual pixels</h1><p class="s">Chromium screenshots at device scale 1 and 2, enlarged with nearest-neighbour. '
            f'Tab colours: Chrome light #f1f3f4-ish and dark #35363a. 16 and 24 px are hand-drawn masters (except B round 1); 32 px and all 2x are the SVG.</p>'
            f'{"".join(rows)}{SCALE_JS}</body></html>\n')
    write("render/favicons.html", html)


if __name__ == "__main__":
    build_files()
    build_strips()
    build_board()
    build_favsheet()
    print("gen ok")
