"""Direction A prototypes, generated from the published Neritic data (v1, fetched 2026-10-15)."""
import json, math, datetime as dt
from pathlib import Path

HERE = Path(__file__).parent
D = HERE / "data"
TODAY = dt.date(2026, 10, 15)
ST = json.load(open(D / "stations-small.json"))
PI = [p for p in json.load(open(D / "port-intel-0bfda277e6.json"))["ports"] if p["port_code"] == 550][0]
CELLS = json.load(open(D / "charm-pda-oct15.geojson"))
OBS = json.load(open(D / "observations-7ad3890a8d.json"))
SCW, MW = ST["HABs-SantaCruzWharf"], ST["HABs-MontereyWharf"]
NEAR = sorted((f["properties"]["d"], f["properties"]["p"]) for f in CELLS["features"] if f["properties"]["d"] <= 15)

FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">'

# provisional mark, candidate B ("Depth N"), and the outlined wordmark
WORD = (HERE.parent / "logo/out/B-wordmark.svg").read_text()
WORD_PATH = WORD[WORD.index("<path"):WORD.index("</svg>")]
def brand(dark=False):
    ink = "#ffffff" if dark else "#0b2238"
    return (f'<a class="brand" href="mycoast.html" aria-label="Neritic home">'
            f'<svg width="27" height="27" viewBox="8 4.2 32 35.3" aria-hidden="true"><defs><clipPath id="bnA"><rect x="0" y="8.6" width="48" height="30.9"/></clipPath></defs>'
            f'<path clip-path="url(#bnA)" d="M10.6 44V8.6L22.6 12.9C25.9 14.1 27.6 16 28.9 19.2L37.4 39.5V4" stroke="{ink}" stroke-width="5.2" stroke-linejoin="miter" stroke-miterlimit="12" fill="none"/>'
            f'<circle cx="27.4" cy="7.2" r="3" fill="#e5902a"/></svg>'
            f'<svg height="17" viewBox="6.6 -87.3 319.0 91.3" fill="{ink}" aria-hidden="true" style="margin-bottom:2px">{WORD_PATH}</svg></a>')

def mast(current):
    items = [("mycoast.html", "My Coast"), ("map.html", "Map"), ("station.html", "Blooms"), ("#", "Fisheries")]
    nav = "".join(f'<a href="{h}"{" aria-current=\"page\"" if t == current else ""}>{t}</a>' for h, t in items)
    return f'''<header class="mast"><div class="mast-in">{brand()}<nav class="nav" aria-label="Primary">{nav}</nav>
<div class="mast-end"><span class="preview-tag">Research preview</span><button class="menu-btn" type="button" aria-haspopup="menu">Menu<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M2 4h10M2 7h10M2 10h10" stroke="currentColor" stroke-width="1.5"/></svg></button></div></div></header>
<nav class="tabs" aria-label="Primary">{nav}</nav>'''

FOOT = '''<footer class="foot"><div class="foot-in"><p>Neritic is a research preview and an independent research project, not affiliated with or endorsed by NOAA, CDFW or CDPH. It does not replace them: closures, advisories and seafood safety come from the agencies.</p>
<nav aria-label="More"><a href="#">Research view</a><a href="#">Data status</a><a href="#">About</a><a href="#">Data &amp; sources</a></nav>
<p class="proto">Prototype, Direction A (scientific editorial). Mark and wordmark: provisional candidate B. Data: the published Neritic dataset as of Thu 15 Oct 2026.</p></div></footer>'''

def page(title, current, body, extra_head="", extra_tail=""):
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>{title} — Neritic</title>{FONTS}<link rel="stylesheet" href="a.css">{extra_head}</head><body>{mast(current)}{body}{FOOT}{extra_tail}</body></html>'''

def ago(d):
    n = (TODAY - d).days
    return "today" if n == 0 else "yesterday" if n == 1 else f"{n} days ago"
def dfmt(d, wd=True):
    return (d.strftime("%a ") if wd else "") + f"{d.day} {d.strftime('%b')}"
def P(s): return dt.date.fromisoformat(s[:10])

# ---------------------------------------------------------------- shared facts
pda = SCW["summaries"]["pDA"]; pn = SCW["summaries"]["pn_seriata"]
last = P(pda["last_date"])
lead2 = PI["charm"]["leads"][2]["variables"]["particulate_domoic"]
MED, LO, HI = round(lead2["median"] * 100), round(lead2["min"] * 100), round(lead2["max"] * 100)
N_CELLS, NEAREST = lead2["n"], NEAR[0][0]
SRC = {
    "sn019": "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-019.aspx",
    "cdfw_anch": "https://wildlife.ca.gov/News/Archive/commercial-and-recreational-take-restrictions-in-place-for-northern-anchovy-in-monterey-bay-due-to-public-health-hazard",
    "sn008": "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-008.aspx",
    "sn018": "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-018.aspx",
    "sn021": "https://www.cdph.ca.gov/Programs/OPA/Pages/SN26-021.aspx",
    "cdfw": "https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories",
    "cdph": "https://www.cdph.ca.gov/Programs/CEH/DRSEM/Pages/EMB/Shellfish/Marine-Biotoxin-Monitoring-Program.aspx",
    "scw": SCW["source_url"], "nws": "https://forecast.weather.gov/shmrn.php?mz=pzz535",
    "charm": "https://coastwatch.pfeg.noaa.gov/erddap/griddap/wvcharmV3_2day.html",
    "viirs": "https://coastwatch.noaa.gov/erddap/griddap/noaacwN21VIIRSchlaSectorUSDaily.html",
}

# ---------------------------------------------------------------- model distance strip
def distance_strip(w=640, phone=False):
    h = 118 if not phone else 132
    x0, x1 = 14, w - 14
    sx = lambda km: x0 + (x1 - x0) * km / 15
    rows = []
    rows.append(f'<defs><pattern id="hz{w}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill="#fff"/><line x1="0" y1="0" x2="0" y2="6" stroke="#9aa6af" stroke-width="1.4"/></pattern></defs>')
    yb = 46
    rows.append(f'<rect x="{sx(0):.1f}" y="{yb-16}" width="{sx(NEAREST)-sx(0):.1f}" height="32" fill="url(#hz{w})" stroke="#9aa6af" stroke-width="1"/>')
    rows.append(f'<text x="{(sx(0)+sx(NEAREST))/2:.1f}" y="{yb+4}" text-anchor="middle" style="font-size:{12 if phone else 13}px;fill:#34424d;font-weight:600;paint-order:stroke;stroke:#fff;stroke-width:4px">no model value</text>')
    for d, p in NEAR:
        rows.append(f'<rect x="{sx(d)-2.2:.1f}" y="{yb-14}" width="4.4" height="28" fill="none" stroke="#5a3d9e" stroke-width="1.2" opacity=".85"/>')
    # harbor
    rows.append(f'<line x1="{sx(0)}" y1="{yb-26}" x2="{sx(0)}" y2="{yb+26}" stroke="#0c1821" stroke-width="2"/>')
    rows.append(f'<text x="{sx(0)}" y="{yb-32}" class="lbl" style="font-size:13px">Monterey harbor</text>')
    rows.append(f'<text x="{sx(NEAREST):.1f}" y="{yb-32}" style="fill:#5a3d9e;font-weight:600;font-size:13px">{N_CELLS} offshore model cells</text>' if not phone else
                f'<text x="{x1}" y="{yb-32}" text-anchor="end" style="fill:#5a3d9e;font-weight:600;font-size:13px">{N_CELLS} offshore cells</text>')
    # axis
    ya = yb + 36
    rows.append(f'<line x1="{x0}" y1="{ya}" x2="{x1}" y2="{ya}" stroke="#b9c3ca"/>')
    for km in (0, 5, 10, 15):
        rows.append(f'<line x1="{sx(km):.1f}" y1="{ya}" x2="{sx(km):.1f}" y2="{ya+5}" stroke="#b9c3ca"/><text x="{sx(km):.1f}" y="{ya+19}" text-anchor="middle">{km} km</text>')
    rows.append(f'<line x1="{sx(NEAREST):.1f}" y1="{ya}" x2="{sx(NEAREST):.1f}" y2="{ya+5}" stroke="#5a3d9e"/><text x="{sx(NEAREST):.1f}" y="{ya+33 if phone else ya+19}" text-anchor="{"middle"}" style="fill:#5a3d9e">{"" if not phone else ""}{NEAREST:.1f} km</text>' if True else "")
    return f'<svg class="chart" viewBox="0 0 {w} {h + (14 if phone else 0)}" role="img" aria-label="Distance from Monterey harbor: no model value from 0 to {NEAREST:.1f} km; {N_CELLS} offshore cells between {NEAREST:.1f} and 15 km.">{"".join(rows)}</svg>'

# ---------------------------------------------------------------- 12-month evidence chart
T0, T1 = dt.date(2025, 10, 15), dt.date(2026, 10, 19)
def evidence_chart(w, phone=False, rows=("pn", "pda", "chl", "model"), today_label=True):
    pad_l = 14 if phone else 250
    pad_r = 14 if phone else 24
    x0, x1 = pad_l, w - pad_r
    span = (T1 - T0).days
    sx = lambda d: x0 + (x1 - x0) * (d - T0).days / span
    rh = 92 if not phone else 96        # plot height per row
    head = 0 if not phone else 34       # phone: label above the row
    gap = 40 if not phone else 30
    out = []
    out.append(f'<defs><pattern id="hg{w}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#c4ccd2" stroke-width="1.2"/></pattern></defs>')
    y = 28
    samples = [(P(r["t"]), r) for r in SCW["rows"]]
    def gaps(top, height, label=True):
        # spans with no sample for more than 13 days, including the run up to today
        ds = [d for d, _ in samples] + [TODAY]
        g = []
        for a, b in zip(ds, ds[1:]):
            if (b - a).days > 13:
                g.append(f'<rect x="{sx(a)+3:.1f}" y="{top}" width="{max(0, sx(b)-sx(a)-6):.1f}" height="{height}" fill="url(#hg{w})"/>')
        return "".join(g)
    def ylabel(top, title, unit, where):
        if phone:
            return (f'<text x="{x0}" y="{top-18}" class="lbl">{title}</text>'
                    f'<text x="{x0}" y="{top-3}" class="unit">{unit.split(",")[0]}{", log scale" if "log" in unit else ""} · {where}</text>')
        return (f'<text x="0" y="{top+14}" class="lbl">{title}</text><text x="0" y="{top+31}" class="unit">{unit}</text>'
                f'<text x="0" y="{top+48}" class="unit">{where}</text>')
    for key in rows:
        top = y + head
        if key == "pn":
            lo, hi = 2, 7
            sy = lambda v: top + rh - (math.log10(v) - lo) / (hi - lo) * rh
            out.append(ylabel(top, "Pseudo-nitzschia", "cells/L, larger cells, log scale", "Santa Cruz Wharf · 41 km NNW"))
            out.append(gaps(top, rh))
            for t, lab in ((2, "100"), (4, "10k" if phone else "10,000"), (6, "1M" if phone else "1,000,000")):
                yy = top + rh - (t - lo) / (hi - lo) * rh
                out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#e6eaed"/><text x="{(x0+2) if phone else (x0-16)}" y="{(yy-4) if phone else (yy+4):.1f}" text-anchor="{'start' if phone else 'end'}" class="tick">{lab}</text>')
            for d, r in samples:
                v = r["pn_seriata"]
                if v is None: continue
                if v == 0:
                    out.append(f'<circle cx="{sx(d):.1f}" cy="{top+rh:.1f}" r="2.8" fill="#fff" stroke="#0a6a60" stroke-width="1.2"/>')
                else:
                    out.append(f'<circle cx="{sx(d):.1f}" cy="{sy(max(v,100)):.1f}" r="3.2" fill="#0a6a60"/>')
            mx = max(((r["pn_seriata"] or 0), P(r["t"])) for r in SCW["rows"])
            lab = "2.2 million, 12-month high" if phone else f"{mx[0]:,.0f}, highest in 12 months"
            right = sx(mx[1]) < x0 + (x1 - x0) * 0.55
            out.append(f'<text x="{sx(mx[1]) + (8 if right else -8):.1f}" y="{sy(mx[0])+4:.1f}" text-anchor="{"start" if right else "end"}" style="font-size:12px;paint-order:stroke;stroke:#f5f7f8;stroke-width:4px">{lab}</text>')
            ld = P(pn["last_date"]); lv = pn["last_value"]
            out.append(f'<circle cx="{sx(ld):.1f}" cy="{sy(lv):.1f}" r="7" fill="none" stroke="#0c1821" stroke-width="1.5"/>')
            out.append(f'<text x="{sx(ld)-10:.1f}" y="{sy(lv)-22:.1f}" text-anchor="end" style="fill:#0c1821;font-weight:600;font-size:13px;paint-order:stroke;stroke:#f5f7f8;stroke-width:5px">4,900 · 30 Sep</text>')
        elif key == "pda":
            hi = 12
            sy = lambda v: top + rh - v / hi * rh
            out.append(ylabel(top, "Domoic acid in seawater", "ng/mL, particulate", "Santa Cruz Wharf · 41 km NNW"))
            out.append(gaps(top, rh))
            for t in (0, 5, 10):
                yy = sy(t)
                out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#e6eaed"/><text x="{(x0+2) if phone else (x0-16)}" y="{(yy-4) if phone else (yy+4):.1f}" text-anchor="{'start' if phone else 'end'}" class="tick">{t}</text>')
            for d, r in samples:
                v = r["pDA"]
                if v is None: continue
                if v == 0:
                    out.append(f'<circle cx="{sx(d):.1f}" cy="{sy(0):.1f}" r="2.8" fill="#fff" stroke="#0a6a60" stroke-width="1.2"/>')
                else:
                    out.append(f'<circle cx="{sx(d):.1f}" cy="{sy(v):.1f}" r="3.2" fill="#0a6a60"/>')
            ld = P(pda["last_date"]); lv = pda["last_value"]
            out.append(f'<circle cx="{sx(ld):.1f}" cy="{sy(lv):.1f}" r="7" fill="none" stroke="#0c1821" stroke-width="1.5"/>')
            out.append(f'<text x="{sx(ld)-10:.1f}" y="{sy(lv)-12:.1f}" text-anchor="end" style="fill:#0c1821;font-weight:600;font-size:13px;paint-order:stroke;stroke:#f5f7f8;stroke-width:5px">0.20 · 30 Sep</text>')
            mx = max((r["pDA"] or 0, P(r["t"])) for r in SCW["rows"])
            lab = f"{mx[0]:.1f}, 12-month high" if phone else f"{mx[0]:.1f}, highest in 12 months"
            right = sx(mx[1]) < x0 + (x1 - x0) * 0.55
            out.append(f'<text x="{sx(mx[1]) + (8 if right else -8):.1f}" y="{sy(mx[0])+4:.1f}" text-anchor="{"start" if right else "end"}" style="font-size:12px;paint-order:stroke;stroke:#f5f7f8;stroke-width:4px">{lab}</text>')
        elif key == "chl":
            hist = [(P(h["date"]), h["value"]) for h in PI["chlorophyll"]["history"] if P(h["date"]) >= T0]
            lo, hi = math.log10(0.3), math.log10(10)
            sy = lambda v: top + rh - (math.log10(v) - lo) / (hi - lo) * rh
            out.append(ylabel(top, "Chlorophyll, satellite", "mg/m³, log scale", "8-day median within 15 km"))
            for t in (0.3, 1, 3, 10):
                yy = sy(t)
                out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#e6eaed"/><text x="{(x0+2) if phone else (x0-16)}" y="{(yy-4) if phone else (yy+4):.1f}" text-anchor="{'start' if phone else 'end'}" class="tick">{t:g}</text>')
            first = hist[0][0]
            out.append(f'<rect x="{x0}" y="{top}" width="{sx(first)-x0-4:.1f}" height="{rh}" fill="url(#hg{w})"/>')
            out.append(f'<text x="{(x0+sx(first))/2:.1f}" y="{top+rh/2+4:.1f}" text-anchor="middle" style="font-size:12px;fill:#52606b;paint-order:stroke;stroke:#f5f7f8;stroke-width:5px">not kept before {dfmt(first, False)}</text>')
            seg = []; prev = None
            for d, v in hist:
                if prev and (d - prev).days > 2:
                    out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="#0a6a60" stroke-width="1.8"/>'); seg = []
                seg.append(f"{sx(d):.1f},{sy(max(v,0.3)):.1f}"); prev = d
            out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="#0a6a60" stroke-width="1.8"/>')
            d, v = hist[-1]
            out.append(f'<circle cx="{sx(d):.1f}" cy="{sy(v):.1f}" r="3.5" fill="#0a6a60"/>')
        elif key == "model":
            hist = [(P(h["date"]), h["value"]) for h in PI["charm"]["history"]["particulate_domoic"]]
            leads = [(P(l["valid_date"]), l["variables"]["particulate_domoic"]["median"]) for l in PI["charm"]["leads"]]
            sy = lambda v: top + rh - v * rh
            out.append(ylabel(top, "NOAA model, offshore", "% chance pDA > 500 ng/L", f"{NEAREST:.1f}–15 km out, not the harbor"))
            out.append(f'<rect x="{x0}" y="{top}" width="{x1-x0}" height="{rh}" fill="#faf8fe" stroke="#d9cff0" stroke-dasharray="3 3"/>')
            for t in (0, 0.5, 1):
                yy = sy(t)
                out.append(f'<text x="{(x0+4) if phone else (x0-16)}" y="{(yy+12 if t==1 else yy-4) if phone else (yy+4):.1f}" text-anchor="{'start' if phone else 'end'}" class="tick">{int(t*100)}%</text>')
            first = hist[0][0]
            out.append(f'<text x="{(x0+sx(first))/2:.1f}" y="{top+rh/2+4:.1f}" text-anchor="middle" style="font-size:12px;fill:#52606b">not kept before {dfmt(first, False)}</text>')
            seg = []; prev = None
            for d, v in hist:
                if prev and (d - prev).days > 2:
                    out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="#5a3d9e" stroke-width="1.6"/>'); seg = []
                seg.append(f"{sx(d):.1f},{sy(v):.1f}"); prev = d
            out.append(f'<polyline points="{" ".join(seg)}" fill="none" stroke="#5a3d9e" stroke-width="1.6"/>')
            pts = " ".join(f"{sx(d):.1f},{sy(v):.1f}" for d, v in [hist[-1]] + leads[1:])
            out.append(f'<polyline points="{pts}" fill="none" stroke="#5a3d9e" stroke-width="1.6" stroke-dasharray="3 3"/>')
            dv, vv = leads[2]
            out.append(f'<rect x="{sx(dv)-4:.1f}" y="{sy(vv)-4:.1f}" width="8" height="8" fill="#fff" stroke="#5a3d9e" stroke-width="1.6"/>')
            out.append(f'<text x="{sx(dv)-10:.1f}" y="{sy(vv)-10:.1f}" text-anchor="end" style="fill:#5a3d9e;font-weight:600;font-size:13px;paint-order:stroke;stroke:#faf8fe;stroke-width:4px">{MED}% offshore median</text>')
        y += rh + head + gap
    # month axis
    ya = y - gap + 14
    months = []
    m = dt.date(2025, 11, 1)
    while m < T1:
        months.append(m); m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    for m in months:
        if phone and m.month % 3 != 1 and m.month not in (10,):
            continue
        lab = m.strftime("%b") + (f" {m.year}" if m.month == 1 else "")
        out.append(f'<line x1="{sx(m):.1f}" y1="{ya-8}" x2="{sx(m):.1f}" y2="{ya-3}" stroke="#b9c3ca"/><text x="{sx(m):.1f}" y="{ya+10}" text-anchor="middle">{lab}</text>')
    # today
    out.append(f'<line x1="{sx(TODAY):.1f}" y1="14" x2="{sx(TODAY):.1f}" y2="{ya-8}" stroke="#0c1821" stroke-width="1.2"/>')
    out.append(f'<text x="{sx(TODAY):.1f}" y="9" text-anchor="end" style="fill:#0c1821;font-weight:600;font-size:12px">Today, Thu 15 Oct</text>')
    H = ya + 18
    aria = "Twelve months of evidence near Monterey: Pseudo-nitzschia and domoic acid at Santa Cruz Wharf (41 km away), satellite chlorophyll within 15 km, and NOAA's offshore model median."
    return f'<svg class="chart" viewBox="0 0 {w} {H}" role="img" aria-label="{aria}">{"".join(out)}</svg>'

# ---------------------------------------------------------------- the notice block
def notices_block(compact=False):
    return f'''<div class="notice-head"><span class="count">2 notices apply to commercial fishing here</span><span class="unverified">Not verified by a person</span></div>
<p class="agency-links">Neritic's copy of the agencies' pages, checked Thu 15 Oct. Their wording controls: <a class="ext" href="{SRC['cdfw']}">CDFW</a> · <a class="ext" href="{SRC['cdph']}">CDPH</a></p>
<ul class="notice-list">
<li><div class="agency">CDFW<span>since 11 Sep</span></div><div><p class="notice-title">Northern anchovy: no take for human consumption, Pigeon Point to Point Lobos</p>
<p class="notice-meta">Commercial and recreational. Monterey is inside the notice's latitudes; its south limit (36°31.46′ N) is 8.4 km south of the harbor.</p>
<p class="notice-links"><a class="ext" href="{SRC['cdfw_anch']}">CDFW news release</a></p></div></li>
<li><div class="agency">CDPH<span>since 14 Sep</span></div><div><p class="notice-title">Do not eat northern anchovies from Pigeon Point to Point Lobos</p>
<p class="notice-meta">Health advisory, same waters.</p><p class="notice-links"><a class="ext" href="{SRC['sn019']}">CDPH SN26-019</a></p></div></li>
</ul>
<p class="also"><b>Also in these waters, for sport shellfish harvest:</b> the statewide mussel quarantine (<a class="ext" href="{SRC['sn008']}">CDPH SN26-008</a>), and the Monterey County bivalve warning, which <span class="status-word">CDPH appears to have lifted</span> (<a class="ext" href="{SRC['sn021']}">SN26-021</a>, original <a class="ext" href="{SRC['sn018']}">SN26-018</a>). Neritic treats it as neither in effect nor lifted until a person checks.</p>'''

def mrow(q, sub, v, u, where, when, kind=""):
    return (f'<div class="mrow"><p class="q"><span class="kind {kind}"></span>{q}<small>{sub}</small></p>'
            f'<div class="mline"><span class="v">{v}<span class="u">{u}</span></span><span class="where">{where}</span><span class="when">{when}</span></div></div>')
def measures_table():
    late = f'{dfmt(last)} · <span class="age-late">{ago(last)}</span>'
    return ('<div class="measures">'
        + mrow("Pseudo-nitzschia", "the algae that can make domoic acid", "4,900", "cells/L", "<b>Santa Cruz Wharf</b>, 41 km NNW", late)
        + mrow("Domoic acid", "in seawater, not seafood", "0.20", "ng/mL", "<b>Santa Cruz Wharf</b>, 41 km NNW", late)
        + mrow("Chlorophyll", "from satellite, algae of every kind", "1.0", "mg/m³", "<b>Within 5 km</b>, median of 24 clear pixels", f"{dfmt(dt.date(2026,10,12))} · {ago(dt.date(2026,10,12))}", "sat")
        + '</div><p class="near-note"><b>Monterey Wharf, 0.6 km away,</b> is sampled weekly for water temperature (16.2 °C on Wed 14 Oct). Pseudo-nitzschia was last counted there in Aug 2025 and domoic acid last measured in Aug 2022.</p>')

def model_block(w=640):
    return f'''<div class="model"><h3><span class="tag">NOAA C-HARM model · valid Thu 15 Oct · issued Wed 14 Oct</span>No model value for Monterey's nearshore waters</h3>
<figure class="model-fig" style="margin:16px 0 0"><div class="fig-desk">{distance_strip(w)}</div><div class="fig-phone">{distance_strip(360, phone=True)}</div></figure>
<p class="model-cap">Offshore, {NEAREST:.1f}–15 km out, the {N_CELLS} cells have a median <span class="offshore-value">{MED}%</span> chance (range {LO}–{HI}%) that particulate domoic acid exceeds 500 ng/L. That describes open water, not the harbor, and it is a probability for seawater, not a seafood test.</p></div>'''

# ================================================================= My Coast
def mycoast():
    body = f'''<main class="page" id="main">
<div class="lede-grid"><div class="title-block">
<h1 class="page-title">Monterey</h1><p class="kicker-line">CDFW port, Monterey County</p>
<p class="trip-line"><button class="pick" type="button">Commercial fishing</button><span class="dot-sep">·</span><button class="pick" type="button">any species</button><span class="dot-sep">·</span><button class="pick" type="button">today, Thu 15 Oct</button><span class="sr">Tap any part to change it.</span></p>
<p class="answer"><strong>Two official notices apply here</strong>, both about northern anchovy. <strong>No recent water sample comes from Monterey itself</strong>: the nearest was taken {ago(last)}, at Santa Cruz Wharf, 41 km away. NOAA's bloom model has no value for these nearshore waters.</p>
</div></div>

<section class="sec notices-sec" aria-labelledby="n-h"><h2 id="n-h">Official notices</h2><div class="body">{notices_block()}</div>
<div class="notes"><p>A notice missing here does not mean the water is open or the seafood safe.</p><p>Transcribed from agency pages by Neritic, not checked by a person yet.</p></div></section>

<section class="sec" aria-labelledby="e-h"><h2 id="e-h">Bloom and toxin evidence</h2><div class="body">
<p class="lead-sentence">The nearest recent samples are <b>41 km away</b>, at Santa Cruz Wharf.</p>
{measures_table()}
{model_block()}
<a class="more-link" href="evidence.html">The 12-month record and every source</a></div>
<div class="notes"><p>Samples describe seawater at the pier on that day. Only CDPH and CDFW test seafood.</p><p>Chlorophyll measures algae of every kind, not toxin.</p><p>Sources: <a class="ext" href="{SRC['scw']}">CalHABMAP, SCCOOS</a> · <a class="ext" href="{SRC['viirs']}">NOAA CoastWatch VIIRS</a> · <a class="ext" href="{SRC['charm']}">NOAA C-HARM v3.1</a></p></div></section>

<section class="sec" aria-labelledby="c-h"><h2 id="c-h">Wind and sea</h2><div class="body">
<p class="cond">NE wind 5 to 10 kt, seas around 3 ft today.</p>
<p class="cond-sub">Waves 3.6 ft now at the Monterey Canyon Outer buoy. <a class="ext" href="{SRC['nws']}">NWS marine forecast, Monterey Bay (PZZ535)</a></p></div>
<div class="notes"><p>Wind and sea, not a bloom or fishing forecast. NWS issued 8:14 AM PDT.</p></div></section>

<section class="sec" aria-labelledby="b-h"><h2 id="b-h">Before you leave</h2><div class="body">
<ul class="checks">
<li><input type="checkbox" id="c1"><label for="c1">Read the two anchovy notices in the agencies' words<span class="src"><a class="ext" href="{SRC['cdfw_anch']}">CDFW</a> · <a class="ext" href="{SRC['sn019']}">CDPH SN26-019</a></span></label></li>
<li><input type="checkbox" id="c2"><label for="c2">Check CDFW's commercial rules and permits for your species<span class="src"><a class="ext" href="https://wildlife.ca.gov/Fishing/Commercial">CDFW commercial fishing</a></span></label></li>
<li><input type="checkbox" id="c3"><label for="c3">Read NWS's full marine forecast for your zone<span class="src"><a class="ext" href="{SRC['nws']}">NWS PZZ535</a></span></label></li>
<li><input type="checkbox" id="c4"><label for="c4">If you harvest shellfish, confirm the Monterey County warning with CDPH<span class="src"><a class="ext" href="{SRC['sn021']}">SN26-021, the later release</a></span></label></li>
</ul></div>
<div class="notes"><p>Ticks stay on this device. A tick means you opened the source, not that anything is safe or open.</p></div></section>
</main>'''
    css = "<style>.fig-phone{display:none}@media(max-width:760px){.fig-desk{display:none}.fig-phone{display:block}}</style>"
    return page("My Coast: Monterey", "My Coast", body, css)

# ================================================================= Evidence
def evidence():
    body = f'''<main class="page" id="main">
<div class="lede-grid"><div class="title-block"><p class="kicker-line" style="margin:0 0 8px"><a href="mycoast.html">← Monterey</a></p>
<h1 class="page-title" style="font-size:var(--t-5)">Bloom and toxin evidence near Monterey</h1>
<p class="answer">Measured at Santa Cruz Wharf, 41 km away, about weekly; the last sample was taken <span class="age-late">{ago(last)}</span>. Satellite chlorophyll within 15 km. NOAA's model is shown only for its offshore cells, {NEAREST:.1f}–15 km out, because it has no value at the harbor.</p></div></div>
<section class="sec chart-sec" aria-labelledby="r-h"><h2 id="r-h" class="sr">Twelve months of evidence</h2><div class="body wide">
<div class="fig-desk">{evidence_chart(1176)}</div><div class="fig-phone">{evidence_chart(360, phone=True)}</div>
<p class="small muted" style="margin-top:12px">Filled dots: measured. Open dots: a reported zero (not counted or not quantified, which is not the same as absent). Hatching: no sample for more than two weeks. Dashed line: NOAA's forecast days.</p></div></section>
<section class="sec" aria-labelledby="h-h"><h2 id="h-h">At the harbor</h2><div class="body">{model_block()}</div>
<div class="notes"><p>The model cannot resolve water this close to shore. Neritic never fills that gap.</p></div></section>
<section class="sec" aria-labelledby="l-h"><h2 id="l-h">Latest values</h2><div class="body">{measures_table()}</div>
<div class="notes"><p>Samples describe seawater at the pier. Only CDPH and CDFW test seafood.</p></div></section>
<section class="sec" aria-labelledby="s-h"><h2 id="s-h">Sources</h2><div class="body">
<table class="src-table"><tbody>
<tr><td class="q">Shore samples</td><td class="where"><a class="ext" href="{SRC['scw']}">CalHABMAP via SCCOOS ERDDAP</a>, Santa Cruz Wharf (HAB_SCW)</td><td class="when">retrieved Thu 15 Oct, 4:32 AM</td></tr>
<tr><td class="q">Satellite chlorophyll</td><td class="where"><a class="ext" href="{SRC['viirs']}">NOAA CoastWatch, VIIRS NOAA-21/20</a>, 750 m</td><td class="when">newest Mon 12 Oct</td></tr>
<tr><td class="q">Bloom model</td><td class="where"><a class="ext" href="{SRC['charm']}">NOAA C-HARM v3.1</a>, 3 km cells</td><td class="when">issued Wed 14 Oct</td></tr>
</tbody></table></div></section>
</main>'''
    css = "<style>.fig-phone{display:none}.chart-sec>.body.wide{grid-column:1/4}@media(max-width:1100px){.chart-sec>.body.wide{grid-column:1/3}}@media(max-width:760px){.fig-desk{display:none}.fig-phone{display:block}.sec>.body.wide{grid-column:1}}.src-table{width:100%;border-collapse:collapse;border-top:1px solid var(--rule-2)}.src-table td{padding:14px 16px 14px 0;border-bottom:1px solid var(--rule);vertical-align:baseline}.src-table td.q{width:180px}.src-table td.when{text-align:right}@media(max-width:760px){.src-table td{display:block;padding:2px 0;border:0;text-align:left!important}.src-table tr{display:block;padding:12px 0;border-bottom:1px solid var(--rule)}}</style>"
    return page("HAB evidence near Monterey", "My Coast", body, css)

# ================================================================= Station
def station():
    stations = [s for s in OBS["stations"]]
    regions = {"north_coast": "North Coast", "san_francisco": "San Francisco Bay area", "monterey_bay": "Monterey Bay", "central_coast": "Central Coast", "southern_california": "Southern California"}
    idx = []
    for rk, rn in regions.items():
        ss = [s for s in stations if s["region"] == rk]
        if not ss: continue
        items = []
        for s in ss:
            summ = {x["variable"]: x for x in s["summaries"]}
            ld = summ["pDA"]["last_date"]
            lab = f'last toxin sample {dfmt(P(ld), False)} {P(ld).year}' if ld else "toxin not measured"
            cur = ' aria-current="page"' if s["station_id"] == "HABs-SantaCruzWharf" else ""
            items.append(f'<li><a href="#"{cur}>{s["name"]}</a><span>{lab}</span></li>')
        idx.append(f'<h3>{rn}</h3><ul>{"".join(items)}</ul>')
    pn_max = SCW["summaries"]["pn_seriata"]["max_last_365d"]; pda_max = SCW["summaries"]["pDA"]["max_last_365d"]
    chl = SCW["summaries"]["chl_extracted"]; tmp = SCW["summaries"]["temp"]
    body = f'''<main class="page st" id="main">
<div class="st-grid"><aside class="st-index" aria-label="Shore stations"><p class="st-index-h">Shore stations</p>{"".join(idx)}</aside>
<div class="st-main">
<p class="kicker-line" style="margin:0 0 6px">Shore station · Monterey Bay · CalHABMAP</p>
<h1 class="page-title">Santa Cruz Wharf</h1>
<p class="answer" style="margin-top:16px">Seawater from this pier is sampled about weekly for domoic acid, Pseudo-nitzschia, chlorophyll and temperature. <strong>The last sample was taken <span class="age-late">{ago(last)}</span></strong>; newer results may not be published yet.</p>
<p class="st-notice"><span class="unverified">Not verified by a person</span> 3 official notices cover these waters. <a href="mycoast.html">Read them</a> · <a class="ext" href="{SRC['cdfw']}">CDFW</a> · <a class="ext" href="{SRC['cdph']}">CDPH</a></p>
<section class="st-chart" aria-labelledby="sc-h"><h2 id="sc-h" class="sr">Twelve months at this pier</h2>
<div class="fig-desk">{station_chart(900)}</div><div class="fig-phone">{station_chart(360, phone=True)}</div>
<p class="small muted" style="margin-top:10px">Filled dots: measured. Open dots: a reported zero, which means at or below what the lab can count, not absent. Hatching: no sample for more than two weeks.</p></section>
<section class="st-latest" aria-labelledby="sl-h"><h2 id="sl-h">The latest sample, {dfmt(last)} {last.year}</h2>
<dl class="units">
<div><dt>Domoic acid in seawater</dt><dd><span class="v">0.20<span class="u">ng/mL</span></span><span class="ctx">highest in 12 months: {pda_max:.1f}</span></dd></div>
<div><dt>Pseudo-nitzschia, larger cells</dt><dd><span class="v">4,900<span class="u">cells/L</span></span><span class="ctx">highest in 12 months: {pn_max:,.0f}</span></dd></div>
<div><dt>Chlorophyll, extracted</dt><dd><span class="v">{chl['last_value']:.2f}<span class="u">mg/m³</span></span><span class="ctx">algae of every kind</span></dd></div>
<div><dt>Water temperature</dt><dd><span class="v">{tmp['last_value']:.1f}<span class="u">°C</span></span><span class="ctx">at sampling</span></dd></div>
</dl>
<p class="small muted" style="margin-top:16px">A pier sample describes the water there that day. It is not a seafood test; CDPH tests shellfish and the agencies decide closures. Source: <a class="ext" href="{SRC['scw']}">CalHABMAP via SCCOOS ERDDAP</a>, retrieved Thu 15 Oct, 4:32 AM. 36.958° N, 122.017° W.</p></section>
</div></div></main>'''
    css = '''<style>
.st-grid{display:grid;grid-template-columns:220px minmax(0,1fr);gap:var(--s-8)}
.st-index{font-size:15px;position:sticky;top:84px;align-self:start;max-height:calc(100vh - 100px);overflow:auto}
.st-index-h{font-weight:700;margin-bottom:var(--s-3)}
.st-index h3{font-family:var(--f-text);font-size:var(--t-1);font-weight:600;color:var(--ink-3);margin:var(--s-4) 0 var(--s-1)}
.st-index ul{list-style:none;margin:0;padding:0}.st-index li{padding:6px 0;border-bottom:1px solid var(--rule)}
.st-index li a{text-decoration:none;color:var(--ink);font-weight:600;display:block}.st-index li a[aria-current]{color:var(--action)}
.st-index li a[aria-current]::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--action);margin:0 6px 2px -12px}
.st-index li span{font-size:var(--t-1);color:var(--ink-3)}
.st-notice{margin-top:var(--s-5);padding:var(--s-3) 0;border-top:1px solid var(--rule);border-bottom:1px solid var(--rule);display:flex;flex-wrap:wrap;gap:var(--s-2) var(--s-3);align-items:center;font-size:15px}
.st-chart{margin-top:var(--s-6)}
.st-latest{margin-top:var(--s-7);border-top:1px solid var(--rule);padding-top:var(--s-6)}
.st-latest h2{font-size:var(--t-3)}
.units{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:var(--s-5);margin:var(--s-5) 0 0}
.units dt{font-size:15px;color:var(--ink-2)}.units dd{margin:4px 0 0}.units .ctx{display:block;font-size:var(--t-1);color:var(--ink-3);margin-top:2px}
.fig-phone{display:none}
@media(max-width:1100px){.st-grid{grid-template-columns:minmax(0,1fr)}.st-index{display:none}.units{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:760px){.fig-desk{display:none}.fig-phone{display:block}.units{grid-template-columns:minmax(0,1fr)}.units>div{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:baseline;border-bottom:1px solid var(--rule);padding-bottom:var(--s-3)}.units dd{margin:0;text-align:right}}
</style>'''
    return page("Santa Cruz Wharf", "Blooms", body, css)

def station_chart(w, phone=False):
    # the pier's own record: toxin and cells, with the highest values named
    svg = evidence_chart(w, phone, rows=("pda", "pn"))
    return svg.replace("Santa Cruz Wharf · 41 km NNW", "this pier")

# ================================================================= Map
def mapview():
    feats = json.dumps(CELLS, separators=(",", ":"))
    stations = json.dumps([{"n": SCW["name"], "lon": SCW["lon"], "lat": SCW["lat"], "recent": True},
                           {"n": "Monterey Wharf", "lon": MW["lon"], "lat": MW["lat"], "recent": False}])
    body = f'''<main class="mapview" id="main">
<aside class="panel" aria-label="Map layers">
<div class="panel-head"><h1>Monterey Bay</h1><button class="pick" type="button" style="font-size:15px">Change place</button></div>
<p class="map-notice"><span class="unverified">Not verified by a person</span><span>3 official notices cover these waters. <a href="mycoast.html">Read them</a> · <a class="ext" href="{SRC['cdfw']}">CDFW</a> · <a class="ext" href="{SRC['cdph']}">CDPH</a></span></p>
<fieldset class="layers"><legend class="sr">What the map shows</legend>
<div class="layer open"><label><input type="radio" name="layer" checked><span class="ln">Bloom forecast<span class="src">NOAA C-HARM v3.1 · 3 km cells</span></span></label>
<div class="layer-more"><div>
<p class="var"><label for="var" class="sr">Quantity</label><select id="var"><option>Particulate domoic acid above 500 ng/L</option><option>Pseudo-nitzschia above 10,000 cells/L</option><option>Domoic acid per cell above 10 pg</option></select></p>
<div class="ramp" aria-hidden="true">{"".join(f'<i style="background:{c}"></i>' for c in RAMP)}</div>
<div class="ramp-ticks"><span>0%</span><span>50%</span><span>100%</span></div>
<p class="ramp-note">Chance, in 10% steps. Blank sea near the coast: no model value.</p>
<p class="ramp-note">Issued Wed 14 Oct. A probability for seawater, not a toxin measurement or a closure.</p>
</div></div></div>
<div class="layer"><label><input type="radio" name="layer"><span class="ln">Satellite chlorophyll<span class="src">NOAA-21/20 VIIRS and Sentinel-3 · newest Mon 12 Oct</span></span></label><div class="layer-more"><div></div></div></div>
<div class="layer"><label><input type="radio" name="layer"><span class="ln">Surface currents<span class="src">HF radar · observed today 8:00 AM · 2 km</span></span></label><div class="layer-more"><div></div></div></div>
<div class="layer"><label><input type="radio" name="layer"><span class="ln">Seafloor only<span class="src">NOAA relief and depth contours</span></span></label><div class="layer-more"><div></div></div></div>
</fieldset>
<p class="panel-foot">Cells are drawn from the published C-HARM grid (run issued Wed 14 Oct). The dashed ring is the 15 km Neritic summarises for the harbor. Click any water for its values and the nearest samples.</p>
</aside>
<div class="mapwrap"><div id="map" role="region" aria-label="Map of Monterey Bay"></div>
<div class="timebar" role="group" aria-label="Forecast day"><span class="tb-issued">Valid day</span>
<div class="seg" style="--i:2"><span class="seg-ind" aria-hidden="true"></span>
<button type="button" aria-pressed="false">Tue 13<small>nowcast</small></button><button type="button" aria-pressed="false">Wed 14<small>+1 day</small></button><button type="button" aria-pressed="true">Thu 15<small>today, +2</small></button><button type="button" aria-pressed="false">Fri 16<small>+3 days</small></button></div></div>
</div></main>'''
    head = '''<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet"><style>
.mapview{display:grid;grid-template-columns:400px minmax(0,1fr);height:calc(100vh - 61px);background:var(--sheet)}
.panel{border-right:1px solid var(--rule);overflow:auto;padding:var(--s-5) var(--s-5) var(--s-6);display:flex;flex-direction:column;gap:var(--s-5)}
.panel-head{display:flex;align-items:baseline;justify-content:space-between;gap:var(--s-3)}.panel-head h1{font-size:var(--t-4)}
.map-notice{display:grid;gap:6px;font-size:15px;padding-bottom:var(--s-4);border-bottom:1px solid var(--rule)}.map-notice .unverified{justify-self:start}
.layers{border:0;margin:0;padding:0;display:grid}
.layer{border-bottom:1px solid var(--rule)}
.layer>label{display:grid;grid-template-columns:28px minmax(0,1fr);align-items:start;padding:var(--s-3) 0;cursor:pointer;min-height:56px}
.layer input{appearance:none;width:18px;height:18px;border:1.5px solid var(--rule-2);border-radius:50%;margin:3px 0 0;display:grid;place-items:center;cursor:pointer}
.layer input:checked{border-color:var(--brand)}.layer input:checked::after{content:"";width:8px;height:8px;border-radius:50%;background:var(--brand)}
.ln{font-weight:600}.ln .src{display:block;font-weight:400;font-size:var(--t-1);color:var(--ink-3);margin-top:1px}
.layer-more{display:grid;grid-template-rows:0fr;transition:grid-template-rows var(--d-state) var(--ease-out)}
.layer-more>div{overflow:hidden;padding-left:28px}
.layer.open .layer-more{grid-template-rows:1fr}.layer.open .layer-more>div{padding-bottom:var(--s-4)}
.var select{width:100%;height:44px;font:500 15px var(--f-text);border:1px solid var(--rule-2);border-radius:var(--r);padding:0 10px;background:var(--sheet);color:var(--ink)}
.ramp{display:grid;grid-template-columns:repeat(10,1fr);height:12px;margin-top:var(--s-4);border-radius:2px;overflow:hidden}
.ramp-ticks{display:flex;justify-content:space-between;font-size:var(--t-1);color:var(--ink-3);margin-top:4px}
.ramp-note{font-size:var(--t-1);color:var(--ink-2);margin-top:var(--s-2)}
.panel-foot{font-size:var(--t-1);color:var(--ink-3);margin-top:auto}
.mapwrap{position:relative;min-width:0}#map{position:absolute;inset:0}
.timebar{position:absolute;left:50%;bottom:28px;transform:translateX(-50%);background:var(--sheet);border:1px solid var(--rule-2);border-radius:var(--r);padding:6px 6px 6px 14px;display:flex;align-items:center;gap:12px;box-shadow:0 1px 2px rgba(12,24,33,.08)}
.tb-issued{font-size:var(--t-1);color:var(--ink-3);white-space:nowrap}
.seg{position:relative;display:grid;grid-template-columns:repeat(4,92px)}
.seg button{position:relative;z-index:1;height:48px;border:0;background:none;font:600 15px var(--f-text);color:var(--ink);cursor:pointer;display:grid;align-content:center;line-height:1.15}
.seg button small{font-weight:400;font-size:12px;color:var(--ink-3)}
.seg button[aria-pressed=true]{color:#fff}.seg button[aria-pressed=true] small{color:#c9d6e2}
.seg-ind{position:absolute;top:0;bottom:0;left:0;width:92px;background:var(--brand);border-radius:4px;transform:translateX(calc(var(--i)*92px));transition:transform var(--d-state) var(--ease-move)}
.map-cap{position:absolute;left:16px;top:14px;max-width:420px;background:rgba(255,255,255,.94);border:1px solid var(--rule);border-radius:var(--r);padding:8px 12px;font-size:var(--t-1);color:var(--ink-2)}
.maplibregl-ctrl-attrib{font-size:11px}
.foot{display:none}
@media(max-width:760px){
 .mapview{display:block;height:calc(100svh - 56px);position:relative}
 .mapwrap{position:absolute;inset:0}
 .panel{position:absolute;left:0;right:0;bottom:56px;z-index:5;max-height:42%;background:var(--sheet);border-right:0;border-top:1px solid var(--rule-2);border-radius:12px 12px 0 0;padding:10px var(--s-4) var(--s-4);gap:var(--s-3);box-shadow:0 -2px 10px rgba(12,24,33,.08)}
 .panel::before{content:"";display:block;width:40px;height:4px;border-radius:2px;background:var(--rule-2);margin:0 auto 4px}
 .panel-head h1{font-size:20px}.panel-foot{display:none}
 .map-notice{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px}
 .map-notice{font-size:15px;padding-bottom:var(--s-3)}
 .maplibregl-ctrl-top-right{top:76px}
 .timebar{bottom:auto;top:10px;left:10px;right:10px;transform:none;padding:4px;gap:0}.tb-issued{display:none}
 .seg{grid-template-columns:repeat(4,1fr);width:100%}.seg-ind{width:25%;transform:translateX(calc(var(--i)*100%))}
 .map-cap{display:none}
}
</style>'''
    tail = f'''<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script><script>
const CELLS={feats};const ST={stations};const RAMP={json.dumps(RAMP)};
const phone=matchMedia('(max-width:760px)').matches;
const map=new maplibregl.Map({{container:'map',style:'https://tiles.openfreemap.org/styles/positron',center:phone?[-121.98,36.70]:[-122.0,36.78],zoom:phone?8.85:9.7,attributionControl:{{compact:true}}}});
map.addControl(new maplibregl.NavigationControl({{showCompass:false}}),'top-right');
if(phone)map.setPadding({{top:70,bottom:Math.round(innerHeight*0.40),left:0,right:0}});
function ring(lon,lat,km){{const c=[];for(let i=0;i<=96;i++){{const a=i/96*2*Math.PI;c.push([lon+km/(111.32*Math.cos(lat*Math.PI/180))*Math.cos(a),lat+km/110.57*Math.sin(a)]);}}return {{type:'Feature',geometry:{{type:'Polygon',coordinates:[c]}}}};}}
map.on('load',()=>{{
 const steps=['step',['get','p'],RAMP[0]];for(let i=1;i<10;i++)steps.push(i/10,RAMP[i]);
 map.addSource('cells',{{type:'geojson',data:CELLS}});
 const first=map.getStyle().layers.find(l=>l.type==='symbol')?.id;
 map.addLayer({{id:'cells',type:'fill',source:'cells',paint:{{'fill-color':steps,'fill-opacity':.58,'fill-outline-color':'rgba(255,255,255,.55)'}}}},first);
 map.addSource('ring',{{type:'geojson',data:ring(-121.89469,36.60024,15)}});
 map.addLayer({{id:'ring',type:'line',source:'ring',paint:{{'line-color':'#0c1821','line-width':1.2,'line-dasharray':[3,3]}}}});
 map.addSource('st',{{type:'geojson',data:{{type:'FeatureCollection',features:ST.map(s=>({{type:'Feature',properties:s,geometry:{{type:'Point',coordinates:[s.lon,s.lat]}}}}))}}}});
 map.addLayer({{id:'st',type:'circle',source:'st',paint:{{'circle-radius':6,'circle-color':['case',['get','recent'],'#0a6a60','#ffffff'],'circle-stroke-color':'#0a6a60','circle-stroke-width':2}}}});
 map.addLayer({{id:'st-l',type:'symbol',source:'st',layout:{{'text-field':['get','n'],'text-font':['Noto Sans Regular'],'text-size':13,'text-offset':['case',['get','recent'],['literal',[0.9,0]],['literal',[0,1.1]]],'text-anchor':['case',['get','recent'],'left','top']}},paint:{{'text-color':'#0c1821','text-halo-color':'#fff','text-halo-width':1.6}}}});
 new maplibregl.Marker({{element:Object.assign(document.createElement('div'),{{className:'harbor',innerHTML:'<span></span><b>Monterey harbor</b>'}})}}).setLngLat([-121.89469,36.60024]).addTo(map);
 map.addSource('rl',{{type:'geojson',data:{{type:'Feature',properties:{{t:'15 km'}},geometry:{{type:'Point',coordinates:[-121.89469,36.60024+15/110.57]}}}}}});
 map.addLayer({{id:'rl',type:'symbol',source:'rl',layout:{{'text-field':['get','t'],'text-font':['Noto Sans Regular'],'text-size':12}},paint:{{'text-color':'#34424d','text-halo-color':'#fff','text-halo-width':2}}}});
}});
document.querySelectorAll('.seg button').forEach((b,i)=>b.addEventListener('click',()=>{{document.querySelectorAll('.seg button').forEach(x=>x.setAttribute('aria-pressed',x===b));b.parentElement.style.setProperty('--i',i);}}));
document.querySelectorAll('.layer input').forEach(r=>r.addEventListener('change',()=>{{document.querySelectorAll('.layer').forEach(l=>l.classList.toggle('open',l.contains(r)));}}));
</script><style>.harbor{{display:flex;align-items:center;gap:6px;transform:translateX(calc(50% - 7px))}}.harbor span{{width:14px;height:14px;border:2.5px solid #0c1821;background:#fff;transform:rotate(45deg)}}.harbor b{{font:600 13px var(--f-text);color:#0c1821;text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff}}</style>'''
    return page("Ocean Map: Monterey Bay", "Map", body, head, tail)

# sequential, light-to-dark on a light basemap; higher chance is darker, in 10 % classes
RAMP = ["#f4f1fa", "#e6def4", "#d6c9ec", "#c5b3e2", "#b29cd6", "#9f85c9", "#8a6dbb", "#7457aa", "#5e4297", "#462e80"]

for name, fn in (("mycoast", mycoast), ("evidence", evidence), ("station", station), ("map", mapview)):
    (HERE / f"{name}.html").write_text(fn())
print("built")
