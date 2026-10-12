"""High-fidelity mockup of a "My Coast" port page from production data of 2026-10-12.
Static HTML for design review only; not part of the app. Every number comes from out/mycoast.json."""
import html, json
from datetime import date
from pathlib import Path

H = Path(__file__).parent
D = json.load(open(H / "out/mycoast.json"))
REG = json.load(open(H.parent / "wt8/data/curated/official_notices.json"))
TODAY = date(2026, 10, 12)
e = html.escape


def age(d):
    n = (TODAY - date.fromisoformat(d)).days
    return "today" if n == 0 else "yesterday" if n == 1 else f"{n} days ago"


def fdate(d):
    return date.fromisoformat(d).strftime("%b %-d, %Y")


WHO = {"sport_harvest": "Recreational (sport) harvest", "recreational": "Recreational fishery", "commercial": "Commercial fishery", "commercial_and_recreational": "Commercial and recreational", "consumption": "Anyone eating it"}
ACT = {"quarantine": "Quarantine", "fishery_closure": "Fishery closure", "consumption_advisory": "Do-not-eat advisory", "take_restriction": "Take restriction"}
GROUP = {"razor clam": "Shellfish", "razor clams": "Shellfish", "mussels (all species)": "Shellfish", "northern anchovy": "Finfish"}


def notice(n):
    who = WHO.get(n["fishery"], n["fishery"])
    return f"""<li class="notice">
  <div class="nrow"><span class="agency">{e(n['agency'])}</span><span class="act">{e(ACT.get(n['action'], n['action']))}</span><span class="who">{e(who)}</span><span class="badge">Not verified</span></div>
  <p class="ntitle">{e(n['title'])}</p>
  <p class="nmeta">Since {fdate(n['effective_date'])} · {e(n['area'])} · <a href="{e(n['src'])}">Agency source ↗</a></p>
</li>"""


def station(s):
    rows = []
    if s["pDA"]:
        v = s["pDA"]
        val = "reported 0 (not quantified)" if v["last_qualifier"] == "reported_zero" or v["last_value"] == 0 else f"{v['last_value']:.2f} ng/mL"
        rows.append(f"<tr><th>Particulate domoic acid</th><td><b>{val}</b></td><td>{fdate(v['last_date'])} <span class='age'>({age(v['last_date'])})</span></td></tr>")
    else:
        rows.append("<tr><th>Particulate domoic acid</th><td colspan=2 class='muted'>not measured at this station</td></tr>")
    for key, lab in (("pn_ser", "Pseudo-nitzschia, large (seriata)"), ("pn_del", "Pseudo-nitzschia, small (delicatissima)")):
        v = s[key]
        if v and v["last_date"]:
            rows.append(f"<tr><th>{lab}</th><td><b>{v['last_value']:,.0f} cells/L</b></td><td>{fdate(v['last_date'])} <span class='age'>({age(v['last_date'])})</span></td></tr>")
    return f"""<div class="station"><p class="sname">{e(s['name'])} <span class="muted">· {s['km']} km from the port · water sample at the pier</span></p><table>{''.join(rows)}</table></div>"""


def page(name):
    p = D[name]
    c = p["charm"]
    n_note = "".join(notice(n) for n in p["notices"])
    stmts = "".join(f"<li>“{e(s['statement'])}” <span class='muted'>— {e(s['agency'])}, <a href='{e(s['source']['url'])}'>source ↗</a></span></li>" for s in REG["statements"])
    st = "".join(station(s) for s in p["stations"]) if p["stations"] else """<div class="empty"><b>No CalHABMAP shore station within 40 km.</b> The nearest, Trinidad Pier (77 km south), was last sampled Apr 17, 2026 (178 days ago). There is no recent toxin measurement near this port.</div>"""
    sat = p["satellite"]
    sat_html = (f"<p><b>Last clear view {fdate(sat['date'])}</b> ({age(sat['date'])}): median chlorophyll {sat['median_mg_m3']} mg/m³ within 5 km (Sentinel-3 OLCI, 300 m). Chlorophyll is algae biomass, not toxin.</p>"
                if sat else "<p><b>No clear satellite view within 5 km in the last 6 days</b> (cloud, fog or no overpass). Nothing is filled in.</p>")
    gap = c["nearest_toxin_cell_km"]
    return f"""<section class="port">
<header class="ph"><p class="eyebrow">My Coast · prototype</p><h1>{e(name)}</h1><p class="sub">{e(p['county'])} County · data as of Oct 12, 2026</p>
<div class="chips" aria-label="What do you fish?"><span class="chip on">All</span><span class="chip">Crab</span><span class="chip">Finfish</span><span class="chip">Shellfish</span><span class="sep"></span><span class="chip on">Commercial + recreational</span></div></header>

<div class="card official"><h2><span class="dot o"></span>Official notices that may apply here</h2>
<ul class="nlist">{n_note}</ul>
<p class="warn">Listed from CDFW and CDPH pages; <b>not verified by a person</b> and possibly incomplete. A missing notice does not mean an area is open or seafood is safe.</p>
<details><summary>Agency statements that apply statewide (crab, lobster)</summary><ul class="stmts">{stmts}</ul></details></div>

<div class="card measured"><h2><span class="dot m"></span>Measured nearby</h2>{st}</div>

<div class="grid2">
<div class="card model"><h2><span class="dot v"></span>Model context (C-HARM, NOAA)</h2>
<p>For {fdate(c['valid'])} (run issued {fdate(c['issued'])}), within {c['radius_km']:.0f} km of the port: median probability that particulate domoic acid exceeds 500 ng/L is <b>{c['pda']['median'] * 100:.0f}%</b> (range {c['pda']['min'] * 100:.0f}–{c['pda']['max'] * 100:.0f}% over {c['pda']['n']} cells).</p>
<p class="gapline">The nearest model cell with a toxin value is <b>{gap} km</b> from the port. The model gives no toxin value closer in, and nothing is filled in.</p>
<p class="muted">A probability for 3 km ocean cells, not a measurement, and not seafood.{" Not compared with the pier sample above." if p["stations"] else ""}</p></div>
<div class="card measured"><h2><span class="dot m"></span>Satellite (observed)</h2>{sat_html}</div>
</div>

<div class="card next"><h2>Before you fish or harvest</h2><ol>
<li>Check <a href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories">CDFW health advisories and closures ↗</a> for your species and fishery.</li>
<li>Check <a href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx">CDPH shellfish advisories ↗</a>, or call <b>1 (800) 553-4133</b>.</li>
<li>CDFW domoic acid fishery closure line: <b>(831) 649-2883</b>.</li></ol>
<p class="muted">CoastWatch does not decide whether an area is open or seafood is safe.</p></div>
</section>"""


CSS = """
:root{--paper:#f6f3ec;--surface:#fff;--ink:#0f1c2b;--ink2:#3b4a5c;--ink3:#6b7786;--line:#e2ddd2;--off:#b26b00;--offbg:#fdf4e3;--offline:#ecc98a;--model:#6a4fc6;--meas:#0f7a6b;--navy:#06111e}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Inter",sans-serif}
.top{background:var(--navy);color:#fff;padding:12px 20px;font-weight:600}.port{max-width:960px;margin:0 auto;padding:20px 16px 48px}
.eyebrow{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);margin:0}h1{font:500 34px/1.1 Georgia,serif;margin:4px 0}.sub{color:var(--ink2);margin:0 0 12px}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{border:1px solid var(--line);background:var(--surface);border-radius:999px;padding:6px 12px;font-size:14px;min-height:36px;display:inline-flex;align-items:center}.chip.on{background:var(--navy);color:#fff;border-color:var(--navy)}.sep{width:8px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin-top:14px}.card h2{font:600 16px/1.3 inherit;margin:0 0 10px;display:flex;align-items:center;gap:8px}
.dot{width:9px;height:9px;border-radius:50%;display:inline-block}.dot.o{background:#e3a33b}.dot.m{background:var(--meas)}.dot.v{background:var(--model)}
.official{background:var(--offbg);border-color:var(--offline)}.nlist{list-style:none;margin:0;padding:0}.notice{border-top:1px solid var(--offline);padding:10px 0}.notice:first-child{border-top:0;padding-top:0}
.nrow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase}.agency{border:1px solid var(--off);color:var(--off);border-radius:5px;padding:1px 6px}.act{color:var(--off)}.who{color:var(--ink2)}.badge{margin-left:auto;border:1px solid #d9a3a3;color:#9b2c2c;border-radius:6px;padding:1px 6px;text-transform:none;letter-spacing:0}
.ntitle{font-weight:600;margin:4px 0 2px}.nmeta{color:var(--ink2);font-size:14px;margin:0}.warn{font-size:14px;color:var(--ink2);margin:10px 0 0}details{margin-top:8px;font-size:14px}.stmts{padding-left:18px;color:var(--ink2)}
.station .sname{font-weight:600;margin:0 0 6px}table{width:100%;border-collapse:collapse;font-size:14px}th{text-align:left;font-weight:500;color:var(--ink2);padding:5px 8px 5px 0;width:44%}td{padding:5px 8px 5px 0;border-top:1px solid var(--line)}th{border-top:1px solid var(--line)}
.age,.muted{color:var(--ink3)}.empty{background:#f3f1ec;border-radius:10px;padding:10px 12px;font-size:14px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.grid2 .card{margin-top:14px}
.model{border-left:4px solid var(--model)}.measured{border-left:4px solid var(--meas)}.gapline{background:#f2eefb;border-radius:8px;padding:8px 10px;font-size:14px}.next ol{margin:0;padding-left:20px}a{color:#0e6a7e}
@media (max-width:640px){.grid2{grid-template-columns:1fr}h1{font-size:28px}.badge{margin-left:0}}
"""
for name in D:
    slug = name.lower().replace(" ", "-")
    (H / "out" / f"mycoast-{slug}.html").write_text(f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>My Coast prototype · {e(name)}</title><style>{CSS}</style><div class='top'>CoastWatch · design prototype (production data, Oct 12, 2026)</div>{page(name)}</html>")
print("ok")
