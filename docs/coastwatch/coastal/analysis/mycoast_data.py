# /// script
# dependencies = ["numpy"]
# ///
import gzip, json, math
from datetime import date
from pathlib import Path
import numpy as np
H = Path(__file__).parent
TODAY = date(2026, 10, 12)
pi = {p["display_name"]: p for p in json.load(open(H / "cur/port-intel.json"))["ports"]}
pa = {p["port"]: p for p in json.load(open(H / "out/ports_audit.json"))}
obs = json.load(open(H.parent / "review/obs.json"))
off = json.load(open(H.parent / "wt8/data/curated/official_notices.json"))
def km(a, b, c, d):
    x = math.radians(d - b) * math.cos(math.radians((a + c) / 2)); y = math.radians(c - a); return 6371 * math.hypot(x, y)
# OLCI: last valid day within 5 km of the port and its median chl
L0, DL, O0, DO, W, Hh, CH = 42.09875, -0.0025, -125.49875, 0.0025, 3400, 3880, 512
days = sorted(p.name for p in (H / "cur/olci").iterdir())
grids = {}
for d in days:
    full = np.full((Hh, W), np.nan, dtype=np.float32)
    for f in (H / "cur/olci" / d).glob("*.u16.gz"):
        rr, cc = map(int, f.name.split(".")[0].split("_"))
        a = np.frombuffer(gzip.decompress(f.read_bytes()), dtype="<u2").astype(np.float32)
        h = min(CH, Hh - rr * CH); w = min(CH, W - cc * CH)
        a = a.reshape(h, w) if a.size == h * w else a.reshape(CH, CH)[:h, :w]
        v = np.where(a == 65535, np.nan, 10 ** (a * 9.155552842799158e-05 - 3.0))
        full[rr * CH: rr * CH + h, cc * CH: cc * CH + w] = v
    grids[d] = full
def sat(lat, lon, rad=5):
    r0 = int((lat - L0) / DL); c0 = int((lon - O0) / DO); n = int(rad / 0.28) + 1
    for d in reversed(days):
        win = grids[d][max(0, r0 - n): r0 + n, max(0, c0 - n): c0 + n]
        v = win[np.isfinite(win)]
        if v.size >= 5:
            return {"date": d, "median_mg_m3": round(float(np.median(v)), 2), "pixels": int(v.size)}
    return None
out = {}
for name in ["Crescent City", "Santa Cruz", "Morro Bay"]:
    p = pi[name]; la, lo = p["lat"], p["lon"]
    st = []
    for s in obs["stations"]:
        if s["lat"] is None: continue
        dkm = km(la, lo, s["lat"], s["lon"])
        if dkm > 40: continue
        sm = {x["variable"]: x for x in s["summaries"]}
        st.append({"name": s["name"], "km": round(dkm, 1), "last_sample": s["sample_times"][-1][:10],
                   "pDA": {k: sm["pDA"][k] for k in ("last_value", "last_date", "last_qualifier", "max_last_365d", "n_last_365d")} if sm.get("pDA") and sm["pDA"]["n_measured"] else None,
                   "pn_ser": {k: sm["pn_seriata"][k] for k in ("last_value", "last_date")} if sm.get("pn_seriata") and sm["pn_seriata"]["n_measured"] else None,
                   "pn_del": {k: sm["pn_delicatissima"][k] for k in ("last_value", "last_date")} if sm.get("pn_delicatissima") and sm["pn_delicatissima"]["n_measured"] else None})
    st.sort(key=lambda x: x["km"])
    county = p["county"]
    notes = [{k: r[k] for k in ("id", "agency", "action", "fishery", "title", "effective_date", "species")} | {"src": r["sources"][0]["url"], "area": r["area"]["description"]}
             for r in off["records"] if r["area"].get("type") == "statewide" or county.lower() in (r["area"]["description"] + " " + r["title"]).lower()
             or (name == "Santa Cruz" and "Pigeon Point" in r["area"]["description"])]
    lead1 = next(l for l in p["charm"]["leads"] if l["lead_days"] == 1)
    out[name] = {"county": county, "lat": la, "lon": lo, "notices": notes, "stations": st[:3], "charm": {"issued": p["charm"]["issued_date"], "valid": lead1["valid_date"], "radius_km": p["charm"]["radius_km"], "pda": lead1["variables"]["particulate_domoic"], "pn": lead1["variables"]["pseudo_nitzschia"], "nearest_toxin_cell_km": pa[name]["nearest_toxin_cell_km"], "nearest_bloom_cell_km": pa[name]["nearest_bloom_cell_km"]}, "satellite": sat(la, lo)}
(H / "out/mycoast.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, v["county"], "notices", [n["id"] for n in v["notices"]]); print("  stations", [(s["name"], s["km"], s["pDA"] and (s["pDA"]["last_value"], s["pDA"]["last_date"])) for s in v["stations"]]); print("  charm", v["charm"]["nearest_toxin_cell_km"], v["charm"]["pda"]); print("  sat", v["satellite"])
