# /// script
# dependencies = ["numpy"]
# ///
import gzip, json, math
from datetime import date
from pathlib import Path
import numpy as np
H = Path(__file__).parent
TODAY = date(2026, 10, 12)
ports = json.load(open(H / "cur/port-intel.json"))["ports"]
obs = json.load(open(H.parent / "review/obs.json"))
off = json.load(open(H.parent / "wt8/data/curated/official_notices.json"))
pda = np.frombuffer(gzip.decompress((H / "cur/lead0_particulate_domoic.u16.gz").read_bytes()), "<u2").reshape(391, 351) != 65535
pn = np.frombuffer(gzip.decompress((H / "cur/lead0_pseudo_nitzschia.u16.gz").read_bytes()), "<u2").reshape(391, 351) != 65535
I, J = np.nonzero(pda); clat = 31.3 + 0.03 * I; clon = -127.5 + 0.03 * J
I2, J2 = np.nonzero(pn); blat = 31.3 + 0.03 * I2; blon = -127.5 + 0.03 * J2
def km(a, b, c, d):
    p = math.radians
    x = (np.radians(d) - p(b)) * np.cos(np.radians((a + c) / 2)); y = np.radians(c) - p(a)
    return 6371 * np.sqrt(x * x + y * y)
def last_of(s, vars_):
    se = {x["variable"]: x["values"] for x in s["series"]}
    for i in range(len(s["sample_times"]) - 1, -1, -1):
        if any(se.get(v, [None] * len(s["sample_times"]))[i] is not None for v in vars_):
            return s["sample_times"][i][:10]
    return None
st = [(s["name"], s["lat"], s["lon"], last_of(s, ["pDA", "tDA", "dDA"])) for s in obs["stations"] if s["lat"] is not None]
stp = [(s["name"], s["lat"], s["lon"], last_of(s, ["pn_seriata", "pn_delicatissima"])) for s in obs["stations"] if s["lat"] is not None]
sta = [(s["name"], s["lat"], s["lon"], s["sample_times"][-1][:10] if s["sample_times"] else None) for s in obs["stations"] if s["lat"] is not None]
def recent(la, lo, lst, days, rad=25):
    return [s[0] for s in lst if s[3] and float(km(la, lo, s[1], s[2])) <= rad and (TODAY - date.fromisoformat(s[3])).days <= days]
rows = []
for p in ports:
    la, lo = p["lat"], p["lon"]
    d = sorted(((float(km(la, lo, s[1], s[2])), s) for s in st), key=lambda x: x[0])
    near = d[0]
    age = (TODAY - date.fromisoformat(near[1][3])).days if near[1][3] else None
    within = [(round(x, 1), s[0], (TODAY - date.fromisoformat(s[3])).days) for x, s in d if x <= 25 and s[3]]
    rec14 = [w for w in within if w[2] <= 14]; rec30 = [w for w in within if w[2] <= 30]
    county = p["county"]
    notices = [r["id"] for r in off["records"] if r["area"].get("type") == "statewide" or county.lower() in (r["area"].get("description", "") + " " + r["title"]).lower()]
    rows.append({"port": p["display_name"], "county": county, "region": p["region"], "nearest_station": near[1][0], "station_km": round(near[0], 1), "last_toxin_sample": near[1][3], "age_days": age,
                 "stations_25km_sampled_14d": len(rec14), "stations_25km_sampled_30d": len(rec30),
                 "toxin_25km_30d": recent(la, lo, st, 30), "pn_25km_30d": recent(la, lo, stp, 30), "any_25km_30d": recent(la, lo, sta, 30), "toxin_25km_90d": recent(la, lo, st, 90),
                 "nearest_toxin_cell_km": round(float(km(la, lo, clat, clon).min()), 1), "nearest_bloom_cell_km": round(float(km(la, lo, blat, blon).min()), 1),
                 "charm_nearest_cell_km_portintel": p["charm"]["nearest_cell_km"], "county_or_statewide_notices": notices})
(H / "out/ports_audit.json").write_text(json.dumps(rows, indent=1))
for r in rows:
    print(f"{r['port'][:16]:16} {r['county'][:11]:11} st {r['nearest_station'][:22]:22} {r['station_km']:6} km last {r['last_toxin_sample']} ({r['age_days']}d) n14 {r['stations_25km_sampled_14d']} n30 {r['stations_25km_sampled_30d']} tox {r['nearest_toxin_cell_km']} bloom {r['nearest_bloom_cell_km']} notices {len(r['county_or_statewide_notices'])}")
n = len(rows)
print("toxin<=30d", sum(bool(r["toxin_25km_30d"]) for r in rows), "toxin<=90d", sum(bool(r["toxin_25km_90d"]) for r in rows), "PN<=30d", sum(bool(r["pn_25km_30d"]) for r in rows), "any sample<=30d", sum(bool(r["any_25km_30d"]) for r in rows))
print("ports", n, "with station<=25km sampled<=14d:", sum(r["stations_25km_sampled_14d"] > 0 for r in rows), "<=30d:", sum(r["stations_25km_sampled_30d"] > 0 for r in rows), "toxin cell <=3km:", sum(r["nearest_toxin_cell_km"] <= 3 for r in rows), "<=5km:", sum(r["nearest_toxin_cell_km"] <= 5 for r in rows))
