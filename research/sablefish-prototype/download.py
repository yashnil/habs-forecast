"""Download WCGBTS haul + sablefish catch tables (2 requests total), cache raw JSON."""
import json, pathlib, datetime, requests, urllib.parse
RAW = pathlib.Path(__file__).parent / "raw"
BASE = "https://www.webapps.nwfsc.noaa.gov/trips/api/v1/source/"
PROJ = "project=Groundfish%20Slope%20and%20Shelf%20Combination%20Survey"
reqs = {
 "haul": BASE + "trawl.operation_haul_fact/selection.json?filters=" + PROJ +
   ",year>=2003,year<=2025&variables=trawl_id,project,year,pass,vessel,tow,datetime_utc_iso,sampling_start_hhmmss,"
   "latitude_dd,longitude_dd,depth_hi_prec_m,area_swept_ha_der,temperature_at_gear_c_der,temperature_at_surface_c_der,performance,station_invalid",
 "catch_sablefish": BASE + "trawl.catch_fact/selection.json?filters=" + PROJ +
   ",field_identified_taxonomy_dim$common_name=sablefish,date_dim$year>=2003,date_dim$year<=2025"
   "&variables=trawl_id,common_name,scientific_name,year,total_catch_wt_kg,total_catch_numbers,cpue_kg_per_ha_der",
}
meta = {}
for name, url in reqs.items():
    out = RAW / f"{name}.json"
    if out.exists():
        print("cached", out); continue
    t = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r = requests.get(url, timeout=300, headers={"User-Agent": "research-prototype (offline analysis)"})
    print(name, r.status_code, len(r.content))
    r.raise_for_status()
    out.write_bytes(r.content)
    meta[name] = {"url": url, "retrieved_utc": t, "status": r.status_code, "bytes": len(r.content), "rows": len(r.json())}
mp = RAW / "provenance.json"
old = json.loads(mp.read_text()) if mp.exists() else {}
old.update(meta); mp.write_text(json.dumps(old, indent=1))
print(json.dumps(old, indent=1))
