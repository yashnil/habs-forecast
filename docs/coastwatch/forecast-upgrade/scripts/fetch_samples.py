"""Fetch real subsets of candidate layers for three California test regions.

Writes NetCDF subsets to <out>/<region>/<dataset>_<date>.nc and a JSON log of what was
requested, what came back (bytes, seconds, time stamps, valid fraction) and every failure.
Nothing is smoothed or gap-filled here: these are the published values.

usage: python fetch_samples.py <out-dir> [days=14]
"""
from __future__ import annotations
import json, sys, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np
import xarray as xr

REGIONS = {
    "monterey": dict(lat=(36.40, 37.20), lon=(-122.60, -121.75)),
    "north_coast": dict(lat=(40.40, 41.40), lon=(-124.70, -123.95)),
    "socal_bight": dict(lat=(32.60, 34.10), lon=(-119.00, -117.10)),
}
CWWC = "https://coastwatch.pfeg.noaa.gov/erddap"
CWC = "https://coastwatch.noaa.gov/erddap"
# id, server, variable, lat descending?, has altitude axis, lon 0-360?
LAYERS = [
    ("olci300_s3a_CI", CWC, "noaacwS3AOLCIchlaSectorCIDaily", "chlor_a", True, True, False),
    ("olci300_s3a_DI", CWC, "noaacwS3AOLCIchlaSectorDIDaily", "chlor_a", True, True, False),
    ("viirs750_1day", CWWC, "erdVHNchla1day", "chla", True, True, False),
    ("viirs_n20_4km", CWWC, "nesdisVHNnoaa20chlaDaily", "chlor_a", True, True, False),
    ("modis_aqua_4km_nrt", CWWC, "erdMH1chla1day_R2022NRT", "chlorophyll", True, False, False),
    ("dineof_2km_sq", CWWC, "noaacwNPPN20S3ASCIDINEOF2kmDaily", "chlor_a", True, True, False),
    ("charm_nowcast", CWWC, "wvcharmV3_0day", "particulate_domoic", False, False, True),
]

def get(url, timeout=240):
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CoastWatch-research/0.1"}), timeout=timeout) as r:
        b = r.read()
    return b, time.time() - t

def times(server, ds, n):
    b, _ = get(f"{server}/griddap/{ds}.json?time[last-{n-1}:1:last]")
    return [r[0] for r in json.loads(b)["table"]["rows"]]

def main(out: Path, days: int):
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for lid, server, ds, var, lat_desc, alt, lon360 in LAYERS:
        try:
            ts = times(server, ds, days)
        except Exception as e:
            log.append(dict(layer=lid, dataset=ds, error=f"time axis: {e}"))
            continue
        for region, R in REGIONS.items():
            # sector datasets only cover one side of 120W
            if lid.endswith("_CI") and R["lon"][0] >= -120: continue
            if lid.endswith("_DI") and R["lon"][1] <= -120: continue
            lo0, lo1 = R["lon"]
            if lid.endswith("_CI"): lo1 = min(lo1, -120.0)
            if lid.endswith("_DI"): lo0 = max(lo0, -120.0)
            if lon360: lo0, lo1 = lo0 + 360, lo1 + 360
            la = f"({R['lat'][1]}):({R['lat'][0]})" if lat_desc else f"({R['lat'][0]}):({R['lat'][1]})"
            for t in ts:
                q = f"{var}[({t})]" + ("[(0.0)]" if alt else "") + f"[{la}][({lo0}):({lo1})]"
                url = f"{server}/griddap/{ds}.nc?{q}"
                rec = dict(layer=lid, dataset=ds, region=region, time=t)
                try:
                    b, s = get(url)
                    f = out / region / f"{lid}_{t[:10]}.nc"
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_bytes(b)
                    with xr.open_dataset(f) as d:
                        a = d[var].values.squeeze()
                        rec.update(bytes=len(b), seconds=round(s, 2), shape=list(a.shape), valid=int(np.isfinite(a).sum()), cells=int(a.size))
                except urllib.error.HTTPError as e:
                    rec["error"] = f"HTTP {e.code}"
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"[:200]
                log.append(rec)
                print(json.dumps(rec), flush=True)
    (out / "fetch-log.json").write_text(json.dumps(log, indent=1))

if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 14)
