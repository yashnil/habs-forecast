# /// script
# dependencies = ["numpy"]
# ///
"""Nearshore valid fraction of the published OLCI 300 m daily chlorophyll grids (CoastWatch's copy
of NOAA CoastWatch's Sentinel-3 sector composites), using the coastal geography from audit.py.
A pixel missing on every day is structural (land/coast flagging); one missing on some days is
weather or overpass (cloud, fog, glint, QC flags; the grid alone cannot tell which)."""
import gzip, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
g = np.load(HERE / "out" / "geo.npz")
lat, lon, dist, reg, sfbay = g["lat"], g["lon"], g["dist"], g["reg"], g["sfbay"]
REG = ["North Coast", "Mendocino/Sonoma", "San Francisco Bay area", "Monterey Bay", "Central Coast", "Southern California"]
L0, DL, O0, DO, W, H, CH = 42.09875, -0.0025, -125.49875, 0.0025, 3400, 3880, 512
LON, LAT = np.meshgrid(lon, lat)
sel = (dist >= 0) & (dist < 10) & (reg >= 0)
r = np.floor((LAT[sel] - L0) / DL + 0.5).astype(int)
c = np.floor((LON[sel] - O0) / DO + 0.5).astype(int)
inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
days = sorted(p.name for p in (HERE / "cur" / "olci").iterdir())
valid = np.zeros((len(days), sel.sum()), dtype=bool)
for k, d in enumerate(days):
    full = np.full((H, W), False)
    for f in (HERE / "cur" / "olci" / d).glob("*.u16.gz"):
        rr, cc = map(int, f.name.split(".")[0].split("_"))
        a = np.frombuffer(gzip.decompress(f.read_bytes()), dtype="<u2")
        h = min(CH, H - rr * CH); w = min(CH, W - cc * CH)
        a = a.reshape(h, w) if a.size == h * w else a.reshape(CH, CH)[:h, :w]
        full[rr * CH: rr * CH + h, cc * CH: cc * CH + w] = a != 65535
    v = np.zeros(sel.sum(), dtype=bool)
    v[inside] = full[r[inside], c[inside]]
    valid[k] = v
D = dist[sel]; R = reg[sel]; SF = sfbay[sel]
out = {"days": days, "rows": []}
for ri, name in enumerate(REG + ["All California"]):
    m = (R == ri) & ~SF if ri < len(REG) else np.ones_like(R, dtype=bool)
    for lo, hi in [(0, 1), (1, 3), (3, 10), (0, 10)]:
        s = m & (D >= lo) & (D < hi)
        if not s.any():
            continue
        per_day = [round(float(valid[k][s].mean() * 100), 1) for k in range(len(days))]
        never = round(float((~valid[:, s].any(axis=0)).mean() * 100), 1)
        out["rows"].append({"region": name, "band_km": f"{lo}-{hi}", "valid_%_by_day": per_day, "median_day_%": float(np.median(per_day)), "never_valid_in_6_days_%": never})
(HERE / "out" / "olci_audit.json").write_text(json.dumps(out, indent=1))
for row in out["rows"]:
    print(row["region"][:22].ljust(22), row["band_km"].ljust(5), "median", row["median_day_%"], "never", row["never_valid_in_6_days_%"], row["valid_%_by_day"])
