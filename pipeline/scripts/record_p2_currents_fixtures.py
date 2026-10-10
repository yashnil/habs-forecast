"""Record HF-radar fixtures (ucsdHfrW2) from the live ERDDAP, byte-for-byte NetCDF-3.

For each fixture region (fixtures.SATELLITE_REGIONS), one request covering the fixture
window by exact grid indices, plus the full latitude/longitude axes and the time list.
The fixture fetcher serves sub-slices of these files (values untouched).

    uv run python scripts/record_p2_currents_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.fixtures import SATELLITE_REGIONS  # noqa: E402
from coastwatch_pipeline.http import fetch  # noqa: E402
from coastwatch_pipeline.sources import currents as c  # noqa: E402

T0, T1 = "2026-10-07T06:00:00Z", "2026-10-08T12:00:00Z"  # fixture now is 2026-10-08T18:00Z
dst = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "currents"
dst.mkdir(parents=True, exist_ok=True)
lat_csv = fetch(f"{c.SERVER}/griddap/{c.DATASET}.csv0?latitude").body
lon_csv = fetch(f"{c.SERVER}/griddap/{c.DATASET}.csv0?longitude").body
(dst / "latitude.csv").write_bytes(lat_csv)
(dst / "longitude.csv").write_bytes(lon_csv)
lat = np.array([float(x) for x in lat_csv.decode().split()])
lon = np.array([float(x) for x in lon_csv.decode().split()])
times = fetch(f"{c.SERVER}/griddap/{c.DATASET}.csv0?" + quote(f"time[({T0}):1:({T1})]", safe=":(),")).body
(dst / "times.csv").write_bytes(times)
index: dict = {"times": [t.strip() for t in times.decode().split()], "regions": {}}
for region, (s, n, w, e) in SATELLITE_REGIONS.items():
    ii = np.nonzero((lat >= s) & (lat <= n))[0]
    jj = np.nonzero((lon >= w) & (lon <= e))[0]
    i0, i1, j0, j1 = int(ii[0]), int(ii[-1]), int(jj[0]), int(jj[-1])
    sel = f"[({T0}):1:({T1})][{i0}:1:{i1}][{j0}:1:{j1}]"
    url = f"{c.SERVER}/griddap/{c.DATASET}.nc?" + quote(",".join(v + sel for v in c.VARS), safe=",():")
    body = fetch(url, retries=4).body
    (dst / f"{region}.nc").write_bytes(body)
    index["regions"][region] = {"file": f"{region}.nc", "i0": i0, "i1": i1, "j0": j0, "j1": j1, "url": url, "bytes": len(body)}
    print(region, i0, i1, j0, j1, len(body))
(dst / "index.json").write_text(json.dumps(index, indent=1) + "\n")
