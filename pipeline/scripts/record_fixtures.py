"""Record small, deterministic upstream fixtures for tests (run manually, needs network).

    uv run python scripts/record_fixtures.py

C-HARM: a Monterey Bay / Gulf of the Farallones subset for each lead of the newest run.
GIBS: capabilities trimmed to the two chlorophyll layers, one healthy tile, one tile from
the newest advertised date (often broken), and the legend SVG.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coastwatch_pipeline.http import fetch  # noqa: E402
from coastwatch_pipeline.sources import charm, gibs  # noqa: E402

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
SUBSET = {"lat_min": 36.0, "lat_max": 38.4, "lon_min": 236.9, "lon_max": 238.4}


def record_charm() -> None:
    out = FIX / "charm"
    out.mkdir(parents=True, exist_ok=True)
    times = {}
    for lead in charm.LEADS:
        t = fetch(charm.latest_time_url(lead)).body.decode()
        (out / f"lead{lead}_time.csv").write_text(t)
        vt = charm.parse_time(t.strip().splitlines()[-1])
        times[lead] = vt.isoformat()
        ts = vt.strftime("%Y-%m-%dT%H:%M:%SZ")
        sel = f"[({ts})][({SUBSET['lat_min']}):({SUBSET['lat_max']})][({SUBSET['lon_min']}):({SUBSET['lon_max']})]"
        q = ",".join(f"{v.name}{sel}" for v in charm.VARIABLES)
        url = f"{charm.SERVER}/griddap/{charm.dataset_id(lead)}.nc?" + quote(q, safe=",():")
        (out / f"lead{lead}.nc").write_bytes(fetch(url).body)
        print("lead", lead, vt, url)
    (out / "recorded.json").write_text(json.dumps({"subset": SUBSET, "valid_times": times}, indent=2) + "\n")


def record_gibs() -> None:
    out = FIX / "gibs"
    out.mkdir(parents=True, exist_ok=True)
    xml = fetch(gibs.CAPS_URL).body.decode()
    blocks = []
    for gl in gibs.LAYERS:
        m = re.search(rf"<ows:Identifier>{gl.layer_id}</ows:Identifier>", xml)
        s = xml.rfind("<Layer>", 0, m.start())
        e = xml.find("</Layer>", m.end()) + len("</Layer>")
        blocks.append(xml[s:e])
    (out / "capabilities.xml").write_text(
        '<?xml version="1.0"?><Capabilities xmlns:ows="http://www.opengis.net/ows/1.1" '
        'xmlns:xlink="http://www.w3.org/1999/xlink"><Contents>' + "".join(blocks) + "</Contents></Capabilities>\n"
    )
    caps = gibs.parse_capabilities(xml, gibs.LAYERS[0].layer_id)
    z, y, x = gibs.PROBE_TILES[0]
    from datetime import date, timedelta

    for label, d in (("newest", caps.default), ("previous", (date.fromisoformat(caps.default) - timedelta(days=1)).isoformat())):
        url = f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/{gibs.LAYERS[0].layer_id}/default/{d}/{caps.tms}/{z}/{y}/{x}.png"
        try:
            r = fetch(url, retries=0)
            (out / f"tile_{label}.png").write_bytes(r.body)
            print(label, d, r.status, gibs.png_color_type(r.body), len(r.body))
        except Exception as e:
            print(label, d, "error", e)
    leg = [u for u in caps.legend_urls if u.endswith("_H.svg")][0]
    (out / "legend_H.svg").write_bytes(fetch(leg).body)


if __name__ == "__main__":
    record_charm()
    record_gibs()
