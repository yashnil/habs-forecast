"""Record NOAA-21 / NOAA-20 VIIRS 750 m fixtures (needs network), added beside the existing ones.

NOAA ends Suomi-NPP data on 2026-11-02 (docs/coastwatch/33-suomi-npp-continuity.md); the VIIRS
layer and the port chlorophyll now come from the NOAA-21 and NOAA-20 daily merges on NOAA
CoastWatch (central) ERDDAP. This records, byte for byte:

1. satellite scenes for the three fixture regions on the same days as the S-NPP fixtures
   (tests/fixtures/satellite/<region>/viirs750_<n21|n20>_<date>.nc, indexed by the upstream
   time stamp read from each file), requested exactly as the pipeline requests them;
2. the port daily requests for Monterey and Moss Landing (tests/fixtures/recorded/, keyed by
   exact URL), by running the fixture pipeline with a recorder.

Existing fixtures (S-NPP, OLCI, C-HARM, official pages) are kept unchanged; the S-NPP ones
remain for the fallback tests. The rolling NOAA-21/20 archives keep about 8 months, so
recording must happen while 2026-10-01..04 is still served.

    uv run python scripts/record_jpss_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.context import RunContext  # noqa: E402
from coastwatch_pipeline.fixtures import FIXTURES, SATELLITE_REGIONS, FixtureFetcher, fixture_context  # noqa: E402
from coastwatch_pipeline.http import FetchError, fetch  # noqa: E402
from coastwatch_pipeline.pipeline import run_pipeline  # noqa: E402
from coastwatch_pipeline.sources import satellite as sat  # noqa: E402
from coastwatch_pipeline.sources.charm import parse_netcdf  # noqa: E402

DAYS = {
    "monterey": ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"],
    "north_coast": ["2026-10-01"],
    "socal_bight": ["2026-10-03"],
}
SHORT = {"noaacwN21VIIRSchlaSectorUSDaily": "n21", "noaacwN20VIIRSchlaSectorUSDaily": "n20"}


def record_scenes() -> None:
    root = FIXTURES / "satellite"
    idx = json.loads((root / "index.json").read_text())
    p = sat.VIIRS_JPSS
    for plat, sectors in p.platforms:
        ds = sectors[0][0]
        ctx = RunContext(out_dir=Path(tempfile.mkdtemp()), fetcher=lambda u: fetch(u, retries=4, backoff=5))
        times = sat.list_times(ctx, p.server, ds, date(2026, 9, 30))
        for region, days in DAYS.items():
            lat_s, lat_n, lon_w, lon_e = SATELLITE_REGIONS[region]
            target = sat.target_for(p, sat.Domain(lat_s, lat_n, lon_w, lon_e))
            entries = [e for e in idx.setdefault(region, {}).get(ds, []) if e["time"][:10] not in days]
            for d in days:
                t = next(x for x in times if x.startswith(d))
                url = sat.scene_url(p.server, ds, p.variable, t, target, *sectors[0][1:])
                body = fetch(url, retries=4, backoff=5).body
                arrays, _, _ = parse_netcdf(body)
                ft = datetime.fromtimestamp(float(np.atleast_1d(arrays["time"])[0]), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                assert ft == t, (ft, t)
                rel = f"{region}/viirs750_{SHORT[ds]}_{d}.nc"
                (root / rel).write_bytes(body)
                entries.append({"time": t, "file": rel})
                print("recorded", plat, region, t, len(body), "bytes")
                time.sleep(1.5)
            idx[region][ds] = sorted(entries, key=lambda e: e["time"])
    (root / "index.json").write_text(json.dumps(idx, indent=1) + "\n")


REC = FIXTURES / "recorded"
PORT_PATTERN = r"noaacwN2[01]VIIRSchlaSectorUSDaily\.nc\?chlor_a.*%5D%5B\((36\.7|36\.9)"


def record_ports() -> None:
    index = json.loads((REC / "index.json").read_text())

    class Recorder(FixtureFetcher):
        def __call__(self, url):
            try:
                return super().__call__(url)
            except FetchError:
                if not re.search(PORT_PATTERN, url):
                    raise
                r = fetch(url, retries=4, backoff=5)
                name = hashlib.sha1(url.encode()).hexdigest()[:16] + ".bin"
                (REC / name).write_bytes(r.body)
                index[url] = {"file": name, "content_type": r.content_type}
                (REC / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
                print("recorded", url[:140])
                time.sleep(1.5)
                return r

    out = Path(tempfile.mkdtemp()) / "v1"
    run_pipeline(fixture_context(out, fetcher=Recorder()), ("charm", "cdfw_ports", "official", "port_intel"))


if __name__ == "__main__":
    record_scenes()
    record_ports()
