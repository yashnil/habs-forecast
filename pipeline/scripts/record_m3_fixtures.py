"""Record M3 fixtures (needs network): CalHABMAP station CSVs (all 17 stations), the
180-day C-HARM nowcast history for the two Monterey Bay stations inside the C-HARM fixture
subset, NOAA FOSS landings and the BLS CPI-U responses.

    uv run python scripts/record_m3_fixtures.py

Adds to tests/fixtures/recorded/ (keyed by exact URL; POSTs by URL#sha256(body)[:16])
without touching the M2 recordings. Other stations' C-HARM history is deliberately not
recorded so the fixture exercises the 'history unavailable' path.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.fixtures import FIXTURES, FixtureFetcher, fixture_context  # noqa: E402
from coastwatch_pipeline.http import FetchError, fetch, post_json  # noqa: E402
from coastwatch_pipeline.pipeline import run_pipeline  # noqa: E402

REC = FIXTURES / "recorded"
RECORD = [
    r"erddap\.sccoos\.org/erddap/tabledap/HABs-",
    # Santa Cruz Wharf (36.958 N) and Monterey Wharf (36.604 N): 180-day history boxes
    r"wvcharmV3_0day\.nc\?pseudo_nitzschia%5B\(2026-04-.*%5D%5B\((36\.803|36\.449)\)",
    r"apps-st\.fisheries\.noaa\.gov/ods/foss/landings/",
    r"api\.bls\.gov/publicAPI/v1/timeseries/data/",
]
index: dict[str, dict] = json.loads((REC / "index.json").read_text())


def _save(key: str, body: bytes, ctype: str) -> None:
    ext = ".html" if "text/html" in ctype else (".csv" if "csv" in ctype else (".json" if "json" in ctype else ".bin"))
    name = hashlib.sha1(key.encode()).hexdigest()[:16] + ext
    (REC / name).write_bytes(body)
    index[key] = {"file": name, "content_type": ctype}
    print("recorded", key[:140])


class Recorder(FixtureFetcher):
    def __call__(self, url):
        try:
            return super().__call__(url)
        except FetchError:
            if not any(re.search(p, url) for p in RECORD):
                raise
            r = fetch(url)
            _save(url, r.body, r.content_type)
            return r

    def post(self, url, body):
        try:
            return super().post(url, body)
        except FetchError:
            if not any(re.search(p, url) for p in RECORD):
                raise
            r = post_json(url, body)
            _save(f"{url}#{hashlib.sha256(body).hexdigest()[:16]}", r.body, r.content_type)
            return r


out = Path("/tmp/cw-record-m3")
shutil.rmtree(out, ignore_errors=True)
run_pipeline(fixture_context(out, fetcher=Recorder()))
(REC / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
print(len(index), "responses in index")
