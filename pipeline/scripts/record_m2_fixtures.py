"""Record M2 fixtures (needs network): official pages, CDPH geometry, and ERDDAP port history
for the Monterey Bay ports inside the C-HARM fixture subset.

    uv run python scripts/record_m2_fixtures.py

Runs the fixture pipeline with a fetcher that serves existing fixtures and records real
responses for allow-listed URLs into tests/fixtures/recorded/ (keyed by exact URL).
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
from coastwatch_pipeline.http import FetchError, fetch  # noqa: E402
from coastwatch_pipeline.pipeline import run_pipeline  # noqa: E402
from coastwatch_pipeline.sources.official import BROWSER_UA  # noqa: E402

REC = FIXTURES / "recorded"
# Monterey (36.60 N) and Moss Landing (36.80 N): ERDDAP history boxes start near these latitudes
RECORD = [
    r"services2\.arcgis\.com/wi1yEacfYjH5viqb",
    r"cdph\.ca\.gov/Programs/OPA/Pages/Shellfish-Advisories",
    r"wildlife\.ca\.gov/(Fishing/Ocean/Health-Advisories|Conservation/Marine/Whale-Safe-Fisheries)",
    r"wvcharmV3_0day\.nc\?pseudo_nitzschia%5B\(2026-09-08.*%5D%5B\((36\.4|36\.6)",
    r"erdVHNchla8day\.nc\?chla.*%5D%5B\((36\.7|36\.9)",
]
index: dict[str, dict] = {}


class Recorder(FixtureFetcher):
    def __call__(self, url):
        try:
            return super().__call__(url)
        except FetchError:
            if not any(re.search(p, url) for p in RECORD):
                raise
            headers = {"User-Agent": BROWSER_UA} if ("cdph.ca.gov" in url or "wildlife.ca.gov" in url) else None
            r = fetch(url, headers=headers)
            name = hashlib.sha1(url.encode()).hexdigest()[:16] + (".html" if "text/html" in r.content_type else ".bin")
            (REC / name).write_bytes(r.body)
            index[url] = {"file": name, "content_type": r.content_type}
            print("recorded", url[:120])
            return r


if REC.exists():
    shutil.rmtree(REC)
REC.mkdir(parents=True)
(REC / "index.json").write_text("{}")
out = Path("/tmp/cw-record")
shutil.rmtree(out, ignore_errors=True)
run_pipeline(fixture_context(out, fetcher=Recorder()))
(REC / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
print(len(index), "responses recorded")
