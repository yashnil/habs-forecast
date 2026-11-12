"""Extend the recorded CalHABMAP station CSVs back to observations.WINDOW_START (needs network).

    uv run python scripts/record_obs_history_fixtures.py

The station responses in tests/fixtures/recorded/ were recorded on 2026-10-08 for
`time>=2014-01-01`. The pipeline now asks for the record from WINDOW_START. Re-recording
the whole response today would also change the 2014-2026 rows (late-published laboratory
values, samples after the fixture clock), so this script records ONLY the earlier rows
(`WINDOW_START <= time < 2014-01-01`, same columns, same order) from SCCOOS ERDDAP and
writes them in front of the rows already recorded. Each file is therefore two real
responses of the same request, joined at 2014-01-01: nothing is edited or invented.
The entry keyed by the old request URL is removed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.fixtures import FIXTURES  # noqa: E402
from coastwatch_pipeline.http import FetchError, fetch  # noqa: E402
from coastwatch_pipeline.sources import observations as obs  # noqa: E402

OLD_START = "2014-01-01"
UA = {"User-Agent": "fixture-recorder/1.0"}  # a generic client name: recording is a one-off, not the pipeline
REC = FIXTURES / "recorded"
index: dict[str, dict] = json.loads((REC / "index.json").read_text())


def earlier_url(key: str) -> str:
    q = ",".join(obs.COLUMNS) + f"&time>={obs.WINDOW_START}T00:00:00Z&time<{OLD_START}T00:00:00Z" + '&orderBy("time")'
    return f"{obs.SERVER}/tabledap/{obs.dataset_id(key)}.csv?" + quote(q, safe=',&=()"').replace(">=", "%3E=").replace("<", "%3C")


def earlier_rows(key: str) -> list[bytes]:
    for attempt in range(4):
        try:
            body = fetch(earlier_url(key), headers=UA, retries=0).body
            return body.splitlines(keepends=True)[2:]
        except FetchError as e:
            # ERDDAP answers 404 "Your query produced no matching results" for a station with no earlier rows
            if "no matching results" in str(e) or "404" in str(e):
                probe = fetch(f"{obs.SERVER}/tabledap/{obs.dataset_id(key)}.csv?time&orderByMin(%22time%22)", headers=UA).body.splitlines()
                if probe[-1].decode()[:10] >= OLD_START:
                    return []
            if attempt == 3:
                raise
            time.sleep(20)
    return []


for key in obs.STATIONS:
    old = obs.request_url(key, OLD_START)
    new = obs.request_url(key)
    if new in index:
        print("already recorded", key)
        continue
    entry = index[old]
    lines = (REC / entry["file"]).read_bytes().splitlines(keepends=True)
    head, rows = lines[:2], lines[2:]
    before = earlier_rows(key)
    name = hashlib.sha1(new.encode()).hexdigest()[:16] + ".csv"
    (REC / name).write_bytes(b"".join(head + before + rows))
    (REC / entry["file"]).unlink()
    del index[old]
    index[new] = {"file": name, "content_type": entry["content_type"]}
    print(f"{key}: {len(before)} earlier row(s) + {len(rows)} recorded")

(REC / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
