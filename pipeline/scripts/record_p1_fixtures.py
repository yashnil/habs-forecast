"""Build satellite fixtures from ERDDAP responses recorded on 2026-10-09.

The responses were fetched from the live endpoints by
docs/coastwatch/forecast-upgrade/scripts/fetch_samples.py (byte-for-byte ERDDAP NetCDF-3).
This copies a subset into tests/fixtures/satellite/<region>/ and writes an index of
(dataset, upstream time stamp, file) read from each file's own time variable.

    uv run python scripts/record_p1_fixtures.py <samples-dir>
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.sources.charm import parse_netcdf  # noqa: E402

DATASET = {
    "olci300_s3a_CI": "noaacwS3AOLCIchlaSectorCIDaily",
    "olci300_s3a_DI": "noaacwS3AOLCIchlaSectorDIDaily",
    "viirs750_1day": "erdVHNchla1day",
}
# region -> (layer, dates)
KEEP = {
    "monterey": [("olci300_s3a_CI", [f"2026-10-0{d}" for d in range(1, 8)]), ("viirs750_1day", ["2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04"])],
    "north_coast": [("olci300_s3a_CI", ["2026-10-01", "2026-10-02"]), ("viirs750_1day", ["2026-10-01"])],
    "socal_bight": [("olci300_s3a_DI", ["2026-10-03", "2026-10-06"]), ("viirs750_1day", ["2026-10-03"])],
}

src = Path(sys.argv[1])
dst = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "satellite"
index: dict = {}
for region, items in KEEP.items():
    for layer, dates in items:
        for d in dates:
            f = src / region / f"{layer}_{d}.nc"
            arrays, _, _ = parse_netcdf(f.read_bytes())
            t = datetime.fromtimestamp(float(np.atleast_1d(arrays["time"])[0]), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            rel = f"{region}/{layer}_{d}.nc"
            (dst / region).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst / rel)
            index.setdefault(region, {}).setdefault(DATASET[layer], []).append({"time": t, "file": rel})
(dst / "index.json").write_text(json.dumps(index, indent=1) + "\n")
print(json.dumps({r: {k: len(v) for k, v in d.items()} for r, d in index.items()}))
