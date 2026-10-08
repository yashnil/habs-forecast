"""Build data/reference/charm_ocean_mask.json from a published C-HARM grid (run manually).

    uv run python scripts/build_ocean_mask.py ../coastwatch-web/public/data/v1

The mask marks C-HARM v3.1 cells that carry a Pseudo-nitzschia nowcast value (ocean inside
the model domain). It is used only to draw latitude-defined official notice areas over
water near the coast. Stored as run-length encoded rows (west to east).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.models import Manifest  # noqa: E402
from coastwatch_pipeline.process import grid as codec  # noqa: E402

out_dir = Path(sys.argv[1])
m = Manifest.model_validate_json((out_dir / "manifest.json").read_text())
lyr = next(lyr for lyr in m.layers if lyr.layer_id == "charm_pseudo_nitzschia_lead0")
g = lyr.grid
vals = codec.decode((out_dir / g.url).read_bytes(), g.width, g.height, g.scale_factor, g.add_offset)
ocean = np.isfinite(vals)
rows = []
for r in range(g.height):
    runs, c = [], 0
    while c < g.width:
        if ocean[r, c]:
            s = c
            while c < g.width and ocean[r, c]:
                c += 1
            runs.append([s, c - s])
        else:
            c += 1
    rows.append(runs)
ref = Path(__file__).resolve().parents[2] / "data" / "reference" / "charm_ocean_mask.json"
ref.write_text(
    json.dumps(
        {
            "description": "C-HARM v3.1 ocean cells (cells with a Pseudo-nitzschia nowcast value). Used only to draw latitude-defined official notice areas over nearshore water.",
            "source": f"{lyr.provenance.source_url} (valid {lyr.time.valid_time})",
            "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "lat_first": g.lat_first,
            "lat_step": g.lat_step,
            "lon_first": g.lon_first,
            "lon_step": g.lon_step,
            "height": g.height,
            "width": g.width,
            "rows_rle": rows,
        },
        separators=(",", ":"),
    )
    + "\n"
)
print(ref, int(ocean.sum()), "ocean cells")
