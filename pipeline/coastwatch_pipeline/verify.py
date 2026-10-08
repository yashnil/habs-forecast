"""Verify published C-HARM artifacts against the source at representative points.

For each point and each published layer:
1. grid   - the published value grid decoded at the point's cell
2. image  - the reprojected PNG pixel at the point has the palette colour of that value
            (or is transparent when there is no value)
3. source - (live) ERDDAP's own value for the same cell agrees within quantization error
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from .http import fetch
from .models import LayerArtifact, Manifest
from .process import grid as gridcodec
from .process.mercator import MercatorImage
from .process.palette import palette_color
from .sources import charm

POINTS: list[tuple[str, float, float]] = [
    ("Off Crescent City", 41.70, -124.35),
    ("Off Eureka", 40.85, -124.30),
    ("Off Fort Bragg", 39.45, -123.90),
    ("Off Bodega Bay", 38.25, -123.15),
    ("Gulf of the Farallones", 37.70, -122.80),
    ("Off Half Moon Bay", 37.45, -122.60),
    ("Off Santa Cruz", 36.90, -122.10),
    ("Monterey Bay (mid-bay)", 36.80, -121.95),
    ("Monterey Wharf (nearshore)", 36.604, -121.889),
    ("Off Morro Bay", 35.35, -121.00),
    ("Santa Barbara Channel", 34.30, -119.80),
    ("San Pedro Channel", 33.60, -118.30),
    ("Off San Diego", 32.70, -117.35),
]


def _cell(layer: LayerArtifact, lat: float, lon: float) -> tuple[int, int, float, float] | None:
    g = layer.grid
    assert g is not None
    r = int(math.floor((lat - g.lat_first) / g.lat_step + 0.5))
    c = int(math.floor((lon - g.lon_first) / g.lon_step + 0.5))
    if not (0 <= r < g.height and 0 <= c < g.width):
        return None
    return r, c, g.lat_first + r * g.lat_step, g.lon_first + c * g.lon_step


def _erddap_point(dataset: str, valid_time: str, lat: float, lon: float) -> dict[str, float]:
    lon360 = lon + 360 if lon < 0 else lon
    sel = f"[({valid_time})][({lat:.4f})][({lon360:.4f})]"
    q = ",".join(f"{v.name}{sel}" for v in charm.VARIABLES)
    url = f"{charm.SERVER}/griddap/{dataset}.csv0?" + quote(q, safe=",():")
    line = fetch(url, retries=4).body.decode().strip().splitlines()[-1].split(",")
    # csv0 columns: time, latitude, longitude, var1, var2, var3
    out = {}
    for i, v in enumerate(charm.VARIABLES):
        s = line[3 + i]
        out[v.name] = float("nan") if s in ("NaN", "") else float(s)
    out["_lat"], out["_lon"] = float(line[1]), float(line[2])
    return out


def verify_charm(out_dir: Path, live: bool = True) -> dict:
    m = Manifest.model_validate_json((out_dir / "manifest.json").read_text())
    layers = [lyr for lyr in m.layers if lyr.group_id == charm.GROUP_ID]
    rows: list[dict] = []
    cache: dict[tuple, dict[str, float]] = {}
    images: dict[str, np.ndarray] = {}
    grids: dict[str, np.ndarray] = {}
    for lyr in layers:
        g, im = lyr.grid, lyr.image
        assert g and im
        grids[lyr.layer_id] = gridcodec.decode((out_dir / g.url).read_bytes(), g.width, g.height, g.scale_factor, g.add_offset, g.nodata)
        images[lyr.layer_id] = np.asarray(Image.open(out_dir / im.url).convert("RGBA"))
    for name, lat, lon in POINTS:
        for lyr in layers:
            cell = _cell(lyr, lat, lon)
            row = {"point": name, "lat": lat, "lon": lon, "layer_id": lyr.layer_id, "valid_date": lyr.time.valid_date}
            if cell is None:
                row.update(status="outside_grid")
                rows.append(row)
                continue
            r, c, clat, clon = cell
            val = float(grids[lyr.layer_id][r, c])
            row.update(cell_lat=round(clat, 4), cell_lon=round(clon, 4), grid_value=None if math.isnan(val) else round(val, 6))
            # image pixel at the cell centre
            im = lyr.image
            mi = MercatorImage(im.width, im.height, im.bounds_lnglat[0], im.bounds_lnglat[2], im.bounds_lnglat[1], im.bounds_lnglat[3])
            pr, pc = mi.pixel_of(clat, clon)
            px = images[lyr.layer_id][pr, pc]
            if math.isnan(val):
                image_ok = int(px[3]) == 0
            else:
                image_ok = int(px[3]) == 255 and tuple(int(x) for x in px[:3]) == palette_color(val, lyr.palette)
            row["image_matches_value"] = bool(image_ok)
            source_ok = None
            if live:
                key = (lyr.provenance.dataset_id, lyr.time.valid_time, round(clat, 4), round(clon, 4))
                if key not in cache:
                    cache[key] = _erddap_point(lyr.provenance.dataset_id, lyr.time.valid_time, clat, clon)
                    time.sleep(0.3)
                src = cache[key]
                sv = src[lyr.variable]
                row["source_value"] = None if math.isnan(sv) else round(sv, 6)
                row["source_cell"] = [src["_lat"], src["_lon"]]
                tol = lyr.grid.max_quantization_error + 1e-6
                source_ok = (math.isnan(sv) and math.isnan(val)) or (not math.isnan(sv) and not math.isnan(val) and abs(sv - val) <= tol)
                row["source_matches_grid"] = bool(source_ok)
            row["status"] = "pass" if image_ok and source_ok is not False else "FAIL"
            rows.append(row)
    checked = [r for r in rows if r["status"] != "outside_grid"]
    failed = [r for r in checked if r["status"] == "FAIL"]
    return {
        "manifest_generated_at": m.generated_at,
        "live_source_comparison": live,
        "summary": {
            "layers": len(layers),
            "points": len(POINTS),
            "comparisons": len(checked),
            "outside_grid": len(rows) - len(checked),
            "no_value_cells": sum(1 for r in checked if r.get("grid_value") is None),
            "failures": len(failed),
            "all_passed": not failed and bool(checked),
        },
        "rows": rows,
    }
