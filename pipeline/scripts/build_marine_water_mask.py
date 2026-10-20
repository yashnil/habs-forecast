"""Build data/reference/marine_water_mask.json: connected marine bays and estuaries on the OLCI lattice.

    uv run python scripts/build_marine_water_mask.py

Why: the satellite chlorophyll scope used to be the C-HARM ocean mask (3 km) dilated 4 cells
(about 12 km) landward. That keeps the open-coast nearshore strip but cuts most enclosed water
farther from the model's ocean cells: 95% of San Francisco Bay (and San Pablo and Suisun Bays)
was left out, though NOAA's OLCI product observes it. The scope is now that same dilated
C-HARM ocean (offshore extent and open-coast strip unchanged) plus the marine water in this mask.

Input (committed, reproducible): coastwatch-web/public/basemap/water-fallback.json, the Ocean
Map's bundled copy of the OpenMapTiles `water` layer, `ocean` class only (the sea, bays and
estuaries up to the OSM coastline; lakes, reservoirs, lagoons and rivers excluded). It is built
by scripts/build_water_fallback.py.

Method (deterministic; no network):
- the ocean polygons are drawn on the Sentinel-3 OLCI 300 m lattice (0.0025 deg) over the
  California satellite domain, at 2x supersampling, filled and with their outlines drawn, so a
  cell counts as marine water when any part of it touches the ocean polygons (narrow sloughs
  and channels, such as Elkhorn Slough, stay connected);
- only water connected (8-neighbour) to the dilated C-HARM ocean scope is kept, so the mask can
  only add bays and estuaries that open to the coastal ocean.
Stored as run-length encoded rows (west to east), north to south, with the inputs' sha256.
VIIRS 750 m cell centres lie exactly on this lattice (every third cell).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.sources import satellite  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "coastwatch-web" / "public" / "basemap" / "water-fallback.json"
OUT = REPO / "data" / "reference" / satellite.MARINE_WATER_MASK
SS = 2


def rle(mask: np.ndarray) -> list[list[list[int]]]:
    rows = []
    for row in mask:
        d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
        rows.append([[int(s), int(e - s)] for s, e in zip(starts, ends)])
    return rows


def rasterise(src: dict, t: satellite.Target) -> np.ndarray:
    q = float(src["quant"])
    h, w = t.height * SS, t.width * SS
    lon0 = t.lon_first - t.step / 2  # west edge of the first column
    lat0 = t.lat_first + t.step / 2  # north edge of the first row
    img = Image.new("1", (w, h), 0)
    dr = ImageDraw.Draw(img)
    rings: list[list[tuple[float, float]]] = []
    for poly in src["polygons"]:
        for ri, ring in enumerate(poly):
            xy = np.cumsum(np.asarray(ring, dtype=np.float64).reshape(-1, 2), axis=0)
            lon = xy[:, 0] / q * 360.0 - 180.0
            lat = np.degrees(np.arctan(np.sinh(math.pi - 2.0 * math.pi * xy[:, 1] / q)))
            pts = list(zip(((lon - lon0) / t.step * SS).tolist(), ((lat0 - lat) / t.step * SS).tolist()))
            dr.polygon(pts, fill=1 if ri == 0 else 0)
            if ri == 0:
                rings.append(pts)
    for pts in rings:  # cells the shoreline passes through touch water
        dr.line(pts + pts[:1], fill=1, width=1)
    a = np.asarray(img, dtype=bool)
    return a.reshape(t.height, SS, t.width, SS).any(axis=(1, 3))


def main(out: Path = OUT) -> None:
    raw = SRC.read_bytes()
    src = json.loads(raw)
    t = satellite.target_for(satellite.OLCI, satellite.CALIFORNIA)
    water = rasterise(src, t)
    scope = satellite.charm_scope(t)
    lab, _ = label(water, structure=np.ones((3, 3), dtype=bool))
    keep = np.unique(lab[water & scope])
    marine = np.isin(lab, keep[keep > 0])
    charm_ref = REPO / "data" / "reference" / "charm_ocean_mask.json"
    out.write_text(
        json.dumps(
            {
                "description": (
                    "Marine bays and estuaries for the satellite chlorophyll scope: cells of the Sentinel-3 OLCI 300 m lattice that touch "
                    "OpenStreetMap ocean water (OpenMapTiles water layer, ocean class; lakes, reservoirs, lagoons and rivers excluded) "
                    "and connect to the dilated C-HARM ocean scope. Built by pipeline/scripts/build_marine_water_mask.py."
                ),
                "source": f"{src['source']}; {src['tiles']} (built {src['built']})",
                "source_file": str(SRC.relative_to(REPO)),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "charm_ocean_mask_sha256": hashlib.sha256(charm_ref.read_bytes()).hexdigest(),
                "coastal_dilate_cells": satellite.COASTAL_DILATE_CELLS,
                "lat_first": t.lat_first,
                "lat_step": -t.step,
                "lon_first": t.lon_first,
                "lon_step": t.step,
                "height": t.height,
                "width": t.width,
                "rows_rle": rle(marine),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    print(out, f"{int(water.sum())} water cells, {int(marine.sum())} connected marine cells, {int((marine & ~scope).sum())} beyond the dilated C-HARM scope")


if __name__ == "__main__":
    main()
