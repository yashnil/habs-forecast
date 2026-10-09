"""Render published C-HARM value grids with the design-reset forecast palette.

Reads the uint16 value grids that the production pipeline publishes on GitHub Pages
(the same numbers the live map uses) and writes Web Mercator PNGs for the prototype.
Only the colour mapping differs from production: a 10-class stepped ramp instead of a
continuous one. Reprojection is the pipeline's nearest-neighbour method, so every
pixel is one real model cell; no smoothing, no data-dependent range.

usage: python render_rasters.py <pages-v1-dir> <out-dir>
"""

from __future__ import annotations

import gzip
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

R = 6378137.0

# Forecast probability ramp "cw-probability-classes-v1": 10 display classes of 10
# percentage points. They are colour steps for reading the map, not risk categories;
# exact values come from the grid. Built in OKLCH: lightness rises in equal steps
# 0.36 -> 0.80 (adjacent classes ~1.2:1), so order survives greyscale and colour-vision
# deficiency; hue turns dusk violet -> mauve -> rose at low chroma, so the common
# 60-80 % range reads as a mid tone and the coastline, land and labels stay legible.
# "No value" is never a colour on this ramp: the map hatches water without a value and
# the raster is opaque, so hatching only shows where there is no model value. No class
# uses the amber reserved for official notices or the teal reserved for measurements.
FORECAST_CLASSES = [
    "#3a385b",
    "#4c436a",
    "#5f4e79",
    "#735986",
    "#886492",
    "#9c709c",
    "#af7ea4",
    "#c28cab",
    "#d39cb3",
    "#e5abbc",
]


if os.environ.get("CW_PALETTE"):  # design exploration only: comma-separated hexes
    FORECAST_CLASSES = os.environ["CW_PALETTE"].split(",")


def merc_y(lat):
    return R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def lat_of(y):
    return np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)


def render(values: np.ndarray, g: dict, upsample: int = 4) -> tuple[Image.Image, list]:
    h, w = values.shape
    west = g["lon_first"] - g["lon_step"] / 2
    east = g["lon_first"] + g["lon_step"] * (w - 0.5)
    south = g["lat_first"] - g["lat_step"] / 2
    north = g["lat_first"] + g["lat_step"] * (h - 0.5)
    width = w * upsample
    height = int(round(width * (merc_y(north) - merc_y(south)) / (R * math.radians(east - west))))
    yt, yb = merc_y(north), merc_y(south)
    lats = lat_of(yt - (np.arange(height) + 0.5) / height * (yt - yb))
    lons = west + (np.arange(width) + 0.5) / width * (east - west)
    rows = np.clip(np.floor((lats - g["lat_first"]) / g["lat_step"] + 0.5).astype(int), 0, h - 1)
    cols = np.clip(np.floor((lons - g["lon_first"]) / g["lon_step"] + 0.5).astype(int), 0, w - 1)
    v = values[rows[:, None], cols[None, :]]
    rgba = np.zeros(v.shape + (4,), np.uint8)
    ok = np.isfinite(v)
    cls = np.clip((np.where(ok, v, 0) * 10).astype(int), 0, 9)
    pal = np.array([[int(c[i : i + 2], 16) for i in (1, 3, 5)] for c in FORECAST_CLASSES], np.uint8)
    rgba[..., :3] = pal[cls]
    rgba[..., 3] = np.where(ok, 255, 0)
    corners = [[west, north], [east, north], [east, south], [west, south]]
    return Image.fromarray(rgba, "RGBA"), corners


def main(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # The prototype also ships the published value grids unchanged, so the map can read
    # out the exact model value under the pointer, as production does.
    grids = out.parent / "grids"
    grids.mkdir(exist_ok=True)
    manifest = json.loads((src / "manifest.json").read_text())
    index = {}
    for layer in manifest["layers"]:
        g = layer.get("grid")
        if not g:
            continue
        raw = np.frombuffer(gzip.open(src / g["url"]).read(), "<u2").reshape(g["height"], g["width"])
        vals = np.where(raw == g["nodata"], np.nan, raw * g["scale_factor"] + g["add_offset"])
        img, corners = render(vals, g)
        name = f"{layer['variable']}-lead{layer['time']['lead_days']}.png"
        img.save(out / name, optimize=True)
        shutil.copyfile(src / g["url"], grids / f"{layer['layer_id']}.u16.gz")
        meta = {k: g[k] for k in ("width", "height", "lat_first", "lat_step", "lon_first", "lon_step", "scale_factor", "add_offset", "nodata")}
        index[layer["layer_id"]] = {"url": f"img/{name}", "corners": corners, "grid": {"url": f"grids/{layer['layer_id']}.u16.gz", **meta}}
    (out / "rasters.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} rasters to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
