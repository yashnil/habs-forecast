"""Render a regular lat/lon grid as Web Mercator XYZ tiles (256 px, palette PNGs).

Every tile pixel takes the value of the source cell containing the pixel centre
(nearest neighbour), so a pixel is always one real cell: nothing is interpolated or
gap-filled, and cells without a value stay transparent. Above the native zoom, the
browser enlarges tiles (nearest), never the pipeline.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .mercator import SourceGrid

TILE = 256


def lon_to_x(lon, z):
    return (np.asarray(lon) + 180.0) / 360.0 * (2**z) * TILE


def lat_to_y(lat, z):
    lr = np.radians(np.asarray(lat))
    return (1 - np.log(np.tan(lr) + 1 / np.cos(lr)) / math.pi) / 2 * (2**z) * TILE


def x_to_lon(x, z):
    return np.asarray(x) / ((2**z) * TILE) * 360.0 - 180.0


def y_to_lat(y, z):
    n = math.pi - 2 * math.pi * np.asarray(y) / ((2**z) * TILE)
    return np.degrees(np.arctan(np.sinh(n)))


def native_zoom(cell_m: float, lat: float) -> int:
    """Smallest zoom whose pixel is no larger than one source cell at `lat`."""
    for z in range(4, 15):
        if 156543.034 / 2**z * math.cos(math.radians(lat)) <= cell_m:
            return z
    return 14


@dataclass
class TileSet:
    n_tiles: int = 0
    bytes: int = 0
    tiles: list[str] = field(default_factory=list)


def render_tiles(
    values: np.ndarray,
    src: SourceGrid,
    bounds: tuple[float, float, float, float],
    zooms: range,
    to_index: Callable[[np.ndarray], np.ndarray],
    colours: list[tuple[int, int, int]],
    out_dir: Path,
) -> TileSet:
    """Write {z}/{x}/{y}.png under out_dir for every tile intersecting `bounds`
    (west, south, east, north) that has at least one valid pixel. `to_index` maps
    values to palette indices (0 = transparent); `colours` is the palette."""
    return render_stack([(values, src, to_index)], bounds, zooms, colours, out_dir)


def render_stack(
    layers: list[tuple[np.ndarray, SourceGrid, Callable[[np.ndarray], np.ndarray]]],
    bounds: tuple[float, float, float, float],
    zooms: range,
    colours: list[tuple[int, int, int]],
    out_dir: Path,
) -> TileSet:
    """Like render_tiles for several grids drawn top to bottom: each tile pixel takes the
    first layer with a non-transparent index at that pixel, each layer sampled on its own
    lattice (nearest cell). Grids keep their native cell edges; nothing is resampled
    between them."""
    west, south, east, north = bounds
    pal = [c for rgb in colours for c in rgb] + [0, 0, 0] * (256 - len(colours))
    ts = TileSet()
    for z in zooms:
        x0, x1 = int(lon_to_x(west, z) // TILE), int(lon_to_x(east, z) // TILE)
        y0, y1 = int(lat_to_y(north, z) // TILE), int(lat_to_y(south, z) // TILE)
        px = (np.arange(x0 * TILE, (x1 + 1) * TILE) + 0.5)
        lons = x_to_lon(px, z)
        cols = [src.cell_index(np.full_like(lons, src.lat_first), lons)[1] for _, src, _ in layers]
        for ty in range(y0, y1 + 1):
            py = np.arange(ty * TILE, (ty + 1) * TILE) + 0.5
            lats = y_to_lat(py, z)
            idx: np.ndarray | None = None
            for (values, src, to_index), cl in zip(layers, cols):
                rows, _ = src.cell_index(lats, np.full_like(lats, src.lon_first))
                ok = (rows[:, None] >= 0) & (cl[None, :] >= 0)
                strip = np.full(ok.shape, np.nan)
                if ok.any():
                    strip[ok] = values[np.broadcast_to(rows[:, None], ok.shape)[ok], np.broadcast_to(cl[None, :], ok.shape)[ok]]
                li = to_index(strip).astype(np.uint8)
                if idx is None:
                    idx = li
                else:
                    gap = idx == 0
                    idx[gap] = li[gap]
            assert idx is not None
            for i, tx in enumerate(range(x0, x1 + 1)):
                tile = idx[:, i * TILE : (i + 1) * TILE]
                if not tile.any():
                    continue
                im = Image.fromarray(np.ascontiguousarray(tile), mode="P")
                im.putpalette(pal)
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True, transparency=0)
                p = out_dir / str(z) / str(tx) / f"{ty}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(buf.getvalue())
                ts.n_tiles += 1
                ts.bytes += len(buf.getvalue())
                ts.tiles.append(f"{z}/{tx}/{ty}")
    return ts


def write_chunks(codes_blob_fn: Callable[[np.ndarray], bytes], values: np.ndarray, rows: int, cols: int, out_dir: Path) -> tuple[list[str], int]:
    """Split `values` into rows x cols chunks and write the ones with any finite value
    as {r}_{c}.u16.gz. Returns (present keys, total bytes)."""
    present, total = [], 0
    h, w = values.shape
    for r in range(math.ceil(h / rows)):
        for c in range(math.ceil(w / cols)):
            block = values[r * rows : (r + 1) * rows, c * cols : (c + 1) * cols]
            if not np.isfinite(block).any():
                continue
            blob = codes_blob_fn(block)
            p = out_dir / f"{r}_{c}.u16.gz"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
            present.append(f"{r}_{c}")
            total += len(blob)
    return present, total
