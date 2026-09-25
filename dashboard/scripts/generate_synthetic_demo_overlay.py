#!/usr/bin/env python3
"""
Rebuild dashboard/data/overlay.png with ocean-only pixels (transparent on land).

Uses the same CA offshore polyline fallback as export_map_snapshot.py --mask-fallback
so the map shows chlorophyll only over water (coastal strip / ocean), not inland.

  python3 dashboard/scripts/generate_synthetic_demo_overlay.py
  cd coastwatch-web && npm run sync-data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_DASH = Path(__file__).resolve().parents[1]
if str(_DASH) not in sys.path:
    sys.path.insert(0, str(_DASH))

import struct
import zlib

from ocean_mask import cell_centers_north_row0, fallback_ocean_mask  # noqa: E402


def _write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    """Minimal RGBA PNG writer (stdlib only)."""
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected HxWx4 uint8")
    h, w = rgba.shape[0], rgba.shape[1]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    raw = b"".join(b"\x00" + rgba[y, :, :].tobytes() for y in range(h))
    compressed = zlib.compress(raw, 9)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _ylgnbu_rgba(t: np.ndarray) -> np.ndarray:
    """Rough YlGnBu-style RGBA, t in [0,1]."""
    shape = t.shape
    u = np.clip(t.astype(np.float64).ravel(), 0.0, 1.0)
    xp = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=np.float64)
    r = np.interp(u, xp, [255.0, 199.0, 65.0, 34.0])
    g = np.interp(u, xp, [255.0, 233.0, 182.0, 94.0])
    bch = np.interp(u, xp, [217.0, 180.0, 196.0, 168.0])
    rgb = np.stack([r, g, bch], axis=-1).reshape(shape + (3,))
    a = np.full(shape + (1,), 255, dtype=np.uint8)
    rgb_u8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return np.concatenate([rgb_u8, a], axis=-1)


def main() -> None:
    snap_path = _DASH / "data" / "snapshot.json"
    out_png = _DASH / "data" / "overlay.png"
    manifest = json.loads(snap_path.read_text())
    b = manifest["bounds"]
    south, west, north, east = b["south"], b["west"], b["north"], b["east"]
    vmin = float(manifest["colormap"]["vmin"])
    vmax = float(manifest["colormap"]["vmax"])

    n_lat, n_lon = 400, 360
    lat_asc = np.linspace(south, north, n_lat)
    lon_asc = np.linspace(west, east, n_lon)
    La, Lo = np.meshgrid(lat_asc, lon_asc, indexing="ij")
    cx, cy = (west + east) / 2, (south + north) / 2
    dist = np.sqrt((Lo - cx) ** 2 + (La - cy) ** 2)
    z = vmin + (vmax - vmin) * np.exp(-(dist / 2.8) ** 2)
    z += 0.04 * (vmax - vmin) * np.sin(La * 0.9) * np.cos(Lo * 0.7)

    if lat_asc[0] < lat_asc[-1]:
        z = np.flipud(z)

    lat_sorted = np.sort(lat_asc)
    lon_sorted = np.sort(lon_asc)
    tnorm = (z - vmin) / max(vmax - vmin, 1e-9)
    rgba = _ylgnbu_rgba(tnorm.astype(np.float64))

    La2, Lo2 = cell_centers_north_row0(lat_sorted, lon_sorted)
    ocean = fallback_ocean_mask(La2, Lo2)
    rgba = np.array(rgba, copy=True, dtype=np.uint8)
    rgba[~ocean, 3] = 0
    rgba[rgba[..., 3] < 12, 3] = 0

    _write_rgba_png(out_png, rgba)
    print(f"Wrote ocean-masked demo overlay: {out_png}")


if __name__ == "__main__":
    main()
