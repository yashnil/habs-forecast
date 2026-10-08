"""Fixed palettes. Ranges never depend on the data being rendered (safety rule R6)."""

from __future__ import annotations

import numpy as np

from ..models import Palette, PaletteStop

# Single-hue magenta ramp, OKLCH L 0.36 -> 0.92 (monotone), hue 345.
# Low end recedes toward the navy surface; high end is brightest. Magenta is
# kept distinct from water blues and from the reserved status colours.
_PROB_HEX = [
    "#5d2748",
    "#7f2a60",
    "#a12e79",
    "#be3b90",
    "#d753a6",
    "#e872b9",
    "#f394cb",
    "#f9b6db",
    "#fed8ec",
]

PROBABILITY = Palette(
    id="cw-probability-magenta-v1",
    domain=[0.0, 1.0],
    stops=[
        PaletteStop(value=round(i / (len(_PROB_HEX) - 1), 6), color=c)
        for i, c in enumerate(_PROB_HEX)
    ],
)


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def apply_palette(values: np.ndarray, palette: Palette) -> np.ndarray:
    """Map values to RGBA uint8. NaN -> fully transparent. Values are clipped to the
    palette domain; validation rejects out-of-range data before this is called."""
    lo, hi = palette.domain
    xs = np.array([s.value for s in palette.stops], dtype=np.float64)
    cols = np.array([_hex_to_rgb(s.color) for s in palette.stops], dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(v)
    t = np.clip(np.where(valid, v, lo), lo, hi)
    t = (t - lo) / (hi - lo)
    out = np.zeros(v.shape + (4,), dtype=np.uint8)
    for ch in range(3):
        out[..., ch] = np.round(np.interp(t, xs, cols[:, ch])).astype(np.uint8)
    out[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return out


def palette_color(value: float, palette: Palette) -> tuple[int, int, int]:
    rgba = apply_palette(np.array([value]), palette)[0]
    return int(rgba[0]), int(rgba[1]), int(rgba[2])
