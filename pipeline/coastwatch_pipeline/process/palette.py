"""Fixed palettes. Ranges never depend on the data being rendered (safety rule R6)."""

from __future__ import annotations

import numpy as np

from ..models import Palette, PaletteStop

# C-HARM probability, "cw-probability-classes-v1" (design reset, approved 2026-10-09):
# ten 10-percentage-point display classes, stepped (not interpolated). They are colour
# steps for reading the map, not risk categories; exact values stay in the value grid.
# OKLCH lightness rises in equal steps 0.36 -> 0.80 so order survives greyscale and
# colour-vision deficiency; low chroma keeps the common 60-80 % range a mid tone so the
# coastline and labels stay legible. "No value" is never a ramp colour: the map hatches
# water without a value and draws the raster opaque. Mirrors coastwatch-web
# src/lib/palette.ts (FORECAST_CLASSES), which a test keeps in sync.
_PROB_CLASSES = [
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

PROBABILITY = Palette(
    id="cw-probability-classes-v1",
    domain=[0.0, 1.0],
    stops=[PaletteStop(value=round(i / 10, 6), color=c) for i, c in enumerate(_PROB_CLASSES)],
    interpolation="step",
)

# Satellite chlorophyll-a, "cw-chlorophyll-log-v1": log10 domain 0.05-50 mg m-3, fixed.
# Observation family: deep blue -> teal -> green -> pale lime, lightness rising
# monotonically. Avoids the violet model ramp, the amber reserved for official notices
# and the teal used for station markers.
_CHL_HEX = ["#102a4c", "#14466e", "#126483", "#147f85", "#2a9a7c", "#5bb267", "#9bc851", "#d6dd5e", "#f3ef9c"]
CHL_LOG_DOMAIN = (float(np.log10(0.05)), float(np.log10(50.0)))

CHLOROPHYLL = Palette(
    id="cw-chlorophyll-log-v1",
    domain=[round(CHL_LOG_DOMAIN[0], 6), round(CHL_LOG_DOMAIN[1], 6)],
    stops=[
        PaletteStop(value=round(CHL_LOG_DOMAIN[0] + (CHL_LOG_DOMAIN[1] - CHL_LOG_DOMAIN[0]) * i / (len(_CHL_HEX) - 1), 6), color=c)
        for i, c in enumerate(_CHL_HEX)
    ],
    interpolation="linear",
    scale="log10",
)

# Age of a composite pixel in days (0 = observed on the reference date). Categorical,
# neutral greys-to-blue so it never reads as a value scale; index = age in days.
AGE_COLOURS = ["#e8f1f8", "#bcd3e6", "#8fb2d0", "#6790b5", "#4a7097", "#365477", "#273d58", "#1c2c40"]


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _position(values: np.ndarray, palette: Palette) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(v)
    if palette.scale == "log10":
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(valid & (v > 0), np.log10(np.where(v > 0, v, 1.0)), np.nan)
        valid = np.isfinite(v)
    lo, hi = palette.domain
    return np.clip(np.where(valid, v, lo), lo, hi), valid


def apply_palette(values: np.ndarray, palette: Palette) -> np.ndarray:
    """Map values to RGBA uint8. NaN -> fully transparent. Values are clipped to the
    palette domain; validation rejects out-of-range data before this is called."""
    t, valid = _position(values, palette)
    xs = np.array([s.value for s in palette.stops], dtype=np.float64)
    cols = np.array([_hex_to_rgb(s.color) for s in palette.stops], dtype=np.float64)
    out = np.zeros(t.shape + (4,), dtype=np.uint8)
    if palette.interpolation == "step":
        idx = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 1)
        out[..., :3] = cols[idx].astype(np.uint8)
    else:
        for ch in range(3):
            out[..., ch] = np.round(np.interp(t, xs, cols[:, ch])).astype(np.uint8)
    out[..., 3] = np.where(valid, 255, 0).astype(np.uint8)
    return out


def palette_indices(values: np.ndarray, palette: Palette, n: int = 254) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Quantize values to n colours for 8-bit palette PNG tiles. Index 0 is transparent
    (no value); indices 1..n sample the palette evenly across its domain."""
    t, valid = _position(values, palette)
    lo, hi = palette.domain
    k = np.clip(np.floor((t - lo) / (hi - lo) * n), 0, n - 1).astype(np.int32) + 1
    idx = np.where(valid, k, 0).astype(np.uint8)
    centres = lo + (np.arange(n) + 0.5) / n * (hi - lo)
    if palette.scale == "log10":
        centres = 10**centres
    rgb = apply_palette(centres, palette)[:, :3]
    return idx, [(0, 0, 0)] + [tuple(int(c) for c in row) for row in rgb]


def palette_color(value: float, palette: Palette) -> tuple[int, int, int]:
    rgba = apply_palette(np.array([value]), palette)[0]
    return int(rgba[0]), int(rgba[1]), int(rgba[2])
