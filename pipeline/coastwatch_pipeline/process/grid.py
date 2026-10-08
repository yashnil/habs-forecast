"""Quantized value grids: uint16 little-endian, gzip-compressed, deterministic bytes."""

from __future__ import annotations

import gzip
import io

import numpy as np

NODATA = 65535
MAX_CODE = 65534


def encode(values: np.ndarray, lo: float, hi: float) -> tuple[bytes, float, float, float]:
    """Return (gzip bytes, scale_factor, add_offset, max_quantization_error)."""
    scale = (hi - lo) / MAX_CODE
    v = np.asarray(values, dtype=np.float64)
    codes = np.full(v.shape, NODATA, dtype="<u2")
    ok = np.isfinite(v)
    codes[ok] = np.clip(np.round((v[ok] - lo) / scale), 0, MAX_CODE).astype("<u2")
    buf = io.BytesIO()
    # mtime=0 keeps output byte-identical for identical input
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(codes.tobytes(order="C"))
    return buf.getvalue(), scale, lo, scale / 2


def decode(blob: bytes, width: int, height: int, scale: float, offset: float, nodata: int = NODATA) -> np.ndarray:
    codes = np.frombuffer(gzip.decompress(blob), dtype="<u2").reshape(height, width)
    out = codes.astype(np.float64) * scale + offset
    out[codes == nodata] = np.nan
    return out
