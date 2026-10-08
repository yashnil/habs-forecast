from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_bytes(path: Path, data: bytes) -> None:
    _atomic_write(path, data)


def png_bytes(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def write_png(path: Path, rgba: np.ndarray) -> None:
    _atomic_write(path, png_bytes(rgba))


def write_json(path: Path, obj: object) -> None:
    _atomic_write(path, (json.dumps(obj, indent=2, sort_keys=False) + "\n").encode("utf-8"))
