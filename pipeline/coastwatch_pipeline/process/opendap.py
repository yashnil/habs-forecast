"""Minimal OPeNDAP (DAP2) reader for THREDDS servers: binary `.dods` data and `.das` attributes.

Used for NOAA NDBC's THREDDS copy of the HFRNet surface currents, which is served by OPeNDAP
only (no ERDDAP, no NetCDF subset service). Only what that needs is supported: numeric arrays
and scalars, including arrays inside Grid and Structure declarations. Values are returned as
stored (packed integers); scale_factor, add_offset and _FillValue are applied by the caller from
the attributes, exactly as for the NetCDF responses elsewhere in the pipeline.

DAP2 data encoding (OPeNDAP DAP2 spec, section 7.3): after the constrained DDS and the line
"Data:", each declared variable follows in declaration order as XDR (big endian). An array starts
with its length written twice; Int16/UInt16 occupy 4 bytes each; Byte arrays are packed and
padded to a multiple of 4 bytes. A Grid sends its array, then its map vectors.
"""

from __future__ import annotations

import re
import struct

import numpy as np

_TYPES = {
    "Byte": (np.dtype(np.uint8), 1),
    "Int16": (np.dtype(">i4"), 4),
    "UInt16": (np.dtype(">u4"), 4),
    "Int32": (np.dtype(">i4"), 4),
    "UInt32": (np.dtype(">u4"), 4),
    "Float32": (np.dtype(">f4"), 4),
    "Float64": (np.dtype(">f8"), 8),
}
_DECL = re.compile(r"^\s*(Byte|Int16|UInt16|Int32|UInt32|Float32|Float64|String|Url)\s+(\w+)((?:\[\s*\w+\s*=\s*\d+\s*\])*)\s*;", re.M)
_MARK = b"\nData:\n"


def parse_dods(body: bytes) -> dict[str, np.ndarray]:
    """Arrays by name, in their declared shape. A name declared more than once (the map vectors
    repeated in every Grid) keeps its first occurrence. Raises ValueError on anything malformed,
    so a truncated or unexpected response is never read as data."""
    i = body.find(_MARK)
    if i < 0:
        head = body[:200].decode("utf-8", "replace")
        raise ValueError(f"not a DAP2 data response: {head!r}")
    dds, buf = body[:i].decode("utf-8", "replace"), body[i + len(_MARK) :]
    if not dds.lstrip().startswith("Dataset"):
        raise ValueError("DAP2 response without a Dataset declaration")
    off, out = 0, {}
    for typ, name, dims in _DECL.findall(dds):
        if typ in ("String", "Url"):
            raise ValueError(f"string variable {name} requested; not supported")
        dt, size = _TYPES[typ]
        shape = tuple(int(n) for n in re.findall(r"=\s*(\d+)", dims))
        if not shape:
            if off + 4 > len(buf):
                raise ValueError(f"response ends inside scalar {name}")
            if typ == "Byte":
                val = np.array(buf[off + 3], dtype=np.uint8)  # XDR pads a byte to 4, value last
                off += 4
            else:
                val = np.frombuffer(buf, dt, 1, off)[0]
                off += size
            out.setdefault(name, np.asarray(val))
            continue
        if off + 8 > len(buf):
            raise ValueError(f"response ends before array {name}")
        n1, n2 = struct.unpack(">II", buf[off : off + 8])
        off += 8
        n = int(np.prod(shape))
        if n1 != n or n2 != n:
            raise ValueError(f"array {name}: declared {n} values, response says {n1}/{n2}")
        nbytes = (n + 3) // 4 * 4 if typ == "Byte" else n * size
        if off + nbytes > len(buf):
            raise ValueError(f"response ends inside array {name}")
        out.setdefault(name, np.frombuffer(buf, dt, n, off).reshape(shape))
        off += nbytes
    if off != len(buf):
        raise ValueError(f"{len(buf) - off} unexpected bytes after the declared variables")
    return out


_ATTR = re.compile(r"^\s*(Byte|Int16|UInt16|Int32|UInt32|Float32|Float64|String)\s+(\w+)\s+(.*?);?\s*$")


def parse_das(text: str) -> dict[str, dict[str, object]]:
    """Attributes per variable (top-level containers only): numbers as float, strings as str."""
    out: dict[str, dict[str, object]] = {}
    depth, cur = 0, None
    for line in text.splitlines():
        s = line.strip()
        if s.endswith("{"):
            depth += 1
            if depth == 2:
                cur = s[:-1].strip()
                out[cur] = {}
            continue
        if s == "}":
            depth -= 1
            if depth < 2:
                cur = None
            continue
        if cur is None or depth != 2:
            continue
        m = _ATTR.match(s)
        if not m:
            continue
        typ, key, val = m.groups()
        val = val.rstrip(";").strip()
        if typ == "String":
            out[cur][key] = val.strip('"')
        else:
            try:
                out[cur][key] = float(val.split(",")[0])
            except ValueError:
                continue
    return out


def encode_dods(dds: str, arrays: list[tuple[str, np.ndarray]]) -> bytes:
    """The inverse of parse_dods, for test fixtures: `arrays` in declaration order, typed as
    declared in `dds`."""
    types = {name: typ for typ, name, _ in _DECL.findall(dds)}
    parts = [dds.encode(), _MARK]
    for name, a in arrays:
        typ = types[name]
        dt, _ = _TYPES[typ]
        flat = np.asarray(a).ravel()
        parts.append(struct.pack(">II", flat.size, flat.size))
        raw = flat.astype(dt).tobytes()
        if typ == "Byte":
            raw += b"\0" * ((-len(raw)) % 4)
        parts.append(raw)
    return b"".join(parts)
