from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from scipy.io import netcdf_file

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.http import FetchError, Response


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "v1"


@pytest.fixture
def ctx(out: Path):
    return fixture_context(out)


def make_nc(
    *,
    time: datetime = datetime(2026, 10, 7, 12, tzinfo=timezone.utc),
    lat=None,
    lon=None,
    values: dict[str, np.ndarray] | None = None,
    version: str = "3.1",
    drop: tuple[str, ...] = (),
    fill: float = -99999.0,
) -> bytes:
    """Build a small ERDDAP-like NetCDF-3 file for malformed-input tests."""
    lat = np.round(36.01 + 0.03 * np.arange(10), 4) if lat is None else np.asarray(lat)
    lon = np.round(237.81 + 0.03 * np.arange(10), 4) if lon is None else np.asarray(lon)
    names = ("pseudo_nitzschia", "particulate_domoic", "cellular_domoic")
    if values is None:
        rng = np.random.default_rng(0)
        values = {n: rng.uniform(0, 1, (lat.size, lon.size)) for n in names}
    buf = io.BytesIO()
    f = netcdf_file(buf, "w", version=1)
    f.product_version = version
    f.history = "test fixture"
    f.createDimension("time", 1)
    f.createDimension("latitude", lat.size)
    f.createDimension("longitude", lon.size)
    if "time" not in drop:
        t = f.createVariable("time", "d", ("time",))
        t[:] = [time.timestamp()]
    la = f.createVariable("latitude", "d", ("latitude",))
    la[:] = lat
    lo = f.createVariable("longitude", "d", ("longitude",))
    lo[:] = lon
    for n in names:
        if n in drop:
            continue
        v = f.createVariable(n, "f", ("time", "latitude", "longitude"))
        v._FillValue = np.float32(fill)
        v.missing_value = np.float32(fill)
        a = np.array(values[n], dtype=np.float32)
        a[~np.isfinite(a)] = fill
        v[:] = a[None, ...]
    f.flush()
    data = buf.getvalue()
    f.close()
    return data


def static(body: bytes, status: int = 200, ctype: str = "application/octet-stream"):
    return lambda url: Response(url=url, status=status, content_type=ctype, body=body)


def failing(msg: str = "HTTP 503"):
    def fn(url):
        raise FetchError(f"{msg} for {url}")

    return fn


__all__ = ["make_nc", "static", "failing", "FixtureFetcher"]
