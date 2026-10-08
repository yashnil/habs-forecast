"""Reproject a regular lat/lon grid to a Web Mercator (EPSG:3857) image.

Placing an equirectangular image by its four corners on a Mercator map stretches it
linearly in Mercator y, which misplaces features by up to ~18 km over 32-42 N. We
instead resample onto pixels that are uniform in Mercator y, so corner placement is
exact. Nearest-neighbour sampling keeps every pixel equal to a real source cell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

R_EARTH = 6378137.0


def lat_to_merc_y(lat: np.ndarray | float) -> np.ndarray | float:
    lat_r = np.radians(lat)
    return R_EARTH * np.log(np.tan(np.pi / 4 + lat_r / 2))


def merc_y_to_lat(y: np.ndarray | float) -> np.ndarray | float:
    return np.degrees(2 * np.arctan(np.exp(np.asarray(y) / R_EARTH)) - np.pi / 2)


@dataclass(frozen=True)
class SourceGrid:
    """Regular grid described by cell centres. lat/lon steps may be negative."""

    lat_first: float
    lat_step: float
    lon_first: float
    lon_step: float
    height: int
    width: int

    @property
    def west(self) -> float:
        return min(self.lon_first, self.lon_first + self.lon_step * (self.width - 1)) - abs(self.lon_step) / 2

    @property
    def east(self) -> float:
        return max(self.lon_first, self.lon_first + self.lon_step * (self.width - 1)) + abs(self.lon_step) / 2

    @property
    def south(self) -> float:
        return min(self.lat_first, self.lat_first + self.lat_step * (self.height - 1)) - abs(self.lat_step) / 2

    @property
    def north(self) -> float:
        return max(self.lat_first, self.lat_first + self.lat_step * (self.height - 1)) + abs(self.lat_step) / 2

    def cell_index(self, lat: np.ndarray | float, lon: np.ndarray | float):
        """Row/col of the cell containing (lat, lon); -1 where outside the grid."""
        r = np.floor((np.asarray(lat) - self.lat_first) / self.lat_step + 0.5).astype(np.int64)
        c = np.floor((np.asarray(lon) - self.lon_first) / self.lon_step + 0.5).astype(np.int64)
        inside = (r >= 0) & (r < self.height) & (c >= 0) & (c < self.width)
        return np.where(inside, r, -1), np.where(inside, c, -1)


@dataclass(frozen=True)
class MercatorImage:
    width: int
    height: int
    west: float
    east: float
    south: float
    north: float

    def pixel_of(self, lat: float, lon: float) -> tuple[int, int]:
        """(row, col) of the image pixel containing a lat/lon."""
        x0, x1 = self.west, self.east
        y_top, y_bot = lat_to_merc_y(self.north), lat_to_merc_y(self.south)
        col = int(math.floor((lon - x0) / (x1 - x0) * self.width))
        row = int(math.floor((y_top - lat_to_merc_y(lat)) / (y_top - y_bot) * self.height))
        return row, col

    @property
    def corners(self):
        return (
            (self.west, self.north),
            (self.east, self.north),
            (self.east, self.south),
            (self.west, self.south),
        )


def plan_image(src: SourceGrid, upsample: int = 4) -> MercatorImage:
    """Choose an output size whose pixels are ~1/upsample of a source cell at mid-latitude."""
    width = src.width * upsample
    lon_span = src.east - src.west
    y_span = float(lat_to_merc_y(src.north) - lat_to_merc_y(src.south))
    x_span = R_EARTH * math.radians(lon_span)
    height = int(round(width * y_span / x_span))
    return MercatorImage(width, height, src.west, src.east, src.south, src.north)


def resample_to_mercator(values: np.ndarray, src: SourceGrid, img: MercatorImage) -> np.ndarray:
    """Nearest-neighbour resample of `values` (height x width, source order) onto the image."""
    if values.shape != (src.height, src.width):
        raise ValueError(f"values shape {values.shape} != grid {(src.height, src.width)}")
    y_top, y_bot = lat_to_merc_y(img.north), lat_to_merc_y(img.south)
    ys = y_top - (np.arange(img.height) + 0.5) / img.height * (y_top - y_bot)
    lats = merc_y_to_lat(ys)
    lons = img.west + (np.arange(img.width) + 0.5) / img.width * (img.east - img.west)
    rows, _ = src.cell_index(lats, np.full_like(lats, src.lon_first))
    _, cols = src.cell_index(np.full_like(lons, src.lat_first), lons)
    out = np.full((img.height, img.width), np.nan, dtype=np.float64)
    rr = rows[:, None]
    cc = cols[None, :]
    ok = (rr >= 0) & (cc >= 0)
    out[ok] = values[np.broadcast_to(rr, ok.shape)[ok], np.broadcast_to(cc, ok.shape)[ok]]
    return out
