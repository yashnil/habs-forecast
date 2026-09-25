"""
Ocean/land masking for dashboard overlays.

Primary method: Natural Earth 110m land polygons via GeoPandas (fast spatial join).
Fallback: simple CA offshore polyline if GeoPandas / download fails.

Image row 0 = north (Folium ImageOverlay convention).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

# Approximate mainland shoreline fallback (latitude °N, longitude °W).
_CA_COAST_LAT = np.array(
    [42.0, 41.0, 40.0, 39.0, 38.0, 37.6, 37.0, 36.6, 36.0, 35.5, 35.0, 34.5, 34.0, 33.5, 33.0, 32.5, 32.0],
    dtype=np.float64,
)
_CA_COAST_LON = np.array(
    [-124.6, -124.3, -124.0, -123.8, -123.5, -123.0, -122.4, -121.9, -121.7, -121.2, -120.8, -120.2, -119.5, -118.8, -118.2, -117.3, -117.0],
    dtype=np.float64,
)

NE_LAND_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"


def _interp_coast_lon(lat: np.ndarray) -> np.ndarray:
    lat_cl = np.clip(lat, _CA_COAST_LAT.min(), _CA_COAST_LAT.max())
    return np.interp(lat_cl, _CA_COAST_LAT, _CA_COAST_LON)


def fallback_ocean_mask(lat_cell: np.ndarray, lon_cell: np.ndarray, buffer_deg: float = 0.08) -> np.ndarray:
    """True = ocean (CA window). West of interpolated coast + small buffer."""
    max_ocean_lon = _interp_coast_lon(lat_cell) + buffer_deg
    in_domain = (lat_cell >= 31.5) & (lat_cell <= 42.8) & (lon_cell >= -126.5) & (lon_cell <= -116.8)
    offshore = lon_cell < max_ocean_lon
    return in_domain & offshore


@lru_cache(maxsize=1)
def _natural_earth_land_gdf():
    import geopandas as gpd

    return gpd.read_file(NE_LAND_URL)


def geopandas_ocean_mask(lat_cell: np.ndarray, lon_cell: np.ndarray) -> np.ndarray:
    """True = ocean (point not inside Natural Earth land)."""
    import geopandas as gpd

    land = _natural_earth_land_gdf()
    flat_lat = lat_cell.ravel()
    flat_lon = lon_cell.ravel()
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(flat_lon, flat_lat),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, land[["geometry"]], predicate="within", how="left")
    ocean = joined["index_right"].isna().to_numpy()
    return ocean.reshape(lat_cell.shape)


def ocean_mask_grid(
    lat_ascending: np.ndarray,
    lon_ascending: np.ndarray,
    *,
    method: str = "geopandas",
) -> np.ndarray:
    """
    Ocean mask with shape (n_lat, n_lon), row 0 = north (matches flipped z/rgba).

    method: "geopandas" (default) or "fallback".
    """
    La, Lo = cell_centers_north_row0(lat_ascending, lon_ascending)
    if method == "fallback":
        return fallback_ocean_mask(La, Lo)
    try:
        return geopandas_ocean_mask(La, Lo)
    except Exception:
        return fallback_ocean_mask(La, Lo)


def natural_earth_ocean_mask(lat_cell: np.ndarray, lon_cell: np.ndarray) -> np.ndarray:
    """Alias for cartopy-era name; uses GeoPandas."""
    return geopandas_ocean_mask(lat_cell, lon_cell)


def cell_centers_north_row0(
    lat_ascending: np.ndarray,
    lon_ascending: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lat_desc = lat_ascending[::-1].copy()
    La, Lo = np.meshgrid(lat_desc, lon_ascending, indexing="ij")
    return La, Lo


def mask_rgba_ocean_only(
    rgba: np.ndarray,
    *,
    lat_ascending: np.ndarray,
    lon_ascending: np.ndarray,
    mask_method: str = "geopandas",
) -> np.ndarray:
    """Set alpha=0 on land. rgba rows north → south (row 0 = north)."""
    La, Lo = cell_centers_north_row0(lat_ascending, lon_ascending)
    if mask_method == "fallback":
        ocean = fallback_ocean_mask(La, Lo)
    else:
        try:
            ocean = geopandas_ocean_mask(La, Lo)
        except Exception:
            ocean = fallback_ocean_mask(La, Lo)
    out = np.array(rgba, copy=True, dtype=np.uint8)
    out[~ocean, 3] = 0
    # Remove faint halos: zero near-transparent pixels
    out[out[..., 3] < 12, 3] = 0
    return out


def visible_geographic_bounds(
    rgba: np.ndarray,
    *,
    lat_north: float,
    lat_south: float,
    lon_west: float,
    lon_east: float,
    pad_deg: float = 0.35,
) -> Tuple[float, float, float, float]:
    """Tight (south, west, north, east) from pixels with meaningful alpha."""
    H, W = rgba.shape[0], rgba.shape[1]
    alpha = rgba[..., 3] > 40
    ys, xs = np.where(alpha)
    if ys.size == 0:
        return lat_south - pad_deg, lon_west - pad_deg, lat_north + pad_deg, lon_east + pad_deg

    rmin, rmax = int(ys.min()), int(ys.max())
    cmin, cmax = int(xs.min()), int(xs.max())

    def lat_for_row(r: int) -> float:
        if H <= 1:
            return lat_north
        return lat_north - r * (lat_north - lat_south) / (H - 1)

    def lon_for_col(c: int) -> float:
        if W <= 1:
            return lon_west
        return lon_west + c * (lon_east - lon_west) / (W - 1)

    north_v = lat_for_row(rmin)
    south_v = lat_for_row(rmax)
    west_v = lon_for_col(cmin)
    east_v = lon_for_col(cmax)
    return (
        south_v - pad_deg,
        west_v - pad_deg,
        north_v + pad_deg,
        east_v + pad_deg,
    )
