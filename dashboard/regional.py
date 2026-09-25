"""Regional summaries along the CA coast for the dashboard manifest."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

# (key, south_lat, north_lat, short label)
CA_REGIONS: List[Tuple[str, float, float, str]] = [
    ("north_coast", 40.0, 42.9, "North coast"),
    ("north_bay", 38.2, 40.0, "North Bay / Mendocino–Sonoma"),
    ("central_bay", 37.0, 38.2, "SF Bay / Marin"),
    ("monterey", 35.8, 37.0, "Monterey / Santa Cruz"),
    ("morro", 34.6, 35.8, "Morro / Big Sur"),
    ("southern", 32.0, 34.6, "Southern CA"),
]


def _lat_grid(lat_north: float, lat_south: float, H: int, W: int) -> np.ndarray:
    if H <= 1:
        v = np.array([0.5 * (lat_north + lat_south)], dtype=np.float64)
    else:
        v = lat_north - np.arange(H, dtype=np.float64) * (lat_north - lat_south) / (H - 1)
    return np.broadcast_to(v[:, np.newaxis], (H, W))


def compute_regional_algae(
    z: np.ndarray,
    ocean: np.ndarray,
    *,
    lat_north: float,
    lat_south: float,
) -> Dict[str, Any]:
    """
    z, ocean: shape (H, W), row 0 = north. Returns dict for snapshot.json.
    Tiers are relative to all ocean pixels on this map (tertiles).
    """
    H, W = z.shape
    lat_row = _lat_grid(lat_north, lat_south, H, W)

    valid = ocean & np.isfinite(z)
    vals_all = z[valid]
    if vals_all.size < 30:
        p33 = p67 = float(np.nanmedian(vals_all)) if vals_all.size else 0.0
    else:
        p33, p67 = np.nanpercentile(vals_all, [33.3, 66.7])

    regions_out: Dict[str, Any] = {}
    for key, lo, hi, label in CA_REGIONS:
        m = valid & (lat_row >= lo) & (lat_row < hi)
        vv = z[m]
        n = int(np.sum(m))
        if n < 8:
            regions_out[key] = {
                "label": label,
                "lat_range": [lo, hi],
                "status": "not_enough_water_pixels",
                "pixel_count": n,
            }
            continue
        med = float(np.nanmedian(vv))
        if med <= p33:
            tier = "lower_than_most_water_on_this_map"
            tier_short = "Lower"
        elif med >= p67:
            tier = "higher_than_most_water_on_this_map"
            tier_short = "Higher"
        else:
            tier = "typical_for_this_map"
            tier_short = "Typical"

        regions_out[key] = {
            "label": label,
            "lat_range": [lo, hi],
            "median_log_chl": med,
            "tier": tier,
            "tier_short": tier_short,
            "pixel_count": n,
        }

    return {
        "reference_percentiles": {"p33": float(p33), "p67": float(p67)},
        "regions": regions_out,
    }
