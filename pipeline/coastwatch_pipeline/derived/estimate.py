"""Gap-filled chlorophyll ("estimated, not observed") on the Sentinel-3 OLCI 300 m lattice.

Method (validated on held-out OLCI days: pipeline/scripts/validate_chl_estimate.py, results in
data/reference/chl_estimate_validation.json):

1. Background: each pixel's recent typical value, an exponentially weighted mean of its own valid
   log10 observations (e-folding TAU_DAYS), carried from run to run as a small state grid. It
   stands in for the research's "recent median" and is updated causally (only past days).
2. Pixels with too little recent history of their own (below MIN_WEIGHT, e.g. the never-observed
   strip next to the shore) borrow a background from nearby pixels that have one (Gaussian,
   BACKGROUND_L_KM), only when enough of them lie close by; otherwise they stay empty.
3. Anomaly: observed minus background at the pixels of the latest clear view, spread to the gaps
   by a normalised Gaussian on a ANOMALY_L_KM scale (optimal-interpolation style). Where no
   observation lies near enough, the anomaly is 0 (background only).
   Where none lies within ~3 km, a wider borrowing (BACKGROUND_WIDE_L_KM) fills the pixel as a
   separate, less certain class. Borrowed backgrounds within 0.6 km of shore get a validated shoreward offset (chlorophyll rises
   toward shore; without it they read 0.2-0.6 log10 low).
4. Estimate = background + anomaly, ONLY at water pixels in the satellite scope that have no
   observation in the latest clear view. Observed pixels are never written or changed.

Every estimated pixel gets a class (distance band x open coast / bay x own / borrowed background)
and the uncertainty grid holds the RMSE measured for that class in the validation.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, distance_transform_edt, gaussian_filter, label

TAU_DAYS = 20.0  # background memory (e-folding, days)
MIN_WEIGHT = 0.5  # own background needs at least about one valid view in the last ~2 weeks
BACKGROUND_L_KM = 1.0  # borrowing scale for pixels without their own background
BACKGROUND_MIN_SUPPORT = 0.05  # Gaussian weight share of nearby pixels with a background
# second tier, so the coast and the bays are filled even where the nearest recent views are a few
# km away (fog banks): a wider borrowing scale, its own (larger, measured) uncertainty class
BACKGROUND_WIDE_L_KM = 5.0
BACKGROUND_WIDE_MIN_SUPPORT = 1e-3
ANOMALY_L_KM = 3.0  # same-period anomaly scale (validated: 3 km)
ANOMALY_MIN_SUPPORT = 1e-3
BAY_OPEN_KM = 1.0  # water narrower than about 2 x this (harbour mouths, the Golden Gate) closes bays
KM_PER_DEG = 111.32

# distance bands (km from the nearest land cell) used for validation and uncertainty
BANDS = ((0.0, 0.6, "0-0.6 km"), (0.6, 3.0, "0.6-3 km"), (3.0, math.inf, "over 3 km"))


def cell_km(step: float, lat_mid: float) -> tuple[float, float]:
    """(north-south, east-west) size of a cell in km at lat_mid."""
    return step * KM_PER_DEG, step * KM_PER_DEG * math.cos(math.radians(lat_mid))


def gauss_avg(values: np.ndarray, valid: np.ndarray, l_km: float, dy_km: float, dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalised Gaussian average of `values` over `valid` cells: (average, support).
    support is the Gaussian weight share of valid cells around each cell (0..1)."""
    sig = (l_km / dy_km, l_km / dx_km)
    v = np.where(valid, values, 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    num = gaussian_filter(v, sig, mode="constant", truncate=3.0)
    den = gaussian_filter(w, sig, mode="constant", truncate=3.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return avg, den


# ---------------------------------------------------------------- background state
def update_background(b: np.ndarray, w: np.ndarray, days_since: float, day_log10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Advance the background by `days_since` days and add one day's valid log10 values.
    b: weighted mean (NaN where no weight), w: effective number of recent views."""
    w = w * math.exp(-max(days_since, 0.0) / TAU_DAYS)
    ok = np.isfinite(day_log10)
    b0 = np.where(np.isfinite(b), b, 0.0)
    nb = np.where(ok, (b0 * w + np.where(ok, day_log10, 0.0)) / (w + 1.0), b)
    nw = w + ok
    nb = np.where(nw > 0, nb, np.nan)
    return nb, nw


# ---------------------------------------------------------------- geometry
def distance_km(water: np.ndarray, step: float, lat_mid: float) -> np.ndarray:
    """Distance from each water cell to the nearest non-water cell, km (0 on land)."""
    dy, dx = cell_km(step, lat_mid)
    return distance_transform_edt(water, sampling=(dy, dx))


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= r * r


# Closing lines across the mouths of the named bays, as (lat, lon) polylines drawn land to land
# on the marine water mask (data/reference/marine_water_mask.json). The mask counts a cell as
# water when any part of it touches the sea, which widens mouths such as the Golden Gate (about
# 2.5 km on the mask) beyond what the morphological opening below can close, and its simplified
# outlines leave gaps in thin barrier spits (Humboldt Bay's South Spit, San Diego's Silver
# Strand); the lines follow those spits. Water cut off from the open sea by these lines is a bay
# whatever the width of its mouth. Checked against the mask by tests/test_derived.py.
BAY_MOUTHS: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = (
    ("San Francisco Bay (Golden Gate; with San Pablo and Suisun Bays)", ((37.800, -122.4788), (37.840, -122.4788))),
    ("Tomales Bay", ((38.2375, -122.9950), (38.2420, -122.9660))),
    ("Humboldt Bay (entrance and South Spit)", ((40.7638, -124.2265), (40.7510, -124.2345), (40.7313, -124.2488), (40.6925, -124.2600))),
    ("Morro Bay (entrance and sand spit)", ((35.3740, -120.8613), (35.3300, -120.8613))),
    ("Elkhorn Slough (Moss Landing)", ((36.8080, -121.7838), (36.8230, -121.7838))),
    ("Mission Bay", ((32.7540, -117.2405), (32.7775, -117.2405))),
    ("San Diego Bay (entrance)", ((32.6875, -117.2440), (32.6890, -117.2200))),
    ("San Diego Bay (Silver Strand)", ((32.6863, -117.1813), (32.6713, -117.1650), (32.6588, -117.1563), (32.6463, -117.1463),
                                       (32.6388, -117.1425), (32.6288, -117.1388), (32.6163, -117.1338), (32.6038, -117.1300))),
)


def mouth_cells(shape: tuple[int, int], lat_first: float, lon_first: float, step: float) -> np.ndarray:
    """Cells crossed by the BAY_MOUTHS closing lines (8-connected lines: they block 4-connected water)."""
    h, w = shape
    out = np.zeros(shape, dtype=bool)
    for _, pts in BAY_MOUTHS:
        for (la0, lo0), (la1, lo1) in zip(pts[:-1], pts[1:]):
            n = int(max(abs(la1 - la0), abs(lo1 - lo0)) / step * 4) + 2
            r = np.round((lat_first - np.linspace(la0, la1, n)) / step).astype(int)
            c = np.round((np.linspace(lo0, lo1, n) - lon_first) / step).astype(int)
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            out[r[ok], c[ok]] = True
    return out


def bay_mask(water: np.ndarray, step: float, lat_mid: float, origin: tuple[float, float] | None = None) -> np.ndarray:
    """Enclosed water (bays, estuaries, harbours, sloughs): water that is not part of the open
    coastal ocean. The open ocean is the water that stays connected to the domain's western
    (offshore) edge after a morphological opening of radius BAY_OPEN_KM, which cuts narrow
    mouths such as Humboldt Bay's entrance; it is then grown back by the same radius so the open
    coast's own nearshore strip stays 'open coast'. With `origin` (lat, lon of the first cell
    centre), water cut off from the western edge by the BAY_MOUTHS closing lines is a bay too
    (San Francisco Bay's mouth is too wide on the mask for the opening)."""
    dy, _ = cell_km(step, lat_mid)
    r = max(1, int(round(BAY_OPEN_KM / dy)))
    k = _disk(r)
    opened = binary_opening(water, structure=k)
    lab, _ = label(opened)
    edge = np.unique(lab[:, 0])
    edge = edge[edge > 0]
    open_sea = np.isin(lab, edge)
    open_sea = binary_dilation(open_sea, structure=_disk(r + 1)) & water
    bay = water & ~open_sea
    if origin is not None:
        cut = water & ~mouth_cells(water.shape, origin[0], origin[1], step)
        lab, _ = label(cut)  # 4-connected: a diagonal step across a closing line does not count
        edge = np.unique(lab[:, 0])
        edge = edge[edge > 0]
        bay |= cut & ~np.isin(lab, edge)
    return bay


# ---------------------------------------------------------------- estimate
BACKGROUND_TIERS = ("own", "borrowed", "borrowed_wide")
CLASS_KEYS: list[str] = []  # filled below: index (1-based code) -> "band|type|background"
for _b in BANDS:
    for _t in ("open", "bay"):
        for _g in BACKGROUND_TIERS:
            CLASS_KEYS.append(f"{_b[2]}|{_t}|{_g}")


def class_codes(dist: np.ndarray, bay: np.ndarray, tier: np.ndarray) -> np.ndarray:
    """Per-cell class code (1..len(CLASS_KEYS)); 0 = none. tier: 0 own, 1 borrowed, 2 wide, -1 none."""
    code = np.zeros(dist.shape, dtype=np.uint8)
    nt = len(BACKGROUND_TIERS)
    for bi, (lo, hi, _) in enumerate(BANDS):
        in_band = (dist > lo) & (dist <= hi) if lo > 0 else (dist <= hi)
        for ti, tsel in enumerate((~bay, bay)):
            for gi in range(nt):
                code[in_band & tsel & (tier == gi)] = 1 + bi * 2 * nt + ti * nt + gi
    return code


def estimate(
    obs_log10: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    fill_area: np.ndarray,
    dist: np.ndarray,
    bay: np.ndarray,
    dy_km: float,
    dx_km: float,
    nearshore_offset: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """(estimate log10, class code). The estimate is NaN wherever a pixel is observed, outside
    `fill_area`, or without enough recent nearby data. `nearshore_offset` ({"open", "bay"}: log10)
    is added to borrowed backgrounds within 0.6 km of shore (validated: chlorophyll rises toward
    shore, so a background borrowed from farther out reads low)."""
    own = np.isfinite(b) & (w >= MIN_WEIGHT)
    b_own = np.where(own, b, np.nan)
    borrowed, support = gauss_avg(b_own, own, BACKGROUND_L_KM, dy_km, dx_km)
    wide, wsupport = gauss_avg(b_own, own, BACKGROUND_WIDE_L_KM, dy_km, dx_km)
    near = dist <= BANDS[0][1]
    shoreward = np.where(near & bay, nearshore_offset["bay"], np.where(near, nearshore_offset["open"], 0.0))
    t1 = ~own & (support >= BACKGROUND_MIN_SUPPORT)
    t2 = ~own & ~t1 & (wsupport >= BACKGROUND_WIDE_MIN_SUPPORT)
    tier = np.where(own, 0, np.where(t1, 1, np.where(t2, 2, -1)))
    bg = np.where(own, b_own, np.where(t1, borrowed + shoreward, np.where(t2, wide + shoreward, np.nan)))
    obs = np.isfinite(obs_log10)
    an_ok = obs & own
    anom, an_sup = gauss_avg(np.where(an_ok, obs_log10 - b_own, 0.0), an_ok, ANOMALY_L_KM, dy_km, dx_km)
    anom = np.where(an_sup > ANOMALY_MIN_SUPPORT, anom, 0.0)
    target = fill_area & ~obs & np.isfinite(bg)
    est = np.where(target, bg + anom, np.nan)
    code = np.where(target, class_codes(dist, bay, tier), 0).astype(np.uint8)
    return est, code

