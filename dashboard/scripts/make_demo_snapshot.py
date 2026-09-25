#!/usr/bin/env python3
"""
Create a demo overlay + manifest for the California nearshore strip (no NetCDF required).

Uses Natural Earth land polygons (via GeoPandas) by default so coloring stays on the ocean.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_DASH = Path(__file__).resolve().parents[1]
if str(_DASH) not in sys.path:
    sys.path.insert(0, str(_DASH))

import snapshot_utils as su  # noqa: E402
from ocean_mask import mask_rgba_ocean_only, ocean_mask_grid, visible_geographic_bounds  # noqa: E402
from regional import compute_regional_algae  # noqa: E402


def synthetic_logchl(La: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Smooth, broad-scale demo field (ln-like scale) — not real ocean data."""
    base = -0.42 + 0.018 * (La - 36.2)
    gyre = 0.38 * np.exp(-((La - 36.4) ** 2 / 2.4 + (L + 122.2) ** 2 / 3.0))
    along = 0.07 * np.sin(0.2 * (L + 125.0))
    return (base + gyre + along).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None, help="Default: dashboard/data next to this repo")
    ap.add_argument(
        "--mask-fallback",
        action="store_true",
        help="Use fast polyline mask (offline / CI) instead of Natural Earth via GeoPandas.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir) if args.outdir else _DASH / "data"
    mm = "fallback" if args.mask_fallback else "geopandas"

    lon = np.arange(-126.0, -117.0, 0.04, dtype=np.float64)
    lat = np.arange(32.0, 42.5, 0.04, dtype=np.float64)
    La, L = np.meshgrid(lat, lon, indexing="ij")
    z = synthetic_logchl(La, L)
    z = np.flipud(z)

    lat_asc = np.sort(lat)
    lon_asc = np.sort(lon)
    ocean = ocean_mask_grid(lat_asc, lon_asc, method=mm)
    regional = compute_regional_algae(
        z,
        ocean,
        lat_north=float(lat.max()),
        lat_south=float(lat.min()),
    )

    finite = z[np.isfinite(z)]
    vmin = float(np.nanpercentile(finite, 5))
    vmax = float(np.nanpercentile(finite, 95))
    if vmax <= vmin:
        vmax = vmin + 0.01

    cmap = "YlGnBu"
    rgba = su.logchl_to_rgba(z, vmin=vmin, vmax=vmax, cmap_name=cmap)
    rgba = mask_rgba_ocean_only(
        rgba,
        lat_ascending=lat_asc,
        lon_ascending=lon_asc,
        mask_method=mm,
    )
    south_t, west_t, north_t, east_t = visible_geographic_bounds(
        rgba,
        lat_north=float(lat.max()),
        lat_south=float(lat.min()),
        lon_west=float(lon.min()),
        lon_east=float(lon.max()),
    )
    overlay_path = outdir / "overlay.png"
    manifest_path = outdir / "snapshot.json"
    su.write_overlay_png(rgba, overlay_path)
    land_mask = "fallback_polyline" if mm == "fallback" else "natural_earth_geopandas"
    manifest = su.build_manifest(
        bounds=(south_t, west_t, north_t, east_t),
        time_label="demo-static",
        variable="log_chl_demo",
        data_source="synthetic_demo",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        notes="Synthetic field for UI testing only — replace via export_map_snapshot.py.",
        regional_algae=regional,
        land_mask=land_mask,
    )
    su.write_manifest(manifest, manifest_path)
    print(f"Wrote {overlay_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
