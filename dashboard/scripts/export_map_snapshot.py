#!/usr/bin/env python3
"""
Export a single-time-slice field from NetCDF to a Folium-friendly overlay PNG + manifest.

Typical use after model export or on the observation cube:

  python dashboard/scripts/export_map_snapshot.py \\
    --nc /path/to/HAB_convLSTM_core_v1_clean.nc \\
    --var log_chl \\
    --time -1 \\
    --outdir dashboard/data

Or for predictions:

  python dashboard/scripts/export_map_snapshot.py \\
    --nc /path/to/pinn__best.nc \\
    --var log_chl_pred \\
    --time -1 \\
    --outdir dashboard/data \\
    --source pinn_forecast
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_DASH = Path(__file__).resolve().parents[1]
if str(_DASH) not in sys.path:
    sys.path.insert(0, str(_DASH))

import snapshot_utils as su  # noqa: E402
from ocean_mask import mask_rgba_ocean_only, ocean_mask_grid, visible_geographic_bounds  # noqa: E402
from regional import compute_regional_algae  # noqa: E402


def _pick_var(ds: xr.Dataset, var: str | None) -> str:
    if var and var in ds:
        return var
    for cand in ("log_chl_pred", "log_chl", "chl_log", "chl_ln"):
        if cand in ds:
            return cand
    raise ValueError(f"No known chlorophyll variable in dataset; got {list(ds.data_vars)}")


def _canonicalize(da: xr.DataArray) -> xr.DataArray:
    dims = set(da.dims)
    if "lat" not in dims and "latitude" in dims:
        da = da.rename({"latitude": "lat"})
    if "lon" not in dims and "longitude" in dims:
        da = da.rename({"longitude": "lon"})
    return da.transpose("lat", "lon", missing_dims="ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description="NetCDF slice → dashboard overlay + manifest")
    ap.add_argument("--nc", required=True, help="Path to NetCDF")
    ap.add_argument("--var", default=None, help="Variable name (default: auto-detect)")
    ap.add_argument("--time", default="-1", help="Time index (int) or ISO date string; -1 = last")
    ap.add_argument("--outdir", required=True, help="Output directory (e.g. dashboard/data)")
    ap.add_argument(
        "--source",
        default="netcdf",
        help="Label stored in manifest data_source (e.g. observations, pinn_forecast)",
    )
    ap.add_argument("--vmin", type=float, default=None, help="Color scale min (default: p2)")
    ap.add_argument("--vmax", type=float, default=None, help="Color scale max (default: p98)")
    ap.add_argument("--cmap", default="YlGnBu", help="Matplotlib colormap name")
    ap.add_argument(
        "--composite-days",
        type=int,
        default=8,
        help="Metadata only: typical composite length for your dataset (shown to end users).",
    )
    ap.add_argument(
        "--mask-fallback",
        action="store_true",
        help="Use fast polyline coast mask instead of Natural Earth land polygons (offline / CI).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ds = xr.open_dataset(args.nc)
    vname = _pick_var(ds, args.var)
    da = _canonicalize(ds[vname])

    if "time" in da.dims:
        tsel = args.time.strip()
        if tsel == "-1":
            da2 = da.isel(time=-1)
            tlabel = str(da2.time.values)
        else:
            try:
                idx = int(tsel)
                da2 = da.isel(time=idx)
                tlabel = str(da2.time.values)
            except ValueError:
                da2 = da.sel(time=np.datetime64(tsel), method="nearest")
                tlabel = str(da2.time.values)
    else:
        da2 = da
        tlabel = None

    z = np.asarray(da2.values, dtype=np.float64)
    lat = np.asarray(da2["lat"].values, dtype=np.float64)
    lon = np.asarray(da2["lon"].values, dtype=np.float64)
    if lat.ndim == 1 and lon.ndim == 1:
        lat_north = float(np.max(lat))
        lat_south = float(np.min(lat))
        lon_west = float(np.min(lon))
        lon_east = float(np.max(lon))
        # Folium ImageOverlay: image row 0 = north. If lat increases south→north, flip rows.
        if lat[0] < lat[-1]:
            z = np.flipud(z)
    else:
        raise ValueError("Expected 1D lat/lon coordinates")

    finite = z[np.isfinite(z)]
    if finite.size == 0:
        raise ValueError("No finite values in slice")

    vmin = args.vmin if args.vmin is not None else float(np.nanpercentile(finite, 2))
    vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(finite, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    lat_asc = np.sort(lat)
    lon_asc = np.sort(lon)
    mm = "fallback" if args.mask_fallback else "geopandas"
    ocean = ocean_mask_grid(lat_asc, lon_asc, method=mm)
    regional = compute_regional_algae(
        z,
        ocean,
        lat_north=lat_north,
        lat_south=lat_south,
    )

    rgba = su.logchl_to_rgba(z, vmin=vmin, vmax=vmax, cmap_name=args.cmap)
    rgba = mask_rgba_ocean_only(
        rgba,
        lat_ascending=lat_asc,
        lon_ascending=lon_asc,
        mask_method=mm,
    )
    south, west, north, east = visible_geographic_bounds(
        rgba,
        lat_north=lat_north,
        lat_south=lat_south,
        lon_west=lon_west,
        lon_east=lon_east,
    )
    overlay_path = outdir / "overlay.png"
    manifest_path = outdir / "snapshot.json"

    su.write_overlay_png(rgba, overlay_path)
    land_mask = "fallback_polyline" if mm == "fallback" else "natural_earth_geopandas"
    manifest = su.build_manifest(
        bounds=(south, west, north, east),
        time_label=tlabel,
        variable=vname,
        data_source=args.source,
        vmin=vmin,
        vmax=vmax,
        cmap=args.cmap,
        notes="Exported from project NetCDF; not a regulatory product.",
        composite_days_hint=args.composite_days,
        regional_algae=regional,
        land_mask=land_mask,
    )
    su.write_manifest(manifest, manifest_path)
    print(f"Wrote {overlay_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
