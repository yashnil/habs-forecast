#!/usr/bin/env python
"""
California Coastal HABs ML - Data Freeze v1.0
Build an analysis-ready NetCDF with mask, target engineering,
monthly climatology & anomalies, rolling stats, lags, and physics-derived drivers.
"""

import os, json, argparse, pathlib
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from dask.diagnostics import ProgressBar

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
EARTH_RADIUS_M = 6371000.0  # mean radius

def load_params(path):
    with open(path, "r") as f:
        P = yaml.safe_load(f)
    # normalize booleans / None
    if isinstance(P.get("drop_low_coverage"), str) and P["drop_low_coverage"].lower() in ("none","null","na",""):
        P["drop_low_coverage"] = None
    return P

def decode_time_if_needed(ds):
    # convert "days since ..." if not datetime64
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        ds = xr.decode_cf(ds)
    return ds

def build_masks(ds, method):
    """
    Returns (static_ocean_mask, valid_mask_t)
    static_ocean_mask: 1 where this grid cell is ever valid (across time)
    valid_mask_t     : per-time valid mask (finite log_chl)
    """
    if method != "non_nan_log_chl":
        raise ValueError(f"Unknown mask method: {method}")
    finite_t = np.isfinite(ds["log_chl"])  # (time,lat,lon)
    static_mask = finite_t.any("time").astype("uint8")   # (lat,lon)
    valid_mask_t = finite_t.astype("uint8")              # (time,lat,lon)
    return static_mask, valid_mask_t

def add_chl_linear(ds, floor):
    chl = np.exp(ds["log_chl"])  # includes floor
    chl_af = (chl - floor).clip(min=0.0)
    ds["chl_lin"] = chl_af.astype("float32")
    ds["chl_lin"].attrs.update(long_name="chlorophyll-a above detection floor", units="mg m-3", note=f"exp(log_chl)-{floor}")
    floored = (chl <= (floor + 1e-9)).astype("uint8")
    ds["chl_is_floored"] = xr.DataArray(floored, dims=("time","lat","lon"), attrs={"long_name":"pixel at detection floor prior to log"})
    return ds

def add_monthly_anom(ds):
    chl_lin_masked = ds["chl_lin"].where(ds["valid_mask"] == 1)
    clim = chl_lin_masked.groupby("time.month").mean("time", skipna=True)
    anom = chl_lin_masked.groupby("time.month") - clim
    ds["chl_anom_monthly"] = anom.astype("float32")
    ds["chl_anom_monthly"].attrs.update(long_name="chlorophyll-a monthly anomaly", units="mg m-3")
    return ds

def add_rolling(ds, windows):
    # windows in timesteps; each timestep = 8d
    for w in windows:
        win_days = 8 * w
        roll = ds["chl_lin"].rolling(time=w, min_periods=1)
        ds[f"chl_roll{win_days}d_mean"] = roll.mean().astype("float32")
        ds[f"chl_roll{win_days}d_std"]  = roll.std().astype("float32")

    # drivers
    skip_vars = {
        "log_chl","chl_lin","chl_is_floored","chl_anom_monthly",
        "valid_mask","ocean_mask_static","valid_frac"
    }
    for dv in ds.data_vars:
        if dv in skip_vars: 
            continue
        if ds[dv].dims != ("time","lat","lon"):
            continue
        for w in windows:
            win_days = 8 * w
            roll = ds[dv].rolling(time=w, min_periods=1)
            ds[f"{dv}_roll{win_days}d_mean"] = roll.mean().astype("float32")
            ds[f"{dv}_roll{win_days}d_std"]  = roll.std().astype("float32")
    return ds

def add_lags(ds, lags):
    lag_vars = [v for v in ds.data_vars if ds[v].dims == ("time","lat","lon")]
    for lv in lag_vars:
        for lag in lags:
            lag_days = 8 * lag
            ds[f"{lv}_lag{lag_days}d"] = ds[lv].shift(time=lag).astype(ds[lv].dtype)
    return ds

def _grid_metrics(lat, lon):
    # lat, lon 1-D arrays (deg)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    dlat = np.gradient(lat_rad)  # size nlat
    dlon = np.gradient(lon_rad)  # size nlon
    dy = EARTH_RADIUS_M * dlat  # m
    # broadcast cos(lat) over lon
    coslat = np.cos(lat_rad)[:, None]
    dx = EARTH_RADIUS_M * coslat * dlon  # broadcast to (lat, lon)
    # convert to DataArrays
    dy_da = xr.DataArray(dy, dims=("lat",), coords={"lat":lat})
    dx_da = xr.DataArray(np.broadcast_to(dlon, (lat.size, lon.size)), dims=("lat","lon"), coords={"lat":lat,"lon":lon})
    # scale dx by coslat*R
    dx_da = EARTH_RADIUS_M * np.cos(lat_rad)[:,None] * dx_da
    return dx_da, dy_da

def add_physics(ds, river_decay_km=50.0):
    rho_air = 1.225
    Cd = 1.3e-3
    u10 = ds["u10"]; v10 = ds["v10"]
    wind_speed = np.sqrt(u10**2 + v10**2)
    tau_x = rho_air * Cd * wind_speed * u10
    tau_y = rho_air * Cd * wind_speed * v10
    ds["tau_x"] = tau_x.astype("float32")
    ds["tau_y"] = tau_y.astype("float32")
    ds["tau_mag"] = np.sqrt(tau_x**2 + tau_y**2).astype("float32")
    ds["wind_speed"] = wind_speed.astype("float32")
    # crude upwelling proxy
    ds["upwelling_wind_proxy"] = (-v10).clip(min=0) * ds["tau_mag"]

    # currents
    if "uo" in ds and "vo" in ds:
        uo = ds["uo"]; vo = ds["vo"]
        ds["cur_speed"] = np.sqrt(uo**2 + vo**2).astype("float32")
        ds["cur_dir_rad"] = np.arctan2(vo, uo).astype("float32")

        # divergence & vorticity (approx; treat deg as linear scaled by metric factors)
        dx, dy = _grid_metrics(ds["lat"].values, ds["lon"].values)
        du_dx = uo.differentiate("lon") / dx
        dv_dy = vo.differentiate("lat") / dy
        ds["cur_div"] = (du_dx + dv_dy).astype("float32")
        dv_dx = vo.differentiate("lon") / dx
        du_dy = uo.differentiate("lat") / dy
        ds["cur_vort"] = (dv_dx - du_dy).astype("float32")

    # SSH gradient
    if "zos" in ds:
        dx, dy = _grid_metrics(ds["lat"].values, ds["lon"].values)
        dssh_dx = ds["zos"].differentiate("lon") / dx
        dssh_dy = ds["zos"].differentiate("lat") / dy
        ds["ssh_grad_mag"] = np.sqrt(dssh_dx**2 + dssh_dy**2).astype("float32")

    # river influence
    if "dist_river_km" in ds:
        ds["river_rank"] = np.exp(-ds["dist_river_km"] / river_decay_km).astype("float32")

    return ds

def add_valid_fraction(ds):
    """
    Compute fraction of the *expected* ocean footprint that is valid each timestep.
    expected ocean footprint = ocean_mask_static == 1
    """
    ocean = ds["ocean_mask_static"]
    ocean_ct = ocean.sum(("lat","lon"))
    # broadcast ocean over time
    ocean_b = ocean.broadcast_like(ds["valid_mask"])
    valid_now = ds["valid_mask"] * ocean_b
    vf = valid_now.sum(("lat","lon")) / ocean_ct
    ds["valid_frac"] = vf
    ds["valid_frac"].attrs["comment"] = "fraction of static ocean footprint with valid data"
    return ds

def apply_drop_low(ds, thresh):
    if thresh is None:
        return ds
    keep = (ds["valid_frac"] >= (1.0 - thresh)).compute().values  # numpy bool array
    drop_n = int((~keep).sum())
    if drop_n == ds.sizes["time"]:
        print("[data_freeze][warn] All timesteps below coverage threshold; keeping all instead.")
        return ds
    if drop_n:
        print(f"[data_freeze] Dropping {drop_n} timesteps (<{(1-thresh)*100:.0f}% coverage).")
        ds = ds.isel(time=np.where(keep)[0])
    return ds


def write_outputs(ds, out_path, params):
    # ---- 3a.  NaN audit ----
    print("\n[data_freeze] NaN summary (valid ocean only)")
    ocean_b = ds["ocean_mask_static"].broadcast_like(ds["valid_mask"])
    for v in ds.data_vars:
        da = ds[v].where(ocean_b == 1)
        nan_pct = float(da.isnull().mean().compute()) * 100
        print(f"    {v:>25s} : {nan_pct:5.1f}% NaN")

    # ---- 3b.  eager compute with progress bar ----
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds = ds.chunk({"time": -1})              # one chunk along time
    print("\n[data_freeze] Computing Dask graph …")
    with ProgressBar():
        ds = ds.compute()

    # ---- 3c.  write NetCDF ----
    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    print(f"[data_freeze] Writing NetCDF → {out_path}")
    ds.to_netcdf(out_path, encoding=enc)

    # ---- 3d.  metadata JSON ----
    meta_path = out_path + ".json"
    meta = dict(params)
    meta.update({
        "n_time": int(ds.sizes["time"]),
        "n_lat" : int(ds.sizes["lat"]),
        "n_lon" : int(ds.sizes["lon"]),
        "variables": sorted(list(ds.data_vars))
    })
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[data_freeze] Wrote metadata JSON → {meta_path}\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML with data_freeze_v1 parameters")
    args = ap.parse_args()
    P = load_params(args.config)

    ds = xr.open_dataset(P["source_path"], chunks={"time": P.get("chunk_time",64)})
    ds = decode_time_if_needed(ds)

    static_mask, valid_mask_t = build_masks(ds, P["coastal_mask_method"])
    ds["valid_mask"] = valid_mask_t
    ds["ocean_mask_static"] = static_mask
    ds["ocean_mask_static"].attrs.update(
        long_name="pixels ever valid across record",
        comment="static coastal ocean footprint derived from finite log_chl"
    )




    ds = add_chl_linear(ds, P["floor_mg_m3"])
    ds = add_monthly_anom(ds)
    ds = add_rolling(ds, P["roll_windows"])
    ds = add_lags(ds, P["lag_steps"])
    if P.get("include_physics_derived", True):
        ds = add_physics(ds, river_decay_km=P.get("river_decay_km",50.0))
    ds = add_valid_fraction(ds)
    ds = apply_drop_low(ds, P["drop_low_coverage"])

    # Re-chunk for efficient write
    ds = ds.chunk({"time": P.get("chunk_time", 64)})

    # QC print (compute on masked ocean only to avoid NaNs)
    chl_valid = ds["chl_lin"].where(ds["valid_mask"] == 1)
    chl_min = float(chl_valid.min(skipna=True).compute())
    chl_max = float(chl_valid.max(skipna=True).compute())
    valid_mean = float(ds["valid_frac"].mean().compute())
    n_time = ds.sizes["time"]

    print("[data_freeze] QC summary")
    print(f"  chl_lin min: {chl_min:.4f} mg m-3")
    print(f"  chl_lin max: {chl_max:.4f} mg m-3")
    print(f"  mean valid fraction: {valid_mean:.3f}")
    print(f"  timesteps: {n_time}")


    write_outputs(ds, P["output_path"], P)

if __name__ == "__main__":
    main()
