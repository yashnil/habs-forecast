# scripts/_feature_utils.py
import numpy as np
import xarray as xr

# ------------------------------------------------------------------
# Helper: add sin/cos of day-of-year
# ------------------------------------------------------------------
def add_seasonal_coords(ds: xr.Dataset) -> xr.Dataset:
    """Adds `sin_doy` and `cos_doy` (0-mean annual cycle) to *ds*."""
    doy = xr.DataArray(ds["time"].dt.dayofyear,
                       coords={"time": ds["time"]},
                       dims="time")
    ds["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    ds["cos_doy"] = np.cos(2 * np.pi * doy / 365)
    return ds

# ------------------------------------------------------------------
# Main feature-builder
# ------------------------------------------------------------------
def build_design_matrix(
    ds: xr.Dataset,
    pred_vars: list[str],
    forecast_lag: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Returns
    -------
    X_all : xr.DataArray  (dims: time, lat, lon, var)
    y     : xr.DataArray  (dims: time, lat, lon)

    Features included
    -----------------
    • 12 current predictors (t)
    • 12 × lag-1, lag-2, lag-3                     → 36
    • 12 three-composite trailing means            → 12
    • sin_doy, cos_doy                              → 2
    • nflh_anom (30-day), flh_kd                    → 2
    • curl_uv, dist_river_km, log1p_dist_river      → 3   ← NEW
      --------------------------------------------------------------
      total = 12 + 36 + 12 + 2 + 2 + 3 = 67 columns
    """
    # ── add seasonal sine/cosine ──────────────────────────────────
    ds = add_seasonal_coords(ds)

    # ── merge derivatives group if it exists in the cube ──────────
    #    (silently skip if user hasn’t run add_derivatives.py yet)
    if "curl_uv" not in ds.data_vars:
        try:
            deriv = xr.open_dataset(ds.encoding["source"], group="derivatives")
            ds = xr.merge([ds, deriv])
        except Exception:
            pass

    # ── core predictor snapshots (t) ──────────────────────────────
    X_curr = ds[pred_vars]

    # helper for lags
    def _lag(k: int):
        return (
            X_curr.shift(time=+k)
                  .to_array("var")
                  .assign_coords(var=[f"{v}_lag{k}" for v in pred_vars])
        )

    X_lag1, X_lag2, X_lag3 = (_lag(k) for k in (1, 2, 3))

    # trailing 3-composite mean
    roll3 = (
        X_curr.rolling(time=3, min_periods=1)
              .mean()
              .to_array("var")
              .assign_coords(var=[f"{v}_roll3" for v in pred_vars])
    )

    # seasonality (broadcast to lat/lon)
    season = xr.concat([ds["sin_doy"], ds["cos_doy"]], dim="var")
    season = (
        season.assign_coords(var=["sin_doy", "cos_doy"])
              .broadcast_like(X_curr.isel(time=0))
    )

    # optical indices
    nflh_anom = ds["nflh"] - ds["nflh"].rolling(time=30, min_periods=15).mean()
    flh_kd    = xr.where(ds["Kd_490"] > 0, ds["nflh"] / ds["Kd_490"], np.nan)
    optics = (
        xr.concat([nflh_anom, flh_kd], dim="var")
          .assign_coords(var=["nflh_anom", "flh_kd"])
          .broadcast_like(X_curr.isel(time=0))
    )

    # derivatives block (only if variables are present)
    if {"curl_uv", "dist_river_km", "log1p_dist_river"} <= set(ds.data_vars):
        derivs = (
            ds[["curl_uv", "dist_river_km", "log1p_dist_river"]]
              .to_array("var")
              .broadcast_like(X_curr.isel(time=0))
        )
    else:
        # create a 0-length placeholder *with the same non-var dims*
        derivs = xr.DataArray(
            np.empty(
                (0, ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"]),
                dtype=np.float32,
            ),
            dims   = ["var", "time", "lat", "lon"],
            coords = {
                "var" : [],
                "time": ds["time"],
                "lat" : ds["lat"],
                "lon" : ds["lon"],
            },
        )

    # ── concatenate all feature blocks ────────────────────────────
    X_all = xr.concat(
        [
            X_curr.to_array("var"),
            X_lag1, X_lag2, X_lag3,
            roll3,
            season,
            optics,
            derivs,      # ← NEW block appended
        ],
        dim="var"
    )

    # ── target shifted forward by forecast_lag ────────────────────
    y = ds["log_chl"].shift(time=-forecast_lag)

    # ── drop final (forecast_lag + 3) composites to keep shapes aligned
    valid_slice = slice(None, -forecast_lag - 3)
    return X_all.isel(time=valid_slice), y.isel(time=valid_slice)