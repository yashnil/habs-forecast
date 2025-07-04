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
def build_design_matrix(ds: xr.Dataset,
                        pred_vars: list[str],
                        forecast_lag: int) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Returns
    -------
    X_all : xr.DataArray  (dims: time, lat, lon, var)
    y     : xr.DataArray  (dims: time, lat, lon)

    Features included
    -----------------
    • Current predictors (t)
    • Lag-1 .. Lag-3      (t-1, t-2, t-3)
    • Trailing 3-composite rolling mean of each predictor
    • sin(day-of-year), cos(day-of-year)
    • Optical indices: nflh anomaly (90-day) & nflh / Kd_490
    Target
    ------
    • y = log10(chlor_a) shifted by –forecast_lag  (≈ t + 1 composite)
    """
    ds = add_seasonal_coords(ds)

    # ── core predictor snapshots ──────────────────────────────────
    X_curr = ds[pred_vars]

    def _lag(k: int):
        return (X_curr.shift(time=+k)
                      .to_array("var")
                      .assign_coords(var=[f"{v}_lag{k}" for v in pred_vars]))

    X_lag1 = _lag(1)
    X_lag2 = _lag(2)
    X_lag3 = _lag(3)

    # ── 3-composite trailing mean ─────────────────────────────────
    roll3 = (X_curr
             .rolling(time=3, min_periods=1)
             .mean()
             .to_array("var")
             .assign_coords(var=[f"{v}_roll3" for v in pred_vars]))

    # ── seasonality  (broadcast to lat/lon) ───────────────────────
    season = xr.concat([ds["sin_doy"], ds["cos_doy"]], dim="var")
    season = (season.assign_coords(var=["sin_doy", "cos_doy"])
                    .broadcast_like(X_curr.isel(time=0)))

    # ── optical indices ───────────────────────────────────────────
    nflh_anom = ds["nflh"] - ds["nflh"].rolling(time=30, min_periods=15).mean()
    # flh_kd    = ds["nflh"] / ds["Kd_490"]
    flh_kd = xr.where(ds["Kd_490"] > 0,
                  ds["nflh"] / ds["Kd_490"],
                  np.nan)
    optics = xr.concat([nflh_anom, flh_kd], dim="var") \
               .assign_coords(var=["nflh_anom", "flh_kd"]) \
               .broadcast_like(X_curr.isel(time=0))

    # ── concatenate all feature blocks ────────────────────────────
    X_all = xr.concat(
        [
            X_curr.to_array("var"),  #  t
            X_lag1,                  #  t-1
            X_lag2,                  #  t-2
            X_lag3,                  #  t-3
            roll3,                   #  3-comp trailing mean
            season,                  #  sin/cos
            optics                   #  optical indices
        ],
        dim="var"
    )

    # ── target (shift forward) ────────────────────────────────────
    y = ds["log_chl"].shift(time=-forecast_lag)

    # ── trim last   forecast_lag + 3   composites ─────────────────
    valid_slice = slice(None, -forecast_lag - 3)
    return X_all.isel(time=valid_slice), y.isel(time=valid_slice)
