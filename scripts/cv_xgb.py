
#!/usr/bin/env python3
# scripts/cv_xgb.py
"""
Year-blocked cross-validation for the tuned XGBoost HAB regressor.
– Design matrix: 42 features (t, lag-1..3, roll-3, sin/cos DOY, optical indices).
– Folds: leave one calendar year out (2016-2020); 2021 remains untouched
– Outputs: per-year RMSE, MAE, R²  →  Models/cv_metrics.tsv
"""
import pathlib, yaml, numpy as np, xarray as xr, pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from _feature_utils import build_design_matrix 

# ─── config & paths ───────────────────────────────────────────────
root = pathlib.Path(__file__).resolve().parents[1]
cfg  = yaml.safe_load(open(root / "config.yaml"))
cube = pathlib.Path(cfg["data_root"]) / "HAB_cube_2016_2021.nc"

LAT_MIN, LAT_MAX = 32.0, 50.0
LON_MIN, LON_MAX = -125.0, -117.0
COAST_R      = int(cfg.get("coast_radius_cells", 4))
FORECAST_LAG = int(cfg.get("forecast_lag",       1))
PRED_VARS = [
    "sst","t2m","u10","v10","avg_sdswrf",
    "Kd_490","nflh","so","thetao","uo","vo","zos"
]

best_params = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.018858981562472215,
    "subsample": 0.6643226915701169,
    "colsample_bytree": 0.8496171077713075,
    "min_child_weight": 4.308082067248617,
    "gamma": 2.092433875822892,
    "reg_lambda": 0.06867791794984984,
    "random_state": 42,          # keep reproducibility
    "objective": "reg:squarederror"
}

out_tsv = root / "Models" / "cv_metrics.tsv"
out_tsv.parent.mkdir(exist_ok=True)

# ─── helper: build design matrix identical to train_baseline ──────
def build_design():
    ds = xr.open_dataset(cube, chunks={"time": 50})
    ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

    # coastal mask
    ocean = ~ds["chlor_a"].isel(time=0).isnull()
    d2l   = xr.apply_ufunc(distance_transform_edt, ~ocean,
                           input_core_dims=[("lat","lon")],
                           output_core_dims=[("lat","lon")], dask="allowed")
    coast = (ocean & (d2l <= COAST_R)).broadcast_like(ds["chlor_a"])
    coast = coast.isel(time=slice(None, -FORECAST_LAG-3))


    X_all, y = build_design_matrix(ds, PRED_VARS, FORECAST_LAG)

    # stack to samples
    def stack(da):
        extra=[d for d in da.dims if d not in ("time","lat","lon")]
        return (da.transpose("time","lat","lon",*extra)
                 .stack(sample=("time","lat","lon"))
                 .transpose("sample",*extra))

    Xst, yst = stack(X_all), stack(y)
    mask = (coast.stack(sample=("time","lat","lon")).values &
            ~np.isnan(Xst).any("var") & np.isfinite(yst))

    X_np  = Xst.values[mask].astype("float32")
    y_np  = yst.values[mask].astype("float32")
    times = Xst["time"].values[mask]
    return X_np, y_np, times

X, y, t = build_design()

# ─── cross-validation loop ────────────────────────────────────────
metrics = []
for yr in range(2016, 2021):          # leave 2021 for final hold-out
    train = t.astype("datetime64[Y]") != np.datetime64(f"{yr}")
    val   = ~train

    model = XGBRegressor(**best_params)
    model.fit(X[train], y[train])
    pred = model.predict(X[val])

    rmse = np.sqrt(mean_squared_error(y[val], pred))
    mae  = mean_absolute_error(y[val], pred)
    r2   = r2_score(y[val], pred)
    metrics.append({"year": yr, "rmse": rmse, "mae": mae, "r2": r2})
    print(f"Fold {yr}: RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}")

df = pd.DataFrame(metrics)
df.to_csv(out_tsv, sep="\t", index=False)

print("\n── summary ──")
print(df.describe()[["rmse","mae","r2"]])
print(f"\n✓ per-year metrics saved → {out_tsv}")

'''
Run Instructions:

micromamba activate habs
cd ~/Desktop/habs-forecast
python scripts/cv_xgb.py
'''