import xarray as xr
import numpy as np
import pathlib

# ------------------------------------------------------------------
# 0. paths & config
# ------------------------------------------------------------------
INFILE  = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_freeze_v1.nc")
OUTFILE = INFILE.with_name("HAB_convLSTM_core_v1.nc")   # change as you like
CHUNK_T = 64                                            # 1-yr chunks (~64*8-d)
COMPRESS = dict(zlib=True, complevel=4)

# ------------------------------------------------------------------
# 1. open lazily
# ------------------------------------------------------------------
ds = xr.open_dataset(INFILE, chunks={"time": CHUNK_T})
ds = ds.chunk({"time": CHUNK_T}).copy()

# ------------------------------------------------------------------
# 2. define the canonical variable groups
# ------------------------------------------------------------------
SATELLITE = ["log_chl", "Kd_490", "nflh"]
METEO     = ["u10","v10","wind_speed","tau_mag","avg_sdswrf","tp","t2m","d2m"]
OCEAN     = ["uo","vo","cur_speed","cur_div","cur_vort","zos","ssh_grad_mag","so","thetao"]
DERIVED   = ["chl_anom_monthly",
             "chl_roll24d_mean","chl_roll24d_std",
             "chl_roll40d_mean","chl_roll40d_std"]
STATIC    = ["river_rank","dist_river_km","ocean_mask_static"]
ALL_VARS  = SATELLITE + METEO + OCEAN + DERIVED + STATIC

# ------------------------------------------------------------------
# 3. helper builders for missing “value-added” fields
# ------------------------------------------------------------------
def ensure_wind_speed(ds):
    if "wind_speed" not in ds and {"u10","v10"}.issubset(ds):
        ds["wind_speed"] = np.hypot(ds["u10"], ds["v10"])
    return ds

def ensure_tau_mag(ds):
    if "tau_mag" not in ds and {"tau_x","tau_y"}.issubset(ds):
        ds["tau_mag"] = np.hypot(ds["tau_x"], ds["tau_y"])
    return ds

def ensure_cur_speed(ds):
    if "cur_speed" not in ds and {"uo","vo"}.issubset(ds):
        ds["cur_speed"] = np.hypot(ds["uo"], ds["vo"])
    return ds

def ensure_cur_div_vort(ds):
    if {"cur_div","cur_vort"}.issubset(ds):
        return ds  # already present
    if {"uo","vo"}.issubset(ds) and {"lat","lon"}.issubset(ds.coords):
        # crude 1st-order centred differences in degrees → m later if wanted
        dy = np.gradient(ds["lat"])
        dx = np.gradient(ds["lon"])
        u  = ds["uo"]; v = ds["vo"]
        dv_dy = (v.diff("lat")/dy[:-1]).pad(lat=(1,0))
        du_dx = (u.diff("lon")/dx[:-1]).pad(lon=(1,0))
        ds["cur_div"]  = du_dx + dv_dy
        ds["cur_vort"] = dv_dy - du_dx
    return ds

def ensure_ssh_grad_mag(ds):
    if "ssh_grad_mag" not in ds and "zos" in ds:
        dη_dy = ds["zos"].differentiate("lat", edge_order=2)
        dη_dx = ds["zos"].differentiate("lon", edge_order=2)
        ds["ssh_grad_mag"] = np.hypot(dη_dx, dη_dy)
    return ds

def ensure_river_rank(ds):
    if "river_rank" not in ds and "dist_river_km" in ds:
        # 1/exp(distance) → rank 1 at river mouth, ~0 offshore
        ds["river_rank"] = xr.apply_ufunc(lambda d: np.exp(-d/10.), ds["dist_river_km"])
    return ds

# apply builders
for fn in (ensure_wind_speed, ensure_tau_mag, ensure_cur_speed,
           ensure_cur_div_vort, ensure_ssh_grad_mag, ensure_river_rank):
    ds = fn(ds)

# ------------------------------------------------------------------
# 4. sanity-check availability & trim
# ------------------------------------------------------------------
missing = [v for v in ALL_VARS if v not in ds]
if missing:
    print("⚠️  Warning: missing vars →", missing)

keep_vars = [v for v in ALL_VARS if v in ds]
ds_core = ds[keep_vars + ["lat","lon","time"]]   # keep coords

# ------------------------------------------------------------------
# 5. write out compressed NetCDF
# ------------------------------------------------------------------
encoding = {v: COMPRESS for v in keep_vars}
ds_core.to_netcdf(OUTFILE, format="NETCDF4", engine="netcdf4", encoding=encoding)

print(f"✅  Wrote trimmed dataset → {OUTFILE.relative_to(INFILE.parent)} "
      f"({len(keep_vars)} variables; original had {len(ds.data_vars)})")
