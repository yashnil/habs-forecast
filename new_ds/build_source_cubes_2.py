#!/usr/bin/env python3
"""
build_source_cubes.py
──────────────────────────────────────────────────────────────────────────────
Create *separate* 8‑day NetCDF cubes for MODIS, ERA‑5 and CMEMS
(2003‑01‑01 → 2021‑06‑30, 851 composites).  No spatial re‑gridding yet.

Outputs  →  .../Data/Finalized/
            modis_8day_native.nc
            era5_8day_native.nc
            cmems_8day_native.nc
"""
# ── std / third‑party ──────────────────────────────────────────────────────
import pathlib, re, datetime as dt, numpy as np, xarray as xr, pandas as pd
from align_utils import to_datetime                     # helper you already have

# ── paths ──────────────────────────────────────────────────────────────────
ROOT    = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
P_MODIS = ROOT / "Processed" / "modis_l3m"
P_ERA5  = ROOT / "Data" / "era5"
P_CMEMS = ROOT / "Data" / "newcmems"
P_OUT   = ROOT / "Data" / "Finalized";  P_OUT.mkdir(parents=True, exist_ok=True)

WIN_START, WIN_END = np.datetime64("2003-01-01"), np.datetime64("2021-06-30")

# ── helper: calendar‑reset 8‑day composites ───────────────────────────────
def eightday_calendar(da, how="mean"):
    doy    = da.time.dt.dayofyear
    offset = ((doy - 1) % 8).astype("timedelta64[D]").data
    blk_t  = (da.time.data - offset).astype("datetime64[ns]")
    g      = da.assign_coords(block_time=("time", blk_t)).groupby("block_time")
    fn     = g.mean if how == "mean" else g.sum
    return fn("time").rename({"block_time": "time"}).sortby("time")

# ════════════════════════════════════════════════════════════════════════════
# 1)  M O D I S
# ════════════════════════════════════════════════════════════════════════════
def build_modis() -> xr.Dataset:
    products = {
        "chlor_a": ("chlorophyll", "chlor_a"),
        "Kd_490" : ("kd490",       "Kd_490"),
        "nflh"   : ("nFLH",        "nflh"),
    }
    date_re, rasters = re.compile(r"AQUA_MODIS\.(\d{8})_"), {}

    for var, (subdir, ncvar) in products.items():
        frames = []
        for fp in sorted((P_MODIS / subdir).glob("*_4km_L3m.nc")):
            m = date_re.search(fp.name)
            if not m:
                continue
            t0 = np.datetime64(dt.datetime.strptime(m.group(1), "%Y%m%d"))
            if not (WIN_START <= t0 <= WIN_END):
                continue

            ds_tmp = xr.open_dataset(fp, engine="netcdf4", decode_cf=False)
            da_raw = ds_tmp[ncvar]

            # ── KD 490 special‑case: int16 + scale/offset
            if ncvar == "Kd_490":
                scale   = da_raw.attrs.get("scale_factor", 1.0)
                offset  = da_raw.attrs.get("add_offset",   0.0)
                fill_i  = da_raw.attrs.get("_FillValue", None)
                fill_f  = None if fill_i is None else fill_i * scale + offset

                da_raw  = da_raw.astype("float32") * scale + offset
                if fill_f is not None:
                    da_raw = da_raw.where(da_raw != fill_f)

                for k in ("scale_factor", "add_offset", "_FillValue"):
                    da_raw.attrs.pop(k, None)           # scrub CF attrs
            else:                                        # chlor_a, nflh
                da_raw = da_raw.astype("float32")
                # chlor_a / nflh keep their own _FillValue but xarray now sees
                # real floats, so NaNs will propagate automatically later.

            frames.append(da_raw.expand_dims(time=[t0]))

        if not frames:
            raise RuntimeError(f"No composites found for {var}")

        da = xr.concat(frames, "time").sortby("time")

        # drop duplicate timestamps, keep first
        _, keep = np.unique(da.time.values, return_index=True)
        if len(keep) != len(da.time):
            da = da.isel(time=keep)

        rasters[var] = da

    common = sorted(set.intersection(*(set(r.time.values) for r in rasters.values())))
    modis  = xr.merge([r.sel(time=common).to_dataset(name=v)
                       for v, r in rasters.items()])

    out = P_OUT / "modis_8day_native.nc"
    modis.to_netcdf(out, encoding={v: {"zlib": False} for v in modis.data_vars})
    print(f"✔︎ wrote {out.name}  → {len(common)} composites")
    return modis

# ════════════════════════════════════════════════════════════════════════════
# 2)  E R A ‑ 5   (6 h → 1 d → 8 d)
# ════════════════════════════════════════════════════════════════════════════
def build_era5(modis_time: xr.DataArray) -> xr.Dataset:
    var_specs = {
        "tp"        : ("data_stream-oper_stepType-accum.nc",      "sum"),
        "avg_sdswrf": ("data_stream-oper_stepType-avg.nc",        "mean"),
        "u10"       : ("data_stream-oper_stepType-instant copy.nc","mean"),
        "v10"       : ("data_stream-oper_stepType-instant copy.nc","mean"),
        "t2m"       : ("data_stream-oper_stepType-instant.nc",    "mean"),
        "d2m"       : ("data_stream-oper_stepType-instant.nc",    "mean"),
    }
    bands = []
    for v, (fname, agg) in var_specs.items():
        ds = (xr.open_dataset(P_ERA5/fname, engine="netcdf4",
                              chunks={"valid_time": 672})
                .rename({"valid_time": "time"})
                .pipe(to_datetime, "time")
                .sel(time=slice(WIN_START, WIN_END))
                .astype("float32"))
        daily = ds.resample(time="1D").sum() if agg == "sum" else ds.resample(time="1D").mean()
        bands.append(eightday_calendar(daily[v]).to_dataset(name=v))

    era8 = xr.merge(bands).sortby("time")
    era8 = era8.sel(time=np.intersect1d(era8.time, modis_time))

    out = P_OUT / "era5_8day_native.nc"
    era8.to_netcdf(out, encoding={v: {"zlib": False} for v in era8.data_vars})
    print(f"✔︎ wrote {out.name}  → {len(era8.time)} composites")
    return era8

# ════════════════════════════════════════════════════════════════════════════
# 3)  C M E M S   (1 d → 8 d)
# ════════════════════════════════════════════════════════════════════════════
def build_cmems(modis_time: xr.DataArray) -> xr.Dataset:
    files = [
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_1752164427171.nc",
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_1752164567692.nc",
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_1752164711016.nc",
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_1752164747982.nc",
        "cmems_mod_glo_phy_my_0.083deg_P1D-m_1752164779197.nc",
    ]
    keep  = ["so", "thetao", "uo", "vo", "zos"]
    pieces = []
    for fp in files:
        ds = (xr.open_dataset(P_CMEMS/fp, engine="netcdf4",
                              chunks={"time": 365})
                .rename({"latitude": "lat", "longitude": "lon"})
                .pipe(to_datetime, "time")
                .sel(time=slice(WIN_START, WIN_END))
                .astype("float32"))
        if "depth" in ds.dims:
            ds = ds.isel(depth=0, drop=True)
        pieces.append(ds[keep])

    cm = xr.concat(pieces, "time").sortby("time")
    cm8 = eightday_calendar(cm).sel(time=np.intersect1d(cm.time, modis_time))

    out = P_OUT / "cmems_8day_native.nc"
    cm8.to_netcdf(out, encoding={v: {"zlib": False} for v in cm8.data_vars})
    print(f"✔︎ wrote {out.name}  → {len(cm8.time)} composites")
    return cm8

# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("⏳  MODIS …"); modis_ds = build_modis()
    print("⏳  ERA‑5 …");  era_ds   = build_era5(modis_ds.time)
    print("⏳  CMEMS …");  cmems_ds = build_cmems(modis_ds.time)
    print("✅  All three source cubes built:")
    for p in P_OUT.glob("*_8day_native.nc"):
        print("   ", p)
