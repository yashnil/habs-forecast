"""Pairwise validation of S-NPP / NOAA-20 / NOAA-21 VIIRS chl over California.

usage: analyze.py <workdir> <etopo.nc>
Writes results/*.csv and results/summary.json under <workdir>.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

W = sys.argv[1]
ETOPO = sys.argv[2]
OUT = os.path.join(W, "results")
os.makedirs(OUT, exist_ok=True)
PRODS = ["snpp", "n20", "n21"]
PAIRS = [("n20", "snpp"), ("n21", "snpp"), ("n21", "n20")]
PRIMARY_START = pd.Timestamp("2026-07-01")

# ---------------------------------------------------------------- load
ref_lat = ref_lon = None
dates = sorted({os.path.basename(p)[:10] for p in [q for p in PRODS for q in glob.glob(os.path.join(W, "raw", p, "*.nc"))]})
cube = {p: [] for p in PRODS}
avail = {p: [] for p in PRODS}
coord_maxdiff = 0.0
for d in dates:
    for p in PRODS:
        f = os.path.join(W, "raw", p, f"{d}.nc")
        if not os.path.exists(f):
            cube[p].append(None); avail[p].append(False); continue
        ds = xr.open_dataset(f, mask_and_scale=True)
        v = "chla" if "chla" in ds else "chlor_a"
        a = ds[v].values.squeeze().astype("float64")
        lat, lon = ds.latitude.values, ds.longitude.values
        if ref_lat is None:
            ref_lat, ref_lon = lat, lon
        else:
            if lat.shape != ref_lat.shape or lon.shape != ref_lon.shape:
                raise SystemExit(f"shape mismatch {p} {d}")
            coord_maxdiff = max(coord_maxdiff, np.abs(lat - ref_lat).max(), np.abs(lon - ref_lon).max())
        # exclusions: NaN, fill (-999 or any <=0), keep >0
        a[~np.isfinite(a)] = np.nan
        a[a <= 0] = np.nan
        cube[p].append(a.astype("float32")); avail[p].append(True)
        ds.close()
assert coord_maxdiff <= 1e-6, coord_maxdiff
shape = (len(ref_lat), len(ref_lon))
for p in PRODS:
    cube[p] = np.stack([c if c is not None else np.full(shape, np.nan, "float32") for c in cube[p]])
    avail[p] = np.array(avail[p])
dts = pd.to_datetime(dates)

# ---------------------------------------------------------------- ocean mask + distance to shore
et = xr.open_dataset(ETOPO)
ev = et["altitude"]
elat, elon = et.latitude.values, et.longitude.values
alt = ev.values.astype(float)
if elat[0] > elat[-1]:
    elat, alt = elat[::-1], alt[::-1]
land = alt >= 0
dy = abs(elat[1] - elat[0]) * 111.195
dx = abs(elon[1] - elon[0]) * 111.195 * np.cos(np.deg2rad(37.25))
dist_e = ndimage.distance_transform_edt(~land, sampling=(dy, dx))  # km from ocean node to nearest land node
iy = np.abs(elat[:, None] - ref_lat[None, :]).argmin(0)
ix = np.abs(elon[:, None] - ref_lon[None, :]).argmin(0)
alt_g = alt[np.ix_(iy, ix)]
dist = dist_e[np.ix_(iy, ix)]
ever = np.zeros(shape, bool)
for p in PRODS:
    ever |= np.isfinite(cube[p]).any(0)
ocean = (alt_g < 0) & ever
LAT2 = np.broadcast_to(ref_lat[:, None], shape)
region = np.where(LAT2 >= 40, "north", np.where(LAT2 >= 34.5, "central", "south"))
dband = np.select([dist <= 5, dist <= 20, dist <= 100], ["0-5km", "5-20km", "20-100km"], ">100km")
n_ocean = int(ocean.sum())
# sanity: valid retrievals that fall outside ocean mask
outside = {p: int((np.isfinite(cube[p]) & ~ocean[None]).sum()) for p in PRODS}
for p in PRODS:
    cube[p][:, ~ocean] = np.nan


# ---------------------------------------------------------------- metrics
def stats(y, x):
    """y, x log10 arrays (paired, finite)."""
    n = len(x)
    if n < 3:
        return dict(n=n)
    d = y - x
    med = float(np.median(d))
    r = float(np.corrcoef(x, y)[0, 1])
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    return dict(n=n, median_log10_ratio=med, median_ratio=10 ** med,
                mad_log10=float(np.median(np.abs(d - med))),
                mean_log10_ratio=float(d.mean()),
                rmse_log10=float(np.sqrt(np.mean(d ** 2))),
                r_log10=r, rma_slope_log10=float(np.sign(r) * sy / sx))


L = {p: np.log10(cube[p]) for p in PRODS}
valid = {p: np.isfinite(cube[p]) for p in PRODS}

rows = []
for i, d in enumerate(dates):
    row = dict(date=d, snpp_avail=bool(avail["snpp"][i]), n20_avail=bool(avail["n20"][i]), n21_avail=bool(avail["n21"][i]))
    for p in PRODS:
        row[f"cov_{p}"] = 100 * valid[p][i].sum() / n_ocean if avail[p][i] else np.nan
        nv = valid[p][i].sum()
        row[f"frac_gt200_{p}"] = float((cube[p][i] > 200).sum() / nv) if nv else np.nan
    row["cov_union_n21n20"] = 100 * (valid["n20"][i] | valid["n21"][i]).sum() / n_ocean
    row["cov_union_all3"] = 100 * (valid["n20"][i] | valid["n21"][i] | valid["snpp"][i]).sum() / n_ocean
    for a, b in PAIRS:
        m = valid[a][i] & valid[b][i]
        s = stats(L[a][i][m], L[b][i][m])
        for k, v in s.items():
            row[f"{a}_{b}_{k}"] = v
    rows.append(row)
daily = pd.DataFrame(rows)
daily.to_csv(os.path.join(OUT, "daily_metrics.csv"), index=False)

windows = {
    "primary_2026-07-01_on": dts >= PRIMARY_START,
    "extended_2026-02_to_06": dts < PRIMARY_START,
    "all": np.ones(len(dts), bool),
}
pooled_rows, strat_rows = [], []
for wname, wm in windows.items():
    for a, b in PAIRS:
        m = valid[a] & valid[b] & wm[:, None, None]
        s = stats(L[a][m], L[b][m]); s.update(window=wname, pair=f"{a}/{b}", stratum="all", level="all")
        s["n_days_with_pairs"] = int((m.sum((1, 2)) > 0).sum())
        pooled_rows.append(s)
        # chl class by S-NPP (for n21/n20 pair: class by N20 since S-NPP is not in the pair)
        cls_src = cube["snpp"] if b == "snpp" else cube["n20"]
        cls = np.digitize(cls_src, [0.3, 1, 3, 10])
        names = ["<0.3", "0.3-1", "1-3", "3-10", ">10"]
        for strat, lab in [("region", np.broadcast_to(region, cube["snpp"].shape)),
                           ("dist_to_shore", np.broadcast_to(dband, cube["snpp"].shape))]:
            for lv in np.unique(lab[0][ocean]):
                mm = m & (lab == lv)
                s = stats(L[a][mm], L[b][mm]); s.update(window=wname, pair=f"{a}/{b}", stratum=strat, level=lv)
                strat_rows.append(s)
        for k, nm in enumerate(names):
            mm = m & (cls == k) & np.isfinite(cls_src)
            s = stats(L[a][mm], L[b][mm]); s.update(window=wname, pair=f"{a}/{b}", stratum="chl_class_by_" + ("snpp" if b == "snpp" else "n20"), level=nm)
            strat_rows.append(s)
pooled = pd.DataFrame(pooled_rows)
strat = pd.DataFrame(strat_rows)
pooled.to_csv(os.path.join(OUT, "pooled_metrics.csv"), index=False)
strat.to_csv(os.path.join(OUT, "stratified_metrics.csv"), index=False)

# coverage / implausible pooled per window
cov_rows = []
for wname, wm in windows.items():
    sel = daily[wm]
    tri = sel[sel.snpp_avail & sel.n20_avail & sel.n21_avail]
    r = dict(window=wname, n_days=len(sel), n_days_all3=len(tri))
    for c in ["cov_snpp", "cov_n20", "cov_n21", "cov_union_n21n20", "cov_union_all3"]:
        r[f"mean_{c}"] = float(tri[c].mean()) if len(tri) else np.nan
        r[f"median_{c}"] = float(tri[c].median()) if len(tri) else np.nan
    if len(tri):
        r["days_union_ge_snpp"] = int((tri.cov_union_n21n20 >= tri.cov_snpp).sum())
        r["pct_days_union_ge_snpp"] = 100 * r["days_union_ge_snpp"] / len(tri)
        r["days_n21_ge_snpp"] = int((tri.cov_n21 >= tri.cov_snpp).sum())
        r["days_n20_ge_snpp"] = int((tri.cov_n20 >= tri.cov_snpp).sum())
        r["days_n21_ge_n20"] = int((tri.cov_n21 >= tri.cov_n20).sum())
    for p in PRODS:
        v = valid[p] & wm[:, None, None]
        nv = int(v.sum())
        r[f"nvalid_{p}"] = nv
        r[f"frac_gt200_{p}"] = float((cube[p][v] > 200).sum() / nv) if nv else np.nan
        r[f"frac_gt100_{p}"] = float((cube[p][v] > 100).sum() / nv) if nv else np.nan
        r[f"p99_{p}"] = float(np.percentile(cube[p][v], 99)) if nv else np.nan
        r[f"max_{p}"] = float(cube[p][v].max()) if nv else np.nan
        r[f"median_{p}"] = float(np.median(cube[p][v])) if nv else np.nan
    cov_rows.append(r)
cov = pd.DataFrame(cov_rows)
cov.to_csv(os.path.join(OUT, "coverage_implausible.csv"), index=False)

# weekly stability (ISO week, pooled pairs within week)
wk = dts.to_period("W-SUN")
wrows = []
for w in sorted(set(wk)):
    wm = np.asarray(wk == w)
    r = dict(week_start=str(w.start_time.date()), n_days=int(wm.sum()))
    for a, b in PAIRS:
        m = valid[a] & valid[b] & wm[:, None, None]
        n = int(m.sum())
        r[f"{a}_{b}_n"] = n
        r[f"{a}_{b}_median_log10"] = float(np.median(L[a][m] - L[b][m])) if n >= 50 else np.nan
    wrows.append(r)
weekly = pd.DataFrame(wrows)
weekly.to_csv(os.path.join(OUT, "weekly_stability.csv"), index=False)

summary = dict(dates=[dates[0], dates[-1]], n_dates=len(dates),
               days_avail={p: int(avail[p].sum()) for p in PRODS},
               missing={p: [d for d, a in zip(dates, avail[p]) if not a] for p in PRODS},
               grid=dict(nlat=shape[0], nlon=shape[1], lat=[float(ref_lat[0]), float(ref_lat[-1])],
                         lon=[float(ref_lon[0]), float(ref_lon[-1])]),
               coord_maxdiff=float(coord_maxdiff), n_ocean=n_ocean,
               n_etopo_ocean=int((alt_g < 0).sum()), n_ever_valid=int(ever.sum()),
               valid_outside_ocean_mask=outside,
               ocean_by_region={k: int((ocean & (region == k)).sum()) for k in ["north", "central", "south"]},
               ocean_by_dband={k: int((ocean & (dband == k)).sum()) for k in ["0-5km", "5-20km", "20-100km", ">100km"]})
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
print(json.dumps(summary)[:3000])
print(pooled.round(4).to_string())
print(cov.round(4).T.to_string())
print(weekly.round(3).to_string())
print(strat[strat.window != "extended_2026-02_to_06"].round(3).to_string())
