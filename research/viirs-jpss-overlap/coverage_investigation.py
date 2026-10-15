"""Investigation of the failed coverage criterion (union N21 u N20 >= S-NPP on >= 80% of days).

usage: coverage_investigation.py <archive_dir> <etopo.nc> <out_dir>

<archive_dir> holds raw/{snpp,n20,n21}/YYYY-MM-DD.nc (stride-4 California subsets on one lattice),
time_map_*.csv and, optionally, raw/n20_4km/YYYY-MM-DD.nc (nesdisVHNnoaa20chlaDaily, 0.0375 deg).
Loading, validity (finite & > 0), ocean mask (ETOPO altitude < 0 & ever valid in any of the three
750 m products on any date), regions and distance-to-shore bands are copied from analyze.py so the
numbers are directly comparable with report.md. Writes small coverage_*.csv tables to <out_dir>.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

W, ETOPO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MAPDIR = sys.argv[4] if len(sys.argv) > 4 else None  # optional: write validity maps of failing days here
os.makedirs(OUT, exist_ok=True)
PRODS = ["snpp", "n20", "n21"]

# ---------------------------------------------------------------- load (as analyze.py)
ref_lat = ref_lon = None
dates = sorted({os.path.basename(q)[:10] for p in PRODS for q in glob.glob(os.path.join(W, "raw", p, "*.nc"))})
cube = {p: [] for p in PRODS}
avail = {p: [] for p in PRODS}
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
            assert np.abs(lat - ref_lat).max() <= 1e-6 and np.abs(lon - ref_lon).max() <= 1e-6
        a[~np.isfinite(a)] = np.nan
        a[a <= 0] = np.nan
        cube[p].append(a.astype("float32")); avail[p].append(True)
        ds.close()
shape = (len(ref_lat), len(ref_lon))
for p in PRODS:
    cube[p] = np.stack([c if c is not None else np.full(shape, np.nan, "float32") for c in cube[p]])
    avail[p] = np.array(avail[p])
dts = pd.to_datetime(dates)

et = xr.open_dataset(ETOPO)
elat, elon = et.latitude.values, et.longitude.values
alt = et["altitude"].values.astype(float)
if elat[0] > elat[-1]:
    elat, alt = elat[::-1], alt[::-1]
land = alt >= 0
dy = abs(elat[1] - elat[0]) * 111.195
dx = abs(elon[1] - elon[0]) * 111.195 * np.cos(np.deg2rad(37.25))
dist_e = ndimage.distance_transform_edt(~land, sampling=(dy, dx))
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
assert n_ocean == 61784, n_ocean
for p in PRODS:
    cube[p][:, ~ocean] = np.nan
valid = {p: np.isfinite(cube[p]) for p in PRODS}

# ---------------------------------------------------------------- 98 three-sensor days
tri = avail["snpp"] & avail["n20"] & avail["n21"]
ti = np.where(tri)[0]
assert len(ti) == 98, len(ti)
tmap = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(W, "time_map_*.csv")))]).drop_duplicates("date").set_index("date")


def hhmm(s):
    return s[11:16] if isinstance(s, str) and len(s) > 15 else ""


STRATA = {"all": ocean}
for r in ["north", "central", "south"]:
    STRATA[r] = ocean & (region == r)
for b in ["0-5km", "5-20km", "20-100km", ">100km"]:
    STRATA[b] = ocean & (dband == b)
STRATA["nearshore_<5km"] = ocean & (dist <= 5)
STRATA["offshore_>=5km"] = ocean & (dist > 5)
STRATA["offshore_>20km"] = ocean & (dist > 20)
NS = {k: int(v.sum()) for k, v in STRATA.items()}
VOID_R = 10

rows = []
for i in ti:
    S, A, B = valid["snpp"][i], valid["n20"][i], valid["n21"][i]
    U = A | B
    d = dates[i]
    r = dict(date=d, month=d[:7],
             snpp_time=hhmm(tmap.loc[d, "snpp_time"]) if d in tmap.index else "",
             n20_time=hhmm(tmap.loc[d, "n20_time"]) if d in tmap.index else "",
             n21_time=hhmm(tmap.loc[d, "n21_time"]) if d in tmap.index else "")
    for k, m in STRATA.items():
        r[f"cov_snpp_{k}"] = 100 * (S & m).sum() / NS[k]
        r[f"cov_union_{k}"] = 100 * (U & m).sum() / NS[k]
        if k == "all":
            r["cov_n20_all"] = 100 * (A & m).sum() / NS[k]
            r["cov_n21_all"] = 100 * (B & m).sum() / NS[k]
            r["cov_snpp_n21_all"] = 100 * ((S | B) & m).sum() / NS[k]
            r["cov_snpp_n20_all"] = 100 * ((S | A) & m).sum() / NS[k]
    so = S & ~U  # S-NPP-only cells (missing in the union)
    uo = U & ~S
    r["n_snpp_valid"] = int(S.sum())
    r["n_snpp_only"] = int(so.sum())
    r["n_union_only"] = int(uo.sum())
    r["n_both"] = int((S & U).sum())
    for k in ["north", "central", "south", "0-5km", "5-20km", "20-100km", ">100km"]:
        r[f"snpp_only_{k}"] = int((so & STRATA[k]).sum())
        r[f"union_only_{k}"] = int((uo & STRATA[k]).sum())
    # how much of the S-NPP-valid area each JPSS sensor also saw (clear-sky agreement)
    r["p_n20_given_snpp"] = 100 * (A & S).sum() / max(S.sum(), 1)
    r["p_n21_given_snpp"] = 100 * (B & S).sum() / max(S.sum(), 1)
    # spatial footprint: share of latitude rows with ocean that contain any valid retrieval
    rows_ocean = ocean.any(1)
    for p, V in [("snpp", S), ("n20", A), ("n21", B)]:
        r[f"rows_with_data_{p}"] = 100 * V.any(1)[rows_ocean].sum() / rows_ocean.sum()
    # "JPSS void": S-NPP-only cell with no valid N20 or N21 cell within +-VOID_R lattice cells (about
    # +-30 km) while S-NPP is >= 50% valid over the ocean cells of that window. Large voids under clear
    # S-NPP sky point to missing swath/granule or swath-edge (view-angle) gaps rather than cloud.
    oc = ocean.astype(float)
    win = 2 * VOID_R + 1
    dens_oc = ndimage.uniform_filter(oc, win, mode="constant")
    dens_u = ndimage.uniform_filter(U.astype(float), win, mode="constant")
    dens_s = ndimage.uniform_filter(S.astype(float), win, mode="constant") / np.maximum(dens_oc, 1e-9)
    void = (dens_u < 1e-9) & (dens_s >= 0.5) & ocean
    r["n_snpp_only_in_jpss_void"] = int((so & void).sum())
    r["share_snpp_only_in_jpss_void"] = (so & void).sum() / max(so.sum(), 1)
    # isolation: share of cells with <= 4 of their 8 neighbours valid in the same product (cloud-edge speckle)
    nS = ndimage.convolve(S.astype(int), np.ones((3, 3), int), mode="constant") - S
    nU = ndimage.convolve(U.astype(int), np.ones((3, 3), int), mode="constant") - U
    r["share_snpp_only_isolated"] = (so & (nS <= 4)).sum() / max(so.sum(), 1)
    r["share_union_only_isolated"] = (uo & (nU <= 4)).sum() / max(uo.sum(), 1)
    r["share_snpp_valid_isolated"] = (S & (nS <= 4)).sum() / max(S.sum(), 1)
    r["share_union_valid_isolated"] = (U & (nU <= 4)).sum() / max(U.sum(), 1)
    for p, V in [("n20", A), ("n21", B)]:
        nV = ndimage.convolve(V.astype(int), np.ones((3, 3), int), mode="constant") - V
        r[f"share_{p}_valid_isolated"] = (V & (nV <= 4)).sum() / max(V.sum(), 1)
    rows.append(r)
day = pd.DataFrame(rows)
day["cov_snpp"] = day.cov_snpp_all
day["cov_union"] = day.cov_union_all
day["deficit_pts"] = day.cov_snpp - day.cov_union
day["ratio_union_snpp"] = np.where(day.cov_snpp > 0, day.cov_union / day.cov_snpp.where(day.cov_snpp > 0), np.nan)
day["fail"] = day.cov_union < day.cov_snpp
for p in ["n20", "n21"]:
    day[f"{p}_stamp_h"] = [int(t[:2]) + int(t[3:5]) / 60 if t else np.nan for t in day[f"{p}_time"]]
EMPTY = 0.05  # % of ocean cells; a product at or below this is "empty" for the day
day["empty_snpp"] = day.cov_snpp <= EMPTY
day["empty_n20"] = day.cov_n20_all <= EMPTY
day["empty_n21"] = day.cov_n21_all <= EMPTY
day["all_three_empty"] = day.empty_snpp & day.empty_n20 & day.empty_n21

# ---------------------------------------------------------------- cause classification (failing days)
# One manual input: validity maps (S-NPP / N20 / N21) of every failing day were inspected by eye
# (maps written by the optional 4th argument). VISUAL records whether a straight-edged no-data boundary
# (swath edge, inter-orbit gap, scan stripe) is visible in a JPSS product over open water that S-NPP saw.
VISUAL = {
    "2026-03-09": (False, "no straight edges; S-NPP keeps more scattered cells in the northern cloud field"),
    "2026-03-24": (True, "N20 and N21 both end at straight swath edges; clear water W of ~123W seen only by S-NPP"),
    "2026-03-31": (False, "cloud pattern similar in all three; S-NPP slightly larger clear patches"),
    "2026-04-04": (True, "N20 straight-edged diagonal gap near 121W; N21 striped scan-pattern loss near 35N"),
    "2026-04-14": (False, "no edges; S-NPP keeps more scattered offshore cells in the south"),
    "2026-04-15": (False, "N21 lacks the central coast 34-37N that S-NPP and N20 see; edge not clearly straight"),
    "2026-04-23": (True, "N20 near-empty with a straight SE edge and stripes; N21 nearly matches S-NPP"),
    "2026-05-01": (False, "near-total cloud"),
    "2026-05-07": (False, "N21 empty; near-total cloud"),
    "2026-05-09": (False, "N21 empty, N20 near-empty (both stamped ~18:25Z, earlier than usual)"),
    "2026-05-11": (False, "N21 misses the clear patch near 41-42N that S-NPP and N20 see"),
    "2026-05-16": (True, "N21 ends at a straight western swath edge; N20 cloudier offshore"),
    "2026-05-17": (False, "N21 near-empty (partial day)"),
    "2026-05-21": (False, "near-total cloud"),
    "2026-05-22": (False, "near-total cloud"),
    "2026-05-27": (True, "N21 ends at a straight western swath edge; N20 cloudier offshore"),
    "2026-05-31": (True, "N20 straight eastern edge and a stripe; N21 nearly matches S-NPP"),
    "2026-06-06": (True, "N20 diagonal stripe gap; N21 straight western edge (N21 stamped 18:06Z)"),
    "2026-06-08": (False, "N21 empty; near-total cloud"),
    "2026-06-12": (False, "N21 near-empty; N20 thin stripe gap"),
    "2026-09-20": (False, "near-total cloud"),
    "2026-09-22": (False, "S-NPP keeps scattered offshore cells 33-36N in broken cloud"),
    "2026-10-02": (False, "thin straight gap lines in N20 and N21 but small; mostly cloud differences"),
}
# Rules, applied in order; the first match is the primary cause, later matches are listed as secondary.
#  a  platform outage / partial day: one JPSS sensor has < 25% of the other's coverage and < 25% of its
#     agreement with S-NPP's clear area (other sensor and S-NPP >= 2%), or one JPSS sensor is <= 0.5%
#     while S-NPP is >= 2%.
#  b  near-total cloud: S-NPP coverage < 10% (a few hundred cells decide the result).
#  e  swath geometry: straight-edged JPSS no-data boundary over water S-NPP saw (VISUAL), or >= 25% of
#     the S-NPP-only cells lie in a "JPSS void" (no N20/N21 retrieval within ~30 km under >= 50% clear S-NPP).
#  d  near-shore: union >= S-NPP once cells <= 5 km from shore are dropped, or >= 50% of the S-NPP-only
#     cells are within 20 km of shore (8% of ocean cells are).
#  c  S-NPP's pass saw more clear sky / kept more cloud-edge cells (different clouds on the same day).
#  m  marginal: deficit < 0.5 points (flag only).
cause, second = [], []
for _, r in day.iterrows():
    c = []
    lo, hi = sorted([r.cov_n20_all, r.cov_n21_all])
    plo, phi = sorted([r.p_n20_given_snpp, r.p_n21_given_snpp])
    if (hi >= 2 and r.cov_snpp >= 2 and lo < 0.25 * hi and plo < 0.25 * phi) or (lo <= 0.5 and r.cov_snpp >= 2):
        c.append("a_platform_outage_partial")
    if r.cov_snpp < 10:
        c.append("b_near_total_cloud")
    if VISUAL.get(r.date, (False, ""))[0] or r.share_snpp_only_in_jpss_void >= 0.25:
        c.append("e_swath_geometry")
    so = max(r.n_snpp_only, 1)
    near_share = (r["snpp_only_0-5km"] + r["snpp_only_5-20km"]) / so
    if (r["cov_union_offshore_>=5km"] >= r["cov_snpp_offshore_>=5km"]) or near_share >= 0.5:
        c.append("d_nearshore")
    c.append("c_snpp_saw_more_clear_sky")
    if r.deficit_pts < 0.5:
        c.append("m_marginal_<0.5pt")
    cause.append(c[0]); second.append(";".join(c[1:]))
day["visual_straight_edge"] = [VISUAL.get(d, (np.nan, ""))[0] for d in day.date]
day["visual_note"] = [VISUAL.get(d, (None, ""))[1] for d in day.date]
day["cause"] = np.where(day.fail, cause, "")
day["secondary"] = np.where(day.fail, second, "")
day["snpp_only_share_within_20km"] = (day["snpp_only_0-5km"] + day["snpp_only_5-20km"]) / day.n_snpp_only.clip(lower=1)
day["dominant_region_snpp_only"] = day[["snpp_only_north", "snpp_only_central", "snpp_only_south"]].idxmax(axis=1).str.replace("snpp_only_", "")
day["fail_offshore_>=5km"] = day["cov_union_offshore_>=5km"] < day["cov_snpp_offshore_>=5km"]

fail = day[day.fail].copy()
cols = ["date", "snpp_time", "n20_time", "n21_time", "cov_snpp", "cov_n20_all", "cov_n21_all", "cov_union", "deficit_pts",
        "ratio_union_snpp", "cause", "secondary", "visual_straight_edge", "visual_note", "share_snpp_only_in_jpss_void",
        "share_snpp_only_isolated", "share_union_only_isolated", "p_n20_given_snpp", "p_n21_given_snpp", "rows_with_data_n20", "rows_with_data_n21",
        "n_snpp_valid", "n_snpp_only", "n_union_only", "snpp_only_share_within_20km", "dominant_region_snpp_only",
        "snpp_only_north", "snpp_only_central", "snpp_only_south", "snpp_only_0-5km", "snpp_only_5-20km", "snpp_only_20-100km", "snpp_only_>100km",
        "cov_snpp_north", "cov_union_north", "cov_snpp_central", "cov_union_central", "cov_snpp_south", "cov_union_south",
        "cov_snpp_nearshore_<5km", "cov_union_nearshore_<5km", "cov_snpp_5-20km", "cov_union_5-20km",
        "cov_snpp_offshore_>=5km", "cov_union_offshore_>=5km", "fail_offshore_>=5km"]
fail[cols].rename(columns={"cov_n20_all": "cov_n20", "cov_n21_all": "cov_n21"}).round(3).to_csv(
    os.path.join(OUT, "coverage_failing_days.csv"), index=False)

# ---------------------------------------------------------------- strata table
srows = []
for k in STRATA:
    for sel_name, sel in [("all_98_days", day), ("failing_23_days", fail)]:
        s, u = sel[f"cov_snpp_{k}"], sel[f"cov_union_{k}"]
        dd = s - u
        srows.append(dict(stratum=k, n_cells=NS[k], days=sel_name, n_days=len(sel),
                          mean_cov_snpp=s.mean(), mean_cov_union=u.mean(), median_cov_snpp=s.median(), median_cov_union=u.median(),
                          days_union_ge_snpp=int((u >= s).sum()), pct_days_union_ge_snpp=100 * (u >= s).mean(),
                          median_deficit_pts=dd.median(), p90_deficit_pts=dd.quantile(0.9)))
strata = pd.DataFrame(srows)
strata.round(3).to_csv(os.path.join(OUT, "coverage_strata.csv"), index=False)

# ---------------------------------------------------------------- cell-level loss on failing days
fi = [ti[j] for j in np.where(day.fail.values)[0]]
cell = {}
S_f = np.stack([valid["snpp"][i] for i in fi]); U_f = np.stack([valid["n20"][i] | valid["n21"][i] for i in fi])
lost = S_f & ~U_f; gained = U_f & ~S_f
cell["failing_days_snpp_valid_cells"] = int(S_f.sum())
cell["failing_days_snpp_valid_missing_in_union"] = int(lost.sum())
cell["failing_days_union_valid_missing_in_snpp"] = int(gained.sum())
for k in ["0-5km", "5-20km", "20-100km", ">100km", "north", "central", "south"]:
    cell[f"lost_share_{k}"] = float(lost[:, STRATA[k]].sum() / lost.sum())
    cell[f"ocean_share_{k}"] = NS[k] / n_ocean
S_a = np.stack([valid["snpp"][i] for i in ti]); U_a = np.stack([valid["n20"][i] | valid["n21"][i] for i in ti])
cell["all_days_snpp_valid_cells"] = int(S_a.sum())
cell["all_days_snpp_valid_missing_in_union"] = int((S_a & ~U_a).sum())
cell["all_days_union_valid_missing_in_snpp"] = int((U_a & ~S_a).sum())
cell["all_days_share_snpp_cells_also_in_union"] = float((S_a & U_a).sum() / S_a.sum())

# ---------------------------------------------------------------- baselines and criterion variants
var = []


def add(name, passed, note=""):
    passed = np.asarray(passed, bool)
    var.append(dict(criterion=name, n_days=int(passed.size), days_pass=int(passed.sum()), pct_days_pass=100 * passed.mean(), note=note))


s, u = day.cov_snpp.values, day.cov_union.values
add("ORIGINAL union >= S-NPP (per day)", u >= s, "acceptance criterion as proposed: FAILED (needs >= 80%)")
add("union >= 0.95 x S-NPP", u >= 0.95 * s, "tolerance variant")
add("union >= 0.90 x S-NPP", u >= 0.90 * s, "tolerance variant")
add("union >= S-NPP - 1 pt", u >= s - 1, "tolerance variant")
add("union >= S-NPP - 2 pt", u >= s - 2, "tolerance variant")
add("union >= S-NPP - 5 pt", u >= s - 5, "tolerance variant")
add("ORIGINAL test, excluding the 8 days on which all three products are empty", (u >= s)[~day.all_three_empty.values],
    "0 >= 0 ties removed; these days carry no information")
add("union >= S-NPP, days with S-NPP >= 10% only", (u >= s)[s >= 10], "excludes near-total cloud days")
add("union >= 0.9 x S-NPP, days with S-NPP >= 10% only", (u >= 0.9 * s)[s >= 10], "")
add("union >= S-NPP offshore (>= 5 km) only", day["cov_union_offshore_>=5km"] >= day["cov_snpp_offshore_>=5km"], "stratum")
add("union >= S-NPP near-shore (<= 5 km) only", day["cov_union_nearshore_<5km"] >= day["cov_snpp_nearshore_<5km"], "stratum")
for k in ["north", "central", "south"]:
    add(f"union >= S-NPP, {k} region only", day[f"cov_union_{k}"] >= day[f"cov_snpp_{k}"], "stratum")
add("BASELINE N21 >= N20", day.cov_n21_all >= day.cov_n20_all, "two near-identical JPSS sensors")
add("BASELINE N20 >= N21", day.cov_n20_all >= day.cov_n21_all, "")
add("BASELINE N21 >= S-NPP", day.cov_n21_all >= s, "single sensor vs single sensor")
add("BASELINE N20 >= S-NPP", day.cov_n20_all >= s, "")
add("BASELINE S-NPP u N21 >= N20", day.cov_snpp_n21_all >= day.cov_n20_all, "same 'two-sensor union vs third sensor' test, sensors permuted")
add("BASELINE S-NPP u N20 >= N21", day.cov_snpp_n20_all >= day.cov_n21_all, "same test, sensors permuted")
# S-NPP against itself on adjacent days (all S-NPP days, not only the 98)
si = np.where(avail["snpp"])[0]
covS = {dates[i]: valid["snpp"][i].sum() / n_ocean * 100 for i in si}
nxt, two = [], []
for i in si:
    d = dts[i]
    dn, dp = (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if dn in covS:
        nxt.append(covS[dn] >= covS[dates[i]])
    if dn in covS and dp in covS:
        jn, jp = dates.index(dn), dates.index(dp)
        two.append(((valid["snpp"][jn] | valid["snpp"][jp]).sum() / n_ocean * 100) >= covS[dates[i]])
add("BASELINE S-NPP(day+1) >= S-NPP(day)", nxt, "S-NPP against itself one day later, all consecutive S-NPP day pairs")
add("BASELINE S-NPP(day-1) u S-NPP(day+1) >= S-NPP(day)", two, "two S-NPP looks a day apart vs one look")
variants = pd.DataFrame(var)

# continuous summaries
rr = day.ratio_union_snpp.dropna().values
summ = dict(mean_cov_snpp=float(s.mean()), mean_cov_union=float(u.mean()), median_cov_snpp=float(np.median(s)),
            median_cov_union=float(np.median(u)), n_days_ratio_defined=int(len(rr)), median_ratio=float(np.median(rr)),
            p10_ratio=float(np.quantile(rr, 0.1)), p25_ratio=float(np.quantile(rr, 0.25)),
            pooled_ratio_sum_union_over_sum_snpp=float(u.sum() / s.sum()))
for name, val in summ.items():
    variants = pd.concat([variants, pd.DataFrame([dict(criterion="SUMMARY " + name, n_days=98, days_pass=np.nan, pct_days_pass=np.nan, note=f"{val:.4f}")])])
# ISO-week pooled coverage (weeks with >= 3 three-sensor days)
day["week"] = pd.to_datetime(day.date).dt.to_period("W-SUN")
wk = day.groupby("week").agg(n=("date", "size"), s=("cov_snpp", "mean"), u=("cov_union", "mean"))
wk3 = wk[wk.n >= 3]
add_rows = dict(criterion="weekly mean union >= weekly mean S-NPP (weeks with >= 3 three-sensor days)", n_days=len(wk3),
                days_pass=int((wk3.u >= wk3.s).sum()), pct_days_pass=100 * (wk3.u >= wk3.s).mean(), note="unit = week, not day")
variants = pd.concat([variants, pd.DataFrame([add_rows])])
variants.round(3).to_csv(os.path.join(OUT, "coverage_criterion_variants.csv"), index=False)

# ---------------------------------------------------------------- shortfall stats and seasonal
def q(x):
    x = np.asarray(x, float)
    return dict(n=len(x), median=np.median(x), p25=np.quantile(x, .25), p75=np.quantile(x, .75), p90=np.quantile(x, .9), max=x.max(), mean=x.mean())


short = []
for k in ["all", "north", "central", "south", "nearshore_<5km", "5-20km", "offshore_>=5km"]:
    dd = fail[f"cov_snpp_{k}"] - fail[f"cov_union_{k}"]
    row = dict(stratum=k, metric="deficit_pts_on_23_failing_days (S-NPP - union; negative = union higher)", **q(dd))
    short.append(row)
short.append(dict(stratum="all", metric="ratio_union_over_snpp_on_failing_days", **q(fail.ratio_union_snpp)))
short.append(dict(stratum="all", metric="ratio_union_over_snpp_days_with_snpp>0", **q(day.ratio_union_snpp.dropna())))
short.append(dict(stratum="all", metric="deficit_pts_all_98_days", **q(day.deficit_pts)))
short = pd.DataFrame(short)
short.round(3).to_csv(os.path.join(OUT, "coverage_shortfall.csv"), index=False)

mon = day.groupby("month").agg(n_days=("date", "size"), n_fail=("fail", "sum"), mean_cov_snpp=("cov_snpp", "mean"),
                               mean_cov_union=("cov_union", "mean"), median_ratio=("ratio_union_snpp", "median")).reset_index()
mon["pct_union_ge_snpp"] = 100 * (1 - mon.n_fail / mon.n_days)
mon.round(3).to_csv(os.path.join(OUT, "coverage_by_month.csv"), index=False)

# ---------------------------------------------------------------- 4 km NOAA-20 fallback (if present)
k4dir = os.path.join(W, "raw", "n20_4km")
k4rows = []
if os.path.isdir(k4dir) and glob.glob(os.path.join(k4dir, "*.nc")):
    g_iy = g_ix = None
    for j, i in enumerate(ti):
        d = dates[i]
        f = os.path.join(k4dir, f"{d}.nc")
        if not os.path.exists(f):
            k4rows.append(dict(date=d, present=False)); continue
        ds = xr.open_dataset(f, mask_and_scale=True)
        a = ds["chlor_a"].values.squeeze().astype("float64")
        la, lo = ds.latitude.values, ds.longitude.values
        ds.close()
        if g_iy is None:
            g_iy = np.abs(la[:, None] - ref_lat[None, :]).argmin(0)
            g_ix = np.abs(lo[:, None] - ref_lon[None, :]).argmin(0)
            res = abs(la[1] - la[0])
            assert np.abs(la[g_iy] - ref_lat).max() <= res / 2 + 1e-9 and np.abs(lo[g_ix] - ref_lon).max() <= res / 2 + 1e-9
            lat_desc = bool(la[0] > la[-1])
            oc4 = np.zeros(a.shape, bool)
            YY, XX = np.meshgrid(g_iy, g_ix, indexing="ij")
            oc4[YY[ocean], XX[ocean]] = True  # 4 km cells containing >= 1 sampled ocean cell
        v4 = np.isfinite(a) & (a > 0)
        V = v4[np.ix_(g_iy, g_ix)] & ocean  # 750 m sample cell lies inside a valid 4 km cell
        S, A, B = valid["snpp"][i], valid["n20"][i], valid["n21"][i]
        U = A | B
        r = dict(date=d, present=True, fail=bool(day.fail.iloc[j]),
                 cov_snpp=100 * S.sum() / n_ocean, cov_n20_750=100 * A.sum() / n_ocean, cov_union=100 * U.sum() / n_ocean,
                 cov_4km_footprint=100 * V.sum() / n_ocean,
                 cov_4km_native=100 * (v4 & oc4).sum() / oc4.sum(),
                 cov_union_or_4km=100 * (U | V).sum() / n_ocean,
                 share_snpp_cells_in_valid_4km=100 * (S & V).sum() / max(S.sum(), 1),
                 share_snpp_cells_in_union=100 * (S & U).sum() / max(S.sum(), 1),
                 share_snpp_cells_in_union_or_4km=100 * (S & (U | V)).sum() / max(S.sum(), 1),
                 share_snpp_only_cells_in_valid_4km=100 * (S & ~U & V).sum() / max((S & ~U).sum(), 1),
                 share_n20_750_cells_in_valid_4km=100 * (A & V).sum() / max(A.sum(), 1),
                 cov_4km_outside_n20_750=100 * (V & ~A).sum() / n_ocean,
                 cov_4km_outside_union=100 * (V & ~U).sum() / n_ocean,
                 cov_4km_near_lt5km=100 * (V & STRATA["nearshore_<5km"]).sum() / NS["nearshore_<5km"],
                 cov_snpp_near_lt5km=100 * (S & STRATA["nearshore_<5km"]).sum() / NS["nearshore_<5km"],
                 lat_descending=lat_desc)
        # coverage of the 4 km product on its own grid within the same ocean area: 4 km cells containing >= 1 ocean sample
        k4rows.append(r)
k4 = pd.DataFrame(k4rows)

if len(k4) and "cov_4km_footprint" in k4:
    k4.round(3).to_csv(os.path.join(OUT, "coverage_4km.csv"), index=False)
    p = k4[k4.present]
    k4sum = dict(n_days=int(len(p)),
                 days_4km_footprint_ge_snpp=int((p.cov_4km_footprint >= p.cov_snpp).sum()),
                 days_union_or_4km_ge_snpp=int((p.cov_union_or_4km >= p.cov_snpp).sum()),
                 failing_days_rescued_by_4km=int((p.fail & (p.cov_union_or_4km >= p.cov_snpp)).sum()),
                 dates_absent=k4[~k4.present].date.tolist(),
                 failing_days_absent=[d for d in k4[~k4.present].date if d in set(fail.date)],
                 n_4km_ocean_cells=int(oc4.sum()),
                 mean_cov_4km_native=float(p.cov_4km_native.mean()),
                 mean_cov_4km_footprint=float(p.cov_4km_footprint.mean()),
                 mean_cov_snpp_same_days=float(p.cov_snpp.mean()),
                 days_union_ge_snpp_same_days=int((p.cov_union >= p.cov_snpp).sum()), mean_cov_n20_750=float(p.cov_n20_750.mean()),
                 mean_cov_union=float(p.cov_union.mean()), mean_cov_union_or_4km=float(p.cov_union_or_4km.mean()),
                 mean_share_snpp_cells_in_valid_4km=float(p.share_snpp_cells_in_valid_4km.mean()),
                 mean_share_snpp_cells_in_union=float(p.share_snpp_cells_in_union.mean()),
                 mean_share_snpp_cells_in_union_or_4km=float(p.share_snpp_cells_in_union_or_4km.mean()),
                 mean_share_snpp_only_cells_in_valid_4km_failing=float(p[p.fail].share_snpp_only_cells_in_valid_4km.mean()),
                 mean_share_n20_750_cells_in_valid_4km=float(p.share_n20_750_cells_in_valid_4km.mean()),
                 mean_cov_4km_outside_n20_750=float(p.cov_4km_outside_n20_750.mean()),
                 corr_cov_4km_vs_n20_750=float(np.corrcoef(p.cov_4km_footprint, p.cov_n20_750)[0, 1]),
                 corr_cov_4km_vs_snpp=float(np.corrcoef(p.cov_4km_footprint, p.cov_snpp)[0, 1]),
                 mean_cov_4km_near_lt5km=float(p["cov_4km_near_lt5km"].mean()), mean_cov_snpp_near_lt5km=float(p["cov_snpp_near_lt5km"].mean()))
else:
    k4sum = dict(note="4 km files not present; part skipped")

# empties and time stamps
diag = dict(days_all_three_empty=day.date[day.all_three_empty].tolist(),
            n20_n21_stamps_on_all_empty_days=[f"{a}/{b}" for a, b in zip(day.n20_time[day.all_three_empty], day.n21_time[day.all_three_empty])],
            days_snpp_empty_others_not=day.date[day.empty_snpp & ~(day.empty_n20 & day.empty_n21)].tolist(),
            days_n20_empty_others_not=day.date[day.empty_n20 & ~(day.empty_snpp & day.empty_n21)].tolist(),
            days_n21_empty_others_not=day.date[day.empty_n21 & ~(day.empty_snpp & day.empty_n20)].tolist())
st = []
for p, o in [("n20", "n21"), ("n21", "n20")]:
    for _, r in day.iterrows():
        oc_ = r[f"cov_{o}_all"]
        if oc_ >= 2:
            st.append(dict(sensor=p, stamp_h=r[f"{p}_stamp_h"], rel=r[f"cov_{p}_all"] / oc_))
st = pd.DataFrame(st)
early = st.stamp_h < 18.75
diag["jpss_sensor_days_other_ge2pct"] = int(len(st))
diag["early_stamp_lt_18:45Z_n"] = int(early.sum())
diag["early_stamp_median_cov_ratio_to_other_jpss"] = float(st.rel[early].median()) if early.any() else None
diag["early_stamp_share_ratio_lt_0.5"] = float((st.rel[early] < 0.5).mean()) if early.any() else None
diag["normal_stamp_median_cov_ratio_to_other_jpss"] = float(st.rel[~early].median())
diag["normal_stamp_share_ratio_lt_0.5"] = float((st.rel[~early] < 0.5).mean())
nz = day[~day.all_three_empty]
for k in ["snpp", "n20", "n21", "union"]:
    diag[f"mean_share_valid_isolated_{k}"] = float(nz[f"share_{k}_valid_isolated"].mean())
diag["mean_share_snpp_only_isolated_failing"] = float(day[day.fail].share_snpp_only_isolated.mean())
diag["mean_share_union_only_isolated_failing"] = float(day[day.fail].share_union_only_isolated.mean())
diag["visual_straight_edge_failing_days"] = int(day[day.fail].visual_straight_edge.astype(bool).sum())
mpp = dict()
for p in ["snpp", "n20", "n21"]:
    mpp[p] = float(day[f"cov_{p}_all" if p != "snpp" else "cov_snpp"].mean())
diag["mean_cov_single_sensors"] = mpp
# pairwise overlap of clear areas (Jaccard), pooled over the 98 days
def jac(a, b):
    a, b = valid[a][ti], valid[b][ti]
    return float((a & b).sum() / (a | b).sum())
diag["jaccard_n20_n21"] = jac("n20", "n21"); diag["jaccard_snpp_n20"] = jac("snpp", "n20"); diag["jaccard_snpp_n21"] = jac("snpp", "n21")
out = dict(n_ocean=n_ocean, diagnostics=diag, strata_cells=NS, cell_level=cell, summary=summ, four_km=k4sum,
           cause_counts=fail.cause.value_counts().to_dict(),
           fail_dates=fail.date.tolist())
def flat(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{pre}{k}.")
        else:
            yield f"{pre}{k}", ";".join(map(str, v)) if isinstance(v, list) else v
pd.DataFrame(list(flat(out)), columns=["key", "value"]).to_csv(os.path.join(OUT, "coverage_summary.csv"), index=False)
if MAPDIR:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(MAPDIR, exist_ok=True)
    for dd in fail.date:
        i = dates.index(dd)
        fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
        for c, p in enumerate(PRODS):
            ax[c].imshow(valid[p][i], cmap="Greys", aspect="auto", interpolation="nearest",
                         extent=[ref_lon.min(), ref_lon.max(), ref_lat.min(), ref_lat.max()])
            ax[c].set_title(f"{dd} {p} {100 * valid[p][i].sum() / n_ocean:.1f}%", fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(MAPDIR, f"valid_{dd}.png"), dpi=70); plt.close(fig)
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_rows", 200)
print(json.dumps(out, indent=1, default=float))
print(day[["date","cov_snpp","cov_n20_all","cov_n21_all","cov_union","n20_time","n21_time","share_snpp_only_in_jpss_void","n_snpp_only_in_jpss_void","share_snpp_only_isolated","share_union_only_isolated","share_snpp_valid_isolated","share_union_valid_isolated","fail"]].round(3).to_string())
print(fail[["date", "cov_snpp", "cov_n20_all", "cov_n21_all", "cov_union", "deficit_pts", "cause", "secondary", "p_n20_given_snpp",
            "p_n21_given_snpp", "rows_with_data_n20", "rows_with_data_n21", "snpp_only_share_within_20km", "dominant_region_snpp_only",
            "fail_offshore_>=5km"]].round(2).to_string())
print(strata.round(2).to_string())
print(variants.round(2).to_string())
print(short.round(2).to_string())
print(mon.round(2).to_string())
