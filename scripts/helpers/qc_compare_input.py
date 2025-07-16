
#!/usr/bin/env python3
"""
QC comparison: unfilled vs filled CA coastal cube (MODIS optical vars).

Inputs
------
HAB_master_8day_4km_coastal_CA.nc              # original, masked, NOT gap-filled
HAB_master_8day_4km_coastal_CA_climclip.nc     # gap-filled (climatology+clip)

Outputs (printed)
-----------------
• Stripe size & coverage
• For each var (chlor_a, Kd_490, nflh, log_chl):
    - counts of observed vs synthetic pixels
    - distribution stats (obs vs synthetic)
    - 99th-percentile ratio synthetic/observed (outlier flag)
    - max abs diff at observed pixels (should be ~0 if obs preserved)
• Per-time coverage improvement summary
• Top-20 extreme synthetic values table
Optional CSV export of synthetic extremes.

Notes
-----
Large operational L4 ocean-colour products routinely apply spatiotemporal
interpolation / EOF reconstructions (e.g., DINEOF) and range checks to
produce cloud-free fields; comparison to observed distributions is a key
QC step before downstream ecological modeling. This script mimics that
QC workflow at regional scale.  See refs in header comments below.
"""

from __future__ import annotations
import pathlib
import numpy as np
import xarray as xr
import pandas as pd

try:
    from tabulate import tabulate
    _HAVE_TAB = True
except ImportError:  # graceful fallback
    _HAVE_TAB = False

# ------------------------------------------------------------------
# paths
# ------------------------------------------------------------------
ROOT = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research")
FIN  = ROOT / "Data" / "Finalized"

RAW_FP = FIN / "HAB_master_8day_4km_coastal_CA.nc"              # before fill
FIL_FP = FIN / "HAB_master_8day_4km_coastal_CA_climclip.nc"      # after fill

OUT_CSV = FIN / "qc_ca_coastal_impute_extremes.csv"              # optional

VARS = ["chlor_a", "Kd_490", "nflh", "log_chl"]                  # log_chl in filled file only


# ------------------------------------------------------------------
# open datasets
# ------------------------------------------------------------------
print("🔹 opening datasets …")
raw = xr.open_dataset(RAW_FP)
fil = xr.open_dataset(FIL_FP)

# force canonical dim order
raw = raw.transpose("time", "lat", "lon")
fil = fil.transpose("time", "lat", "lon")

# ------------------------------------------------------------------
# derive CA coastal stripe mask from *raw* (union of optics)
# ------------------------------------------------------------------
stripe2d = (
    raw["chlor_a"].notnull().any("time") |
    raw["Kd_490"].notnull().any("time")  |
    raw["nflh"].notnull().any("time")
)
stripe2d.name = "stripe"
stripe3d = stripe2d.expand_dims(time=raw.time)

n_stripe = int(stripe2d.sum())
print(f"   stripe pixels: {n_stripe:,} "
      f"({n_stripe / (raw.sizes['lat']*raw.sizes['lon']):.2%} of regional grid)\n")

# convenience boolean NDArrays
stripe3d_bool = stripe3d.data


# ------------------------------------------------------------------
# util: stats helper
# ------------------------------------------------------------------
def _stats(arr: np.ndarray) -> dict:
    """Return dict of distribution stats ignoring NaNs."""
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return dict(n=0, min=np.nan, p01=np.nan, p05=np.nan,
                    med=np.nan, p95=np.nan, p99=np.nan, max=np.nan)
    return dict(
        n   = v.size,
        min = np.nanmin(v),
        p01 = np.nanpercentile(v, 1),
        p05 = np.nanpercentile(v, 5),
        med = np.nanmedian(v),
        p95 = np.nanpercentile(v, 95),
        p99 = np.nanpercentile(v, 99),
        max = np.nanmax(v),
    )


# ------------------------------------------------------------------
# per-variable comparisons
# ------------------------------------------------------------------
rows = []
extreme_records = []

for v in VARS:
    if v not in fil:
        print(f"⚠︎ {v} not in filled file; skipping")
        continue

    print(f"▶ {v} …")

    # observed mask: cells that had data in RAW (if log_chl use chlor_a)
    if v == "log_chl":
        obs3d = raw["chlor_a"].notnull() & stripe3d
    else:
        obs3d = raw[v].notnull() & stripe3d

    # synthetic mask: stripe pixels that were NaN in RAW but valid in FIL
    syn3d = (~obs3d) & stripe3d & fil[v].notnull()

    n_obs_px = int(obs3d.any("time").sum())  # count of stripe pixels ever observed
    n_syn_px = int(syn3d.any("time").sum())  # stripe pixels we fabricated at least once

    # value arrays
    obs_vals = fil[v].where(obs3d).values
    syn_vals = fil[v].where(syn3d).values

    s_obs = _stats(obs_vals)
    s_syn = _stats(syn_vals)

    # ratio of 99th percentiles (synthetic vs observed)
    ratio_p99 = np.nan
    if np.isfinite(s_obs["p99"]) and np.isfinite(s_syn["p99"]) and s_obs["p99"] != 0:
        ratio_p99 = s_syn["p99"] / s_obs["p99"]

    # check obs preserved (compare raw vs fil at obs cells, excluding log_chl)
    max_abs_diff = np.nan
    if v != "log_chl" and v in raw:
        diff = (fil[v] - raw[v]).where(obs3d)
        max_abs_diff = float(np.nanmax(np.abs(diff.values)))

    rows.append([
        v,
        n_obs_px, n_syn_px,
        s_obs["min"], s_obs["p01"], s_obs["med"], s_obs["p99"], s_obs["max"],
        s_syn["min"], s_syn["p01"], s_syn["med"], s_syn["p99"], s_syn["max"],
        ratio_p99, max_abs_diff
    ])

    # collect extreme synthetic values (top 20 by value)
    if s_syn["n"] > 0:
        da_syn = fil[v].where(syn3d)
        flat = da_syn.stack(z=("time","lat","lon"))
        flat = flat.dropna("z")
        if flat.sizes["z"] > 0:
            topN = flat.sortby(flat, ascending=False).isel(z=slice(0,20))
            # counts of obs in RAW at those pixels
            if v == "log_chl":
                obs_count_src = raw["chlor_a"].notnull()
            else:
                obs_count_src = raw[v].notnull()
            obs_cnt = obs_count_src.sel(time=topN["time"], lat=topN["lat"], lon=topN["lon"])
            # store
            for i in range(topN.sizes["z"]):
                extreme_records.append(dict(
                    var=v,
                    time=np.datetime_as_string(topN["time"].values[i], unit="D"),
                    lat=float(topN["lat"].values[i]),
                    lon=float(topN["lon"].values[i]),
                    value=float(topN.values[i]),
                    raw_obs=int(obs_count_src.sel(lat=topN["lat"].values[i],
                                                 lon=topN["lon"].values[i]).sum().item())
                ))

print()

# ------------------------------------------------------------------
# pretty-print table
# ------------------------------------------------------------------
colnames = [
    "var", "obs_px", "syn_px",
    "obs_min", "obs_p01", "obs_med", "obs_p99", "obs_max",
    "syn_min", "syn_p01", "syn_med", "syn_p99", "syn_max",
    "syn_p99/obs_p99", "max_abs_diff_obs"
]

if _HAVE_TAB:
    print(tabulate(rows, headers=colnames, floatfmt=".3g", tablefmt="github"))
else:
    # fallback plain
    print("\t".join(colnames))
    for r in rows:
        print("\t".join(str(x) for x in r))

print()

# ------------------------------------------------------------------
# time-coverage improvement
# ------------------------------------------------------------------
print("Time coverage (fraction of stripe with data) …")
stripe_area = stripe3d.sum(("lat","lon"))
for v in VARS:
    if v not in fil:
        continue
    if v == "log_chl":
        raw_cov = raw["chlor_a"].where(stripe3d).notnull().mean(("lat","lon"))
    else:
        raw_cov = raw[v].where(stripe3d).notnull().mean(("lat","lon"))
    fil_cov = fil[v].where(stripe3d).notnull().mean(("lat","lon"))
    # summary numbers
    rc = raw_cov.values
    fc = fil_cov.values
    print(f"  {v:7s} raw mean={rc.mean():.3f} filled mean={fc.mean():.3f} "
          f"(Δ={fc.mean()-rc.mean():+.3f}) "
          f"raw min={rc.min():.3f} max={rc.max():.3f}  filled min={fc.min():.3f} max={fc.max():.3f}")

print()

# ------------------------------------------------------------------
# extreme synthetic values
# ------------------------------------------------------------------
if extreme_records:
    df_ext = pd.DataFrame(extreme_records)
    df_ext = df_ext.sort_values("value", ascending=False)
    df_ext.to_csv(OUT_CSV, index=False)
    print(f"Top synthetic values written → {OUT_CSV} (first 10 shown)")
    print(df_ext.head(10).to_string(index=False))
else:
    print("No synthetic (filled) values found?")


print("\n✅ QC compare done.\n")

'''
(habs) yashnilmohanty@Prabhus-MBP-M1 habs-forecast % python scripts/helpers/qc_compare_input.py
🔹 opening datasets …
   stripe pixels: 1,626 (2.82% of regional grid)

▶ chlor_a …
▶ Kd_490 …
▶ nflh …
▶ log_chl …

| var     |   obs_px |   syn_px |   obs_min |   obs_p01 |   obs_med |   obs_p99 |   obs_max |   syn_min |   syn_p01 |   syn_med |   syn_p99 |   syn_max |   syn_p99/obs_p99 |   max_abs_diff_obs |
|---------|----------|----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-------------------|--------------------|
| chlor_a |     1626 |     1626 |    0.0566 |    0.192  |     1.37  |    30.9   |     86.3  |    0.192  |    0.237  |     2.59  |    30.9   |    30.9   |                 1 |                  0 |
| Kd_490  |     1626 |     1626 |    0.0236 |    0.0408 |     0.124 |     3.74  |      6.4  |    0.0408 |    0.0484 |     0.2   |     3.74  |     3.74  |                 1 |                  0 |
| nflh    |     1625 |     1626 |   -0.32   |    0.0238 |     0.177 |     0.735 |      2.63 |    0.0238 |    0.0238 |     0.209 |     0.735 |     0.735 |                 1 |                  0 |
| log_chl |     1626 |     1626 |   -2.87   |   -1.65   |     0.316 |     3.43  |      4.46 |   -1.65   |   -1.44   |     0.952 |     3.43  |     3.43  |                 1 |                nan |

Time coverage (fraction of stripe with data) …
  chlor_a raw mean=0.025 filled mean=0.028 (Δ=+0.003) raw min=0.010 max=0.028  filled min=0.028 max=0.028
  Kd_490  raw mean=0.025 filled mean=0.028 (Δ=+0.003) raw min=0.010 max=0.028  filled min=0.028 max=0.028
  nflh    raw mean=0.023 filled mean=0.028 (Δ=+0.005) raw min=0.004 max=0.028  filled min=0.028 max=0.028
  log_chl raw mean=0.025 filled mean=0.028 (Δ=+0.003) raw min=0.010 max=0.028  filled min=0.028 max=0.028

Top synthetic values written → /Users/yashnilmohanty/Desktop/HABs_Research/Data/Finalized/qc_ca_coastal_impute_extremes.csv (first 10 shown)
    var       time       lat         lon     value  raw_obs
chlor_a 2021-06-26 36.937500 -121.854164 30.945389        2
chlor_a 2021-06-26 39.104168 -123.770836 30.945389      757
chlor_a 2021-06-26 37.895832 -122.895836 30.945389      758
chlor_a 2021-06-18 37.895832 -122.895836 30.945389      758
chlor_a 2021-06-18 36.937500 -121.854164 30.945389        2
chlor_a 2021-06-26 41.520832 -124.104164 30.945389      572
chlor_a 2021-06-26 40.979168 -124.104164 30.945389       94
chlor_a 2021-06-26 39.229168 -123.812500 30.945389      712
chlor_a 2021-06-26 39.187500 -123.854164 30.945389      746
chlor_a 2021-06-26 39.187500 -123.812500 30.945389      756

✅ QC compare done.

(habs) yashnilmohanty@Prabhus-MBP-M1 habs-forecast % 

'''
