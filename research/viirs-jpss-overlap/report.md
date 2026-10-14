# VIIRS chlorophyll overlap over California: S-NPP vs NOAA-20 vs NOAA-21
Prepared 2026-10-13 (data retrieved 07:31–07:56Z). Research only: no repo edits, nobody contacted.
Working dir: research3/overlap/

## 0. Headline
* **S-NPP coverage gap.** `erdVHNchla1day` has **no time steps from 2026-06-15 through 2026-09-19**. Its 2026 axis jumps from 2026-06-14 to 2026-09-20. Because of that, the requested window (2026-07-01 onward) has only **12 days with all three sensors** (Sep 20, 22–26, 29, 30, Oct 1–4).
  * To get a usable sample before the rolling N20/N21 archives age out, I also archived and analysed every **2026-02-01..2026-06-30 day on which S-NPP exists**. That adds 86 more three-sensor days.
  * Results are given for three windows: "primary" (Jul 1 onward, as requested), "extended" (Feb–Jun) and "all".
* **NOAA-21 agrees with S-NPP better than NOAA-20 does on every pooled metric:**
  * smaller bias: −0.018 vs −0.045 log10;
  * lower RMSE: 0.095 vs 0.148;
  * higher r: 0.973 vs 0.961;
  * RMA slope near 1: 1.02 vs 1.20;
  * steadier week to week: weekly SD 0.018 vs 0.027.
* **Recommendation: NOAA-21 first, NOAA-20 to fill gaps.** Neither product is bias-corrected to S-NPP.
* **Acceptance:** r ≥ 0.9 passes, the implausible-value test passes and bias stability passes. Union coverage ≥ S-NPP fails narrowly: 76.5% of days against the 80% threshold.

## 1. Data provenance
| key | product | ERDDAP | variable | time stamp used |
|---|---|---|---|---|
| snpp | erdVHNchla1day | https://coastwatch.pfeg.noaa.gov/erddap/griddap/ | chla (long_name "Chlorophyll a") | 12:00Z daily stamp |
| n20 | noaacwN20VIIRSchlaSectorUSDaily | https://coastwatch.noaa.gov/erddap/griddap/ | chlor_a (long_name "Chlorophyll Concentration, OC3 Algorithm") | the single step for that UTC date, 18:2x–19:4xZ |
| n21 | noaacwN21VIIRSchlaSectorUSDaily | same | chlor_a (long_name "Chlorophyll Concentration, OCI Algorithm") | same |

* **Dimension order** (checked in the .dds) is time, altitude(1), latitude, longitude for all three products. Latitude is descending on both servers.
* **Lattice check.** Both servers put cell centres at …x.xx125/…x.xx625/…x.xx875/…x.xx375 on a 0.0075° lattice. I requested lattice-aligned bounds so that both servers pick the same stride-4 cells:
  * lat (42.49875):4:(32.00625) → 350 rows, 42.49875 to 32.02875;
  * lon (−126.99375):4:(−116.00625) → 367 columns, −126.99375 to −116.01375;
  * about 3.3 km by 2.7 km sampling.
  * The largest coordinate difference across all 490 files was 1.4e-14°, well inside the 1e-6° requirement.
* **Example URLs:**
  * `https://coastwatch.pfeg.noaa.gov/erddap/griddap/erdVHNchla1day.nc?chla[(2026-10-04T12:00:00Z)][(0.0)][(42.49875):4:(32.00625)][(-126.99375):4:(-116.00625)]`
  * `https://coastwatch.noaa.gov/erddap/griddap/noaacwN21VIIRSchlaSectorUSDaily.nc?chlor_a[(2026-10-04T19:37:51Z)][(0.0)][(42.49875):4:(32.00625)][(-126.99375):4:(-116.00625)]`
  * Brackets are percent-encoded in the manifest.
* **Time mapping.** I listed each time axis and mapped every step to its UTC date. There was one step per date and no duplicates. The mapping is in `time_map_*.csv`.
* **Archive:** `raw/<snpp|n20|n21>/<YYYY-MM-DD>.nc`, with 112 S-NPP, 192 N20 and 186 N21 files. Total 258,338,448 bytes:
  * S-NPP: 58.9 MB
  * N20: 101.3 MB
  * N21: 98.1 MB
* **Coastline reference:** `raw/etopo/etopo180_CA.nc` (ETOPO1 `etopo180` from pfeg, 880 kB).
* **Manifest:** `manifest.csv`, one row per product-date, with URL, HTTP status, bytes, retry count, seconds and retrieval UTC.
  * All requests eventually returned HTTP 200. There were 16 transient 503s, each retried after a backoff.
  * Requests were sequential with at least 1.5 s between them.
* **Dates covered:**
  * Primary, 2026-07-01..2026-10-06 (98 dates). N20 and N21 cover Jul 1–Oct 5; S-NPP has 13 days (Sep 20, 22–26, 29, 30, Oct 1–4, Oct 6).
  * Extended: 99 S-NPP dates between 2026-02-01 and 2026-06-14. The S-NPP 2026 axis has 29 Jan, 12 Feb, 21 Mar, 24 Apr, 30 May and 12 Jun days. N20/N21 days with no S-NPP before July were not archived.
* **Missing days and why.** Each of these is absent from the server's time axis, not a failed download:
  * S-NPP: 2026-06-15..2026-09-19 (outage or not reprocessed on ERDDAP), plus Sep 21, Sep 27, Sep 28 and Oct 5. There are also scattered gaps Feb–Jun.
  * N20: 2026-04-09, 05-06, 05-10, 05-19, and 10-06 (not yet produced).
  * N21: 2026-02-01..02-08 (archive starts 02-09), 05-10, 06-03, and 10-06.
  * Some days exist on the axis but are almost empty: N21 on 05-09 (0.0% coverage), 05-17 (3.0%) and 06-12 (2.8%). These look like partial passes or outages, not clouds, because S-NPP and N20 were 40–50% covered on 05-17 and 06-12.
* **Three-sensor days:** 98 in total (primary 12, extended 86).
* **Scripts:** `download.py` (fetch, time mapping, manifest) and `analyze.py` (masking, pairing, metrics). Outputs are in `results/`:
  * `daily_metrics.csv`
  * `pooled_metrics.csv`
  * `stratified_metrics.csv`
  * `coverage_implausible.csv`
  * `weekly_stability.csv`
  * `summary.json`
  * `analysis_stdout.txt` (full printout)

## 2. Methods
* **Validity.** I set to invalid any value that is NaN or non-finite, equal to the fill value (−999, declared as `_FillValue`/`missing_value` and removed by CF decoding), or ≤ 0. No other quality filter was applied:
  * l2_flags and qa_score were not used;
  * the products are taken as served.
* **Ocean mask.** A cell is ocean if:
  * the nearest ETOPO1 node has altitude < 0;
  * and the cell had a valid retrieval from at least one sensor on at least one of the 197 dates.
  * The second condition removes Death Valley and other below-sea-level land.
  * Result: 61,784 sampled ocean cells (north 7,765, central 28,774, south 25,245).
  * Valid retrievals on cells ETOPO calls land (lakes, bays, coast pixels within about 1 arc-minute of the coastline) were dropped from all statistics.
  * The domain includes Mexican waters between 32.0 and 32.5°N and the San Francisco Bay outlet.
* **Distance to shore.** I ran a Euclidean distance transform (scipy `distance_transform_edt`) on the ETOPO1 1-arc-minute grid:
  * land = altitude ≥ 0;
  * spacing in km: dy = 1.853, dx = 1.853·cos 37.25°;
  * each sampled cell takes the nearest node's distance.
  * Bands (sampled ocean cells): 0–5 km (1,175), 5–20 km (3,801), 20–100 km (14,976), >100 km (41,832).
  * Resolution is about 1.5–2 km, so the 0–5 km band is approximate and includes some partly land-contaminated pixels.
* **Pairing.** Cell-to-cell on the identical lattice, same UTC date. S-NPP is a 12Z-stamped daily composite; N20/N21 are daily merges near 19Z. The overpasses are tens of minutes to a few hours apart, so some of the scatter is real change in the water.
* **Metrics.** All use d = log10(A) − log10(B).
  * Median and MAD (median |d − median|, unscaled); mean; RMSE = sqrt(mean d²).
  * Pearson r on log10.
  * Type-II reduced-major-axis slope = sign(r)·sd(log A)/sd(log B), with A on the y axis.
* **Coverage** = valid ocean cells ÷ 61,784, in percent, per day.
* **Implausible** = fraction of valid ocean values above 200 mg m^-3. The fraction above 100 is also reported.
* **Strata:**
  * region: north ≥ 40°N, central 34.5–40°N, south < 34.5°N;
  * distance band;
  * chl class from the S-NPP value. For the N21/N20 pair, S-NPP is not in the pair, so the class comes from N20.
* **Stability.** I pooled the pairs within ISO weeks (Mon–Sun) and took the median d, requiring n ≥ 50. My summary statistics use only weeks with n ≥ 5,000.
* **How I applied "stable"** (this rule is mine):
  * stable = SD of weekly medians ≤ 0.03 log10 (about 7%);
  * and no weekly median more than 0.1 log10 from zero.

## 3. Pooled metrics
"all" = 2026-02-01..2026-10-05. Ratio is A/B. Primary window in brackets.

| pair | n pairs | days | median log10 (ratio) | MAD | RMSE log10 | r(log10) | RMA slope |
|---|---|---|---|---|---|---|---|
| N20/SNPP all | 1,178,016 | 92 | −0.045 (0.901) | 0.055 | 0.148 | 0.961 | 1.204 |
| [N20/SNPP primary] | 141,485 | 12 | −0.063 (0.865) | 0.045 | 0.133 | 0.958 | 1.188 |
| N20/SNPP extended | 1,036,531 | 80 | −0.042 (0.907) | 0.056 | 0.149 | 0.961 | 1.206 |
| N21/SNPP all | 1,068,299 | 89 | −0.020 (0.956) | 0.042 | 0.095 | 0.973 | 1.021 |
| [N21/SNPP primary] | 131,368 | 12 | −0.029 (0.935) | 0.037 | 0.104 | 0.961 | 1.071 |
| N21/SNPP extended | 936,931 | 77 | −0.018 (0.959) | 0.042 | 0.093 | 0.975 | 1.015 |
| N21/N20 all | 1,785,298 | 169 | +0.040 (1.097) | 0.046 | 0.129 | 0.963 | 0.898 |
| [N21/N20 primary] | 898,773 | 93 | +0.047 (1.115) | 0.043 | 0.120 | 0.963 | 0.929 |
| N21/N20 extended | 886,525 | 76 | +0.033 (1.079) | 0.048 | 0.139 | 0.962 | 0.876 |

* **Per day,** counting only days with at least 1,000 pairs:
  * N20/SNPP: 80 days; median daily r 0.960, minimum 0.839; r ≥ 0.9 on 77 of 80 days; SD of daily median bias 0.051.
  * N21/SNPP: 72 days; median daily r 0.976, minimum 0.930; r ≥ 0.9 on 72 of 72 days; SD of daily median bias 0.026.
  * N21/N20: 140 days; median daily r 0.966, minimum 0.747; r ≥ 0.9 on 138 of 140 days.
* **Coverage** (% of 61,784 ocean cells, 98 three-sensor days, mean / median):

| | S-NPP | N20 | N21 | union N21∪N20 | union of all three |
|---|---|---|---|---|---|
| all | 26.1 / 23.9 | 24.3 / 19.7 | 25.3 / 21.3 | 32.5 / 30.1 | 37.2 / 35.6 |
| primary (12 d) | 25.3 / 22.3 | 26.3 / 24.8 | 24.6 / 23.7 | 31.5 / 31.0 | 35.2 / 35.5 |

* **How often each source matches or beats S-NPP coverage** (98 days):
  * union N21∪N20 ≥ S-NPP on 75 days (76.5%); primary window 9 of 12 (75%);
  * N21 ≥ S-NPP on 45 days; N20 ≥ S-NPP on 36 days; N21 ≥ N20 on 47 days.
  * The union exceeds S-NPP coverage by a median of 2.4 points (mean 6.4); the median ratio is 1.09.
  * On average N21 alone supplies 71% of the union cells and N20 alone 72%.
* **Days the union falls short of S-NPP** (23):
  * 03-09, 03-24, 03-31, 04-04, 04-14, 04-15, 04-23, 05-01, 05-07, 05-09, 05-11, 05-16, 05-17, 05-21, 05-22, 05-27, 05-31, 06-06, 06-08, 06-12, 09-20, 09-22, 10-02.
  * Several are N21 or N20 outages or partial days (05-09, 05-17, 06-12) or nearly cloud-covered days (S-NPP < 5%).
  * The worst is 03-24: S-NPP 73.2% against a union of 48.2%.
* **Values above 200 mg m^-3** (pooled, all windows):
  * S-NPP 0.0016% (above 100: 0.0038%);
  * N20 0.0038% (above 100: 0.026%);
  * N21 0.0049% (above 100: 0.017%).
  * Maximum values were about 386–388 mg m^-3 for all three sensors.
  * 99th percentiles: S-NPP 6.3, N20 10.2, N21 7.9.
  * Median chl: S-NPP 0.34, N20 0.28, N21 0.33.

## 4. Stratified pooled metrics (window "all")
Each entry is median log10 bias / RMSE / r / RMA slope.

| stratum | N20/SNPP | N21/SNPP | N21/N20 |
|---|---|---|---|
| north ≥40°N | −0.020 / 0.154 / 0.937 / 1.19 | −0.024 / 0.100 / 0.965 / 0.99 | +0.023 / 0.138 / 0.940 / 0.88 |
| central 34.5–40 | −0.020 / 0.143 / 0.954 / 1.19 | −0.014 / 0.106 / 0.964 / 1.03 | +0.032 / 0.137 / 0.957 / 0.91 |
| south <34.5 | −0.084 / 0.151 / 0.941 / 1.20 | −0.025 / 0.075 / 0.971 / 1.02 | +0.057 / 0.114 / 0.959 / 0.91 |
| 0–5 km | +0.010 / 0.260 / 0.917 / 1.29 | −0.021 / 0.185 / 0.933 / 1.09 | +0.004 / 0.244 / 0.917 / 0.87 |
| 5–20 km | −0.030 / 0.235 / 0.938 / 1.31 | −0.019 / 0.158 / 0.950 / 1.08 | +0.035 / 0.211 / 0.934 / 0.88 |
| 20–100 km | −0.039 / 0.143 / 0.957 / 1.19 | −0.026 / 0.096 / 0.970 / 1.00 | +0.034 / 0.131 / 0.952 / 0.92 |
| >100 km | −0.051 / 0.121 / 0.955 / 1.19 | −0.017 / 0.068 / 0.972 / 1.00 | +0.044 / 0.095 / 0.969 / 0.89 |
| chl <0.3 (class) | −0.073 / 0.137 / 0.829 / 1.67 | −0.014 / 0.063 / 0.902 / 1.26 | +0.059 / 0.098 / 0.908 / 0.89 |
| 0.3–1 | −0.032 / 0.089 / 0.877 / 1.16 | −0.018 / 0.069 / 0.898 / 0.96 | +0.021 / 0.088 / 0.846 / 1.07 |
| 1–3 | +0.011 / 0.184 / 0.708 / 1.86 | −0.056 / 0.125 / 0.781 / 1.39 | −0.030 / 0.174 / 0.565 / 1.54 |
| 3–10 | +0.143 / 0.368 / 0.410 / 2.58 | −0.011 / 0.249 / 0.509 / 2.08 | −0.118 / 0.276 / 0.399 / 1.89 |
| >10 | +0.087 / 0.512 / −0.01 / n/m | −0.068 / 0.367 / 0.17 / n/m | −0.351 / 0.533 / 0.18 / n/m |

* **Pair counts per chl class:**
  * N20/SNPP: 495k, 474k, 159k, 47k, 2.9k;
  * N21/SNPP: 423k, 435k, 158k, 49k, 2.8k.
* **Primary-window strata** are in `results/stratified_metrics.csv` and look much the same. N20's south-region bias is −0.083 there.
* **Within-class r is low by construction.** A narrow class range shrinks r, and slopes inside a class are inflated by regression to the mean. Even so, the class pattern matters:
  * N20 reads low in clear water (−0.07 log10 below 0.3 mg m^-3) and high in moderate to high chl (+0.14 at 3–10). That matches its RMA slope of about 1.2, a stretched dynamic range compared with S-NPP.
  * N21 stays within ±0.06 log10 of S-NPP in every class, with lower RMSE.
  * Agreement degrades for everyone in bloom-level and nearshore water: RMSE 0.18–0.37 log10 above 1 mg m^-3 and within 20 km of shore. These are exactly the waters that matter for HAB work.
  * The low-chl offset (OC3 vs OCI blending below about 0.25 mg m^-3 would produce this kind of pattern) fits the metadata labels, but I have not verified it.

## 5. Stability
Weekly median log10 ratio, pooled pairs within each week. n is in thousands of pairs.

| week (Mon) | N20/SNPP | n | N21/SNPP | n | N21/N20 | n |
|---|---|---|---|---|---|---|
| 01-26 | −0.057 | 24 | – | – | – | – |
| 02-02 | −0.057 | 159 | – | – | – | – |
| 02-09 | −0.074 | 27 | −0.049 | 21 | +0.049 | 21 |
| 03-02 | −0.017 | 91 | −0.029 | 92 | +0.034 | 98 |
| 03-09 | −0.039 | 9 | +0.010 | 8 | +0.084 | 6 |
| 03-16 | −0.048 | 215 | −0.013 | 208 | +0.044 | 216 |
| 03-23 | −0.051 | 122 | −0.030 | 129 | +0.029 | 117 |
| 03-30 | −0.033 | 104 | −0.028 | 93 | +0.008 | 85 |
| 04-06 | −0.081 | 25 | −0.033 | 28 | +0.037 | 23 |
| 04-13 | −0.029 | 43 | +0.009 | 52 | +0.041 | 62 |
| 04-20 | −0.057 | 30 | −0.032 | 33 | +0.015 | 29 |
| 04-27 | −0.078 | 4 | −0.003 | 4 | +0.042 | 19 |
| 05-04 | −0.111 | 7 | +0.009 | 17 | +0.085 | 10 |
| 05-11 | −0.006 | 50 | −0.024 | 42 | +0.025 | 41 |
| 05-18 | −0.045 | 13 | −0.005 | 36 | +0.035 | 53 |
| 05-25 | −0.017 | 37 | −0.005 | 59 | +0.011 | 39 |
| 06-01 | −0.009 | 40 | +0.004 | 63 | +0.008 | 31 |
| 06-08 | −0.025 | 37 | −0.029 | 52 | +0.044 | 38 |
| 06-29 … 09-07 | (no S-NPP) | | (no S-NPP) | | +0.019, +0.036, +0.071, +0.048, +0.064, +0.013, +0.044, +0.058, +0.027, +0.061, +0.034 | 5–151 |
| 09-14 (Sep 20 only) | +0.001 | 0.2 | −0.088 | 0.4 | +0.032 | 29 |
| 09-21 | −0.059 | 68 | −0.031 | 64 | +0.043 | 86 |
| 09-28 | −0.066 | 73 | −0.027 | 67 | +0.051 | 95 |
| 10-05 | – | – | – | – | +0.061 | 9 |

Summary over weeks with n ≥ 5,000:

| pair | weeks | mean | SD | min | max | range |
|---|---|---|---|---|---|---|
| N20/SNPP | 19 | −0.046 | 0.027 | −0.111 | −0.006 | 0.104 |
| N21/SNPP | 17 | −0.018 | 0.018 | −0.049 | +0.010 | 0.059 |
| N21/N20 | 30 | +0.041 | 0.020 | +0.008 | +0.085 | 0.077 |

* The N21–N20 offset holds steady at +0.01 to +0.09 through July–September, the period S-NPP is missing, so neither product drifted relative to the other during the gap.
* The September–October N20/SNPP and N21/SNPP biases (−0.06 and −0.03) are close to the spring values.
* No trend is visible, but the record is about 8 months long and has a 3-month hole in the S-NPP reference.

## 6. Acceptance against the proposed criteria
| criterion | N21 vs S-NPP | N20 vs S-NPP | verdict |
|---|---|---|---|
| \|median log10 bias\| documented and stable | −0.020 (ratio 0.956); weekly SD 0.018, all weeks within ±0.05 | −0.045 (ratio 0.901); weekly SD 0.027, one week at −0.111 (n = 7k) | **PASS** for N21. **Marginal pass** for N20: the SD test passes, and the range test fails on one low-n week. |
| r(log10) ≥ 0.9 | 0.973 pooled; every day ≥ 0.93 | 0.961 pooled; 77 of 80 days ≥ 0.9 | **PASS** for both. Within-class r drops to 0.4–0.8 above 1 mg m^-3. |
| union N21∪N20 coverage ≥ S-NPP on ≥ 80% of days | 75 of 98 days = 76.5% (primary 9 of 12 = 75%) | (same union) | **FAIL**, by 3.5 points. Mean union coverage is 32.5% against S-NPP's 26.1%, so on average the union clearly covers more. |
| fraction > 200 mg m^-3 ≤ 1% | 0.0049% | 0.0038% | **PASS** by more than two orders of magnitude (S-NPP 0.0016%). |

The coverage failure has three main causes:
* single-sensor outages or partial days (N21 on 05-09, 05-17 and 06-12);
* spring days when S-NPP's 12Z-stamped composite simply saw more clear sky (03-24, 04-04, 06-06);
* nearly cloud-covered days where a few cells decide the result.

If the threshold were applied on a tolerance basis (union ≥ 0.95 × S-NPP), more days would pass. I have not recomputed under that rule because it was not one of the proposed criteria.

## 7. Recommendation: NOAA-21 first, NOAA-20 as gap-fill
Reasons:
1. **Consistency with the S-NPP record.**
   * N21's median bias is −0.02 log10 (about −4.5%), about half of N20's −0.045 (about −10%).
   * N21's RMSE is 0.095 against 0.148.
   * N21's RMA slope is about 1.0 against about 1.2, so N21 keeps S-NPP's dynamic range while N20 stretches it.
   * N21 is better in every region and every distance band. The south is where the difference is largest: N21 −0.025 vs N20 −0.084.
2. **Stability.** N21's weekly SD is 0.018 against 0.027 for N20, and its daily median-bias SD is 0.026 against 0.051. Every N21 day has r ≥ 0.93.
3. **Coverage.** Taken alone the two are similar: mean 25.3% for N21 vs 24.3% for N20; N21 ≥ S-NPP on 45 days against 36. Most of the coverage benefit comes from the union, which is why N20 should stay as gap-fill.
   * Order matters only where both are valid. In those cells N21 takes priority under this recommendation.
4. **Algorithm labels.** The metadata says N20 is "OC3" and N21 is "OCI". If that is right, N21 would share the OCI-style clear-water treatment of the NASA/NOAA standard products. N21's small low-chl bias (−0.014 vs N20's −0.073 below 0.3 mg m^-3) fits that, but **I have not verified it.** Confirm with NOAA STAR/CoastWatch documentation before relying on it.
5. **Operational caveat.** N21 had a few near-empty days (05-09, 05-17, 06-12) and its archive starts on 02-09. The fill logic must fall through to N20 at the cell level, not only when a whole day is missing.

* **Do not bias-correct either product to S-NPP.**
  * Document the offsets in provenance: N20 is about 10% lower than S-NPP, N21 about 4% lower, and N21 is about 10% higher than N20.
  * Keep the sensor of origin per cell, so that the seam between N21 and N20 cells can be audited. The N21–N20 offset is +0.04 log10 and larger in clear water.

## 8. Caveats
* **The S-NPP reference has a 3-month hole** (Jun 15–Sep 19). The requested July-onward window holds only 12 three-sensor days and about 130–140k pairs, so the primary-window numbers are thin. The Feb–Jun extension is spring/early summer, so seasonal representativeness is limited. There is no July–August S-NPP comparison at all.
* **These are NRT products, not science quality.** Calibration, algorithms and masks can change without notice, and the rolling N20/N21 archives may be reprocessed or dropped. The NetCDFs archived here are the record of what was served on 2026-10-13.
* **Stride-4 sampling** (about 3 km) uses 1 in 16 native cells. Spatial autocorrelation means effective sample sizes are far smaller than n, and sub-3-km nearshore structure is undersampled.
* **No quality filtering** beyond validity: l2_flags and qa_score were not used. Stray light, glint and cloud-adjacency residuals stay in, especially nearshore. The 0–5 km band is also approximate because of the 1-arc-minute ETOPO coastline.
* **Time mismatch.** The S-NPP composites are stamped 12Z; the N20/N21 merges are at about 18:3x–19:4xZ. Diurnal and advective change adds to the apparent disagreement.
* **No in-situ validation.** All metrics measure agreement between sensors, not accuracy. S-NPP is used as the continuity reference, not as truth. Satellite chlorophyll is not a toxin measurement.
* **Coverage depends on the mask.** "Ocean" requires at least one valid retrieval somewhere in the 197 days, which slightly raises coverage percentages for all sensors equally.
* **Weekly stability** is based on uneven weekly sample sizes. Weeks under 5,000 pairs are shown but left out of the SD and range statistics.
