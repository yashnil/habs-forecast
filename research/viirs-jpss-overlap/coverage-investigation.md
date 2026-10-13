# Coverage criterion investigation: NOAA-21 ∪ NOAA-20 vs S-NPP over California
Prepared 2026-10-13. Research only: no repo edits, nobody contacted. Follows `report.md` (same archive, masks and definitions).
Script: `coverage_investigation.py`. Tables: `results/coverage_*.csv`.

This concerns only the VIIRS 750 m chlorophyll layer. It does not concern C-HARM: C-HARM is NOAA's model, and this migration does not change its S-NPP input.

## 0. Headline
* **The criterion failed and is not waived.** The test was "union N21∪N20 coverage ≥ S-NPP coverage on ≥ 80% of days". The result is 75 of 98 days, **76.5%, below the 80% threshold**. Nothing in this note changes that verdict.
  * The failure is not driven by the near-shore band. Within 5 km of shore the union matches or beats S-NPP on 88% of days, with mean coverage of 48.3% against 33.8%. The lost cells are spread across the domain in proportion to ocean area: 1.8% of them lie within 5 km of shore, and 1.9% of the ocean does.
* **Causes of the 23 failing days:**

  | cause | days |
  |---|---|
  | swath geometry: straight-edged swath or stripe gaps in N20/N21 over water S-NPP saw | 7 |
  | S-NPP's pass simply saw more clear sky | 7 |
  | single-platform outage or partial day (all N21) | 5 |
  | near-total cloud (S-NPP < 10%) | 4 |
  | near-shore masking | 0 |

  5 of the 23 days fall short by less than 0.5 points.
* **The shortfall is usually small.** On the failing days, the union is a median **1.1 points** below S-NPP:
  * p25 0.6, p75 4.5, p90 8.3, max 25.0 (2026-03-24);
  * median ratio union/S-NPP 0.91.
  * Across all 98 days the deficit exceeds 1 point on 15 days, 2 points on 8 days and 5 points on 4 days.
* **On average the union covers more ocean than S-NPP:** mean 32.5% against 26.1%, median 30.1% against 23.9%; summed over all days, ratio 1.25.
  * It covers 82% of the cells S-NPP covered.
  * It also adds 669,825 cell-days S-NPP lacked, against 280,739 S-NPP cell-days it lacks.
* **The per-day "≥" test is close to a cloud lottery.** Comparable tests behave like this:
  * N21 ≥ N20 on 48% of days;
  * S-NPP against itself one day later, 51%;
  * the same "two-platform union vs the third platform" test with the sensors permuted passes on 88% and 94%.
  * Tolerance variants of the original test: union ≥ 0.95×S-NPP passes on 83.7% of days, ≥ 0.90× on 88.8%, ≥ S-NPP − 2 points on 91.8%.
  * These variants are **not** used to declare a pass.
* **The 4 km NOAA-20 fallback is the same NOAA-20 data, binned coarser.** It cannot fill NOAA-20 outages: it is just as empty on N20's empty days.
  * Measured as area, it would put "something" over 46–50% of the S-NPP-only cells on failing days.
  * Union plus 4 km footprint ≥ S-NPP on 83 of 86 days. That is a resolution effect (a 4 km cell counts if any part of it was clear), not 750 m coverage.
  * It must stay a separately labelled layer and is not a fix for this criterion.
* **User-visible consequence.** On about 1 day in 4 (23 of 98; 23 of 90 days with any data), the 750 m layer will show less ocean than S-NPP would have. The shortfall is typically about 1 point of the California ocean area, and roughly 1 day in 12 it is 2 points or more. It falls mostly offshore (more than 100 km out) in the central and southern regions and is most frequent in May (10 of 27 May days failed). On the other 3 days in 4 the layer shows more.

## 1. Methods
* **Data and definitions** are identical to `analyze.py`:
  * the same raw stride-4 archive (`research3/overlap/raw/{snpp,n20,n21}`);
  * validity = finite and > 0;
  * ocean = ETOPO1 altitude < 0 and valid at least once in any of the three products, giving 61,784 cells;
  * regions north ≥ 40°N, central 34.5–40°N, south < 34.5°N;
  * distance-to-shore bands from the ETOPO EDT.
  * The union is cell-wise OR of N21 and N20 validity. This is equivalent to "N21 first, N20 fills": coverage does not depend on priority.
  * The script asserts the 61,784-cell ocean mask and the 98 three-sensor days. It reproduces `daily_metrics.csv` (75 of 98).
* **Per day,** I computed the following for each stratum (region, distance band, ≤ 5 km vs > 5 km, > 20 km):
  * coverage of S-NPP, N20, N21 and the union;
  * counts of S-NPP-only and union-only cells;
  * clear-area agreement (share of S-NPP-valid cells also valid in N20 or N21);
  * a "JPSS void" share: S-NPP-only cells with no N20/N21 retrieval within ±10 lattice cells (about ±30 km) while S-NPP is ≥ 50% valid in that window;
  * a cloud-edge "isolation" share: valid cells with ≤ 4 of 8 neighbours valid;
  * the N20/N21 time stamps from `time_map_*.csv`.
* **Cause classification** applies ordered rules (listed in the script); the first match is the primary cause and later matches are recorded:
  * **a, outage/partial:** one JPSS sensor ≤ 0.5% while S-NPP ≥ 2%, or below 25% of the other JPSS sensor in both coverage and agreement with S-NPP;
  * **b, near-total cloud:** S-NPP < 10%;
  * **e, swath geometry:** a straight-edged JPSS no-data boundary over water S-NPP saw, or JPSS-void share ≥ 25%;
  * **d, near-shore:** union ≥ S-NPP once cells ≤ 5 km from shore are dropped, or ≥ 50% of S-NPP-only cells within 20 km of shore;
  * **c, otherwise:** different clear sky.
  * **One input is manual.** I inspected the validity maps of all 23 failing days by eye; the script's optional 4th argument redraws them. The straight-edge observations are recorded in the script (`VISUAL`) and in the CSV.
* **4 km product:** `nesdisVHNnoaa20chlaDaily` (pfeg).
  * The .dds shows dimensions time, altitude(1), latitude, longitude. Global extent; latitude descending.
  * The global attributes say platform NOAA-20, title "NOAA-20 VIIRS, Near Real-Time, Global 4km, Level 3", chlor_a "OC3 Algorithm". The `summary` attribute says "Suomi-NPP", which looks like a metadata copy error.
  * I requested the 98 three-sensor dates sequentially, at least 1.5 s apart, with a backoff on 503s:
    * **Retrieved:** 86 files (HTTP 200, 340,804 bytes each). 4 needed one retry after a 503.
    * **Not on the time axis:** 12 dates (03-20, 03-21, 03-27…04-04 and 05-04).
    * The URL and status for every date are logged in `viirs-work/coverage/manifest_n20_4km.csv`, with the .dds/.das saved next to it.
  * **Stored in** `research3/overlap/raw/n20_4km/`.
  * **Mapping to the 750 m cells.** Each 750 m sample cell is mapped to the 4 km cell containing it; the centre offset is at most half a cell, which the script asserts.
  * **Two measures:**
    * "footprint" = share of the 61,784 ocean sample cells lying in a valid 4 km cell;
    * "native" = valid share of the 39,620 4 km cells that contain at least one ocean sample.

## 2. Which days fail and why
Coverage is in % of ocean cells. The time stamps are the N20/N21 daily-merge stamps (UTC). Every S-NPP stamp is 12:00Z.

The near-shore column is the share of S-NPP-only cells within 20 km of shore; 8% of ocean cells are.

| date | N20/N21 stamp | S-NPP | N20 | N21 | union | deficit | union/S-NPP | primary cause | where S-NPP-only cells are | near-shore | map notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 03-09 | 19:16/19:11 | 26.3 | 19.4 | 16.1 | 26.0 | 0.27 | 0.99 | c clear sky (marginal) | central | 11% | no edges; S-NPP more cells in northern broken cloud |
| 03-24 | 18:41/19:28 | 73.2 | 38.4 | 36.9 | 48.2 | **25.01** | 0.66 | e swath geometry | central | 4% | N20 and N21 both end at straight swath edges; clear water west of ~123°W seen only by S-NPP (68% of S-NPP-only cells in a JPSS void) |
| 03-31 | 19:04/19:00 | 12.3 | 9.3 | 7.2 | 11.2 | 1.11 | 0.91 | c clear sky | central | 2% | similar cloud pattern |
| 04-04 | 19:26/19:23 | 56.0 | 42.0 | 33.8 | 55.9 | 0.16 | 1.00 | e swath geometry (marginal) | central | 4% | N20 straight diagonal gap near 121°W; N21 striped loss near 35°N |
| 04-14 | 19:34/18:51 | 24.0 | 17.2 | 20.5 | 23.0 | 0.99 | 0.96 | c clear sky | south | 2% | no edges |
| 04-15 | 19:22/19:17 | 23.8 | 13.9 | 11.3 | 19.2 | 4.61 | 0.81 | c clear sky | central | 18% | N21 lacks central coast 34–37°N |
| 04-23 | 19:18/19:20 | 28.3 | 8.4 | 21.7 | 27.5 | 0.75 | 0.97 | e swath geometry | south | 6% | N20 near-empty with straight edge and stripes |
| 05-01 | 19:22/19:16 | 3.7 | 1.7 | 1.3 | 2.4 | 1.32 | 0.64 | b near-total cloud | south | 51% | — |
| 05-07 | 19:10/19:05 | 4.0 | 2.9 | 0.1 | 2.9 | 1.02 | 0.74 | a outage (N21) | central | 35% | N21 empty; also near-total cloud |
| 05-09 | 18:28/18:24 | 7.4 | 0.8 | 0.0 | 0.8 | 6.58 | 0.11 | a outage (N21) | central | 6% | N21 empty, N20 near-empty, both stamped earlier than usual; the 4 km N20 is also empty |
| 05-11 | 19:30/19:28 | 12.4 | 8.6 | 2.4 | 9.5 | 2.93 | 0.76 | c clear sky | north | 9% | N21 misses the clear patch at 41–42°N |
| 05-16 | 19:35/19:33 | 47.1 | 21.4 | 28.4 | 38.4 | 8.74 | 0.81 | e swath geometry | central | 2% | N21 straight western swath edge; N20 cloudier offshore |
| 05-17 | 19:22/19:16 | 50.0 | 43.3 | 3.0 | 45.5 | 4.40 | 0.91 | a outage (N21) | central | 8% | N21 partial day |
| 05-21 | 18:56/18:56 | 8.8 | 3.8 | 7.0 | 7.8 | 1.00 | 0.89 | b near-total cloud | north | 15% | — |
| 05-22 | 19:26/19:22 | 2.4 | 0.9 | 0.7 | 1.3 | 1.10 | 0.54 | b near-total cloud | central | 38% | — |
| 05-27 | 19:30/19:28 | 28.9 | 15.1 | 9.5 | 19.1 | **9.82** | 0.66 | e swath geometry | south | 7% | N21 straight western swath edge; N20 cloudier offshore |
| 05-31 | 19:08/19:08 | 48.5 | 28.4 | 47.4 | 48.5 | 0.03 | 1.00 | e swath geometry (marginal) | central | 15% | N20 straight eastern edge and stripe |
| 06-06 | 19:40/18:06 | 60.5 | 50.1 | 35.2 | 55.8 | 4.71 | 0.92 | e swath geometry | south | 3% | N20 diagonal stripe gap; N21 straight western edge (stamped 18:06) |
| 06-08 | 19:11/19:05 | 4.4 | 3.6 | 0.3 | 3.8 | 0.52 | 0.88 | a outage (N21) | south | 37% | N21 empty; also near-total cloud |
| 06-12 | 19:32/19:27 | 49.3 | 47.2 | 2.8 | 47.5 | 1.75 | 0.96 | a outage (N21) | central | 9% | N21 near-empty |
| 09-20 | 19:11/19:09 | 2.2 | 0.9 | 1.3 | 2.0 | 0.20 | 0.91 | b near-total cloud (marginal) | south | 16% | — |
| 09-22 | 19:26/19:17 | 27.7 | 19.9 | 18.7 | 25.9 | 1.78 | 0.94 | c clear sky | central | 5% | S-NPP more scattered offshore cells 33–36°N |
| 10-02 | 19:35/19:28 | 22.6 | 17.6 | 18.7 | 22.1 | 0.43 | 0.98 | c clear sky (marginal) | central | 16% | thin straight gap lines in both, small |

Counts by primary cause:

| cause | days | median deficit (pts) | sum of deficits (pts) |
|---|---|---|---|
| e swath geometry | 7 | 4.7 | 49.2 |
| c S-NPP saw more clear sky | 7 | 1.1 | 12.1 |
| a single-platform outage/partial (N21 every time) | 5 | 1.8 | 14.3 |
| b near-total cloud | 4 | 1.05 | 3.6 |
| d near-shore masking | 0 | – | – |

**(a) Outages and partial days.** N21 was empty or near-empty on 05-07, 05-09, 05-17, 06-08 and 06-12.
* On 05-17 and 06-12 the sky was clear enough for S-NPP and N20 to see 43–50%. Those are processing or pass gaps, not clouds.
* N20 is the fill on those days, and the union falls short by 1.8–4.4 points.
* S-NPP has outages too: it was empty on 04-21 (N21 30%) and 06-07 (N20 5%, N21 3%). The union wins those days.

**(b) Near-total cloud.** On 4 days S-NPP < 10% decides the result, plus 3 more already counted under (a) (05-07, 05-09, 06-08).
* Between about 120 and 820 cells settle the comparison.
* Deficits are 0.2–1.3 points.

**(c) Overpass time and clear sky.**
* **The time stamps are not overpass times.** The S-NPP stamp is a nominal 12:00Z composite stamp. The N20/N21 stamps (18:06–19:43Z on failing days) are the centre times of the US-sector daily merges.
* **Real overpass offsets cannot be measured from this archive.** None of the products carries a per-pixel or per-granule time in the subsets. So I cannot test from this archive whether S-NPP's late-life orbit puts it over California at a different local time, or whether marine-layer timing explains the differences.
* **What the data do show on the 7 "c" days:**
  * The S-NPP-only cells are not clustered in a region or band. They are mostly central and offshore, in proportion to ocean area.
  * The cloud fields look the same shape in all three products, with slightly different extents.
  * **S-NPP does not keep systematically more cloud-edge cells.** The isolated-cell share is similar for all three: S-NPP 12.9%, N20 12.3%, N21 14.1%. On failing days the S-NPP-only cells are no more isolated than the union-only cells (32% vs 35%).
* **The difference is mostly not cloud.** S-NPP alone does average slightly more coverage than either JPSS sensor alone: 26.1% against 24.3% (N20) and 25.3% (N21). However, the swath-geometry category (e) shows that most of the large deficits come from where each platform's swath edges fall, not from cloud.

**(e) Swath geometry** (the largest contributor to deficit size).
* **What the maps show.** On 7 failing days, N20 and/or N21 have straight-edged no-data boundaries (swath edges, inter-orbit gaps, scan stripes) across open water that S-NPP saw.
* **Why the union cannot always fill them.** The two JPSS platforms share an orbit plane offset by part of an orbit, so their swath edges fall in different places. The union usually fills one platform's edge with the other. It fails when both edges fall over California (03-24) or when the other platform is cloudier there (05-16, 05-27).
* **S-NPP has the same artefact on other days,** where the union then wins:
  * 05-18: S-NPP 9.4% against union 79.3%;
  * 05-13: 8.1% against 58.2%;
  * 04-16: 28.5% against 69.8%;
  * 06-13: 6.7% against 41.1%.
  * So this is geometric noise on both sides, not a JPSS defect.

**(d) Near-shore masking: no evidence.**
* On the failing days the lost S-NPP cells lie 1.8% within 0–5 km and 6.1% within 5–20 km, against ocean shares of 1.9% and 6.2%.
* Near shore the union is better: within 0–5 km it matches or beats S-NPP on 86 of 98 days (88%), and on 15 of the 23 failing days.

**Seasonal pattern** (from `coverage_by_month.csv`).

| month | days | fail | union ≥ S-NPP | mean S-NPP | mean union |
|---|---|---|---|---|---|
| Feb | 4 | 0 | 100% | 14.6 | 16.2 |
| Mar | 21 | 3 | 86% | 42.7 | 48.1 |
| Apr | 23 | 4 | 83% | 24.1 | 32.3 |
| May | 27 | 10 | **63%** | 16.6 | 22.4 |
| Jun (1–14) | 11 | 3 | 73% | 27.1 | 35.2 |
| Sep (20–30) | 8 | 2 | 75% | 23.1 | 31.8 |
| Oct (1–4) | 4 | 1 | 75% | 29.6 | 31.1 |

* May is both the cloudiest month in the sample and the worst for the test. All 5 of N21's near-empty failing days fall in May–June.
* Taking May out would raise the rate to 58 of 71 (82%). That is reported only to locate the failure. **Dropping months is not a valid way to pass.**

**Sample bias from the S-NPP gap.**
* 86 of the 98 days are Feb–Jun; only 12 are Sep 20–Oct 4.
* There is **no July–August comparison** because S-NPP has no data 06-15..09-19. That period is peak marine-layer season off California, which would test low-coverage days most.
* The summer pass rate cannot be inferred. With 12 days (9 passing) the autumn sample is too small to say anything.

**NRT latency and missing granules.**
* **All three products are empty on 8 of the 98 days:** 03-19, 04-06, 04-12, 04-19, 04-25, 05-02, 05-04, 06-11.
  * On every one, the N20/N21 stamps are early (15:52–18:03Z, against the usual 18:5x–19:4x). That is consistent with the daily merges lacking the western-US passes on those days.
  * The fact that S-NPP is empty on the same days suggests a common upstream NRT gap rather than cloud. I have not confirmed this with NOAA.
  * These 8 days count as "passes" (0 ≥ 0). Without them, the original test gives **67 of 90 = 74.4%**.
* Days missing from the time axes are listed in report.md §1 and were not part of the 98.
* Early stamps outside the all-empty days are only a weak indicator: 7 sensor-days, of which 2 had less than half the other JPSS sensor's coverage.

**Flags and QA cannot be assessed from the archive.**
* The archived subsets contain only `chla`/`chlor_a`.
* **S-NPP** (`erdVHNchla1day`) is pre-masked. Its summary says "NOAA-MSL12 NRT processing with LAND+CLDICE+HIGLINT+HISATZEN+HISOLZEN masking".
* **What the servers offer today** (checked in their .dds):
  * N21 (`noaacwN21VIIRSchlaSectorUSDaily`) serves `l2_flags`, `qa_score`, `swath_latitude/longitude` and `graphics`.
  * N20 (`noaacwN20VIIRSchlaSectorUSDaily`) serves `qa_score`, swath coordinates and `graphics`, but **no `l2_flags`**.
* **What this rules out for now.** Whether N20/N21 apply stricter cloud-adjacency, glint or view-angle masks than S-NPP's ERD composite cannot be tested here. Neither can the HISATZEN threshold that produces the straight swath edges.
* A follow-up could fetch `qa_score`/`l2_flags` for the failing days while they are still on the rolling servers.

## 3. Size of the shortfall
**Deficit on the 23 failing days** (S-NPP − union, in percentage points of the stratum's ocean cells; a negative value means the union is higher). From `coverage_shortfall.csv`:

| stratum | median | p25 | p75 | p90 | max |
|---|---|---|---|---|---|
| all ocean | **1.11** | 0.63 | 4.51 | **8.31** | 25.01 |
| north | 0.68 | −0.40 | 6.67 | 10.38 | 12.16 |
| central | 0.91 | 0.24 | 2.47 | 7.34 | 51.55 |
| south | 2.04 | 0.57 | 3.41 | 6.75 | 19.68 |
| near-shore ≤ 5 km | **−5.28** | −9.36 | 2.30 | 5.72 | 9.96 |
| 5–20 km | 0.42 | −3.34 | 3.70 | 8.92 | 15.68 |
| offshore > 5 km | **1.24** | 0.76 | 4.46 | **8.64** | 25.44 |

* **Ratio union/S-NPP on the failing days:** median 0.91, p25 0.75, p75 0.96, max 0.999.
* **Over all days with S-NPP > 0** (88 days): median 1.08, p10 0.86, p25 1.00.

**Coverage by stratum** (mean %, and days on which the union ≥ S-NPP; `coverage_strata.csv`):

| stratum | cells | all 98 days: S-NPP / union | union ≥ S-NPP (98 days) | 23 failing days: S-NPP / union | union ≥ S-NPP (23 days) |
|---|---|---|---|---|---|
| all | 61,784 | 26.1 / 32.5 | 75 (76.5%) | 27.1 / 23.7 | 0 |
| north | 7,765 | 28.5 / 35.0 | 64 (65.3%) | 36.8 / 34.4 | 7 |
| central | 28,774 | 27.4 / 34.9 | 74 (75.5%) | 28.9 / 24.9 | 4 |
| south | 25,245 | 24.0 / 29.1 | 71 (72.4%) | 22.1 / 19.0 | 3 |
| 0–5 km | 1,175 | 33.8 / 48.3 | 86 (87.8%) | 40.0 / 44.6 | 15 |
| 5–20 km | 3,801 | 39.7 / 50.0 | 73 (74.5%) | 47.1 / 46.7 | 11 |
| 20–100 km | 14,976 | 31.1 / 40.5 | 73 (74.5%) | 35.0 / 33.7 | 7 |
| > 100 km | 41,832 | 22.9 / 27.7 | 63 (64.3%) | 22.1 / 17.4 | 2 |
| offshore > 5 km | 60,609 | 26.0 / 32.2 | 73 (74.5%) | 26.9 / 23.3 | 0 |

* The per-stratum pass rates are lower than the overall rate for small strata (north 65%), even though the union's mean is higher in every stratum. Smaller areas make the per-day ≥ test noisier (see §5).

**Cell level, failing days.**
* S-NPP had 385,279 valid cell-days, and **101,204 (26%) of them are missing in the union**. The union had 52,231 cell-days S-NPP lacked.
* **Where the lost cells are:**
  * by distance: 1.8% within 0–5 km, 6.1% within 5–20 km, 21.4% within 20–100 km, 70.7% beyond 100 km (ocean shares 1.9 / 6.2 / 24.2 / 67.7%);
  * by region: north 12.2%, central 50.8%, south 37.0% (ocean shares 12.6 / 46.6 / 40.9%).
* **Over all 98 days,** 82.2% of S-NPP's 1,581,430 valid cell-days are also valid in the union.

## 4. The 4 km NOAA-20 product as a separately labelled fallback
From `coverage_4km.csv`; 86 dates present; 21 of the 23 failing days present (03-31 and 04-04 are not on the 4 km axis).

**It is the same NOAA-20 data at coarser resolution.**
* On 74 of the 78 dates where N20 750 m has data, every N20-valid 750 m cell lies inside a valid 4 km cell.
* Its native coverage is about 1.6× the N20 750 m coverage (median ratio 1.57), with a day-to-day correlation of 0.94.
* On N20's empty or near-empty days it is just as empty: 04-21 0%; 05-09 0%; 05-14 3.0%; 05-30 5.3%; 06-04 3.0%.
* **What it can add.** Its extra coverage comes from 4 km bins that hold some clear 750 m pixels. Mean native coverage on the 86 days:

  | layer | mean coverage |
  |---|---|
  | 4 km | 33.1% |
  | N20 750 m | 23.2% |
  | S-NPP | 24.9% |
  | union | 31.5% |

  * Near shore (≤ 5 km) the 4 km footprint averages 47.0%, against 34.3% for S-NPP.
* **What it cannot do.** It cannot fill a NOAA-20 outage, and it does not help when both JPSS swaths miss an area. On 03-24 it had only 19.6%, less than N20 750 m. On 4 dates (03-24, 05-09, 06-05, 06-09) it lacks part of N20's 750 m data, presumably because of different binning or day boundaries; not resolved here.

**Effect on the result, if it were evaluated as a footprint:**
* On failing days, a median of 50% (mean 46%) of the S-NPP-only 750 m cells fall inside a valid 4 km cell.
* Union ∪ 4 km footprint ≥ S-NPP on 83 of 86 days, and 18 of the 21 failing days present would "pass".
  * **This is not like-for-like.** A 4 km cell counts as covered when any part of it was clear, so footprint coverage is inflated by resolution.
  * It must never be merged into 750 m pixels or counted as 750 m coverage.
  * As a separately labelled, coarser layer it would put some observation over about half of the area the 750 m layer loses on failing days.
* **Conclusion: it does not change the result for the 750 m criterion.** It is a possible secondary display, with its own label and resolution, for days the 750 m union is thin.

## 5. Is the criterion well posed?
* **Per-day coverage is dominated by clouds and swath placement that differ between passes.** Baselines on the same 98 days (`coverage_criterion_variants.csv`):

| test | days passing |
|---|---|
| N21 ≥ N20 (two near-identical sensors) | 47 / 98 (48%); N20 ≥ N21 59 / 98 (both include 8 empty-day ties) |
| N21 ≥ S-NPP / N20 ≥ S-NPP | 45 (46%) / 36 (37%) |
| S-NPP(day+1) ≥ S-NPP(day) | 50 / 98 (51%) |
| S-NPP(day−1) ∪ S-NPP(day+1) ≥ S-NPP(day) | 74 / 87 (85%) |
| S-NPP ∪ N21 ≥ N20 | 86 / 98 (88%) |
| S-NPP ∪ N20 ≥ N21 | 92 / 98 (94%) |
| **N21 ∪ N20 ≥ S-NPP (the criterion)** | **75 / 98 (76.5%)** |

* **A two-look union beats a single look on roughly 76–94% of days,** depending on which platforms are paired.
  * The JPSS pair scores lowest. Their clear areas overlap slightly more (Jaccard 0.52, against 0.48–0.51 for pairs with S-NPP), and S-NPP's single-sensor coverage is slightly higher (26.1% vs 24.3/25.3%).
  * So 76.5% is not an outlier against the noise baseline. Even so, it is genuinely the weakest of the three pairings.
* **A strict per-day "≥" at 80% gives no credit for size.**
  * 5 of the 23 failures are under 0.5 points. 8 "passes" are 0 = 0 days with no data at all. The worst failure (25 points) counts the same as a 0.03-point one.
  * At the same time it hides real losses: per-day ≥ says nothing about a 25-point miss.
* **Tolerance variants** (all 98 days):

  | variant | days passing |
  |---|---|
  | union ≥ 0.95×S-NPP | 82 (83.7%) |
  | union ≥ 0.90×S-NPP | 87 (88.8%) |
  | union ≥ S-NPP − 1 pt | 83 (84.7%) |
  | union ≥ S-NPP − 2 pt | 90 (91.8%) |
  | union ≥ S-NPP − 5 pt | 94 (95.9%) |
  | original test, all-empty days excluded | 67 / 90 (74.4%) |
  | original test, S-NPP ≥ 10% days only | 51 / 67 (76.1%) |
  | ≥ 0.9×, S-NPP ≥ 10% days only | 62 / 67 (92.5%) |
  | weekly mean union ≥ weekly mean S-NPP (weeks with ≥ 3 days) | 17 / 17 weeks |

* **A better-posed criterion for future migrations,** to be fixed before looking at data:
  1. mean union coverage ≥ mean S-NPP coverage over the window, overall and in each region and in the ≤ 5 km band;
  2. union ≥ 0.9 × S-NPP on ≥ 80% of informative days (S-NPP ≥ 10%, not all-empty);
  3. report the share of S-NPP-valid cells retained, here 82%, and the worst-day deficit.
  4. **Noise baseline.** Calibrate the per-day thresholds against an inter-sensor baseline such as the permuted-union rates above.
* **This criterion is not applied retroactively.** The proposed criterion was the original one, and it **failed**.

## 6. User-visible consequence and how to communicate it
**What users will see.**
* Compared with what S-NPP would have given, the NOAA-21-first, NOAA-20-fill layer will show **less ocean on about 1 day in 4** (23 of 98; 23 of 90 days with any data).
  * The shortfall is **typically about 1 point** of California ocean area (median 1.1 points, about 9% of what S-NPP saw that day).
  * About **1 day in 12** it is 2 points or more (8 of 98), and 4 of 98 days exceed 5 points; the worst was 25 points on 03-24.
  * It falls mostly **offshore (> 100 km) in the central and southern regions**, and most often in **May** (10 of 27 May days).
* **On the other days the layer shows more ocean,** on average 6.4 points more (32.5% vs 26.1%), including near shore.

**How the app should communicate it.** The app already has the pieces: gaps shown as "no observation, not low chlorophyll", per-pixel dates, per-region coverage fractions and platform labels.
* **Keep gaps visibly blank.** Never interpolate across a swath-edge stripe.
  * Straight-edged blank bands are a normal artefact of the satellite's swath. The coverage note can say "blank areas include gaps between satellite passes as well as cloud".
* **Show the day's per-region coverage next to the map** (already planned). Optionally flag a day as "low coverage" when a region's coverage falls well below its recent median.
* **Name the platform per pixel** (NOAA-21 or NOAA-20 fill) and state in provenance that the source changed from S-NPP on the migration date, with the measured offsets from report.md.
* **Do not describe the new layer as having "the same or better coverage every day".** "On average more coverage; on some days less, usually by a small amount" is accurate.
* **If the 4 km NOAA-20 product is ever shown,** it must be a separate, labelled, coarser layer ("NOAA-20, 4 km"), never blended into the 750 m pixels.
* **Science framing is unchanged:**
  * these are observations, not forecasts;
  * satellite chlorophyll is not a toxin measurement.

## 7. Caveats
* **Stride-4 sampling** (about 3 km, 1 in 16 native cells). Coverage fractions are estimates. Single-cell edge effects, such as cloud-edge speckle and thin stripes, are undersampled.
* **These are NRT products.** Masks, algorithms and the rolling archives can change. What is analysed is what was served on 2026-10-13. Empty days may be NRT gaps that later reprocessing would fill.
* **No flags.** l2_flags and qa_score were not archived, and N20 does not serve l2_flags. The masking explanation for the swath edges (view-angle limits) is inferred from the straight geometry, not verified from flags.
* **The overpass-time explanation cannot be tested.** The stamps are nominal (12Z for S-NPP, daily-merge centre for N20/N21), so no per-pixel overpass times exist in the archive.
* **Seasonal bias:**
  * 86 of the 98 days are Feb–mid-June and 12 are late Sep–early Oct;
  * there is no July–August comparison because of S-NPP's 3-month hole (06-15..09-19);
  * May, the worst month, carries 27 days.
* **The classification has a manual component** (visual straight-edge flags) and thresholds I chose. The counts per category could shift by a day or two under different thresholds. The overall picture would not change: the failure is not near-shore, and it comes from a mix of geometry, N21 partial days and cloud.
* **The 4 km comparison** covers 86 of 98 dates; 12 are not on the pfeg axis. Its footprint measure is resolution-inflated by construction.
* **No in-situ data.** Coverage is not accuracy, and S-NPP is the continuity reference, not truth.
