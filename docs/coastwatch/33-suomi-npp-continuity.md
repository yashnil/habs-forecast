# 33 — Satellite continuity: the end of Suomi-NPP data (2026-11-02)

**Status:** audit and validation done; migration planned, not yet implemented. The C-HARM input watch and the monitoring fixes are in this PR. **No production dataset changed.**

**Evidence:**
- [dependency audit](../../research/viirs-jpss-overlap/snpp-dependency-audit.md): live checks on 2026-10-13, 07:02–07:11 UTC;
- [overlap validation](../../research/viirs-jpss-overlap/report.md): 98 three-sensor days; scripts, download log and metrics in [research/viirs-jpss-overlap](../../research/viirs-jpss-overlap). The 258 MB raw archive is kept outside the repo.

## 1. What depends on Suomi-NPP

| Dependency | Where | What it feeds | Who can fix it |
|---|---|---|---|
| `erdVHNchla1day` (S-NPP VIIRS 750 m daily) | `sources/satellite.py:154-168`, `sources/multisensor.py:229-251` | VIIRS map layer, the VIIRS half of the multi-sensor layer, the satellite readout fallback, verify-satellite, fixtures | **CoastWatch** |
| `erdVHNchla8day` (S-NPP 8-day) | `sources/port_intel.py:48` | Port and My Coast chlorophyll panel | **CoastWatch** |
| C-HARM v3.1 ocean-colour input | NOAA: `satellite="S-NPP"`; product "C-HARM v3.1 model output with SNPP VIIRS and WCOFS data"; history "DINEOF gap filling … daily S-NPP NOAA VIIRS chlorophyll", 2026-10-12 | Every C-HARM forecast | **NOAA only** |
| GIBS chlorophyll imagery | `sources/gibs.py` | Visual layer | Not S-NPP (already NOAA-20) |

- **Cut-off:** 2026-11-02, 13:00 UTC (NESDIS; CoastWatch "Update #1", 2026-10-01).
- **NOAA's guidance:** move to NOAA-21 first, NOAA-20 second.
- **C-HARM:** no NOAA, CoastWatch, SCCOOS or CeNCOOS statement about it was found. Whether NOAA moves it to NOAA-20/21, recalibrates it, keeps running it on ageing input, or pauses it is **unknown**. Replacing CoastWatch's own VIIRS source does not change C-HARM's input.
- **Already gapped:** S-NPP `erdVHNchla1day` has no data at all from 2026-06-15 to 2026-09-19.

## 2. Replacements (live 2026-10-13)

| Product | Server | Grid | Variable | Archive | Latest seen | California request |
|---|---|---|---|---|---|---|
| **`noaacwN21VIIRSchlaSectorUSDaily`** (NOAA-21) | coastwatch.noaa.gov | 0.0075°, **the same lattice as `erdVHNchla1day`** | `chlor_a`, mg m⁻³; `l2_flags`, `qa_score` | rolling, from 2026-02-09 | 2026-10-05 | HTTP 200 |
| **`noaacwN20VIIRSchlaSectorUSDaily`** (NOAA-20) | coastwatch.noaa.gov | same | same | rolling, from 2026-02-01 | 2026-10-05 | HTTP 200 |
| `nesdisVHNnoaa20chlaDaily` (NOAA-20, 4 km) | pfeg | 0.0375° | `chlor_a` | from 2021 | 2026-10-10 | HTTP 200, fallback on another server |

- **No NOAA-20/21 750 m 8-day product exists.** The port panel should use an 8-day median of the daily pixels that CoastWatch computes itself.
- **Not recommended:** the gap-filled DINEOF products are 9 km and themselves use S-NPP.

## 3. Overlap validation (98 days, February–October 2026, California, the same cells)

| Pair | Median log₁₀ bias (ratio) | RMSE (log₁₀) | r (log₁₀) | RMA slope |
|---|---|---|---|---|
| NOAA-20 / S-NPP | −0.045 (0.90) | 0.148 | 0.961 | 1.20 |
| **NOAA-21 / S-NPP** | **−0.020 (0.96)** | **0.095** | **0.973** | **1.02** |
| NOAA-21 / NOAA-20 | +0.040 | 0.129 | 0.963 | 0.90 |

- **Coverage:** the mean share of ocean cells with a value is 26.1% for S-NPP, 24.3% for NOAA-20, 25.3% for NOAA-21, and **32.5% for NOAA-21 and NOAA-20 together**.
- **Acceptance:**
  - bias documented and stable: pass for NOAA-21, marginal for NOAA-20;
  - r ≥ 0.9: pass for both;
  - implausible values ≤ 1%: pass;
  - the combined coverage at least matching S-NPP on 80% of days: **fail (76.5%)**. There will be somewhat more days with less coverage than S-NPP gave.
- **Caveats:**
  - near-real-time products, with no flag filtering applied;
  - every 4th cell sampled;
  - different overpass times;
  - poorer agreement within 5 km of shore and above 1 mg m⁻³;
  - no in-situ check.
- **No bias correction.** CoastWatch publishes the agency's values and states the offset.

**Recommendation:** NOAA-21 first, with NOAA-20 filling cells NOAA-21 lacks. Each pixel comes from one platform and is never averaged. The 4 km NOAA-20 product is a labelled fallback on another server.

## 4. Migration plan (staging first)

1. **Implementation** (branch `feat/viirs-jpss`; file and line references in the audit):
   - a NOAA-21/NOAA-20 product on the same lattice, using overpass-time stamps;
   - the port 8-day median computed from the daily pixels;
   - verify-satellite selecting by each layer's own `dataset_id`;
   - web labels from `platforms` ("VIIRS · NOAA-21");
   - a new processing version, so S-NPP days are never reused as NOAA-20/21;
   - **no new manifest fields**, because the demo and released web reject them.
2. **Provenance:**
   - published S-NPP artifacts keep `erdVHNchla1day` / `erdVHNchla8day` and their dates;
   - new artifacts carry the new dataset ids and platforms;
   - port history marks the change point and the measured offset.
3. **Staging:**
   - re-record fixtures;
   - run the pipeline to the **staging** channel only;
   - run verify-satellite live;
   - compare the staging and production manifests;
   - view staging in the private preview;
   - simulate the cut-off: S-NPP returns nothing newer, and NOAA-21 or NOAA-20 fails alone.
4. **Merge by about 2026-10-26**, so a week of NOAA-20/21 runs exists before 2026-11-02. Keep the S-NPP fallback until it goes stale, then remove it.
5. **Health thresholds:** set the VIIRS latency threshold from the observed 3–8 days (NOAA-20/21 lagged about 7.5 days on 10-13).

## 5. C-HARM (in this PR)

- **Watched:** NOAA's own C-HARM metadata (`satellite`, `platform`, `product_name`, `product_version`, `date_modified`) is now kept with each run in the existing free-form `upstream_metadata`.
- **`charm_input_changed`:** raised once when any of these differs from the previous run; it stays open a week. The app should then state that forecasts before and after may not compare directly.
- **`charm_input_ageing`:** raised when runs after 2026-11-02 still declare S-NPP.
- **Unchanged:** if runs stop, the existing staleness checks and the app's freshness labels apply. A missing or stale run is never "no risk".

## 6. Decisions for the owner

- **Primary platform:** NOAA-21 first, as NOAA recommends and as the validation supports, or NOAA-20 first.
- **NOAA CoastWatch help desk:** whether to ask about the OC3/OCI algorithm labels and C-HARM's plans. This is outreach and needs your approval.
- **Timing:** approval to implement `feat/viirs-jpss` now, so it can merge before 2026-10-26.
