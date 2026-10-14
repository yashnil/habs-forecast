# Sablefish survey-density model, California: offline research prototype

**Research only.** Not connected to CoastWatch, not published, not a fishing recommendation. Prepared 2026-10-13 as the "prototype only if labels exist" step of the fishing-opportunity feasibility study ([docs/coastwatch/research/fishing-opportunity-feasibility.md](../../docs/coastwatch/research/fishing-opportunity-feasibility.md)).

**Verdict: NO-GO for any user-facing use.** The model ranks where the NOAA trawl survey finds more or less sablefish modestly better than a depth × latitude climatology. It does **not** know how much sablefish there is, its intervals under-cover, and it fails when the Southern California Bight is held out.

## Data

| | |
|---|---|
| Source | NOAA NWFSC FRAM Data Warehouse, West Coast Groundfish Bottom Trawl Survey ("Groundfish Slope and Shelf Combination Survey"), the fishery-*independent* survey: catch per tow with position, date, depth and area swept |
| API | `https://www.webapps.nwfsc.noaa.gov/trips/api/v1/source/trawl.operation_haul_fact/selection.json` and `.../trawl.catch_fact/selection.json` (the older `/data/api/v1/` path returns 404; it moved in April 2025) |
| Retrieved | 2026-10-13 05:33Z, two requests: 15,133 hauls, 9,695 sablefish catch records |
| Used | California (lat < 42° N), `performance == "Satisfactory"`, `station_invalid == 0`: **7,714 tows**, 4,700 (61%) with sablefish. Zero-catch tows come from joining catch onto hauls by `trawl_id`. |
| Gaps | 2019 half coverage (2 vessels); **2020 no survey**; 2013 reduced |
| Splits | train 2003–16 (4,903 tows) · tune 2017–19 (970) · **test 2021–25 once (1,841)** |
| Not committed | raw responses and the derived parquet (re-create with `download.py` then `build.py`) |

## Method

- **Response:** density in kg/ha including zeros; hurdle models (occurrence plus log-normal positive density) and a Poisson gradient-boosting model.
- **Covariates:** log depth, latitude, longitude, bottom temperature measured on the tow, day of year. Year is excluded, so nothing is learned from a year the model has not seen.
- **Baselines:** depth × 1° latitude climatology; persistence by spatial bin, both frozen and rolling.
- **Models:** HistGradientBoosting hurdle (with and without temperature), pyGAM hurdle, HistGradientBoosting Poisson.
- **Tuning:** on 2017–19 only.
- **Holdouts:** leave one region out (north ≥ 40° N, central, south < 34.5° N) and leave one of the 5 main vessels out, both within 2003–19.
- **Uncertainty:** cluster bootstrap over year × latitude band (1,000 reps). Intervals by quantile gradient boosting and conformalized quantile regression.

## Results (test 2021–25, scored once)

| | Spearman ρ | AUC | Brier | Poisson deviance | Mean predicted (observed 16.0 kg/ha) |
|---|---|---|---|---|---|
| Climatology | 0.731 | 0.861 | 0.151 | **69.4** | 5.9 |
| GBM hurdle | **0.789** | 0.892 | 0.139 | 74.7 | 5.6 |
| GAM hurdle | 0.776 | 0.887 | **0.138** | 76.1 | 5.7 |

- **Skill against climatology (GBM hurdle, 95% CI):**
  - ΔSpearman +0.058 [+0.034, +0.083];
  - ΔAUC +0.031 [+0.015, +0.048];
  - Brier skill +0.08 [+0.02, +0.14];
  - **Poisson-deviance skill −0.08 [−0.11, +0.05], so no skill on level.**
- **Level shift:** 2021–25 averaged about 2.6× the training years; every frozen method under-predicted by about 2.7×.
- **Calibration:** occurrence calibration slope 0.87–0.94, but the intercept is +0.6 to +0.7, so presence is under-predicted in the test years.
- **Intervals:** 80% and 95% intervals covered 72% and 86% of test tows. Conformal correction did not help, because the shift came after the calibration years.
- **Holdouts:**
  - Leave one vessel out: the model wins in all 5 cases.
  - Leave the south out: the GBM's Brier skill is −0.05 [−0.07, −0.03] and the GAM collapses (AUC 0.50).
- **Bottom temperature** adds nothing beyond depth, position and season.

Full metrics: [results/results.json](results/results.json). Figures: [figs/](figs/).

## What it can and cannot claim

**Can (research framing only):**
- the *relative* pattern of NOAA-survey sablefish density in trawlable soft-bottom habitat, about 55–1,280 m, June–October, annual resolution;
- that this pattern held in five unseen years, modestly better than a depth × latitude table.

**Cannot:**
- catch, catch rates or fishing opportunity for any gear;
- abundance or year-to-year change;
- legal fishing areas: RCAs, EFH areas, MPAs, the Cowcod Conservation Areas, seasons and trip limits;
- rocky, untrawlable or nearshore habitat;
- daily or weekly timing;
- anything about toxins or seafood.

## Conditions before any exhibit (research mode only)

- level-free display (ranks or quantiles);
- explicit labels ("NOAA survey density, trawlable habitat, not catch, not legal guidance");
- no forecast framing;
- review by a NWFSC or PFMC groundfish scientist.

**Next research steps:** an sdmTMB delta-gamma model with spatial and spatiotemporal fields, checked against the design-based index; size-stratified models; a replication species (Dover sole).

## Reproduce

```
uv run --python 3.12 --with pandas --with numpy --with scipy==1.14.1 --with scikit-learn --with pyarrow --with matplotlib --with pygam python -I download.py
uv run … python -I build.py
uv run … python -I model.py   # about 11 minutes; bootstrap seed 20261013
```
