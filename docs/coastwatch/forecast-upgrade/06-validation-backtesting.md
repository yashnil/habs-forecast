# 6. Validation and backtesting requirements

Nothing new is presented as predictive until it passes the matching gate below, and an independent HAB
scientist has reviewed the result. That review is still pending for the existing product too.

## 6.1 Ground truth available

| Truth | Coverage | Use |
|---|---|---|
| CalHABMAP pier samples (pDA, cDA, *Pseudo-nitzschia*, chlorophyll) | 10–17 piers, weekly, 2014– | Primary target for C-HARM and ML evaluation |
| HF-radar surface currents | Hourly, 2012–, partial coverage | Current and transport verification |
| Global Drifter Program trajectories | Sparse | Lagrangian verification |
| Satellite chlorophyll at the next clear overpass | Cloud-limited | Feature-displacement checks for transport |
| CDPH shellfish toxin results and closures | Event records | Context only: not a target for water forecasts |

## 6.2 Archives needed, and where they are

| Archive | Where | Note |
|---|---|---|
| C-HARM v3.1 nowcasts and forecasts | `wvcharmV3_*` on ERDDAP, 2022-11 onwards | Fixed-lead history available, so C-HARM can be scored directly |
| VIIRS chlorophyll (S-NPP 2012–, NOAA-20 2017–) | ERDDAP science-quality datasets | Long enough for 2014– targets |
| OLCI 300 m | **Not on NOAA's ERDDAP beyond 90 days.** Use EUMETSAT/Copernicus (account) or NASA OB.DAAC (Earthdata). | Needed before OLCI-based features can be trained |
| PACE | 2024– via Earthdata | Too short to train on yet |
| WCOFS | AWS bucket listing starts in July 2024 (`202407/`… folders, then `YYYY/MM/DD/`). Earlier runs would have to come from NOAA archives. | Operational since 2021, so at most about 5 seasons |
| Upwelling indices CUTI/BEUTI | NOAA ERD, 1988– | Long, cheap predictors |

## 6.3 Tests and acceptance gates

| Product | Test | Baselines | Metrics | Gate to show it publicly (experimental label) |
|---|---|---|---|---|
| **C-HARM display** (existing) | Match each pier sample to the C-HARM cell nowcast and +1/+2/+3 forecasts issued before it, Nov 2022 to now | Climatology by station and month; persistence of the last sample | Brier skill score, AUC, reliability, hit and false-alarm rates at 0.5 ng/mL | Publish the scores as information whatever the result, as a description of the agency model. CoastWatch doesn't gate an agency product. |
| **Satellite chlorophyll layers** | Compare against pier chlorophyll (extracted, with replicates) on matching days, nearshore vs offshore pixels | — | Bias and scatter on a log scale; share of piers with a valid pixel within 1 km | Show as observations with quality flags; publish the matchup statistics in methods |
| **Latest clear-view composite** | Check that the age of each pixel is correct, and that composite values equal source values | — | Exact equality | Unit tests |
| **Surface drift (B)** | Hindcast: run WCOFS forecasts from past runs, compare with HF-radar-derived trajectories and drifters at 24, 48 and 72 h | HF-radar persistence (today's currents held fixed); no motion | Separation distance (km) by lead; skill score vs persistence | Positive skill vs persistence at 24–48 h in the regions shown. Ensemble spread shown, and capped at 72 h. |
| **Station ML (C)** | Rolling-origin backtest 2016–2026, leave-one-event-out | Persistence, climatology, C-HARM nearest valid cell | Brier skill score (BSS) with confidence intervals, reliability, precision/recall at decision thresholds, lead-time stratified | BSS > 0 vs **both** persistence and C-HARM with a confidence interval excluding 0. Reliable calibration. HAB-scientist sign-off. |

## 6.4 Things a backtest must not do

- Tune on the test years, or let gap-filled satellite fields "see" data from after the forecast time. DINEOF uses a 180-day window, so archived gap-filled fields must be the ones produced at the time.
- Count autocorrelated weekly samples as independent events.
- Report accuracy on the 96 % of samples that are below threshold as skill.
- Treat "no sample" as "no toxin".
