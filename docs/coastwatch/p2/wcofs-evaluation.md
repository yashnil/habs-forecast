# WCOFS forecast currents: evaluation against HF radar (P2 phase D)

Status:
- **experimental research, not user-facing;**
- the P1 proof of concept ([`wcofs_poc.py`](../../../pipeline/scripts/wcofs_poc.py)) is kept as it was;
- this phase adds a skill check ([`wcofs_eval.py`](../../../pipeline/scripts/wcofs_eval.py)), with the full results in [`evidence/wcofs-vs-hfradar-skill.json`](evidence/wcofs-vs-hfradar-skill.json).

## 1. Question

In P1, four hourly comparisons in Monterey Bay showed realistic speeds but weak pattern agreement (correlation 0.25–0.45). The question is whether that was bad luck, a convention bug, the choice of comparison, or the model.

## 2. Method (2026-10-09)

- **Forecasts:** WCOFS runs 2026-10-06 and 2026-10-07 t03z. Every 3-hourly regular-grid step HF radar has observed: 44 steps, leads 3–72 h. Surface level only, read by byte range (about 34 MB and 5 s per step).
- **Observations:** HF-radar 2 km totals (`ucsdHfrW2`) at the same hour. WCOFS (0.04°) is sampled at the nearest cell to every valid radar cell.
- **Regions:**
  - Monterey Bay (36.45–37.15 N);
  - Gulf of the Farallones (37.3–38.1 N);
  - the central-coast shelf (34.4–35.6 N);
  - the Southern California Bight (33.3–34.45 N).
- **Metrics:**
  - vector RMSE;
  - complex correlation (magnitude, and angle = mean rotation);
  - mean speeds;
  - skill = 1 − RMSE_model / RMSE_baseline.
- **Baselines:**
  - *persistence*: the radar field at the issue hour carried forward. This is optimistic, because real-time radar arrives hours late. For daily means, it is the radar mean over the 24 h before issue.
  - *zero*: predict no current.
- **Daily means:** lead-day-1 24-hour means, 8 WCOFS steps against 24 radar hours (at least 18 valid). These suppress the tide and the daily sea breeze, which a 3-hourly comparison aliases.
- **Independence:** WCOFS assimilates HF radar, so early leads are not independent of the observations. That favours the model.

## 3. Results

Median over the steps in each lead band:

| Region | Correlation 3–24 h | Correlation 27–48 h | Skill vs persistence 3–24 h | Skill vs persistence 27–48 h | Skill vs zero 3–24 h |
|---|---|---|---|---|---|
| Monterey Bay | 0.19–0.25 | 0.14–0.37 | **−0.16 to −0.42** | **−0.42 to −0.64** | 0.01–0.09 |
| Gulf of the Farallones | 0.32–0.43 | 0.31–0.40 | 0.01–0.11 | 0.06–0.14 | −0.06 to −0.12 |
| Central-coast shelf | 0.53–0.64 | 0.44–0.51 | −0.34 to −0.50 | −0.15 to −0.24 | **+0.29 to +0.34** |
| Southern California Bight | 0.38–0.43 | 0.24–0.33 | −0.03 to −0.11 | −0.27 to −0.28 | −0.07 to +0.01 |

Lead-day-1 daily means:

| Region | Correlation (two runs) | Mean rotation | Skill vs zero | Skill vs persistence |
|---|---|---|---|---|
| Monterey Bay | 0.12, 0.42 | **134°, 167°** | −0.12, −0.29 | −1.1, −1.5 |
| Gulf of the Farallones | 0.39, 0.13 | 53°, 21° | −0.04, −0.15 | −1.1, −1.1 |
| Central-coast shelf | 0.61, 0.52 | 6°, 8° | +0.30, +0.25 | −0.58, −0.29 |
| Southern California Bight | 0.42, 0.26 | −10°, 4° | −0.43, −0.76 | −0.57, −1.3 |

Speeds are realistic everywhere: about 0.14–0.26 m/s modelled against 0.17–0.30 m/s observed.

## 4. Why Monterey Bay agrees poorly

- **Not a convention bug.** On the open shelf the same code gives correlation 0.5–0.64 with a mean rotation of 6–8°. A sign or rotation error would show up there too.
- **The bay's circulation is not resolved.** In Monterey Bay the daily-mean patterns are rotated 134–167°, nearly opposite. The bay's recirculation and the upwelling shadow are a few tens of km across. WCOFS has about 4 km cells, so a few cells across the bay mouth, and assimilates at that scale.
- **The persistence baseline is hard to beat.** Over a day or two, coastal surface currents are dominated by slowly changing wind-driven and eddy flows. The radar field from yesterday is a strong forecast.
- **The sample is small.** Two runs in one season. This is enough to say "not ready", not enough to estimate skill reliably.

## 5. Recommendation

1. **Do not show WCOFS as a user-facing forecast.** It does not beat persistence in any region tested except, marginally, the Gulf of the Farallones. In bays it can point the wrong way.
2. **If it is ever shown,** only on the open shelf, at regional zoom, as "model context". It would need its skill printed in the layer and a check that passes the gate below.
3. **Validation gate before any user-facing forecast currents:**
   - **Period:** at least 60 days covering upwelling and relaxation seasons, by region and lead day.
   - **Metrics:** daily-mean currents; positive skill against *latency-aware* persistence at 24–48 h; mean rotation within ±30°; RMSE below the zero-current RMSE.
   - **Coverage:** report results separately for bays, the shelf and offshore.
4. **Continue to defer drift or particle outlooks and any HAB transport forecast.** PR #10's gates still apply.

## 6. Cost of a proper hindcast

Per run, 24 steps × about 34 MB is about 0.8 GB of range reads, about 2 minutes. 60 days is about 50 GB and 2 hours, on a laptop or a free runner. It needs no paid infrastructure.
