# Station baselines (research only, not in the app)

`baselines.py` asks what a shore station's own past says about its next weekly sample, as the bar
any future station-level forecast (or assistant answer) must clear before it is shown. Stdlib
Python, deterministic; `RESULTS.md` is its output on the observations snapshot used by the web
tests (generated 2026-10-08; CalHABMAP via Neritic's pipeline). Re-run on the current production
`observations-*.json` before quoting any number.

## Findings (temporal holdout: fit < 2022, score 2022 → 2026-10)

- **Pseudo-nitzschia ≥ 10,000 cells/L** (1,793 test pairs, 468 events): persistence, as a
  calibrated two-state Markov chain, has Brier skill **0.15** over the station's base rate and AUC
  0.77; seasonal climatology 0.11 / 0.76. Every station with enough events has positive persistence
  skill (0.10–0.25).
- **Particulate domoic acid ≥ 0.5 ng/mL** (1,384 pairs, **41 events**): pooled persistence skill
  0.07, seasonal −0.01; per-station values swing from −0.06 to 0.34 on 5–11 events each. Too few
  events to rank methods; any claimed pDA skill at a pier needs far more than this to stand.

## What this means for v3 and later

1. A model's *next-sample* claim at a pier is only informative if it beats persistence (≈ 0.15 BSS
   for cells). This is the station-scale analogue of the PINN audit's finding that climatology +
   AR(1) already matches the published networks (research/pinn_audit on PR #49): baselines first.
2. The record context shipped in v3 (`src/lib/history.ts`) describes the past and makes no
   prediction. These numbers are why: the past is informative but weakly so, and saying "likely
   to stay high" would claim skill not shown here.
3. Not done, needs owner approval or more data: C-HARM's own skill at the piers on the same pairs
   (needs archived daily C-HARM cells at each pier; the pipeline keeps only current runs), event
   thresholds tied to agency action levels, lead times beyond one sample.

## Caveats

Weekly sampling is irregular (holidays, staffing); 3–14-day steps only. Pseudo-nitzschia counts are
of two size classes, not species, and not toxicity. The split is one temporal holdout, not a
cross-validated estimate; confidence intervals are not computed.
