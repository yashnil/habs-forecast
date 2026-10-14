# Audit and reproduction of the physics-guided chlorophyll forecasting study

**Scope.** I re-ran the existing checkpoints against the frozen data cube, read the experiment code, and retrained under one controlled protocol.

- No original file was edited; the scripts import `pinn/*.py` unchanged.
- No checkpoint in `~/HAB_Models` was written.

**Status (2026-10-14).**

- §1–§3 are complete: 12 controlled runs (4 arms × 3 seeds) finished on 2026-10-14.
- Nothing here has been reviewed by the author.
- Numbers in this file come from `results/`. They are research diagnostics, not CoastWatch forecasts.

## Summary

1. **The published numbers reproduce exactly.** ConvLSTM test RMSE is 0.8006, and the PINN (λ=5, κ=100) is 0.8007 [README §4.1: 0.801 / 0.801].
2. **They were produced by two errors that partly cancel.**
   - Training adds the predicted change to the *z-scored* last log-chlorophyll.
   - Diagnostics adds it to the *raw* value, then subtracts one constant bias fitted over all periods, test included.
   - Scored the way it was trained, each model is *better* than published: test 0.768 (ConvLSTM) and 0.761 (PINN), against 0.801.
   - The test leakage from the bias step itself is negligible (0.8006 → 0.8010 with a train-only bias).
3. **A two-parameter statistical baseline matches both networks.**
   - The baseline is per-pixel monthly climatology plus AR(1) on anomalies, fitted on train only.
   - It gets test RMSE 0.762, 14.3% better than persistence.
   - Both networks' differences from it are inside the 95% block-bootstrap interval.
   - Persistence is the only baseline in the paper, and it is a weak one here.
4. **The physics term has no effect as implemented.**
   - The λ and κ ablation tables are the same seven runs relabelled: row *i* of each matches to about 1e-8.
   - The physics residual is about 1e-8, against a data loss of about 0.3. Even at the λ clamp of 1000, physics contributes about 3e-5 of the loss.
   - κ never moves from its initial value; the change is at the level of weight decay.
   - So "optimal λ=5, κ=100" is an artefact of run order, and "PINN matches ConvLSTM" is true for the trivial reason that they are almost the same objective.
   - The controlled retraining (§3) confirms this directly. With the physics term on and off, the same-seed test RMSE is identical to 5 decimals in all 3 seeds.
   - Across 12 runs, the seed-to-seed spread (sd 0.005–0.008) is as large as the gap between the published checkpoints.
5. **Several reported analyses are not what they are described as:**
   - the imputation table (§4.4);
   - the lead-time table (§4.2);
   - the "PINN" row's provenance (§4.1).

   Details are in §2.

The safe public statement remains: **the PINN matches, and does not beat, the ConvLSTM.** It should be strengthened to say that neither beats a climatology + AR(1) baseline at 8 days.

## 1. Reproduction and re-scoring (complete)

Run `rescore.py`, then `bootstrap.py` (commands in §5). Every forecast below is scored on the same 1,373,970 ocean pixel-times. The test set has 110 composites (2019-01 to 2021-06) and 186,990 pixel-times. Each row lists test, val and train RMSE in ln(mg m⁻³), then test skill vs persistence.

**Baselines (fitted on train only):**

| Forecast | Test | Val | Train | Test skill |
|---|---:|---:|---:|---:|
| Persistence | 0.8889 | 0.8543 | 0.8927 | — |
| AR(1) pooled, y = 0.731·x + 0.137 | 0.8226 | 0.7932 | 0.8304 | 7.5% |
| Climatology + AR(1) on anomalies, φ = 0.42 | **0.7618** | 0.7129 | 0.7537 | **14.3%** |

**ConvLSTM** (`vanilla_best.pt`):

| Protocol | Test | Val | Train | Test skill |
|---|---:|---:|---:|---:|
| Published: raw anchor, bias fitted on all periods | 0.8006 | 0.7622 | 0.7987 | 9.9% |
| Raw anchor, bias fitted on train only | 0.8010 | 0.7622 | 0.7986 | 9.9% |
| Raw anchor, no bias correction | 0.9752 | 0.9695 | 1.0030 | −9.7% |
| As trained: z-anchor, no correction | **0.7679** | 0.7317 | 0.7714 | 13.6% |

**PINN (λ=5, κ=100)** (`convLSTM_best.pt`):

| Protocol | Test | Val | Train | Test skill |
|---|---:|---:|---:|---:|
| Published: raw anchor, bias fitted on all periods | 0.8007 | 0.7650 | 0.8002 | 9.9% |
| Raw anchor, bias fitted on train only | 0.8015 | 0.7652 | 0.8001 | 9.8% |
| Raw anchor, no bias correction | 0.9535 | 0.9494 | 0.9907 | −7.3% |
| As trained: z-anchor, no correction | **0.7607** | 0.7226 | 0.7621 | 14.4% |

Test-period differences, from a moving-block bootstrap (blocks of 6 composites, 2,000 resamples):

| Difference in test RMSE | Point | 95% interval |
|---|---:|---|
| ConvLSTM (as trained) − climatology AR(1) | +0.006 | [−0.010, +0.020] |
| PINN (as trained) − climatology AR(1) | −0.001 | [−0.017, +0.011] |
| PINN checkpoint − ConvLSTM checkpoint (as trained) | −0.007 | [−0.012, −0.002] |
| Climatology AR(1) − persistence | −0.127 | [−0.152, −0.104] |

The PINN-checkpoint advantage compares two *different single training runs*. They differ in dropout, κ, λ and RNG state as well as physics. It does not survive the controlled comparison in §3: physics changes nothing, and seed variance alone is sd ≈ 0.005. It must not be quoted as a PINN advantage.

## 2. Code audit findings

Each finding is verified in code or artefacts unless marked *suspected*.

| # | Finding | Where | Effect on reported results |
|---|---|---|---|
| A1 | **Train/inference anchor mismatch.** Training forms `X[:,-1,log_chl] + Δ` from the *z-scored* last frame (NaN→0) and compares it with *raw* targets. Diagnostics adds Δ to the *raw* last frame. With σ = 1.217, the error has a state-dependent part of about 0.18·(C_t − μ) that a constant cannot remove. | `pinn/pinn_model.py:229,241-243`; `pinn/diagnostics.py:232-238`; `convLSTM/diagnostics.py:241` | All §4.1 RMSEs are worse than the models deserve (0.80 vs 0.76–0.77). The "residual on persistence" in README §3.1 is not what is trained. |
| A2 | **Test-label bias correction.** One mean error is fitted over train, val and test, then subtracted. | `pinn/diagnostics.py:518-521`; `convLSTM/diagnostics.py:508-511` | Leakage, but small in practice: test +0.0004 to +0.0008 when fitted on train only. It mostly masks A1. |
| A3 | **The physics term is about 1e-8** of the loss scale. The residual mixes z-scored u, v with metre and second grid spacing (dx = 4 km, dt = 8 d). | `pinn_model.py:81-91,263-296` | λ·L_phys ≤ 3e-5 against L_sup ≈ 0.3, so there is no regularising effect. Confirmed in the §3 training logs (phys ≈ 3e-8). |
| A4 | **The ablations are seed-order artefacts.** Row *i* of `ablation_lambda.csv` equals row *i* of `ablation_kappa.csv`, to about 1e-8. Seeds are set once at import, so each run continues the RNG stream. Weights come from the final epoch, not the best one. | `pinn/ablation_studies.py:107-198`; `paper_results/ablation/*` | §4.3 and the "optimal λ=5, κ=100" claim are unsupported. The 0.797–0.810 test spread is really a seven-sample estimate of run variance, and it is larger than the reported model gaps. |
| A5 | **κ is not learned.** final κ = initial κ, to about 1e-5. | `ablation_*.csv`; checkpoint κ = 99.996 | "κ learned during training" should be removed. |
| A6 | **`optimal_parameters.json` was hand-written,** and the optimised PINN came from uncommitted code. No script writes that file, and `pinn_model.py` never had κ=100 or a fixed λ in git history. | `paper_results/optimal_parameters.json`; `run_paper_experiments.py:173-176` | The optimised checkpoint cannot be regenerated from the repository. |
| A7 | **The "PINN" row cannot be reproduced.** `Diagnostics_PINN` was computed from `convLSTM_best.pt`, which was later overwritten (17 Dec 2025, κ = 100). | `Diagnostics_PINN/diagnostics_summary.json` | That row is now unreproducible. Candidate originals: `convLSTM_PINN_best.pt`, `best_pinn.pt` (unlabelled). |
| A8 | **The arms differ in more than physics.** Only the PINN applies `Dropout2d(0.1)` to its one-channel output, which zeroes the whole Δ map for about 10% of samples. README §3.2 says the ConvLSTM has it too. | `pinn_model.py:210,217`; `pinn/vanilla_model.py` | The ConvLSTM–PINN comparison is confounded. §3 removes this: both arms use the same module. |
| A9 | **The architecture is not convolutional in space.** "ConvLSTM" is a per-pixel LSTM with 1×1 convolutions. Spatial context enters only through the physics term, which is inactive (A3). | `pinn_model.py:189-218` | Describe it as a pixel-wise LSTM, or add spatial kernels. |
| A10 | **Imputation rows are not like-for-like.** One model is used, never retrained. The targets are imputed too. Normalisation is recomputed on the imputed data. `pixel_ok` becomes all-True, so the filled rows score different pixels. | `pinn/imputation_sensitivity.py:80-292` | §4.4 and the "masking beats imputation" claim are unsupported. That also explains the odd pattern of lowest RMSE but lowest r for "original". |
| A11 | **Lead times are not forecasts.** The 16/24/32-day rows compare the 8-day prediction with later observations, against honest (k+1)-step persistence. | `lead_skill_summary.py:106-113` | §4.2 measures how fast a stale 8-day forecast decays, not multi-lead skill. Relabel it, or do autoregressive rollouts. |
| A12 | **An input uses the test period.** `chl_anom_monthly` subtracts a monthly climatology computed over all years. *Suspected small.* | `convLSTM/prepare_data.py:60-64` | Recompute it on train years only. |
| A13 | **Truth is cleaned with whole-record thresholds.** The outlier threshold and 3×3 median replacement use the whole record, including test targets. | `convLSTM/scrub_outliers.py:42-66` | Disclose it, or use a train-only threshold. It affects model and persistence equally. |
| A14 | **The evaluated cube has no temporal gaps** inside the ocean footprint: 1,626 pixels × 845 steps, all finite. The step that gap-filled `log_chl` is not in `new_ds/`. | `summary.json` (`n_masked_nan_last: 0`) | README §2–§4 describes "native MODIS gaps". The evaluation actually scores gap-filled values as truth, and that should be stated. |

## 3. Controlled retraining (complete)

**Design.** This was set before the results.

- Same data, loaders, model module (including dropout), optimiser, schedule and early stopping as `pinn_model.py`.
- Arms are `λ ∈ {0, adaptive}` × `anchor ∈ {z (as published), raw (true residual on persistence)}`.
- Seeds 0, 1, 2, each reseeded per run.
- Scored on the full grid, with no bias correction, using the forecast each arm was trained to produce.
- The reimplemented loss was checked against the original `train_one` on the same batch; the difference is exactly 0.

Source: `results/controlled/*.json` and `results/controlled/SUMMARY.md`, from `summarize_controlled.py`. Scores are full-grid RMSE in ln(mg m⁻³).

| Arm | Test RMSE (mean ± sd, 3 seeds) | Per seed (0, 1, 2) | Val RMSE | Epochs run |
|---|---|---|---|---|
| λ=0, z-anchor (ConvLSTM as published, with dropout) | 0.7692 ± 0.0054 | 0.7707, 0.7736, 0.7632 | 0.7330 ± 0.0067 | 28, 25, 31 |
| λ adaptive, z-anchor (PINN as published) | 0.7692 ± 0.0054 | 0.7707, 0.7736, 0.7632 | 0.7330 ± 0.0067 | 28, 25, 31 |
| λ=0, raw anchor | 0.7776 ± 0.0082 | 0.7847, 0.7685, 0.7795 | 0.7449 ± 0.0074 | 16, 25, 17 |
| λ adaptive, raw anchor | 0.7776 ± 0.0082 | 0.7847, 0.7685, 0.7795 | 0.7449 ± 0.0074 | 16, 25, 17 |
| *Reference: persistence* | 0.8889 | | 0.8543 | |
| *Reference: climatology + AR(1) (§1)* | 0.7618 | | 0.7129 | |

These are the paired same-seed differences in test RMSE.

**Physics on − off:**
- z anchor: 0.00000, 0.00000, 0.00000.
- raw anchor: 0.00000, 0.00000, 0.00000.
- In the z arms, physics was active from epoch 21 to the end (25–31). The largest physics loss logged was about 6e-8, against a supervised loss of about 0.3. κ stayed at 25.000.
- Raw-anchor seeds 0 and 2 early-stopped before epoch 21. **Physics never switched on** in those runs.
- The original protocol has the same property: any run that early-stops before epoch 21 is a plain ConvLSTM, whatever its label.

**Raw anchor − z anchor:** +0.0140, −0.0050, +0.0163.

**Outcomes:**
- **H1 is confirmed, and in a stronger form:** the arms are identical, not merely within the seed spread. As implemented, the PINN is the ConvLSTM.
- **H2 is not supported.**
  - The raw anchor averages +0.008 worse, worse in 2 of 3 seeds.
  - That is within the seed spread, and two raw runs stopped early.
  - So a true residual on persistence gives no gain here.
  - The z-anchor model learns its own persistence mapping, which acts like the damped persistence in the AR baselines.
- **H3 is confirmed.**
  - None of the 12 runs beats the climatology + AR(1) baseline: the best is 0.7632, the baseline 0.7618.
  - The single best network run ties it.

**Hypotheses, stated in advance:**

- **H1:** the adaptive-λ and λ=0 arms differ by less than the seed spread. That is expected from A3.
- **H2:** the raw anchor is no worse than the z anchor.
- **H3:** no arm beats the climatology AR(1) beyond the bootstrap interval.

## 4. Recommendations, in order

1. **Fix the evaluation.**
   - Score the forecast the model was trained to produce, with no bias correction fitted on evaluation data.
   - Add the climatology + AR(1) baseline to every table.
2. **Make physics matter, or drop the claim.**
   - Nondimensionalise the residual: physical u, v in m s⁻¹, consistent units, and the residual in the normalised log space.
   - Log the ratio of L_phys to L_sup.
   - Learn log κ with a sensible scale.
   - Then re-run λ and κ sweeps with reseeding and at least 3 seeds per setting.
   - Start physics before early stopping can end training. Otherwise, check that it ever switched on.
3. **Correct the record.** README §4.2–§4.4 and §5 need the changes above, and the paper may need an erratum for §4.1's skill and the ablation claims. That is the author's decision. Nothing here changes CoastWatch, which publishes no model forecasts from this research.
4. **Provenance.**
   - Commit the code that produced the optimised checkpoint, or retrain it from committed code.
   - Version checkpoints by content hash, never by a reused filename.

## 5. Reproduce

```bash
PY=~/.local/share/mamba/envs/habs/bin/python
CUBE="$HABS_DATA_ROOT/Data/Derived/HAB_convLSTM_core_v1_clean.nc"    # 890 MB, not in git
$PY research/pinn_audit/rescore.py --freeze "$CUBE" \
    --ckpt convlstm=$HOME/HAB_Models/vanilla_best.pt --ckpt pinn_opt=$HOME/HAB_Models/convLSTM_best.pt \
    --out research/pinn_audit/out                         # ~25 min CPU
$PY research/pinn_audit/bootstrap.py --freeze "$CUBE" --out research/pinn_audit/out
HABS_MODEL_DIR=/tmp/scratch $PY research/pinn_audit/controlled.py --check --out /tmp/ctl   # loss equivalence
HABS_MODEL_DIR=/tmp/scratch $PY research/pinn_audit/controlled.py --seeds 0 1 2 --lam adaptive --anchor z --out /tmp/ctl
```

Checkpoint hashes (md5):

- `vanilla_best.pt`: `23e63437c1fece6ee4744fc47f110f95`
- `convLSTM_best.pt`: `d12e8a3920985580a36223201c668f7d`
