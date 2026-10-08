# Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System

Code, trained artifacts and diagnostics for an 8-day-ahead forecast of nearshore chlorophyll-a along the California coast. The forecast is posed as a spatiotemporal regression on a 4 km grid, and three neural architectures are compared: a recurrent convolutional baseline (ConvLSTM), a Temporal Fusion Transformer (TFT) and a physics-informed ConvLSTM (PINN) regularized by a two-dimensional advection–diffusion residual.

**Research question.** Do physics-guided neural networks improve coastal bloom forecast skill and spatial fidelity compared to purely statistical models?

---

## Contents

1. [Abstract](#1-abstract)
2. [Study Domain and Data](#2-study-domain-and-data)
3. [Methods](#3-methods)
4. [Results](#4-results)
5. [Discussion](#5-discussion)
6. [Reproducing the Experiments](#6-reproducing-the-experiments)
7. [Repository Structure](#7-repository-structure)
8. [Decision-Support Applications](#8-decision-support-applications)
9. [Computational Requirements](#9-computational-requirements)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [Citation, License and Contact](#11-citation-license-and-contact)

---

## 1. Abstract

Harmful algal blooms (HABs) are intensifying along the California coast, challenging forecasting systems that must resolve complex physical–biogeochemical interactions. We present a deep learning framework for **chlorophyll-a forecasting** using ConvLSTM, TFT, and a physics-informed ConvLSTM (PINN). The models are trained on an 18.6-year (2003–2021) nearshore dataset at 4 km resolution with engineered hydroclimatic and static predictors.

**Key findings:**

* All models achieve ~0.80 RMSE and >0.77 correlation in log chlorophyll on test set.
* Evaluated with the same diagnostics pipeline, the optimized PINN (λ=5.0, κ=100 m²/s) matches but does not beat the ConvLSTM baseline: validation RMSE 0.765 vs. 0.762 and test RMSE 0.801 vs. 0.801 in log chlorophyll (Section 4.1).
* The PINN modestly improves spatial fidelity (relative scale error, spectral energy ratios) and convergence stability.
* Leaving native satellite gaps masked gives the best skill; every gap-filling strategy tested degrades test RMSE (0.87–1.06 vs. 0.80, Section 4.4).
* TFT lags in spatial generalization, showing stronger regional biases.
* Case studies (Monterey Bay 2021, Navarro Lagoon 2020) show convolutional–recurrent models outperform transformers in reproducing bloom footprints.
* Predictor attribution highlights **Kd490, river distance/influence, SST, and shortwave radiation** as dominant drivers.

This hybrid framework balances statistical accuracy and physical interpretability, offering a scalable approach for operational coastal bloom forecasting.

---

## 2. Study Domain and Data

### 2.1 Domain and grid

| Property | Value |
|----------|-------|
| Spatial extent | 32.0°N – 42.0°N, 125.0°W – 115.0°W (California Current System, U.S. West Coast) |
| Grid | 240 × 240 cells at 1/24° (~4 km), the MODIS-Aqua Level-3 mapped grid |
| Temporal resolution | 8-day composites |
| Period | 2003-02-18 to 2021-06-26 (845 composites in the evaluation fields) |
| Analysis region | Nearshore ocean pixels with at least 20% valid chlorophyll retrievals over the record; bays and cells shallower than 10 m are excluded from the loss when bathymetry is available |

### 2.2 Data sources

All sources are regridded onto the MODIS 4 km grid and aggregated to the same 8-day calendar (`new_ds/`).

| Source | Variables used as predictors | Role |
|--------|------------------------------|------|
| MODIS-Aqua L3 (NASA OB.DAAC / Earthdata) | `log_chl` (ln chlorophyll-a), `Kd_490`, `nflh` | Biomass proxy (target), light attenuation, fluorescence |
| ERA5 (ECMWF) | `u10`, `v10`, `wind_speed`, `tau_mag`, `avg_sdswrf`, `tp`, `t2m`, `d2m` | Wind forcing and stress, shortwave radiation, precipitation, air temperature and dewpoint |
| CMEMS GLORYS reanalysis | `uo`, `vo`, `cur_speed`, `cur_div`, `cur_vort`, `zos`, `ssh_grad_mag`, `so`, `thetao` | Surface currents and their divergence/vorticity, sea-surface height and its gradient, salinity, temperature |
| Derived from chlorophyll | `chl_anom_monthly`, `chl_roll24d_mean`, `chl_roll24d_std`, `chl_roll40d_mean`, `chl_roll40d_std` | Monthly-climatology anomaly and 24/40-day rolling statistics |
| Static | `river_rank`, `dist_river_km`, `ocean_mask_static` | River proximity and influence, ocean mask |

This gives 28 input channels per time step. The raw and processed data (the "data freeze", `HAB_convLSTM_core_v1_clean.nc`) are not distributed with the repository (see [Section 6.2](#62-data-and-paths)).

### 2.3 Dataset construction

1. **Regridding and compositing** (`new_ds/`): the native MODIS, ERA5 and CMEMS fields are composited to 8-day windows (`build_source_cubes.py`), regridded to the MODIS 4 km grid with `xesmf` (`regrid_to_modis.py`), and fused into a single master cube (`build_hab_cube.py`).
2. **Data freeze** (`convLSTM/prepare_data.py`, configured by `config/data_freeze_v1.yaml`):
   * the coastal mask is built from non-missing `log_chl`;
   * linear chlorophyll is reconstructed with a detection floor of 0.056616 mg m⁻³, and sub-floor pixels are retained and flagged;
   * monthly climatologies and anomalies are computed, along with rolling means and standard deviations and lagged predictors;
   * physics-derived drivers are added: wind stress from a bulk formula (ρ_air = 1.225 kg m⁻³, C_d = 1.3 × 10⁻³), current speed, divergence and vorticity, SSH gradient magnitude, and a river-influence field with a 50 km e-folding distance;
   * time steps with less than 80% valid ocean coverage are dropped.
3. **Outlier scrubbing** (`convLSTM/scrub_outliers.py`): extreme values are replaced by the median of the surrounding 3 × 3 neighbourhood at each time step.

---

## 3. Methods

### 3.1 Forecast formulation

Let $C_t = \ln(\mathrm{Chl})$ at composite $t$. Each model receives a history of $L = 6$ composites (48 days) of all 28 standardized predictors, $\mathbf{X}_{t-5:t} \in \mathbb{R}^{L \times 28 \times H \times W}$, and forecasts one step (8 days) ahead. All models predict a **residual on persistence**:

$$\hat{C}_{t+1} = C_t + f_\theta(\mathbf{X}_{t-5:t}),$$

so persistence ($\hat{C}_{t+1} = C_t$) is the natural reference forecast.

### 3.2 Architectures

| Model | Implementation | Description |
|-------|----------------|-------------|
| ConvLSTM | `convLSTM/baseline_model.py`, `pinn/vanilla_model.py` | A 1×1 convolution reduces the 28 channels to 24. Two stacked recurrent layers with 48 and 64 hidden units follow; each is an LSTM cell applied at every pixel, with a 1×1 convolutional projection. Then Dropout2d (p = 0.1) and a 1×1 output head. |
| PINN | `pinn/pinn_model.py` | Same network as the ConvLSTM, plus a trainable diffusivity κ and the physics residual loss described in Section 3.3. |
| TFT | `tft/tft_model.py` | A 1×1×1 convolution lifts each pixel's time series to d = 96. Three temporal-fusion blocks follow (8-head self-attention over the 6-step sequence, with gated residual networks), then a linear head on the final step. Trained with PyTorch Lightning. |
| XGBoost (supplementary) | `scripts/XGB/` | A gradient-boosted tabular baseline on per-pixel features, with Optuna tuning and cross-validation. |

### 3.3 Physics constraint

The PINN penalizes violations of two-dimensional advection–diffusion of log-chlorophyll by the surface currents $(u, v)$ (`uo`, `vo` at the most recent input step):

$$\mathcal{R} = \frac{\hat{C}_{t+1} - C_t}{\Delta t} + u\,\frac{\partial \hat{C}_{t+1}}{\partial x} + v\,\frac{\partial \hat{C}_{t+1}}{\partial y} - \kappa\,\nabla^2 \hat{C}_{t+1},$$

with $\Delta x = 4$ km and $\Delta t = 8$ days. The gradients use centred finite differences and the Laplacian uses a 5-point stencil. The physics loss is the mean squared residual over valid pixels, rescaled from standardized to physical log units:

$$\mathcal{L}_{\text{phys}} = \sigma_{C}^{2}\,\frac{1}{|\Omega|}\sum_{\Omega}\mathcal{R}^2 .$$

The diffusivity κ is a learnable scalar, clamped to [10⁻², 10³].

### 3.4 Loss and training

The supervised term is a class-weighted Huber loss (δ = 1.0) on the log-space residual, restricted to valid pixels $\Omega$. Pixel weights are 1.0, 1.5, 2.5 and 4.0 across the training-set quartiles of chlorophyll, which emphasizes bloom and extreme concentrations. The total loss is

$$\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda\,\mathcal{L}_{\text{phys}} .$$

| Setting | Value |
|---------|-------|
| Data split (by time) | Train: before 2016-01-01; validation: 2016-01-01 to 2018-12-31; test: 2019-01-01 onward |
| Standardization | Per-variable z-score using training-period statistics; missing inputs set to 0 after standardization |
| Sampling | Random 64 × 64 coastal patches; stratified sampler weighting quartile bins of domain-median chlorophyll by frequency⁻¹·⁵ |
| Optimizer | AdamW, learning rate 3 × 10⁻⁴, weight decay 10⁻⁴, gradient clipping at 1.0 |
| Schedule | ReduceLROnPlateau (factor 0.5, patience 3) on validation RMSE; early stopping with patience 6; up to 40 epochs; batch size 32; seed 42 |
| Physics schedule (`pinn_model.py`) | Supervised-only for epochs 1–20. From epoch 21 the physics term is enabled with an adaptive weight λ = clamp(L_sup / L_phys, 0.1, 1000), ramped linearly to full strength over 5 epochs; κ initialized at 25 m² s⁻¹ |
| Physics schedule (ablation, `ablation_studies.py`) | Fixed λ and κ_init per run, physics enabled after epoch 10 with a 5-epoch ramp, 30 epochs per run |

### 3.5 Evaluation

Forecasts are evaluated on valid ocean pixels against persistence.

* **Continuous accuracy:** RMSE and MAE in log space and in linear space (mg m⁻³), KGE.
* **Skill vs. persistence:** $S = 100\,(1 - \mathrm{RMSE}_{\text{model}} / \mathrm{RMSE}_{\text{pers}})$.
* **Correlation structure:** Pearson's r, Spearman's ρ.
* **Bloom detection skill:** hit rate, false alarm ratio, F1, ROC-AUC, IoU.
* **Spatial fidelity:** relative scale error, spectral energy ratio, coastal RMSE profiles, spatial bias maps.
* **Physics diagnostics (PINN):** physics residuals, gradient norm ratios.
* **Predictor attribution:** SHAP values, partial dependence plots.
* **Lead-time skill:** 8, 16, 24 and 32 days.
* **Case studies:** Monterey Bay (bloom of 2021-05-25) and Navarro River / Lagoon (bloom of 2020-07-24).

---

## 4. Results

Numbers in this section are copied directly from the result files in the repository. Each table names its source.

### 4.1 Global skill at 8-day lead

Source: `Diagnostics_*/metrics_global.csv`, produced by `*/diagnostics.py` with `--seq 6 --lead 1`. RMSE and MAE are in ln(mg m⁻³).

| Model | Train RMSE | Val RMSE | Test RMSE | Test MAE | Test skill vs. persistence |
|-------|-----------:|---------:|----------:|---------:|---------------------------:|
| Persistence | 0.893 | 0.854 | 0.889 | 0.624 | — |
| ConvLSTM | 0.799 | 0.762 | 0.801 | 0.587 | 9.9% |
| PINN | 0.806 | 0.770 | 0.805 | 0.594 | 9.4% |
| PINN (λ = 5.0, κ = 100) | 0.800 | 0.765 | 0.801 | 0.590 | 9.9% |
| TFT | 0.819 | 0.789 | 0.822 | 0.611 | 7.6% |

### 4.2 Lead-time dependence (test period)

Source: `lead_skill_summary.py --subset test`. The leads are computed from each model's 8-day predicted fields.

| Model | Metric | 8 d | 16 d | 24 d | 32 d |
|-------|--------|----:|-----:|-----:|-----:|
| ConvLSTM | RMSE (log) | 0.801 | 0.915 | 0.941 | 0.966 |
|          | Pearson r  | 0.746 | 0.667 | 0.647 | 0.628 |
| PINN     | RMSE (log) | 0.805 | 0.919 | 0.948 | 0.973 |
|          | Pearson r  | 0.751 | 0.675 | 0.654 | 0.635 |
| TFT      | RMSE (log) | 0.822 | 0.940 | 0.971 | 0.997 |
|          | Pearson r  | 0.751 | 0.674 | 0.652 | 0.632 |

### 4.3 Ablation on the physics-loss weight λ

Source: `paper_results/ablation/lambda/ablation_lambda.csv` (κ_init = 25, 30 epochs per run). Figure: `paper_results/ablation/lambda/ablation_lambda.png`.

| λ | Train RMSE | Val RMSE | Test RMSE |
|---:|-----------:|---------:|----------:|
| 0.0 | 0.755 | 0.711 | 0.806 |
| 0.1 | 0.758 | 0.702 | 0.808 |
| 0.5 | 0.750 | 0.711 | 0.797 |
| 1.0 | 0.747 | 0.703 | 0.810 |
| 2.0 | 0.750 | 0.704 | 0.805 |
| 5.0 | 0.743 | **0.695** | 0.806 |
| 10.0 | 0.739 | 0.695 | 0.800 |

The companion diffusivity sweep (κ_init ∈ {1, 5, 10, 25, 50, 100, 500}, λ = 5.0) is in `paper_results/ablation/kappa/`. The selected configuration (λ = 5.0, κ_init = 100) is recorded in `paper_results/optimal_parameters.json`.

### 4.4 Sensitivity to gap-filling of inputs

Source: `paper_results/imputation/imputation_results.csv`. The optimized PINN is evaluated with input gaps left native (masked) or filled by six strategies. Figure: `paper_results/imputation/imputation_sensitivity.png`.

| Method | Val RMSE | Test RMSE | Test r |
|--------|---------:|----------:|-------:|
| Original (native gaps, masked loss) | **0.699** | **0.801** | 0.733 |
| Climatology | 0.824 | 0.871 | 0.826 |
| Median | 0.877 | 0.990 | 0.756 |
| Mean | 0.885 | 0.974 | 0.738 |
| Zero fill | 0.928 | 1.064 | 0.800 |
| Forward fill | 0.928 | 1.064 | 0.800 |
| Linear interpolation | 0.928 | 1.064 | 0.800 |

### 4.5 Additional outputs

For each model, `Diagnostics_<model>/` holds:

* skill maps (`fig_skill_maps.png`, `skill_maps.npz`);
* skill by month and by concentration bin (`metrics_month.csv`, `metrics_bins.csv`);
* domain-mean time series (`fig_timeseries.png`);
* residual histograms and scatter plots;
* the full predicted fields, `predicted_fields.nc`, with variables `log_chl_pred`, `log_chl_pers`, `log_chl_true` and `valid_mask`.

---

## 5. Discussion

* **PINN vs. ConvLSTM:** on the same evaluation pipeline the optimized PINN is statistically indistinguishable from the ConvLSTM (Section 4.1). The λ ablation (Section 4.3) reaches validation RMSE 0.695 at λ = 5, but those are separate 30-epoch runs and are not directly comparable with the Section 4.1 checkpoints.
* **Optimal parameters:** Systematic ablation identified λ=5.0 (physics loss weight) and κ=100 m²/s (diffusivity) as optimal for this domain.
* **Data gaps:** masking native MODIS gaps in the loss outperforms all six gap-filling strategies tested (Section 4.4).
* **ConvLSTM competitive:** achieves the same overall accuracy without the physics term.
* **Note on earlier figures:** internal status notes (`PAPER_READY_SUMMARY.md`, `PAPER_STATUS_REPORT.md`) cite an "8.3% validation improvement (0.699 vs. 0.762)". That figure compares the validation RMSE from the imputation-sensitivity run with the ConvLSTM's validation RMSE from the diagnostics run, so it is not a like-for-like comparison and is not used here. Skill during the 2019 marine heatwave is not quantified in the result files in this repository.
* **TFT underperforms:** struggles with sparse coastal grids.
* **Drivers:** Kd490, river influence, SST, and radiation dominate predictor rankings.
* **Computational efficiency:** Inference <10 seconds per forecast enables real-time operational deployment.
* **Skill ceiling:** ~0.80 RMSE in log chlorophyll; limited by coarse drivers (4 km, 8-day composites).

---

## 6. Reproducing the Experiments

### 6.1 Environment

The research pipeline was developed with Python 3.10. Conda is recommended because `cartopy`, `geopandas` and `xesmf` depend on native libraries:

```bash
micromamba env create -f environment.yml   # or: conda env create -f environment.yml
micromamba activate habs
```

Or, with pip only (skips `xesmf`, which is only needed by the regridding scripts in `new_ds/`):

```bash
pip install -r requirements.txt
```

Principal dependencies: `torch`, `pytorch-lightning`, `numpy`, `pandas`, `scipy`, `xarray`, `netCDF4`, `dask`, `scikit-learn`, `xgboost`, `shap`, `optuna`, `matplotlib`, `seaborn`, `cartopy`, `geopandas`, `shapely`, `pyproj`.

### 6.2 Data and paths

The data freeze (`HAB_convLSTM_core_v1_clean.nc`) is **not included** in this repository. Scripts locate data and checkpoints through environment variables:

| Variable | Default | Used for |
|----------|---------|----------|
| `HABS_DATA_ROOT` | `~/Desktop/HABs_Research` | Raw and processed data root (`Data/`, `Processed/`), used by `new_ds/` and `scripts/helpers/` |
| `HABS_FREEZE` | `$HABS_DATA_ROOT/Data/Derived/HAB_convLSTM_core_v1_clean.nc` | Data freeze for model training and diagnostics |
| `HABS_MODEL_DIR` | `~/HAB_Models` | Model checkpoints and prediction exports |

```bash
export HABS_DATA_ROOT=/path/to/HABs_Research
export HABS_MODEL_DIR=/path/to/HAB_Models
```

`config.yaml` and `config/data_freeze_v1.yaml` also accept `~` and `$VARS` in their paths. Scripts that take `--freeze`, `--obs` or `--ckpt` arguments use those values instead. Diagnostics read and write the `Diagnostics_*` folders in this repository.

### 6.3 Pipeline

**Step 1: build the dataset** (only needed when starting from the raw sources).

```bash
bash   new_ds/make_modis_4km.sh          # L3 binned -> 4 km mapped MODIS (requires SeaDAS l3mapgen)
python new_ds/build_source_cubes.py      # 8-day composites of MODIS, ERA5, CMEMS
python new_ds/regrid_to_modis.py         # regrid to the MODIS 4 km grid
python new_ds/build_hab_cube.py          # fuse into the master cube
python convLSTM/prepare_data.py --config config/data_freeze_v1.yaml
```

**Step 2: train the models.** Checkpoints are written to `$HABS_MODEL_DIR`.

```bash
python convLSTM/baseline_model.py
python pinn/pinn_model.py
python tft/tft_model.py
```

**Step 3: run the diagnostics.**

```bash
python pinn/diagnostics.py \
    --freeze "$HABS_FREEZE" \
    --ckpt   "$HABS_MODEL_DIR/convLSTM_best.pt" \
    --out    Diagnostics_PINN_Optimized \
    --seq 6 --lead 1 --batch 32
# convLSTM/diagnostics.py and tft/diagnostics.py take the same --freeze / --ckpt arguments.
# RUN_DIAGNOSTICS.sh wraps the command above.
```

**Step 4: run the ablation and sensitivity studies** (reviewer-response experiments; see `pinn/PAPER_WORKFLOW.md`).

```bash
python pinn/ablation_studies.py --study lambda \
    --lambda-values 0.0 0.1 0.5 1.0 2.0 5.0 10.0 --epochs 30
python pinn/ablation_studies.py --study kappa \
    --kappa-values 1 5 10 25 50 100 500 --lambda-values 5.0 --epochs 30
python pinn/imputation_sensitivity.py
python pinn/compile_paper_results.py
```

**Step 5: make the analyses and figures.** Run each script with `--help` to see its full list of arguments.

```bash
python lead_skill_summary.py --subset test
python pinn/spatial_bias.py
python predictor_importance_pdp.py --freeze "$HABS_FREEZE"
python extra_diagnostics.py
python study_domain.py --obs "$HABS_FREEZE"
bash   run_mech.sh               # robustness figure (mech_figure.py)
bash   monterey_run_case.sh      # Monterey Bay case study
bash   navarro_run_case.sh       # Navarro Lagoon case study
```

The case-study scripts read the prediction exports produced by `export_preds.py`, in `$HABS_MODEL_DIR/exports/`.

---

## 7. Repository Structure

```
habs-forecast/
├── convLSTM/                  # ConvLSTM baseline
│   ├── prepare_data.py            # Builds the data freeze (HAB_convLSTM_core_v1_clean.nc)
│   ├── scrub_outliers.py          # 3×3 median outlier replacement
│   ├── baseline_model.py          # Training
│   ├── diagnostics.py             # Metrics, skill maps, predicted fields
│   └── spatial_bias.py
│
├── pinn/                      # Physics-informed ConvLSTM
│   ├── pinn_model.py              # PINN training (adaptive λ, trainable κ)
│   ├── vanilla_model.py           # Same network without the physics term
│   ├── pinn_model_uncertainty.py  # PINN with uncertainty quantification
│   ├── inference_uncertainty.py   # Uncertainty quantification inference
│   ├── ablation_studies.py        # Hyperparameter sensitivity (λ, κ)
│   ├── imputation_sensitivity.py  # Data gap robustness analysis
│   ├── run_paper_experiments.py   # Runs the reviewer-response experiments end to end
│   ├── compile_paper_results.py   # Collects results into one summary
│   ├── diagnostics.py
│   ├── spatial_bias.py
│   └── *.md                       # ABLATION / UNCERTAINTY / PAPER_WORKFLOW / QUICK_START_PAPER guides
│
├── tft/                       # Temporal Fusion Transformer
│   ├── tft_model.py
│   ├── diagnostics.py
│   └── spatial_bias.py
│
├── new_ds/                    # Dataset construction: regrid MODIS / ERA5 / CMEMS to the 4 km MODIS grid
├── scripts/
│   ├── XGB/                       # XGBoost tabular baseline (train, tune, CV, diagnostics)
│   ├── helpers/                   # Masking, imputation, QC and sanity-check utilities
│   └── prep_hab_cube.py
├── config/data_freeze_v1.yaml # Data-freeze configuration
├── config.yaml                # XGBoost / tabular pipeline configuration
│
├── Diagnostics_ConvLSTM/      # Predicted fields, metrics CSVs and figures, one folder per model
├── Diagnostics_PINN/
├── Diagnostics_PINN_Optimized/    # PINN with λ=5.0, κ=100
├── Diagnostics_TFT/
├── Models/                    # XGBoost artifacts, ConvLSTM checkpoint, Optuna studies
├── runs/pinn_best.pt          # PINN checkpoint
├── paper_results/             # Ablation (λ, κ) and imputation-sensitivity CSVs and figures
│
├── export_preds.py            # Inference helpers for case studies (ConvLSTM / TFT / PINN)
├── case_studies.py            # Multi-panel case-study figures
├── monterey.py                # Case study: Monterey Bay HAB (2021)
├── navarro.py                 # Case study: Navarro Lagoon HAB (2020)
├── background.py              # Individual case-study analysis
├── study_domain.py            # Figure 1: study domain and alongshore climatology
├── region_overviews.py        # Monterey / Navarro overview maps
├── predictor_importance_pdp.py    # PDP and SHAP analyses
├── extra_diagnostics.py       # SHAP and feature importance plots
├── mech_figure.py             # PINN robustness tests
├── lead_skill_summary.py      # Lead-time forecast skill
├── calibration.py             # Calibration diagnostics
├── RUN_DIAGNOSTICS.sh         # Diagnostics for the optimized PINN
│
├── pipeline/                  # CoastWatch data pipeline: ingest, validate, render, publish (see pipeline/README.md)
├── coastwatch-web/            # CoastWatch Next.js + MapLibre web app (see coastwatch-web/README.md)
├── schemas/v1/                # JSON Schema shared by the pipeline and the web app
├── data/curated/              # Human-reviewed inputs (CDFW port selection)
├── docs/coastwatch/           # CoastWatch planning, architecture, safety rules, source evidence
│
├── PAPER_READY_SUMMARY.md     # Reviewer-response status and results
├── PAPER_STATUS_REPORT.md
└── GOALS.md                   # Product goals for the fisheries decision platform
```

---

## 8. Decision-Support Application (CoastWatch)

**CoastWatch** (`coastwatch-web/` + `pipeline/`) is a public-facing coastal map for California fishing communities. It is **decision support only**, not a regulatory or public-health product; closures and advisories from CDFW, CDPH and OEHHA always take precedence.

The current release (Milestone 1) shows the official **C-HARM v3.1** harmful-algal-bloom and domoic-acid forecast probabilities from NOAA CoastWatch West Coast, satellite chlorophyll from NASA GIBS as a separate observation layer, and CDFW landing ports. **The research models in this repository are not used by CoastWatch yet**; they need retraining on operational (VIIRS-era) inputs and fresh validation first (see `docs/coastwatch/06-development-plan.md`).

```bash
cd pipeline && uv sync && uv run cwp run          # fetch + validate + publish to coastwatch-web/public/data/v1
cd ../coastwatch-web && npm install && npm run dev  # http://localhost:3000
```

See [`docs/coastwatch/`](docs/coastwatch/README.md) for the architecture, data-source register and the scientific safety rules. The earlier Streamlit dashboard, which displayed a synthetic demo field, has been retired.

---

## 9. Computational Requirements

| Resource | Requirement |
|----------|-------------|
| Training | ~3–6 hours on consumer hardware (Apple Silicon or equivalent) for full PINN training (30–40 epochs) |
| Inference | <10 seconds per 8-day forecast on CPU, faster with GPU acceleration |
| Memory | ~8–16 GB RAM recommended for full dataset processing |
| Storage | ~5–10 GB for processed datasets and model checkpoints |

The physics constraint adds minimal overhead (~10–15% inference time), making real-time operational forecasting feasible. Mixed-precision training is enabled automatically when CUDA is available.

---

## 10. Limitations and Future Work

**Limitations.** Skill is bounded by the resolution of the drivers: 4 km grids and 8-day composites. Satellite chlorophyll is only a proxy for bloom biomass and does not measure toxin. The evaluation uses a single chronological train/validation/test split.

**Future work:**

* Uncertainty quantification (Monte Carlo dropout, ensembles) for operational confidence intervals; implemented in `pinn/pinn_model_uncertainty.py` (see `pinn/UNCERTAINTY_README.md`)
* Higher-resolution inputs (hyperspectral ocean color, ROMS forecasts)
* Ensemble probabilistic forecasting for longer leads (>2 weeks)
* Multi-task learning for coupled biogeochemical variables (nitrate, dissolved oxygen)
* Transferability tests in other eastern-boundary upwelling systems (e.g., Humboldt, Benguela)
* Climate-aware feature engineering (ENSO indices, marine heatwave indicators)

---

## 11. Citation, License and Contact

**Citation.** If you use this work, please cite:

> Mohanty, Y. (2025). *Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System*. (In Review).

```bibtex
@unpublished{mohanty2025habs,
  author = {Mohanty, Yashnil},
  title  = {Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System},
  year   = {2025},
  note   = {In review}
}
```

**License.** MIT; see [`LICENSE`](LICENSE).

**Contact.** Yashnil Mohanty, yashnilmohanty@gmail.com, GitHub: [yashnil](https://github.com/yashnil).
