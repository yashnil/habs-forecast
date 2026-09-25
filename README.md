# 🌊 Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System

## 🌲 Overview  

This project develops a physics-guided deep learning framework for forecasting **harmful algal blooms (HABs)** along the California Current System. HABs, driven by climate variability and nutrient enrichment, threaten ecosystems, fisheries, and public health. Our models integrate **satellite ocean color, ocean reanalysis, and atmospheric predictors** to test whether adding **physical transport constraints** improves bloom prediction.

We compare three architectures:  
- **ConvLSTM** – convolutional long short-term memory network  
- **TFT** – Temporal Fusion Transformer  
- **PINN** – physics-informed ConvLSTM enforcing 2D advection–diffusion dynamics  

The central question: *Do physics-guided neural networks improve coastal bloom forecast skill and spatial fidelity compared to purely statistical models?*  

---

## 📘 Abstract  

Harmful algal blooms (HABs) are intensifying along the California coast, challenging forecasting systems that must resolve complex physical–biogeochemical interactions. We present a deep learning framework for **chlorophyll-a forecasting** using ConvLSTM, TFT, and a physics-informed ConvLSTM (PINN). The models are trained on an 18.6-year (2003–2021) nearshore dataset at 4 km resolution with engineered hydroclimatic and static predictors.  

**Key Findings:**  
* All models achieve ~0.80 RMSE and >0.77 correlation in log chlorophyll on test set.  
* The PINN shows 8.3% validation improvement (RMSE: 0.699 vs. 0.762 baseline) with optimal parameters (λ=5.0, κ=100 m²/s).  
* The PINN modestly improves spatial fidelity (relative scale error, spectral energy ratios) and convergence stability.  
* Model demonstrates resilience to climate extremes (2019 marine heatwave) with <3% skill degradation.  
* Robustness analysis confirms model performance is maintained across diverse data imputation methods.  
* TFT lags in spatial generalization, showing stronger regional biases.  
* Case studies (Monterey Bay 2021, Navarro Lagoon 2020) show convolutional–recurrent models outperform transformers in reproducing bloom footprints.  
* Predictor attribution highlights **Kd490, river distance/influence, SST, and shortwave radiation** as dominant drivers.  

This hybrid framework balances statistical accuracy and physical interpretability, offering a scalable approach for operational coastal bloom forecasting.  

---

## 📁 Repository Structure  

```
habs-forecast/
├── convLSTM/                  # ConvLSTM baseline
│   ├── prepare_data.py            # Builds the data-freeze NetCDF (HAB_convLSTM_core_v1_clean.nc)
│   ├── scrub_outliers.py          # 3×3 median outlier replacement
│   ├── baseline_model.py          # Training
│   ├── diagnostics.py             # Metrics, skill maps, predicted fields
│   └── spatial_bias.py
│
├── pinn/                      # Physics-informed ConvLSTM
│   ├── pinn_model.py              # Main PINN model (optimal: λ=5.0, κ=100)
│   ├── vanilla_model.py           # Baseline ConvLSTM (no physics)
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
├── config.yaml                # XGBoost / tabular pipeline config
│
├── Diagnostics_ConvLSTM/      # Predicted fields, metrics CSVs, figures per model
├── Diagnostics_PINN/
├── Diagnostics_PINN_Optimized/    # PINN with λ=5.0, κ=100
├── Diagnostics_TFT/
├── Models/                    # XGBoost artifacts, ConvLSTM checkpoint, Optuna studies
├── runs/pinn_best.pt          # PINN checkpoint
├── paper_results/             # Ablation (λ, κ) and imputation-sensitivity CSVs + figures
│
├── export_preds.py            # Inference helpers for case studies (ConvLSTM / TFT / PINN)
├── case_studies.py            # Multi-panel case-study figures
├── monterey.py                # Case study: Monterey Bay HAB (2021)
├── navarro.py                 # Case study: Navarro Lagoon HAB (2020)
├── background.py              # Individual case-study analysis
├── study_domain.py            # Figure 1: study domain + alongshore climatology
├── region_overviews.py        # Monterey / Navarro overview maps
├── predictor_importance_pdp.py    # PDP + SHAP analyses
├── extra_diagnostics.py       # SHAP + feature importance plots
├── mech_figure.py             # PINN robustness tests
├── lead_skill_summary.py      # Lead time forecast skill
├── calibration.py             # Calibration diagnostics
├── RUN_DIAGNOSTICS.sh         # Diagnostics for the optimized PINN
│
├── dashboard/                 # Streamlit decision-support dashboard (see dashboard/README.md)
├── coastwatch-web/            # Next.js + Mapbox web front end (see coastwatch-web/README.md)
│
├── PAPER_READY_SUMMARY.md     # Reviewer-response status and results
├── PAPER_STATUS_REPORT.md
└── GOALS.md                   # Product goals for the fisheries decision platform
```

---

## 🧪 Experiments Overview  

| Model | Key Purpose |
|-------|-------------|
| **ConvLSTM** | Data-driven spatiotemporal benchmark |
| **TFT** | Attention-based probabilistic forecasting |
| **PINN** | Physics-informed ConvLSTM with advection–diffusion constraint |

---

## 📊 Evaluation Metrics  

* **RMSE, MAE, KGE** (continuous accuracy)  
* **Spearman’s ρ, Pearson’s r** (correlation structure)  
* **Bloom detection skill:** hit rate, false alarm ratio, F1, ROC-AUC, IoU  
* **Spatial fidelity:** relative scale error, spectral energy ratio, coastal RMSE profiles  
* **Physics diagnostics (PINN):** physics residuals, gradient norm ratios  
* **Predictor attribution:** SHAP values, PDPs, scatterplots  

---

## 📡 Data Sources  

| Dataset | Variables | Relevance |
|---------|-----------|------------|
| **MODIS-Aqua (NASA Earthdata)** | Chlorophyll-a (ln), Kd490, nFLH, SST | Biomass proxy, optical/light drivers |
| **ERA5 (ECMWF)** | Wind (u, v, τ), radiation, precipitation, T2m, dewpoint | Meteorological forcing |
| **CMEMS GLORYS** | Currents, SSH, vorticity, salinity | Transport, circulation, retention |
| **Static features** | River distance, coastal mask, latitude | Nutrient influence, spatial context |

---

## 🔍 Key Insights  

* **PINN improves realism:** 8.3% validation improvement (RMSE: 0.699 vs. 0.762) with optimal hyperparameters, plus stronger fidelity in bloom structure.  
* **Optimal parameters:** Systematic ablation identified λ=5.0 (physics loss weight) and κ=100 m²/s (diffusivity) as optimal for this domain.  
* **Climate resilience:** PINN maintains skill during extreme climate events (2019 marine heatwave) with <3% degradation vs. >3% for baseline.  
* **Data robustness:** Model performance robust to data gaps; native MODIS gaps handled effectively through masked loss computation.  
* **ConvLSTM competitive:** achieves similar overall accuracy without physics, but less robust in structure and climate extremes.  
* **TFT underperforms:** struggles with sparse coastal grids.  
* **Drivers:** Kd490, river influence, SST, and radiation dominate predictor rankings.  
* **Computational efficiency:** Inference <10 seconds per forecast enables real-time operational deployment.  
* **Skill ceiling:** ~0.80 RMSE in log chlorophyll; limited by coarse drivers (4 km, 8-day composites).  

---

## 💻 Getting Started  

### 📦 Dependencies  

The research pipeline was developed with Python 3.10. Conda is recommended because `cartopy`, `geopandas` and `xesmf` depend on native libraries:  

```bash
micromamba env create -f environment.yml   # or: conda env create -f environment.yml
micromamba activate habs
```

Or with pip only (skips `xesmf`, which is only needed for the regridding scripts in `new_ds/`):  

```bash
pip install -r requirements.txt
```

Key libraries:  
* `torch`, `pytorch-lightning` (PyTorch)  
* `scikit-learn`, `xgboost`, `shap`, `optuna`  
* `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`  
* `xarray`, `netCDF4`, `dask`, `cartopy`, `geopandas`, `shapely`, `pyproj`  

### 🗂 Data and paths  

The training data (the data-freeze file `HAB_convLSTM_core_v1_clean.nc`, built by `convLSTM/prepare_data.py` from the `new_ds/` cubes) is **not included** in this repository. Scripts locate data and checkpoints through environment variables:  

| Variable | Default | Used for |
|----------|---------|----------|
| `HABS_DATA_ROOT` | `~/Desktop/HABs_Research` | Raw/processed data root (`Data/`, `Processed/`), used by `new_ds/` and `scripts/helpers/` |
| `HABS_FREEZE` | `$HABS_DATA_ROOT/Data/Derived/HAB_convLSTM_core_v1_clean.nc` | Data-freeze file for model training and diagnostics |
| `HABS_MODEL_DIR` | `~/HAB_Models` | Model checkpoints and prediction exports |

```bash
export HABS_DATA_ROOT=/path/to/HABs_Research
export HABS_MODEL_DIR=/path/to/HAB_Models
```

`config.yaml` and `config/data_freeze_v1.yaml` also accept `~` and `$VARS` in their paths. Scripts that take `--freeze` / `--obs` / `--ckpt` arguments use those instead. Diagnostics read and write the `Diagnostics_*` folders in this repo.  

### ▶️ Running Models  

```bash
python convLSTM/baseline_model.py
python pinn/pinn_model.py
python tft/tft_model.py
```

### 📈 Diagnostics and Plots  

Diagnostics scripts need the data-freeze file and a trained checkpoint:  

```bash
python pinn/diagnostics.py \
    --freeze /path/to/HAB_convLSTM_core_v1_clean.nc \
    --ckpt   /path/to/checkpoint.pt \
    --out    Diagnostics_PINN_Optimized \
    --seq 6 --lead 1 --batch 32
# convLSTM/diagnostics.py and tft/diagnostics.py take the same --freeze / --ckpt arguments
```

Other analyses (run each with `--help` to see its arguments):  

```bash
python pinn/spatial_bias.py
python predictor_importance_pdp.py --freeze /path/to/HAB_convLSTM_core_v1_clean.nc
python extra_diagnostics.py
python mech_figure.py --obs ... --pred ... --start ... --end ...
python lead_skill_summary.py
python monterey.py --obs ... --pred ...
python navarro.py  --obs ... --pred ...

# Additional analyses
python pinn/ablation_studies.py --study lambda  # Find optimal λ
python pinn/ablation_studies.py --study kappa   # Find optimal κ
python pinn/imputation_sensitivity.py          # Data gap robustness
```

See `pinn/PAPER_WORKFLOW.md` and `pinn/QUICK_START_PAPER.md` for the full reviewer-response workflow.  

### ⚙️ Hyperparameter Optimization

The PINN model benefits from systematic hyperparameter tuning:

```bash
# Run ablation studies to find optimal parameters
python pinn/ablation_studies.py \
    --study lambda \
    --lambda-values 0.0 0.1 0.5 1.0 2.0 5.0 10.0 \
    --epochs 30

python pinn/ablation_studies.py \
    --study kappa \
    --kappa-values 1 5 10 25 50 100 500 \
    --lambda-values 5.0 \
    --epochs 30
```

**Current optimal parameters:** λ=5.0, κ=100 m²/s (from validation RMSE optimization)

---

## 🗺 Dashboard and Web App  

Two decision-support front ends visualize a chlorophyll snapshot exported from the pipeline. They are **not** regulatory or public-health products.  

* **`dashboard/`**: a lightweight Streamlit app. Quick start:  
  ```bash
  pip install -r dashboard/requirements.txt
  python dashboard/scripts/make_demo_snapshot.py   # synthetic demo data
  streamlit run dashboard/app.py
  ```
  Use `dashboard/scripts/export_map_snapshot.py` to export real model output. See [`dashboard/README.md`](dashboard/README.md).  
* **`coastwatch-web/`**: a Next.js 15 + Mapbox front end with NASA GIBS chlorophyll tiles, ports and regional summaries. It needs a Mapbox token in `.env.local` (copy from `.env.example`). See [`coastwatch-web/README.md`](coastwatch-web/README.md).  

A GitHub Actions workflow (`.github/workflows/dashboard-demo.yml`) smoke-tests demo snapshot generation.  

---

## 📑 Paper and Citation  

If you use this work, please cite:  
> Mohanty, Y. (2025). *Physics-Guided Neural Forecasts of Nearshore Harmful Algal Blooms in the California Current System*. (In Review).  

---

## 💻 Computational Requirements

* **Training:** ~3-6 hours on consumer hardware (Apple Silicon or equivalent) for full PINN training (30-40 epochs)
* **Inference:** <10 seconds per 8-day forecast on CPU, faster with GPU acceleration
* **Memory:** ~8-16 GB RAM recommended for full dataset processing
* **Storage:** ~5-10 GB for processed datasets and model checkpoints

The physics constraint adds minimal overhead (~10-15% inference time), making real-time operational forecasting feasible.

## 🧠 Future Work  

* Uncertainty quantification (Monte Carlo Dropout, ensembles) for operational confidence intervals
* Higher-resolution inputs (hyperspectral ocean color, ROMS forecasts)  
* Ensemble probabilistic forecasting for longer leads (>2 weeks)  
* Multi-task learning for coupled biogeochemical variables (nitrate, DO)  
* Transferability tests in other upwelling systems (e.g., Humboldt, Benguela)
* Climate-aware feature engineering (ENSO indices, marine heatwave indicators)  

---

## 📬 Contact  

Questions or feedback? Reach out to:  

- 📧 yashnilmohanty@gmail.com  
- 🔗 GitHub: yashnil  

---

## 🏷 License  

This project is licensed under the MIT License. See [`LICENSE`](LICENSE).  
```
