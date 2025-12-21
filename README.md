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
│
├── convLSTM/               # ConvLSTM model + diagnostics
│   ├── baseline\_model.py
│   ├── diagnostics.py
│   └── spatial\_bias.py
│
├── pinn/                   # Physics-informed ConvLSTM
│   ├── pinn\_model.py          # Main PINN model (optimal: λ=5.0, κ=100)
│   ├── vanilla\_model.py       # Baseline ConvLSTM (no physics)
│   ├── pinn\_model\_uncertainty.py  # PINN with uncertainty quantification
│   ├── ablation\_studies.py    # Hyperparameter sensitivity (λ, κ)
│   ├── imputation\_sensitivity.py  # Data gap robustness analysis
│   ├── diagnostics.py
│   ├── spatial\_bias.py
│   └── inference\_uncertainty.py   # Uncertainty quantification inference
│
├── tft/                    # Temporal Fusion Transformer
│   ├── tft\_model.py
│   ├── diagnostics.py
│   └── spatial\_bias.py
│
├── Diagnostics\_ConvLSTM/   # Predicted fields + plots
├── Diagnostics\_PINN/
├── Diagnostics\_TFT/
│
├── monterey.py             # Case study: Monterey Bay HAB (2021)
├── navarro.py              # Case study: Navarro Lagoon HAB (2020)
│
├── predictor\_importance\_pdp.py   # PDP + SHAP analyses
├── extra\_diagnostics.py          # SHAP + feature importance plots
├── mech\_figure.py                # PINN robustness tests
├── lead\_skill\_summary.py         # Lead time forecast skill

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

Install required packages:  
```

pip install -r requirements.txt

```

Key libraries:  
* `torch`, `torchvision` (PyTorch)  
* `scikit-learn`, `xgboost`, `shap`  
* `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`  
* `xarray`, `rasterio`, `geopandas`, `cartopy`, `netCDF4`  

### ▶️ Running Models  

Activate environment and run:  
```

micromamba activate habs
cd \~/Desktop/habs-forecast
python convLSTM/baseline\_model.py
python pinn/pinn\_model.py
python tft/tft\_model.py

```

### 📈 Diagnostics and Plots  

```bash
# Standard diagnostics
python convLSTM/diagnostics.py
python pinn/diagnostics.py
python pinn/spatial\_bias.py
python predictor\_importance\_pdp.py
python extra\_diagnostics.py
python mech\_figure.py
python lead\_skill\_summary.py
python monterey.py
python navarro.py

# Additional analyses
python pinn/ablation_studies.py --study lambda  # Find optimal λ
python pinn/ablation_studies.py --study kappa   # Find optimal κ
python pinn/imputation_sensitivity.py          # Data gap robustness
```

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

This project is licensed under the MIT License.  
```
