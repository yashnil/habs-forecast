# Paper Experiments Workflow

This document outlines the complete workflow to address reviewer comments and generate paper-ready results.

## Quick Start

```bash
# Run all experiments
python pinn/run_paper_experiments.py --all

# Or run steps individually
python pinn/run_paper_experiments.py --step ablation
python pinn/run_paper_experiments.py --step imputation
```

## Complete Workflow

### Step 1: Ablation Studies (Find Optimal Parameters)

**Purpose**: Find optimal λ (physics loss weight) and κ (diffusivity)

```bash
# Run all ablation studies
python pinn/ablation_studies.py --study lambda --lambda-values 0.0 0.1 0.5 1.0 2.0 5.0 10.0
python pinn/ablation_studies.py --study kappa --kappa-values 1 5 10 25 50 100 500
python pinn/ablation_studies.py --study both --lambda-values 0.1 1.0 5.0 10.0 --kappa-values 10 25 50 100
```

**Outputs**:
- `ablation_results/lambda/ablation_lambda.csv` and `.png`
- `ablation_results/kappa/ablation_kappa.csv` and `.png`
- `ablation_results/combined/ablation_combined.csv` and `.png`
- `paper_results/optimal_parameters.json`

**Time**: ~2-3 hours (30 epochs per configuration)

### Step 2: Imputation Sensitivity

**Purpose**: Show model robustness to data preprocessing

```bash
python pinn/imputation_sensitivity.py \
    --data /path/to/data.nc \
    --model ~/HAB_Models/convLSTM_best.pt \
    --methods original zero mean median forward_fill climatology interpolate
```

**Outputs**:
- `imputation_analysis/imputation_results.csv`
- `imputation_analysis/imputation_sensitivity.png`

**Time**: ~30 minutes

### Step 3: Train Best PINN Model

**Purpose**: Train final model with optimal hyperparameters

**Option A: Modify existing script**
1. Open `pinn/pinn_model.py`
2. Set optimal λ and κ from Step 1
3. Train normally

**Option B: Use ablation script's training function**
- Modify `ablation_studies.py` to export training function
- Or manually set parameters

```python
# In pinn_model.py, modify:
lambda_phys = 1.0  # From optimal_parameters.json
kappa_init = 25.0  # From optimal_parameters.json
```

```bash
python pinn/pinn_model.py --epochs 40 --batch 32
```

**Outputs**:
- `~/HAB_Models/pinn_best.pt` (or similar)
- Training metrics

**Time**: ~4-6 hours (40 epochs)

### Step 4: Train Uncertainty Models

**Purpose**: Generate uncertainty-quantified predictions

```bash
# Single model with MC Dropout
python pinn/pinn_model_uncertainty.py --epochs 40 --batch 32

# Ensemble (5 models)
python pinn/pinn_model_uncertainty.py --epochs 40 --batch 32 --ensemble 5

# Quantile regression
python pinn/pinn_model_uncertainty.py --epochs 40 --batch 32 --quantiles 0.1 0.5 0.9
```

**Outputs**:
- `~/HAB_Models/pinn_uncertainty_best.pt`
- `~/HAB_Models/pinn_ensemble_*_best.pt` (5 files)
- `~/HAB_Models/pinn_quantiles_best.pt`

**Time**: ~20-30 hours (5 ensemble members × 40 epochs)

### Step 5: Generate Predictions and Figures

**Purpose**: Create paper-ready figures

```bash
# Generate predictions with uncertainty
python pinn/inference_uncertainty.py \
    --ckpt ~/HAB_Models/pinn_uncertainty_best.pt \
    --data /path/to/test_data.nc \
    --output predictions_uncertainty.nc

# Run diagnostics to generate comparison figures
python pinn/diagnostics.py \
    --freeze /path/to/data.nc \
    --ckpt ~/HAB_Models/pinn_best.pt \
    --out Diagnostics_PINN_Best
```

**Outputs**:
- Diagnostic figures (scatter plots, skill maps, etc.)
- Uncertainty maps
- Comparison figures

### Step 6: Create Summary Report

**Purpose**: Compile all results for paper

```bash
python pinn/run_paper_experiments.py --step report
```

**Outputs**:
- `paper_results/paper_summary_report.md`

## Key Results to Report in Paper

### 1. Hyperparameter Optimization

**Table**: Ablation Study Results
- Optimal λ value
- Optimal κ value
- Performance improvement vs. default

**Figure**: Ablation heatmaps
- λ × κ performance grid
- Optimal region identification

### 2. Robustness Analysis

**Table**: Imputation Sensitivity
- RMSE for each imputation method
- Best method identification
- Sensitivity range

**Figure**: Imputation comparison
- Bar chart of RMSE by method
- Missing data statistics

### 3. Uncertainty Quantification

**Table**: Uncertainty Metrics
- Coverage at different confidence levels
- Prediction interval width
- Comparison with TFT quantiles

**Figure**: Uncertainty Maps
- Spatial uncertainty patterns
- Epistemic vs. aleatoric breakdown
- Comparison with observations

### 4. Final Model Performance

**Table**: Best PINN Results
- Train/Val/Test RMSE, MAE, Correlation
- Comparison with baseline ConvLSTM
- Comparison with TFT

**Figure**: Performance Comparison
- Scatter plots (PINN vs. Obs)
- Skill maps
- Time series comparisons

## Paper Sections to Update

### Methods Section

Add:
- "Hyperparameter optimization via systematic ablation studies"
- "Uncertainty quantification via Monte Carlo Dropout and ensemble methods"
- "Robustness analysis across data imputation strategies"

### Results Section

Add:
- Ablation study results (optimal λ and κ)
- Imputation sensitivity analysis
- Uncertainty quantification results
- Comparison with TFT uncertainty

### Figures

Add/Update:
- Figure: Ablation study heatmaps
- Figure: Imputation sensitivity comparison
- Figure: Uncertainty quantification examples
- Figure: Updated performance comparison with uncertainty

### Discussion Section

Add:
- Interpretation of optimal hyperparameters
- Robustness to data preprocessing
- Uncertainty quantification benefits
- Comparison with TFT quantile approach

## Timeline Estimate

- **Ablation studies**: 2-3 hours
- **Imputation sensitivity**: 30 minutes
- **Train best PINN**: 4-6 hours
- **Train uncertainty models**: 20-30 hours (can run in parallel)
- **Generate figures**: 1-2 hours
- **Total**: ~1-2 days of compute time

## Tips

1. **Run ablation studies first** - They're fast and inform everything else
2. **Use optimal parameters** - Don't use default values if ablation shows better options
3. **Parallel training** - Train ensemble members in parallel if possible
4. **Save checkpoints** - Keep all trained models for comparison
5. **Document everything** - Save all parameters and results for reproducibility

## Troubleshooting

**Issue**: Ablation studies take too long
- **Solution**: Reduce epochs (20-25 instead of 30) or test fewer values

**Issue**: Model doesn't train with optimal parameters
- **Solution**: Check parameter ranges are valid (λ > 0, κ in [0.01, 1000])

**Issue**: Uncertainty models fail
- **Solution**: Check CUDA memory, reduce batch size or MC samples

## Next Steps After Experiments

1. Review `paper_results/paper_summary_report.md`
2. Extract key metrics for paper tables
3. Select best figures for paper
4. Update paper text with new results
5. Address reviewer comments with specific citations

