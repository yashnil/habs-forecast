# PINN Ablation Studies and Imputation Sensitivity Analysis

This document describes the ablation studies and sensitivity analyses for the PINN model.

## Overview

Two main analysis scripts are provided:

1. **`ablation_studies.py`** - Systematic ablation on physics loss weight (λ) and diffusivity (κ)
2. **`imputation_sensitivity.py`** - Sensitivity analysis for different data imputation methods

## Ablation Studies

### Physics Loss Weight (λ) Ablation

Tests the sensitivity of model performance to the physics loss weight parameter.

**Current Implementation**: λ is auto-scaled based on the ratio of supervised loss to physics loss, with a ramp-up schedule.

**Ablation Tests**: Fixed λ values to understand the trade-off between data fit and physics constraints.

```bash
# Test different λ values
python pinn/ablation_studies.py \
    --study lambda \
    --lambda-values 0.0 0.1 0.5 1.0 2.0 5.0 10.0 \
    --epochs 30
```

**Outputs**:
- `ablation_lambda.csv` - Results table
- `ablation_lambda.png` - Visualization plots

**Key Metrics**:
- Validation/test RMSE vs λ
- Final learned κ vs λ
- Convergence speed (best epoch)

### Diffusivity (κ) Ablation

Tests the sensitivity to initial diffusivity parameter and how it evolves during training.

**Current Implementation**: κ is initialized to 25 m²/s and is trainable (clamped to [0.01, 1000]).

**Ablation Tests**: Different initial κ values to understand optimal initialization.

```bash
# Test different initial κ values
python pinn/ablation_studies.py \
    --study kappa \
    --kappa-values 1 5 10 25 50 100 500 \
    --epochs 30
```

**Outputs**:
- `ablation_kappa.csv` - Results table
- `ablation_kappa.png` - Visualization plots

**Key Metrics**:
- RMSE vs initial κ
- Final κ vs initial κ (how much it changes)
- RMSE improvement vs baseline

### Combined Ablation

Tests all combinations of λ and κ to find optimal hyperparameters.

```bash
# Combined study
python pinn/ablation_studies.py \
    --study both \
    --lambda-values 0.1 1.0 10.0 \
    --kappa-values 10 25 100 \
    --epochs 25
```

**Outputs**:
- `ablation_combined.csv` - Results table
- `ablation_combined.png` - Heatmap visualization

## Imputation Sensitivity Analysis

Tests how different data imputation methods affect model performance.

### Available Methods

1. **original** - No imputation (may have NaNs)
2. **zero** - Zero-fill missing values
3. **mean** - Mean imputation (training data)
4. **median** - Median imputation (training data)
5. **forward_fill** - Forward-fill along time
6. **climatology** - 8-day block climatology
7. **interpolate** - Linear interpolation

### Usage

```bash
# Test all methods
python pinn/imputation_sensitivity.py \
    --data /path/to/data.nc \
    --model ~/HAB_Models/convLSTM_best.pt \
    --methods original zero mean median forward_fill climatology interpolate \
    --output imputation_results.csv
```

### Outputs

- CSV file with metrics for each method
- Visualization plots showing:
  - RMSE comparison across methods
  - Correlation comparison
  - Missing data statistics
  - RMSE vs data completeness

## Interpretation Guidelines

### λ (Physics Loss Weight)

- **λ = 0**: Pure data-driven model (no physics)
- **Small λ (0.1-1.0)**: Physics as weak regularization
- **Medium λ (1.0-5.0)**: Balanced physics and data
- **Large λ (>5.0)**: Strong physics constraints (may over-constrain)

**Expected Behavior**:
- Very low λ: Similar to vanilla ConvLSTM
- Optimal λ: Best validation/test performance
- Very high λ: May hurt performance if physics is too restrictive

### κ (Diffusivity)

- **Small κ (1-10)**: Weak diffusion (advection-dominated)
- **Medium κ (10-100)**: Balanced transport
- **Large κ (100-1000)**: Strong diffusion (smoothing)

**Expected Behavior**:
- κ too small: May not capture mixing processes
- Optimal κ: Learned value that balances advection and diffusion
- κ too large: Over-smoothing, loss of spatial structure

### Imputation Methods

**Best Practices**:
- **Climatology**: Best for seasonal variables (preserves temporal structure)
- **Interpolation**: Good for short gaps
- **Mean/Median**: Simple but may introduce bias
- **Zero-fill**: Worst option (introduces artificial values)

**Sensitivity Indicators**:
- Large RMSE differences → High sensitivity (need careful imputation)
- Small RMSE differences → Low sensitivity (robust to imputation)

## Example Workflow

### 1. Run Ablation Studies

```bash
# Lambda ablation
python pinn/ablation_studies.py --study lambda --lambda-values 0.0 0.1 0.5 1.0 2.0 5.0 10.0

# Kappa ablation
python pinn/ablation_studies.py --study kappa --kappa-values 1 5 10 25 50 100 500

# Combined (smaller grid for speed)
python pinn/ablation_studies.py --study both \
    --lambda-values 0.1 1.0 10.0 \
    --kappa-values 10 25 100
```

### 2. Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df_lambda = pd.read_csv("ablation_results/ablation_lambda.csv")
df_kappa = pd.read_csv("ablation_results/ablation_kappa.csv")

# Find optimal parameters
best_lambda = df_lambda.loc[df_lambda["val_rmse"].idxmin()]
best_kappa = df_kappa.loc[df_kappa["val_rmse"].idxmin()]

print(f"Optimal λ: {best_lambda['lambda']:.2f}")
print(f"Optimal κ_init: {best_kappa['kappa_init']:.1f}")
```

### 3. Test Imputation Sensitivity

```bash
python pinn/imputation_sensitivity.py \
    --data /path/to/data.nc \
    --model ~/HAB_Models/convLSTM_best.pt \
    --output imputation_results.csv
```

### 4. Use Optimal Parameters

Update `pinn_model.py` with optimal values:

```python
# In training script
LAMBDA_PHYS = 1.0  # From ablation study
KAPPA_INIT = 25.0  # From ablation study
```

## Notes

- Ablation studies use shorter training (30 epochs) for speed
- For final model, retrain with optimal parameters for full epochs
- Imputation sensitivity can be run with or without retraining
- Results may vary with different random seeds (consider ensemble)

## References

- Physics loss weight: Balancing data fit vs. physics constraints
- Diffusivity: Physical parameter for advection-diffusion equation
- Imputation: Critical for handling missing satellite data

