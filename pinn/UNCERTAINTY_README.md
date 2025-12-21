# PINN Uncertainty Quantification

This document describes the uncertainty quantification methods added to the Physics-Informed Neural Network (PINN) model to complement TFT quantile predictions.

## Overview

The enhanced PINN model (`pinn_model_uncertainty.py`) supports three complementary uncertainty quantification methods:

1. **Monte Carlo Dropout** - Epistemic uncertainty (model uncertainty)
2. **Ensemble Methods** - Robust predictions via multiple models
3. **Quantile Regression** - Predictive intervals (similar to TFT)

## Methods

### 1. Monte Carlo Dropout

**What it does**: Estimates epistemic uncertainty by sampling multiple predictions with dropout enabled at inference time.

**How it works**:
- Dropout layers remain active during inference
- Multiple forward passes (default: 20) generate a distribution of predictions
- Mean and standard deviation capture model uncertainty

**Usage**:
```python
# Training (dropout is automatically enabled)
python pinn/pinn_model_uncertainty.py --epochs 40 --batch 16

# Inference with MC dropout
python pinn/inference_uncertainty.py \
    --ckpt ~/HAB_Models/pinn_uncertainty_best.pt \
    --data path/to/data.nc \
    --output predictions.nc \
    --mc-samples 20
```

### 2. Ensemble Methods

**What it does**: Trains multiple models with different random seeds and aggregates their predictions.

**How it works**:
- Trains N independent models (default: 5)
- Each model uses a different random seed
- Predictions are aggregated (mean ± std) for robust estimates

**Usage**:
```bash
# Train ensemble of 5 models
python pinn/pinn_model_uncertainty.py \
    --epochs 40 \
    --batch 16 \
    --ensemble 5

# Models will be saved as:
# ~/HAB_Models/pinn_ensemble_1_best.pt
# ~/HAB_Models/pinn_ensemble_2_best.pt
# ... etc
```

**Inference**:
```python
# Load and use ensemble
from pinn.inference_uncertainty import load_ensemble, ensemble_predict

models = load_ensemble(pathlib.Path("~/HAB_Models"), Cin=30)
mean, std, predictions = ensemble_predict(models, X_batch)
```

### 3. Quantile Regression

**What it does**: Directly predicts multiple quantiles (e.g., 0.1, 0.5, 0.9) for predictive intervals, similar to TFT.

**How it works**:
- Model outputs multiple quantile predictions simultaneously
- Uses quantile loss (pinball loss) during training
- Provides uncertainty bounds directly

**Usage**:
```bash
# Train with quantile regression (predict 10th, 50th, 90th percentiles)
python pinn/pinn_model_uncertainty.py \
    --epochs 40 \
    --batch 16 \
    --quantiles 0.1 0.5 0.9
```

**Output**: Model outputs 3 channels corresponding to the requested quantiles.

## Combining Methods

You can combine methods for more robust uncertainty estimates:

### Ensemble + MC Dropout
```python
# Train ensemble
python pinn/pinn_model_uncertainty.py --ensemble 5

# Inference with both ensemble and MC dropout
# (combines ensemble variance with MC dropout variance)
```

### Quantiles + MC Dropout
```python
# Train with quantiles
python pinn/pinn_model_uncertainty.py --quantiles 0.1 0.5 0.9

# Inference adds MC dropout uncertainty on top of quantile spread
```

## Comparison with TFT

| Method | TFT | PINN (Enhanced) |
|--------|-----|-----------------|
| Quantile Regression | ✅ Built-in | ✅ Added |
| Epistemic Uncertainty | ❌ | ✅ MC Dropout |
| Ensemble Methods | ❌ | ✅ Added |
| Physics Constraints | ❌ | ✅ Built-in |

## Output Format

### Single Model with MC Dropout
```python
mean, std = mc_dropout_predict(model, X, n_samples=20)
# mean: (B, H, W) - mean prediction
# std: (B, H, W) - epistemic uncertainty
```

### Ensemble
```python
mean, std, predictions = ensemble_predict(models, X)
# mean: (B, H, W) - ensemble mean
# std: (B, H, W) - ensemble uncertainty
# predictions: List of (B, H, W) individual model predictions
```

### Quantile Regression
```python
quantiles, mean, std = model(X, return_uncertainty=True)
# quantiles: (B, n_quantiles, H, W) - quantile predictions
# mean: (B, H, W) - median/mean
# std: (B, H, W) - uncertainty from quantile spread
```

## NetCDF Output

The inference script saves predictions to NetCDF with the following structure:

```python
xr.Dataset({
    "pred_mean": (["time", "lat", "lon"], ...),      # Mean prediction
    "pred_std": (["time", "lat", "lon"], ...),       # Uncertainty (std)
    "pred_quantiles": (["time", "quantile", "lat", "lon"], ...),  # If quantiles used
    "valid_mask": (["time", "lat", "lon"], ...)      # Valid pixels
})
```

## Configuration

Key parameters in `pinn_model_uncertainty.py`:

```python
MC_DROPOUT_RATE = 0.1      # Dropout rate for uncertainty
MC_SAMPLES = 20            # Number of MC samples
ENSEMBLE_SIZE = 5          # Default ensemble size
QUANTILES = None           # Quantiles to predict (e.g., [0.1, 0.5, 0.9])
```

## Example Workflow

### 1. Train Ensemble with Quantiles
```bash
python pinn/pinn_model_uncertainty.py \
    --epochs 40 \
    --batch 16 \
    --ensemble 5 \
    --quantiles 0.1 0.5 0.9
```

### 2. Generate Predictions with Uncertainty
```bash
python pinn/inference_uncertainty.py \
    --ensemble-dir ~/HAB_Models \
    --data /path/to/test_data.nc \
    --output predictions_uncertainty.nc \
    --mc-samples 20
```

### 3. Visualize Uncertainty
```python
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("predictions_uncertainty.nc")

# Plot mean prediction
ds.pred_mean.isel(time=0).plot()

# Plot uncertainty
ds.pred_std.isel(time=0).plot()

# Plot quantile intervals
ds.pred_quantiles.isel(time=0, quantile=0).plot(label="10th percentile")
ds.pred_quantiles.isel(time=0, quantile=2).plot(label="90th percentile")
```

## Performance Considerations

- **MC Dropout**: Adds ~20x inference time (20 samples)
- **Ensemble**: Adds ~Nx training time (N models) but same inference time as single model
- **Quantiles**: Minimal overhead (single forward pass, multiple outputs)

## References

- **Monte Carlo Dropout**: Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
- **Quantile Regression**: Koenker & Bassett (1978) "Regression Quantiles"
- **Ensemble Methods**: Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty"

