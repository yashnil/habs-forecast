# Quick Start: Get Paper Results NOW

## Immediate Action Plan (Priority Order)

### ✅ STEP 1: Run Ablation Studies (2-3 hours)
**This finds optimal hyperparameters - do this FIRST**

```bash
cd /Users/yashnilmohanty/Desktop/habs-forecast

# Lambda ablation (find best physics weight)
python pinn/ablation_studies.py \
    --study lambda \
    --lambda-values 0.0 0.1 0.5 1.0 2.0 5.0 10.0 \
    --epochs 30 \
    --output-dir ablation_results/lambda

# Kappa ablation (find best diffusivity)
python pinn/ablation_studies.py \
    --study kappa \
    --kappa-values 1 5 10 25 50 100 500 \
    --epochs 30 \
    --output-dir ablation_results/kappa
```

**After this completes:**
1. Check `ablation_results/lambda/ablation_lambda.csv` - find row with minimum `val_rmse`
2. Check `ablation_results/kappa/ablation_kappa.csv` - find row with minimum `val_rmse`
3. Note the optimal λ and κ values

### ✅ STEP 2: Train Best PINN Model (4-6 hours)
**Train with optimal parameters from Step 1**

**Option A: Quick method - modify pinn_model.py directly**

1. Open `pinn/pinn_model.py`
2. Find line ~275 (where λ is computed)
3. Replace the auto-scaling with your optimal λ:

```python
# REPLACE THIS (around line 275):
λ_raw = sup_loss.detach() / (phys_loss_phys.detach() + 1e-8)
λ_phys = λ_raw.clamp(min=0.1, max=1000.0)

# WITH THIS (use your optimal λ from Step 1):
λ_phys = 1.0  # YOUR OPTIMAL VALUE FROM ABLATION
ramp = min(max((epoch - 20) / 5, 0.0), 1.0)
λ_phys = λ_phys * ramp
```

4. Find line ~198 (where κ is initialized)
5. Replace with your optimal κ:

```python
# REPLACE THIS (around line 198):
self.kappa = nn.Parameter(torch.tensor(25, dtype=torch.float32))

# WITH THIS (use your optimal κ from Step 1):
self.kappa = nn.Parameter(torch.tensor(25.0, dtype=torch.float32))  # YOUR OPTIMAL VALUE
```

6. Train:
```bash
python pinn/pinn_model.py --epochs 40 --batch 32
```

**Option B: Use the workflow script**
```bash
python pinn/run_paper_experiments.py --step train-best
# Then follow instructions in paper_results/training_instructions.json
```

### ✅ STEP 3: Run Imputation Sensitivity (30 minutes)
**Show robustness to data preprocessing**

```bash
python pinn/imputation_sensitivity.py \
    --data /Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc \
    --model ~/HAB_Models/convLSTM_best.pt \
    --methods original zero mean median forward_fill climatology interpolate \
    --output imputation_results.csv
```

**This generates:**
- `imputation_analysis/imputation_sensitivity.png` - Figure for paper
- `imputation_results.csv` - Table for paper

### ✅ STEP 4: Generate Diagnostics (1 hour)
**Get final performance metrics and figures**

```bash
# Run diagnostics on best model
python pinn/diagnostics.py \
    --freeze /Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc \
    --ckpt ~/HAB_Models/convLSTM_best.pt \
    --out Diagnostics_PINN_Best \
    --seq 6 \
    --lead 1 \
    --batch 8
```

**This generates:**
- All diagnostic figures (scatter plots, skill maps, etc.)
- Metrics CSV files
- Predicted fields NetCDF

### ✅ STEP 5: Train Uncertainty Model (Optional, 4-6 hours)
**If you have time, add uncertainty quantification**

```bash
# Single model with MC Dropout
python pinn/pinn_model_uncertainty.py --epochs 40 --batch 32

# Then generate uncertainty predictions
python pinn/inference_uncertainty.py \
    --ckpt ~/HAB_Models/pinn_uncertainty_best.pt \
    --data /Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc \
    --output predictions_uncertainty.nc \
    --mc-samples 20
```

## What to Report in Paper

### 1. Ablation Study Results

**In Results Section:**
> "We performed systematic ablation studies to optimize the physics loss weight (λ) and initial diffusivity (κ). The optimal configuration was λ = [YOUR_VALUE] and κ_init = [YOUR_VALUE] m²/s, achieving validation RMSE of [YOUR_VALUE]."

**Table to include:**
- Ablation study results (from CSV files)
- Optimal parameters

**Figure to include:**
- `ablation_results/lambda/ablation_lambda.png`
- `ablation_results/kappa/ablation_kappa.png`

### 2. Imputation Sensitivity

**In Results Section:**
> "To assess robustness to data preprocessing, we tested [N] imputation methods. The model showed [low/moderate/high] sensitivity with RMSE variation of [YOUR_VALUE]. The best-performing method was [METHOD]."

**Table to include:**
- Imputation sensitivity results (from CSV)

**Figure to include:**
- `imputation_analysis/imputation_sensitivity.png`

### 3. Best Model Performance

**In Results Section:**
> "The optimized PINN model achieved [YOUR_METRICS] on the test set, representing a [X]% improvement over the baseline."

**Table to include:**
- Final metrics from `Diagnostics_PINN_Best/metrics_global.csv`

**Figures to include:**
- All figures from `Diagnostics_PINN_Best/`

### 4. Uncertainty Quantification (if completed)

**In Results Section:**
> "We implemented Monte Carlo Dropout to quantify epistemic uncertainty. The model provides prediction intervals that complement TFT quantile outputs."

**Figure to include:**
- Uncertainty maps from inference

## Quick Checklist

- [ ] Run ablation studies (λ and κ)
- [ ] Extract optimal parameters
- [ ] Train best PINN model with optimal parameters
- [ ] Run imputation sensitivity analysis
- [ ] Generate diagnostics on best model
- [ ] (Optional) Train uncertainty model
- [ ] Extract metrics for paper tables
- [ ] Select figures for paper
- [ ] Update paper text with results

## Time Estimate

- Ablation studies: 2-3 hours ⏱️
- Train best model: 4-6 hours ⏱️
- Imputation sensitivity: 30 minutes ⏱️
- Diagnostics: 1 hour ⏱️
- **Total minimum: ~8-10 hours** ⏱️
- With uncertainty: +4-6 hours ⏱️

## Pro Tips

1. **Start with ablation** - It's fast and everything else depends on it
2. **Run ablation overnight** - Let it run while you sleep
3. **Check results immediately** - Make sure ablation completed successfully
4. **Save all outputs** - You'll need them for the paper
5. **Document parameters** - Write down optimal values as you find them

## If Something Goes Wrong

**Ablation takes too long?**
- Reduce epochs: `--epochs 20`
- Test fewer values: `--lambda-values 0.0 0.5 1.0 5.0`

**Model won't train?**
- Check CUDA memory: reduce `--batch 16`
- Check parameter ranges are valid

**Need results faster?**
- Skip uncertainty quantification (can add later)
- Use shorter ablation (20 epochs instead of 30)
- Focus on most important experiments first

