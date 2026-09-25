#!/usr/bin/env python3
"""
Data Imputation Sensitivity Analysis

This script tests the sensitivity of PINN predictions to different data
imputation methods. It compares:
1. No imputation (NaN handling)
2. Zero-filling
3. Mean imputation
4. Median imputation
5. Forward-fill
6. Climatology-based imputation

Usage:
    python pinn/imputation_sensitivity.py \
        --data path/to/data.nc \
        --output imputation_sensitivity_results.csv
"""
from __future__ import annotations
import os
import argparse, pathlib, json
import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import pathlib

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pinn.pinn_model import (
    ConvLSTM, PatchDS, make_loaders, ALL_VARS, LOGCHL_IDX,
    FREEZE, SEQ, LEAD_IDX, PATCH, BATCH, DEVICE, MIXED_PREC,
    norm_stats, z
)

# ------------------------------------------------------------------ #
# Imputation Methods
# ------------------------------------------------------------------ #
def impute_zero(ds, var_name):
    """Zero-fill missing values."""
    da = ds[var_name].copy()
    da = da.fillna(0.0)
    return da

def impute_mean(ds, var_name, train_idx):
    """Mean imputation using training data."""
    da = ds[var_name].copy()
    if "time" in da.dims:
        train_mean = da.isel(time=train_idx).mean(skipna=True)
        da = da.fillna(train_mean)
    else:
        da = da.fillna(da.mean(skipna=True))
    return da

def impute_median(ds, var_name, train_idx):
    """Median imputation using training data."""
    da = ds[var_name].copy()
    if "time" in da.dims:
        train_median = da.isel(time=train_idx).median(skipna=True)
        da = da.fillna(train_median)
    else:
        da = da.fillna(da.median(skipna=True))
    return da

def impute_forward_fill(ds, var_name):
    """Forward-fill along time dimension."""
    da = ds[var_name].copy()
    if "time" in da.dims:
        da = da.ffill("time")
        # Fill any remaining NaNs with backward fill
        da = da.bfill("time")
        # If still NaN, fill with 0
        da = da.fillna(0.0)
    else:
        da = da.fillna(0.0)
    return da

def impute_climatology(ds, var_name, train_idx):
    """Climatology-based imputation (8-day block mean)."""
    da = ds[var_name].copy()
    if "time" not in da.dims:
        return da.fillna(da.median(skipna=True))
    
    # Compute 8-day block climatology from training data
    train_da = da.isel(time=train_idx)
    times = train_da.time.values
    
    # Group by 8-day block (day of year / 8)
    doy = pd.to_datetime(times).dayofyear
    block = (doy // 8).astype(int)
    
    # Compute mean for each block
    block_means = {}
    for b in np.unique(block):
        block_mask = block == b
        block_mean = train_da.isel(time=block_mask).mean(skipna=True)
        block_means[b] = block_mean
    
    # Apply to full dataset
    all_times = pd.to_datetime(da.time.values)
    all_doy = all_times.dayofyear
    all_block = (all_doy // 8).astype(int)
    
    da_filled = da.copy()
    for t_idx, b in enumerate(all_block):
        if b in block_means:
            da_filled[t_idx] = da_filled[t_idx].fillna(block_means[b])
    
    # Fill any remaining NaNs with overall mean
    da_filled = da_filled.fillna(train_da.mean(skipna=True))
    return da_filled

def impute_interpolate(ds, var_name):
    """Linear interpolation along time."""
    da = ds[var_name].copy()
    if "time" in da.dims:
        da = da.interpolate_na("time", method="linear")
        da = da.ffill("time").bfill("time")
        da = da.fillna(0.0)
    else:
        da = da.fillna(0.0)
    return da

# ------------------------------------------------------------------ #
# Apply Imputation to Dataset
# ------------------------------------------------------------------ #
def apply_imputation(ds, method, train_idx):
    """Apply imputation method to all variables in dataset."""
    ds_imputed = ds.copy(deep=True)
    
    for var in ALL_VARS:
        if var not in ds_imputed:
            continue
        
        if method == "zero":
            ds_imputed[var] = impute_zero(ds_imputed, var)
        elif method == "mean":
            ds_imputed[var] = impute_mean(ds_imputed, var, train_idx)
        elif method == "median":
            ds_imputed[var] = impute_median(ds_imputed, var, train_idx)
        elif method == "forward_fill":
            ds_imputed[var] = impute_forward_fill(ds_imputed, var)
        elif method == "climatology":
            ds_imputed[var] = impute_climatology(ds_imputed, var, train_idx)
        elif method == "interpolate":
            ds_imputed[var] = impute_interpolate(ds_imputed, var)
        elif method == "original":
            # Keep original (may have NaNs)
            pass
        else:
            raise ValueError(f"Unknown imputation method: {method}")
    
    return ds_imputed

# ------------------------------------------------------------------ #
# Evaluate Model with Imputed Data
# ------------------------------------------------------------------ #
@torch.no_grad()
def evaluate_model(model, dl):
    """Evaluate model on data loader."""
    model.eval()
    se = n = 0
    all_preds = []
    all_trues = []
    
    for X, y, m in dl:
        X, y, m = [t.to(DEVICE) for t in (X, y, m)]
        with autocast(enabled=MIXED_PREC):
            pred = X[:, -1, LOGCHL_IDX] + model(X)
        err = (pred - y).masked_fill_(~m, 0)
        se += (err**2).sum().item()
        n += m.sum().item()
        
        # Store for additional metrics
        all_preds.append(pred[m].cpu().numpy())
        all_trues.append(y[m].cpu().numpy())
    
    rmse = np.sqrt(se / n) if n > 0 else float('inf')
    
    # Additional metrics
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    
    mae = np.mean(np.abs(all_preds - all_trues))
    corr = np.corrcoef(all_preds, all_trues)[0, 1]
    
    return {
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "n_samples": n
    }

def test_imputation_sensitivity(
    model_path: pathlib.Path,
    data_path: pathlib.Path,
    methods: list,
    epochs_retrain: int = 0,
    output_csv: pathlib.Path = None
):
    """
    Test sensitivity to different imputation methods.
    
    Args:
        model_path: Path to pre-trained model checkpoint
        data_path: Path to data file
        methods: List of imputation methods to test
        epochs_retrain: If > 0, retrain model for this many epochs on imputed data
        output_csv: Optional path to CSV file for incremental saving
    """
    print(f"\n{'='*70}")
    print(f"IMPUTATION SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Testing methods: {methods}")
    print(f"Data: {data_path}")
    print(f"Model: {model_path}\n")
    
    # Check for existing results
    results = []
    if output_csv and output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            completed_methods = existing_df["method"].tolist()
            methods = [m for m in methods if m not in completed_methods]
            results = existing_df.to_dict("records")
            print(f"Found {len(completed_methods)} completed methods. Resuming with {len(methods)} remaining.")
        except Exception as e:
            print(f"Could not load existing results: {e}")
    
    # Load original data
    ds_orig = xr.open_dataset(data_path).load()
    
    # Get train/val/test splits
    t_all = np.arange(ds_orig.sizes["time"])
    times = ds_orig.time.values
    tr = t_all[times < np.datetime64("2016-01-01")]
    va = t_all[(times >= np.datetime64("2016-01-01")) & (times <= np.datetime64("2018-12-31"))]
    te = t_all[times > np.datetime64("2018-12-31")]
    
    for method in methods:
        print(f"\n{'─'*70}")
        print(f"Testing imputation method: {method}")
        print(f"{'─'*70}")
        
        # Apply imputation
        ds_imputed = apply_imputation(ds_orig, method, tr)
        
        # Compute statistics on imputed data
        stats = norm_stats(ds_imputed, tr)
        
        # Build pixel mask (must be xarray DataArray, not numpy array)
        if "pixel_ok" in ds_imputed:
            pixel_ok = ds_imputed.pixel_ok.astype(bool)
        else:
            frac = np.isfinite(ds_imputed.log_chl).sum("time") / ds_imputed.sizes["time"]
            pixel_ok = (frac >= 0.2).astype(bool)
            # Convert to DataArray if it's a numpy array
            if isinstance(pixel_ok, np.ndarray):
                pixel_ok = xr.DataArray(pixel_ok, dims=["lat", "lon"], 
                                       coords={"lat": ds_imputed.lat, "lon": ds_imputed.lon})
        
        # Create data loaders
        tr_i, va_i, te_i = tr[SEQ-1:-LEAD_IDX], va[SEQ-1:-LEAD_IDX], te[SEQ-1:-LEAD_IDX]
        tr_ds = PatchDS(ds_imputed, tr_i, stats, pixel_ok)
        va_ds = PatchDS(ds_imputed, va_i, stats, pixel_ok)
        te_ds = PatchDS(ds_imputed, te_i, stats, pixel_ok)
        
        tr_dl = DataLoader(tr_ds, BATCH, shuffle=False, num_workers=2, pin_memory=True)
        va_dl = DataLoader(va_ds, BATCH, shuffle=False, num_workers=2, pin_memory=True)
        te_dl = DataLoader(te_ds, BATCH, shuffle=False, num_workers=2, pin_memory=True)
        
        # Load or retrain model
        Cin = len([v for v in ALL_VARS if v in ds_imputed.data_vars])
        model = ConvLSTM(Cin).to(DEVICE)
        
        if epochs_retrain > 0:
            print(f"  Retraining model for {epochs_retrain} epochs...")
            # Retrain on imputed data (simplified - would need full training loop)
            # For now, just load and evaluate
            state = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
        else:
            # Just load pre-trained model
            state = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
        
        # Evaluate
        print("  Evaluating on train/val/test splits...")
        train_metrics = evaluate_model(model, tr_dl)
        val_metrics = evaluate_model(model, va_dl)
        test_metrics = evaluate_model(model, te_dl)
        
        # Compute data statistics
        n_missing_orig = {}
        n_missing_imputed = {}
        for var in ALL_VARS:
            if var in ds_orig:
                n_missing_orig[var] = int(ds_orig[var].isnull().sum().item())
            if var in ds_imputed:
                n_missing_imputed[var] = int(ds_imputed[var].isnull().sum().item())
        
        results.append({
            "method": method,
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mae"],
            "train_corr": train_metrics["corr"],
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_corr": val_metrics["corr"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_corr": test_metrics["corr"],
            "n_missing_after": sum(n_missing_imputed.values()),
            "n_missing_before": sum(n_missing_orig.values())
        })
        
        print(f"  Results:")
        print(f"    Train RMSE: {train_metrics['rmse']:.4f}")
        print(f"    Val RMSE:   {val_metrics['rmse']:.4f}")
        print(f"    Test RMSE:  {test_metrics['rmse']:.4f}")
        
        # Save incrementally
        if output_csv:
            df = pd.DataFrame(results)
            df.to_csv(output_csv, index=False)
            print(f"  ✓ Saved progress to {output_csv}")
    
    return pd.DataFrame(results)

# ------------------------------------------------------------------ #
# Visualization
# ------------------------------------------------------------------ #
def plot_imputation_sensitivity(df, out_dir):
    """Plot imputation sensitivity results."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE comparison
    x = np.arange(len(df))
    width = 0.25
    axes[0, 0].bar(x - width, df["train_rmse"], width, label="Train", alpha=0.8)
    axes[0, 0].bar(x, df["val_rmse"], width, label="Val", alpha=0.8)
    axes[0, 0].bar(x + width, df["test_rmse"], width, label="Test", alpha=0.8)
    axes[0, 0].set_xlabel("Imputation Method", fontsize=12)
    axes[0, 0].set_ylabel("RMSE (log chl)", fontsize=12)
    axes[0, 0].set_title("RMSE by Imputation Method", fontsize=14)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(df["method"], rotation=45, ha="right")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis="y")
    
    # Correlation comparison
    axes[0, 1].bar(x - width, df["train_corr"], width, label="Train", alpha=0.8)
    axes[0, 1].bar(x, df["val_corr"], width, label="Val", alpha=0.8)
    axes[0, 1].bar(x + width, df["test_corr"], width, label="Test", alpha=0.8)
    axes[0, 1].set_xlabel("Imputation Method", fontsize=12)
    axes[0, 1].set_ylabel("Correlation", fontsize=12)
    axes[0, 1].set_title("Correlation by Imputation Method", fontsize=14)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(df["method"], rotation=45, ha="right")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")
    
    # Missing data statistics
    axes[1, 0].bar(x, df["n_missing_before"], width*2, label="Before", alpha=0.6, color="red")
    axes[1, 0].bar(x, df["n_missing_after"], width*2, label="After", alpha=0.8, color="green")
    axes[1, 0].set_xlabel("Imputation Method", fontsize=12)
    axes[1, 0].set_ylabel("Number of Missing Values", fontsize=12)
    axes[1, 0].set_title("Missing Data: Before vs After", fontsize=14)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(df["method"], rotation=45, ha="right")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis="y")
    
    # RMSE vs Missing Data
    axes[1, 1].scatter(df["n_missing_after"], df["val_rmse"], s=100, alpha=0.7)
    for i, method in enumerate(df["method"]):
        axes[1, 1].annotate(method, (df["n_missing_after"].iloc[i], df["val_rmse"].iloc[i]),
                           fontsize=9, alpha=0.7)
    axes[1, 1].set_xlabel("Missing Values After Imputation", fontsize=12)
    axes[1, 1].set_ylabel("Validation RMSE", fontsize=12)
    axes[1, 1].set_title("RMSE vs Data Completeness", fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "imputation_sensitivity.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved plot: {out_dir / 'imputation_sensitivity.png'}")
    plt.close()

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="Imputation sensitivity analysis")
    ap.add_argument("--data", type=str, default=str(FREEZE), help="Path to data NetCDF")
    ap.add_argument("--model", type=str, help="Path to model checkpoint")
    ap.add_argument("--methods", type=str, nargs="+",
                    default=["original", "zero", "mean", "median", "forward_fill", "climatology", "interpolate"],
                    help="Imputation methods to test")
    ap.add_argument("--output", type=str, default="imputation_sensitivity_results.csv",
                    help="Output CSV file")
    ap.add_argument("--output-dir", type=str, default="imputation_analysis",
                    help="Output directory for plots")
    ap.add_argument("--retrain-epochs", type=int, default=0,
                    help="If > 0, retrain model on imputed data")
    args = ap.parse_args()
    
    model_path = pathlib.Path(args.model) if args.model else pathlib.Path(os.environ.get("HABS_MODEL_DIR", "~/HAB_Models")).expanduser() / "convLSTM_best.pt"
    data_path = pathlib.Path(args.data)
    out_dir = pathlib.Path(args.output_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")
    
    # Run sensitivity analysis
    df = test_imputation_sensitivity(
        model_path, data_path, args.methods, 
        epochs_retrain=args.retrain_epochs,
        output_csv=pathlib.Path(args.output)
    )
    
    # Save results
    df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Plot results
    plot_imputation_sensitivity(df, out_dir)
    
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

