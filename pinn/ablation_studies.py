#!/usr/bin/env python3
"""
PINN Ablation Studies: Physics Loss Weight (λ) and Diffusivity (κ)

This script performs systematic ablation studies to understand:
1. Sensitivity to physics loss weight (λ)
2. Sensitivity to diffusivity parameter (κ)
3. Optimal hyperparameter combinations

Usage:
    # Ablation on λ (physics loss weight)
    python pinn/ablation_studies.py --study lambda --lambda-values 0.0 0.1 0.5 1.0 2.0 5.0 10.0

    # Ablation on κ (diffusivity)
    python pinn/ablation_studies.py --study kappa --kappa-values 1 5 10 25 50 100 500

    # Combined study
    python pinn/ablation_studies.py --study both --lambda-values 0.1 1.0 10.0 --kappa-values 10 25 100
"""
from __future__ import annotations
import argparse, json, pathlib, itertools, sys
import numpy as np
import pandas as pd
import xarray as xr
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import base model components
from pinn.pinn_model import (
    ConvLSTM, PxLSTM, PatchDS, make_loaders, physics_residual,
    ALL_VARS, LOGCHL_IDX, FREEZE, SEQ, LEAD_IDX, PATCH, BATCH, EPOCHS,
    DEVICE, MIXED_PREC, SCALER, STRATIFY, WEIGHT_EXP, HUBER_DELTA,
    norm_stats, z
)

# ------------------------------------------------------------------ #
# Modified training function with explicit λ and κ control
# ------------------------------------------------------------------ #
def train_with_params(
    tr_dl, va_dl, te_dl, qthr, stats,
    lambda_phys: float = None,  # If None, use auto-scaling
    kappa_init: float = 25.0,
    kappa_trainable: bool = True,
    epochs: int = 30,  # Shorter for ablation
    verbose: bool = False
):
    """
    Train PINN with specified λ and κ parameters.
    
    Args:
        lambda_phys: Fixed physics loss weight (if None, uses auto-scaling)
        kappa_init: Initial diffusivity value
        kappa_trainable: Whether κ is trainable or fixed
        epochs: Number of training epochs
        verbose: Print training progress
    """
    Cin = len([v for v in ALL_VARS if v in xr.open_dataset(FREEZE).data_vars])
    
    # Create model with specified κ
    class ConvLSTM_Ablation(ConvLSTM):
        def __init__(self, Cin, kappa_init=25.0, kappa_trainable=True):
            super().__init__(Cin)
            if not kappa_trainable:
                # Make kappa a buffer (non-trainable)
                self.register_buffer('kappa', torch.tensor(kappa_init, dtype=torch.float32))
            else:
                self.kappa = nn.Parameter(torch.tensor(kappa_init, dtype=torch.float32))
    
    net = ConvLSTM_Ablation(Cin, kappa_init=kappa_init, kappa_trainable=kappa_trainable).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = ReduceLROnPlateau(opt, 'min', patience=3, factor=0.5)
    
    best = 1e9
    best_epoch = 0
    val_rmses = []
    
    for ep in range(1, epochs + 1):
        net.train()
        for X, y, m in tr_dl:
            X, y, m = [t.to(DEVICE) for t in (X, y, m)]
            
            with autocast(enabled=MIXED_PREC):
                last_log = X[:, -1, LOGCHL_IDX]
                delta = net(X)
                tgt = y - last_log
                sup_err = (delta - tgt).masked_fill_(~m, 0)
                
                # Class-weighted Huber loss
                chl_lin = torch.exp(y)
                w = torch.where(chl_lin < qthr[0], 1.0,
                        torch.where(chl_lin < qthr[1], 1.5,
                            torch.where(chl_lin < qthr[2], 2.5, 4.0)))
                w = w * m
                abs_err = sup_err.abs()
                quadratic = torch.minimum(abs_err, torch.tensor(HUBER_DELTA, device=abs_err.device))
                linear = abs_err - quadratic
                huber = 0.5 * quadratic**2 + HUBER_DELTA * linear
                sup_loss = (w * huber).sum() / w.sum()
                
                # Physics loss
                use_phys = (ep > 10)  # Start physics earlier for ablation
                phys_loss = torch.tensor(0.0, device=DEVICE)
                if use_phys:
                    logC_ = last_log.unsqueeze(1)
                    logCp_ = (last_log + delta).unsqueeze(1)
                    u_ = X[:, -1, ALL_VARS.index("uo")].unsqueeze(1)
                    v_ = X[:, -1, ALL_VARS.index("vo")].unsqueeze(1)
                    res = physics_residual(logCp_, logC_, u_, v_, net.kappa)
                    sq = (res[:, 0][m]).pow(2)
                    phys_loss = sq.sum() / (m.sum().clamp(min=1))
                    
                    mu, sd = stats["log_chl"]
                    phys_loss_phys = phys_loss * (sd**2)
                    
                    # Use explicit λ or auto-scaling
                    if lambda_phys is not None:
                        λ_phys = lambda_phys
                        # Ramp up over first 5 physics epochs
                        ramp = min(max((ep - 10) / 5, 0.0), 1.0)
                        λ_phys = λ_phys * ramp
                    else:
                        # Auto-scaling (original method)
                        λ_raw = sup_loss.detach() / (phys_loss_phys.detach() + 1e-8)
                        λ_phys = λ_raw.clamp(min=0.1, max=1000.0)
                        ramp = min(max((ep - 10) / 5, 0.0), 1.0)
                        λ_phys = λ_phys * ramp
                else:
                    λ_phys = 0.0
                    phys_loss_phys = torch.tensor(0.0, device=DEVICE)
                
                loss = sup_loss + λ_phys * phys_loss_phys
            
            SCALER.scale(loss).backward()
            SCALER.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            SCALER.step(opt)
            SCALER.update()
            opt.zero_grad()
            
            if kappa_trainable:
                with torch.no_grad():
                    net.kappa.clamp_(1e-2, 1e3)
        
        # Validation
        @torch.no_grad()
        def rmse(dl, net):
            net.eval()
            se = n = 0
            for X, y, m in dl:
                X, y, m = [t.to(DEVICE) for t in (X, y, m)]
                with autocast(enabled=MIXED_PREC):
                    p = X[:, -1, LOGCHL_IDX] + net(X)
                err = (p - y).masked_fill_(~m, 0)
                se += (err**2).sum().item()
                n += m.sum().item()
            return np.sqrt(se / n) if n > 0 else float('inf')
        
        val_rm = rmse(va_dl, net)
        val_rmses.append(val_rm)
        sched.step(val_rm)
        
        if verbose and ep % 5 == 0:
            kappa_val = net.kappa.item() if isinstance(net.kappa, nn.Parameter) else float(net.kappa)
            print(f"E{ep:02d}  val RMSE={val_rm:.4f}  κ={kappa_val:.2f}  λ={λ_phys:.4f}")
        
        if val_rm < best - 1e-3:
            best = val_rm
            best_epoch = ep
    
    # Final evaluation
    @torch.no_grad()
    def rmse(dl, net):
        net.eval()
        se = n = 0
        for X, y, m in dl:
            X, y, m = [t.to(DEVICE) for t in (X, y, m)]
            with autocast(enabled=MIXED_PREC):
                p = X[:, -1, LOGCHL_IDX] + net(X)
            err = (p - y).masked_fill_(~m, 0)
            se += (err**2).sum().item()
            n += m.sum().item()
        return np.sqrt(se / n) if n > 0 else float('inf')
    
    final_kappa = net.kappa.item() if isinstance(net.kappa, nn.Parameter) else float(net.kappa)
    metrics = {
        "train": rmse(tr_dl, net),
        "val": rmse(va_dl, net),
        "test": rmse(te_dl, net),
        "best_epoch": best_epoch,
        "final_kappa": final_kappa
    }
    
    return metrics, net

# ------------------------------------------------------------------ #
# Ablation Studies
# ------------------------------------------------------------------ #
def ablation_lambda(lambda_values, kappa_init=25.0, epochs=30):
    """Ablation study on physics loss weight λ."""
    print(f"\n{'='*70}")
    print(f"LAMBDA (λ) ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Testing λ values: {lambda_values}")
    print(f"Fixed κ = {kappa_init}")
    print(f"Epochs per run: {epochs}\n")
    
    tr_dl, va_dl, te_dl, qthr, stats = make_loaders()
    results = []
    
    # Check for existing results to resume
    out_dir = pathlib.Path("paper_results") / "ablation" / "lambda"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "ablation_lambda.csv"
    
    if csv_file.exists():
        existing_df = pd.read_csv(csv_file)
        completed = existing_df["lambda"].tolist()
        remaining = [λ for λ in lambda_values if λ not in completed]
        print(f"Found {len(completed)} completed runs. Resuming with {len(remaining)} remaining values.")
        results = existing_df.to_dict("records")
        lambda_values = remaining
    
    for i, λ in enumerate(lambda_values):
        print(f"\n[{i+1}/{len(lambda_values)}] Training with λ = {λ}")
        try:
            metrics, net = train_with_params(
                tr_dl, va_dl, te_dl, qthr, stats,
                lambda_phys=λ,
                kappa_init=kappa_init,
                kappa_trainable=True,
                epochs=epochs,
                verbose=True
            )
            
            result = {
                "lambda": λ,
                "kappa_init": kappa_init,
                "final_kappa": metrics["final_kappa"],
                "train_rmse": metrics["train"],
                "val_rmse": metrics["val"],
                "test_rmse": metrics["test"],
                "best_epoch": metrics["best_epoch"]
            }
            results.append(result)
            
            # Save incrementally after each run
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False)
            
            print(f"  Results: train={metrics['train']:.4f}, val={metrics['val']:.4f}, test={metrics['test']:.4f}")
            print(f"  Final κ = {metrics['final_kappa']:.2f}")
            print(f"  ✓ Saved progress to {csv_file}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  Continuing with next value...")
            continue
    
    return pd.DataFrame(results)

def ablation_kappa(kappa_values, lambda_phys=None, epochs=30):
    """Ablation study on diffusivity κ."""
    print(f"\n{'='*70}")
    print(f"KAPPA (κ) ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Testing κ values: {kappa_values}")
    if lambda_phys is not None:
        print(f"Fixed λ = {lambda_phys}")
    else:
        print(f"Auto-scaling λ")
    print(f"Epochs per run: {epochs}\n")
    
    # Check if data file exists
    if not FREEZE.exists():
        raise FileNotFoundError(
            f"Data file not found: {FREEZE}\n"
            f"Please update FREEZE path in pinn_model.py or provide correct path."
        )
    
    tr_dl, va_dl, te_dl, qthr, stats = make_loaders()
    results = []
    
    # Check for existing results to resume
    out_dir = pathlib.Path("paper_results") / "ablation" / "kappa"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_file = out_dir / "ablation_kappa.csv"
    
    if csv_file.exists():
        try:
            existing_df = pd.read_csv(csv_file)
            completed = existing_df["kappa_init"].tolist()
            remaining = [κ for κ in kappa_values if κ not in completed]
            print(f"Found {len(completed)} completed runs. Resuming with {len(remaining)} remaining values.")
            results = existing_df.to_dict("records")
            kappa_values = remaining
        except (pd.errors.EmptyDataError, KeyError, IndexError):
            print("Existing CSV file is empty or invalid. Starting fresh.")
            results = []
    
    for i, κ in enumerate(kappa_values):
        print(f"\n[{i+1}/{len(kappa_values)}] Training with κ_init = {κ}")
        try:
            metrics, net = train_with_params(
                tr_dl, va_dl, te_dl, qthr, stats,
                lambda_phys=lambda_phys,
                kappa_init=κ,
                kappa_trainable=True,
                epochs=epochs,
                verbose=True
            )
            
            result = {
                "kappa_init": κ,
                "lambda": lambda_phys if lambda_phys is not None else "auto",
                "final_kappa": metrics["final_kappa"],
                "train_rmse": metrics["train"],
                "val_rmse": metrics["val"],
                "test_rmse": metrics["test"],
                "best_epoch": metrics["best_epoch"]
            }
            results.append(result)
            
            # Save incrementally after each run
            df = pd.DataFrame(results)
            df.to_csv(csv_file, index=False)
            
            print(f"  Results: train={metrics['train']:.4f}, val={metrics['val']:.4f}, test={metrics['test']:.4f}")
            print(f"  Final κ = {metrics['final_kappa']:.2f} (init: {κ})")
            print(f"  ✓ Saved progress to {csv_file}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print(f"  Continuing with next value...")
            continue
    
    return pd.DataFrame(results)

def ablation_combined(lambda_values, kappa_values, epochs=25):
    """Combined ablation study on both λ and κ."""
    print(f"\n{'='*70}")
    print(f"COMBINED ABLATION STUDY: λ × κ")
    print(f"{'='*70}")
    print(f"λ values: {lambda_values}")
    print(f"κ values: {kappa_values}")
    print(f"Total combinations: {len(lambda_values) * len(kappa_values)}")
    print(f"Epochs per run: {epochs}\n")
    
    # Check if data file exists
    if not FREEZE.exists():
        raise FileNotFoundError(
            f"Data file not found: {FREEZE}\n"
            f"Please update FREEZE path in pinn_model.py or provide correct path."
        )
    
    tr_dl, va_dl, te_dl, qthr, stats = make_loaders()
    results = []
    
    total = len(lambda_values) * len(kappa_values)
    idx = 0
    
    for λ, κ in itertools.product(lambda_values, kappa_values):
        idx += 1
        print(f"\n[{idx}/{total}] Training with λ = {λ}, κ_init = {κ}")
        metrics, net = train_with_params(
            tr_dl, va_dl, te_dl, qthr, stats,
            lambda_phys=λ,
            kappa_init=κ,
            kappa_trainable=True,
            epochs=epochs,
            verbose=False
        )
        
        results.append({
            "lambda": λ,
            "kappa_init": κ,
            "final_kappa": metrics["final_kappa"],
            "train_rmse": metrics["train"],
            "val_rmse": metrics["val"],
            "test_rmse": metrics["test"],
            "best_epoch": metrics["best_epoch"]
        })
        
        print(f"  Results: val={metrics['val']:.4f}, test={metrics['test']:.4f}, final_κ={metrics['final_kappa']:.2f}")
    
    return pd.DataFrame(results)

# ------------------------------------------------------------------ #
# Visualization
# ------------------------------------------------------------------ #
def plot_ablation_results(df, study_type, out_dir):
    """Plot ablation study results."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if study_type == "lambda":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE vs lambda
        axes[0, 0].plot(df["lambda"], df["val_rmse"], "o-", label="Validation", linewidth=2)
        axes[0, 0].plot(df["lambda"], df["test_rmse"], "s--", label="Test", linewidth=2)
        axes[0, 0].set_xlabel("Physics Loss Weight (λ)", fontsize=12)
        axes[0, 0].set_ylabel("RMSE (log chl)", fontsize=12)
        axes[0, 0].set_title("RMSE vs Physics Loss Weight", fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Final kappa vs lambda
        axes[0, 1].plot(df["lambda"], df["final_kappa"], "o-", color="green", linewidth=2)
        axes[0, 1].set_xlabel("Physics Loss Weight (λ)", fontsize=12)
        axes[0, 1].set_ylabel("Final κ", fontsize=12)
        axes[0, 1].set_title("Learned Diffusivity vs λ", fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Train vs Val RMSE
        axes[1, 0].scatter(df["train_rmse"], df["val_rmse"], c=df["lambda"], cmap="viridis", s=100)
        axes[1, 0].plot([0.6, 0.9], [0.6, 0.9], "r--", alpha=0.5)
        axes[1, 0].set_xlabel("Train RMSE", fontsize=12)
        axes[1, 0].set_ylabel("Validation RMSE", fontsize=12)
        axes[1, 0].set_title("Train vs Validation RMSE", fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label="λ")
        
        # Best epoch vs lambda
        axes[1, 1].bar(range(len(df)), df["best_epoch"], color="steelblue", alpha=0.7)
        axes[1, 1].set_xticks(range(len(df)))
        axes[1, 1].set_xticklabels([f"{λ:.2f}" for λ in df["lambda"]], rotation=45)
        axes[1, 1].set_xlabel("Physics Loss Weight (λ)", fontsize=12)
        axes[1, 1].set_ylabel("Best Epoch", fontsize=12)
        axes[1, 1].set_title("Convergence Speed vs λ", fontsize=14)
        axes[1, 1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig(out_dir / "ablation_lambda.png", dpi=300, bbox_inches="tight")
        print(f"\nSaved plot: {out_dir / 'ablation_lambda.png'}")
        
    elif study_type == "kappa":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE vs kappa
        axes[0, 0].semilogx(df["kappa_init"], df["val_rmse"], "o-", label="Validation", linewidth=2)
        axes[0, 0].semilogx(df["kappa_init"], df["test_rmse"], "s--", label="Test", linewidth=2)
        axes[0, 0].set_xlabel("Initial κ", fontsize=12)
        axes[0, 0].set_ylabel("RMSE (log chl)", fontsize=12)
        axes[0, 0].set_title("RMSE vs Initial Diffusivity", fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Final vs initial kappa
        axes[0, 1].loglog(df["kappa_init"], df["final_kappa"], "o-", color="green", linewidth=2)
        axes[0, 1].plot([1, 1000], [1, 1000], "r--", alpha=0.5, label="y=x")
        axes[0, 1].set_xlabel("Initial κ", fontsize=12)
        axes[0, 1].set_ylabel("Final κ", fontsize=12)
        axes[0, 1].set_title("Learned vs Initial κ", fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Change in kappa
        kappa_change = df["final_kappa"] - df["kappa_init"]
        axes[1, 0].plot(df["kappa_init"], kappa_change, "o-", color="purple", linewidth=2)
        axes[1, 0].axhline(0, color="r", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Initial κ", fontsize=12)
        axes[1, 0].set_ylabel("Δκ (final - initial)", fontsize=12)
        axes[1, 0].set_title("Change in Diffusivity During Training", fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # RMSE improvement
        baseline_rmse = df[df["kappa_init"] == df["kappa_init"].median()]["val_rmse"].values[0]
        rmse_improvement = baseline_rmse - df["val_rmse"]
        axes[1, 1].plot(df["kappa_init"], rmse_improvement, "o-", color="orange", linewidth=2)
        axes[1, 1].axhline(0, color="r", linestyle="--", alpha=0.5)
        axes[1, 1].set_xlabel("Initial κ", fontsize=12)
        axes[1, 1].set_ylabel("RMSE Improvement", fontsize=12)
        axes[1, 1].set_title("RMSE Improvement vs Baseline", fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out_dir / "ablation_kappa.png", dpi=300, bbox_inches="tight")
        print(f"\nSaved plot: {out_dir / 'ablation_kappa.png'}")
        
    elif study_type == "both":
        # Heatmap of validation RMSE
        pivot = df.pivot(index="kappa_init", columns="lambda", values="val_rmse")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis_r", ax=axes[0], cbar_kws={"label": "Val RMSE"})
        axes[0].set_xlabel("Physics Loss Weight (λ)", fontsize=12)
        axes[0].set_ylabel("Initial κ", fontsize=12)
        axes[0].set_title("Validation RMSE: λ × κ", fontsize=14)
        
        # Best combination
        best_idx = df["val_rmse"].idxmin()
        best_row = df.loc[best_idx]
        axes[1].text(0.1, 0.8, f"Best Configuration:", fontsize=14, weight="bold", transform=axes[1].transAxes)
        axes[1].text(0.1, 0.7, f"  λ = {best_row['lambda']:.2f}", fontsize=12, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.6, f"  κ_init = {best_row['kappa_init']:.1f}", fontsize=12, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.5, f"  κ_final = {best_row['final_kappa']:.2f}", fontsize=12, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.4, f"  Val RMSE = {best_row['val_rmse']:.4f}", fontsize=12, transform=axes[1].transAxes)
        axes[1].text(0.1, 0.3, f"  Test RMSE = {best_row['test_rmse']:.4f}", fontsize=12, transform=axes[1].transAxes)
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(out_dir / "ablation_combined.png", dpi=300, bbox_inches="tight")
        print(f"\nSaved plot: {out_dir / 'ablation_combined.png'}")
    
    plt.close()

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(description="PINN ablation studies")
    ap.add_argument("--study", type=str, choices=["lambda", "kappa", "both"], required=True,
                    help="Type of ablation study")
    ap.add_argument("--lambda-values", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                    help="Physics loss weight values to test")
    ap.add_argument("--kappa-values", type=float, nargs="+", default=[1, 5, 10, 25, 50, 100, 500],
                    help="Initial diffusivity values to test")
    ap.add_argument("--epochs", type=int, default=30, help="Epochs per run")
    ap.add_argument("--output-dir", type=str, default="ablation_results", help="Output directory")
    args = ap.parse_args()
    
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.study == "lambda":
        df = ablation_lambda(args.lambda_values, epochs=args.epochs)
        df.to_csv(out_dir / "ablation_lambda.csv", index=False)
        plot_ablation_results(df, "lambda", out_dir)
        
    elif args.study == "kappa":
        # Use first lambda value if provided, otherwise None (auto-scale)
        lambda_phys = args.lambda_values[0] if args.lambda_values else None
        df = ablation_kappa(args.kappa_values, lambda_phys=lambda_phys, epochs=args.epochs)
        df.to_csv(out_dir / "ablation_kappa.csv", index=False)
        plot_ablation_results(df, "kappa", out_dir)
        
    elif args.study == "both":
        df = ablation_combined(args.lambda_values, args.kappa_values, epochs=args.epochs)
        df.to_csv(out_dir / "ablation_combined.csv", index=False)
        plot_ablation_results(df, "both", out_dir)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {out_dir}")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()

