#!/usr/bin/env python3
"""
Compile Paper Results

This script compiles all experimental results into a single summary
for easy inclusion in the paper.

Usage:
    python pinn/compile_paper_results.py --output paper_results_summary.md
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def load_ablation_results(results_dir: pathlib.Path):
    """Load ablation study results."""
    results = {}
    
    # Lambda ablation
    lambda_file = results_dir / "ablation" / "lambda" / "ablation_lambda.csv"
    if lambda_file.exists():
        df = pd.read_csv(lambda_file)
        best = df.loc[df["val_rmse"].idxmin()]
        results["lambda"] = {
            "optimal": float(best["lambda"]),
            "val_rmse": float(best["val_rmse"]),
            "test_rmse": float(best["test_rmse"]),
            "all_results": df.to_dict("records")
        }
    
    # Kappa ablation
    kappa_file = results_dir / "ablation" / "kappa" / "ablation_kappa.csv"
    if kappa_file.exists():
        df = pd.read_csv(kappa_file)
        best = df.loc[df["val_rmse"].idxmin()]
        results["kappa"] = {
            "optimal": float(best["kappa_init"]),
            "final_kappa": float(best["final_kappa"]),
            "val_rmse": float(best["val_rmse"]),
            "test_rmse": float(best["test_rmse"]),
            "all_results": df.to_dict("records")
        }
    
    # Combined ablation
    combined_file = results_dir / "ablation" / "combined" / "ablation_combined.csv"
    if combined_file.exists():
        df = pd.read_csv(combined_file)
        best = df.loc[df["val_rmse"].idxmin()]
        results["combined"] = {
            "optimal_lambda": float(best["lambda"]),
            "optimal_kappa": float(best["kappa_init"]),
            "val_rmse": float(best["val_rmse"]),
            "test_rmse": float(best["test_rmse"]),
            "all_results": df.to_dict("records")
        }
    
    return results

def load_imputation_results(results_dir: pathlib.Path):
    """Load imputation sensitivity results."""
    imputation_file = results_dir / "imputation" / "imputation_results.csv"
    if not imputation_file.exists():
        return None
    
    df = pd.read_csv(imputation_file)
    best = df.loc[df["val_rmse"].idxmin()]
    
    return {
        "best_method": best["method"],
        "best_val_rmse": float(best["val_rmse"]),
        "best_test_rmse": float(best["test_rmse"]),
        "sensitivity_range": float(df["val_rmse"].max() - df["val_rmse"].min()),
        "all_results": df.to_dict("records")
    }

def load_diagnostics_metrics(diagnostics_dir: pathlib.Path):
    """Load diagnostics metrics."""
    metrics_file = diagnostics_dir / "metrics_global.csv"
    if not metrics_file.exists():
        return None
    
    df = pd.read_csv(metrics_file)
    return df.to_dict("records")

def create_paper_summary(results_dir: pathlib.Path, output_file: pathlib.Path):
    """Create comprehensive paper summary."""
    
    ablation = load_ablation_results(results_dir)
    imputation = load_imputation_results(results_dir)
    
    # Try to find diagnostics
    diagnostics_dirs = [
        pathlib.Path("Diagnostics_PINN"),
        pathlib.Path("Diagnostics_PINN_Best"),
        results_dir / "diagnostics"
    ]
    diagnostics = None
    for d in diagnostics_dirs:
        if d.exists():
            diagnostics = load_diagnostics_metrics(d)
            break
    
    # Create markdown summary
    summary = f"""# PINN Model Results Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Hyperparameter Optimization

### Physics Loss Weight (λ)

"""
    
    if "lambda" in ablation:
        λ_data = ablation["lambda"]
        summary += f"""**Optimal Value**: λ = {λ_data['optimal']:.2f}

**Performance**:
- Validation RMSE: {λ_data['val_rmse']:.4f}
- Test RMSE: {λ_data['test_rmse']:.4f}

**All Tested Values**:
| λ | Val RMSE | Test RMSE |
|---|---|---|
"""
        for r in λ_data['all_results']:
            summary += f"| {r['lambda']:.2f} | {r['val_rmse']:.4f} | {r['test_rmse']:.4f} |\n"
    
    summary += "\n### Diffusivity (κ)\n\n"
    
    if "kappa" in ablation:
        κ_data = ablation["kappa"]
        summary += f"""**Optimal Initial Value**: κ_init = {κ_data['optimal']:.1f} m²/s

**Final Learned Value**: κ_final = {κ_data['final_kappa']:.2f} m²/s

**Performance**:
- Validation RMSE: {κ_data['val_rmse']:.4f}
- Test RMSE: {κ_data['test_rmse']:.4f}

**All Tested Values**:
| κ_init | κ_final | Val RMSE | Test RMSE |
|---|---|---|---|
"""
        for r in κ_data['all_results']:
            summary += f"| {r['kappa_init']:.1f} | {r['final_kappa']:.2f} | {r['val_rmse']:.4f} | {r['test_rmse']:.4f} |\n"
    
    summary += "\n### Combined Optimal Configuration\n\n"
    
    if "combined" in ablation:
        comb_data = ablation["combined"]
        summary += f"""**Optimal Parameters**:
- λ = {comb_data['optimal_lambda']:.2f}
- κ_init = {comb_data['optimal_kappa']:.1f} m²/s

**Best Performance**:
- Validation RMSE: {comb_data['val_rmse']:.4f}
- Test RMSE: {comb_data['test_rmse']:.4f}
"""
    
    summary += "\n## 2. Data Imputation Sensitivity\n\n"
    
    if imputation:
        summary += f"""**Best Method**: {imputation['best_method']}

**Performance**:
- Validation RMSE: {imputation['best_val_rmse']:.4f}
- Test RMSE: {imputation['best_test_rmse']:.4f}

**Sensitivity Range**: {imputation['sensitivity_range']:.4f} RMSE

**All Methods**:
| Method | Val RMSE | Test RMSE | Val MAE | Test MAE | Val Corr | Test Corr |
|---|---|---|---|---|---|---|
"""
        for r in imputation['all_results']:
            summary += f"| {r['method']} | {r['val_rmse']:.4f} | {r['test_rmse']:.4f} | {r['val_mae']:.4f} | {r['test_mae']:.4f} | {r['val_corr']:.4f} | {r['test_corr']:.4f} |\n"
    else:
        summary += "*Imputation sensitivity analysis not yet completed.*\n"
    
    summary += "\n## 3. Final Model Performance\n\n"
    
    if diagnostics:
        summary += "**Global Metrics** (from diagnostics):\n\n"
        summary += "| Subset | RMSE | MAE | Correlation |\n"
        summary += "|---|---|---|---|\n"
        for d in diagnostics:
            subset = d.get('subset', 'unknown')
            # Extract metrics (adjust column names as needed)
            rmse = d.get('rmse', d.get('RMSE', 'N/A'))
            mae = d.get('mae', d.get('MAE', 'N/A'))
            corr = d.get('corr', d.get('correlation', d.get('r', 'N/A')))
            summary += f"| {subset} | {rmse} | {mae} | {corr} |\n"
    else:
        summary += "*Diagnostics not yet run. Run diagnostics.py to generate metrics.*\n"
    
    summary += "\n## 4. Key Findings for Paper\n\n"
    
    if "combined" in ablation and imputation:
        summary += f"""1. **Optimal Hyperparameters**: λ = {ablation['combined']['optimal_lambda']:.2f}, κ_init = {ablation['combined']['optimal_kappa']:.1f} m²/s
2. **Best Performance**: Validation RMSE = {ablation['combined']['val_rmse']:.4f}, Test RMSE = {ablation['combined']['test_rmse']:.4f}
3. **Robustness**: Model shows {'low' if imputation['sensitivity_range'] < 0.05 else 'moderate'} sensitivity to imputation (range: {imputation['sensitivity_range']:.4f} RMSE)
4. **Best Imputation**: {imputation['best_method']} method performs best
"""
    
    summary += "\n## 5. Figures for Paper\n\n"
    summary += "1. Ablation study heatmaps (λ × κ performance)\n"
    summary += "2. Imputation sensitivity comparison bar chart\n"
    summary += "3. Final model diagnostics (scatter plots, skill maps, etc.)\n"
    summary += "4. (If available) Uncertainty quantification examples\n"
    
    summary += "\n## 6. Paper Text Suggestions\n\n"
    
    if "combined" in ablation:
        summary += f"""### Results Section

"We performed systematic ablation studies to optimize the physics loss weight (λ) and initial diffusivity (κ). The optimal configuration was λ = {ablation['combined']['optimal_lambda']:.2f} and κ_init = {ablation['combined']['optimal_kappa']:.1f} m²/s, achieving validation RMSE of {ablation['combined']['val_rmse']:.4f} and test RMSE of {ablation['combined']['test_rmse']:.4f}."

"""
    
    if imputation:
        summary += f"""### Robustness Analysis

"To assess robustness to data preprocessing, we tested {len(imputation['all_results'])} imputation methods including zero-fill, mean/median imputation, forward-fill, climatology-based, and linear interpolation. The model showed {'low' if imputation['sensitivity_range'] < 0.05 else 'moderate'} sensitivity with RMSE variation of {imputation['sensitivity_range']:.4f}. The best-performing method was {imputation['best_method']}, achieving validation RMSE of {imputation['best_val_rmse']:.4f}."

"""
    
    summary += "\n---\n"
    summary += "*This summary was automatically generated. Review and verify all values before including in paper.*\n"
    
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"✓ Paper summary compiled: {output_file}")
    
    # Also save as JSON for programmatic access
    json_file = output_file.with_suffix('.json')
    json_data = {
        "ablation": ablation,
        "imputation": imputation,
        "diagnostics": diagnostics,
        "generated": datetime.now().isoformat()
    }
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ JSON data saved: {json_file}")

def main():
    ap = argparse.ArgumentParser(description="Compile paper results")
    ap.add_argument("--results-dir", type=str, default="paper_results",
                   help="Results directory")
    ap.add_argument("--output", type=str, default="paper_results_summary.md",
                   help="Output summary file")
    args = ap.parse_args()
    
    results_dir = pathlib.Path(args.results_dir)
    output_file = pathlib.Path(args.output)
    
    if not results_dir.exists():
        print(f"⚠ Warning: Results directory not found: {results_dir}")
        print("  Run experiments first using run_paper_experiments.py")
        return
    
    create_paper_summary(results_dir, output_file)

if __name__ == "__main__":
    main()

