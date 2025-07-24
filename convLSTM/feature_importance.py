
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths & variables
repo_root = Path().resolve()                # your repo root
model_path = repo_root/"Models"/"xgb_chla_reg.joblib"
PRED_VARS = [
    "sst", "t2m", "u10", "v10", "avg_sdswrf",
    "Kd_490", "nflh", "so", "thetao", "uo", "vo", "zos"
]

# Load the trained regressor
reg = joblib.load(model_path)

# before saving, ensure the dir exists
out_dir = Path("/Users/yashnilmohanty/Desktop/habs-all-diagnostics")
out_dir.mkdir(parents=True, exist_ok=True)

# Plot
importances = reg.feature_importances_
plt.figure(figsize=(6,4))
plt.barh(PRED_VARS, importances)
plt.xlabel("Importance (Gain)")
plt.title("XGBoost Feature Importances for log₁₀(chlor_a) Forecast")
plt.tight_layout()
plt.savefig(out_dir / "feature_importance.png", dpi=300)
plt.show()
