# debug_samples.py
from data_cubes import make_loaders
import torch

tr_dl, va_dl, te_dl = make_loaders(batch=1, workers=0)

print("Inspecting first 10 training tiles:")
for idx, (X, y, mask) in enumerate(tr_dl):
    # X: (1,4,20,64,64), y: (1,64,64), mask: (1,64,64)
    n_obs = (~torch.isnan(y) & mask.bool()).sum().item()
    n_tot = mask.sum().item()
    print(f" Sample {idx:2d}: total water pixels in patch = {n_tot:4.0f}, "
          f"finite-target pixels = {n_obs:4.0f}")
    if idx >= 9:
        break
