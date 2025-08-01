# drop this snippet into an IPython / python -i session
import torch, xarray as xr, numpy as np, pathlib
from baseline_model import make_loaders, physics_residual, LOGCHL_IDX, U_IDX, V_IDX, _FLOOR

FREEZE = pathlib.Path("/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc")
tr_dl, *_ , stats = make_loaders()        # uses your existing helper
X, y, m = next(iter(tr_dl))
mu_u, sd_u = stats["uo"]; mu_v, sd_v = stats["vo"]

with torch.no_grad():
    # pretend prediction == last input (Δ=0) just to probe scale
    C_last_lin = torch.exp(X[:,-1,LOGCHL_IDX]) - _FLOOR
    C_pred_lin = C_last_lin.clone()
    u_lin = (X[:,-1,U_IDX] * sd_u + mu_u)
    v_lin = (X[:,-1,V_IDX] * sd_v + mu_v)
    res = physics_residual(C_pred_lin, C_last_lin, u_lin, v_lin)
print(res.abs().mean().item())            # expect ~8e-5
