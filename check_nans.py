#!/usr/bin/env python3
import sys, pathlib
# ── make "tft" importable exactly like diagnostics.py did
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT / "tft"))

import xarray as xr, torch, numpy as np
from tft.test import melt_to_df, make_tensor, PixelTSD, ALL_VARS

FREEZE = pathlib.Path(
    "/Users/yashnilmohanty/Desktop/HABs_Research/Data/Derived/HAB_convLSTM_core_v1_clean.nc"
)
SEQ, LEAD = 6, 1

# 1 ─ open freeze & reproduce pixel_ok the same way test.py did
ds = xr.open_dataset(FREEZE); ds.load()
frac = np.isfinite(ds.log_chl).sum("time") / ds.sizes["time"]
pixel_ok = (frac >= .20).values           # True where ≥20 % finite

# 2 ─ recreate the “valid_core” mask that PixelTSD uses
df   = melt_to_df(ds)
feat = make_tensor(df[ALL_VARS],
                   (df.sid.nunique(), df.t.nunique(), len(ALL_VARS)))
chl  = feat[..., 0]
valid_core = torch.zeros_like(chl, dtype=torch.bool)
hist_ok = torch.isfinite(
    torch.stack([chl[:, i:i+SEQ] for i in range(chl.shape[1]-SEQ+1)], -1)
).all(1)
hist_ok = hist_ok[:, :-LEAD] if LEAD > 0 else hist_ok
future_ok = torch.isfinite(chl[:, SEQ-1+LEAD:])
valid_core[:, SEQ-1:-LEAD] = hist_ok & future_ok         # (P, T)

# 3 ─ flatten to 2-D “was ever used” mask
ever_used = valid_core.any(1).numpy()                    # (P,) → by sid
sid_to_ij = df.groupby("sid")[["iy","ix"]].first()
used_map  = np.zeros_like(pixel_ok)
for sid, row in sid_to_ij.iterrows():
    used_map[row.iy, row.ix] = ever_used[sid]

# 4 ─ disagreement statistics
both_good    = pixel_ok & used_map
pixel_only   = pixel_ok & ~used_map
used_only    = used_map & ~pixel_ok

print("Pixels marked good by freeze      :", pixel_ok.sum())
print("Pixels ever used in PixelTSD      :", used_map.sum())
print("   ↳ of which BOTH masks agree    :", both_good.sum())
print("   ↳ good in freeze but never used:", pixel_only.sum())
print("   ↳ used but not flagged good    :", used_only.sum())

'''
Pixels marked good by freeze      : 1626
Pixels ever used in PixelTSD      : 1626
   ↳ of which BOTH masks agree    : 1626
   ↳ good in freeze but never used: 0
   ↳ used but not flagged good    : 0

'''