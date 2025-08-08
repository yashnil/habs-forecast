
import torch, pathlib, os, datetime
from pprint import pprint

CKPT = pathlib.Path("/Users/yashnilmohanty/HAB_Models/convLSTM_best.pt")

# ── 1.  Basic file sanity: modified time & size
print("File modified:", datetime.datetime.fromtimestamp(CKPT.stat().st_mtime))
print("File size    :", CKPT.stat().st_size/1e6, "MB")

# ── 2.  Inspect state-dict keys
state = torch.load(CKPT, map_location="cpu")
print("Keys count   :", len(state))
print("Sample keys  :", list(state)[:8])

# ── 3.  Instantiate *your* model class and load strictly
from diagnostics import ConvLSTM   # exact class you added
model = ConvLSTM(Cin=30)                              # 30 = number of channels
model.load_state_dict(state, strict=True)             # will raise if mismatch
print("✓ state-dict loaded with strict=True")

# ── 4.  Spot-check a unique value (e.g., trained κ)
print("kappa =", model.kappa.item())                  # should be ~25 → 1000
