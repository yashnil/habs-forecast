#!/usr/bin/env python3
"""
ConvLSTM regressor — 8-day HAB forecast (log-chlorophyll).

Inputs
  X : (B, T=4, C=18, 64, 64)   from data_cubes.py
  y : scalar (B,)              log10-chl at t+1 (tile centre)
  m : (B, 64, 64)              water mask (not yet used in the loss)

The network is pixel-wise: an LSTMCell is run for every (H,W) location,
followed by a 1×1 conv and global-avg-pool.
"""
from __future__ import annotations
import pathlib, numpy as np, torch, torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_cubes import make_loaders, CHANS, SEQ          # CHANS=18, SEQ=4

root   = pathlib.Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────── pixel-wise LSTM block ───────────────────────────────
class PixelLSTM(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.cell  = nn.LSTMCell(cin, cout)
        self.conv1 = nn.Conv2d(cout, cout, kernel_size=1)

    @staticmethod
    def _zeros(n: int, feat: int, dtype, dev):
        z = torch.zeros(n, feat, dtype=dtype, device=dev)
        return z.clone(), z.clone()           # h , c

    def forward(self, x, hc=None):            # x (B,C,H,W)
        B, C, H, W = x.shape
        flat = x.permute(0,2,3,1).reshape(B*H*W, C)  # (BHW,C)
        if hc is None:
            hc = self._zeros(B*H*W, self.cell.hidden_size, flat.dtype, flat.device)
        h, c = self.cell(flat, hc)
        h_map = h.reshape(B, H, W, -1).permute(0,3,1,2)   # (B,cout,H,W)
        return self.conv1(h_map), (h, c)

# ─────────────── ConvLSTM net ─────────────────────────────────────
class ConvLSTMNet(nn.Module):
    def __init__(self, chans: int = CHANS):
        super().__init__()
        self.l1   = PixelLSTM(chans, 32)
        self.l2   = PixelLSTM(32,   64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):                     # x (B,T,C,H,W)
        h1 = h2 = None
        for t in range(x.size(1)):
            out1, h1 = self.l1(x[:, t], h1)
            out2, h2 = self.l2(out1,    h2)
        return self.head(self.pool(out2)).squeeze(1)  # (B,)

# ───────────── helper to run one epoch ────────────────────────────
@torch.no_grad()
def run_epoch(dl, model, loss_fn):
    tot, n = 0.0, 0
    model.eval()
    for X, y, _ in dl:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        tot += loss_fn(pred, y).item() * X.size(0)
        n   += X.size(0)
    return np.sqrt(tot / n)                  # RMSE on log-scale

def train_epoch(dl, model, loss_fn, opt):
    model.train()
    for X, y, _ in dl:
        X, y = X.to(device), y.to(device)
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()

# ──────────────── training loop ───────────────────────────────────
def main():
    tr_dl, va_dl, te_dl = make_loaders(batch=8, workers=4)

    model = ConvLSTMNet().to(device)
    loss_fn = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5, verbose=True)

    best_val, patience = np.inf, 10
    epoch = 0
    while patience:
        epoch += 1
        train_epoch(tr_dl, model, loss_fn, opt)
        tr_rmse = run_epoch(tr_dl, model, loss_fn)
        va_rmse = run_epoch(va_dl, model, loss_fn)
        sched.step(va_rmse)

        lr = opt.param_groups[0]["lr"]
        print(f"E{epoch:02d}  train {tr_rmse:.3f}  val {va_rmse:.3f}  lr {lr:.1e}")

        if va_rmse < best_val:
            best_val, patience = va_rmse, 10
            torch.save(model.state_dict(), root / "Models/convLSTM_best.pt")
        else:
            patience -= 1

    # ───────── test on 2021 ───────────────────────────────────────
    model.load_state_dict(torch.load(root / "Models/convLSTM_best.pt"))
    test_rmse = run_epoch(te_dl, model, loss_fn)
    print(f"\nTEST 2021  RMSE(log) = {test_rmse:.3f}")

    out = root / "Models/convLSTM_metrics.txt"
    out.write_text(f"val_rmse={best_val:.5f}\ntest_rmse={test_rmse:.5f}\n")
    print("✓ metrics written to", out)

if __name__ == "__main__":
    main()
