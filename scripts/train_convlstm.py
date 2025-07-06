#!/usr/bin/env python3
"""
ConvLSTM regressor for 8-day HAB forecast.

X shape : (B, T=4, C=18, 64, 64)
y        : scalar log10-chlor_a at t+1 (central pixel)
"""
from __future__ import annotations
import pathlib, torch, torch.nn as nn, numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_cubes import make_loaders, CHANS, SEQ
root   = pathlib.Path(__file__).resolve().parents[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ───────────────── pixel-wise Conv-LSTM block ────────────────────
class PixelLSTM(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.cell  = nn.LSTMCell(cin, cout)
        self.conv3 = nn.Conv2d(cout, cout, 3, padding=1)

    def _init_state(self, BHW, hidden, dtype, dev):
        h = torch.zeros(BHW, hidden, dtype=dtype, device=dev)
        c = torch.zeros_like(h)
        return h, c

    def forward(self, x, hc=None):            # x (B,C,H,W)
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1).reshape(B*H*W, C)       # (BHW,C)
        if hc is None:
            hc = self._init_state(B*H*W,
                                   self.cell.hidden_size,
                                   x.dtype, x.device)
        h, c = self.cell(x, hc)
        h_map = h.reshape(B, H, W, -1).permute(0,3,1,2)  # (B,Cout,H,W)
        return self.conv3(h_map), (h, c)

# ─────────────────── ConvLSTM network ─────────────────────────────
class ConvLSTMNet(nn.Module):
    def __init__(self, chans: int = CHANS):
        super().__init__()
        self.l1   = PixelLSTM(chans, 32)
        self.l2   = PixelLSTM(32,   64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(64, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):                     # x (B,T,C,H,W)
        h1 = h2 = None
        for t in range(x.size(1)):
            out1, h1 = self.l1(x[:, t], h1)
            out2, h2 = self.l2(out1,    h2)
        return self.head(self.pool(out2)).squeeze(1)

# ────────────────── helpers ───────────────────────────────────────
def run_epoch(dl, model, crit, optim=None):
    rmses = []
    for X, y in dl:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = crit(pred, y)
        if optim:
            optim.zero_grad(); loss.backward(); optim.step()
        rmses.append(loss.detach().cpu().item())
    return np.sqrt(np.mean(rmses))            # RMSE (log-space)

# ────────────────── training loop ─────────────────────────────────
def train():
    tr_dl, va_dl, te_dl = make_loaders(batch=8, workers=4)
    model = ConvLSTMNet().to(device)
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = ReduceLROnPlateau(opt, patience=3, factor=0.5, verbose=True)

    best, patience, ep = np.inf, 10, 0
    while patience:
        ep += 1
        tr_rmse = run_epoch(tr_dl, model, crit, opt)
        va_rmse = run_epoch(va_dl, model, crit)
        sched.step(va_rmse)

        lr = opt.param_groups[0]['lr']
        print(f"E{ep:02d}  train {tr_rmse:.3f}  val {va_rmse:.3f}  lr {lr:.1e}")

        if va_rmse < best:
            best, patience = va_rmse, 10
            torch.save(model.state_dict(), root/'Models/convLSTM_best.pt')
        else:
            patience -= 1

    # ─── test on 2021 ────────────────────────────────────────────
    model.load_state_dict(torch.load(root/'Models/convLSTM_best.pt'))
    te_rmse = run_epoch(te_dl, model, crit)
    print(f"\nTEST RMSE(log) = {te_rmse:.3f}")
    with open(root/'Models/convLSTM_metrics.txt','w') as f:
        f.write(f"val_rmse={best:.4f}\ntest_rmse={te_rmse:.4f}\n")

if __name__ == "__main__":
    train()
