#!/usr/bin/env python3
"""
ConvLSTM regressor — 8-day HAB forecast (log-chlorophyll), predicting
the increment Δ = log_chl(t+1) − log_chl(t) and then reconstructing full
log_chl for RMSE computation.
"""
from __future__ import annotations
import pathlib, numpy as np, torch, torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_cubes import make_loaders, CHANS, SEQ, PATCH

root   = pathlib.Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PixelLSTM(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.cell  = nn.LSTMCell(cin, cout)
        self.conv1 = nn.Conv2d(cout, cout, 1)
    @staticmethod
    def _zeros(n, feat, dtype, dev):
        z = torch.zeros(n, feat, dtype=dtype, device=dev)
        return z.clone(), z.clone()
    def forward(self, x, hc=None):
        B,C,H,W = x.shape
        flat = x.permute(0,2,3,1).reshape(B*H*W, C)
        if hc is None:
            hc = self._zeros(B*H*W, self.cell.hidden_size,
                              flat.dtype, flat.device)
        h,c = self.cell(flat, hc)
        h_map = h.reshape(B,H,W,-1).permute(0,3,1,2)
        return self.conv1(h_map), (h,c)

class ConvLSTMNet(nn.Module):
    def __init__(self, chans: int = CHANS):
        super().__init__()
        self.l1   = PixelLSTM(chans, 32)
        self.l2   = PixelLSTM(32,   64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        h1=h2=None
        for t in range(x.size(1)):
            out1, h1 = self.l1(x[:,t], h1)
            out2, h2 = self.l2(out1,  h2)
        return self.head(self.pool(out2)).squeeze(1)  # Δ

@torch.no_grad()
def run_epoch(dl, model, loss_fn):
    model.eval()
    tot, n = 0.0, 0
    for X, y, _ in dl:
        X,y = X.to(device), y.to(device)
        pers       = X[:, -1, 14, PATCH//2, PATCH//2]
        delta_pred = model(X)
        y_pred     = pers + delta_pred
        loss       = loss_fn(y_pred, y)
        tot       += loss.item() * X.size(0)
        n         += X.size(0)
    return np.sqrt(tot/n)

def train_epoch(dl, model, loss_fn, opt):
    model.train()
    for X, y, _ in dl:
        X,y = X.to(device), y.to(device)
        pers       = X[:, -1, 14, PATCH//2, PATCH//2]
        delta_true = y - pers
        delta_pred = model(X)
        loss       = loss_fn(delta_pred, delta_true)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

def main():
    tr_dl, va_dl, te_dl = make_loaders(batch=8, workers=4)
    model   = ConvLSTMNet().to(device)
    loss_fn = nn.MSELoss()
    opt     = torch.optim.Adam(
                  model.parameters(),
                  lr=5e-4,
                  weight_decay=1e-5
              )
    sched   = ReduceLROnPlateau(opt, mode="min",
                                patience=3, factor=0.5, verbose=True)

    best_val, patience = np.inf, 10
    epoch = 0
    while patience:
        epoch += 1
        train_epoch(tr_dl, model, loss_fn, opt)
        tr_rmse = run_epoch(tr_dl, model, loss_fn)
        va_rmse = run_epoch(va_dl, model, loss_fn)
        sched.step(va_rmse)
        print(f"E{epoch:02d}  train {tr_rmse:.3f}  val {va_rmse:.3f}  "
              f"lr {opt.param_groups[0]['lr']:.1e}")
        if va_rmse < best_val:
            best_val, patience = va_rmse, 10
            torch.save(model.state_dict(),
                       root/"Models"/"convLSTM_best.pt")
        else:
            patience -= 1

    model.load_state_dict(torch.load(root/"Models"/"convLSTM_best.pt"))
    test_rmse = run_epoch(te_dl, model, loss_fn)
    print(f"\nTEST 2021  RMSE(log) = {test_rmse:.3f}")
    (root/"Models"/"convLSTM_metrics.txt").write_text(
        f"val_rmse={best_val:.5f}\n"
        f"test_rmse={test_rmse:.5f}\n"
    )
    print("✓ metrics written.")

if __name__ == "__main__":
    main()
