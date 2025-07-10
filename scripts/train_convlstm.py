#!/usr/bin/env python3
"""
ConvLSTM regressor — 8‐day HAB forecast, predicting a full 64×64 patch.

Inputs: (SEQ=4, C=20, 64, 64)
Outputs: Δlog_chl patch (64×64) → add persistence → full log_chl next composite.
Loss: sum of squared errors over water‐only, observed pixels.
"""
from __future__ import annotations
import pathlib, numpy as np, torch, torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_cubes import make_loaders, CHANS, SEQ, PATCH, LOGCHL_IDX

root   = pathlib.Path(__file__).resolve().parents[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PixelLSTM(nn.Module):
    def __init__(self, cin:int, cout:int):
        super().__init__()
        self.cell  = nn.LSTMCell(cin, cout)
        self.conv1 = nn.Conv2d(cout, cout, 1)
    @staticmethod
    def _zeros(n, feat, dtype, dev):
        z = torch.zeros(n, feat, dtype=dtype, device=dev)
        return z.clone(), z.clone()
    def forward(self, x, hc=None):
        B,C,H,W = x.shape
        flat    = x.permute(0,2,3,1).reshape(B*H*W, C)
        if hc is None:
            hc = self._zeros(B*H*W, self.cell.hidden_size, flat.dtype, flat.device)
        h,c = self.cell(flat, hc)
        hmap = h.reshape(B,H,W,-1).permute(0,3,1,2)
        return self.conv1(hmap), (h,c)


class ConvLSTMNet(nn.Module):
    def __init__(self, chans:int=CHANS):
        super().__init__()
        self.l1   = PixelLSTM(chans, 32)
        self.l2   = PixelLSTM(32,   64)
        self.head = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x):
        h1 = h2 = None
        for t in range(x.size(1)):
            out1, h1 = self.l1(x[:,t], h1)
            out2, h2 = self.l2(out1,  h2)
        return self.head(out2).squeeze(1)  # (B,H,W)


@torch.no_grad()
def run_epoch(dl, model):
    model.eval()
    tot_sqerr, tot_count = 0.0, 0
    for X, y_true, mask in dl:
        X, y_true, mask = [t.to(device) for t in (X,y_true,mask)]
        # persistence = last‐frame log_chl
        pers       = X[:, -1, LOGCHL_IDX]      # (B,H,W)
        delta_pred = model(X)                  # (B,H,W)
        pred       = pers + delta_pred         # (B,H,W)

        # mask = water & observed
        obs_mask   = (~torch.isnan(y_true)) & mask.bool()
        err_map    = torch.where(obs_mask, pred - y_true,
                                 torch.zeros_like(pred))
        tot_sqerr += (err_map*err_map).sum().item()
        tot_count += obs_mask.sum().item()

    return np.sqrt(tot_sqerr / tot_count)


def train_epoch(dl, model, opt):
    model.train()
    for X, y_true, mask in dl:
        X, y_true, mask = [t.to(device) for t in (X,y_true,mask)]
        pers     = X[:, -1, LOGCHL_IDX]
        true_dp  = y_true - pers
        dp_pred  = model(X)

        obs_mask = (~torch.isnan(y_true)) & mask.bool()
        err_map  = torch.where(obs_mask, dp_pred - true_dp,
                               torch.zeros_like(dp_pred))
        loss     = (err_map*err_map).sum()  # sum‐MSE

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def main():
    tr_dl, va_dl, te_dl = make_loaders(batch=8, workers=4)

    model = ConvLSTMNet().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5, verbose=True)

    best_val, patience = np.inf, 10
    for epoch in range(1,51):
        train_epoch(tr_dl, model, opt)
        tr_rmse = run_epoch(tr_dl, model)
        va_rmse = run_epoch(va_dl, model)
        sched.step(va_rmse)

        print(f"E{epoch:02d} train {tr_rmse:.3f}  val {va_rmse:.3f}  "
              f"lr {opt.param_groups[0]['lr']:.1e}")

        if va_rmse < best_val:
            best_val, patience = va_rmse, 10
            torch.save(model.state_dict(), root/"Models"/"convLSTM_best.pt")
        else:
            patience -= 1
            if patience == 0:
                break

    # final test
    model.load_state_dict(torch.load(root/"Models"/"convLSTM_best.pt"))
    test_rmse = run_epoch(te_dl, model)
    print(f"\nTEST 2021 RMSE(log_chl) = {test_rmse:.3f}")

    (root/"Models"/"convLSTM_metrics.txt").write_text(
        f"val_rmse={best_val:.5f}\n"
        f"test_rmse={test_rmse:.5f}\n"
    )
    print("✓ metrics written.")


if __name__ == "__main__":
    main()
