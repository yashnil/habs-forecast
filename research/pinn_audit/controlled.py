#!/usr/bin/env python3
"""Controlled multi-seed comparison: ConvLSTM (no physics) vs PINN, one protocol.

Reuses the original data pipeline, model, loss constants and physics residual
from ``pinn/pinn_model.py`` (imported, not copied or edited). Only these change:

* DataLoader workers forced to 0 (macOS spawn would pickle the 4 GB cube per worker).
* Per-run seed, epochs and output directory.
* ``--anchor``:
    ``z``   the original behaviour: the residual is added to the *z-scored* last
            log-chlorophyll (``X[:, -1, log_chl]``), as pinn_model.py trains it.
    ``raw`` the residual is added to the *raw* last log-chlorophyll (z*sd+mu),
            i.e. a true residual on persistence, which is what README §3.1 states
            and what pinn/diagnostics.py assumes at inference.
* ``--lam``: ``adaptive`` (pinn_model.py: clamp(L_sup/L_phys, 0.1, 1000), ramped
  in from epoch 21) or a fixed number (0 = no physics = the ConvLSTM arm).

Evaluation is full-grid on every 8-day step, without any bias correction, with
the forecast the model was trained to produce. Writes one JSON per run.

``--check`` verifies that the reimplemented loss equals the original
``train_one`` loss on one batch (anchor=z, adaptive λ) before anything else.
"""
from __future__ import annotations

import argparse, importlib.util, json, math, pathlib, random, sys, time

import numpy as np
import torch
import torch.nn as nn

REPO = pathlib.Path(__file__).resolve().parents[2]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.argv, argv = [sys.argv[0]], sys.argv  # pinn_model parses no args at import, but be safe
    spec.loader.exec_module(mod)
    sys.argv = argv
    return mod


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def batch_loss(M, net, X, y, m, qthr, stats, epoch, lam, anchor):
    """Original pinn_model.train_one loss, with the anchor and λ made explicit."""
    i = M.LOGCHL_IDX
    z_last = X[:, -1, i]
    mu, sd = stats["log_chl"]
    last = z_last if anchor == "z" else z_last * sd + mu
    delta = net(X)
    err = (delta - (y - last)).masked_fill(~m, 0)
    chl_lin = torch.exp(y)
    w = torch.where(chl_lin < qthr[0], 1.0, torch.where(chl_lin < qthr[1], 1.5, torch.where(chl_lin < qthr[2], 2.5, 4.0))) * m
    a = err.abs(); quad = torch.minimum(a, torch.tensor(M.HUBER_DELTA, device=a.device))
    sup = (w * (0.5 * quad ** 2 + M.HUBER_DELTA * (a - quad))).sum() / w.sum()
    if lam == 0:
        return sup, sup.detach(), torch.tensor(0.0)
    u = X[:, -1, M.ALL_VARS.index("uo")].unsqueeze(1)
    v = X[:, -1, M.ALL_VARS.index("vo")].unsqueeze(1)
    res = M.physics_residual((last + delta).unsqueeze(1), last.unsqueeze(1), u, v, net.kappa)
    phys = (res[:, 0][m]).pow(2).sum() / m.sum().clamp(min=1) * sd ** 2
    if lam == "adaptive":
        lam_v = (sup.detach() / (phys.detach() + 1e-8)).clamp(min=0.1, max=1000.0)
        ramp = min(max((epoch - 20) / 5, 0.0), 1.0) if epoch > 20 else 0.0
        lam_v = lam_v * ramp
    else:
        lam_v = float(lam) * (min(max((epoch - 20) / 5, 0.0), 1.0) if epoch > 20 else 0.0)
    return sup + lam_v * phys, sup.detach(), phys.detach()


@torch.no_grad()
def patch_rmse(M, dl, net, stats, anchor, dev):
    net.eval(); se = n = 0
    mu, sd = stats["log_chl"]
    for X, y, m in dl:
        X, y, m = X.to(dev), y.to(dev), m.to(dev)
        z_last = X[:, -1, M.LOGCHL_IDX]
        last = z_last if anchor == "z" else z_last * sd + mu
        e = (last + net(X) - y).masked_fill(~m, 0)
        se += (e ** 2).sum().item(); n += m.sum().item()
    return math.sqrt(se / n)


@torch.no_grad()
def full_grid(D, net, ds, stats, varlist, anchor, dev, seq=6, lead=1):
    """Full-grid forecasts for every core time; returns per-split RMSE (model, persistence)."""
    net.eval()
    ok = D.build_pixel_ok_if_needed(ds)
    la_any = np.where(ok.any("lon"))[0]; lo_any = np.where(ok.any("lat"))[0]
    la, lo = slice(la_any.min(), la_any.max() + 1), slice(lo_any.min(), lo_any.max() + 1)
    okv = ok.isel(lat=la, lon=lo).values
    logc = ds.log_chl.isel(lat=la, lon=lo).values.astype(np.float32)
    mu, sd = stats["log_chl"]
    core = np.arange(seq - 1, ds.sizes["time"] - lead)
    t = ds.time.values[core + lead]
    out = {k: [0.0, 0.0, 0] for k in ("train", "val", "test")}
    split = np.where(t < np.datetime64("2016-01-01"), "train",
                     np.where(t <= np.datetime64("2018-12-31"), "val", "test"))
    for i in range(0, len(core), 8):
        ks = core[i:i + 8]
        X = torch.from_numpy(np.stack([D.make_input(ds, varlist, stats, k, seq, la, lo) for k in ks])).to(dev)
        delta = net(X).float().cpu().numpy()
        last_raw = logc[ks]; truth = logc[ks + lead]
        z_last = np.nan_to_num((last_raw - mu) / sd, nan=0.0)
        pred = (z_last if anchor == "z" else last_raw) + delta
        msk = okv[None] & np.isfinite(truth) & np.isfinite(last_raw)
        for j, k in enumerate(ks):
            s = out[split[i + j]]
            mm = msk[j]
            s[0] += float(((pred[j] - truth[j])[mm] ** 2).sum())
            s[1] += float(((last_raw[j] - truth[j])[mm] ** 2).sum())
            s[2] += int(mm.sum())
    return {k: {"rmse": math.sqrt(a / n), "rmse_persistence": math.sqrt(b / n), "n": n} for k, (a, b, n) in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lam", default="adaptive")
    ap.add_argument("--anchor", choices=["z", "raw"], default="z")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--kappa", type=float, default=25.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    lam = a.lam if a.lam == "adaptive" else float(a.lam)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)

    M = load("pinn_model", "pinn/pinn_model.py")
    D = load("pinn_diag", "pinn/diagnostics.py")
    from torch.utils.data import DataLoader
    M.DataLoader = lambda *x, **k: DataLoader(*x, **{**k, "num_workers": 0, "pin_memory": False})
    M.DEVICE = dev
    t0 = time.time()
    tr_dl, va_dl, te_dl, qthr, stats = M.make_loaders()
    ds = tr_dl.dataset.ds
    varlist = [v for v in M.ALL_VARS if v in ds.data_vars]
    print(f"[ctl] data loaded in {time.time()-t0:.0f}s; Cin={len(varlist)} device={dev}", flush=True)

    if a.check:
        # Equivalence: original train_one loss vs batch_loss(anchor=z, adaptive), same batch, same weights.
        seed_all(0); net = M.ConvLSTM(len(varlist)).to(dev); net.eval()   # eval: dropout off for comparison
        X, y, m = next(iter(tr_dl)); X, y, m = X.to(dev), y.to(dev), m.to(dev)
        captured = {}
        class Opt:  # capture the loss the original computes
            param_groups = [{}]
            def zero_grad(self): pass
        orig_scale = M.SCALER.scale
        M.SCALER.scale = lambda loss: captured.setdefault("loss", loss.detach()) * 0 + loss * 0
        M.SCALER.unscale_ = lambda o: None; M.SCALER.step = lambda o: None; M.SCALER.update = lambda: None
        net.train = lambda mode=True: net      # keep eval mode inside train_one
        M.train_one([(X.cpu(), y.cpu(), m.cpu())], net, Opt(), qthr, stats, 25, True)
        mine, _, _ = batch_loss(M, net, X, y, m, qthr, stats, 25, "adaptive", "z")
        M.SCALER.scale = orig_scale
        diff = abs(float(captured["loss"]) - float(mine))
        print(f"[ctl] equivalence: original={float(captured['loss']):.6f} mine={float(mine):.6f} |diff|={diff:.2e}")
        assert diff < 1e-4 * max(1.0, abs(float(mine))), "loss reimplementation differs from pinn_model.train_one"
        return

    for seed in a.seeds:
        tag = f"lam-{a.lam}_anchor-{a.anchor}_k{a.kappa:g}_seed{seed}"
        if (out / f"{tag}.json").exists():
            print(f"[ctl] skip {tag} (done)"); continue
        seed_all(seed); M.SEED = seed
        trd = M.PatchDS(ds, tr_dl.dataset.tids, stats, tr_dl.dataset.mask)
        vad = M.PatchDS(ds, va_dl.dataset.tids, stats, va_dl.dataset.mask)
        tl = DataLoader(trd, M.BATCH, sampler=tr_dl.sampler, drop_last=True)
        vl = DataLoader(vad, M.BATCH, shuffle=False)
        net = M.ConvLSTM(len(varlist)).to(dev)
        with torch.no_grad():
            net.kappa.fill_(a.kappa)
        opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=3, factor=.5)
        best, bad, hist, t1 = 1e9, 0, [], time.time()
        for ep in range(1, a.epochs + 1):
            net.train()
            for X, y, m in tl:
                X, y, m = X.to(dev), y.to(dev), m.to(dev)
                loss, sup, phys = batch_loss(M, net, X, y, m, qthr, stats, ep, lam, a.anchor)
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); opt.zero_grad()
                with torch.no_grad():
                    net.kappa.clamp_(1e-2, 1e3)
            vr = patch_rmse(M, vl, net, stats, a.anchor, dev); sch.step(vr)
            hist.append({"epoch": ep, "val_patch_rmse": vr, "sup": float(sup), "phys": float(phys), "kappa": float(net.kappa)})
            print(f"[ctl] {tag} E{ep:02d} val={vr:.4f} sup={float(sup):.4f} phys={float(phys):.3g} kappa={float(net.kappa):.4f}", flush=True)
            if vr < best - 1e-3:
                best, bad = vr, 0; torch.save(net.state_dict(), out / f"{tag}.pt")
            else:
                bad += 1
                if bad == 6:
                    break
        net.load_state_dict(torch.load(out / f"{tag}.pt", map_location=dev))
        res = full_grid(D, net, ds, stats, varlist, a.anchor, dev)
        json.dump({"tag": tag, "seed": seed, "lam": a.lam, "anchor": a.anchor, "kappa_init": a.kappa,
                   "kappa_final": float(net.kappa), "epochs_run": len(hist), "best_val_patch_rmse": best,
                   "train_seconds": time.time() - t1, "full_grid": res, "history": hist},
                  open(out / f"{tag}.json", "w"), indent=1)
        print(f"[ctl] {tag} done: " + " ".join(f"{k}={v['rmse']:.4f}/pers {v['rmse_persistence']:.4f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
