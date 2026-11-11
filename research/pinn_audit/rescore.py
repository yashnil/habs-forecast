#!/usr/bin/env python3
"""Re-score existing ConvLSTM / PINN checkpoints under alternative evaluation protocols.

Read-only with respect to the original research: imports the model and input
builder from ``pinn/diagnostics.py`` unchanged, never writes checkpoints, and
writes only to ``--out``.

Protocols (all on the same pixels, times and checkpoints):

  published   pred = raw log_chl[t] + delta, minus a constant bias computed over
              ALL subsets (train+val+test) -- what pinn/diagnostics.py does.
  no_bias     pred = raw log_chl[t] + delta, no bias correction.
  train_bias  pred = raw log_chl[t] + delta, minus a bias fitted on train only.
  as_trained  pred = z(log_chl[t]) (NaN->0) + delta -- the quantity the training
              loop optimised and scored (pinn_model.py rmse()).

Baselines fitted on train only: persistence, pooled AR(1) (y = a*x + b), and
AR(1) on monthly per-pixel climatology anomalies.
"""
from __future__ import annotations
import argparse, importlib.util, json, pathlib, sys, time
import numpy as np, pandas as pd, torch, xarray as xr

REPO = pathlib.Path(__file__).resolve().parents[2]


def load_diag():
    spec = importlib.util.spec_from_file_location("pinn_diag", REPO / "pinn/diagnostics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def splits(times):
    t = np.asarray(times)
    return {"train": t < np.datetime64("2016-01-01"),
            "val": (t >= np.datetime64("2016-01-01")) & (t <= np.datetime64("2018-12-31")),
            "test": t > np.datetime64("2018-12-31")}


def rmse(p, y, m):
    d = (p - y)[m]
    return float(np.sqrt(np.mean(d * d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=6)
    ap.add_argument("--lead", type=int, default=1)
    args = ap.parse_args()
    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); torch.set_num_threads(8)

    D = load_diag()
    ds = xr.open_dataset(args.freeze, engine="netcdf4").load()
    pixel_ok = D.build_pixel_ok_if_needed(ds)
    sp_in = splits(ds.time.values)
    lat_any = np.where(pixel_ok.any("lon"))[0]; lon_any = np.where(pixel_ok.any("lat"))[0]
    la, lo = slice(lat_any.min(), lat_any.max() + 1), slice(lon_any.min(), lon_any.max() + 1)
    ok = pixel_ok.isel(lat=la, lon=lo).values

    core = np.arange(args.seq - 1, ds.sizes["time"] - args.lead)
    tpred = ds.time.values[core + args.lead]
    sp = splits(tpred)
    logc = ds.log_chl.isel(lat=la, lon=lo).values.astype(np.float32)
    last, truth = logc[core], logc[core + args.lead]
    mask = ok[None] & np.isfinite(truth)
    print(f"[audit] Tpred={len(core)} grid={ok.shape} masked px={int(mask.sum())}", flush=True)
    print(f"[audit] masked px with NaN last_log: {int((mask & ~np.isfinite(last)).sum())}", flush=True)

    rows, coverage = [], {}
    # ---- statistical baselines (fit on train targets only) ----------------
    trm = mask & sp["train"][:, None, None] & np.isfinite(last)
    a, b = np.polyfit(last[trm], truth[trm], 1)
    months = pd.DatetimeIndex(ds.time.values).month.values
    clim = np.full((13,) + ok.shape, np.nan, np.float32)
    tr_idx = np.where(sp_in["train"])[0]
    for mth in range(1, 13):
        sel = tr_idx[months[tr_idx] == mth]
        with np.errstate(all="ignore"):
            clim[mth] = np.nanmean(logc[sel], 0)
    c_last, c_tgt = clim[months[core]], clim[months[core + args.lead]]
    an_x, an_y = last - c_last, truth - c_tgt
    trm2 = trm & np.isfinite(an_x) & np.isfinite(an_y)
    phi = float(np.sum(an_x[trm2] * an_y[trm2]) / np.sum(an_x[trm2] ** 2))
    base = {"persistence": last, "ar1_pooled": a * last + b, "ar1_clim_anom": c_tgt + phi * an_x}
    json.dump({"ar1_pooled": [float(a), float(b)], "ar1_clim_anom_phi": phi},
              open(out / "baseline_params.json", "w"), indent=2)

    # common evaluation mask: every forecast finite (excludes NaN last_log pixels)
    def finite_all(preds):
        m = mask.copy()
        for p in preds: m &= np.isfinite(p)
        return m

    preds = dict(base)
    mu, sd = None, None
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        state = torch.load(path, map_location="cpu")
        has_kappa = "kappa" in state
        if not has_kappa:  # vanilla ConvLSTM: same graph, no physics parameter
            state = dict(state, kappa=torch.tensor(float("nan")))
        cin = state["reduce.weight"].shape[1]
        varlist = [v for v in D.ALL_VARS if v in ds.data_vars][:cin]
        stats = D.norm_stats(ds, sp_in["train"], varlist)
        mu, sd = stats["log_chl"]
        net = D.ConvLSTM(cin); net.load_state_dict(state, strict=True); net.eval()
        delta = np.empty_like(last); t0 = time.time()
        for i in range(0, len(core), 16):
            ks = core[i:i + 16]
            X = np.stack([D.make_input(ds, varlist, stats, k, args.seq, la, lo) for k in ks], 0)
            with torch.no_grad():
                delta[i:i + len(ks)] = net(torch.from_numpy(X)).numpy()
        print(f"[audit] {name}: inference {time.time()-t0:.0f}s kappa={float(state.get('kappa', torch.tensor(float('nan')))):.3g}", flush=True)
        np.save(out / f"delta_{name}.npy", delta)
        raw = last + delta
        z_last = np.nan_to_num((last - mu) / sd, nan=0.0)
        m_all = mask & np.isfinite(raw)
        bias_all = float(np.mean((raw - truth)[m_all]))
        m_tr = m_all & sp["train"][:, None, None]
        bias_tr = float(np.mean((raw - truth)[m_tr]))
        preds[f"{name}:published"] = raw - bias_all
        preds[f"{name}:no_bias"] = raw
        preds[f"{name}:train_bias"] = raw - bias_tr
        preds[f"{name}:as_trained"] = z_last + delta
        coverage[name] = {"bias_all": bias_all, "bias_train": bias_tr, "kappa": float(state["kappa"]) if has_kappa else None}

    m_common = finite_all(list(preds.values()))
    m_pub = mask & np.isfinite(last)  # what diagnostics.py effectively scores
    for key, p in preds.items():
        for sname, s in sp.items():
            sel = s[:, None, None]
            rows.append({"forecast": key, "subset": sname,
                         "rmse_log_common": rmse(p, truth, m_common & sel),
                         "rmse_log_pers_common": rmse(last, truth, m_common & sel),
                         "n_common": int((m_common & sel).sum())})
    df = pd.DataFrame(rows)
    df["skill_pct_common"] = 100 * (1 - df.rmse_log_common / df.rmse_log_pers_common)
    df.to_csv(out / "rescore.csv", index=False)
    json.dump({"log_chl_mu_sd": [mu, sd], "checkpoints": coverage,
               "n_masked": int(mask.sum()), "n_masked_nan_last": int((mask & ~np.isfinite(last)).sum()),
               "n_common": int(m_common.sum())}, open(out / "summary.json", "w"), indent=2)
    print(df.pivot(index="forecast", columns="subset", values="rmse_log_common").round(4).to_string())


if __name__ == "__main__":
    main()
