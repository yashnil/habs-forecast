"""Build data/reference/chl_outlook_reference.json: climatology + AR(1) for the 8-day chlorophyll outlook,
and measure its skill on held-out years with the same protocol as the research lead experiment.

    python3 pipeline/scripts/build_chl_outlook_reference.py --cube PATH/HAB_convLSTM_core_v1_clean.nc

Needs xarray + netCDF4 (not pipeline dependencies: this runs once, offline, by a person; the
pipeline only reads the JSON it writes).

Input: the research training cube (MODIS-Aqua L3 8-day composites, 4 km, 2003-01-01..2021-06-26,
log_chl = ln(max(chl, 0.0566)), footprint `ocean_mask_static` = pixels ever valid). SHA-256 and
origin: ~/Documents/CoastWatch-research-archive/convlstm/README.md.

What the outlook forecasts: the MEAN chlorophyll of the next 8-day composite (days 1-8 after the
predictor composite), never a day-8 map and never toxin.

Protocol (same as scratchpad/science/scripts/lead_experiment.py and research/pinn_audit):
- train 2003-2015, validation 2016-2018, test 2019-2021-06; everything fitted on train only;
- target: log_chl of composite t+k; predictors: composite t only (causal);
- baselines: persistence (composite t), climatology of the target period, climatology + AR(1);
- variants chosen on VALIDATION only (climatology: monthly, 8-day period, 8-day period smoothed;
  AR(1): one coefficient or per pixel shrunk to it); the chosen one is scored once on TEST;
- 95% CI of RMSE differences by block bootstrap (blocks of 6 composites, 1000 draws).
The shipped reference refits the chosen climatology on all years 2003-2021-06; the AR(1)
coefficients shipped are the train-fitted ones that were tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "reference" / "chl_outlook_reference.json"
LN10 = float(np.log(10.0))
NP = 46  # 8-day periods per year (period p starts on day-of-year 8p+1)


def period_of(times: np.ndarray) -> np.ndarray:
    doy = (times.astype("datetime64[D]") - times.astype("datetime64[Y]")).astype(int)
    return np.minimum(doy // 8, NP - 1)


def clim_fit(C: np.ndarray, keys: np.ndarray, sel: np.ndarray, n: int, smooth: bool) -> np.ndarray:
    out = np.stack([np.nanmean(C[sel & (keys == k)], 0) for k in range(n)])
    if smooth:  # circular 1-2-1 over adjacent periods
        out = 0.25 * np.roll(out, 1, 0) + 0.5 * out + 0.25 * np.roll(out, -1, 0)
    return out


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e**2)))


def boot_ci(se_a: np.ndarray, se_b: np.ndarray, rng: np.random.Generator) -> list[float]:
    """95% CI of RMSE(a) - RMSE(b); se_* are per-composite mean squared errors."""
    n_t = len(se_a)
    nb = int(np.ceil(n_t / 6))
    d = []
    for _ in range(1000):
        st = rng.integers(0, n_t - 5, nb)
        idx = np.concatenate([np.arange(s, s + 6) for s in st])[:n_t]
        d.append(np.sqrt(se_a[idx].mean()) - np.sqrt(se_b[idx].mean()))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main() -> None:
    import xarray as xr

    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--results", default=None, help="also write the full experiment results here")
    a = ap.parse_args()
    sha = hashlib.sha256(Path(a.cube).read_bytes()).hexdigest()
    d = xr.open_dataset(a.cube)
    m = d.ocean_mask_static.values.astype(bool)
    times = d.time.values
    t_d = times.astype("datetime64[D]")
    T = len(times)
    C = d.log_chl.values.astype(np.float64)[:, m]  # (T, P) ln(mg m-3), finite in the footprint
    P = C.shape[1]
    train = t_d < np.datetime64("2016-01-01")
    val = (t_d >= np.datetime64("2016-01-01")) & (t_d <= np.datetime64("2018-12-31"))
    test = t_d > np.datetime64("2018-12-31")
    months = times.astype("datetime64[M]").astype(int) % 12
    periods = period_of(times)
    rng = np.random.default_rng(0)

    variants = {
        "monthly": (months, 12, False),
        "8day": (periods, NP, False),
        "8day_smoothed": (periods, NP, True),
    }
    results: dict = {"cube_sha256": sha, "n_pixels": int(P), "protocol": "train 2003-2015, val 2016-2018, test 2019-2021-06; fitted on train"}
    fits = {}
    for name, (keys, n, sm) in variants.items():
        cl = clim_fit(C, keys, train, n, sm)
        fits[name] = (cl, keys)

    def evaluate(split: np.ndarray, k: int, cname: str, phi_mode: str) -> dict:
        cl, keys = fits[cname]
        Cl = cl[keys]
        A = C - Cl
        t0 = np.arange(0, T - k)
        tr = t0[train[t0] & train[t0 + k]]
        x, y = A[tr].ravel(), A[tr + k].ravel()
        phi = float((x * y).sum() / (x * x).sum())
        if phi_mode == "per_pixel":
            xp, yp = A[tr], A[tr + k]
            num, den = (xp * yp).sum(0), (xp * xp).sum(0)
            lam = den.mean() * 0.5  # shrink toward the pooled phi
            phi_px = (num + lam * phi) / (den + lam)
        else:
            phi_px = np.full(P, phi)
        ts = t0[split[t0] & split[t0 + k]]
        y = C[ts + k]
        preds = {"persistence": C[ts], "climatology": Cl[ts + k], "clim+AR1": Cl[ts + k] + phi_px[None] * A[ts]}
        out = {"phi": round(phi, 4), "n_pixel_times": int(y.size), "n_composites": int(len(ts)),
               "period": f"{str(t_d[ts[0] + k])}..{str(t_d[ts[-1] + k])} (target composites)"}
        se = {}
        for nm, p in preds.items():
            e = p - y
            out[nm] = {"rmse_ln": round(rmse(e), 4), "bias_ln": round(float(e.mean()), 4)}
            se[nm] = (e**2).mean(1)
        for nm in ("persistence", "climatology"):
            out[nm]["minus_climAR1_95ci"] = boot_ci(se[nm], se["clim+AR1"], rng)
        out["_se"] = se
        out["_phi_px"] = phi_px
        out["_ts"] = ts
        return out

    # 1. choose variants on validation (lead 1 = 8 days)
    choice = {}
    for cname in variants:
        for pm in ("global", "per_pixel"):
            r = evaluate(val, 1, cname, pm)
            choice[f"{cname}|{pm}"] = r["clim+AR1"]["rmse_ln"]
            results.setdefault("validation_choice", {})[f"{cname}|{pm}"] = {k: v for k, v in r.items() if not k.startswith("_")}
    best = min(choice, key=choice.get)
    cname, pm = best.split("|")
    # the research baseline configuration, for comparison on test
    results["chosen"] = {"climatology": cname, "ar1": pm, "val_rmse_ln": choice[best]}
    print("validation clim+AR1 RMSE:", json.dumps(choice), "-> chosen", best, flush=True)

    # 2. score the chosen variant once on test, leads 1..4 (8..32 days); and the research config
    for k in (1, 2, 3, 4):
        r = evaluate(test, k, cname, pm)
        if k == 1:
            ref = evaluate(test, 1, "monthly", "global")
            r["research_config_monthly_global"] = {"clim+AR1_rmse_ln": ref["clim+AR1"]["rmse_ln"], "climatology_rmse_ln": ref["climatology"]["rmse_ln"]}
            r["chosen_minus_research_95ci"] = boot_ci(r["_se"]["clim+AR1"], ref["_se"]["clim+AR1"], rng)
            # per-pixel expected error (val + test, never train) for the uncertainty grid
            both = val | test
            rr = evaluate(both, 1, cname, pm)
            cl, keys = fits[cname]
            Cl = cl[keys]
            A = C - Cl
            ts = rr["_ts"]
            e_ar = Cl[ts + 1] + rr["_phi_px"][None] * A[ts] - C[ts + 1]
            e_cl = Cl[ts + 1] - C[ts + 1]
            px_rmse_ar = np.sqrt((e_ar**2).mean(0))
            px_rmse_cl = np.sqrt((e_cl**2).mean(0))
            phi_px_1 = rr["_phi_px"]
        results[f"test_lead_{k * 8}d"] = {kk: v for kk, v in r.items() if not kk.startswith("_")}
        print(k * 8, "d test:", json.dumps({kk: v for kk, v in r.items() if not kk.startswith("_")}), flush=True)
    phis = [results[f"test_lead_{k * 8}d"]["phi"] for k in (1, 2, 3, 4)]

    # 3. the shipped reference: chosen climatology refitted on all years; tested AR(1) coefficients
    keys_all, n, sm = variants[cname]
    cl_all = clim_fit(C, keys_all, np.ones(T, bool), n, sm)
    if n == 12:  # expand to 8-day periods: month of each period's first day
        first = (np.datetime64("2001-01-01") + np.arange(NP) * 8).astype("datetime64[M]").astype(int) % 12
        cl_all = cl_all[first]
    rows, cols = np.nonzero(m)
    lat, lon = d.lat.values, d.lon.values
    test1 = results["test_lead_8d"]
    ref = {
        "description": (
            "Per-pixel 8-day-period chlorophyll climatology (ln mg m-3) and AR(1) coefficients for the Neritic 8-day "
            "chlorophyll outlook (mean of the next 8 days). Built by pipeline/scripts/build_chl_outlook_reference.py "
            "from the research training cube."
        ),
        "source": "MODIS-Aqua L3 8-day chlorophyll-a (OCI), 4 km, 2003-01-01..2021-06-26, research cube HAB_convLSTM_core_v1_clean.nc",
        "source_sha256": sha,
        "native_resolution": "4 km (1/24 degree) MODIS-Aqua L3 mapped grid",
        "units": "ln(mg m-3); ln(max(chl, 0.0566)) as in the cube",
        "grid": {"lat_first": float(lat[0]), "lat_step": float(lat[1] - lat[0]), "lon_first": float(lon[0]), "lon_step": float(lon[1] - lon[0]), "height": int(m.shape[0]), "width": int(m.shape[1])},
        "pixels": [[int(r), int(c)] for r, c in zip(rows, cols)],
        "periods": NP,
        "period_rule": "period p covers day-of-year 8p+1 .. 8p+8 (the last period of a year is 5-6 days)",
        "climatology_variant": cname,
        "climatology_years": "2003-2021-06 (all years); skill below was measured with a 2003-2015 fit",
        "climatology_ln": [[round(float(v), 3) for v in cl_all[:, i]] for i in range(P)],
        "ar1_mode": pm,
        "ar1_phi_by_lag_days": {str(8 * k): round(phis[k - 1], 4) for k in (1, 2, 3, 4)},
        "ar1_phi_pixel_8d": [round(float(v), 4) for v in phi_px_1] if pm == "per_pixel" else None,
        "expected_rmse_ln_climAR1": [round(float(v), 3) for v in px_rmse_ar],
        "expected_rmse_ln_climatology": [round(float(v), 3) for v in px_rmse_cl],
        "expected_rmse_basis": "per-pixel RMSE on 2016-2021-06 (validation + test, never training), 8-day lead",
        "skill_8d": {
            "metric": "RMSE ln(mg m-3)",
            "model": test1["clim+AR1"]["rmse_ln"],
            "climatology": test1["climatology"]["rmse_ln"],
            "persistence": test1["persistence"]["rmse_ln"],
            "persistence_minus_model_95ci": test1["persistence"]["minus_climAR1_95ci"],
            "climatology_minus_model_95ci": test1["climatology"]["minus_climAR1_95ci"],
            "test_period": test1["period"],
            "n": test1["n_pixel_times"],
            "n_composites": test1["n_composites"],
        },
        "skill_by_lead": {f"{8 * k}d": {nm: results[f"test_lead_{8 * k}d"][nm]["rmse_ln"] for nm in ("clim+AR1", "climatology", "persistence")} for k in (1, 2, 3, 4)},
    }
    OUT.write_text(json.dumps(ref, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    if a.results:
        Path(a.results).write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
