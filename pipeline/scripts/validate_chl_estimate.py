"""Held-out validation of the gap-filled chlorophyll estimate (coastwatch_pipeline.derived.estimate)
on recent daily Sentinel-3 OLCI 300 m data. Writes data/reference/chl_estimate_validation.json, which
the pipeline reads for the per-class uncertainty grid and the `validation` block of v1/derived/index.json.

    uv run python scripts/validate_chl_estimate.py --stacks DIR

DIR holds olci_<region>_stack.npz files: daily OLCI chlorophyll (mg m-3; Sentinel-3A first,
Sentinel-3B filling its gaps, as the pipeline mosaics them) for six coastal regions on the OLCI
lattice, 2026-08-19..2026-10-13 (56 days), from NOAA CoastWatch ERDDAP
noaacwS3{A,B}OLCIchlaSector{CI,DI}Daily. They were built by the 2026-10-19 science study
(scratchpad science/scripts/fetch.py + load.py); arrays: M (days, rows, cols), days, lat, lon.
Only M, days, lat and lon are used: water, distance to shore and bays come from the pipeline's
own masks, cropped to each region, so the validation sees exactly the production geometry.

Protocol (causal, as a live run would see the data):
- target days: from day 21 on (three weeks of history first), with >40% of the water within 3 km
  of shore observed; up to 10 per region, evenly spread;
- cloud scenario: on target day d, hide the valid pixels where another random day was cloudy
  (realistic gap shapes) for the whole 7-day window d-6..d, so they are gaps of the latest clear
  view, exactly the pixels the estimate fills;
- never-observed scenario: hide every water pixel within 0.6 km of shore on ALL days (pixels the
  satellite never sees), so the estimate must borrow its background from farther out;
- the background state is replayed day by day through d with the hidden values removed; the
  latest clear view is d-6..d with them removed; the estimate is scored against day d's real
  observation at each hidden pixel (log10 mg m-3);
- baselines: 'older view' (the pixel's last valid observation before the 7-day window, up to 30 days
  old: what an "include older views" option would show); 'background only' (no same-period
  anomaly); 'nearest observed' (copy the nearest pixel of the latest clear view). The app itself
  shows nothing in these pixels (no observation within 7 days).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.derived import estimate as E  # noqa: E402
from coastwatch_pipeline.sources import satellite  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "reference" / "chl_estimate_validation.json"
REGIONS = ["monterey", "sfbay", "tomales", "humboldt", "sandiego", "sbchannel"]
WINDOW = 7
OLDER_MAX = 30
MIN_N = 200
# clouds; and the satellite never seeing the water within 0.3 km (the outermost pixel row, the
# realistic case: 21-58% of it is never valid), within 0.6 km (a harder test) or within 2 km (a
# fog-bound coast: tests the wide borrowing tier)
SCENARIOS = ("clouds", "never_observed_0.3", "never_observed_0.6", "never_observed_2.0")
NEAR = "0-0.6 km"


def crop_index(target, lat: np.ndarray, lon: np.ndarray) -> tuple[slice, slice]:
    r = np.round((target.lat_first - lat) / target.step).astype(int)
    c = np.round((lon - target.lon_first) / target.step).astype(int)
    assert np.allclose(target.lat_first - r * target.step, lat, atol=1e-4), "stack is not on the OLCI lattice"
    assert np.all(np.diff(r) == 1) and np.all(np.diff(c) == 1)
    return slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacks", required=True)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    target = satellite.target_for(satellite.OLCI, satellite.CALIFORNIA)
    lat_mid = target.lat_first - target.step * target.height / 2
    dy, dx = E.cell_km(target.step, lat_mid)
    water_all = satellite.marine_water(target)
    scope_all = satellite.coastal_mask(target)
    dist_all = E.distance_km(water_all, target.step, lat_mid)
    bay_all = E.bay_mask(water_all, target.step, lat_mid)

    rows: list[tuple] = []  # (region, scenario, class, method, sq_err_sum, err_sum, n)
    days_used: dict[str, list[str]] = {}
    span = [None, None]
    for region in REGIONS:
        z = np.load(Path(a.stacks) / f"olci_{region}_stack.npz")
        rs, cs = crop_index(target, z["lat"], z["lon"])
        water, dist, bay, scope = water_all[rs, cs], dist_all[rs, cs], bay_all[rs, cs], scope_all[rs, cs]
        fill_area = water & scope
        with np.errstate(divide="ignore", invalid="ignore"):
            M = np.log10(np.where(z["M"] > 0, z["M"], np.nan))
        M[:, ~fill_area] = np.nan
        days = [date.fromisoformat(str(x)[:10]) for x in z["days"]]
        span[0] = min(span[0] or days[0], days[0])
        span[1] = max(span[1] or days[-1], days[-1])
        T = len(days)
        near = fill_area & (dist <= 3)
        cand = [d for d in range(21, T) if np.isfinite(M[d][near]).mean() > 0.4]
        if len(cand) > 10:
            cand = [cand[i] for i in np.linspace(0, len(cand) - 1, 10).astype(int)]
        days_used[region] = [days[d].isoformat() for d in cand]
        for d in cand:
            for scen in SCENARIOS:
                valid = np.isfinite(M[d])
                if scen == "clouds":
                    other = rng.choice([o for o in range(T) if o != d])
                    hide = valid & ~np.isfinite(M[other])
                    first_hidden = d - WINDOW + 1
                else:
                    hide = fill_area & (dist <= float(scen.split("_")[-1]))
                    first_hidden = 0
                if (hide & valid).sum() < 30:
                    continue
                X = M.copy()
                X[first_hidden : d + 1, hide] = np.nan
                b = np.full(M.shape[1:], np.nan)
                w = np.zeros(M.shape[1:])
                prev = None
                for k in range(d + 1):
                    b, w = E.update_background(b, w, (days[k] - prev).days if prev else 0.0, X[k])
                    prev = days[k]
                comp = np.full(M.shape[1:], np.nan)
                for k in range(d, d - WINDOW, -1):
                    f = np.isnan(comp) & np.isfinite(X[k])
                    comp[f] = X[k][f]
                est, code = E.estimate(comp, b, w, fill_area, dist, bay, dy, dx, nearshore_offset={"open": 0.0, "bay": 0.0})
                # baselines
                older = np.full(M.shape[1:], np.nan)
                for k in range(d - WINDOW, -1, -1):
                    if (days[d] - days[k]).days > OLDER_MAX:
                        break
                    f = np.isnan(older) & np.isfinite(X[k])
                    older[f] = X[k][f]
                own = np.isfinite(b) & (w >= E.MIN_WEIGHT)
                bg_only, sup = E.gauss_avg(np.where(own, b, np.nan), own, E.BACKGROUND_L_KM, dy, dx)
                bg_only = np.where(own, b, np.where(sup >= E.BACKGROUND_MIN_SUPPORT, bg_only, np.nan))
                _, (ii, jj) = distance_transform_edt(~np.isfinite(comp), return_indices=True)
                nearest = comp[ii, jj]
                sel = hide & valid
                truth = M[d][sel]
                cls = code[sel]
                preds = {"estimate": est[sel], "older_view": older[sel], "background_only": bg_only[sel], "nearest_observed": nearest[sel]}
                for c in np.unique(cls):
                    if c == 0:
                        continue
                    m_c = cls == c
                    for name, p in preds.items():
                        e = (p - truth)[m_c]
                        ok = np.isfinite(e)
                        rows.append((region, scen, E.CLASS_KEYS[c - 1], name, float((e[ok] ** 2).sum()), float(e[ok].sum()), int(ok.sum()), int(m_c.sum()), days[d].isoformat()))
                # coverage: share of hidden-and-scorable pixels the estimate fills
                rows.append((region, scen, "_all", "_coverage", 0.0, 0.0, int(np.isfinite(est[sel]).sum()), int(sel.sum()), days[d].isoformat()))
        print(region, "target days", len(cand), flush=True)

    # Nearshore offset for borrowed backgrounds: chlorophyll rises toward shore, so a background
    # borrowed from farther out reads low. The offset (per open coast / bay) is the mean bias of the
    # realistic never-observed scenario, regions weighted equally; every score below applies, for
    # each region, the offset fitted on the OTHER regions (leave one region out).
    def region_bias(region: str | None, typ: str) -> float:
        rr = [r for r in rows if r[1] == "never_observed_0.3" and r[3] == "estimate" and r[2] == f"{NEAR}|{typ}|borrowed" and (region is None or r[0] == region)]
        n = sum(r[6] for r in rr)
        return sum(r[5] for r in rr) / n if n else float("nan")

    offsets_loro: dict[tuple[str, str], float] = {}
    shipped: dict[str, float] = {}
    for typ in ("open", "bay"):
        per = {reg: region_bias(reg, typ) for reg in REGIONS}
        have = {k: v for k, v in per.items() if np.isfinite(v)}
        shipped[typ] = round(-float(np.mean(list(have.values()))), 3) if have else 0.0
        for reg in REGIONS:
            others = [v for k, v in have.items() if k != reg]
            offsets_loro[(reg, typ)] = -float(np.mean(others)) if others else 0.0

    def shift(r: tuple) -> float:
        if r[3] in ("estimate", "background_only") and r[2].startswith(NEAR) and "|borrowed" in r[2]:
            return offsets_loro[(r[0], r[2].split("|")[1])]
        return 0.0

    def agg(filt) -> dict:
        out = {}
        for name in ("estimate", "older_view", "background_only", "nearest_observed"):
            rr = [r for r in rows if r[3] == name and filt(r)]
            n = sum(r[6] for r in rr)
            if n:
                sq = sum(r[4] + 2 * shift(r) * r[5] + r[6] * shift(r) ** 2 for r in rr)
                se = sum(r[5] + r[6] * shift(r) for r in rr)
                out[name] = {"rmse_log10": round(float(np.sqrt(sq / n)), 3), "bias_log10": round(se / n, 3), "n": n}
        return out

    by_class = {}
    for key in E.CLASS_KEYS:
        r = agg(lambda r, key=key: r[2] == key)
        if r:
            by_class[key] = r
    band = lambda r, lab: r[2].split("|")[0] == lab  # noqa: E731
    summary = {
        "clouds": {
            "0-0.6 km": agg(lambda r: r[1] == "clouds" and band(r, NEAR) and "|open|" in r[2]),
            "0.6-3 km": agg(lambda r: r[1] == "clouds" and band(r, "0.6-3 km") and "|open|" in r[2]),
            "over 3 km": agg(lambda r: r[1] == "clouds" and band(r, "over 3 km") and "|open|" in r[2]),
            "bays": agg(lambda r: r[1] == "clouds" and "|bay|" in r[2]),
        },
    }
    for scen in SCENARIOS[1:]:
        summary[scen] = {
            "0-0.6 km": agg(lambda r, scen=scen: r[1] == scen and "|open|" in r[2]),
            "bays": agg(lambda r, scen=scen: r[1] == scen and "|bay|" in r[2]),
        }
    by_region = {}
    for region in REGIONS:
        for scen in SCENARIOS:
            for lab in ("0-0.6 km", "0.6-3 km", "over 3 km"):
                for typ in ("open", "bay"):
                    r = agg(lambda r, region=region, scen=scen, lab=lab, typ=typ: r[0] == region and r[1] == scen and band(r, lab) and f"|{typ}|" in r[2])
                    if r:
                        by_region.setdefault(region, {}).setdefault(scen, {})[f"{lab} {typ}"] = r
    cov = {}
    for scen in SCENARIOS:
        rr = [r for r in rows if r[3] == "_coverage" and r[1] == scen]
        cov[scen] = round(sum(r[6] for r in rr) / max(1, sum(r[7] for r in rr)), 3)
    # uncertainty per class: measured RMSE (never-observed classes: the harder 0.6 km scenario
    # counts too); classes measured on fewer than MIN_N pixels take the largest measured RMSE of
    # their band (conservative), or the largest of any class
    unc = {}
    largest = max(v["estimate"]["rmse_log10"] for v in by_class.values() if "estimate" in v)
    for key in E.CLASS_KEYS:
        v = by_class.get(key, {}).get("estimate")
        if v and v["n"] >= MIN_N:
            unc[key] = {"rmse_log10": v["rmse_log10"], "n": v["n"], "basis": "measured"}
        else:
            measured = {k: by_class[k]["estimate"]["rmse_log10"] for k in by_class if "estimate" in by_class[k] and by_class[k]["estimate"]["n"] >= MIN_N}
            same_tier = [x for k, x in measured.items() if k.split("|")[2] == key.split("|")[2]]
            same_band = [x for k, x in measured.items() if k.split("|")[0] == key.split("|")[0]]
            if same_tier:
                unc[key] = {"rmse_log10": max(same_tier + ([v["rmse_log10"]] if v else [])), "n": v["n"] if v else 0, "basis": "too few validation pixels: largest measured RMSE of its background tier"}
            elif same_band:
                unc[key] = {"rmse_log10": max(same_band), "n": v["n"] if v else 0, "basis": "too few validation pixels: largest measured RMSE of its band"}
            else:
                unc[key] = {"rmse_log10": largest, "n": v["n"] if v else 0, "basis": "not measured: largest RMSE measured in any class"}
    wins = 0
    cases = 0
    for region, sc in by_region.items():
        for lab, r in sc.get("clouds", {}).items():
            if "estimate" in r and "older_view" in r and r["estimate"]["n"] >= MIN_N:
                cases += 1
                wins += r["estimate"]["rmse_log10"] < r["older_view"]["rmse_log10"]
    result = {
        "description": "Held-out validation of the gap-filled chlorophyll estimate (log10 mg m-3). Built by pipeline/scripts/validate_chl_estimate.py.",
        "data": f"Sentinel-3 OLCI 300 m daily chlorophyll (NOAA CoastWatch ERDDAP), {span[0]}..{span[1]}, regions: {', '.join(REGIONS)}",
        "target_days": days_used,
        "nearshore_offset_log10": shipped,
        "nearshore_offset_rule": "added to borrowed backgrounds within 0.6 km of shore; mean bias of the never-observed (0.3 km) scenario, regions weighted equally; all scores apply leave-one-region-out offsets",
        "parameters": {"tau_days": E.TAU_DAYS, "min_weight": E.MIN_WEIGHT, "background_l_km": E.BACKGROUND_L_KM, "anomaly_l_km": E.ANOMALY_L_KM, "bay_open_km": E.BAY_OPEN_KM},
        "summary": summary,
        "fill_rate_of_hidden": cov,
        "estimate_beats_older_view": f"{wins} of {cases} region x band cases (cloud scenario, n >= {MIN_N})",
        "by_class": by_class,
        "uncertainty_by_class": unc,
        "by_region": by_region,
    }
    Path(a.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("summary", "fill_rate_of_hidden", "estimate_beats_older_view")}, indent=1))


if __name__ == "__main__":
    main()
