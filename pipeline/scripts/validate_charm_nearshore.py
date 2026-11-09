"""Validate "show the nearest valued C-HARM cell at a nearshore point" against CalHABMAP pier samples.

    cd pipeline && uv run python scripts/validate_charm_nearshore.py [--cache DIR] [--boot 2000]

C-HARM v3.1 masks many cells near shore (toxin variables especially), so the app shows, at a pier
with no C-HARM value, the nowcast of the nearest cell that has one. This script asks whether that
borrowed nowcast discriminates and is calibrated for what was measured in the water at the pier.

Data
- C-HARM v3.1 nowcast (lead 0), NOAA CoastWatch West Coast ERDDAP griddap wvcharmV3_0day
  (0-360 longitudes, 0.03 deg, `time` = valid day 12:00Z). Per station: a box of +/-BOX_KM around
  the station, one request per calendar year, cached under --cache.
- CalHABMAP shore stations, SCCOOS ERDDAP tabledap HABs-<station>. pDA is published in ng/mL of
  seawater (so C-HARM's 500 ng/L threshold is 0.5 ng/mL); Pseudo-nitzschia in cells/L by size group.

Matching
- A sample's day is its local calendar date (UTC-8, fixed, so a sample is never moved across
  midnight by daylight saving). Only same-day nowcasts are used; a day with no C-HARM run is
  skipped (counted, never filled from another day).
- For each sample, per variable: the nearest cell with a finite value in [0, 1] that day
  (distance in km from the sample's own reported position), and its probability.
- Several rows on one station-day are collapsed to the maximum measured value (any exceedance
  that day counts as an event).

Events
- pda_gt_500ngL : pDA > 0.5 ng/mL (= 500 ng/L, C-HARM's particulate domoic acid threshold);
                   model = particulate_domoic probability.
- pda_detected  : pDA > 0 (a reported value above 0; CalHABMAP publishes no detection limit, a
                   reported 0 means not quantified); model = particulate_domoic probability.
                   C-HARM does not forecast this event, so its Brier score is not a fair test;
                   AUC (ranking) is the meaningful number.
- pn_gt_10k     : total Pseudo-nitzschia (seriata + delicatissima groups) > 10,000 cells/L, C-HARM's
                   PN threshold; model = pseudo_nitzschia probability. Needs both groups counted,
                   except that one group alone above 10,000 cells/L is already an event.

Baseline
- Climatology: event frequency at the same station in a +/-1 month window around the sample's
  month, from the station's samples in OTHER years (leave-one-year-out), over the station's whole
  CalHABMAP record since 2014; shrunk toward the station's leave-one-year-out overall rate with a
  weight of 2 samples (and that toward the pooled leave-one-year-out rate of all stations).

Skill
- AUC (Mann-Whitney, ties averaged), Brier of the model and of climatology, BSS = 1 - Bm/Bc.
- 95% CIs by cluster bootstrap over station x calendar-month blocks (weekly samples are
  autocorrelated; resampling single samples would make the intervals too narrow).
- Verdict "skill" only when the AUC CI excludes 0.5 from above AND BSS > 0 (for pda_detected,
  where Brier is not a fair test, on AUC alone, and the verdict says so). Otherwise
  "no demonstrated skill nearshore".

This is a test of a model probability against water samples at piers. Neither describes toxin in
seafood; C-HARM probabilities are not measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.http import USER_AGENT, FetchError, fetch  # noqa: E402
from coastwatch_pipeline.sources import charm, observations  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "reference" / "charm_nearshore_validation.json"
DEFAULT_CACHE = Path(
    "/private/tmp/claude-502/-Users-yashnilmohanty-Desktop-habs-forecast/4bf1be61-568f-430e-8fec-87043cfeb314/scratchpad/charm-pier-cache"
)
KM_PER_DEG = 111.32
BOX_KM = 40.0
CLIM_START = "2014-01-01"
LOCAL_OFFSET_H = -8
PRIOR_WEIGHT = 2.0
DIST_BINS = [(0.0, 5.0, "<5 km"), (5.0, 15.0, "5-15 km"), (15.0, math.inf, ">15 km")]
PDA_THRESHOLD_NG_ML = 0.5  # 500 ng/L
PN_THRESHOLD = 10_000.0
CHARM_VARS = ("pseudo_nitzschia", "particulate_domoic")

EVENTS = {
    "pda_gt_500ngL": {
        "label": "Particulate domoic acid above 500 ng/L (0.5 ng/mL) in the pier water sample",
        "model_variable": "particulate_domoic",
        "brier_fair": True,
    },
    "pda_detected": {
        "label": "Particulate domoic acid reported above 0 (no detection limit is published; 0 = not quantified)",
        "model_variable": "particulate_domoic",
        "brier_fair": False,
    },
    "pn_gt_10k": {
        "label": "Total Pseudo-nitzschia (seriata + delicatissima groups) above 10,000 cells/L",
        "model_variable": "pseudo_nitzschia",
        "brier_fair": True,
    },
}

LOG: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG.append(msg)


# ---------------------------------------------------------------- fetching with a cache
def cached_fetch(url: str, cache: Path, suffix: str) -> bytes:
    p = cache / (hashlib.sha256(url.encode()).hexdigest()[:24] + suffix)
    if p.exists() and p.stat().st_size > 0:
        return p.read_bytes()
    body = fetch(url, timeout=120, retries=4, backoff=10.0, max_seconds=400).body
    p.write_bytes(body)
    time.sleep(1.0)  # be polite to the servers
    return body


# ---------------------------------------------------------------- CalHABMAP
def station_url(key: str) -> str:
    cols = "time,latitude,longitude,pDA,Pseudo_nitzschia_seriata_group,Pseudo_nitzschia_delicatissima_group"
    q = cols + f"&time>={CLIM_START}T00:00:00Z" + '&orderBy("time")'
    return f"{observations.SERVER}/tabledap/{observations.dataset_id(key)}.csv?" + quote(q, safe=',&=()"').replace(">=", "%3E=")


def _num(s: str) -> float | None:
    s = s.strip()
    if not s or s.lower() == "nan":
        return None
    v = float(s)
    return None if math.isnan(v) or v < 0 else v  # negatives rejected, as the pipeline does


def load_station(key: str, cache: Path) -> list[dict]:
    body = cached_fetch(station_url(key), cache, ".csv")
    text = body.decode("utf-8", "replace")
    rows = list(csv.reader(io.StringIO(text)))
    header, units = rows[0], rows[1]
    ix = {c: header.index(c) for c in header}
    expect = {"pDA": "ng/mL", "Pseudo_nitzschia_seriata_group": "cells/L", "Pseudo_nitzschia_delicatissima_group": "cells/L"}
    for c, u in expect.items():
        if units[ix[c]] != u:
            raise ValueError(f"{key}: unexpected units {c}={units[ix[c]]!r}")
    by_day: dict[date, dict] = {}
    for r in rows[2:]:
        if len(r) < len(header):
            continue
        t = datetime.fromisoformat(r[ix["time"]].replace("Z", "+00:00")).astimezone(timezone.utc)
        d = (t + timedelta(hours=LOCAL_OFFSET_H)).date()
        rec = by_day.setdefault(d, {"date": d, "lat": [], "lon": [], "pDA": None, "ser": None, "del": None})
        rec["lat"].append(float(r[ix["latitude"]]))
        rec["lon"].append(float(r[ix["longitude"]]))
        for name, col in (("pDA", "pDA"), ("ser", "Pseudo_nitzschia_seriata_group"), ("del", "Pseudo_nitzschia_delicatissima_group")):
            v = _num(r[ix[col]])
            if v is not None:
                rec[name] = v if rec[name] is None else max(rec[name], v)
    out = []
    for d in sorted(by_day):
        rec = by_day[d]
        rec["lat"], rec["lon"] = float(np.median(rec["lat"])), float(np.median(rec["lon"]))
        out.append(rec)
    return out


def events_of(rec: dict) -> dict[str, float | None]:
    p, s, dl = rec["pDA"], rec["ser"], rec["del"]
    pn: float | None
    if s is not None and dl is not None:
        pn = float(s + dl > PN_THRESHOLD)
    elif (s is not None and s > PN_THRESHOLD) or (dl is not None and dl > PN_THRESHOLD):
        pn = 1.0
    else:
        pn = None
    return {
        "pda_gt_500ngL": None if p is None else float(p > PDA_THRESHOLD_NG_ML),
        "pda_detected": None if p is None else float(p > 0),
        "pn_gt_10k": pn,
    }


# ---------------------------------------------------------------- C-HARM
def charm_coverage(cache: Path) -> tuple[date, date, int]:
    url = f"{charm.SERVER}/griddap/{charm.dataset_id(0)}.csv0?time"
    p = cache / "charm_times.csv"
    body = fetch(url, timeout=120, retries=4, backoff=10.0).body  # always fresh: the record grows daily
    p.write_bytes(body)
    times = [datetime.fromisoformat(s.strip().replace("Z", "+00:00")).date() for s in body.decode().splitlines() if s.strip()]
    return times[0], times[-1], len(times)


def charm_box_url(lat: float, lon: float, start: date, end: date) -> str:
    dlat = BOX_KM / KM_PER_DEG
    dlon = BOX_KM / (KM_PER_DEG * math.cos(math.radians(lat)))
    d = charm.DOMAIN
    la0, la1 = max(lat - dlat, d["lat_min"]), min(lat + dlat, d["lat_max"])
    lo0, lo1 = max(lon + 360 - dlon, d["lon_min"]), min(lon + 360 + dlon, d["lon_max"])
    sel = f"[({start.isoformat()}T12:00:00Z):({end.isoformat()}T12:00:00Z)][({la0:.3f}):({la1:.3f})][({lo0:.3f}):({lo1:.3f})]"
    q = ",".join(f"{v}{sel}" for v in CHARM_VARS)
    return f"{charm.SERVER}/griddap/{charm.dataset_id(0)}.nc?" + quote(q, safe=",():")


def load_charm_box(lat: float, lon: float, first: date, last: date, cache: Path, problems: list[str], label: str):
    """{date: {var: 2-D array}}, lat, lon(-180..180). One request per calendar year."""
    days: dict[date, dict[str, np.ndarray]] = {}
    LAT = LON = None
    for year in range(first.year, last.year + 1):
        s, e = max(first, date(year, 1, 1)), min(last, date(year, 12, 31))
        url = charm_box_url(lat, lon, s, e)
        try:
            blob = cached_fetch(url, cache, ".nc")
            arrays, vattrs, _ = charm.parse_netcdf(blob)
        except (FetchError, ValueError) as ex:
            problems.append(f"{label} C-HARM {year}: {str(ex)[:300]}")
            log(f"  ! {label} {year}: {str(ex)[:200]}")
            continue
        LAT = arrays["latitude"].astype(float)
        LON = arrays["longitude"].astype(float) - 360.0
        tt = [datetime.fromtimestamp(float(t), tz=timezone.utc).date() for t in np.atleast_1d(arrays["time"])]
        for v in CHARM_VARS:
            a = arrays[v].astype(float)
            fv = vattrs[v].get("_FillValue")
            if isinstance(fv, (int, float)):
                a[np.isclose(a, float(fv))] = np.nan
            a[(a < -1e-6) | (a > 1 + 1e-6)] = np.nan
            for i, d in enumerate(tt):
                days.setdefault(d, {})[v] = np.clip(a[i], 0, 1)
    return days, LAT, LON


# ---------------------------------------------------------------- statistics
def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    avg_rank = upper - (counts - 1) / 2.0  # 1-based average rank of each tie group
    r = avg_rank[inv]
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def scores(y, p, c):
    bm, bc = brier(y, p), brier(y, c)
    return auc(y, p), bm, bc, (1 - bm / bc) if bc > 0 else float("nan")


def ci(a: list[float]) -> list[float] | None:
    a = np.asarray([x for x in a if np.isfinite(x)])
    if a.size < 50:
        return None
    return [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]


def r3(x: float) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def skill_block(rows: list[dict], ev: str, n_boot: int, rng: np.random.Generator) -> dict:
    y = np.array([r["y"] for r in rows], dtype=float)
    p = np.array([r["p"] for r in rows], dtype=float)
    c = np.array([r["clim"] for r in rows], dtype=float)
    n, ne = len(rows), int(y.sum()) if rows else 0
    block = {
        "metric": EVENTS[ev]["label"],
        "model_variable": EVENTS[ev]["model_variable"],
        "n": n,
        "n_events": ne,
        "base_rate": r3(ne / n) if n else None,
        "n_stations": len({r["station"] for r in rows}),
        "stations_with_events": sorted({r["station"] for r in rows if r["y"]}),
        "dates": [min(r["date"] for r in rows).isoformat(), max(r["date"] for r in rows).isoformat()] if rows else None,
        "median_distance_km": r3(float(np.median([r["dist"] for r in rows]))) if rows else None,
    }
    if n == 0 or ne == 0 or ne == n:
        block.update(auc=None, auc_95ci=None, brier_model=None, brier_climatology=None, bss=None, bss_95ci=None,
                     note="Too few samples or no events / no non-events: not computable.")  # fmt: skip
        return block
    a, bm, bc, bss = scores(y, p, c)
    clusters = defaultdict(list)
    for i, r in enumerate(rows):
        clusters[(r["station"], r["date"].year, r["date"].month)].append(i)
    groups = [np.array(v) for v in clusters.values()]
    ba, bb = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        sa, _, _, sb = scores(y[idx], p[idx], c[idx])
        ba.append(sa)
        bb.append(sb)
    block.update(
        auc=r3(a),
        auc_95ci=ci(ba),
        brier_model=r3(bm),
        brier_climatology=r3(bc),
        bss=r3(bss),
        bss_95ci=ci(bb),
        mean_model_probability=r3(float(p.mean())),
        mean_climatology_probability=r3(float(c.mean())),
        auc_climatology=r3(auc(y, c)),
        n_clusters=len(groups),
    )
    return block


def verdict(ev: str, b: dict) -> tuple[str, str]:
    if b.get("auc") is None or b.get("auc_95ci") is None:
        return "no demonstrated skill nearshore", f"Not enough events to test ({b['n_events']} events in {b['n']} samples)."
    lo, hi = b["auc_95ci"]
    auc_ok = lo > 0.5
    ci_txt = "lies below 0.5 (ranks worse than chance)" if hi < 0.5 else "includes 0.5"
    if not EVENTS[ev]["brier_fair"]:
        if auc_ok:
            return "skill", (
                f"The nearest valued C-HARM pDA probability ranks detected-vs-not samples better than chance "
                f"(AUC {b['auc']:.2f}, 95% CI {b['auc_95ci'][0]:.2f}-{b['auc_95ci'][1]:.2f}, n={b['n']}); ranking only, "
                f"its probabilities are for a different (500 ng/L) event."
            )
        return "no demonstrated skill nearshore", (
            f"AUC {b['auc']:.2f} (95% CI {b['auc_95ci'][0]:.2f}-{b['auc_95ci'][1]:.2f}) {ci_txt}, n={b['n']}, {b['n_events']} events."
        )
    if auc_ok and b["bss"] is not None and b["bss"] > 0:
        return "skill", (
            f"AUC {b['auc']:.2f} (95% CI {b['auc_95ci'][0]:.2f}-{b['auc_95ci'][1]:.2f}) and BSS {b['bss']:.2f} vs station-month "
            f"climatology, n={b['n']} samples, {b['n_events']} events, {b['dates'][0]} to {b['dates'][1]}."
        )
    why = []
    if not auc_ok:
        why.append(f"AUC 95% CI {b['auc_95ci'][0]:.2f}-{b['auc_95ci'][1]:.2f} {ci_txt}")
    if b["bss"] is None or b["bss"] <= 0:
        why.append(f"BSS {b['bss']:.2f} <= 0 (worse than station-month climatology)" if b["bss"] is not None else "BSS not computable")
    return "no demonstrated skill nearshore", (
        f"{'; '.join(why)} (AUC {b['auc']:.2f}, n={b['n']}, {b['n_events']} events, {b['dates'][0]} to {b['dates'][1]})."
    )


# ---------------------------------------------------------------- climatology
def climatology(all_recs: dict[str, list[dict]], ev: str):
    """Returns f(station, date) -> leave-one-year-out probability."""
    # counts[(scope)][year] = [events, samples]; scope: "all", station, (station, month)
    counts: dict[object, dict[int, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for k, recs in all_recs.items():
        for r in recs:
            y = events_of(r)[ev]
            if y is None:
                continue
            for scope in ("all", k, (k, r["date"].month)):
                c = counts[scope][r["date"].year]
                c[0] += y
                c[1] += 1

    def rate(scopes, year, prior, w):
        k = n = 0.0
        for s in scopes:
            for yr, (e, m) in counts[s].items():
                if yr != year:
                    k, n = k + e, n + m
        return (k + w * prior) / (n + w)

    def f(station: str, d: date) -> float:
        pooled_rate = rate(["all"], d.year, 0.5, 1.0)
        st_rate = rate([station], d.year, pooled_rate, PRIOR_WEIGHT)
        months = [((d.month - 1 + k) % 12) + 1 for k in (-1, 0, 1)]
        return rate([(station, m) for m in months], d.year, st_rate, PRIOR_WEIGHT)

    return f


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--stations", nargs="*", default=list(observations.STATIONS))
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    problems: list[str] = []
    rng = np.random.default_rng(20261020)

    try:
        first, last, n_times = charm_coverage(args.cache)
    except FetchError as ex:
        write_not_validated(args.out, f"C-HARM time axis could not be fetched: {ex}")
        return 1
    span = (last - first).days + 1
    log(f"C-HARM nowcast record {first}..{last}: {n_times} of {span} days have a run")

    all_recs: dict[str, list[dict]] = {}
    for key in args.stations:
        try:
            all_recs[key] = load_station(key, args.cache)
            log(f"{key}: {len(all_recs[key])} sample days since {CLIM_START}")
        except Exception as ex:
            problems.append(f"{key} CalHABMAP: {str(ex)[:300]}")
            log(f"  ! {key}: {ex}")

    clim = {ev: climatology(all_recs, ev) for ev in EVENTS}
    rows: dict[str, list[dict]] = {ev: [] for ev in EVENTS}
    coverage = {}
    for key, recs in all_recs.items():
        in_period = [r for r in recs if first <= r["date"] <= last]
        cov = {"name": observations.STATIONS[key], "sample_days_in_period": len(in_period), "no_charm_run_that_day": 0,
               "no_valued_cell_within_box": {v: 0 for v in CHARM_VARS}, "matched": {v: 0 for v in CHARM_VARS}}  # fmt: skip
        coverage[key] = cov
        if not in_period:
            continue
        lat0 = float(np.median([r["lat"] for r in in_period]))
        lon0 = float(np.median([r["lon"] for r in in_period]))
        cov["position"] = [round(lat0, 4), round(lon0, 4)]
        s0, s1 = min(r["date"] for r in in_period), max(r["date"] for r in in_period)
        days, LAT, LON = load_charm_box(lat0, lon0, s0, s1, args.cache, problems, key)
        if LAT is None:
            continue
        GLA, GLO = np.meshgrid(LAT, LON, indexing="ij")
        dists = defaultdict(list)
        for r in in_period:
            grids = days.get(r["date"])
            if grids is None:
                cov["no_charm_run_that_day"] += 1
                continue
            D = np.hypot((GLA - r["lat"]) * KM_PER_DEG, (GLO - r["lon"]) * KM_PER_DEG * math.cos(math.radians(r["lat"])))
            near = {}
            for v in CHARM_VARS:
                a = grids[v]
                ok = np.isfinite(a)
                if not ok.any():
                    cov["no_valued_cell_within_box"][v] += 1
                    continue
                Dm = np.where(ok, D, np.inf)
                i = np.unravel_index(int(np.argmin(Dm)), Dm.shape)
                near[v] = (float(Dm[i]), float(a[i]))
                cov["matched"][v] += 1
                dists[v].append(float(Dm[i]))
            ys = events_of(r)
            for ev, spec in EVENTS.items():
                v = spec["model_variable"]
                if ys[ev] is None or v not in near:
                    continue
                rows[ev].append({"station": key, "date": r["date"], "y": ys[ev], "p": near[v][1], "dist": near[v][0],
                                 "clim": clim[ev](key, r["date"])})  # fmt: skip
        for v in CHARM_VARS:
            if dists[v]:
                cov[f"nearest_valued_km_{v}"] = {"median": round(float(np.median(dists[v])), 2),
                                                 "min": round(min(dists[v]), 2), "max": round(max(dists[v]), 2)}  # fmt: skip
        log(f"{key}: in period {len(in_period)}, no run {cov['no_charm_run_that_day']}, matched {cov['matched']}")

    results = {}
    for ev in EVENTS:
        rr = rows[ev]
        overall = skill_block(rr, ev, args.boot, rng)
        v, sentence = verdict(ev, overall)
        by_dist = {}
        for lo, hi, lab in DIST_BINS:
            sub = [r for r in rr if lo <= r["dist"] < hi]
            b = skill_block(sub, ev, args.boot, rng)
            b["verdict"], b["verdict_sentence"] = verdict(ev, b)
            by_dist[lab] = b
        results[ev] = {**overall, "verdict": v, "verdict_sentence": sentence, "by_distance_to_nearest_valued_cell": by_dist}
        log(f"{ev}: n={overall['n']} events={overall['n_events']} AUC={overall.get('auc')} {overall.get('auc_95ci')} "
            f"BSS={overall.get('bss')} {overall.get('bss_95ci')} -> {v}")  # fmt: skip
        for lab, b in by_dist.items():
            log(f"    {lab}: n={b['n']} ev={b['n_events']} AUC={b.get('auc')} {b.get('auc_95ci')} BSS={b.get('bss')}")

    runtime = time.monotonic() - t0
    out = {
        "status": "validated",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "description": (
            "Test of showing, at a pier/nearshore point without a C-HARM value, the same-day C-HARM v3.1 nowcast "
            "probability of the nearest cell that has a value, against CalHABMAP pier water samples. Model "
            "probabilities are compared with measured events in seawater at the pier; neither is toxin in seafood."
        ),
        "script": "pipeline/scripts/validate_charm_nearshore.py",
        "period": {"charm_record": [first.isoformat(), last.isoformat()], "charm_days_with_run": n_times,
                   "charm_days_in_span": span, "climatology_samples_from": CLIM_START},  # fmt: skip
        "data_sources": {
            "charm": {
                "dataset": f"{charm.SERVER}/griddap/{charm.dataset_id(0)}.html",
                "variables": {"particulate_domoic": "P(pDA > 500 ng/L)", "pseudo_nitzschia": "P(Pseudo-nitzschia > 10,000 cells/L)"},
                "request_url_pattern": f"{charm.SERVER}/griddap/{charm.dataset_id(0)}.nc?pseudo_nitzschia[(YYYY-MM-DDT12:00:00Z):(YYYY-MM-DDT12:00:00Z)][(lat0):(lat1)][(lon0_360):(lon1_360)],particulate_domoic[...]"
                "  (one request per station per calendar year, box +/-%.0f km)" % BOX_KM,
                "example_request_url": charm_box_url(32.867, -117.257, date(2024, 1, 1), date(2024, 12, 31)),
                "user_agent": USER_AGENT,
            },
            "calhabmap": {
                "datasets": f"{observations.SERVER}/tabledap/HABs-<station>.html ({len(all_recs)} stations)",
                "request_url_pattern": station_url("<station>"),
                "units": {"pDA": "ng/mL (500 ng/L = 0.5 ng/mL)", "Pseudo_nitzschia_*_group": "cells/L"},
            },
        },
        "method": {
            "matching": "Sample day = local calendar date (UTC-8). Same-day nowcast only; days without a C-HARM run are skipped, never filled. Nearest cell with a finite value that day, per variable, from the sample's reported position.",
            "collapse": "Several rows on one station-day: the maximum measured value of each variable.",
            "climatology": "Leave-one-year-out event frequency at the same station in a +/-1 month window, from the station's samples since 2014 in other years, shrunk (weight 2 samples) toward the station's leave-one-year-out overall rate.",
            "confidence_intervals": f"Cluster bootstrap ({args.boot} resamples) over station x calendar-month blocks.",
            "verdict_rule": "'skill' only if the AUC 95% CI lies above 0.5 and BSS > 0 vs climatology (pda_detected: AUC only, since C-HARM does not forecast that event); otherwise 'no demonstrated skill nearshore'.",
            "pda_detected_limit": "pDA > 0 ng/mL as reported; CalHABMAP publishes no detection limit.",
            "pn_total": "seriata + delicatissima groups; both must be counted unless one alone exceeds 10,000 cells/L.",
        },
        "events": results,
        "station_coverage": coverage,
        "data_problems": problems,
        "runtime_s": round(runtime, 1),
        "caveats": [
            "C-HARM v3.1 is a model probability, not a measurement; CalHABMAP values describe one water sample at one pier on one day.",
            "Neither the model nor the samples describe domoic acid in seafood; closures and advisories are CDFW/CDPH decisions.",
            "Samples are weekly or less and autocorrelated; CIs use block bootstrap but remain approximate.",
            "The test covers the C-HARM record on ERDDAP only (v3.1 nowcast, from 2022-11-01) and only the CalHABMAP piers.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    log(f"wrote {args.out} in {runtime:.0f} s")
    return 0


def write_not_validated(path: Path, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": "not_validated", "reason": reason[:500],
                                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}, indent=2) + "\n")  # fmt: skip
    log(f"not validated: {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
