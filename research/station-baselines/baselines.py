"""Station-level baselines: how much does a shore station's own past say about its next sample?

Research only (docs/coastwatch/v3/00-scope-and-dependency.md): nothing here is wired into the app.

Question. Before any learned model (C-HARM, the PINN, a ConvLSTM or an assistant) is trusted to say
something about the next pier sample, it should beat what the station's own record already says.
This scores three baselines on the CalHABMAP series that Neritic publishes:

  base      the station's training-period exceedance rate (one number per station);
  season    the station's training-period rate in the same months (the sample's month +/- 1);
  persist   P(next exceeds | this one exceeds / does not), estimated per station in training
            (a two-state Markov chain), i.e. "it stays as it is" with calibrated odds.

Events: particulate domoic acid >= 0.5 ng/mL (C-HARM's 500 ng/L threshold) and Pseudo-nitzschia
(seriata + delicatissima) >= 10,000 cells/L. A forecast is for the station's next sample of that
variable 3-14 days later (weekly sampling; longer gaps are not "the next week" and are skipped).
Temporal holdout: fit on samples before 2022-01-01, score from 2022-01-01 on. Scores: Brier, Brier
skill against `base`, ROC AUC (Mann-Whitney), with event counts. Stdlib only:

    python3 -I research/station-baselines/baselines.py [observations.json] > out.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[2] / "coastwatch-web/tests/fixture-data/v1/observations-6d325da89a.json"
SPLIT = date(2022, 1, 1)
EVENTS = {"pDA": (["pDA"], 0.5, "pDA >= 0.5 ng/mL"), "pn": (["pn_seriata", "pn_delicatissima"], 10_000.0, "Pseudo-nitzschia >= 10,000 cells/L")}


def samples(st: dict, vars_: list[str]) -> list[tuple[date, float]]:
    series = [s for s in st["series"] if s["variable"] in vars_]
    out = []
    for i, t in enumerate(st["sample_times"]):
        tot = None
        for s in series:
            v = s["values"][i]
            if v is None or (s.get("qualifiers") or {}).get(str(i)) == "rejected_negative":
                continue
            tot = (tot or 0.0) + v
        if tot is not None:
            out.append((date.fromisoformat(t[:10]), tot))
    return out


def pairs(xs: list[tuple[date, float]], thr: float):
    """(date of the issuing sample, event now, event at the next sample) for 3-14 day steps."""
    for (d0, v0), (d1, v1) in zip(xs, xs[1:]):
        if 3 <= (d1 - d0).days <= 14:
            yield d0, d1, v0 >= thr, v1 >= thr


def season(m: int) -> set[int]:
    return {(m + 10) % 12 + 1, m, m % 12 + 1}


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def brier(p: list[float], y: list[bool]) -> float:
    return sum((a - b) ** 2 for a, b in zip(p, y)) / len(p)


def main(path: Path) -> None:
    obs = json.loads(path.read_text())
    print(f"# Station baselines\n\nData: `{path.name}` (generated {obs.get('generated_at')}); split {SPLIT}; next sample 3–14 days later.\n")
    for key, (vars_, thr, label) in EVENTS.items():
        pooled = defaultdict(list)
        rows = []
        for st in obs["stations"]:
            xs = samples(st, vars_)
            pr = list(pairs(xs, thr))
            train = [p for p in pr if p[0] < SPLIT]
            test = [p for p in pr if p[0] >= SPLIT]
            if len(train) < 50 or len(test) < 20:
                continue
            # Laplace-smoothed rates, fitted on training pairs only
            rate = lambda ys: (sum(ys) + 0.5) / (len(ys) + 1)  # noqa: E731
            base = rate([y1 for *_, y1 in train])
            by_m = {m: rate([y1 for _, d1, _, y1 in train if d1.month in season(m)]) for m in range(1, 13)}
            trans = {s: rate([y1 for _, _, y0, y1 in train if y0 == s]) for s in (False, True)}
            y = [y1 for *_, y1 in test]
            f = {"base": [base] * len(test), "season": [by_m[d1.month] for _, d1, _, _ in test], "persist": [trans[y0] for _, _, y0, _ in test]}
            for k, v in f.items():
                pooled[k] += v
            pooled["y"] += y
            b0 = brier(f["base"], y)
            rows.append((st["name"], len(train), len(test), sum(y), {k: (brier(v, y), auc(v, y)) for k, v in f.items()}, b0))
        print(f"## {label}\n")
        print("| Station | train pairs | test pairs | test events | Brier base | BSS season | BSS persist | AUC season | AUC persist |")
        print("|---|---|---|---|---|---|---|---|---|")
        fmt = lambda x: "—" if x is None else f"{x:.2f}"  # noqa: E731
        for name, ntr, nte, ev, sc, b0 in rows:
            bss = lambda k: None if b0 == 0 else 1 - sc[k][0] / b0  # noqa: E731
            print(f"| {name} | {ntr} | {nte} | {ev} | {b0:.3f} | {fmt(bss('season'))} | {fmt(bss('persist'))} | {fmt(sc['season'][1])} | {fmt(sc['persist'][1])} |")
        y = pooled["y"]
        if y:
            b0 = brier(pooled["base"], y)
            print(f"| **pooled** | | {len(y)} | {sum(y)} | {b0:.3f} | {fmt(1 - brier(pooled['season'], y) / b0)} | {fmt(1 - brier(pooled['persist'], y) / b0)} | {fmt(auc(pooled['season'], y))} | {fmt(auc(pooled['persist'], y))} |")
        print()


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
