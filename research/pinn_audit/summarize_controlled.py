#!/usr/bin/env python3
"""Summarise controlled.py runs: per arm mean ± sd over seeds, and paired (same-seed) differences."""
import json, pathlib, statistics as st, sys
from collections import defaultdict

runs = [json.load(open(p)) for p in sorted(pathlib.Path(sys.argv[1]).glob("*.json"))]
arms = defaultdict(dict)
for r in runs:
    arms[(r["lam"], r["anchor"])][r["seed"]] = r
def ms(xs): return f"{st.mean(xs):.4f} ± {st.stdev(xs):.4f}" if len(xs) > 1 else f"{xs[0]:.4f}"
rows = {}
print("| Arm | Seeds | Test RMSE | Val RMSE | Epochs run | κ final |\n|---|---|---|---|---|---|")
for (lam, anc), by in sorted(arms.items()):
    t = [r["full_grid"]["test"]["rmse"] for r in by.values()]; v = [r["full_grid"]["val"]["rmse"] for r in by.values()]
    rows[(lam, anc)] = by
    print(f"| λ={lam}, anchor={anc} | {len(by)} | {ms(t)} | {ms(v)} | {', '.join(str(r['epochs_run']) for r in by.values())} | "
          f"{', '.join(f'{r['kappa_final']:.3f}' for r in by.values())} |")
print("\nPaired same-seed test differences:")
for a, b in [(("adaptive", "z"), ("0", "z")), (("adaptive", "raw"), ("0", "raw")), (("0", "raw"), ("0", "z")), (("adaptive", "raw"), ("adaptive", "z"))]:
    if a in rows and b in rows:
        common = sorted(set(rows[a]) & set(rows[b]))
        d = [rows[a][s]["full_grid"]["test"]["rmse"] - rows[b][s]["full_grid"]["test"]["rmse"] for s in common]
        print(f"- {a} − {b}: " + ", ".join(f"{x:+.5f}" for x in d) + f" (seeds {common})")
pers = runs[0]["full_grid"]["test"]["rmse_persistence"] if runs else None
print(f"\nPersistence test RMSE: {pers:.4f}")
