"""Offline evaluation: golden questions (end to end) and adversarial answers (guard only).

Golden metrics, per case and aggregated:
  tool_recall      expected tools were called
  evidence_recall  required evidence ids are cited in the final answer
  text_ok          required patterns present, forbidden patterns absent
  outcome_ok       abstained when it should (and only then)
  first_pass       the provider's first draft passed the guard (no revision)
  final_guard      final answer passes the guard (fallback guarantees this)
Adversarial metrics: recall per violation code, false positives on controls.
"""
from __future__ import annotations

import json
import pathlib
import re

from .agent import run
from .guard import check
from .providers import MockProvider, ScriptedProvider
from .tools import Toolbox

EVALS = pathlib.Path(__file__).resolve().parent.parent / "evals"


def _scripted(case) -> ScriptedProvider:
    calls = [{"type": "tool_use", "id": f"toolu_s{i}", "name": n, "input": a} for i, (n, a) in enumerate(case["tools"])]
    return ScriptedProvider([{"content": calls, "stop_reason": "tool_use"},
                             {"content": [{"type": "text", "text": case["answer"]}], "stop_reason": "end_turn"}])


def run_golden(tb: Toolbox, provider_factory=None) -> dict:
    provider_factory = provider_factory or (lambda: MockProvider(tb))
    spec = json.loads((EVALS / "golden.json").read_text())
    rows = []
    for c in spec["cases"]:
        t = run(c["question"], provider_factory(), tb)
        called = {x["name"] for x in t.tool_calls}
        cited = set(re.findall(r"\[([a-z_]+:[^\]\s]+)\]", t.answer))
        exp_tools, req = c.get("expect_tools", []), c.get("require_cited", [])
        text_ok = all(re.search(p, t.answer) for p in c.get("require_text", [])) and \
            not any(re.search(p, t.answer) for p in c.get("forbid_text", []))
        want = c.get("expect_outcome")
        outcome_ok = (t.outcome == "abstained") if want == "abstained" else (t.outcome != "abstained")
        row = {
            "id": c["id"], "hard": bool(c.get("hard")), "outcome": t.outcome,
            "tool_recall": sum(x in called for x in exp_tools) / len(exp_tools) if exp_tools else 1.0,
            "evidence_recall": sum(x in cited for x in req) / len(req) if req else 1.0,
            "missing_evidence": [x for x in req if x not in cited],
            "text_ok": text_ok, "outcome_ok": outcome_ok,
            "first_pass": t.first_report.passed, "first_violations": sorted(t.first_report.codes()),
            "final_guard": t.final_report.passed,
        }
        row["pass"] = (row["tool_recall"] == 1 and row["evidence_recall"] == 1 and text_ok and outcome_ok
                       and row["final_guard"])
        rows.append(row)

    def agg(rs):
        n = max(len(rs), 1)
        return {"n": len(rs), "pass_rate": sum(r["pass"] for r in rs) / n,
                "tool_recall": sum(r["tool_recall"] for r in rs) / n,
                "evidence_recall": sum(r["evidence_recall"] for r in rs) / n,
                "text_ok": sum(r["text_ok"] for r in rs) / n, "outcome_ok": sum(r["outcome_ok"] for r in rs) / n,
                "first_pass": sum(r["first_pass"] for r in rs) / n, "final_guard": sum(r["final_guard"] for r in rs) / n,
                "fallback_rate": sum(r["outcome"] == "fallback" for r in rs) / n}
    return {"cases": rows, "core": agg([r for r in rows if not r["hard"]]), "hard": agg([r for r in rows if r["hard"]])}


def run_adversarial(tb: Toolbox) -> dict:
    spec = json.loads((EVALS / "adversarial.json").read_text())
    rows, fp = [], []
    for c in spec["cases"]:
        t = run("adversarial", _scripted(c), tb, revise=False)
        got = t.first_report.codes()
        rows.append({"id": c["id"], "expect": c["expect"], "got": sorted(got), "caught": set(c["expect"]) <= got})
    for c in spec["controls"]:
        t = run("control", _scripted(c), tb, revise=False)
        fp.append({"id": c["id"], "violations": [f"{v.code}: {v.detail}" for v in t.first_report.violations],
                   "clean": t.first_report.passed})
    return {"cases": rows, "controls": fp,
            "recall": sum(r["caught"] for r in rows) / len(rows),
            "false_positive_rate": sum(not r["clean"] for r in fp) / len(fp)}


def main(tb: Toolbox, out: str | None = None) -> int:
    g, a = run_golden(tb), run_adversarial(tb)
    print("Golden (mock provider)")
    for r in g["cases"]:
        flag = "PASS" if r["pass"] else "FAIL"
        extra = f" missing={r['missing_evidence']}" if r["missing_evidence"] else ""
        extra += f" first_guard={r['first_violations']}" if r["first_violations"] else ""
        print(f"  {flag} {'(hard) ' if r['hard'] else ''}{r['id']:<24} outcome={r['outcome']}{extra}")
    for k in ("core", "hard"):
        print(f"  {k}: " + ", ".join(f"{m}={v:.2f}" if isinstance(v, float) else f"{m}={v}" for m, v in g[k].items()))
    print("Adversarial (guard)")
    for r in a["cases"]:
        print(f"  {'CAUGHT' if r['caught'] else 'MISSED'} {r['id']:<24} expect={r['expect']} got={r['got']}")
    for r in a["controls"]:
        print(f"  {'CLEAN ' if r['clean'] else 'FALSE+'} {r['id']:<24} {r['violations']}")
    print(f"  recall={a['recall']:.2f} false_positive_rate={a['false_positive_rate']:.2f}")
    if out:
        pathlib.Path(out).write_text(json.dumps({"golden": g, "adversarial": a}, indent=1))
    return 0 if g["core"]["pass_rate"] == 1.0 and a["recall"] == 1.0 and a["false_positive_rate"] == 0 else 1
