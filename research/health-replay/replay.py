"""Replay pipeline health over every published main-branch run, in order: what the current
rules would have alerted, beside what the rules of the time did. Read-only. From pipeline/:
    ../research/health-replay/fetch.sh /tmp/replay
    uv run python ../research/health-replay/replay.py /tmp/replay
"""
import json, sys
from pathlib import Path
from datetime import datetime, timezone
from coastwatch_pipeline.health import evaluate
from coastwatch_pipeline.models import Manifest
D = Path(sys.argv[1])
runs = {str(r["databaseId"]): r for r in json.load(open(D / "runs.json"))}
order = []
for rid, r in runs.items():
    p = D / "runs" / rid / "manifest.json"
    if r["headBranch"] == "main" and p.exists():
        order.append((r["createdAt"], rid))
order.sort()
prev = None; prev_gen = None; out = []
for created, rid in order:
    raw = json.loads((D / "runs" / rid / "manifest.json").read_text())
    try:
        m = Manifest.model_validate(raw)
    except Exception as e:
        print(rid, created, "SKIP (old schema):", str(e).splitlines()[0]); continue
    now = datetime.fromisoformat(raw["generated_at"].replace("Z", "+00:00"))
    h = evaluate(m, prev, now, prev_gen)
    old = json.loads((D / "runs" / rid / "health.json").read_text()) if (D / "runs" / rid / "health.json").exists() else {}
    fails = [f"{s.source_id}:{s.outcome}" for s in m.sources if s.outcome != "success"]
    print(f"{raw['generated_at']}  {runs[rid]['event'][:8]:8}  non-ok={fails}")
    print(f"   old alerts: {sorted(a['key'] for a in old.get('alerts', []))}")
    print(f"   new alerts: {sorted(a['key'] for a in h['alerts'])}")
    for a in h["alerts"]:
        if a["key"] not in {x["key"] for x in (prev or {}).get("alerts", [])}:
            print(f"     + {a['key']}: {a['message'][:160]}")
    prev, prev_gen = h, raw["generated_at"]
print("\nfreshness table of the last run:")
for sid, f in prev.get("freshness", {}).items():
    print(f"   {sid:14} {f['outcome']:9} newest {f['newest_date']} ({f['basis']}) age {f['age_days']} d  {f['state']}")
