"""Probe ERDDAP griddap datasets: time coverage, last time step, resolution, bounds.

usage: python probe_erddap.py <server> <dataset_id> [...]
Writes one JSON line per dataset to stdout (evidence/erddap-probe.jsonl).
"""
import json, sys, urllib.request, datetime as dt

def get(url, timeout=90):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CoastWatch-research/0.1"}), timeout=timeout) as r:
        return json.load(r)

def probe(server, ds):
    out = {"server": server, "dataset": ds, "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    try:
        info = get(f"{server}/info/{ds}/index.json")
        rows = info["table"]["rows"]
        attrs = {(r[1], r[2]): r[4] for r in rows if r[0] == "attribute"}
        dims = [r[1] for r in rows if r[0] == "dimension"]
        out["title"] = attrs.get(("NC_GLOBAL", "title"))
        out["institution"] = attrs.get(("NC_GLOBAL", "institution"))
        out["license"] = (attrs.get(("NC_GLOBAL", "license")) or "")[:200]
        out["time_coverage_end"] = attrs.get(("NC_GLOBAL", "time_coverage_end"))
        out["variables"] = [r[1] for r in rows if r[0] == "variable"]
        out["dims"] = {d: {k: attrs.get((d, k)) for k in ("actual_range",)} | {"spacing": next((r[4] for r in rows if r[0] == "dimension" and r[1] == d), None)} for d in dims}
        # last two time steps from the axis itself
        if "time" in dims:
            t = get(f"{server}/griddap/{ds}.json?time[last-2:1:last]")
            out["last_times"] = [r[0] for r in t["table"]["rows"]]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out

if __name__ == "__main__":
    server = sys.argv[1]
    for ds in sys.argv[2:]:
        print(json.dumps(probe(server, ds)), flush=True)
