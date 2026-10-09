"""Pipeline health: alerts for problems a person should act on, without alerting on the
ordinary behaviour of the sources.

Evaluated after each run from the new manifest and a small history (health.json, carried
in the published dataset like everything else). What alerts, and why that threshold:

- source_failing      a NOAA source failed MIN_CONSECUTIVE_FAILURES runs in a row (~18 h at
                      the 6-hourly schedule). NOAA's ERDDAP refuses single runs (HTTP 403/502)
                      and serves the next one; one failure is not news.
- satellite_stale     newest Sentinel-3 pixel older than OLCI_MAX_AGE_DAYS, or newest VIIRS
                      pixel older than VIIRS_MAX_AGE_DAYS. Both are latency-aware (OLCI ~1-2 d,
                      VIIRS ~5 d). Statewide some pixel is almost always clear, so this means
                      the feed stopped, not that it was cloudy. Low coverage never alerts.
- charm_missing       newest C-HARM issue date older than CHARM_MAX_AGE_DAYS (daily product).
- hfr_stale           newest HF-radar hour older than HFR_MAX_AGE_HOURS (latency is ~4-15 h).
- hfr_coverage_drop   newest hour has fewer than HFR_COVERAGE_DROP of the window's median
                      valid cells (a network outage, not hour-to-hour variation).
- hfr_region_lost     a region that has had radar coverage before (per health.json history)
                      has none in the 24-hour window. Regions never covered do not alert.
- publish_stale       the previous published manifest is older than PUBLISH_MAX_AGE_HOURS when
                      this run starts (missed runs).
Publication failures of this workflow itself are reported by the workflow (a failed job opens
or updates an issue); this module cannot run if the build fails.
"""

from __future__ import annotations

import json
import statistics
from datetime import date, datetime, timezone
from pathlib import Path

from .models import Manifest

MIN_CONSECUTIVE_FAILURES = 3
OLCI_MAX_AGE_DAYS = 4
VIIRS_MAX_AGE_DAYS = 8
CHARM_MAX_AGE_DAYS = 2
HFR_MAX_AGE_HOURS = 18
HFR_COVERAGE_DROP = 0.5
PUBLISH_MAX_AGE_HOURS = 14
NOAA_SOURCES = ("charm", "satellite_chl", "hf_radar")
HEALTH_FILE = "health.json"


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def evaluate(manifest: Manifest, previous: dict | None, now: datetime, previous_generated_at: str | None = None) -> dict:
    prev = previous or {}
    prev_sources = prev.get("sources", {})
    prev_alerts = {a["key"]: a for a in prev.get("alerts", [])}
    now_iso = now.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sources: dict[str, dict] = {}
    alerts: list[dict] = []

    def alert(key: str, kind: str, message: str) -> None:
        alerts.append({"key": key, "kind": kind, "message": message, "since": prev_alerts.get(key, {}).get("since", now_iso)})

    # consecutive failures per source
    for st in manifest.sources:
        p = prev_sources.get(st.source_id, {})
        n = (p.get("consecutive_failures", 0) + 1) if st.outcome == "failed" else 0
        sources[st.source_id] = {"outcome": st.outcome, "consecutive_failures": n, "last_success_at": st.last_success_at, "error": (st.error or "")[:300] or None}
        if st.source_id in NOAA_SOURCES and n >= MIN_CONSECUTIVE_FAILURES:
            alert(f"source_failing:{st.source_id}", "source_failing", f"{st.title}: failed {n} runs in a row (last success {st.last_success_at or 'never'}). Last error: {(st.error or '')[:200]}")

    layers = {lyr.layer_id: lyr for lyr in manifest.layers}
    today = now.date()

    # satellite freshness (by newest observed pixel)
    for lid, limit, label in (("olci300_chl_latest", OLCI_MAX_AGE_DAYS, "Sentinel-3 OLCI 300 m"), ("viirs750_chl_latest", VIIRS_MAX_AGE_DAYS, "VIIRS 750 m")):
        lyr = layers.get(lid)
        if lyr and lyr.time.observed_date:
            age = (today - date.fromisoformat(lyr.time.observed_date)).days
            if age > limit:
                alert(f"satellite_stale:{lid}", "satellite_stale", f"{label}: newest observation {lyr.time.observed_date} is {age} days old (alert above {limit}).")
        elif "satellite_chl" in sources:
            alert(f"satellite_stale:{lid}", "satellite_stale", f"{label}: not in the published dataset.")

    # C-HARM run
    run = next((r for r in manifest.forecast_runs if r.group_id == "charm"), None)
    if run:
        age = (today - date.fromisoformat(run.issued_date)).days
        if age > CHARM_MAX_AGE_DAYS:
            alert("charm_missing", "charm_missing", f"C-HARM: newest run issued {run.issued_date}, {age} days ago (alert above {CHARM_MAX_AGE_DAYS}).")
    elif "charm" in sources:
        alert("charm_missing", "charm_missing", "C-HARM: no forecast run in the published dataset.")

    # HF radar
    hourly = sorted((lyr for lyr in manifest.layers if lyr.layer_id.startswith("hfr2km_currents_2") and lyr.qc), key=lambda x: x.time.valid_time or "")
    region_cov: dict[str, bool] = {}
    if hourly:
        newest = hourly[-1]
        hours = (now - _dt(newest.time.valid_time or now_iso)).total_seconds() / 3600
        if hours > HFR_MAX_AGE_HOURS:
            alert("hfr_stale", "hfr_stale", f"HF radar: newest hour {newest.time.valid_time} is {hours:.0f} h old (alert above {HFR_MAX_AGE_HOURS} h).")
        counts = [h.qc.n_valid for h in hourly if h.qc]
        med = statistics.median(counts) if counts else 0
        if med and newest.qc and newest.qc.n_valid < HFR_COVERAGE_DROP * med:
            alert("hfr_coverage_drop", "hfr_coverage_drop", f"HF radar: newest hour has {newest.qc.n_valid} valid cells, under {HFR_COVERAGE_DROP:.0%} of the 24 h median ({med:.0f}).")
        for h in hourly:
            for rc in (h.coverage.regions if h.coverage else []):
                region_cov[rc.region_id] = region_cov.get(rc.region_id, False) or rc.observed_fraction > 0
    elif "hf_radar" in sources:
        alert("hfr_stale", "hfr_stale", "HF radar: no hourly currents in the published dataset.")
    seen = dict(prev.get("hfr_regions_last_covered", {}))
    for rid, covered in region_cov.items():
        if covered:
            seen[rid] = now_iso
        elif rid in seen:
            alert(f"hfr_region_lost:{rid}", "hfr_region_lost", f"HF radar: no coverage in {rid} for the last 24 hours (last covered {seen[rid]}).")

    # missed runs
    if previous_generated_at:
        gap = (now - _dt(previous_generated_at)).total_seconds() / 3600
        if gap > PUBLISH_MAX_AGE_HOURS:
            alert("publish_stale", "publish_stale", f"The previously published dataset was {gap:.0f} h old when this run started (runs are every 6 h).")

    return {
        "schema": "coastwatch-health-1",
        "evaluated_at": now_iso,
        "pipeline_run_id": manifest.pipeline_run_id,
        "thresholds": {
            "min_consecutive_failures": MIN_CONSECUTIVE_FAILURES, "olci_max_age_days": OLCI_MAX_AGE_DAYS, "viirs_max_age_days": VIIRS_MAX_AGE_DAYS,
            "charm_max_age_days": CHARM_MAX_AGE_DAYS, "hfr_max_age_hours": HFR_MAX_AGE_HOURS, "hfr_coverage_drop": HFR_COVERAGE_DROP,
            "publish_max_age_hours": PUBLISH_MAX_AGE_HOURS,
        },
        "sources": sources,
        "hfr_regions_last_covered": seen,
        "alerts": alerts,
        "new_alerts": [a["key"] for a in alerts if a["key"] not in prev_alerts],
        "cleared_alerts": [k for k in prev_alerts if k not in {a["key"] for a in alerts}],
    }


def markdown(h: dict, run_url: str = "") -> str:
    lines = [f"Pipeline health at {h['evaluated_at']} (run {h['pipeline_run_id']}{', ' + run_url if run_url else ''})", ""]
    if not h["alerts"]:
        lines.append("All checks clear.")
    for a in h["alerts"]:
        new = " **new**" if a["key"] in h["new_alerts"] else ""
        lines.append(f"- [{a['kind']}]{new} {a['message']} (since {a['since']})")
    if h["cleared_alerts"]:
        lines += ["", "Cleared since the last run: " + ", ".join(h["cleared_alerts"])]
    lines += ["", "Thresholds and rationale: pipeline/coastwatch_pipeline/health.py. The published data keep their own dates; nothing is relabelled."]
    return "\n".join(lines)


def run_health(out_dir: Path, previous_manifest_generated_at: str | None, now: datetime | None = None) -> dict:
    m = Manifest.model_validate_json((out_dir / "manifest.json").read_text())
    p = out_dir / HEALTH_FILE
    previous = json.loads(p.read_text()) if p.exists() else None
    h = evaluate(m, previous, now or datetime.now(timezone.utc), previous_manifest_generated_at)
    p.write_text(json.dumps(h, indent=1) + "\n")
    return h
