"""Run sources independently and publish a manifest.

Failure policy (docs/coastwatch/04-architecture.md §6):
- each source runs in isolation; one failure never blocks the others
- on failure the previous artifacts for that source stay published *with their real
  dates* (the browser classifies them stale/historical) and the failure is recorded
- a forecast lead missing from the newest run is reported missing, never back-filled
- manifest.json is written last and atomically
"""

from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

from .context import RunContext
from .models import ForecastRun, LayerArtifact, Manifest, SourceStatus
from .publish.files import write_json
from .sources import charm, gibs, ports

ALL_SOURCES = ("charm", "gibs_chl", "cdfw_ports")
RETENTION_DAYS = 14


def load_previous(out_dir: Path) -> Manifest | None:
    p = out_dir / "manifest.json"
    if not p.exists():
        return None
    try:
        return Manifest.model_validate_json(p.read_text())
    except Exception:
        return None


def _prev_status(prev: Manifest | None, source_id: str) -> SourceStatus | None:
    if not prev:
        return None
    return next((s for s in prev.sources if s.source_id == source_id), None)


def _prev_layers(prev: Manifest | None, source_id: str) -> list[LayerArtifact]:
    if not prev:
        return []
    return [lyr for lyr in prev.layers if lyr.provenance.source_id == source_id]


def run_pipeline(ctx: RunContext, only: tuple[str, ...] = ALL_SOURCES) -> Manifest:
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    prev = load_previous(ctx.out_dir)
    layers: list[LayerArtifact] = []
    runs: list[ForecastRun] = []
    statuses: list[SourceStatus] = []
    ports_url = prev.ports_url if prev else None

    # ---- C-HARM
    if "charm" in only:
        res = charm.run(ctx)
        prev_s = _prev_status(prev, charm.SOURCE_ID)
        if res.layers:
            layers.extend(res.layers)
            assert res.run is not None
            runs.append(res.run)
            unchanged = prev_s is not None and prev_s.latest_issued_date == (res.latest_issued.isoformat() if res.latest_issued else None)
            outcome = "partial" if res.errors or res.run.leads_missing else ("unchanged" if unchanged else "updated")
            statuses.append(
                SourceStatus(
                    source_id=charm.SOURCE_ID,
                    title="C-HARM v3.1 harmful algal bloom forecast",
                    product_class="official_forecast",
                    last_attempt_at=ctx.now_iso,
                    last_success_at=ctx.now_iso,
                    outcome=outcome,
                    error="; ".join(res.errors) or None,
                    latest_issued_date=res.latest_issued.isoformat() if res.latest_issued else None,
                    latest_valid_date=res.latest_valid.isoformat() if res.latest_valid else None,
                    freshness=charm.FRESHNESS,
                )
            )
        else:
            layers.extend(_prev_layers(prev, charm.SOURCE_ID))
            if prev:
                runs.extend(r for r in prev.forecast_runs if r.source_id == charm.SOURCE_ID)
            statuses.append(_failed(ctx, charm.SOURCE_ID, "C-HARM v3.1 harmful algal bloom forecast", "official_forecast", charm.FRESHNESS, prev_s, res.errors))
    else:
        layers.extend(_prev_layers(prev, charm.SOURCE_ID))
        if prev:
            runs.extend(r for r in prev.forecast_runs if r.source_id == charm.SOURCE_ID)
        _carry(statuses, prev, charm.SOURCE_ID)

    # ---- GIBS satellite chlorophyll
    if "gibs_chl" in only:
        g = gibs.run(ctx)
        prev_s = _prev_status(prev, gibs.SOURCE_ID)
        if g.layers:
            layers.extend(g.layers)
            statuses.append(
                SourceStatus(
                    source_id=gibs.SOURCE_ID,
                    title="NASA GIBS satellite chlorophyll",
                    product_class="observation",
                    last_attempt_at=ctx.now_iso,
                    last_success_at=ctx.now_iso,
                    outcome="partial" if g.errors else "updated",
                    error="; ".join(g.errors + g.notes) or None,
                    latest_valid_date=g.latest_date,
                    freshness=gibs.FRESHNESS,
                )
            )
        else:
            layers.extend(_prev_layers(prev, gibs.SOURCE_ID))
            statuses.append(_failed(ctx, gibs.SOURCE_ID, "NASA GIBS satellite chlorophyll", "observation", gibs.FRESHNESS, prev_s, g.errors + g.notes))
    else:
        layers.extend(_prev_layers(prev, gibs.SOURCE_ID))
        _carry(statuses, prev, gibs.SOURCE_ID)

    # ---- CDFW ports
    if "cdfw_ports" in only:
        pr = ports.run(ctx)
        prev_s = _prev_status(prev, ports.SOURCE_ID)
        port_policy = _ports_freshness()
        if pr.collection:
            write_json(ctx.out_dir / "ports.geojson", pr.collection.model_dump(mode="json"))
            ports_url = "ports.geojson"
            statuses.append(
                SourceStatus(
                    source_id=ports.SOURCE_ID,
                    title="CDFW landing ports (ds3081)",
                    product_class="reference",
                    last_attempt_at=ctx.now_iso,
                    last_success_at=ctx.now_iso,
                    outcome="updated",
                    latest_valid_date=ctx.now.date().isoformat(),
                    freshness=port_policy,
                )
            )
        else:
            statuses.append(_failed(ctx, ports.SOURCE_ID, "CDFW landing ports (ds3081)", "reference", port_policy, prev_s, pr.errors))
    else:
        _carry(statuses, prev, ports.SOURCE_ID)

    manifest = Manifest(
        generated_at=ctx.now_iso,
        pipeline_version=ctx.pipeline_version,
        pipeline_run_id=ctx.run_id,
        layers=layers,
        forecast_runs=runs,
        sources=statuses,
        ports_url=ports_url,
    )
    write_json(ctx.out_dir / "manifest.json", manifest.model_dump(mode="json"))
    prune(ctx.out_dir, manifest, ctx.now.date())
    return manifest


def _ports_freshness():
    from .models import FreshnessPolicy

    return FreshnessPolicy(
        basis="valid_date",
        current_max_age_days=35,
        stale_max_age_days=120,
        note="Reference geometry; changes rarely. Re-fetched every run.",
    )


def _failed(ctx, source_id, title, pclass, policy, prev_s, errors) -> SourceStatus:
    return SourceStatus(
        source_id=source_id,
        title=title,
        product_class=pclass,
        last_attempt_at=ctx.now_iso,
        last_success_at=prev_s.last_success_at if prev_s else None,
        outcome="failed",
        error="; ".join(errors) or "unknown error",
        latest_issued_date=prev_s.latest_issued_date if prev_s else None,
        latest_valid_date=prev_s.latest_valid_date if prev_s else None,
        freshness=policy,
    )


def _carry(statuses: list[SourceStatus], prev: Manifest | None, source_id: str) -> None:
    s = _prev_status(prev, source_id)
    if s:
        statuses.append(s)


def prune(out_dir: Path, manifest: Manifest, today: date) -> None:
    """Delete C-HARM run directories that are unreferenced and older than RETENTION_DAYS."""
    keep = {lyr.image.url.split("/")[1] for lyr in manifest.layers if lyr.image and lyr.image.url.startswith("charm/")}
    root = out_dir / "charm"
    if not root.exists():
        return
    cutoff = today - timedelta(days=RETENTION_DAYS)
    for d in root.iterdir():
        try:
            dd = date.fromisoformat(d.name)
        except ValueError:
            continue
        if d.name not in keep and dd < cutoff:
            shutil.rmtree(d)


def manifest_summary(m: Manifest) -> str:
    out = {
        "generated_at": m.generated_at,
        "layers": len(m.layers),
        "runs": [r.model_dump() for r in m.forecast_runs],
        "sources": [{k: getattr(s, k) for k in ("source_id", "outcome", "latest_issued_date", "latest_valid_date", "error")} for s in m.sources],
    }
    return json.dumps(out, indent=2)
