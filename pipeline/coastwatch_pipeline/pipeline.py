"""Run sources independently and publish a manifest.

Failure policy (docs/coastwatch/04-architecture.md §6):
- each source runs in isolation; one failure never blocks the others
- on failure the previous artifacts for that source stay published *with their real
  dates* (the browser classifies them stale/historical) and the failure is recorded
- a forecast lead missing from the newest run is reported missing, never back-filled
- manifest.json is written last and atomically
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

from .context import RunContext
from .models import (
    ForecastRun,
    FreshnessPolicy,
    LayerArtifact,
    Manifest,
    FisheriesDataset,
    ObservationDataset,
    OfficialDataset,
    PortsCollection,
    SourceStatus,
)
from .publish.files import write_bytes, write_json
from .sources import charm, fisheries, gibs, observations, official, port_intel, ports, satellite

ALL_SOURCES = ("charm", "gibs_chl", "satellite_chl", "cdfw_ports", "official", "port_intel", "calhabmap", "foss_landings")

OFFICIAL_FRESHNESS = FreshnessPolicy(
    basis="reviewed_date",
    current_max_age_days=official.POLICY.verified_max_age_days,
    stale_max_age_days=official.POLICY.aging_max_age_days,
    note=official.POLICY.note,
)
PORT_INTEL_FRESHNESS = FreshnessPolicy(
    basis="valid_date",
    current_max_age_days=2,
    stale_max_age_days=7,
    note="Port summaries are rebuilt from each pipeline run; their inputs carry their own dates.",
)
OBS_FRESHNESS = FreshnessPolicy(
    basis="observed_date",
    current_max_age_days=14,
    stale_max_age_days=45,
    note="Shore stations are sampled about weekly and published after laboratory analysis; each station and variable carries its own sample dates.",
)
FISHERIES_FRESHNESS = FreshnessPolicy(
    basis="valid_date",
    current_max_age_days=730,
    stale_max_age_days=1100,
    note="Annual landings are finalised and published one to two years after the year ends; the date is the end of the latest year included.",
)
FISHERIES_REFRESH_DAYS = 7  # annual data; avoids hitting the BLS API's daily request limit


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
            # "unchanged" only when the same run is republished at the same addresses; new
            # renderings (e.g. a palette change) count as updated so they are verified
            same_paths = {lyr.image.url for lyr in res.layers if lyr.image} == {lyr.image.url for lyr in _prev_layers(prev, charm.SOURCE_ID) if lyr.image}
            unchanged = same_paths and prev_s is not None and prev_s.latest_issued_date == (res.latest_issued.isoformat() if res.latest_issued else None)
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
                    notes=res.run.notes,
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
                    error="; ".join(g.errors) or None,
                    notes=g.notes,
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

    # ---- high-resolution satellite chlorophyll (OLCI 300 m, VIIRS 750 m fallback)
    sat_title = "Satellite chlorophyll-a: Sentinel-3 OLCI 300 m (VIIRS 750 m fallback)"
    if "satellite_chl" in only:
        prev_sat = _prev_layers(prev, satellite.SOURCE_ID)
        sres = satellite.run(ctx, prev_sat)
        prev_s = _prev_status(prev, satellite.SOURCE_ID)
        if sres.layers:
            layers.extend(sres.layers)
            olci_ok = any(lyr.layer_id == "olci300_chl_latest" for lyr in sres.layers)
            statuses.append(
                SourceStatus(
                    source_id=satellite.SOURCE_ID,
                    title=sat_title,
                    product_class="observation",
                    last_attempt_at=ctx.now_iso,
                    last_success_at=ctx.now_iso,
                    outcome="partial" if sres.errors or not olci_ok else "updated",
                    error="; ".join(sres.errors)[:800] or None,
                    notes=sres.notes[:40],
                    latest_valid_date=sres.latest_date,
                    freshness=satellite.FRESHNESS,
                )
            )
        else:
            layers.extend(prev_sat)
            statuses.append(_failed(ctx, satellite.SOURCE_ID, sat_title, "observation", satellite.FRESHNESS, prev_s, sres.errors + sres.notes))
    else:
        layers.extend(_prev_layers(prev, satellite.SOURCE_ID))
        _carry(statuses, prev, satellite.SOURCE_ID)

    # ---- CDFW ports
    if "cdfw_ports" in only:
        pr = ports.run(ctx)
        prev_s = _prev_status(prev, ports.SOURCE_ID)
        port_policy = _ports_freshness()
        if pr.collection:
            body = (json.dumps(pr.collection.model_dump(mode="json"), indent=2) + "\n").encode()
            ports_url = f"ports-{hashlib.sha256(body).hexdigest()[:10]}.geojson"
            write_bytes(ctx.out_dir / ports_url, body)
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

    # ---- official notices (human-reviewed registry + page watcher + official geometry)
    official_url = prev.official_url if prev else None
    official_ds: OfficialDataset | None = _load(ctx, official_url, OfficialDataset)
    if "official" in only:
        res = official.run(ctx)
        prev_s = _prev_status(prev, official.SOURCE_ID)
        if res.dataset:
            official_url = _write_hashed(ctx, "official", res.dataset.model_dump(mode="json"))
            official_ds = res.dataset
            watch_failed = any(not w.ok for w in res.dataset.watch)
            statuses.append(
                SourceStatus(
                    source_id=official.SOURCE_ID,
                    title="Official closures and advisories (CDFW, CDPH)",
                    product_class="official_regulatory",
                    last_attempt_at=ctx.now_iso,
                    last_success_at=ctx.now_iso,
                    outcome="partial" if watch_failed or res.dataset.geometry_errors else "updated",
                    error="; ".join(w.error or "" for w in res.dataset.watch if not w.ok) or None,
                    notes=res.notes,
                    latest_valid_date=res.dataset.registry.review.reviewed_at[:10],
                    freshness=OFFICIAL_FRESHNESS,
                )
            )
        else:
            statuses.append(
                _failed(ctx, official.SOURCE_ID, "Official closures and advisories (CDFW, CDPH)", "official_regulatory", OFFICIAL_FRESHNESS, prev_s, res.errors)
            )
    else:
        _carry(statuses, prev, official.SOURCE_ID)

    # ---- port intelligence (depends on the C-HARM layers, ports and official notices above)
    port_intel_url = prev.port_intel_url if prev else None
    if "port_intel" in only:
        ports_coll = _load(ctx, ports_url, PortsCollection)
        pi = port_intel.run(ctx, layers, ports_coll, official_ds)
        prev_s = _prev_status(prev, port_intel.SOURCE_ID)
        if pi.collection:
            port_intel_url = _write_hashed(ctx, "port-intel", pi.collection.model_dump(mode="json"))
            charm_run = next((r for r in runs if r.source_id == charm.SOURCE_ID), None)
            statuses.append(
                SourceStatus(
                    source_id=port_intel.SOURCE_ID,
                    title="Port summaries (C-HARM, satellite chlorophyll, official notices)",
                    product_class="derived_summary",
                    last_attempt_at=ctx.now_iso,
                    last_success_at=ctx.now_iso,
                    outcome="partial" if pi.notes else "updated",
                    notes=pi.notes[:40],
                    latest_issued_date=charm_run.issued_date if charm_run else None,
                    latest_valid_date=ctx.now.date().isoformat(),
                    freshness=PORT_INTEL_FRESHNESS,
                )
            )
        else:
            statuses.append(
                _failed(ctx, port_intel.SOURCE_ID, "Port summaries (C-HARM, satellite chlorophyll, official notices)", "derived_summary", PORT_INTEL_FRESHNESS, prev_s, pi.errors)
            )
    else:
        _carry(statuses, prev, port_intel.SOURCE_ID)

    # ---- measured observations (CalHABMAP); uses ports and the C-HARM layers above
    observations_url = prev.observations_url if prev else None
    if "calhabmap" in only:
        prev_obs = _load(ctx, observations_url, ObservationDataset)
        ob = observations.run(ctx, _load(ctx, ports_url, PortsCollection), layers, prev_obs)
        prev_s = _prev_status(prev, observations.SOURCE_ID)
        title = "CalHABMAP shore-station measurements (domoic acid, Pseudo-nitzschia)"
        if ob.dataset:
            observations_url = _write_hashed(ctx, "observations", ob.dataset.model_dump(mode="json"), compact=True)
            degraded = [s for s in ob.dataset.stations if s.status != "updated"]
            statuses.append(
                SourceStatus(
                    source_id=observations.SOURCE_ID,
                    title=title,
                    product_class="observation",
                    last_attempt_at=ctx.now_iso,
                    last_success_at=ctx.now_iso,
                    outcome="partial" if degraded or ob.errors else "updated",
                    error="; ".join(ob.errors)[:500] or None,
                    notes=ob.notes[:40],
                    latest_valid_date=ob.latest_sample[:10] if ob.latest_sample else None,
                    freshness=OBS_FRESHNESS,
                )
            )
        else:
            statuses.append(_failed(ctx, observations.SOURCE_ID, title, "observation", OBS_FRESHNESS, prev_s, ob.errors))
    else:
        _carry(statuses, prev, observations.SOURCE_ID)

    # ---- historical fisheries exposure (FOSS + BLS CPI); refreshed weekly
    fisheries_url = prev.fisheries_url if prev else None
    if "foss_landings" in only:
        prev_s = _prev_status(prev, fisheries.SOURCE_ID)
        title = "Historical fisheries exposure (NOAA FOSS landings, BLS CPI-U)"
        prev_fish = _load(ctx, fisheries_url, FisheriesDataset)
        # reuse only an artifact built by this same pipeline version (a code change may change the method)
        recent = (
            prev_fish is not None
            and prev_s is not None
            and prev_s.last_success_at is not None
            and all(p.pipeline_version == ctx.pipeline_version for p in prev_fish.provenance)
            and (ctx.now - datetime.fromisoformat(prev_s.last_success_at.replace("Z", "+00:00"))) < timedelta(days=FISHERIES_REFRESH_DAYS)
        )
        if recent:
            assert prev_s is not None
            statuses.append(prev_s.model_copy(update={"last_attempt_at": ctx.now_iso, "outcome": "unchanged", "notes": [f"Annual data; refreshed every {FISHERIES_REFRESH_DAYS} days."]}))
        else:
            fr = fisheries.run(ctx)
            if fr.dataset:
                fisheries_url = _write_hashed(ctx, "fisheries", fr.dataset.model_dump(mode="json"))
                statuses.append(
                    SourceStatus(
                        source_id=fisheries.SOURCE_ID,
                        title=title,
                        product_class="historical_context",
                        last_attempt_at=ctx.now_iso,
                        last_success_at=ctx.now_iso,
                        outcome="updated",
                        notes=fr.notes,
                        latest_valid_date=f"{fr.latest_year}-12-31" if fr.latest_year else None,
                        freshness=FISHERIES_FRESHNESS,
                    )
                )
            else:
                statuses.append(_failed(ctx, fisheries.SOURCE_ID, title, "historical_context", FISHERIES_FRESHNESS, prev_s, fr.errors))
    else:
        _carry(statuses, prev, fisheries.SOURCE_ID)

    manifest = Manifest(
        generated_at=ctx.now_iso,
        pipeline_version=ctx.pipeline_version,
        pipeline_run_id=ctx.run_id,
        layers=layers,
        forecast_runs=runs,
        sources=statuses,
        ports_url=ports_url,
        official_url=official_url,
        port_intel_url=port_intel_url,
        observations_url=observations_url,
        fisheries_url=fisheries_url,
    )
    write_json(ctx.out_dir / "manifest.json", manifest.model_dump(mode="json"))
    prune(ctx.out_dir, manifest, prev)
    return manifest


def _write_hashed(ctx: RunContext, stem: str, obj: dict, compact: bool = False) -> str:
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) if compact else json.dumps(obj, indent=1, ensure_ascii=False)
    body = (text + "\n").encode()
    name = f"{stem}-{hashlib.sha256(body).hexdigest()[:10]}.json"
    write_bytes(ctx.out_dir / name, body)
    return name


def _load(ctx: RunContext, rel: str | None, model):
    if not rel or not (ctx.out_dir / rel).exists():
        return None
    try:
        return model.model_validate_json((ctx.out_dir / rel).read_text())
    except Exception:
        return None


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


def referenced_prefixes(m: Manifest | None) -> set[str]:
    """Directories a manifest references as a whole: tile pyramids and chunked grids."""
    out: set[str] = set()
    for lyr in m.layers if m else []:
        tiles = [lyr.tiles] + ([lyr.composite.age_tiles] if lyr.composite and lyr.composite.age_tiles else [])
        for t in tiles:
            if t and t.relative:
                out.add(t.url_template.split("{z}")[0])
        grids = [lyr.grid] + ([lyr.composite.age_grid] if lyr.composite else [])
        for g in grids:
            if g and g.chunks:
                out.add(g.chunks.url_template.split("{row}")[0])
    return out


def referenced_paths(m: Manifest | None) -> set[str]:
    """Every single artifact file a manifest points to (relative paths)."""
    if not m:
        return set()
    out: set[str] = set()
    for lyr in m.layers:
        if lyr.image:
            out.add(lyr.image.url)
        if lyr.grid and not lyr.grid.chunks:
            out.add(lyr.grid.url)
    for rel in (m.ports_url, m.official_url, m.port_intel_url, m.observations_url, m.fisheries_url):
        if rel:
            out.add(rel)
    return out


def prune(out_dir: Path, manifest: Manifest, previous: Manifest | None) -> None:
    """Keep files referenced by the new manifest and by the previous one (clients and CDN
    caches may still hold the previous manifest for a few minutes); delete everything
    else the pipeline owns. Never touches files outside charm/, satellite/ and the hashed JSON/GeoJSON artifacts."""
    keep = referenced_paths(manifest) | referenced_paths(previous)
    prefixes = tuple(referenced_prefixes(manifest) | referenced_prefixes(previous))
    for root in ("charm", "satellite"):
        base = out_dir / root
        if not base.exists():
            continue
        for f in sorted(base.rglob("*"), reverse=True):
            rel = f.relative_to(out_dir).as_posix()
            if f.is_file() and rel not in keep and not (prefixes and rel.startswith(prefixes)):
                f.unlink()
            elif f.is_dir() and not any(f.iterdir()):
                f.rmdir()
    for pattern in ("ports*.geojson", "official-*.json", "port-intel-*.json", "observations-*.json", "fisheries-*.json"):
        for f in out_dir.glob(pattern):
            if f.name not in keep:
                f.unlink()


def manifest_summary(m: Manifest) -> str:
    out = {
        "generated_at": m.generated_at,
        "layers": len(m.layers),
        "runs": [r.model_dump() for r in m.forecast_runs],
        "sources": [{k: getattr(s, k) for k in ("source_id", "outcome", "latest_issued_date", "latest_valid_date", "error")} for s in m.sources],
    }
    return json.dumps(out, indent=2)
