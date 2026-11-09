"""Curated fisheries status -> `v1/fisheries-status/index.json` (FisheriesStatusIndex, pilot sidecar).

Input: `data/curated/fisheries-status/status.yaml`, transcribed from CDFW/CDPH pages and reviewed in
pull requests. Review state comes ONLY from `review.json` beside it, written by the human review
command (`cwp review-fisheries-status --reviewer NAME --confirm`): a record is `human_verified` only if
the review lists its id with the hash of its current content, so editing a record after review makes it
pending again. The YAML may not claim a review itself. Nothing here expires on its own: `until` is
transcribed, never acted on, and "at least" dates stay in `detail`.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..pilot_models import FisheriesStatusIndex, FisheryStatus, RampState, RampZone

REPO = Path(__file__).resolve().parents[3]
CURATED_DIR = REPO / "data" / "curated" / "fisheries-status"
OUT_REL = "fisheries-status/index.json"
PILOT_PLACES = {"port-592", "port-550", "port-593", "port-452"}


class CurationError(ValueError):
    pass


def record_hash(rec: dict) -> str:
    """Hash of a record's transcribed content (review fields excluded)."""
    body = {k: v for k, v in rec.items() if k not in ("review_state", "reviewed_by", "reviewed_at")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load(curated_dir: Path = CURATED_DIR) -> tuple[dict, dict]:
    raw = yaml.safe_load((curated_dir / "status.yaml").read_text())
    review_path = curated_dir / "review.json"
    review = json.loads(review_path.read_text()) if review_path.exists() else {}
    return raw, review


def _check(raw: dict) -> list[str]:
    problems = []
    ids = [r.get("id") for r in raw.get("records", [])]
    if len(ids) != len(set(ids)):
        problems.append("duplicate record ids")
    for r in raw.get("records", []):
        for k in ("review_state", "reviewed_by", "reviewed_at"):
            if k in r:
                problems.append(f"{r.get('id')}: {k} may not be set in the YAML (the review command records reviews)")
        if not str(r.get("source_url", "")).startswith("https://"):
            problems.append(f"{r.get('id')}: source_url must be an https URL")
        unknown = set(r.get("applies_to_places") or []) - PILOT_PLACES
        if unknown:
            problems.append(f"{r.get('id')}: unknown places {sorted(unknown)}")
    if raw.get("ramp") and any(k in raw["ramp"] for k in ("review_state", "reviewed_by", "reviewed_at")):
        problems.append("ramp: review fields may not be set in the YAML")
    return problems


def build(now_iso: str, curated_dir: Path = CURATED_DIR) -> FisheriesStatusIndex:
    raw, review = load(curated_dir)
    problems = _check(raw)
    if problems:
        raise CurationError("; ".join(problems))
    checked = str(raw["checked_at"])
    reviewed = review.get("records", {})
    records = []
    for r in raw["records"]:
        rec = {"checked_at": checked, **r}
        rec.setdefault("other_sources", [])
        h = record_hash(rec)
        rv = reviewed.get(rec["id"])
        if rv and rv.get("sha256") == h and review.get("reviewer"):
            rec.update(review_state="human_verified", reviewed_by=review["reviewer"], reviewed_at=review.get("reviewed_at"))
        else:
            rec["review_state"] = "pending_human_review"
        records.append(FisheryStatus.model_validate(rec))
    ramp = None
    if raw.get("ramp"):
        rr = {"checked_at": checked, **raw["ramp"]}
        zones = [RampZone.model_validate(z) for z in rr.pop("zones")]
        h = record_hash({**rr, "zones": [z.model_dump() for z in zones]})
        rv = reviewed.get("ramp")
        state = {"review_state": "pending_human_review"}
        if rv and rv.get("sha256") == h and review.get("reviewer"):
            state = {"review_state": "human_verified", "reviewed_by": review["reviewer"], "reviewed_at": review.get("reviewed_at")}
        ramp = RampState.model_validate({**rr, "zones": zones, **state})
    return FisheriesStatusIndex(version=1, generated_at=now_iso, records=records, ramp=ramp)


def ramp_hash(raw: dict) -> str:
    rr = {"checked_at": str(raw["checked_at"]), **raw["ramp"]}
    zones = [RampZone.model_validate(z) for z in rr.pop("zones")]
    return record_hash({**rr, "zones": [z.model_dump() for z in zones]})


def record_review(curated_dir: Path, reviewer: str, ids: list[str] | None, method: str) -> dict:
    """Write review.json for the given record ids (all records and the RAMP state when None).
    Called only by the CLI after a person passed --reviewer and --confirm."""
    if not reviewer.strip():
        raise CurationError("a reviewer name is required")
    raw, review = load(curated_dir)
    problems = _check(raw)
    if problems:
        raise CurationError("; ".join(problems))
    checked = str(raw["checked_at"])
    hashes = {r["id"]: record_hash({"checked_at": checked, **r, "other_sources": r.get("other_sources", [])}) for r in raw["records"]}
    if raw.get("ramp"):
        hashes["ramp"] = ramp_hash(raw)
    want = ids or list(hashes)
    missing = [i for i in want if i not in hashes]
    if missing:
        raise CurationError(f"unknown record ids: {missing}")
    if review.get("reviewer") and review["reviewer"] != reviewer:
        review = {}  # a new reviewer re-reviews from scratch; one name per review file
    recs = review.get("records", {})
    for i in want:
        recs[i] = {"sha256": hashes[i]}
    out = {
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "method": method,
        "records": dict(sorted(recs.items())),
    }
    (curated_dir / "review.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def write(out_dir: Path, idx: FisheriesStatusIndex) -> Path:
    from ..publish.files import write_json

    path = out_dir / OUT_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, idx.model_dump(mode="json"))
    return path
