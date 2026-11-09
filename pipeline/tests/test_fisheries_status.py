"""Curated fisheries status (data/curated/fisheries-status) -> v1/fisheries-status/index.json.
Nothing is verified, lifted or expired by code: only the human review command records a review."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from coastwatch_pipeline.cli import main
from coastwatch_pipeline.fixtures import fixture_context
from coastwatch_pipeline.pilot_models import FisheriesStatusIndex
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.sources import fisheries_status as fs

NOW = "2026-11-08T18:00:00Z"


@pytest.fixture
def curated(tmp_path) -> Path:
    d = tmp_path / "curated"
    shutil.copytree(fs.CURATED_DIR, d)
    (d / "review.json").unlink(missing_ok=True)
    return d


def test_committed_records_are_complete_and_all_pending():
    idx = fs.build(NOW)
    assert idx.version == 1 and len(idx.records) >= 6
    for r in idx.records:
        assert r.review_state == "pending_human_review" and r.reviewed_by is None
        assert r.source_url.startswith("https://") and r.checked_at == "2026-11-08"
        assert len(r.headline) <= 110 and "safe" not in r.headline.lower()
        assert r.until is None  # nothing here expires on its own; "at least" dates live in detail
    assert idx.ramp and idx.ramp.review_state == "pending_human_review"
    by = {r.id: r for r in idx.records}
    dc = by["dungeness-commercial-2026-27"]
    assert (dc.level, dc.cause, dc.next_review) == ("delayed", "whale_entanglement", "2026-12-15")
    assert "at least January 1, 2027" in dc.detail and "not a domoic acid closure" in dc.detail
    assert by["dungeness-recreational-2026-27"].effective == "2026-11-07"
    an = by["anchovy-monterey-bay-2026-09"]
    assert (an.level, an.cause, an.effective) == ("restricted", "domoic_acid", "2026-09-11") and "port-452" not in an.applies_to_places
    assert by["mussel-quarantine-2026"].level == "watch"  # announced end held for a person to confirm
    assert by["salmon-commercial-2026"].level == "closed"


def test_ramp_zones_and_pilot_ports():
    ramp = fs.build(NOW).ramp
    zones = {z.zone: z for z in ramp.zones}
    assert set(zones) == {"1", "2", "3", "4", "5"}
    assert zones["4"].applies_to_places == ["port-592", "port-550", "port-593"] and zones["3"].applies_to_places == ["port-452"]
    assert zones["4"].lat_north == pytest.approx(37 + 11 / 60, abs=1e-3) and zones["4"].lat_south == 36.0  # Pigeon Point, Lopez Point
    for z in zones.values():
        assert z.lat_north > z.lat_south
    assert ramp.last_assessment == "2026-10-30" and ramp.next_assessment == "2026-12-15"


def test_yaml_cannot_claim_a_review(curated):
    raw = yaml.safe_load((curated / "status.yaml").read_text())
    raw["records"][0]["review_state"] = "human_verified"
    (curated / "status.yaml").write_text(yaml.safe_dump(raw))
    with pytest.raises(fs.CurationError, match="review_state may not be set"):
        fs.build(NOW, curated)


def test_review_requires_a_name_and_confirm(curated):
    assert main(["review-fisheries-status", "--curated", str(curated), "--reviewer", "Jane Doe"]) == 2
    assert main(["review-fisheries-status", "--curated", str(curated), "--reviewer", "claude", "--confirm"]) == 2
    assert main(["review-fisheries-status", "--curated", str(curated), "--reviewer", "  ", "--confirm"]) == 2
    assert not (curated / "review.json").exists()
    assert all(r.review_state == "pending_human_review" for r in fs.build(NOW, curated).records)


def test_a_human_review_verifies_until_the_record_changes(curated):
    assert main(["review-fisheries-status", "--curated", str(curated), "--reviewer", "Jane Doe", "--ids", "salmon-commercial-2026", "--confirm"]) == 0
    by = {r.id: r for r in fs.build(NOW, curated).records}
    assert by["salmon-commercial-2026"].review_state == "human_verified" and by["salmon-commercial-2026"].reviewed_by == "Jane Doe"
    assert by["dungeness-commercial-2026-27"].review_state == "pending_human_review"
    # editing a reviewed record makes it pending again
    raw = yaml.safe_load((curated / "status.yaml").read_text())
    next(r for r in raw["records"] if r["id"] == "salmon-commercial-2026")["detail"] += " Edited."
    (curated / "status.yaml").write_text(yaml.safe_dump(raw))
    assert {r.id: r for r in fs.build(NOW, curated).records}["salmon-commercial-2026"].review_state == "pending_human_review"


def test_pipeline_writes_the_sidecar_outside_the_manifest(tmp_path):
    out = tmp_path / "v1"
    run_pipeline(fixture_context(out))
    data = json.loads((out / "fisheries-status" / "index.json").read_text())
    FisheriesStatusIndex.model_validate(data)
    assert "fisheries-status" not in (out / "manifest.json").read_text()


def test_committed_schema_has_the_contract_fields():
    sch = json.loads((fs.REPO / "schemas" / "v1" / "fisheries-status.schema.json").read_text())
    assert set(sch["required"]) == {"version", "generated_at", "records", "ramp"}
    assert set(sch["$defs"]["FisheryStatus"]["required"]) == {
        "id", "fishery", "label", "level", "cause", "headline", "detail", "area_label", "applies_to_places",
        "effective", "until", "next_review", "source_name", "source_url", "checked_at", "review_state",
    }
