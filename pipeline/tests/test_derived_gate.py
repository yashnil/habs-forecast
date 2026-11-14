"""The experimental derived layers (v1/derived/) are staging-only: off by default, never written,
pruned or published by a production run (owner decision for the V15.1 public release)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coastwatch_pipeline import cli
from coastwatch_pipeline.fixtures import fixture_context
from coastwatch_pipeline.pipeline import DERIVED_ENV, derived_enabled, run_pipeline
from coastwatch_pipeline.publish.check import check_published, guard_publish_tree

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "coastwatch-data.yml"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv(DERIVED_ENV, raising=False)


@pytest.mark.parametrize("value,expected", [(None, False), ("", False), ("0", False), ("no", False), ("1", True), ("true", True), ("YES", True)])
def test_env_switch(monkeypatch, value, expected):
    if value is not None:
        monkeypatch.setenv(DERIVED_ENV, value)
    assert derived_enabled() is expected
    # an explicit flag always wins over the environment
    assert derived_enabled(False) is False and derived_enabled(True) is True


def test_production_run_writes_no_derived_and_still_writes_the_sidecars(out):
    run_pipeline(fixture_context(out))
    assert not (out / "derived").exists()
    assert (out / "manifest.json").exists()
    assert (out / "biotoxin" / "index.json").exists() and (out / "fisheries-status" / "index.json").exists()
    report = check_published(str(out))
    assert report["ok"], report["problems"]
    assert not any(c["path"].startswith("derived/") for c in report["checked"])
    assert guard_publish_tree(out.parent, forbid=("v1/derived/",)) == []


def test_production_run_never_touches_an_existing_derived_folder(out):
    """No create, no prune: a stray v1/derived/ is left exactly as found (and the publish guard,
    not the pipeline, refuses it in production)."""
    stray = out / "derived" / "old-0123456789" / "x.u16.gz"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"x")
    (out / "derived" / "index.json").write_text(json.dumps({"version": 1, "layers": []}))
    before = sorted(p.relative_to(out).as_posix() for p in (out / "derived").rglob("*"))
    run_pipeline(fixture_context(out))
    assert sorted(p.relative_to(out).as_posix() for p in (out / "derived").rglob("*")) == before
    bad = guard_publish_tree(out.parent, forbid=("v1/derived/",))
    assert bad and all(b.startswith("v1/derived/") and b.endswith("(forbidden in this channel)") for b in bad)
    assert guard_publish_tree(out.parent) == []  # staging: allowed


def test_staging_opt_in_by_env_writes_derived(monkeypatch, out):
    monkeypatch.setenv(DERIVED_ENV, "1")
    run_pipeline(fixture_context(out), ("satellite_chl",))
    assert (out / "derived" / "index.json").exists()


def test_cli_run_flag_defaults_off(monkeypatch):
    seen: list[bool] = []
    monkeypatch.setattr(cli, "cmd_run", lambda a: seen.append(a.derived) or 0)
    cli.main(["run", "--out", "/nonexistent"])
    cli.main(["run", "--out", "/nonexistent", "--derived"])
    assert seen == [False, True]


def test_workflow_enables_derived_for_staging_only():
    text = WORKFLOW.read_text()
    assert "CWP_DERIVED: ${{ inputs.channel == 'staging' && '1' || '0' }}" in text
    # production (scheduled runs have no inputs: channel is empty) forbids v1/derived/ at the guard
    assert "${{ inputs.channel != 'staging' && '--forbid v1/derived/' || '' }}" in text
    assert "cwp run --out ../publish/v1 --derived" not in text
