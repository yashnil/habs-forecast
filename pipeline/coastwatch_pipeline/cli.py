"""Command line entry point: `cwp run | schema | verify-charm | verify-satellite | verify-currents | health | fixture | …`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .context import RunContext
from .models import SCHEMA_MODELS
from .pipeline import ALL_SOURCES, manifest_summary, run_pipeline

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "coastwatch-web" / "public" / "data" / "v1"


def cmd_run(a: argparse.Namespace) -> int:
    only = tuple(a.only.split(",")) if a.only else ALL_SOURCES
    unknown = set(only) - set(ALL_SOURCES)
    if unknown:
        print(f"unknown sources: {sorted(unknown)}", file=sys.stderr)
        return 2
    m = run_pipeline(RunContext(out_dir=Path(a.out)), only)
    print(manifest_summary(m))
    failed = [s.source_id for s in m.sources if s.outcome == "failed" and s.source_id in only]
    # Exit non-zero so CI makes the failure visible; artifacts for healthy sources were still written.
    return 1 if failed and a.strict else 0


def cmd_schema(a: argparse.Namespace) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_MODELS.items():
        schema = model.model_json_schema()
        schema["$id"] = f"https://coastwatch/schemas/v1/{name}.schema.json"
        (out / f"{name}.schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out / f'{name}.schema.json'}")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from .models import Manifest
    from .verify import verify_charm

    if a.only_if_updated:
        m = Manifest.model_validate_json((Path(a.out) / "manifest.json").read_text())
        st = next((s for s in m.sources if s.source_id == "charm"), None)
        if not st or st.outcome not in ("updated", "partial"):
            # Nothing new was published from C-HARM (failed or unchanged): the previous run
            # was verified when it was published. Do not block publishing a failure status.
            print(f"skip: C-HARM outcome is {st.outcome if st else 'absent'}; no new forecast to verify")
            return 0
    report = verify_charm(Path(a.out), live=not a.offline)
    path = Path(a.report) if a.report else Path(a.out) / "verification" / "charm-points.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    for row in [r for r in report["rows"] if r.get("status") == "FAIL"][:10]:
        print("FAIL", json.dumps(row)[:400])
    print(f"report: {path}")
    return 0 if report["summary"]["all_passed"] else 1


def cmd_health(a: argparse.Namespace) -> int:
    from .health import markdown, run_health

    h = run_health(Path(a.out), a.previous_generated_at or None)
    md = markdown(h, a.run_url or "")
    if a.markdown:
        Path(a.markdown).write_text(md + "\n")
    if a.github_output:
        with open(a.github_output, "a") as f:
            f.write(f"alerts={len(h['alerts'])}\nnew_alerts={len(h['new_alerts'])}\ncleared={len(h['cleared_alerts'])}\n")
    print(md)
    return 0  # alerts are reported, never a reason to withhold valid data


def cmd_verify_currents(a: argparse.Namespace) -> int:
    from .models import Manifest
    from .verify_currents import verify_currents

    if a.only_if_updated:
        m = Manifest.model_validate_json((Path(a.out) / "manifest.json").read_text())
        st = next((s for s in m.sources if s.source_id == "hf_radar"), None)
        if not st or st.outcome not in ("updated", "partial"):
            print(f"skip: HF-radar outcome is {st.outcome if st else 'absent'}; nothing new to verify")
            return 0
    report = verify_currents(Path(a.out), live=not a.offline)
    path = Path(a.report) if a.report else Path(a.out) / "verification" / "currents-points.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    for row in [r for r in report["rows"] if r.get("status") == "FAIL"][:10]:
        print("FAIL", json.dumps(row)[:400])
    print(f"report: {path}")
    return 0 if report["summary"]["all_passed"] else 1


def cmd_verify_satellite(a: argparse.Namespace) -> int:
    from .models import Manifest
    from .verify_satellite import verify_satellite

    if a.only_if_updated:
        m = Manifest.model_validate_json((Path(a.out) / "manifest.json").read_text())
        st = next((s for s in m.sources if s.source_id == "satellite_chl"), None)
        if not st or st.outcome not in ("updated", "partial"):
            print(f"skip: satellite outcome is {st.outcome if st else 'absent'}; nothing new to verify")
            return 0
    report = verify_satellite(Path(a.out), live=not a.offline)
    path = Path(a.report) if a.report else Path(a.out) / "verification" / "satellite-points.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    for row in [r for r in report["rows"] if r.get("status") == "FAIL"][:10]:
        print("FAIL", json.dumps(row)[:400])
    print(f"report: {path}")
    return 0 if report["summary"]["all_passed"] else 1


def cmd_fixture(a: argparse.Namespace) -> int:
    from .fixtures import build_fixture_dataset

    m = build_fixture_dataset(Path(a.out), now=a.now, scenario=a.scenario)
    print(manifest_summary(m))
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    from .publish.check import check_published

    import time

    report = check_published(a.base, a.expect_run)
    for _ in range(a.retries):
        if report["ok"]:
            break
        time.sleep(a.wait)
        report = check_published(a.base, a.expect_run)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    summary = {k: report[k] for k in ("base", "ok", "files_checked", "tiles_validated", "problems") if k in report}
    print(json.dumps(summary, indent=2))
    return 0 if report["ok"] else 1


def cmd_guard(a: argparse.Namespace) -> int:
    from .publish.check import main_guard

    return main_guard(a.root)


def cmd_watch(a: argparse.Namespace) -> int:
    """Report the official pages against the last human review, with memory: `--state` carries
    conditions from run to run so a notification is due only when something appears, changes
    or resolves (watch_official.py). Read-only for the registry."""
    from .watch_official import markdown, run_watch

    state = run_watch(RunContext(out_dir=Path(".")), Path(a.registry), Path(a.state) if a.state else None, a.run_url or "")
    md = markdown(state, a.run_url or "")
    if a.markdown:
        Path(a.markdown).write_text(md + "\n")
    print(json.dumps({k: v for k, v in state.items() if k != "history"}, indent=2, ensure_ascii=False))
    if a.github_output:
        with open(a.github_output, "a") as fh:
            fh.write(f"notify={'true' if state['notify'] else 'false'}\n")
            fh.write(f"conditions={len(state['conditions'])}\n")
            fh.write(f"marker={state['marker']}\n")
            # legacy flag: anything at all differs from the last review (not a reason to notify)
            fh.write(f"changed={'true' if state['conditions'] else 'false'}\n")
    return 0


def cmd_review(a: argparse.Namespace) -> int:
    """Run by a person AFTER checking every record against the official pages."""
    from .sources import official

    if a.status == "human_verified" and not a.confirm:
        print("Refusing: pass --confirm to state that you checked every record against the official sources.", file=sys.stderr)
        return 2
    review = official.record_review(Path(a.registry), RunContext(out_dir=Path(".")), a.reviewer, a.status, a.method)
    print(json.dumps(review, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cwp", description="Neritic data pipeline (formerly CoastWatch)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="fetch, validate, render and publish artifacts")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--only", help=f"comma list of {','.join(ALL_SOURCES)}")
    r.add_argument("--strict", action="store_true", help="exit 1 if any selected source failed")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("schema", help="export JSON Schema")
    s.add_argument("--out", default=str(REPO / "schemas" / "v1"))
    s.set_defaults(fn=cmd_schema)
    v = sub.add_parser("verify-charm", help="compare published C-HARM artifacts with ERDDAP at fixed points")
    v.add_argument("--out", default=str(DEFAULT_OUT))
    v.add_argument("--report")
    v.add_argument("--offline", action="store_true", help="skip live ERDDAP point queries")
    v.add_argument("--only-if-updated", action="store_true", help="skip unless this run published new C-HARM data")
    v.set_defaults(fn=cmd_verify)
    vs = sub.add_parser("verify-satellite", help="check published satellite chlorophyll against ERDDAP, tiles and composites")
    vs.add_argument("--out", default=str(DEFAULT_OUT))
    vs.add_argument("--report")
    vs.add_argument("--offline", action="store_true", help="skip live ERDDAP point queries")
    vs.add_argument("--only-if-updated", action="store_true", help="skip unless this run published new satellite data")
    vs.set_defaults(fn=cmd_verify_satellite)
    vc = sub.add_parser("verify-currents", help="check published HF-radar currents against ERDDAP and the 24-hour mean against the hours")
    vc.add_argument("--out", default=str(DEFAULT_OUT))
    vc.add_argument("--report")
    vc.add_argument("--offline", action="store_true", help="skip live ERDDAP point queries")
    vc.add_argument("--only-if-updated", action="store_true", help="skip unless this run published new HF-radar data")
    vc.set_defaults(fn=cmd_verify_currents)
    hp = sub.add_parser("health", help="evaluate pipeline health (alerts) and update health.json in the dataset")
    hp.add_argument("--out", default=str(DEFAULT_OUT))
    hp.add_argument("--previous-generated-at", help="generated_at of the dataset this run started from")
    hp.add_argument("--markdown", help="write the alert summary (issue body) here")
    hp.add_argument("--github-output", help="append alerts/new_alerts/cleared counts for GitHub Actions")
    hp.add_argument("--run-url")
    hp.set_defaults(fn=cmd_health)
    f = sub.add_parser("fixture", help="build a deterministic dataset from recorded fixtures (no network)")
    f.add_argument("--out", required=True)
    f.add_argument("--now", default="2026-10-08T18:00:00Z")
    f.add_argument("--scenario", default="normal", choices=["normal", "charm-failed"])
    f.set_defaults(fn=cmd_fixture)
    c = sub.add_parser("check-published", help="validate a published dataset (URL or directory) as a browser sees it")
    c.add_argument("--base", required=True, help="e.g. https://raw.githubusercontent.com/<owner>/<repo>/coastwatch-data/v1")
    c.add_argument("--report")
    c.add_argument("--expect-run", help="fail unless the manifest comes from this pipeline run id")
    c.add_argument("--retries", type=int, default=0, help="re-check this many times (e.g. while a CDN cache expires)")
    c.add_argument("--wait", type=float, default=20.0)
    c.set_defaults(fn=cmd_check)
    g = sub.add_parser("guard-publish", help="refuse to publish anything except v1/ pipeline artifacts")
    g.add_argument("--root", required=True)
    g.set_defaults(fn=cmd_guard)
    reg_default = str(REPO / "data" / "curated" / "official_notices.json")
    w = sub.add_parser("watch-official", help="check official pages for changes since the last review (never edits records)")
    w.add_argument("--registry", default=reg_default)
    w.add_argument("--state", help="state file carried between runs (in the dataset: v1/official-watch.json); without it every condition counts as new")
    w.add_argument("--markdown", help="write the per-source review report (issue body) here")
    w.add_argument("--github-output", help="append notify/conditions/marker (and legacy changed) for GitHub Actions")
    w.add_argument("--run-url")
    w.set_defaults(fn=cmd_watch)
    rv = sub.add_parser("review-official", help="record a human review of the official notices registry")
    rv.add_argument("--registry", default=reg_default)
    rv.add_argument("--reviewer", required=True)
    rv.add_argument("--status", choices=["human_verified", "pending_human_review"], default="human_verified")
    rv.add_argument("--method", default="Each record compared with the linked CDFW/CDPH pages; watched pages fingerprinted.")
    rv.add_argument("--confirm", action="store_true")
    rv.set_defaults(fn=cmd_review)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
