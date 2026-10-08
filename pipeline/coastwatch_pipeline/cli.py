"""Command line entry point: `cwp run | schema | verify-charm | fixture`."""

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
    from .verify import verify_charm

    report = verify_charm(Path(a.out), live=not a.offline)
    path = Path(a.report) if a.report else Path(a.out) / "verification" / "charm-points.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"report: {path}")
    return 0 if report["summary"]["all_passed"] else 1


def cmd_fixture(a: argparse.Namespace) -> int:
    from .fixtures import build_fixture_dataset

    m = build_fixture_dataset(Path(a.out), now=a.now)
    print(manifest_summary(m))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cwp", description="CoastWatch pipeline")
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
    v.set_defaults(fn=cmd_verify)
    f = sub.add_parser("fixture", help="build a deterministic dataset from recorded fixtures (no network)")
    f.add_argument("--out", required=True)
    f.add_argument("--now", default="2026-10-08T18:00:00Z")
    f.set_defaults(fn=cmd_fixture)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
