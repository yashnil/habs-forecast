"""CLI: python -m cwi ask "..." | python -m cwi eval | python -m cwi payload "..."."""
from __future__ import annotations

import argparse
import json
import sys

from .agent import SYSTEM_PROMPT, run
from .providers import AnthropicDryRun, MockProvider
from .snapshot import DEFAULT_SNAPSHOT, load_snapshot
from .tools import Toolbox


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="cwi")
    ap.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--trace", action="store_true")
    e = sub.add_parser("eval"); e.add_argument("--out", default=None)
    p = sub.add_parser("payload", help="print the Claude request body for the first turn (never sent)")
    p.add_argument("question")
    args = ap.parse_args(argv)

    tb = Toolbox(load_snapshot(args.snapshot))
    if args.cmd == "ask":
        t = run(args.question, MockProvider(tb), tb)
        print(t.answer)
        if args.trace:
            print("\n--- trace", file=sys.stderr)
            print(json.dumps({"outcome": t.outcome, "tool_calls": t.tool_calls,
                              "first_guard": [v.__dict__ for v in t.first_report.violations]}, indent=1), file=sys.stderr)
        return 0
    if args.cmd == "payload":
        dry = AnthropicDryRun()
        print(json.dumps(dry.build(SYSTEM_PROMPT, [{"role": "user", "content": args.question}], tb.specs()), indent=1))
        return 0
    from .evaluate import main as eval_main
    return eval_main(tb, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
