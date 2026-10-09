from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .http import Fetcher, Poster, fetch, post_json


def _git_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha[:12]
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class RunContext:
    out_dir: Path
    fetcher: Fetcher = fetch
    poster: Poster = post_json
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    pipeline_version: str = field(default_factory=_git_sha)
    # Per-run options, e.g. a smaller satellite domain for fixtures and tests.
    options: dict = field(default_factory=dict)
    run_id: str = field(
        default_factory=lambda: os.environ.get("GITHUB_RUN_ID")
        or "local-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    @property
    def now_iso(self) -> str:
        return iso(self.now)
