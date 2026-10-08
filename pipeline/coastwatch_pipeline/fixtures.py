"""Offline fixture fetcher and deterministic dataset builder.

Fixtures in tests/fixtures were recorded from the live services on 2026-10-08 by
scripts/record_fixtures.py (C-HARM is a Monterey Bay / Farallones subset). They let the
pipeline and the web app be tested without network access.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .context import RunContext
from .http import FetchError, Response
from .models import Manifest
from .pipeline import run_pipeline

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FIXTURE_NOW = "2026-10-08T18:00:00Z"


class FixtureFetcher:
    """Serves recorded responses by URL pattern. `overrides` maps a regex to a callable
    returning a Response (or raising) so tests can inject failures and malformed data."""

    def __init__(self, root: Path = FIXTURES, overrides: dict[str, Callable[[str], Response]] | None = None):
        self.root = root
        self.overrides = overrides or {}
        self.calls: list[str] = []

    def __call__(self, url: str) -> Response:
        self.calls.append(url)
        for pattern, fn in self.overrides.items():
            if re.search(pattern, url):
                return fn(url)
        return self._serve(url)

    def _file(self, url: str, rel: str, ctype: str) -> Response:
        p = self.root / rel
        if not p.exists():
            raise FetchError(f"HTTP 404 for {url}")
        return Response(url=url, status=200, content_type=ctype, body=p.read_bytes())

    def _serve(self, url: str) -> Response:
        m = re.search(r"wvcharmV3_(\d)day\.csv0\?time", url)
        if m:
            return self._file(url, f"charm/lead{m.group(1)}_time.csv", "text/csv")
        m = re.search(r"wvcharmV3_(\d)day\.nc\?", url)
        if m:
            return self._file(url, f"charm/lead{m.group(1)}.nc", "application/x-netcdf")
        if url.endswith("WMTSCapabilities.xml"):
            return self._file(url, "gibs/capabilities.xml", "application/xml")
        m = re.search(r"/default/(\d{4}-\d{2}-\d{2})/", url)
        if m and "gibs.earthdata.nasa.gov" in url:
            caps = (self.root / "gibs" / "capabilities.xml").read_text()
            defaults = re.findall(r"<Default>([^<]+)</Default>", caps)
            from datetime import date, timedelta

            d = m.group(1)
            newest = {x for x in defaults}
            previous = {(date.fromisoformat(x) - timedelta(days=1)).isoformat() for x in defaults}
            if d in newest:
                return self._file(url, "gibs/tile_newest.png", "image/png")
            if d in previous:
                return self._file(url, "gibs/tile_previous.png", "image/png")
            raise FetchError(f"HTTP 404 for {url}")
        if "/legends/" in url and url.endswith(".svg"):
            return self._file(url, "gibs/legend_H.svg", "image/svg+xml")
        if "biosds3081_fpu" in url:
            return self._file(url, "cdfw_ds3081_ports.json", "application/json")
        raise FetchError(f"no fixture for {url}")


def fixture_context(out: Path, now: str = FIXTURE_NOW, fetcher: FixtureFetcher | None = None) -> RunContext:
    return RunContext(
        out_dir=out,
        fetcher=fetcher or FixtureFetcher(),
        now=datetime.fromisoformat(now.replace("Z", "+00:00")).astimezone(timezone.utc),
        pipeline_version="fixture",
        run_id="fixture",
    )


def build_fixture_dataset(out: Path, now: str = FIXTURE_NOW, scenario: str = "normal") -> Manifest:
    """Scenarios: 'normal' (all sources succeed) or 'charm-failed' (a later run in which
    every C-HARM request fails, so the previous run is kept and the failure recorded)."""
    import json

    from .verify import verify_charm

    if out.exists():
        shutil.rmtree(out)
    m = run_pipeline(fixture_context(out, now))
    if scenario == "charm-failed":
        later = (datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%SZ")
        fetcher = FixtureFetcher(overrides={r"wvcharmV3_": _fail_502})
        m = run_pipeline(fixture_context(out, later, fetcher))
    elif scenario != "normal":
        raise ValueError(f"unknown scenario {scenario}")
    report = verify_charm(out, live=False)
    (out / "verification").mkdir(exist_ok=True)
    (out / "verification" / "charm-points.json").write_text(json.dumps(report, indent=2) + "\n")
    return m


def _fail_502(url: str) -> Response:
    raise FetchError(f"HTTP 502 for {url}")
