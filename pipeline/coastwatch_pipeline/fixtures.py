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

# Bounding boxes of the satellite fixtures (real ERDDAP responses, recorded 2026-10-09).
SATELLITE_REGIONS = {
    "monterey": (36.40125, 37.20125, -122.60125, -121.75125),
    "north_coast": (40.40125, 41.40125, -124.70125, -123.95125),
    "socal_bight": (32.60125, 34.10125, -119.00125, -117.10125),
}


class FixtureFetcher:
    """Serves recorded responses by URL pattern. `overrides` maps a regex to a callable
    returning a Response (or raising) so tests can inject failures and malformed data."""

    def __init__(self, root: Path = FIXTURES, overrides: dict[str, Callable[[str], Response]] | None = None, satellite_region: str = "monterey"):
        self.root = root
        self.overrides = overrides or {}
        self.calls: list[str] = []
        self.satellite_region = satellite_region

    def __call__(self, url: str) -> Response:
        self.calls.append(url)
        for pattern, fn in self.overrides.items():
            if re.search(pattern, url):
                return fn(url)
        return self._serve(url)

    def post(self, url: str, body: bytes) -> Response:
        """POST requests are recorded under '<url>#<sha256(body)[:16]>'."""
        import hashlib

        key = f"{url}#{hashlib.sha256(body).hexdigest()[:16]}"
        self.calls.append(key)
        for pattern, fn in self.overrides.items():
            if re.search(pattern, key):
                return fn(key)
        rec = self._recorded(key)
        if rec is None:
            raise FetchError(f"no fixture for POST {key}")
        return rec

    def _file(self, url: str, rel: str, ctype: str) -> Response:
        p = self.root / rel
        if not p.exists():
            raise FetchError(f"HTTP 404 for {url}")
        return Response(url=url, status=200, content_type=ctype, body=p.read_bytes())

    def _recorded(self, url: str) -> Response | None:
        """Responses recorded by scripts/record_m2_fixtures.py, keyed by exact URL."""
        idx = self.root / "recorded" / "index.json"
        if not idx.exists():
            return None
        import json

        entry = json.loads(idx.read_text()).get(url)
        if not entry:
            return None
        return Response(url=url, status=200, content_type=entry["content_type"], body=(self.root / "recorded" / entry["file"]).read_bytes())

    def _satellite(self, url: str) -> Response | None:
        import json
        from urllib.parse import unquote

        m = re.search(r"/griddap/((?:noaacwS3[AB]OLCIchlaSector[A-Z]{2}Daily)|erdVHNchla1day)\.(csv0|nc)\?(.*)$", url)
        if not m:
            return None
        ds, kind, q = m.group(1), m.group(2), unquote(m.group(3))
        idx = json.loads((self.root / "satellite" / "index.json").read_text())
        entries = idx.get(self.satellite_region, {}).get(ds, [])
        if kind == "csv0":
            start = re.search(r"time\[\((\d{4}-\d\d-\d\d)", q)
            times = [e["time"] for e in entries]
            if start and times:
                # like ERDDAP, the start bound snaps to the nearest available time, which can
                # be before it (e.g. the previous evening's overpass)
                from datetime import datetime

                t0 = datetime.fromisoformat(f"{start.group(1)}T00:00:00+00:00")
                gap = lambda t: abs((datetime.fromisoformat(t.replace("Z", "+00:00")) - t0).total_seconds())  # noqa: E731
                times = times[times.index(min(times, key=gap)):]
            if not times:
                raise FetchError(f"HTTP 404 for {url} (no matching results)")
            return Response(url=url, status=200, content_type="text/csv", body=("\n".join(times) + "\n").encode())
        t = re.search(r"\[\((\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\)\]", q)
        hit = next((e for e in entries if t and e["time"] == t.group(1)), None)
        if not hit:
            raise FetchError(f"HTTP 404 for {url}")
        return self._file(url, f"satellite/{hit['file']}", "application/x-netcdf")

    def _serve(self, url: str) -> Response:
        rec = self._recorded(url)
        if rec is not None:
            return rec
        sat = self._satellite(url)
        if sat is not None:
            return sat
        m = re.search(r"wvcharmV3_(\d)day\.csv0\?time", url)
        if m:
            return self._file(url, f"charm/lead{m.group(1)}_time.csv", "text/csv")
        # single-time lead requests only; time-range (history) requests must be recorded explicitly
        m = re.search(r"wvcharmV3_(\d)day\.nc\?pseudo_nitzschia%5B\(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\)%5D", url)
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
    from .sources.satellite import Domain

    f = fetcher or FixtureFetcher()
    lat_s, lat_n, lon_w, lon_e = SATELLITE_REGIONS[f.satellite_region]
    return RunContext(
        out_dir=out,
        options={"satellite_domain": Domain(lat_s, lat_n, lon_w, lon_e)},
        fetcher=f,
        poster=f.post,
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
    from .verify_satellite import verify_satellite

    report = verify_charm(out, live=False)
    (out / "verification").mkdir(exist_ok=True)
    (out / "verification" / "charm-points.json").write_text(json.dumps(report, indent=2) + "\n")
    sat = verify_satellite(out, live=False)
    (out / "verification" / "satellite-points.json").write_text(json.dumps(sat, indent=2) + "\n")
    return m


def _fail_502(url: str) -> Response:
    raise FetchError(f"HTTP 502 for {url}")
