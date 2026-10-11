"""Minimal HTTP client with retries. Upstream servers (ERDDAP, GIBS, ArcGIS) return
intermittent 502/503s, so every fetch retries with backoff before failing loudly."""

from __future__ import annotations

import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

USER_AGENT = "CoastWatch-pipeline/0.1 (+https://github.com/yashnil/habs-forecast)"


CERT_DIR = Path(__file__).parent / "certs"


@lru_cache(maxsize=1)
def tls_context() -> ssl.SSLContext:
    """System trust store plus public intermediates that some agency servers fail to send
    (see certs/*.pem). Verification is never disabled."""
    ctx = ssl.create_default_context()
    for pem in sorted(CERT_DIR.glob("*.pem")):
        ctx.load_verify_locations(cafile=str(pem))
    return ctx


class FetchError(RuntimeError):
    def __init__(self, message: str, detail: str | None = None, *, hang: bool = False, budget: bool = False):
        super().__init__(message)
        self.detail = detail  # start of the server's error message, when it sent one
        # every attempt timed out or could not connect: the server gave no answer at all
        self.hang = hang
        # not requested (or cut short) because the source's time budget was used up
        self.budget = budget


# ---------------------------------------------------------------- time budgets
# One source whose server hangs must not hold the whole refresh past the job's time limit
# (run 38116032798 on 2026-10-11: SCCOOS's ERDDAP hung, ~400 s per station, and the job was
# cancelled at 40 min with nothing published). The pipeline sets a budget per source and one
# for the run; fetch() never waits past the earlier of the two, and fails fast once it is
# used up. The source then fails like any other outage and keeps its last good data.
_budget: dict[str, tuple[str, float, float] | None] = {"source": None, "run": None}


def start_budget(label: str, seconds: float, scope: str = "source") -> None:
    """Start (or replace) the time budget for `scope` ("source" or "run")."""
    _budget[scope] = (label, seconds, time.monotonic() + seconds)


def clear_budgets() -> None:
    _budget["source"] = _budget["run"] = None


def _remaining() -> tuple[float, str] | None:
    """Seconds left before the nearest budget ends, and which budget that is."""
    live = [b for b in _budget.values() if b is not None]
    if not live:
        return None
    label, seconds, deadline = min(live, key=lambda b: b[2])
    return deadline - time.monotonic(), f"{label} time budget ({seconds:.0f} s)"


@dataclass
class Response:
    url: str
    status: int
    content_type: str
    body: bytes


Fetcher = Callable[[str], Response]


def _read_within(r, deadline: float, url: str) -> bytes:
    """Read the whole body, but give up at `deadline` (monotonic seconds): the socket
    timeout applies per read, so a server trickling bytes could otherwise stall a run."""
    chunks = []
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"download exceeded its time limit: {url}")
        b = r.read(1 << 16)
        if not b:
            return b"".join(chunks)
        chunks.append(b)


def fetch(
    url: str,
    *,
    timeout: float = 90.0,
    retries: int = 3,
    backoff: float = 5.0,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
    max_seconds: float = 240.0,
) -> Response:
    last: Exception | None = None
    answered = False  # the server sent an HTTP response on some attempt
    for attempt in range(retries + 1):
        rem = _remaining()
        if rem is not None and rem[0] <= 1:
            raise FetchError(f"{rem[1]} used up; not requested: {url}" + (f" (last: {last})" if last else ""), budget=True)
        t_out, t_max = (timeout, max_seconds) if rem is None else (min(timeout, rem[0]), min(max_seconds, rem[0]))
        req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            deadline = time.monotonic() + t_max
            with urllib.request.urlopen(req, timeout=t_out, context=tls_context()) as r:
                return Response(
                    url=url,
                    status=r.status,
                    content_type=r.headers.get("Content-Type", ""),
                    body=_read_within(r, deadline, url) if method != "HEAD" else b"",
                )
        except urllib.error.HTTPError as e:
            # 4xx other than 429 will not get better by retrying, except 403: NOAA's
            # ERDDAP intermittently answers 403 to cloud runners (seen in staging run
            # 37964087582, 2026-10-09) and serves the same request moments later
            if 400 <= e.code < 500 and e.code not in (403, 429):
                # keep the start of the server's message: ERDDAP answers 404 both for "no
                # matching results" (an answer) and "unknown datasetID" (an outage)
                try:
                    msg = " ".join(e.read(400).decode("utf-8", "replace").split())
                except Exception:
                    msg = ""
                raise FetchError(f"HTTP {e.code} for {url}" + (f" ({msg[:240]})" if msg else ""), detail=msg[:240] or None) from e
            last, answered = e, True
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        if attempt < retries:
            wait = backoff * (2**attempt)
            rem = _remaining()
            if rem is not None and rem[0] <= wait + 1:
                raise FetchError(f"{rem[1]} used up after {attempt + 1} attempts: {url} ({last})", hang=not answered, budget=True)
            time.sleep(wait)
    raise FetchError(f"Failed after {retries + 1} attempts: {url} ({last})", hang=not answered)


def _log(msg: str) -> None:
    import sys

    print(f"[http] {msg}", file=sys.stderr, flush=True)


class CircuitBreaker:
    """Per-run fetcher that stops calling an upstream endpoint after it has failed
    `threshold` times in a row (each failure already includes fetch()'s retries).

    The key is host + path without the query or extension, i.e. one ERDDAP dataset:
    NOAA's ERDDAP has refused one dataset while serving another from the same host
    (staging run 37967595146), so a whole-host breaker would drop working sources.
    Bounds a run's time when an endpoint refuses everything (~35 s per request
    otherwise) and makes the outcome explicit; nothing is written for failed requests.

    Hangs are also counted per host: a server that answers nothing (every attempt timed out
    or could not connect) is down for all its datasets, and each dataset costs minutes to
    find that out (SCCOOS's ERDDAP serves one dataset per CalHABMAP station). After
    `host_threshold` consecutive hangs the host is skipped for the rest of the run. A host
    that answers, even with errors, is never skipped this way.
    """

    def __init__(self, inner: Fetcher = fetch, threshold: int = 2, host_threshold: int = 2):
        self.inner, self.threshold, self.host_threshold = inner, threshold, host_threshold
        self.failures: dict[str, int] = {}
        self.opened: dict[str, str] = {}
        self.host_hangs: dict[str, int] = {}
        self.hosts_down: dict[str, str] = {}

    @staticmethod
    def key(url: str) -> str:
        from urllib.parse import urlsplit

        u = urlsplit(url)
        path = u.path.rsplit(".", 1)[0] if "." in u.path.rsplit("/", 1)[-1] else u.path
        return f"{u.netloc}{path}"

    def __call__(self, url: str) -> Response:
        from urllib.parse import urlsplit

        k, host = self.key(url), urlsplit(url).netloc
        if host in self.hosts_down:
            raise FetchError(f"{host} skipped for the rest of this run: no answer to {self.host_threshold} requests in a row (last: {self.hosts_down[host]}): {url}", hang=True)
        if k in self.opened:
            raise FetchError(f"circuit open for {k} after {self.threshold} consecutive failures this run (last: {self.opened[k]}): {url}")
        t0 = time.monotonic()
        try:
            r = self.inner(url)
            if time.monotonic() - t0 > 20:
                _log(f"slow fetch {time.monotonic() - t0:.0f}s {len(r.body) / 1e6:.1f} MB {url[:160]}")
        except FetchError as e:
            _log(f"fetch failed after {time.monotonic() - t0:.0f}s: {str(e)[:200]}")
            if "HTTP 404" in str(e):  # a definite answer, not an outage
                self.host_hangs[host] = 0
                raise
            if e.budget and not e.hang:  # the budget ran out; says nothing about the server
                raise
            self.failures[k] = self.failures.get(k, 0) + 1
            if self.failures[k] >= self.threshold:
                self.opened[k] = str(e)[:200]
            if e.hang:
                self.host_hangs[host] = self.host_hangs.get(host, 0) + 1
                if self.host_hangs[host] >= self.host_threshold:
                    self.hosts_down[host] = str(e)[:200]
            else:
                self.host_hangs[host] = 0
            raise
        self.failures[k] = 0
        self.host_hangs[host] = 0
        return r


Poster = Callable[[str, bytes], Response]


def post_json(url: str, body: bytes) -> Response:
    """POST a JSON body (used for the BLS public API, which only serves ranges via POST)."""
    return fetch(url, method="POST", data=body, headers={"Content-Type": "application/json"})
