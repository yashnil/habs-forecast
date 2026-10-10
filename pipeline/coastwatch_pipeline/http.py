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
    pass


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
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            deadline = time.monotonic() + max_seconds
            with urllib.request.urlopen(req, timeout=timeout, context=tls_context()) as r:
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
                raise FetchError(f"HTTP {e.code} for {url}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        if attempt < retries:
            time.sleep(backoff * (2**attempt))
    raise FetchError(f"Failed after {retries + 1} attempts: {url} ({last})")


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
    """

    def __init__(self, inner: Fetcher = fetch, threshold: int = 2):
        self.inner, self.threshold = inner, threshold
        self.failures: dict[str, int] = {}
        self.opened: dict[str, str] = {}

    @staticmethod
    def key(url: str) -> str:
        from urllib.parse import urlsplit

        u = urlsplit(url)
        path = u.path.rsplit(".", 1)[0] if "." in u.path.rsplit("/", 1)[-1] else u.path
        return f"{u.netloc}{path}"

    def __call__(self, url: str) -> Response:
        k = self.key(url)
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
                raise
            self.failures[k] = self.failures.get(k, 0) + 1
            if self.failures[k] >= self.threshold:
                self.opened[k] = str(e)[:200]
            raise
        self.failures[k] = 0
        return r


Poster = Callable[[str, bytes], Response]


def post_json(url: str, body: bytes) -> Response:
    """POST a JSON body (used for the BLS public API, which only serves ranges via POST)."""
    return fetch(url, method="POST", data=body, headers={"Content-Type": "application/json"})
