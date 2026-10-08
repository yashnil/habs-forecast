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


def fetch(
    url: str,
    *,
    timeout: float = 90.0,
    retries: int = 3,
    backoff: float = 5.0,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> Response:
    last: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=tls_context()) as r:
                return Response(
                    url=url,
                    status=r.status,
                    content_type=r.headers.get("Content-Type", ""),
                    body=r.read() if method != "HEAD" else b"",
                )
        except urllib.error.HTTPError as e:
            # 4xx other than 429 will not get better by retrying
            if 400 <= e.code < 500 and e.code != 429:
                raise FetchError(f"HTTP {e.code} for {url}") from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        if attempt < retries:
            time.sleep(backoff * (2**attempt))
    raise FetchError(f"Failed after {retries + 1} attempts: {url} ({last})")
