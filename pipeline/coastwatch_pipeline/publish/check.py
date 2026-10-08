"""Check a published dataset the way a browser will see it.

Given a base URL (or a local directory), fetch manifest.json, validate it against the
schema, and confirm that every artifact it references exists, decodes, matches its
declared size, and (for URLs) is served with CORS so the web app can load it.
"""

from __future__ import annotations

import gzip
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..http import USER_AGENT
from ..models import Manifest, PortsCollection

TEST_ORIGIN = "https://coastwatch.example"


@dataclass
class Fetched:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


def _get(base: str, rel: str) -> Fetched:
    if base.startswith(("http://", "https://")):
        url = f"{base.rstrip('/')}/{rel}"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Origin": TEST_ORIGIN})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return Fetched(r.status, r.read(), {k.lower(): v for k, v in r.headers.items()})
        except urllib.error.HTTPError as e:
            return Fetched(e.code, b"", {})
    p = Path(base) / rel
    return Fetched(200, p.read_bytes()) if p.exists() else Fetched(404, b"")


def check_published(base: str, expect_run: str | None = None) -> dict:
    remote = base.startswith(("http://", "https://"))
    problems: list[str] = []
    checked: list[dict] = []

    mf = _get(base, "manifest.json")
    if mf.status != 200:
        return {"base": base, "ok": False, "problems": [f"manifest.json: HTTP {mf.status}"], "checked": []}
    try:
        manifest = Manifest.model_validate_json(mf.body)
    except Exception as e:
        return {"base": base, "ok": False, "problems": [f"manifest.json failed schema validation: {e}"], "checked": []}
    if expect_run and manifest.pipeline_run_id != expect_run:
        problems.append(f"manifest.json is from run {manifest.pipeline_run_id}, expected {expect_run} (not yet propagated?)")
    if remote and mf.headers.get("access-control-allow-origin") not in ("*", TEST_ORIGIN):
        problems.append(f"manifest.json: missing CORS header (got {mf.headers.get('access-control-allow-origin')!r})")

    def record(rel: str, kind: str, f: Fetched, ok: bool, detail: str) -> None:
        cors = f.headers.get("access-control-allow-origin") if remote else "n/a (local)"
        if remote and cors not in ("*", TEST_ORIGIN):
            ok = False
            detail += f"; missing CORS (got {cors!r})"
        checked.append({"path": rel, "kind": kind, "status": f.status, "bytes": len(f.body), "cors": cors, "ok": ok, "detail": detail})
        if not ok:
            problems.append(f"{rel}: {detail}")

    for lyr in manifest.layers:
        if lyr.image:
            f = _get(base, lyr.image.url)
            ok, detail = f.status == 200, f"HTTP {f.status}"
            if ok:
                try:
                    with Image.open(io.BytesIO(f.body)) as im:
                        ok = im.size == (lyr.image.width, lyr.image.height)
                        detail = f"PNG {im.size[0]}x{im.size[1]}" + ("" if ok else f" != declared {lyr.image.width}x{lyr.image.height}")
                except Exception as e:
                    ok, detail = False, f"not a PNG: {e}"
            record(lyr.image.url, "image", f, ok, detail)
        if lyr.grid:
            g = lyr.grid
            f = _get(base, g.url)
            ok, detail = f.status == 200, f"HTTP {f.status}"
            if ok:
                try:
                    n = len(gzip.decompress(f.body))
                    ok = n == g.width * g.height * 2
                    detail = f"{n} bytes decoded" + ("" if ok else f" != {g.width * g.height * 2}")
                except Exception as e:
                    ok, detail = False, f"not gzip: {e}"
            record(g.url, "grid", f, ok, detail)

    if manifest.ports_url:
        f = _get(base, manifest.ports_url)
        ok, detail = f.status == 200, f"HTTP {f.status}"
        if ok:
            try:
                n = len(PortsCollection.model_validate_json(f.body).features)
                detail = f"{n} ports"
            except Exception as e:
                ok, detail = False, f"invalid ports: {e}"
        record(manifest.ports_url, "ports", f, ok, detail)

    return {
        "base": base,
        "manifest_generated_at": manifest.generated_at,
        "pipeline_run_id": manifest.pipeline_run_id,
        "ok": not problems,
        "files_checked": len(checked) + 1,
        "problems": problems,
        "checked": checked,
    }


def guard_publish_tree(root: Path, allowed_top: str = "v1") -> list[str]:
    """Return offending paths if the tree about to be published contains anything other
    than pipeline artifacts. Used by the data workflow before it commits."""
    allowed_suffixes = (".json", ".geojson", ".png", ".u16.gz")
    bad = []
    for p in root.rglob("*"):
        if ".git" in p.relative_to(root).parts or p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        if not rel.startswith(f"{allowed_top}/") or not rel.endswith(allowed_suffixes):
            bad.append(rel)
        elif p.stat().st_size > 20_000_000:
            bad.append(f"{rel} (too large)")
    return bad


def main_guard(root: str) -> int:
    bad = guard_publish_tree(Path(root))
    if bad:
        print(json.dumps({"refusing_to_publish": bad}, indent=2))
        return 1
    print("publish tree OK: only v1/ pipeline artifacts")
    return 0
