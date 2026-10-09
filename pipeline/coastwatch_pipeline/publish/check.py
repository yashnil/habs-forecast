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
from ..models import FisheriesDataset, Manifest, ObservationDataset, OfficialDataset, PortIntelCollection, PortsCollection

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
    tiles_validated = 0

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
        for kind, t in lyr.tile_layers():
            if not t or not t.relative:
                continue
            if not t.sample_tiles:
                record(t.url_template, kind, Fetched(200, b""), False, "relative tile layer lists no sample tiles")
            for key in t.sample_tiles:
                z, x, y = key.split("/")
                rel = t.url_template.replace("{z}", z).replace("{x}", x).replace("{y}", y)
                f = _get(base, rel)
                ok, detail = f.status == 200, f"HTTP {f.status}"
                if ok:
                    try:
                        with Image.open(io.BytesIO(f.body)) as im:
                            ok = im.size == (t.tile_size, t.tile_size) and im.mode == "P"
                            detail = f"tile {im.size[0]}x{im.size[1]} mode {im.mode}"
                    except Exception as e:
                        ok, detail = False, f"not a PNG: {e}"
                record(rel, kind, f, ok, detail)
            if not remote:
                # Before publishing, every tile is checked, not just the samples: the tile
                # directory holds exactly n_tiles files and each one is a 256 px palette PNG.
                # (At the public URL the samples confirm serving; the content is the same commit.)
                root = Path(base) / t.url_template.split("{z}", 1)[0]
                files = sorted(root.rglob("*.png")) if root.is_dir() else []
                bad = []
                for fp in files:
                    try:
                        with Image.open(fp) as im:
                            if im.size != (t.tile_size, t.tile_size) or im.mode != "P":
                                bad.append(f"{fp.relative_to(root).as_posix()} {im.size} {im.mode}")
                    except Exception as e:
                        bad.append(f"{fp.relative_to(root).as_posix()}: {e}")
                why = ([f"{len(files)} tile files != n_tiles {t.n_tiles}"] if t.n_tiles is not None and len(files) != t.n_tiles else []) + (
                    [f"{len(bad)} bad tiles, e.g. {bad[0]}"] if bad else [])
                ok = not why
                detail = f"all {len(files)} tiles decode as {t.tile_size} px palette PNGs" if ok else "; ".join(why)
                tiles_validated += len(files) - len(bad)
                record(t.url_template, f"{kind}_all", Fetched(200, b""), ok, detail)
        grids = [("grid", lyr.grid)] + ([("age_grid", lyr.composite.age_grid)] if lyr.composite else [])
        for kind, g in grids:
            if not g or not g.chunks:
                continue
            ch = g.chunks
            for key in ch.present:
                r, c = (int(v) for v in key.split("_"))
                h = min(ch.rows, g.height - r * ch.rows)
                w = min(ch.cols, g.width - c * ch.cols)
                rel = ch.url_template.format(row=r, col=c)
                f = _get(base, rel)
                ok, detail = f.status == 200 and h > 0 and w > 0, f"HTTP {f.status}"
                if ok:
                    try:
                        n = len(gzip.decompress(f.body))
                        ok = n == h * w * 2
                        detail = f"chunk {key}: {n} bytes decoded" + ("" if ok else f" != {h * w * 2}")
                    except Exception as e:
                        ok, detail = False, f"not gzip: {e}"
                record(rel, kind, f, ok, detail)
        if lyr.vectors:
            for kind, g in (("u_grid", lyr.vectors.u_grid), ("v_grid", lyr.vectors.v_grid)):
                f = _get(base, g.url)
                ok, detail = f.status == 200, f"HTTP {f.status}"
                if ok:
                    try:
                        n = len(gzip.decompress(f.body))
                        ok = n == g.width * g.height * 2
                        detail = f"{kind}: {n} bytes decoded" + ("" if ok else f" != {g.width * g.height * 2}")
                    except Exception as e:
                        ok, detail = False, f"not gzip: {e}"
                record(g.url, kind, f, ok, detail)
            if lyr.vectors.arrows_url:
                f = _get(base, lyr.vectors.arrows_url)
                ok = f.status == 200
                try:
                    ok = ok and json.loads(f.body).get("type") == "FeatureCollection"
                except Exception:
                    ok = False
                record(lyr.vectors.arrows_url, "arrows", f, ok, "GeoJSON FeatureCollection" if ok else "missing or not GeoJSON")
        if lyr.coverage:
            cov_ok = all(0 <= rc.observed_fraction <= 1 for rc in lyr.coverage.regions)
            if not cov_ok:
                problems.append(f"{lyr.layer_id}: coverage fraction out of range")
        if lyr.grid and not lyr.grid.chunks:
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

    for rel, model, kind in (
        (manifest.official_url, OfficialDataset, "official"),
        (manifest.port_intel_url, PortIntelCollection, "port_intel"),
        (manifest.observations_url, ObservationDataset, "observations"),
        (manifest.fisheries_url, FisheriesDataset, "fisheries"),
    ):
        if not rel:
            continue
        f = _get(base, rel)
        ok, detail = f.status == 200, f"HTTP {f.status}"
        if ok:
            try:
                model.model_validate_json(f.body)
                detail = "schema valid"
            except Exception as e:
                ok, detail = False, f"invalid {kind}: {str(e)[:200]}"
        record(rel, kind, f, ok, detail)

    return {
        "base": base,
        "manifest_generated_at": manifest.generated_at,
        "pipeline_run_id": manifest.pipeline_run_id,
        "ok": not problems,
        "files_checked": len(checked) + 1,
        # every tile file, when checking a local tree; 0 at a URL (samples only)
        "tiles_validated": tiles_validated,
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
