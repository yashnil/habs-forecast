"""Verify published satellite chlorophyll artifacts.

For sample pixels with a value (spread across the domain, deterministic):
1. alignment  - the grid cell centre is on the source lattice and, live, ERDDAP returns
                exactly that cell centre for a point query at it
2. source     - (live) ERDDAP's value for the same scene and cell equals the published
                value within quantization error
3. tiles      - the tile pixel containing the cell centre at the native zoom has the
                palette colour of the published value
4. composite  - each latest-clear-view pixel equals the day layer its age points to
Cells without a value are checked to be transparent in the tiles.
"""

from __future__ import annotations

import math
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

from .http import FetchError, fetch
from .models import LayerArtifact, Manifest
from .process.palette import palette_indices
from .process.tiles import TILE, lat_to_y, lon_to_x
from .sources import satellite


def _published(out_dir: Path, lyr: LayerArtifact) -> np.ndarray:
    assert lyr.grid
    return satellite.load_published(out_dir, lyr.grid)


def _age(out_dir: Path, lyr: LayerArtifact) -> np.ndarray:
    assert lyr.composite
    return satellite.load_published(out_dir, lyr.composite.age_grid)


def _sample(values: np.ndarray, n: int) -> list[tuple[int, int]]:
    rr, cc = np.nonzero(np.isfinite(values))
    if not rr.size:
        return []
    idx = np.linspace(0, rr.size - 1, num=min(n, rr.size)).astype(int)
    return [(int(rr[i]), int(cc[i])) for i in idx]


def _tile_index(out_dir: Path, lyr: LayerArtifact, lat: float, lon: float) -> int | None:
    t = lyr.tiles
    assert t
    z = t.max_native_zoom
    x, y = float(lon_to_x(lon, z)), float(lat_to_y(lat, z))
    p = out_dir / t.url_template.format(z=z, x=int(x // TILE), y=int(y // TILE))
    if not p.exists():
        return None
    with Image.open(p) as im:
        return int(np.asarray(im)[int(y % TILE), int(x % TILE)])


class Unreachable(Exception):
    """ERDDAP did not answer (e.g. HTTP 403 to cloud runners): the value could not be
    checked. Reported separately from a mismatch; it still fails verification."""


def _erddap(server: str, ds: str, var: str, t: str, lat: float, lon: float) -> tuple[float, float, float] | None:
    q = f"{var}[({t})][(0.0)][({lat:.5f})][({lon:.5f})]"
    url = f"{server}/griddap/{ds}.csv0?" + quote(q, safe=",():")
    try:
        line = fetch(url, retries=3, backoff=3).body.decode().strip().splitlines()[-1].split(",")
    except FetchError as e:
        if "HTTP 404" in str(e):  # no data at that time and place: a real answer
            return None
        raise Unreachable(str(e)[:200]) from e
    v = float("nan") if line[-1] in ("NaN", "") else float(line[-1])
    return v, float(line[2]), float(line[3])


def verify_satellite(out_dir: Path, live: bool = True, per_layer: int = 8) -> dict:
    m = Manifest.model_validate_json((out_dir / "manifest.json").read_text())
    all_layers = [lyr for lyr in m.layers if lyr.provenance.source_id == satellite.SOURCE_ID and lyr.grid]
    # Only layers made by this run: reused and carried-over layers were verified when they
    # were first published (and their upstream may be the part that is down right now).
    layers = [lyr for lyr in all_layers if lyr.provenance.pipeline_run_id == m.pipeline_run_id]
    rows: list[dict] = []
    days = {lyr.time.observed_date: lyr for lyr in all_layers if lyr.composite is None and lyr.layer_id.startswith("olci")}
    for lyr in layers:
        vals = _published(out_dir, lyr)
        g = lyr.grid
        assert g
        prod = satellite.OLCI if lyr.layer_id.startswith("olci") else satellite.VIIRS
        _, colours = palette_indices(np.array([1.0]), lyr.palette)
        age = _age(out_dir, lyr) if lyr.composite else None
        for r, c in _sample(vals, per_layer):
            lat, lon = g.lat_first + r * g.lat_step, g.lon_first + c * g.lon_step
            v = float(vals[r, c])
            row: dict = {"layer_id": lyr.layer_id, "row": r, "col": c, "lat": round(lat, 6), "lon": round(lon, 6), "value": round(v, 5)}
            # tiles: palette index at the cell centre equals the published value's index
            want = int(palette_indices(np.array([v]), lyr.palette)[0][0])
            got = _tile_index(out_dir, lyr, lat, lon)
            row["tile_matches_value"] = got == want
            ok = got == want
            # composite: value equals the day layer the age points to
            if age is not None and lyr.composite:
                a = int(age[r, c])
                d = (date.fromisoformat(lyr.composite.reference_date) - timedelta(days=a)).isoformat()
                row["observed_date"] = d
                if lyr.layer_id.startswith("olci"):
                    day = days.get(d)
                    dv = float(_published(out_dir, day)[r, c]) if day and day.grid else float("nan")
                    row["composite_matches_day"] = bool(np.isclose(dv, v, rtol=1e-9))
                    ok = ok and row["composite_matches_day"]
                src_layer = days.get(d) if lyr.layer_id.startswith("olci") else None
                times = [t for t in lyr.time.observed_times if t.startswith(d)] if src_layer is None else src_layer.time.observed_times
            else:
                times = lyr.time.observed_times
            if live:
                tol = g.max_quantization_error + 1e-6
                match = None
                unreachable = None
                for plat, sectors in prod.platforms:
                    for ds, lo, hi, _ in sectors:
                        if not ((lo == -180 or lon > lo) and lon < hi):
                            continue
                        for t in times:
                            try:
                                res = _erddap(prod.server, ds, prod.variable, t, lat, lon)
                            except Unreachable as e:
                                unreachable = str(e)
                                res = None
                            time.sleep(0.2)
                            if res and math.isfinite(res[0]) and res[0] > 0:
                                match = (plat, ds, t, *res)
                                break
                        if match:
                            break
                    if match:
                        break
                if match:
                    _, ds, t, sv, slat, slon = match
                    row.update(source_dataset=ds, source_time=t, source_value=round(sv, 5), source_cell=[slat, slon])
                    row["source_cell_matches"] = abs(slat - lat) < 1e-5 and abs(slon - lon) < 1e-5
                    row["source_matches_grid"] = abs(math.log10(sv) - math.log10(v)) <= tol
                    ok = ok and row["source_cell_matches"] and row["source_matches_grid"]
                else:
                    row["source_matches_grid"] = False
                    row["source_error"] = f"UNVERIFIABLE, ERDDAP unreachable: {unreachable}" if unreachable else "no valid value at this cell in the listed scenes"
                    ok = False
            row["status"] = "pass" if ok else "FAIL"
            rows.append(row)
        # empty cells must be transparent in the tiles
        rr, cc = np.nonzero(~np.isfinite(vals))
        for i in np.linspace(0, rr.size - 1, num=min(4, rr.size)).astype(int) if rr.size else []:
            lat, lon = g.lat_first + rr[i] * g.lat_step, g.lon_first + cc[i] * g.lon_step
            got = _tile_index(out_dir, lyr, lat, lon)
            rows.append({"layer_id": lyr.layer_id, "row": int(rr[i]), "col": int(cc[i]), "value": None, "tile_transparent": got in (None, 0), "status": "pass" if got in (None, 0) else "FAIL"})
    failed = [r for r in rows if r["status"] == "FAIL"]
    unverifiable = [r for r in failed if str(r.get("source_error", "")).startswith("UNVERIFIABLE")]
    return {
        "manifest_generated_at": m.generated_at,
        "live_source_comparison": live,
        "summary": {
            "layers": len(layers), "layers_from_earlier_runs_not_rechecked": len(all_layers) - len(layers),
            "checks": len(rows), "failures": len(failed), "unverifiable_upstream_unreachable": len(unverifiable),
            "all_passed": not failed and (bool(rows) or not layers),
        },
        "rows": rows,
    }

