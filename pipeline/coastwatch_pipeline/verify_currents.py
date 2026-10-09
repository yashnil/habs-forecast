"""Verify published HF-radar currents against the upstream ERDDAP and against themselves.

1. source     - (live) for sample cells of this run's hourly layers, ERDDAP's water_u and
                water_v at that hour and cell centre equal the published u and v within
                quantization, and ERDDAP returns exactly that cell centre (alignment)
2. dropped    - (live) for sample cells published without a value, ERDDAP either has no
                value there or the value fails a documented check (sites, HDOP, speed):
                good observations are never discarded silently
3. mean       - each 24-hour-mean cell equals the mean of the published hourly values at
                that cell, with at least MEAN_MIN_HOURS hours
An ERDDAP that does not answer is reported as UNVERIFIABLE (and fails), never as a mismatch.
"""

from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .http import FetchError, fetch
from .models import LayerArtifact, Manifest
from .process import grid as gridcodec
from .sources import currents as cur


def _field(out_dir: Path, lyr: LayerArtifact) -> tuple[np.ndarray, np.ndarray]:
    assert lyr.vectors
    return tuple(  # type: ignore[return-value]
        gridcodec.decode((out_dir / g.url).read_bytes(), g.width, g.height, g.scale_factor, g.add_offset) for g in (lyr.vectors.u_grid, lyr.vectors.v_grid)
    )


def _erddap(t: str, lat: float, lon: float) -> dict | None:
    sel = f"[({t})][({lat:.6f})][({lon:.6f})]"
    url = f"{cur.SERVER}/griddap/{cur.DATASET}.csv0?" + quote(",".join(v + sel for v in cur.VARS), safe=",():")
    try:
        line = fetch(url, retries=3, backoff=3).body.decode().strip().splitlines()[-1].split(",")
    except FetchError as e:
        if "HTTP 404" in str(e):
            return None
        raise RuntimeError(f"UNVERIFIABLE, ERDDAP unreachable: {str(e)[:160]}") from e
    num = lambda x: float("nan") if x in ("NaN", "") else float(x)  # noqa: E731
    return {"lat": float(line[1]), "lon": float(line[2]), "u": num(line[3]), "v": num(line[4]), "hdop": num(line[5]), "sites": num(line[6])}


def _cells(mask: np.ndarray, n: int) -> list[tuple[int, int]]:
    rr, cc = np.nonzero(mask)
    if not rr.size:
        return []
    idx = np.linspace(0, rr.size - 1, num=min(n, rr.size)).astype(int)
    return [(int(rr[i]), int(cc[i])) for i in idx]


def verify_currents(out_dir: Path, live: bool = True, per_layer: int = 6, hourly_layers: int = 2) -> dict:
    m = Manifest.model_validate_json((out_dir / "manifest.json").read_text())
    all_cur = [lyr for lyr in m.layers if lyr.provenance.source_id == cur.SOURCE_ID and lyr.vectors]
    hourly = sorted((lyr for lyr in all_cur if lyr.layer_id.startswith("hfr2km_currents_2")), key=lambda x: x.time.valid_time or "")
    new = [lyr for lyr in hourly if lyr.provenance.pipeline_run_id == m.pipeline_run_id]
    rows: list[dict] = []
    picks = new[-1:] + new[:: max(1, len(new) // max(1, hourly_layers - 1))][: hourly_layers - 1] if new else []
    for lyr in {x.layer_id: x for x in picks}.values():
        u, v = _field(out_dir, lyr)
        g = lyr.vectors.u_grid  # type: ignore[union-attr]
        tol = g.max_quantization_error + 1e-6
        t = lyr.time.valid_time or ""
        for kind, cells in (("value", _cells(np.isfinite(u), per_layer)), ("no_value", _cells(~np.isfinite(u), max(2, per_layer // 3)))):
            for r, c in cells:
                lat, lon = g.lat_first + r * g.lat_step, g.lon_first + c * g.lon_step
                row: dict = {"layer_id": lyr.layer_id, "time": t, "kind": kind, "row": r, "col": c, "lat": round(lat, 6), "lon": round(lon, 6)}
                if kind == "value":
                    row.update(u=round(float(u[r, c]), 5), v=round(float(v[r, c]), 5))
                ok = True
                if live:
                    try:
                        up = _erddap(t, lat, lon)
                        time.sleep(0.2)
                    except RuntimeError as e:
                        up, ok = None, False
                        row["source_error"] = str(e)
                    if up is not None:
                        row["source_cell"] = [up["lat"], up["lon"]]
                        row["source_cell_matches"] = abs(up["lat"] - lat) < 1e-4 and abs(up["lon"] - lon) < 1e-4
                        has = np.isfinite(up["u"]) and np.isfinite(up["v"]) and abs(up["u"]) < 100
                        if kind == "value":
                            row.update(source_u=up["u"], source_v=up["v"])
                            row["source_matches"] = bool(has and abs(up["u"] - u[r, c]) <= tol and abs(up["v"] - v[r, c]) <= tol)
                            ok = ok and row["source_cell_matches"] and row["source_matches"]
                        else:
                            fails = (not has) or not (up["sites"] >= cur.MIN_SITES) or not (up["hdop"] <= cur.HDOP_MAX) or float(np.hypot(up["u"], up["v"])) > cur.SPEED_MAX
                            row["dropped_correctly"] = bool(fails)
                            ok = ok and row["source_cell_matches"] and row["dropped_correctly"]
                    elif "source_error" not in row and kind == "value":
                        ok = False
                        row["source_error"] = "ERDDAP has no value at a published cell"
                row["status"] = "pass" if ok else "FAIL"
                rows.append(row)
    # 24-hour mean equals the mean of the published hours
    mean = next((x for x in all_cur if x.layer_id == "hfr2km_currents_mean24h" and x.provenance.pipeline_run_id == m.pipeline_run_id), None)
    if mean:
        t0, t1 = mean.time.observed_times[0], mean.time.observed_times[-1]
        used = [x for x in hourly if t0 <= (x.time.valid_time or "") <= t1]
        U = np.stack([_field(out_dir, x)[0] for x in used])
        V = np.stack([_field(out_dir, x)[1] for x in used])
        mu, mv = _field(out_dir, mean)
        n = np.isfinite(U).sum(axis=0)
        tol = 2 * mean.vectors.u_grid.max_quantization_error + 1e-6  # type: ignore[union-attr]
        for r, c in _cells(np.isfinite(mu), per_layer * 2):
            eu, ev = np.nanmean(U[:, r, c]), np.nanmean(V[:, r, c])
            ok = n[r, c] >= cur.MEAN_MIN_HOURS and abs(eu - mu[r, c]) <= tol and abs(ev - mv[r, c]) <= tol
            rows.append({"layer_id": mean.layer_id, "kind": "mean", "row": r, "col": c, "hours": int(n[r, c]), "u": round(float(mu[r, c]), 5), "expected_u": round(float(eu), 5), "status": "pass" if ok else "FAIL"})
        for r, c in _cells(~np.isfinite(mu) & (n > 0), 4):
            ok = n[r, c] < cur.MEAN_MIN_HOURS
            rows.append({"layer_id": mean.layer_id, "kind": "mean_withheld", "row": r, "col": c, "hours": int(n[r, c]), "status": "pass" if ok else "FAIL"})
    failed = [r for r in rows if r["status"] == "FAIL"]
    unverifiable = [r for r in failed if str(r.get("source_error", "")).startswith("UNVERIFIABLE")]
    return {
        "manifest_generated_at": m.generated_at,
        "live_source_comparison": live,
        "summary": {
            "hourly_layers_new": len(new), "hourly_layers_checked": len({x.layer_id for x in picks}), "checks": len(rows),
            "failures": len(failed), "unverifiable_upstream_unreachable": len(unverifiable),
            "all_passed": not failed and (bool(rows) or not new),
        },
        "rows": rows,
    }
