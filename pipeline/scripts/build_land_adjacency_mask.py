"""Build data/reference/olci_land_adjacency.json: the first two water rings beside land on the OLCI lattice.

    uv run python scripts/build_land_adjacency_mask.py [--days 45] [--cache /tmp/cw-adjacency]

Why (docs/coastwatch/v13/research/coastal-adjacency.md): in NOAA CoastWatch's Sentinel-3 OLCI
chlorophyll (noaacwS3{A,B}OLCIchlaSector{CI,DI}Daily) the single water pixel next to the
product's own land mask reads about ten times its neighbours on every clear day: land and water
in one 300 m pixel, and stray light from the land. The product carries no flag for it. The
pipeline therefore leaves that ring out (sources/satellite.py LAND_ADJACENCY_MASK).

The ring is defined by the product itself, not by a coastline from somewhere else:

- "land" is a cell with NO value in any of the scenes read (needs network: one request per
  latitude band and scene, only the strip of the lattice near a shore), AND within
  NEAR_SHORE_CELLS of the OpenStreetMap shoreline (the committed basemap water layer). The second
  condition keeps a patch of sea that happened to be under cloud in every scene from being called
  land: a never-valid cell far from any shore is not land.
- ring 1 is every cell that had a value in some scene and touches a land cell (8 neighbours);
  ring 2 is every cell with a value that touches ring 1.

Stored as run-length encoded rows on the OLCI 300 m California lattice (west to east, north to
south), with the scenes used, so the file says what it was made from. VIIRS 750 m cell centres
lie on the same lattice (every third cell).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coastwatch_pipeline.http import FetchError, fetch  # noqa: E402
from coastwatch_pipeline.sources.charm import parse_netcdf  # noqa: E402
from coastwatch_pipeline.sources import satellite  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "reference" / satellite.LAND_ADJACENCY_MASK
# the same cells for the web app (Explore says "left out: beside land" for them): lattice and runs only
WEB = REPO / "coastwatch-web" / "public" / "reference" / "land-adjacency-v1.json"
UA = {"User-Agent": "reference-mask-builder/1.0"}  # a one-off build, not the pipeline
BAND_ROWS = 200  # 0.5 degrees of latitude per request
PAD_CELLS = 24  # the strip read each side of the shoreline cells of a band
NEAR_SHORE_CELLS = 4  # a never-valid cell is land only this close to the OSM shoreline (1.2 km)
EIGHT = np.ones((3, 3), dtype=bool)


def _marine_builder():
    spec = importlib.util.spec_from_file_location("build_marine_water_mask", Path(__file__).with_name("build_marine_water_mask.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shore_strips(osm_water: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Per latitude band, the column range that holds every cell near an OSM shoreline:
    (row0, row1, col0, col1), half-open. Bands with no shoreline are skipped."""
    shore = osm_water & binary_dilation(~osm_water, structure=EIGHT)
    out = []
    h, w = osm_water.shape
    for r0 in range(0, h, BAND_ROWS):
        r1 = min(h, r0 + BAND_ROWS)
        cols = np.nonzero(shore[r0:r1].any(axis=0))[0]
        if cols.size:
            out.append((r0, r1, max(0, int(cols[0]) - PAD_CELLS), min(w, int(cols[-1]) + PAD_CELLS + 1)))
    return out


def scene_times(ds: str, days: int, end: datetime) -> list[str]:
    lo = (end - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
    hi = end.strftime("%Y-%m-%dT23:59:59Z")
    body = fetch(f"{satellite.OLCI.server}/griddap/{ds}.csv0?" + quote(f"time[({lo}):1:({hi})]", safe=":(),"), headers=UA).body
    return [ln.strip() for ln in body.decode().splitlines() if ln.strip()[:1].isdigit()]


def read_valid(ds: str, t: str, target: satellite.Target, strip: tuple[int, int, int, int], cache: Path) -> np.ndarray | None:
    """True where the scene has a positive value, on the strip; None when the scene cannot be read."""
    r0, r1, c0, c1 = strip
    lats, lons = target.lats(), target.lons()
    q = f"chlor_a[({t})][(0.0)][({lats[r0]:.5f}):({lats[r1 - 1]:.5f})][({lons[c0]:.5f}):({lons[c1 - 1]:.5f})]"
    url = f"{satellite.OLCI.server}/griddap/{ds}.nc?" + quote(q, safe=",():")
    f = cache / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".npy")
    if f.exists():
        return np.load(f)
    for attempt in range(3):
        try:
            arrays, _, _ = parse_netcdf(fetch(url, headers=UA, retries=0, timeout=120).body)
            break
        except FetchError as e:
            if "404" in str(e):
                return None
            if attempt == 2:
                print("  skipped", ds, t, strip, str(e)[:80])
                return None
            time.sleep(10)
        except Exception as e:  # a truncated or unreadable response
            print("  unreadable", ds, t, strip, str(e)[:80])
            return None
    lat = np.asarray(arrays["latitude"], dtype=np.float64)
    lon = np.asarray(arrays["longitude"], dtype=np.float64)
    a = np.asarray(arrays["chlor_a"], dtype=np.float64).reshape(lat.size, lon.size)
    rows = np.round((target.lat_first - lat) / target.step).astype(int) - r0
    cols = np.round((lon - target.lon_first) / target.step).astype(int) - c0
    ok_r = (rows >= 0) & (rows < r1 - r0)
    ok_c = (cols >= 0) & (cols < c1 - c0)
    out = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    out[np.ix_(rows[ok_r], cols[ok_c])] = np.isfinite(a[np.ix_(ok_r, ok_c)]) & (a[np.ix_(ok_r, ok_c)] > 0)
    np.save(f, out)
    return out


def ring_from(ever_valid: np.ndarray, read: np.ndarray, osm_water: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(land, ring 1, ring 2) from the cells that ever had a value, the cells that were read at
    all, and the OSM water raster. Ring 1 touches land; ring 2 touches ring 1 and is neither.
    Pure: this is what the tests exercise."""
    near_shore = binary_dilation(~osm_water, structure=EIGHT, iterations=NEAR_SHORE_CELLS)
    land = read & ~ever_valid & near_shore
    ring = ever_valid & binary_dilation(land, structure=EIGHT)
    ring2 = ever_valid & binary_dilation(ring, structure=EIGHT) & ~ring
    return land, ring, ring2


def web_asset(ref: dict) -> str:
    """The web's copy: the lattice and the runs of every cell the OLCI layers leave out (the rings the
    product drops, merged), nothing else (src/nx/explore/adjacency.ts AdjacencyMask)."""
    cells = np.zeros((ref["height"], ref["width"]), dtype=bool)
    for key in ("rows_rle", "rows_rle_ring2")[: satellite.OLCI.land_adjacent_rings]:
        for r, runs in enumerate(ref[key]):
            for a, n in runs:
                cells[r, a : a + n] = True
    keep = {"lat_first": ref["lat_first"], "lon_first": ref["lon_first"], "step": ref["lon_step"], "height": ref["height"], "width": ref["width"], "rows_rle": _marine_builder().rle(cells)}
    return json.dumps(keep, separators=(",", ":")) + "\n"


def write_web(out: Path = WEB) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(web_asset(json.loads(OUT.read_text())))
    print(out, out.stat().st_size, "bytes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-only", action="store_true", help="rewrite the web copy from the committed reference (no network)")
    ap.add_argument("--days", type=int, default=45)
    ap.add_argument("--end", default=None, help="last day read, YYYY-MM-DD (default: yesterday, UTC)")
    ap.add_argument("--cache", default="/tmp/cw-adjacency")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.web_only:
        return write_web()
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc) - timedelta(days=1)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    mb = _marine_builder()
    target = satellite.target_for(satellite.OLCI, satellite.CALIFORNIA)
    src_raw = mb.SRC.read_bytes()
    osm_water = mb.rasterise(json.loads(src_raw), target)
    strips = shore_strips(osm_water)
    lons = target.lons()

    ever = np.zeros((target.height, target.width), dtype=bool)
    read = np.zeros_like(ever)
    used: dict[str, set[str]] = {}
    lats = target.lats()
    jobs: list[tuple[str, str, tuple[int, int, int, int]]] = []
    for _platform, sectors in satellite.OLCI.platforms:
        for ds, lon_lo, lon_hi, lat_max in sectors:
            times = scene_times(ds, args.days, end)
            print(ds, len(times), "scenes")
            for r0, r1, c0, c1 in strips:
                # the part of the strip this sector serves (satellite.scene_url's rule), south of its northern limit
                cols = np.arange(c0, c1)
                keep = cols[(lons[cols] > lon_lo) & (lons[cols] < lon_hi)] if lon_lo > -180 else cols[lons[cols] < lon_hi]
                rows = np.arange(r0, r1)
                rows = rows[lats[rows] <= lat_max]
                if not keep.size or not rows.size:
                    continue
                s = (int(rows[0]), int(rows[-1]) + 1, int(keep[0]), int(keep[-1]) + 1)
                jobs.extend((ds, t, s) for t in times)
    print(len(jobs), "requests")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (ds, t, s), v in zip(jobs, pool.map(lambda j: read_valid(j[0], j[1], target, j[2], cache), jobs)):
            if v is None:
                continue
            used.setdefault(ds, set()).add(t)
            ever[s[0] : s[1], s[2] : s[3]] |= v
            read[s[0] : s[1], s[2] : s[3]] = True

    land, ring, ring2 = ring_from(ever, read, osm_water)
    scope = satellite.coastal_mask(target)
    OUT.write_text(
        json.dumps(
            {
                "description": (
                    "Land adjacency on the Sentinel-3 OLCI 300 m lattice. rows_rle (ring 1): cells that carried a chlorophyll value in at least "
                    "one of the scenes listed and touch (8 neighbours) a cell that carried none and lies within "
                    f"{NEAR_SHORE_CELLS} cells of the OpenStreetMap shoreline. rows_rle_ring2 (ring 2): cells with a value that touch ring 1. "
                    "The satellite chlorophyll layers leave out the rings their product is set to (sources/satellite.py "
                    "Product.land_adjacent_rings): there one pixel holds land and water together, or sits in the land's stray light. Built by pipeline/scripts/build_land_adjacency_mask.py; evidence in "
                    "docs/coastwatch/v13/research/coastal-adjacency.md."
                ),
                "source": f"NOAA CoastWatch ERDDAP {satellite.OLCI.server}: " + ", ".join(sorted(used)),
                "scenes": {ds: {"n": len(ts), "first": min(ts), "last": max(ts)} for ds, ts in sorted(used.items())},
                "built": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "shoreline_file": str(mb.SRC.relative_to(REPO)),
                "shoreline_sha256": hashlib.sha256(src_raw).hexdigest(),
                "near_shore_cells": NEAR_SHORE_CELLS,
                "lat_first": target.lat_first,
                "lat_step": -target.step,
                "lon_first": target.lon_first,
                "lon_step": target.step,
                "height": target.height,
                "width": target.width,
                "n_ring": int(ring.sum()),
                "n_ring_in_scope": int((ring & scope).sum()),
                "n_ring2": int(ring2.sum()),
                "n_ring2_in_scope": int((ring2 & scope).sum()),
                "rows_rle": mb.rle(ring),
                "rows_rle_ring2": mb.rle(ring2),
            },
            separators=(",", ":"),
        )
        + "\n"
    )
    write_web()
    print(OUT, f"{int(land.sum())} land cells near shore, {int(ring.sum())} ring-1 cells ({int((ring & scope).sum())} in the satellite scope), {int(ring2.sum())} ring-2 cells ({int((ring2 & scope).sum())} in scope), {sum(len(v) for v in used.values())} scenes")


if __name__ == "__main__":
    main()
