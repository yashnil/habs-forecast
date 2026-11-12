"""Land adjacency (docs/coastwatch/v13/research/coastal-adjacency.md): the two OLCI 300 m water
pixels beside the product's own land mask are left out; the third pixel, the offshore water, VIIRS
and every other value are untouched."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from coastwatch_pipeline.fixtures import FixtureFetcher, fixture_context
from coastwatch_pipeline.sources import satellite as sat
from coastwatch_pipeline.sources.charm import parse_netcdf

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "data" / "reference" / sat.LAND_ADJACENCY_MASK
WEB = REPO / "coastwatch-web" / "public" / "reference" / "land-adjacency-v1.json"
FIX = Path(__file__).parent / "fixtures" / "satellite"
EIGHT = np.ones((3, 3), dtype=bool)


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_land_adjacency_mask", REPO / "pipeline" / "scripts" / "build_land_adjacency_mask.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def coast(h=40, w=60, shore=30):
    """Sea west of column `shore`, land east of it; every sea cell had a value some day."""
    osm_water = np.zeros((h, w), dtype=bool)
    osm_water[:, :shore] = True
    ever = osm_water.copy()
    return osm_water, ever, np.ones((h, w), dtype=bool)


def test_the_ring_is_the_first_water_cell_and_only_it(builder):
    osm, ever, read = coast()
    land, ring, ring2 = builder.ring_from(ever, read, osm)
    assert ring[:, 29].all(), "the cell that touches land is ring 1"
    assert not ring[:, :29].any()
    assert ring2[:, 28].all(), "the cell that touches ring 1 is ring 2"
    assert not ring2[:, :28].any() and not ring2[:, 29:].any(), "the third cell and everything seaward are in neither"
    assert not ring[:, 30:].any() and land[:, 30:34].all()


def test_a_never_valid_patch_offshore_is_not_land(builder):
    """Sea under cloud in every scene read must not grow a ring around itself."""
    osm, ever, read = coast()
    ever[10:20, 5:12] = False  # 25 cells from the shore: never valid, and not land
    land, ring, ring2 = builder.ring_from(ever, read, osm)
    assert not land[10:20, 5:12].any()
    assert not ring[:, :29].any() and not ring2[:, :28].any()


def test_the_products_own_mask_decides_not_the_coastline(builder):
    """Where the product masks one more water cell than the coastline has, the ring is the first cell
    WITH a value: the cell beside the product's mask."""
    osm, ever, read = coast()
    ever[:, 29] = False  # the product never gives the shoreline cell a value
    _, ring, ring2 = builder.ring_from(ever, read, osm)
    assert ring[:, 28].all() and not ring[:, :28].any() and not ring[:, 29].any()
    assert ring2[:, 27].all() and not ring2[:, :27].any()


def test_bays_a_wide_bay_keeps_its_interior_a_narrow_slough_is_all_ring(builder):
    h, w = 60, 80
    osm = np.zeros((h, w), dtype=bool)
    osm[:, :20] = True  # the open sea
    osm[10:30, 20:60] = True  # a bay 20 cells wide (6 km)
    osm[45:47, 20:70] = True  # a slough 2 cells wide (600 m)
    ever, read = osm.copy(), np.ones((h, w), dtype=bool)
    _, ring, ring2 = builder.ring_from(ever, read, osm)
    out = ring | ring2
    assert not out[12:28, 22:58].any(), "the bay's interior is kept"
    assert ring[10, 25:55].all() and ring[29, 25:55].all(), "its shore cells are ring 1"
    assert ring2[11, 25:55].all() and ring2[28, 25:55].all(), "and the next ones ring 2"
    assert ring[45:47, 21:70].all(), "every cell of a two-cell slough touches land: none is kept"


def test_cells_that_were_not_read_are_never_land(builder):
    osm, ever, read = coast()
    read[:, 30:] = False  # the strip east of the shore was not requested
    land, ring, ring2 = builder.ring_from(ever, read, osm)
    assert not land.any() and not ring.any() and not ring2.any()


def test_drop_sets_only_ring_cells_to_no_value():
    vals = np.arange(1.0, 26.0).reshape(5, 5)
    vals[0, 0] = np.nan
    src = np.ones((5, 5), dtype=np.uint8)
    adj = np.zeros((5, 5), dtype=bool)
    adj[:, 4] = True
    adj[0, 0] = True  # a cell beside land that had no value anyway is not counted
    before = vals.copy()
    n = sat.drop_land_adjacent(vals, src, adj)
    assert n == 5
    assert np.isnan(vals[:, 4]).all() and (src[:, 4] == 0).all()
    keep = ~adj
    assert np.array_equal(vals[keep], before[keep], equal_nan=True), "no other value is altered"
    assert (src[keep] == 1).all()


# ---------------------------------------------------------------- the committed reference
def test_reference_is_on_the_olci_lattice_and_says_what_it_was_made_from():
    m = json.loads(REF.read_text())
    t = sat.target_for(sat.OLCI, sat.CALIFORNIA)
    assert (m["height"], m["width"]) == (t.height, t.width)
    assert m["lat_first"] == t.lat_first and m["lon_first"] == t.lon_first and m["lon_step"] == t.step
    assert len(m["rows_rle"]) == t.height
    assert sum(v["n"] for v in m["scenes"].values()) >= 40, "built from too few scenes to tell land from cloud"
    assert all(k.startswith("noaacwS3") for k in m["scenes"])
    src = REPO / m["shoreline_file"]
    assert hashlib.sha256(src.read_bytes()).hexdigest() == m["shoreline_sha256"], "the shoreline changed: rerun scripts/build_land_adjacency_mask.py"
    ring = sat.land_adjacent(t)
    assert int(ring.sum()) == m["n_ring"] > 10_000
    both = sat.land_adjacent(t, 2)
    assert int(both.sum()) == m["n_ring"] + m["n_ring2"] and (both & ~ring).sum() == m["n_ring2"] > 10_000
    # ring 2 is exactly the cells that touch ring 1 among the cells the product gives a value
    assert not ((both & ~ring) & ~binary_dilation(ring, structure=EIGHT)).any()


def test_reference_has_no_effect_offshore_and_is_one_cell_wide(builder):
    t = sat.target_for(sat.OLCI, sat.CALIFORNIA)
    ring = sat.land_adjacent(t, 2)
    mb = builder._marine_builder()
    osm_water = mb.rasterise(json.loads(mb.SRC.read_bytes()), t)
    # every left-out cell is within 6 cells (1.8 km) of the OSM shoreline: nothing offshore is touched
    near = binary_dilation(~osm_water, structure=EIGHT, iterations=builder.NEAR_SHORE_CELLS + 2)
    assert not (ring & ~near).any()
    # C-HARM ocean cells (3 km) far from shore lose nothing
    far = ~binary_dilation(~osm_water, structure=EIGHT, iterations=12)
    assert not (ring & far).any()
    # two cells wide along an open coast (Monterey Bay's sandy arc, 36.70-36.90 N): in each row what is left
    # out is a short run, not a band
    r0, r1 = (int(round((t.lat_first - la) / t.step)) for la in (36.90, 36.70))
    c0, c1 = (int(round((lo - t.lon_first) / t.step)) for lo in (-121.95, -121.75))
    widths = ring[r0:r1, c0:c1].sum(axis=1)
    # (a few rows are wider where Elkhorn Slough, Moss Landing harbour and the river mouths meet the bay)
    assert np.median(widths) <= 4 and np.percentile(widths, 90) <= 8 and widths.max() <= 20, (int(widths.max()), float(np.median(widths)))
    # the satellite scope keeps almost all of its cells (the ring is a line, not an area)
    scope = sat.coastal_mask(t)
    assert (ring & scope).sum() / scope.sum() < 0.02


def test_san_francisco_bay_keeps_its_open_water():
    """Bays: the ring follows the bay's shore; the open water of the bay (37.70 N 122.30 W and
    37.60 N 122.20 W, South Bay) stays in scope and out of the ring."""
    t = sat.target_for(sat.OLCI, sat.CALIFORNIA)
    ring, scope = sat.land_adjacent(t, 2), sat.coastal_mask(t)
    for lat, lon in ((37.70, -122.30), (37.60, -122.20)):
        r, c = int(round((t.lat_first - lat) / t.step)), int(round((lon - t.lon_first) / t.step))
        box = (slice(r - 3, r + 4), slice(c - 3, c + 4))
        assert scope[box].all() and not ring[box].any(), (lat, lon)


def test_web_copy_holds_the_same_cells(builder):
    assert WEB.read_text() == builder.web_asset(json.loads(REF.read_text())), "rerun scripts/build_land_adjacency_mask.py --web-only"


# ---------------------------------------------------------------- through the pipeline (recorded scenes)
def run_region(tmp_path: Path, region: str):
    out = tmp_path / region / "v1"
    ctx = fixture_context(out, fetcher=FixtureFetcher(satellite_region=region))
    return ctx, sat.run(ctx, [])


def test_olci_layers_leave_the_ring_out_and_say_so(tmp_path):
    ctx, res = run_region(tmp_path, "monterey")
    assert not res.errors, res.errors
    day = next(lyr for lyr in res.layers if lyr.layer_id == "olci300_chl_2026-10-05")  # the clearest shore of the recorded week
    g = day.grid
    t = sat.Target(g.lat_first, g.lon_first, 0.0025, g.height, g.width)
    pub = sat.load_published(ctx.out_dir, g)
    ring = sat.land_adjacent(t, 2)
    assert ring.sum() > 600
    assert not (np.isfinite(pub) & ring).any(), "a cell within two pixels of land carries a value"
    check = next(c for c in day.qc.checks if c.name == "land_adjacency")
    assert check.passed and int(check.detail.split()[0]) > 300 and "within 2 pixel(s) of land" in check.detail
    assert sum(int(c.detail.split()[0]) for lyr in res.layers if lyr.layer_id.startswith("olci300_chl_2") for c in lyr.qc.checks if c.name == "land_adjacency") > 600
    assert day.provenance.upstream_metadata["processing"] == sat.PROCESSING_VERSION == "satellite-processing-5"
    assert any("beside land" in c for c in day.caveats)
    # what the ring took: the upstream scene's values there are far above the cells kept beside them
    arrays, _, _ = parse_netcdf((FIX / "monterey" / "olci300_s3a_CI_2026-10-05.nc").read_bytes())
    lat, lon = np.asarray(arrays["latitude"], float), np.asarray(arrays["longitude"], float)
    a = np.asarray(arrays["chlor_a"], float).reshape(lat.size, lon.size)
    a[~(a > 0)] = np.nan
    rows = np.round((g.lat_first - lat) / 0.0025).astype(int)
    cols = np.round((lon - g.lon_first) / 0.0025).astype(int)
    ok_r, ok_c = (rows >= 0) & (rows < g.height), (cols >= 0) & (cols < g.width)
    up = np.full((g.height, g.width), np.nan)
    up[np.ix_(rows[ok_r], cols[ok_c])] = a[np.ix_(ok_r, ok_c)]
    taken = up[ring & np.isfinite(up) & sat.coastal_mask(t)]
    third = binary_dilation(ring, structure=EIGHT) & ~ring & np.isfinite(pub)
    assert taken.size > 300 and third.sum() > 100
    assert np.median(taken) > 3 * np.median(pub[third]), (float(np.median(taken)), float(np.median(pub[third])))
    # the third pixel is published exactly as before: the upstream value, within quantization
    rr, cc = np.nonzero(third & np.isfinite(up))
    assert rr.size > 100
    assert np.abs(np.log10(pub[rr, cc]) - np.log10(up[rr, cc])).max() <= g.max_quantization_error + 1e-9
    # offshore: every upstream value in scope more than 2 cells from the ring is published
    off = np.isfinite(up) & sat.coastal_mask(t) & ~binary_dilation(ring, structure=EIGHT, iterations=2)
    assert np.isfinite(pub[off]).all()


def test_the_latest_view_has_no_ring_cell_either(tmp_path):
    ctx, res = run_region(tmp_path, "monterey")
    latest = next(lyr for lyr in res.layers if lyr.layer_id == "olci300_chl_latest")
    g = latest.grid
    t = sat.Target(g.lat_first, g.lon_first, 0.0025, g.height, g.width)
    assert not (np.isfinite(sat.load_published(ctx.out_dir, g)) & sat.land_adjacent(t, 2)).any()
    assert any(c.name == "land_adjacency" for c in latest.qc.checks)


def test_viirs_is_not_touched(tmp_path):
    """The 750 m product shows no population of extreme values in its first pixel (the research note):
    the rule is not applied to it, and its layers do not claim it."""
    assert sat.OLCI.land_adjacent_rings == 2 and sat.VIIRS_JPSS.land_adjacent_rings == 0 and sat.VIIRS_SNPP.land_adjacent_rings == 0
    ctx, res = run_region(tmp_path, "monterey")
    v = [lyr for lyr in res.layers if lyr.layer_id.startswith("viirs750")]
    assert v
    for lyr in v:
        assert not any(c.name == "land_adjacency" for c in lyr.qc.checks)
        assert not any("beside land" in c for c in lyr.caveats)
