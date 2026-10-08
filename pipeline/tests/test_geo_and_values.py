"""Projection, raster alignment and numerical round-trip checks."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.fixtures import fixture_context
from coastwatch_pipeline.process import grid as gridcodec
from coastwatch_pipeline.process.mercator import (
    MercatorImage,
    SourceGrid,
    lat_to_merc_y,
    merc_y_to_lat,
    plan_image,
    resample_to_mercator,
)
from coastwatch_pipeline.process.palette import PROBABILITY, apply_palette, palette_color
from coastwatch_pipeline.verify import verify_charm

KM_PER_DEG = 111.32


def test_mercator_round_trip():
    lats = np.linspace(31.3, 43.0, 50)
    assert np.allclose(merc_y_to_lat(lat_to_merc_y(lats)), lats, atol=1e-9)


def test_naive_corner_placement_would_misplace_by_kilometres():
    """Regression for the old export: an equirectangular image stretched between corners.
    Our Mercator resampling must not have this error."""
    south, north, lat = 31.3, 43.0, 37.0
    f = (lat - south) / (north - south)
    y = lat_to_merc_y(south) + f * (lat_to_merc_y(north) - lat_to_merc_y(south))
    naive_lat = float(merc_y_to_lat(y))
    assert abs(naive_lat - lat) * KM_PER_DEG > 10  # >10 km off with the naive approach

    src = SourceGrid(31.31, 0.03, -127.49, 0.03, 390, 350)
    img = plan_image(src)
    row, _ = img.pixel_of(lat, -122.0)
    y_top, y_bot = lat_to_merc_y(img.north), lat_to_merc_y(img.south)
    pix_lat = float(merc_y_to_lat(y_top - (row + 0.5) / img.height * (y_top - y_bot)))
    pixel_km = (y_top - y_bot) / img.height / 1000 * math.cos(math.radians(lat))
    assert abs(pix_lat - lat) * KM_PER_DEG <= pixel_km  # within one pixel (~0.8 km)


def test_every_pixel_samples_the_cell_containing_its_centre():
    src = SourceGrid(36.01, 0.03, -123.09, 0.03, 81, 51)
    vals = np.arange(src.height * src.width, dtype=float).reshape(src.height, src.width)
    img = plan_image(src, upsample=4)
    out = resample_to_mercator(vals, src, img)
    y_top, y_bot = lat_to_merc_y(img.north), lat_to_merc_y(img.south)
    rng = np.random.default_rng(1)
    for _ in range(500):
        r, c = rng.integers(0, img.height), rng.integers(0, img.width)
        lat = float(merc_y_to_lat(y_top - (r + 0.5) / img.height * (y_top - y_bot)))
        lon = img.west + (c + 0.5) / img.width * (img.east - img.west)
        sr = round((lat - src.lat_first) / src.lat_step)
        sc = round((lon - src.lon_first) / src.lon_step)
        assert out[r, c] == vals[sr, sc]


def test_image_corners_match_grid_edges():
    src = SourceGrid(36.01, 0.03, -123.09, 0.03, 81, 51)
    img = plan_image(src)
    assert math.isclose(img.west, -123.105) and math.isclose(img.east, -121.575)
    assert math.isclose(img.south, 35.995) and math.isclose(img.north, 38.425)
    assert img.corners[0] == (img.west, img.north) and img.corners[2] == (img.east, img.south)


def test_grid_round_trip_within_quantization_error():
    rng = np.random.default_rng(2)
    v = rng.uniform(0, 1, (40, 30))
    v[5:9, 2:7] = np.nan
    blob, scale, offset, qerr = gridcodec.encode(v, 0.0, 1.0)
    back = gridcodec.decode(blob, 30, 40, scale, offset)
    assert np.array_equal(np.isnan(back), np.isnan(v))
    ok = ~np.isnan(v)
    assert np.max(np.abs(back[ok] - v[ok])) <= qerr + 1e-12
    assert qerr < 1e-5


def test_palette_is_fixed_and_monotone_in_lightness():
    assert PROBABILITY.domain == (0.0, 1.0)
    lum = [sum(int(s.color[i : i + 2], 16) * w for i, w in ((1, 0.2126), (3, 0.7152), (5, 0.0722))) for s in PROBABILITY.stops]
    assert all(b > a for a, b in zip(lum, lum[1:]))
    rgba = apply_palette(np.array([np.nan, 0.0, 1.0]), PROBABILITY)
    assert rgba[0, 3] == 0 and rgba[1, 3] == 255
    assert palette_color(0.0, PROBABILITY) == (0x5D, 0x27, 0x48)


def test_published_images_and_grids_agree_with_source_points(out):
    run_pipeline(fixture_context(out))
    report = verify_charm(out, live=False)
    s = report["summary"]
    assert s["all_passed"], [r for r in report["rows"] if r["status"] == "FAIL"]
    assert s["comparisons"] >= 50
    # nearshore pier cell has no toxin value in the source, and none in our output
    wharf = [r for r in report["rows"] if r["point"].startswith("Monterey Wharf") and "domoic" in r["layer_id"]]
    assert wharf and all(r["grid_value"] is None for r in wharf)


def test_grid_values_equal_source_netcdf(out):
    """Decoded published grid == values read directly from the recorded ERDDAP file."""
    from coastwatch_pipeline.sources import charm

    m = run_pipeline(fixture_context(out))
    blob = (charm.__file__ and open(fixture_context(out).fetcher.root / "charm" / "lead1.nc", "rb").read())
    arrays, vattrs, _ = charm.parse_netcdf(blob)
    raw = arrays["particulate_domoic"][0].astype(float)
    raw[np.isclose(raw, -99999.0)] = np.nan
    lyr = next(lyr for lyr in m.layers if lyr.layer_id == "charm_particulate_domoic_lead1")
    g = lyr.grid
    dec = gridcodec.decode((out / g.url).read_bytes(), g.width, g.height, g.scale_factor, g.add_offset)
    assert np.array_equal(np.isnan(dec), np.isnan(raw))
    ok = ~np.isnan(raw)
    assert np.max(np.abs(dec[ok] - raw[ok])) <= g.max_quantization_error + 1e-7
    # the image is a palette rendering of exactly these values
    im = np.asarray(Image.open(out / lyr.image.url).convert("RGBA"))
    mi = MercatorImage(lyr.image.width, lyr.image.height, *[lyr.image.bounds_lnglat[i] for i in (0, 2, 1, 3)])
    lat, lon = g.lat_first + 40 * g.lat_step, g.lon_first + 25 * g.lon_step
    pr, pc = mi.pixel_of(lat, lon)
    if not np.isnan(raw[40, 25]):
        assert tuple(im[pr, pc, :3]) == palette_color(float(dec[40, 25]), PROBABILITY)
