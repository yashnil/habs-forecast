"""The satellite chlorophyll scope: the coastal ocean (C-HARM ocean dilated COASTAL_DILATE_CELLS,
unchanged offshore) plus the marine bays and estuaries connected to it (OpenStreetMap ocean water,
data/reference/marine_water_mask.json). Until 2026-10 the scope was the dilated C-HARM ocean alone
and left out 95% of San Francisco Bay."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from coastwatch_pipeline.sources import satellite as sat

REPO = Path(__file__).resolve().parents[2]
OLCI_T = sat.target_for(sat.OLCI, sat.CALIFORNIA)
# the VIIRS 750 m lattice (erdVHNchla1day / NOAA-21 / NOAA-20), as the run resolves it
VIIRS_T = sat.target_for(sat.VIIRS_JPSS, sat.CALIFORNIA)

BAYS = {
    "SF Bay, central (east of Alcatraz)": (37.83, -122.40),
    "SF Bay, south (below San Mateo Bridge)": (37.55, -122.17),
    "San Pablo Bay": (38.05, -122.37),
    "Carquinez Strait": (38.06, -122.20),
    "Suisun Bay": (38.07, -122.03),
    "Tomales Bay": (38.15, -122.90),
    "Bodega Harbor": (38.315, -123.05),
    "Humboldt Bay (Arcata Bay)": (40.83, -124.12),
    "Morro Bay estuary": (35.34, -120.845),
    "Elkhorn Slough": (36.83, -121.76),
    "Mission Bay": (32.775, -117.225),
    "San Diego Bay, south": (32.63, -117.12),
}
LAKES = {
    "Clear Lake": (39.00, -122.77),
    "Lake Berryessa": (38.60, -122.23),
    "San Pablo Reservoir": (37.92, -122.255),
    "Crystal Springs Reservoir": (37.53, -122.36),
    "Calaveras Reservoir": (37.49, -121.82),
    "Lexington Reservoir": (37.20, -121.99),
    "Lake Tahoe": (39.10, -120.03),
    "Lake Elsinore": (33.66, -117.35),
}


def at(t: sat.Target, m: np.ndarray, lat: float, lon: float) -> bool:
    return bool(m[int(round((t.lat_first - lat) / t.step)), int(round((lon - t.lon_first) / t.step))])


@pytest.fixture(scope="module")
def scopes():
    return {t: (sat.charm_scope(t), sat.coastal_mask(t), sat.marine_water(t)) for t in (OLCI_T, VIIRS_T)}


@pytest.mark.parametrize("name", BAYS)
def test_connected_bays_and_estuaries_are_in_scope(scopes, name):
    for t, (_, scope, _) in scopes.items():
        assert at(t, scope, *BAYS[name]), name


def test_sf_bay_was_cropped_by_the_old_scope_and_is_whole_now(scopes):
    old, new, marine = scopes[OLCI_T]
    lats, lons = OLCI_T.lats(), OLCI_T.lons()
    box = ((lats >= 37.42) & (lats <= 38.20))[:, None] & ((lons >= -122.48) & (lons <= -121.75))[None, :]
    water = box & marine
    assert water.sum() > 12000  # ~1,000 km2 of 300 m cells inside the Golden Gate
    assert old[water].mean() < 0.10  # what the research measured: 95% left out
    assert new[water].all()


@pytest.mark.parametrize("name", LAKES)
def test_inland_lakes_and_reservoirs_stay_out(scopes, name):
    for t, (_, scope, _) in scopes.items():
        assert not at(t, scope, *LAKES[name]), name


def test_offshore_and_open_coast_scope_is_unchanged(scopes):
    for t, (old, new, marine) in scopes.items():
        assert not (old & ~new).any()  # nothing is removed
        assert not (new & ~old & ~marine).any()  # only marine water is added
        lons = t.lons()
        assert (new[:, lons < -123.2] == old[:, lons < -123.2]).all()  # offshore of the Golden Gate: identical
        assert (new & ~old).sum() < 0.01 * old.sum()


def test_viirs_cells_sit_on_the_olci_lattice_so_both_scopes_agree(scopes):
    o, v = scopes[OLCI_T][1], scopes[VIIRS_T][1]
    r0 = round((OLCI_T.lat_first - VIIRS_T.lat_first) / OLCI_T.step)
    c0 = round((VIIRS_T.lon_first - OLCI_T.lon_first) / OLCI_T.step)
    sub = o[r0 :: 3, c0 :: 3][: VIIRS_T.height, : VIIRS_T.width]
    assert sub.shape == v.shape and (sub == v).all()


def test_marine_mask_is_reproducible_from_committed_inputs(tmp_path):
    ref = REPO / "data" / "reference" / sat.MARINE_WATER_MASK
    meta = json.loads(ref.read_text())
    src = REPO / meta["source_file"]
    assert hashlib.sha256(src.read_bytes()).hexdigest() == meta["source_sha256"], "water-fallback.json changed: rerun scripts/build_marine_water_mask.py"
    charm = REPO / "data" / "reference" / "charm_ocean_mask.json"
    assert hashlib.sha256(charm.read_bytes()).hexdigest() == meta["charm_ocean_mask_sha256"]
    spec = importlib.util.spec_from_file_location("build_marine_water_mask", REPO / "pipeline" / "scripts" / "build_marine_water_mask.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "mask.json"
    mod.main(out)
    assert json.loads(out.read_text()) == meta


def test_charm_layers_do_not_use_the_satellite_scope():
    src = (REPO / "pipeline" / "coastwatch_pipeline" / "sources" / "charm.py").read_text()
    assert "coastal_mask" not in src and "marine_water" not in src
