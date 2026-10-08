"""Port intelligence: neighbourhood statistics, history, chlorophyll, official relations."""

from __future__ import annotations

import math

import numpy as np

from coastwatch_pipeline.fixtures import FIXTURES, fixture_context
from coastwatch_pipeline.models import PortIntelCollection
from coastwatch_pipeline.pipeline import run_pipeline
from coastwatch_pipeline.process import grid as codec
from coastwatch_pipeline.sources import charm, port_intel


def _run(out):
    m = run_pipeline(fixture_context(out))
    return m, PortIntelCollection.model_validate_json((out / m.port_intel_url).read_text())


def port(coll, name):
    return next(p for p in coll.ports if p.display_name == name)


def test_monterey_official_relations_and_reasons(out):
    _, coll = _run(out)
    rel = {(r.record_id, r.relation) for r in port(coll, "Monterey").official_relations}
    assert rel == {
        ("cdph-2026-annual-mussel-quarantine", "statewide"),
        ("cdph-2026-sn26-018-monterey-bivalves", "same_county"),
        ("cdph-2026-sn26-019-anchovy-central-coast", "port_latitude_within_stated_range"),
        ("cdfw-2026-anchovy-take-restriction-monterey-bay", "port_latitude_within_stated_range"),
    }
    # Santa Cruz is inside the anchovy latitudes but not in Monterey County
    sc = {r.record_id for r in port(coll, "Santa Cruz").official_relations}
    assert "cdph-2026-sn26-019-anchovy-central-coast" in sc and "cdph-2026-sn26-018-monterey-bivalves" not in sc
    # Ventura is near the Northern Channel Islands special advisory
    assert "cdph-nci-bivalve-special-advisory" in {r.record_id for r in port(coll, "Ventura").official_relations}


def test_lead_statistics_match_an_independent_recomputation(out):
    m, coll = _run(out)
    p = port(coll, "Moss Landing")
    lyr = next(lyr for lyr in m.layers if lyr.layer_id == "charm_particulate_domoic_lead1")
    g = lyr.grid
    vals = codec.decode((out / g.url).read_bytes(), g.width, g.height, g.scale_factor, g.add_offset)
    lats = g.lat_first + np.arange(g.height) * g.lat_step
    lons = g.lon_first + np.arange(g.width) * g.lon_step
    expected = []
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            dy = (la - p.lat) * 111.32
            dx = (lo - p.lon) * 111.32 * math.cos(math.radians(p.lat))
            if math.hypot(dx, dy) <= 15 and np.isfinite(vals[i, j]):
                expected.append(vals[i, j])
    s = next(lead for lead in p.charm.leads if lead.lead_days == 1).variables["particulate_domoic"]
    assert s.n == len(expected) and abs(s.median - float(np.median(expected))) < 1e-6


def test_ports_outside_forecast_data_get_no_values_not_zeros(out):
    _, coll = _run(out)
    sd = port(coll, "San Diego")  # outside the fixture C-HARM subset
    assert sd.charm.cells_in_radius == 0
    for lead in sd.charm.leads:
        for s in lead.variables.values():
            assert s.n == 0 and s.median is None
    assert sd.charm.history == {} and sd.charm.history_error
    assert sd.chlorophyll.latest is None and sd.chlorophyll.error


def test_history_is_real_dated_and_never_interpolated(out):
    _, coll = _run(out)
    h = port(coll, "Monterey").charm.history["particulate_domoic"]
    dates = [pt.date for pt in h]
    assert dates == sorted(dates) and dates[-1] == "2026-10-07"
    assert len(set(dates)) == len(dates) and len(dates) <= port_intel.HISTORY_DAYS
    assert all(pt.value is None or 0 <= pt.value <= 1 for pt in h)
    # cross-check one day against the recorded ERDDAP response
    import json

    idx = json.loads((FIXTURES / "recorded" / "index.json").read_text())
    url = next(u for u in idx if "wvcharmV3_0day.nc" in u and "(36.445)" in u)
    arrays, _, _ = charm.parse_netcdf((FIXTURES / "recorded" / idx[url]["file"]).read_bytes())
    a = arrays["particulate_domoic"][-1].astype(float)
    a[np.isclose(a, -99999.0)] = np.nan
    LA, LO = np.meshgrid(arrays["latitude"], arrays["longitude"] - 360, indexing="ij")
    d = port_intel.km(port(coll, "Monterey").lat, port(coll, "Monterey").lon, LA, LO)
    expected = float(np.median(a[(d <= 15) & np.isfinite(a)]))
    assert abs(h[-1].value - expected) < 1e-6


def test_chlorophyll_latest_composite(out):
    _, coll = _run(out)
    chl = port(coll, "Monterey").chlorophyll
    assert chl.error is None and chl.latest and chl.latest.n > 0
    assert chl.latest.median > 0 and chl.composite_days == 8
    assert chl.latest_center_date and chl.latest_center_date <= "2026-10-08"
    assert 0 < chl.latest_valid_fraction <= 1


def test_port_caveats_state_spatial_limits(out):
    _, coll = _run(out)
    for p in coll.ports:
        joined = " ".join(p.caveats)
        assert "do not describe conditions at the dock" in joined
        assert "not toxins" in joined
