# /// script
# requires-python = ">=3.11"
# dependencies = ["shapely>=2.1,<3"]
# ///
"""Build the Ocean Map's California county reference layer from the state's official boundaries.

Source: California Department of Technology, "California County Boundaries and Identifiers"
(shoreline version; boundaries from the California Department of Tax and Fee Administration),
https://gis.data.ca.gov/datasets/CDT::california-county-boundaries-and-identifiers, public domain.
The "with Coastal Buffers" variant of the same dataset is read only to know which counties face
the ocean or San Francisco Bay; its offshore polygons are tax-jurisdiction areas, not marine or
regulatory boundaries, and are never drawn.

Method:
- all 58 counties are checked (names, FIPS codes, valid geometry);
- the polygons are simplified together as a coverage (shapely.coverage_simplify), so a border
  shared by two counties stays one line;
- only the borders *between* counties are kept: the state's outer edge (the coastline and the
  state lines) is left to the basemap, so the map never shows a second coastline;
- one label point per county (pole of inaccessibility), with its FIPS code and whether it faces
  the ocean or the Bay;
- coordinates are rounded to 1e-4 degrees (about 10 m).

Output (committed; it changes only when the source dataset or this script changes):
coastwatch-web/public/basemap/ca-counties.json

usage: uv run --script pipeline/scripts/build_counties.py <shoreline.geojson> <coastal-buffers.geojson> <out.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import shapely
from shapely.geometry import mapping, shape
from shapely.ops import polylabel

TOLERANCE_DEG = 0.0012  # ~110 m: below a pixel at zoom 10, the closest zoom the layer is styled for
SOURCE = {
    "title": "California County Boundaries and Identifiers",
    "publisher": "California Department of Technology (boundaries from the California Department of Tax and Fee Administration)",
    "url": "https://gis.data.ca.gov/datasets/CDT::california-county-boundaries-and-identifiers",
    "service": "https://services3.arcgis.com/uknczv4rpevve42E/arcgis/rest/services/California_County_Boundaries_and_Identifiers_Blue_Version_view/FeatureServer/1",
    "license": "Public domain (no warranty)",
}
COUNTIES_58 = 58


def rnd(geom):
    return shapely.set_precision(geom, 1e-4)


def coords(g, nd=4):
    def r(c):
        return [round(c[0], nd), round(c[1], nd)]

    m = mapping(g)
    if m["type"] == "LineString":
        return {"type": "LineString", "coordinates": [r(c) for c in m["coordinates"]]}
    if m["type"] == "MultiLineString":
        return {"type": "MultiLineString", "coordinates": [[r(c) for c in line] for line in m["coordinates"]]}
    if m["type"] == "Point":
        return {"type": "Point", "coordinates": r(m["coordinates"])}
    raise ValueError(m["type"])


def main(shore_path: str, buffer_path: str, out_path: str) -> None:
    raw = Path(shore_path).read_bytes()
    src = json.loads(raw)
    feats = src["features"]
    names = [f["properties"]["CDT_NAME_SHORT"] for f in feats]
    fips = [f["properties"]["CENSUS_GEOID"] for f in feats]
    assert len(feats) == COUNTIES_58 and len(set(names)) == COUNTIES_58 and len(set(fips)) == COUNTIES_58, (len(feats), len(set(names)))
    polys = [shape(f["geometry"]) for f in feats]
    assert all(p.is_valid for p in polys), "invalid source geometry"

    # which counties face the ocean or the Bay, from the same dataset's coastal-buffer variant
    buf = json.loads(Path(buffer_path).read_bytes())
    facing: dict[str, str] = {}
    for f in buf["features"]:
        p = f["properties"]
        kind = (p.get("OFFSHORE") or "").strip().lower()
        if kind in ("ocean", "bay"):
            n = p.get("CDT_NAME_SHORT")
            facing[n] = "ocean" if kind == "ocean" or facing.get(n) == "ocean" else "bay"
    assert set(facing) <= set(names), set(facing) - set(names)

    simple = list(shapely.coverage_simplify(polys, TOLERANCE_DEG))
    simple = [shapely.make_valid(g) for g in simple]
    borders = shapely.union_all([g.boundary for g in simple])
    outer = shapely.union_all(simple).boundary
    inner = borders.difference(outer.buffer(1e-6))
    inner = shapely.line_merge(shapely.union_all(inner)) if not inner.is_empty else inner

    labels = []
    for f, g, orig in zip(feats, simple, polys):
        p = f["properties"]
        big = max(getattr(orig, "geoms", [orig]), key=lambda x: x.area)  # mainland part (not islands)
        pt = polylabel(big, tolerance=0.005)
        labels.append({
            "type": "Feature",
            "geometry": coords(pt, 3),
            "properties": {"name": p["CDT_NAME_SHORT"], "fips": p["CENSUS_GEOID"], "coast": facing.get(p["CDT_NAME_SHORT"], "inland")},
        })
    labels.sort(key=lambda x: x["properties"]["name"])

    out = {
        "type": "FeatureCollection",
        "source": {**SOURCE, "sha256": hashlib.sha256(raw).hexdigest(), "simplified_deg": TOLERANCE_DEG},
        "features": [
            {"type": "Feature", "geometry": coords(inner), "properties": {"kind": "border"}},
            *[{**l, "properties": {**l["properties"], "kind": "label"}} for l in labels],
        ],
    }
    text = json.dumps(out, separators=(",", ":"))
    Path(out_path).write_text(text)
    n_ocean = sum(v == "ocean" for v in facing.values())
    n_bay = sum(v == "bay" for v in facing.values())
    print(json.dumps({"counties": len(labels), "ocean_facing": n_ocean, "bay_facing": n_bay, "border_vertices": len(shapely.get_coordinates(inner)), "bytes": len(text)}))


if __name__ == "__main__":
    main(*sys.argv[1:4])
