# /// script
# requires-python = ">=3.11"
# dependencies = ["pyshp>=2.3", "shapely>=2.0"]
# ///
"""Build the Fisheries page's small reference map (M7, decision F2).

Output, written to coastwatch-web/src/content/fisheries-geo.json and committed (it changes only
when this script changes):

- ports: CDFW's Marine Landing Data System ports, each with its CDFW port area ("major port").
  CDFW, Marine Region: "Marine Landing Data System Ports - R7 - CDFW [ds3081]", BIOS, licensed
  CC BY 4.0. CDFW publishes these as geocoded *points* ("The point may not represent the exact
  port location"); it publishes no port-area boundaries, so the page draws points, never areas.
  Coordinates are rounded to 0.01° (about 1 km), well below what the small map can show.
- outline: California's outline from Natural Earth 1:50m admin-1 states and provinces (public
  domain), simplified to 0.02°. It is a locator, not a coastline for measurement.

The map carries no landings values: NOAA's annual figures are statewide and CoastWatch does not
divide them among port areas.

usage: uv run --script pipeline/scripts/build_fisheries_geo.py <ne_50m_admin_1_states_provinces.shp> <out.json>
       (Natural Earth: https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip;
        the CDFW points are fetched from the ArcGIS FeatureServer below)
"""

import json
import sys
import urllib.request
from datetime import UTC, datetime

import shapefile
from shapely.geometry import shape

CDFW_ITEM = "https://gis.data.ca.gov/datasets/CDFW::marine-landing-data-system-ports-r7-cdfw-ds3081"
CDFW_QUERY = "https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/biosds3081_fpu/FeatureServer/0/query?where=1%3D1&outFields=PortName,PortCode,MajorPort,MajorPortC&outSR=4326&f=geojson"


def title(s: str) -> str:
    cap = lambda w: "-".join(x.capitalize() for x in w.split("-"))
    return " ".join(cap(w) for w in s.replace("/", " / ").split()).replace(" / ", "/")


def main(shp: str, out: str) -> None:
    with urllib.request.urlopen(CDFW_QUERY, timeout=60) as r:
        gj = json.load(r)
    ports = []
    for f in gj["features"]:
        p = f["properties"]
        lon, lat = f["geometry"]["coordinates"]
        ports.append({"name": title(p["PortName"]), "code": p["PortCode"], "area": title(p["MajorPort"]), "area_code": p["MajorPortC"], "lon": round(lon, 2), "lat": round(lat, 2)})
    ports.sort(key=lambda p: (p["area_code"], -p["lat"]))

    reader = shapefile.Reader(shp)
    fields = [f[0] for f in reader.fields[1:]]
    outline = None
    for sr in reader.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        if rec.get("name") == "California" and rec.get("admin") == "United States of America":
            g = shape(sr.shape.__geo_interface__).simplify(0.02, preserve_topology=True)
            polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
            outline = [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords] for p in polys if p.area > 0.004]
    assert outline, "California not found in the shapefile"

    doc = {
        "built_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "ports_source": {
            "title": "Marine Landing Data System Ports (ds3081)",
            "publisher": "California Department of Fish and Wildlife, Marine Region (BIOS)",
            "url": CDFW_ITEM,
            "license": "CC BY 4.0",
            "note": "Ports and major port areas used for commercial landings reported to CDFW. Points are geocoded and may not represent the exact port location.",
        },
        "outline_source": {"title": "Natural Earth 1:50m admin-1 states and provinces", "url": "https://www.naturalearthdata.com/", "license": "Public domain"},
        "ports": ports,
        "outline": outline,
    }
    with open(out, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
        fh.write("\n")
    print(f"{len(ports)} ports, {len(outline)} outline rings -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
