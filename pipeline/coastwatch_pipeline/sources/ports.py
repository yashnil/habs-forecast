"""California landing ports from CDFW 'Marine Landing Data System Ports' [ds3081].

The curated list (data/curated/ports.json) selects ports by CDFW port code; names,
port areas and coordinates come from CDFW. Replaces the hand-placed harbor points
that put Santa Barbara in the channel and listed Pillar Point twice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..context import RunContext
from ..models import PortFeature, PortProperties, PortsCollection, Provenance, Region

SOURCE_ID = "cdfw_ports"
SERVICE = "https://services2.arcgis.com/Uq9r85Potqm3MfRV/arcgis/rest/services/biosds3081_fpu/FeatureServer/0"
QUERY_URL = SERVICE + "/query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=json&resultRecordCount=2000"
# Coastal California bounding box (lon/lat) used as a sanity check
BBOX = (-124.5, 32.4, -117.0, 42.0)
REPO_ROOT = Path(__file__).resolve().parents[3]
CURATED = REPO_ROOT / "data" / "curated" / "ports.json"


class PortsError(Exception):
    pass


@dataclass
class PortsResult:
    collection: PortsCollection | None
    errors: list[str]


def build(ctx: RunContext, raw: dict, curated: dict) -> PortsCollection:
    if "features" not in raw:
        raise PortsError(f"unexpected ArcGIS response: {str(raw)[:200]}")
    by_code: dict[int, dict] = {}
    for f in raw["features"]:
        a = f.get("attributes", {})
        if isinstance(a.get("PortCode"), int):
            by_code[a["PortCode"]] = f
    features: list[PortFeature] = []
    problems: list[str] = []
    for p in curated["ports"]:
        code = p["port_code"]
        f = by_code.get(code)
        if f is None:
            problems.append(f"port code {code} ({p['display_name']}) not found in ds3081")
            continue
        a, g = f["attributes"], f.get("geometry") or {}
        x, y = g.get("x"), g.get("y")
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            problems.append(f"port code {code}: missing geometry")
            continue
        if not (BBOX[0] <= x <= BBOX[2] and BBOX[1] <= y <= BBOX[3]):
            problems.append(f"port code {code}: point ({x:.4f}, {y:.4f}) outside coastal California")
            continue
        if a.get("MajorPort") != p["expected_port_area"]:
            problems.append(
                f"port code {code}: CDFW port area {a.get('MajorPort')!r} != expected {p['expected_port_area']!r}"
            )
            continue
        features.append(
            PortFeature(
                geometry={"type": "Point", "coordinates": [round(x, 5), round(y, 5)]},
                properties=PortProperties(
                    port_code=code,
                    name=str(a.get("PortName")),
                    display_name=p["display_name"],
                    port_area=str(a.get("MajorPort")),
                    port_area_code=int(a.get("MajorPortC")),
                    county=p["county"],
                    region=p["region"],
                ),
            )
        )
    if problems:
        raise PortsError("; ".join(problems))
    region_ids = {r["id"] for r in curated.get("regions", [])}
    missing_regions = sorted({f.properties.region for f in features} - region_ids)
    if missing_regions:
        raise PortsError(f"ports reference unknown regions {missing_regions}")
    return PortsCollection(
        features=features,
        regions=[Region(**r) for r in curated.get("regions", [])],
        caveats=[
            "Port points are CDFW reference locations for landing records; CDFW notes a point may not represent the exact harbour location.",
        ],
        provenance=Provenance(
            source_id=SOURCE_ID,
            source_name="CDFW Marine Landing Data System Ports [ds3081]",
            source_url="https://www.arcgis.com/home/item.html?id=9a8ab5ffa69e4cbeb8e7a4bd9a3a0dac",
            dataset_id="ds3081",
            institution="California Department of Fish and Wildlife, Marine Region",
            license="CDFW BIOS public dataset; attribute CDFW.",
            retrieved_at=ctx.now_iso,
            request_urls=[QUERY_URL],
            upstream_metadata={"curated_list": "data/curated/ports.json", "curated_last_reviewed": curated.get("last_reviewed", "")},
            pipeline_version=ctx.pipeline_version,
            pipeline_run_id=ctx.run_id,
        ),
    )


def run(ctx: RunContext, curated_path: Path = CURATED) -> PortsResult:
    try:
        curated = json.loads(curated_path.read_text())
        raw = json.loads(ctx.fetcher(QUERY_URL).body)
        return PortsResult(build(ctx, raw, curated), [])
    except Exception as e:
        return PortsResult(None, [f"ports: {e}"])
