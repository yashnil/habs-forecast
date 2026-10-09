"""Snapshot the published CoastWatch dataset for the design-reset prototypes.

Reads a local copy of the production GitHub Pages `v1/` directory (downloaded with
`fetch_pages.sh`) and writes `prototype/data/coastwatch-data.js`, a single script that
sets `window.CW_DATA`. Values are copied, never edited; fields the prototypes do not
display are dropped to keep the file small. The prototypes therefore show exactly the
numbers, dates, labels and review states that production published in that run.

usage: python build_data.py <pages-v1-dir> <prototype-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OBS_VARIABLES = ["pDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"]


def load(src: Path, name: str):
    return json.loads((src / name).read_text())


def main(src: Path, proto: Path) -> None:
    m = load(src, "manifest.json")
    rasters = json.loads((proto / "img" / "rasters.json").read_text())

    layers = []
    for layer in m["layers"]:
        entry = {
            k: layer[k]
            for k in ("layer_id", "group_id", "product_class", "title", "short_title", "variable", "units", "threshold_text", "description", "time", "freshness")
        }
        if layer["layer_id"] in rasters:
            entry["raster"] = rasters[layer["layer_id"]]
        if layer.get("tiles"):
            entry["tiles"] = {k: layer["tiles"][k] for k in ("url_template", "max_native_zoom", "legend_url")}
        layers.append(entry)

    official = load(src, m["official_url"])
    port_intel = load(src, m["port_intel_url"])
    for p in port_intel["ports"]:
        if p.get("chlorophyll"):
            p["chlorophyll"].pop("history", None)

    obs = load(src, m["observations_url"])
    for s in obs["stations"]:
        s.pop("depths_m", None)
        s.pop("request_url", None)
        s["series"] = [x for x in s["series"] if x["variable"] in OBS_VARIABLES]
        s["summaries"] = [x for x in s["summaries"] if x["variable"] in OBS_VARIABLES]
        if s.get("charm"):
            s["charm"]["history"] = {"particulate_domoic": s["charm"]["history"].get("particulate_domoic", [])}
    obs["variables"] = [v for v in obs["variables"] if v["id"] in OBS_VARIABLES]

    data = {
        "snapshot": {
            "base_url": "https://yashnil.github.io/habs-forecast/v1",
            "generated_at": m["generated_at"],
            "pipeline_run_id": m["pipeline_run_id"],
            "pipeline_version": m["pipeline_version"],
            "note": "Copied unchanged from the production GitHub Pages dataset; display-only fields dropped.",
        },
        "sources": m["sources"],
        "forecast_runs": m["forecast_runs"],
        "layers": layers,
        "ports": load(src, m["ports_url"]),
        "official": official,
        "port_intel": port_intel,
        "observations": obs,
        "fisheries": load(src, m["fisheries_url"]),
    }
    out = proto / "data" / "coastwatch-data.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.CW_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB) from run {m['pipeline_run_id']}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
