#!/usr/bin/env bash
# Download the current production dataset from GitHub Pages into <dir> (manifest,
# JSON artifacts and C-HARM value grids), then rebuild the prototype data and rasters.
# usage: scripts/fetch_pages.sh <dir>
set -euo pipefail
DIR=${1:?usage: fetch_pages.sh <dir>}
BASE=https://yashnil.github.io/habs-forecast/v1
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$DIR"
curl -sSf "$BASE/manifest.json" -o "$DIR/manifest.json"
python3 - "$DIR" <<'PY' | while read -r p; do mkdir -p "$DIR/$(dirname "$p")"; curl -sSf "$BASE/$p" -o "$DIR/$p"; done
import json, sys
m = json.load(open(sys.argv[1] + "/manifest.json"))
for k in ("ports_url", "official_url", "port_intel_url", "observations_url", "fisheries_url"):
    print(m[k])
for layer in m["layers"]:
    if layer.get("grid"):
        print(layer["grid"]["url"])
PY
python3 "$HERE/scripts/render_rasters.py" "$DIR" "$HERE/prototype/img"
python3 "$HERE/scripts/build_data.py" "$DIR" "$HERE/prototype"
