#!/usr/bin/env bash
# Copies a pinned, read-only snapshot of real CoastWatch data into public/ for the prototypes.
# Source: the production data branch (coastwatch-data) at a fixed commit, and the basemap assets
# committed on the redesign branch. Nothing is written to either branch. The snapshot is git-ignored.
set -euo pipefail
DATA_COMMIT="${DATA_COMMIT:-0353c1f38c5d2756f679c1cfdd91fd07b0713ea8}"   # data run 38328551760, 2026-10-13 00:42Z
ASSET_COMMIT="${ASSET_COMMIT:-fabe67d2940026dafa328a03ee267a7e6e34557a}" # feat/coastal-redesign: tinted relief
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
OUT="$HERE/public"
git -C "$REPO" cat-file -e "$DATA_COMMIT" 2>/dev/null || git -C "$REPO" fetch -q origin coastwatch-data
rm -rf "$OUT/data" "$OUT/basemap" "$OUT/fonts"
mkdir -p "$OUT/data/v1"

extract() { # commit, tree path, dest dir (keeps path below the tree path)
  local c="$1" p="$2" d="$3"
  git -C "$REPO" archive "$c" "$p" | tar -x -C "$d" --strip-components="$(awk -F/ '{print NF-1}' <<<"$p/")"
}
M=$(git -C "$REPO" show "$DATA_COMMIT:v1/manifest.json")
echo "$M" > "$OUT/data/v1/manifest.json"
for key in ports_url official_url port_intel_url observations_url fisheries_url; do
  f=$(node -e "console.log(JSON.parse(process.argv[1])['$key'])" "$M")
  git -C "$REPO" show "$DATA_COMMIT:v1/$f" > "$OUT/data/v1/$f"
done
# C-HARM run, the 24 h mean currents, the latest OLCI value grid and the multi-sensor colour tiles.
for p in $(node -e '
  const m=JSON.parse(process.argv[1]); const s=new Set();
  for (const L of m.layers) {
    const pick=u=>u&&s.add(u.split("/").slice(0,-1).join("/").replace(/\/(grid|tiles)$/,""));
    if (L.group_id==="charm") pick(L.grid.url);
    if (L.layer_id==="hfr2km_currents_mean24h") pick(L.vectors.u_grid.url);
    if (L.layer_id==="olci300_chl_latest"||L.layer_id==="multisensor_chl_latest") pick((L.tiles||{}).url_template.replace(/\/tiles\/.*$/,"/x"));
  }
  console.log([...s].join(" "))' "$M"); do
  mkdir -p "$OUT/data/v1/$p"
  extract "$DATA_COMMIT" "v1/$p" "$OUT/data/v1/$p"
done
git -C "$REPO" show "$DATA_COMMIT:v1/verification/charm-points.json" > "$OUT/data/v1/charm-points.json" 2>/dev/null || true
mkdir -p "$OUT/basemap" "$OUT/fonts"
extract "$ASSET_COMMIT" coastwatch-web/public/basemap "$OUT/basemap"
extract "$ASSET_COMMIT" coastwatch-web/public/fonts "$OUT/fonts"
echo "{\"data_commit\":\"$DATA_COMMIT\",\"asset_commit\":\"$ASSET_COMMIT\",\"snapshot_at\":\"$(date -u +%FT%TZ)\"}" > "$OUT/data/snapshot.json"
du -sh "$OUT/data" "$OUT/basemap" "$OUT/fonts"
