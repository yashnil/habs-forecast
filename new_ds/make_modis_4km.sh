#!/usr/bin/env bash
set -euo pipefail

RAW="/Users/yashnilmohanty/Desktop/HABs_Research/Data"
OUT="/Users/yashnilmohanty/Desktop/HABs_Research/Processed/modis_l3m"
mkdir -p "$OUT"/{chlorophyll,kd490,nFLH}

command -v l3mapgen >/dev/null 2>&1 || { echo "❌  l3mapgen not on PATH"; exit 1; }

# mapping window
west=-125 east=-115 south=32 north=50

map_one() {                 # $1=in subdir   $2=suffix   $3=out subdir
  local in_dir="$RAW/$1" suf="$2" out_dir="$OUT/$3"
  echo "🗺️  $(basename "$in_dir")  →  $(basename "$out_dir")/"
  map_cnt=0; skip_cnt=0

  find "$in_dir" -name "AQUA_MODIS.20*.L3b.8D.${suf}.x.nc" | sort | while read -r f; do
      [[ $(basename "$f") > "AQUA_MODIS.20210630" ]] && break
      base=${f/_L3b.8D.${suf}.x.nc/}                     # strip suffix
      out="$out_dir/$(basename "${base}")_4km_L3m.nc"
      if [[ -f $out ]]; then
          ((skip_cnt++))
      else
          ((map_cnt++))
          echo "   map  $(basename "$out")"
          l3mapgen ifile="$f" ofile="$out" resolution=4km projection=platecarree \
                   west=$west east=$east north=$north south=$south
      fi
  done
  echo "     mapped: $map_cnt   skipped: $skip_cnt"
}

map_one chlorophyll CHL  chlorophyll
map_one kd490       KD   kd490
map_one nFLH        FLH  nFLH

echo "✅  Mapping complete – verify counts above."
