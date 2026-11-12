#!/usr/bin/env node
/**
 * node scripts/build-shore.mjs: the mainland's shore longitude by latitude (src/nx/chart/shore.json), from the
 * chart's own land (public/chart/land.geojson, built by pipeline/scripts/build_chart.py). For each latitude it
 * keeps the westmost crossing of the mainland polygon (the largest one), so islands (the Farallones, the
 * Channel Islands) never stand in for the coast. Used where the chart must know where the water starts: a
 * regulation line drawn from the shore out to sea, a camera that frames water rather than land.
 * Rerun after build_chart.py changes the land.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const land = JSON.parse(readFileSync(join(here, "../public/chart/land.geojson"), "utf8"));

const area = (ring) => {
  const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
  return (Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys));
};
const rings = land.features.map((f) => f.geometry.coordinates[0]);
const mainland = rings.reduce((a, b) => (area(b) > area(a) ? b : a));

const LAT0 = 32.4, LAT1 = 42.2, STEP = 0.02;
const lon = [];
for (let k = 0; LAT0 + k * STEP <= LAT1 + 1e-9; k++) {
  const lat = LAT0 + k * STEP;
  let west = Infinity;
  for (let i = 0; i < mainland.length - 1; i++) {
    const [x0, y0] = mainland[i], [x1, y1] = mainland[i + 1];
    if ((y0 <= lat && y1 > lat) || (y1 <= lat && y0 > lat)) {
      const x = x0 + ((lat - y0) / (y1 - y0)) * (x1 - x0);
      if (x < west) west = x;
    }
  }
  lon.push(Number.isFinite(west) ? Math.round(west * 1e4) / 1e4 : null);
}
const out = { _source: "scripts/build-shore.mjs from public/chart/land.geojson (mainland, westmost crossing)", lat0: LAT0, step: STEP, lon };
writeFileSync(join(here, "../src/nx/chart/shore.json"), JSON.stringify(out) + "\n");
console.log(`shore.json: ${lon.length} latitudes`);
