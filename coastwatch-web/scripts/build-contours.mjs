#!/usr/bin/env node
/**
 * public/chart/contours.geojson: the depth areas' edges (public/chart/depth.geojson, "shallower than d" for d = 10 …
 * 2000 m) as lines, without the seams where two sources meet (the Monterey inset's rectangle, where the Coastal
 * Relief Model's areas meet ETOPO's) or the frame's own edges. The chart draws every edge as a fine light isobath
 * and the Depth layer's four (20, 50, 100, 200 m) as fathom contours (v14 README §4). Each line carries `d` (m),
 * `fm` (its true fathom value, rounded) and `src`.
 *
 *   node scripts/build-contours.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(fileURLToPath(new URL(".", import.meta.url)), "..");
const chart = JSON.parse(readFileSync(join(root, "public/chart/chart.json"), "utf8"));
const depth = JSON.parse(readFileSync(join(root, "public/chart/depth.geojson"), "utf8"));
const [iw, is, ie, inn] = chart.frame.inset;
const [cw, cs, ce, cn] = chart.frame.ca;
const EPS = 0.004;
const near = (a, b) => Math.abs(a - b) < EPS;
// a segment lies on a seam when both its ends lie on the same edge of the inset or of the frame
const onEdge = (p, q) => {
  for (const [w, s, e, n] of [
    [iw, is, ie, inn],
    [cw, cs, ce, cn],
  ]) {
    if (near(p[0], w) && near(q[0], w)) return true;
    if (near(p[0], e) && near(q[0], e)) return true;
    if (near(p[1], s) && near(q[1], s)) return true;
    if (near(p[1], n) && near(q[1], n)) return true;
  }
  return false;
};
const round = (v) => Math.round(v * 1e5) / 1e5;
const features = [];
for (const f of depth.features) {
  const d = f.properties.d;
  const polys = f.geometry.type === "MultiPolygon" ? f.geometry.coordinates : [f.geometry.coordinates];
  for (const poly of polys)
    for (const ring of poly) {
      let run = [];
      const flush = () => {
        if (run.length >= 2) features.push({ type: "Feature", properties: { d, fm: Math.round(d / 1.8288), src: f.properties.src }, geometry: { type: "LineString", coordinates: run.map(([x, y]) => [round(x), round(y)]) } });
        run = [];
      };
      for (let i = 0; i < ring.length - 1; i++) {
        const p = ring[i], q = ring[i + 1];
        if (onEdge(p, q)) {
          flush();
          continue;
        }
        if (!run.length) run.push(p);
        run.push(q);
      }
      flush();
    }
}
const out = { type: "FeatureCollection", features };
writeFileSync(join(root, "public/chart/contours.geojson"), JSON.stringify(out));
console.log(`contours.geojson: ${features.length} lines`);
