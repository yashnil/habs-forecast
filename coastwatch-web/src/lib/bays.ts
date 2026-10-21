/**
 * California's enclosed bays and estuaries, client side (v6): where satellite chlorophyll is
 * approximate (turbid water) and where the 8-day outlook is never drawn or read.
 *
 * The pipeline masks the outlook's bay cells itself (derived/index.json). This list is the web's
 * own backstop, so an index published before that mask, or a cell that straddles a bay mouth, is
 * still never shown over bay water: MODIS chlorophyll is unreliable in turbid water, and the
 * outlook's error bars there were its smallest only because the bays are always "green".
 *
 * Conservative outlines (lon, lat), drawn over the OpenStreetMap ocean water the satellite scope
 * uses (data/reference/marine_water_mask.json) and checked against it: each is cut at the bay's
 * mouth (the Golden Gate bridge, the Tomales, Humboldt, Morro and San Diego entrances, Elkhorn
 * Slough's Highway 1 bridge), so open-coast water is never inside. They include land freely
 * (land is never drawn). Not a legal or navigational boundary.
 */
export type Bay = { id: string; name: string; ring: [number, number][]; bbox: [number, number, number, number] };

const RINGS: { id: string; name: string; ring: [number, number][] }[] = [
  {
    // San Francisco, San Pablo and Suisun Bays with Carquinez Strait, east of the Golden Gate bridge
    id: "sf-bay",
    name: "San Francisco Bay",
    ring: [
      [-122.476, 37.806], [-122.44, 37.75], [-122.42, 37.65], [-122.4, 37.55], [-122.25, 37.42], [-122.05, 37.38], [-121.9, 37.42], [-121.95, 37.6],
      [-122.15, 37.8], [-122.22, 37.92], [-122.2, 38.0], [-121.8, 38.0], [-121.8, 38.14], [-122.05, 38.16], [-122.25, 38.13], [-122.4, 38.17],
      [-122.52, 38.1], [-122.52, 37.98], [-122.5, 37.93], [-122.5, 37.88], [-122.474, 37.832],
    ],
  },
  { id: "tomales-bay", name: "Tomales Bay", ring: [[-122.958, 38.232], [-122.93, 38.24], [-122.86, 38.17], [-122.79, 38.08], [-122.79, 38.04], [-122.83, 38.04], [-122.9, 38.1], [-122.93, 38.17], [-122.962, 38.212]] },
  { id: "humboldt-bay", name: "Humboldt Bay", ring: [[-124.222, 40.772], [-124.188, 40.84], [-124.15, 40.89], [-124.05, 40.89], [-124.05, 40.78], [-124.18, 40.62], [-124.235, 40.62], [-124.24, 40.7], [-124.232, 40.758]] },
  { id: "morro-bay", name: "Morro Bay", ring: [[-120.858, 35.372], [-120.85, 35.4], [-120.8, 35.4], [-120.8, 35.32], [-120.85, 35.32], [-120.858, 35.345]] },
  { id: "elkhorn-slough", name: "Elkhorn Slough", ring: [[-121.78, 36.8], [-121.78, 36.82], [-121.76, 36.87], [-121.72, 36.87], [-121.72, 36.8]] },
  { id: "san-diego-bay", name: "San Diego Bay", ring: [[-117.237, 32.69], [-117.245, 32.72], [-117.2, 32.75], [-117.1, 32.72], [-117.08, 32.58], [-117.112, 32.585], [-117.147, 32.655], [-117.19, 32.684]] },
];

export const BAYS: Bay[] = RINGS.map((b) => {
  const xs = b.ring.map((p) => p[0]);
  const ys = b.ring.map((p) => p[1]);
  return { ...b, bbox: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] };
});

/** Even-odd point in polygon (ring of [lon, lat]). */
function inRing(ring: [number, number][], lon: number, lat: number): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    if (yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** The bay containing the point, or null (the open coast, or land outside every bay outline). */
export function bayAt(lat: number, lon: number): Bay | null {
  for (const b of BAYS) {
    const [w, s, e, n] = b.bbox;
    if (lon < w || lon > e || lat < s || lat > n) continue;
    if (inRing(b.ring, lon, lat)) return b;
  }
  return null;
}

/** The bays whose outline box meets [w, s, e, n] (a map tile's extent): only these need a per-pixel test. */
export function baysIn(w: number, s: number, e: number, n: number): Bay[] {
  return BAYS.filter((b) => b.bbox[0] <= e && b.bbox[2] >= w && b.bbox[1] <= n && b.bbox[3] >= s);
}

/**
 * Clear every pixel of an RGBA tile (Web Mercator z/x/y, size × size) whose centre lies in a bay.
 * Cheap where it matters: a tile that meets no bay's box is left untouched without a per-pixel test.
 */
export function clearBays(px: Uint8ClampedArray, z: number, x: number, y: number, size: number): void {
  const n = 2 ** z;
  const lonAt = (k: number) => ((x + (k + 0.5) / size) / n) * 360 - 180;
  const latAt = (k: number) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + (k + 0.5) / size)) / n))) * 180) / Math.PI;
  const bays = baysIn(lonAt(-0.5), latAt(size - 0.5), lonAt(size - 0.5), latAt(-0.5));
  if (!bays.length) return;
  const lons = new Float64Array(size);
  for (let k = 0; k < size; k++) lons[k] = lonAt(k);
  for (let py = 0; py < size; py++) {
    const lat = latAt(py);
    const rowBays = bays.filter((b) => lat >= b.bbox[1] && lat <= b.bbox[3]);
    if (!rowBays.length) continue;
    for (let pxi = 0; pxi < size; pxi++) {
      const o = (py * size + pxi) * 4;
      if (px[o + 3] === 0) continue;
      const lon = lons[pxi];
      if (rowBays.some((b) => lon >= b.bbox[0] && lon <= b.bbox[2] && inRing(b.ring, lon, lat))) {
        px[o] = px[o + 1] = px[o + 2] = px[o + 3] = 0;
      }
    }
  }
}

/** The outlook's words in a bay (readouts and the inspector). */
export const NO_OUTLOOK_IN_BAYS = "No outlook in bays (turbid water; MODIS chlorophyll unreliable).";
/** The caveat on observed (and estimated) chlorophyll in a bay. */
export const BAY_TURBID = "Bay chlorophyll is approximate (turbid water)";
