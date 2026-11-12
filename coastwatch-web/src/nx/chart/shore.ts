import SHORE from "./shore.json";

/**
 * Where the water starts: the mainland's shore longitude by latitude (shore.json, built by scripts/build-shore.mjs
 * from the chart's own land). Pure, so cameras (src/nx/explore/camera.ts) and the regulation layer share it.
 */
const S = SHORE as { lat0: number; step: number; lon: (number | null)[] };

/** The mainland's shore longitude at a latitude, or null outside California. */
export function shoreLon(lat: number): number | null {
  const k = (lat - S.lat0) / S.step;
  const i = Math.floor(k);
  if (i < 0 || i + 1 >= S.lon.length) return null;
  const a = S.lon[i], b = S.lon[i + 1];
  if (a == null || b == null) return a ?? b;
  return a + (b - a) * (k - i);
}

/** The eastmost shore between two latitudes: where the water ends in a view of that stretch of coast. */
export function eastmostShore(south: number, north: number): number | null {
  let best: number | null = null;
  for (let lat = south; lat <= north + 1e-9; lat += S.step) {
    const v = shoreLon(lat);
    if (v != null && (best == null || v > best)) best = v;
  }
  return best;
}
