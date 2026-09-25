import type { HarborFeature } from "./types";

function toRad(d: number) {
  return (d * Math.PI) / 180;
}

/** Haversine km */
export function distanceKm(a: [number, number], b: [number, number]): number {
  const R = 6371;
  const [lon1, lat1] = a;
  const [lon2, lat2] = b;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(x)));
}

export function nearestHarbor(
  user: [number, number],
  features: HarborFeature[],
): { feature: HarborFeature; km: number } | null {
  if (!features.length) return null;
  let best = features[0];
  let bestD = distanceKm(user, best.geometry.coordinates);
  for (let i = 1; i < features.length; i++) {
    const f = features[i];
    const d = distanceKm(user, f.geometry.coordinates);
    if (d < bestD) {
      best = f;
      bestD = d;
    }
  }
  return { feature: best, km: bestD };
}

/**
 * @deprecated For the web app use `/api/gibs-chl-meta` and `@/lib/gibs` (VIIRS daily).
 * Heuristic date string YYYY-MM-DD (UTC).
 */
export function gibsChlDate(daysBack = 5): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - daysBack);
  return d.toISOString().slice(0, 10);
}
