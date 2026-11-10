import type { Bounds } from "@/nx/chart/mercator";

/**
 * Today's camera (journeys.md Journey 1; README §2): the port and the water that matters to it in one view,
 * the test area, the buoy and (off Moss Landing) the canyon head, with room around them for the soundings'
 * type. Points far from the port (a buoy 80 km off) are left out so the port never shrinks to a dot. The
 * host fits the box into the free part of the sheet, so the margin and the phone sheet are already excluded.
 */
type P = { lon: number; lat: number };

const KM_LAT = 111.32;
const kmLon = (lat: number) => KM_LAT * Math.cos((lat * Math.PI) / 180);
export const distanceKm = (a: P, b: P) => Math.hypot((a.lon - b.lon) * kmLon((a.lat + b.lat) / 2), (a.lat - b.lat) * KM_LAT);

export function todayBounds(port: P, points: P[], o: { maxKm?: number; minHalfKm?: number; pad?: number; south?: number } = {}): Bounds {
  const maxKm = o.maxKm ?? 70;
  const minHalf = o.minHalfKm ?? 20;
  const pad = o.pad ?? 0.22;
  const near = [port, ...points.filter((p) => Number.isFinite(p.lon) && Number.isFinite(p.lat) && distanceKm(port, p) <= maxKm)];
  let w = Math.min(...near.map((p) => p.lon));
  let e = Math.max(...near.map((p) => p.lon));
  let s = Math.min(...near.map((p) => p.lat));
  let n = Math.max(...near.map((p) => p.lat));
  // breathing room for the type set beside each point (soundings are about 220 px wide)
  const dx = Math.max((e - w) * pad, 0.02);
  const dy = Math.max((n - s) * pad, 0.02);
  w -= dx * 1.4;
  e += dx;
  s -= dy;
  n += dy;
  // never closer than minHalf km either side of the centre: the port's water, not just its harbour
  const cx = (w + e) / 2;
  const cy = (s + n) / 2;
  const hx = Math.max((e - w) / 2, minHalf / kmLon(cy));
  const hy = Math.max((n - s) / 2, (minHalf * 0.8) / KM_LAT);
  // phones: open water below the points, where the soundings' type can sit above the resting sheet
  const below = (o.south ?? 0) * hy;
  return [
    [cx - hx, cy - hy - below],
    [cx + hx, cy + hy],
  ];
}

/** A port's own water before a fishery is chosen: the port with the sea to its west and south-west in view. */
export function portWater(port: P): Bounds {
  return todayBounds(port, [{ lon: port.lon - 0.3, lat: port.lat - 0.08 }], { pad: 0.1 });
}
