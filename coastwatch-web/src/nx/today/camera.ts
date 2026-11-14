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

export function todayBounds(port: P, points: P[], o: { maxKm?: number; minHalfKm?: number; pad?: number; south?: number; eastKm?: number } = {}): Bounds {
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
  // and room east of the port for its name, so the name never sits over the water the soundings need
  const east = (o.eastKm ?? 0) / kmLon(cy);
  return [
    [cx - hx, cy - hy - below],
    [cx + hx + east, cy + hy],
  ];
}

/** A port's own water before a fishery is chosen: the port with the sea to its west and south-west in view. */
export function portWater(port: P): Bounds {
  return todayBounds(port, [{ lon: port.lon - 0.3, lat: port.lat - 0.08 }], { pad: 0.1 });
}

/**
 * The phone's view above the resting sheet: small, so it is composed rather than fitted. The port sits right
 * of centre with room to its east for its name (over land); the readings' points sit high with open water to
 * their west and south, where the soundings' type is set (Soundings.tsx tries those anchors first on a phone),
 * so the test reads over the water it describes, not over the hills. When the two do not both fit, the buoy
 * drops to the margin's water row, by priority.
 */
export function phoneBounds(port: P, points: P[], maxKm = 70): Bounds {
  const near = points.filter((p) => distanceKm(port, p) <= maxKm);
  const k = kmLon(port.lat);
  const west = Math.min(port.lon, ...near.map((p) => p.lon)) - 18 / k;
  const e = port.lon + 26 / k;
  const w = Math.min(west, e - 48 / k);
  const n = Math.max(port.lat, ...near.map((p) => p.lat)) + 10 / KM_LAT;
  const s = Math.min(port.lat, ...near.map((p) => p.lat)) - 26 / KM_LAT;
  return [
    [w, s],
    [e, n],
  ];
}

/**
 * v14 Today (the approved synthesis): a RAMP reader's camera frames the port's whole whale zone, its limits drawn from
 * the shore out at their real latitudes, with the open water west of the port; the ticket stands east of the port (the
 * host brings the port to the ticket's edge). Readings near the port are kept in. Without a zone, the port's water.
 */
export function zoneBounds(port: P, zone: { north: number; south: number } | null, points: P[] = [], maxKm = 70): Bounds {
  if (!zone) return todayBounds(port, points, { minHalfKm: 40 });
  const near = points.filter((p) => distanceKm(port, p) <= maxKm);
  // the zone, but never more than about 0.8° either side of the port (zone 4 runs to Lopez Point)
  const n = Math.max(Math.min(zone.north + 0.04, port.lat + 0.8), ...near.map((p) => p.lat + 0.05));
  const s = Math.min(Math.max(zone.south - 0.04, port.lat - 0.8), ...near.map((p) => p.lat - 0.05));
  const latSpan = n - s;
  const e = port.lon + 0.06;
  const w = Math.min(e - (latSpan * 1.25) / Math.cos((port.lat * Math.PI) / 180), ...near.map((p) => p.lon - 0.1));
  return [
    [w, s],
    [e, n],
  ];
}
