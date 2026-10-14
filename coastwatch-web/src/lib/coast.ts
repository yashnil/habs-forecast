import type { LayerArtifact, Manifest, PortsCollection, ValueGrid } from "@/generated/schema";
import { HARVEST_SITES } from "@/content/coast";
import type { StationLite } from "@/lib/data";
import { cellAt, distanceKm, valueOf, type Cell } from "@/lib/grid";
import { compass } from "@/lib/currents";

/** A location in My Coast: a CDFW port, a named harvest site, or a point chosen on the map. */
export type CoastPlace = {
  key: string; // "port-593", "site-clam-beach", "point-36.950,-122.050"
  kind: "port" | "site" | "point";
  name: string;
  lat: number;
  lon: number;
  county: string | null;
  note: string | null;
};

export type Audience = "all" | "commercial" | "recreational" | "shellfish";
export const AUDIENCES: { id: Audience; label: string }[] = [
  { id: "all", label: "Everything" },
  { id: "commercial", label: "Commercial fishing" },
  { id: "recreational", label: "Recreational fishing" },
  { id: "shellfish", label: "Shellfish harvest" },
];

export function portPlaces(ports: PortsCollection | null): CoastPlace[] {
  return (ports?.features ?? []).map((f) => {
    const [lon, lat] = f.geometry.coordinates as number[];
    return { key: `port-${f.properties.port_code}`, kind: "port" as const, name: f.properties.display_name, lat, lon, county: f.properties.county ?? null, note: "CDFW port" };
  });
}

export function sitePlaces(): CoastPlace[] {
  return HARVEST_SITES.map((s) => ({ key: `site-${s.id}`, kind: "site" as const, name: s.name, lat: s.lat, lon: s.lon, county: s.county, note: `CDPH monitoring site (${s.sampled})` }));
}

const POINT = /^point-(-?\d{1,2}\.\d{1,4}),(-?\d{1,3}\.\d{1,4})$/;
/** California's coastal waters, loosely: a chosen point outside this is refused. */
export const COAST_BOUNDS = { south: 32.4, north: 42.1, west: -126.0, east: -116.9 };

export function pointKey(lat: number, lon: number): string {
  return `point-${lat.toFixed(3)},${lon.toFixed(3)}`;
}

/** Resolve a place key from a link. Unknown or malformed keys resolve to null (the page says so). */
export function resolvePlace(key: string | null | undefined, ports: PortsCollection | null): CoastPlace | null {
  if (!key) return null;
  const m = POINT.exec(key);
  if (m) {
    const lat = Number(m[1]);
    const lon = Number(m[2]);
    if (!(lat >= COAST_BOUNDS.south && lat <= COAST_BOUNDS.north && lon >= COAST_BOUNDS.west && lon <= COAST_BOUNDS.east)) return null;
    return { key: pointKey(lat, lon), kind: "point", name: `${lat.toFixed(3)}°N, ${Math.abs(lon).toFixed(3)}°W`, lat, lon, county: null, note: "Point chosen on the map" };
  }
  return [...portPlaces(ports), ...sitePlaces()].find((p) => p.key === key) ?? null;
}

export function parseAudience(v: string | null | undefined): Audience {
  return v === "commercial" || v === "recreational" || v === "shellfish" ? v : "all";
}

/** The page's link: the place (waters), the home port and the audience. */
export function coastHref(p: { place?: string | null; home?: string | null; for?: Audience; species?: string | null; date?: string | null }): string {
  const q = new URLSearchParams();
  if (p.place) q.set("place", p.place);
  if (p.home) q.set("home", p.home);
  if (p.for && p.for !== "all") q.set("for", p.for);
  if (p.species && p.species !== "any") q.set("species", p.species);
  // the trip day (Pacific date); absent means today
  if (p.date && /^\d{4}-\d{2}-\d{2}$/.test(p.date)) q.set("date", p.date);
  // commas in a point key stay readable in shared links
  const s = q.toString().replace(/%2C/g, ",");
  return s ? `/coast?${s}` : "/coast";
}

// ---------------------------------------------------------------- measurements nearby

export const STATION_RADIUS_KM = 40;
export type Recency = "recent" | "older" | "old" | "none_this_year";
/** Ages for a laboratory result, by sample date (not by when it was published). */
export function recency(days: number): Recency {
  return days <= 14 ? "recent" : days <= 90 ? "older" : days <= 365 ? "old" : "none_this_year";
}
export const RECENCY_LABEL: Record<Recency, string> = {
  recent: "sampled in the last 2 weeks",
  older: "sampled 2 weeks to 3 months ago",
  old: "sampled 3 to 12 months ago",
  none_this_year: "no sample in the last year",
};

/** "today", "12 days ago", "over 4 years ago": exact for recent samples, rounded down past a year. */
export function ageWords(days: number): string {
  if (days === 0) return "today";
  if (days === 1) return "1 day ago";
  if (days < 365) return `${days} days ago`;
  const y = Math.floor(days / 365);
  return y === 1 ? "over a year ago" : `over ${y} years ago`;
}

export function ageDays(date: string, now: Date): number {
  return Math.max(0, Math.floor((now.getTime() - Date.parse(`${date.slice(0, 10)}T12:00:00Z`)) / 86_400_000));
}

export type StationNear = { station: StationLite; km: number; bearing: string };

/** Stations within `radius` km, nearest first; when none, the single nearest station anywhere. */
export function stationsNear(stations: StationLite[] | null, lat: number, lon: number, radius = STATION_RADIUS_KM): { near: StationNear[]; nearestBeyond: StationNear | null } {
  const all = (stations ?? [])
    .map((s) => ({ station: s, km: distanceKm(lat, lon, s.lat, s.lon), bearing: bearingName(lat, lon, s.lat, s.lon) }))
    .sort((a, b) => a.km - b.km);
  const near = all.filter((x) => x.km <= radius);
  return { near, nearestBeyond: near.length ? null : (all[0] ?? null) };
}

export function bearingName(lat1: number, lon1: number, lat2: number, lon2: number): string {
  const dy = lat2 - lat1;
  const dx = (lon2 - lon1) * Math.cos(((lat1 + lat2) / 2) * (Math.PI / 180));
  if (Math.hypot(dx, dy) < 1e-6) return "here";
  const deg = (Math.atan2(dx, dy) * 180) / Math.PI;
  return compass((deg + 360) % 360);
}

// ---------------------------------------------------------------- forecast availability

export type CellValue = { cell: Cell; value: number; km: number; bearing: string };
export type Availability = {
  /** value in the cell that contains the location, when there is one */
  here: CellValue | null;
  /** the nearest cell with a value, when the location's own cell has none */
  nearest: CellValue | null;
  /** cells with a value within `radiusKm` */
  region: { radiusKm: number; n: number; median: number; min: number; max: number } | null;
};

/**
 * Where C-HARM has values relative to a location. Nothing is interpolated: `here` is the value of
 * the cell containing the point (or null), `nearest` is a real cell elsewhere, reported with its
 * distance and direction from the point to the cell centre, and never as a value at the point.
 */
export function availability(g: ValueGrid, codes: Uint16Array, lat: number, lon: number, searchKm = 60, radiusKm = 15): Availability {
  const c = cellAt(g, lat, lon);
  const here = c ? valueOf(g, codes, c) : null;
  const rCells = Math.ceil(searchKm / (Math.abs(g.lat_step) * 111.32 * Math.cos((lat * Math.PI) / 180))) + 1;
  const row0 = Math.floor((lat - g.lat_first) / g.lat_step + 0.5);
  const col0 = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
  let best: CellValue | null = null;
  const inRadius: number[] = [];
  for (let dr = -rCells; dr <= rCells; dr++) {
    for (let dc = -rCells; dc <= rCells; dc++) {
      const row = row0 + dr;
      const col = col0 + dc;
      if (row < 0 || row >= g.height || col < 0 || col >= g.width) continue;
      const cell = { row, col, lat: g.lat_first + row * g.lat_step, lon: g.lon_first + col * g.lon_step };
      const v = valueOf(g, codes, cell);
      if (v === null) continue;
      const km = distanceKm(lat, lon, cell.lat, cell.lon);
      if (km <= radiusKm) inRadius.push(v);
      if (km <= searchKm && (!best || km < best.km)) best = { cell, value: v, km, bearing: bearingName(lat, lon, cell.lat, cell.lon) };
    }
  }
  inRadius.sort((a, b) => a - b);
  const region = inRadius.length
    ? { radiusKm, n: inRadius.length, median: inRadius[Math.floor((inRadius.length - 1) / 2)] * 0.5 + inRadius[Math.ceil((inRadius.length - 1) / 2)] * 0.5, min: inRadius[0], max: inRadius[inRadius.length - 1] }
    : null;
  return {
    here: c && here !== null ? { cell: c, value: here, km: 0, bearing: "here" } : null,
    nearest: c && here !== null ? null : best,
    region,
  };
}

/** The cell's square, for the map (centre ± half a step). */
export function cellPolygon(g: ValueGrid, cell: Cell): GeoJSON.Polygon {
  const hl = Math.abs(g.lat_step) / 2;
  const hw = Math.abs(g.lon_step) / 2;
  return { type: "Polygon", coordinates: [[[cell.lon - hw, cell.lat - hl], [cell.lon + hw, cell.lat - hl], [cell.lon + hw, cell.lat + hl], [cell.lon - hw, cell.lat + hl], [cell.lon - hw, cell.lat - hl]]] };
}

/** The lead to open with: today's valid day when issued, otherwise the latest issued. */
export function defaultLead(m: Manifest, todayIso: string, layers: (LayerArtifact | null)[]): number {
  const today = layers.findIndex((l) => l?.time.valid_date === todayIso);
  if (today >= 0) return today;
  const issued = layers.map((l, i) => (l ? i : -1)).filter((i) => i >= 0);
  return issued.length ? issued[issued.length - 1] : 0;
}
