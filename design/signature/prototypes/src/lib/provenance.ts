import { cellAt, disputedBy, type Grid, nearestValued } from "../data/store";
import { nearestPoint, readPoint } from "../data/points";
import type { Layer, Manifest, Observations, Official, OfficialRecord, Port, Station } from "../data/types";
import { bearingWord, km, type LngLat } from "./geo";
import { daysBetween, TODAY } from "./time";

/**
 * What CoastWatch knows about one point on the water, and where each piece comes from. Every item
 * keeps its own place and date; nothing is moved onto the selected point. The map draws a thread
 * from the point to each item's real location (provenance threads).
 */
export type GapReason =
  | { kind: "outside-domain"; detail: string }
  | { kind: "nearshore-mask"; detail: string; nearest?: { km: number; center: LngLat; value: number } }
  | { kind: "cloud"; detail: string; nearest?: { km: number; center: LngLat; value: number; date: string } }
  | { kind: "no-sampling"; detail: string }
  | { kind: "too-old"; detail: string }
  | { kind: "source-unavailable"; detail: string }
  | { kind: "land"; detail: string };

export interface Reading {
  point: LngLat;
  place: { label: string; port?: Port; portKm?: number };
  model:
    | { status: "value"; value: number; cell: [number, number, number, number]; center: LngLat; layer: Layer }
    | { status: "gap"; reason: GapReason; cell?: [number, number, number, number]; layer?: Layer }
    | { status: "loading" };
  satellite:
    | { status: "value"; value: number; date: string; ageDays: number; pixel: [number, number, number, number]; center: LngLat; layer: Layer }
    | { status: "gap"; reason: GapReason; pixel?: [number, number, number, number]; layer?: Layer }
    | { status: "loading" | "off" };
  /** lastDate/ageDays: the last visit (any quantity); toxinDate/lastValue: the last toxin measurement */
  station: { station: Station; km: number; bearing: string; lastDate: string | null; ageDays: number | null; toxinDate: string | null; lastValue: number | null; tooFar: boolean } | null;
  notices: NoticeGeo[];
  disputed: { record: OfficialRecord; by: NonNullable<ReturnType<typeof disputedBy>> }[];
}
export interface NoticeGeo {
  record: OfficialRecord;
  kind: "lat-band" | "area" | "statewide";
  /** mirrors the redesign's applicability tiers (lib/applicability.ts) for a point on the water */
  tier: "geographic" | "unresolved" | "outside";
  /** for latitude bands: is the point between the official limits, and how far is the nearer limit */
  between?: boolean;
  nearestLimit?: { lat: number; which: "north" | "south"; km: number };
}

export const STATION_FAR_KM = 25;

export function nearestPort(ports: Port[], p: LngLat) {
  let best: Port | undefined; let d = Infinity;
  for (const port of ports) { const k = km(p, [port.lon, port.lat]); if (k < d) { d = k; best = port; } }
  return best ? { port: best, km: d } : null;
}

export function modelReading(grid: Grid | undefined, layer: Layer | undefined, p: LngLat, manifest?: Manifest, failed?: boolean): Reading["model"] {
  if (failed) return { status: "gap", reason: { kind: "source-unavailable", detail: "The forecast could not be loaded just now. Nothing is shown in its place." }, layer };
  if (!grid || !layer) return { status: "loading" };
  const c = cellAt(grid, p[0], p[1]);
  if (!c) return { status: "gap", reason: { kind: "outside-domain", detail: "Outside the area NOAA's C-HARM model covers." }, layer };
  if (!Number.isNaN(c.value)) return { status: "value", value: c.value, cell: c.bounds, center: c.center, layer };
  const n = nearestValued(grid, p[0], p[1], 6);
  void manifest;
  return {
    status: "gap",
    cell: c.bounds,
    layer,
    reason: {
      kind: "nearshore-mask",
      detail: "C-HARM gives no value for this 3 km cell. The model leaves out many cells within about 3–6 km of shore, and bays.",
      nearest: n ? { km: km(p, n.center), center: n.center, value: n.value } : undefined,
    },
  };
}

export async function satelliteReading(layer: Layer | undefined, p: LngLat): Promise<Reading["satellite"]> {
  if (!layer?.grid) return { status: "off" };
  const comp = (layer as unknown as { composite?: { reference_date: string; window_days: number; age_grid: Layer["grid"] } }).composite;
  const r = await readPoint(layer.grid, p[0], p[1]);
  if (!r.inside) return { status: "gap", reason: { kind: "outside-domain", detail: "Outside the area the satellite product covers." }, layer };
  if (r.value != null && comp?.age_grid) {
    const a = await readPoint(comp.age_grid, p[0], p[1]);
    const age = a.value ?? 0;
    const date = new Date(Date.parse(comp.reference_date + "T12:00:00Z") - age * 86400000).toISOString().slice(0, 10);
    return { status: "value", value: r.value, date, ageDays: daysBetween(date, TODAY), pixel: r.bounds, center: r.center, layer };
  }
  const n = await nearestPoint(layer.grid, p[0], p[1], 40);
  let nearest: { km: number; center: LngLat; value: number; date: string } | undefined;
  if (n && n.value != null && comp?.age_grid) {
    const a = await readPoint(comp.age_grid, n.center[0], n.center[1]);
    const date = new Date(Date.parse(comp.reference_date + "T12:00:00Z") - (a.value ?? 0) * 86400000).toISOString().slice(0, 10);
    nearest = { km: km(p, n.center), center: n.center, value: n.value, date };
  }
  return {
    status: "gap",
    pixel: r.bounds,
    layer,
    reason: { kind: "cloud", detail: `No clear satellite view of this 300 m pixel in the last ${comp?.window_days ?? 7} days (cloud, fog or no overpass).`, nearest },
  };
}

export function stationReading(obs: Observations | undefined, p: LngLat): Reading["station"] {
  if (!obs) return null;
  let best: Station | undefined; let d = Infinity;
  for (const s of obs.stations) { const k = km(p, [s.lon, s.lat]); if (k < d) { d = k; best = s; } }
  if (!best) return null;
  const sum = best.summaries.find((x) => x.variable === "pDA");
  const lastDate = best.sample_times.at(-1)?.slice(0, 10) ?? null;
  return {
    station: best, km: d, bearing: bearingWord(p, [best.lon, best.lat]),
    lastDate, ageDays: lastDate ? daysBetween(lastDate, TODAY) : null,
    toxinDate: sum?.last_date ?? null, lastValue: sum?.last_value ?? null,
    tooFar: d > STATION_FAR_KM,
  };
}

/** Geometry relations only. Whether a notice *applies* is decided elsewhere from the official
 *  geography and the reader's species (redesign: lib/applicability.ts); proximity never implies it. */
export function noticeGeometry(off: Official | undefined, p: LngLat): Pick<Reading, "notices" | "disputed"> {
  if (!off) return { notices: [], disputed: [] };
  const notices: NoticeGeo[] = [];
  const disputed: Reading["disputed"] = [];
  for (const r of off.registry.records) {
    if (r.status !== "active") continue;
    const by = disputedBy(off, r.id);
    if (by) { const near = areaDistanceKm(off, r.id, p); if (near != null && near < 30) disputed.push({ record: r, by }); continue; }
    if (r.area.type === "lat_band" && r.area.lat_north != null && r.area.lat_south != null) {
      const between = p[1] <= r.area.lat_north && p[1] >= r.area.lat_south;
      const dn = km(p, [p[0], r.area.lat_north]), ds = km(p, [p[0], r.area.lat_south]);
      notices.push({ record: r, kind: "lat-band", tier: between ? "geographic" : "outside", between, nearestLimit: dn < ds ? { lat: r.area.lat_north, which: "north", km: dn } : { lat: r.area.lat_south, which: "south", km: ds } });
    } else if (r.area.type === "statewide") notices.push({ record: r, kind: "statewide", tier: "geographic" });
    else {
      // county- and area-defined notices are drawn on land and state no offshore reach: on the
      // water they are "may apply", never "applies", and only listed when the area is nearby
      const near = areaDistanceKm(off, r.id, p);
      if (near != null && near < 30) notices.push({ record: r, kind: "area", tier: "unresolved", nearestLimit: undefined });
    }
  }
  return { notices, disputed };
}

type Ring = [number, number][];
function areaDistanceKm(off: Official, id: string, p: LngLat): number | null {
  let best: number | null = null;
  const kx = 111.32 * Math.cos((p[1] * Math.PI) / 180), ky = 111.32;
  for (const f of off.geometry.features) {
    const props = f.properties as { record_ids?: string[] } | null;
    if (!props?.record_ids?.includes(id)) continue;
    const g = f.geometry as { type: string; coordinates: unknown };
    const polys = g.type === "Polygon" ? [g.coordinates as Ring[]] : g.type === "MultiPolygon" ? (g.coordinates as Ring[][]) : [];
    for (const poly of polys) for (const ring of poly) for (let i = 1; i < ring.length; i++) {
      const ax = (ring[i - 1][0] - p[0]) * kx, ay = (ring[i - 1][1] - p[1]) * ky;
      const bx = (ring[i][0] - p[0]) * kx, by = (ring[i][1] - p[1]) * ky;
      const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
      const t = l2 > 0 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / l2)) : 0;
      const d = Math.hypot(ax + t * dx, ay + t * dy);
      best = best == null ? d : Math.min(best, d);
    }
  }
  return best;
}
