/**
 * Provenance threads (signature interaction, PR #43 04-interactions §1): when a reader selects a
 * water point, thin lines run from it to where each statement about it comes from:
 * - the model cell that holds the value (its exact edges), or the nearest cell with a value when
 *   the point's own cell has none;
 * - the shore station that was sampled;
 * - the nearest official latitude limit, drawn straight north or south, as a ruler.
 *
 * Guardrails: a thread is a connector, never a value. It is never coloured by data, it always ends
 * at the source's real location, and the point never inherits the source's value (the text says
 * "it describes that cell, not this one"). Every fact a thread shows is also in the panel's words,
 * so threads are decorative for assistive technology.
 */

export type ThreadTo = "cell" | "station" | "limit" | "pixel";

export type ProvenanceInput = {
  target: { lat: number; lon: number };
  /** the model cell used: its polygon (exact edges) and centre; `own`: the point's own cell */
  cell?: { polygon: GeoJSON.Polygon; lat: number; lon: number; own: boolean } | null;
  stations?: { lat: number; lon: number; name: string; km: number; id?: string; aged?: boolean; focus?: boolean }[];
  /** the satellite pixel used (its centre), when the satellite is in focus */
  pixel?: { lat: number; lon: number } | null;
  /** other stations to draw without a thread (small) */
  others?: { lat: number; lon: number; name: string; id: string; aged: boolean; minor?: boolean }[];
  /** a neighbourhood ring (the model's cells within it), as a polygon */
  ring?: GeoJSON.Polygon | null;
  /** official geometry (lat_limit lines are used for the ruler) */
  official?: GeoJSON.FeatureCollection | null;
};

/** Beyond this, a station still gets a thread, drawn fainter: too far to describe these waters. */
export const STATION_FAR_KM = 25;
/** The limit ruler is drawn only when the limit is this close (about 30 km); farther, the words say it. */
export const LIMIT_RULER_DEG = 0.27;

/** The official latitude limit nearest the point (lines of kind "lat_limit"), or null. */
export function nearestLatLimit(official: GeoJSON.FeatureCollection | null | undefined, lat: number): { lat: number; label: string } | null {
  let best: { lat: number; label: string; d: number } | null = null;
  for (const f of official?.features ?? []) {
    const p = (f.properties ?? {}) as { kind?: string; label?: string };
    if (p.kind !== "lat_limit" || f.geometry.type !== "LineString") continue;
    const coords = f.geometry.coordinates as number[][];
    const flat = coords[0]?.[1];
    if (flat == null) continue;
    const d = Math.abs(flat - lat);
    if (!best || d < best.d) best = { lat: flat, label: p.label ?? "", d };
  }
  return best ? { lat: best.lat, label: best.label } : null;
}

/**
 * The overlay features: the target, the cell outline, one thread per source and a dot at each
 * thread's end, in the order the threads draw (model, then measured, then official).
 */
export function provenanceFeatures(x: ProvenanceInput): GeoJSON.FeatureCollection {
  const t: [number, number] = [x.target.lon, x.target.lat];
  const f: GeoJSON.Feature[] = [{ type: "Feature", properties: { kind: "target" }, geometry: { type: "Point", coordinates: t } }];
  const thread = (to: ThreadTo, end: [number, number], props: Record<string, unknown> = {}) => {
    f.push({ type: "Feature", properties: { kind: "thread", to, ...props }, geometry: { type: "LineString", coordinates: [t, end] } });
    f.push({ type: "Feature", properties: { kind: "end", to, ...props }, geometry: { type: "Point", coordinates: end } });
  };
  if (x.cell) {
    f.push({ type: "Feature", properties: { kind: "cell", own: x.cell.own }, geometry: x.cell.polygon });
    // the point's own cell needs no thread: the outline around the point is the connection
    if (!x.cell.own) thread("cell", [x.cell.lon, x.cell.lat]);
  }
  if (x.ring) f.push({ type: "Feature", properties: { kind: "ring" }, geometry: x.ring });
  for (const o of x.others ?? []) f.push({ type: "Feature", properties: { kind: "station", name: o.name, id: o.id, aged: o.aged, minor: o.minor ?? true }, geometry: { type: "Point", coordinates: [o.lon, o.lat] } });
  for (const s of x.stations ?? []) {
    f.push({ type: "Feature", properties: { kind: "station", name: s.name, id: s.id ?? s.name, far: s.km > STATION_FAR_KM, aged: !!s.aged, focus: !!s.focus }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } });
    thread("station", [s.lon, s.lat], { far: s.km > STATION_FAR_KM });
  }
  if (x.pixel) thread("pixel", [x.pixel.lon, x.pixel.lat]);
  const lim = nearestLatLimit(x.official, x.target.lat);
  const dLat = lim ? Math.abs(lim.lat - x.target.lat) : Infinity;
  if (lim && dLat > 0.002 && dLat <= LIMIT_RULER_DEG) thread("limit", [x.target.lon, lim.lat], { label: lim.label });
  return { type: "FeatureCollection", features: f };
}
