import { distanceKm, type Gap } from "@/lib/grid";
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

export type ThreadTo = "cell" | "station" | "limit" | "pixel" | "obs";

export type ProvenanceInput = {
  target: { lat: number; lon: number };
  /** the model cell used: its polygon (exact edges) and centre; `own`: the point's own cell */
  cell?: { polygon: GeoJSON.Polygon; lat: number; lon: number; own: boolean } | null;
  stations?: { lat: number; lon: number; name: string; km: number; id?: string; aged?: boolean; focus?: boolean }[];
  /** the satellite pixel used (its centre), when the satellite is in focus */
  pixel?: { lat: number; lon: number } | null;
  /** other stations to draw without a thread (small) */
  others?: { lat: number; lon: number; name: string; id: string; aged: boolean; minor?: boolean }[];
  /** NOAA stations behind the conditions card (a buoy's measurements, a tide station's predictions) */
  obs?: { lat: number; lon: number; name: string; what: "buoy" | "tide" }[];
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
/**
 * The text anchor that puts a label on the side of `end` away from `from`: the label's edge named
 * by the anchor sits at the point, so "right" runs the words west of it. Used for the distance
 * labels at a thread's end, so they never cover the harbour's own name when the cell is close.
 */
export function anchorAwayFrom(from: [number, number], end: [number, number]): "left" | "right" | "top" | "bottom" {
  const dx = (from[0] - end[0]) * Math.cos((from[1] * Math.PI) / 180);
  const dy = from[1] - end[1];
  return Math.abs(dx) >= Math.abs(dy) ? (dx > 0 ? "right" : "left") : dy > 0 ? "top" : "bottom";
}

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
    if (!x.cell.own) {
      const km = distanceKm(x.target.lat, x.target.lon, x.cell.lat, x.cell.lon);
      thread("cell", [x.cell.lon, x.cell.lat], { km: +km.toFixed(1), label: `${km < 10 ? km.toFixed(1) : Math.round(km)} km to the nearest model cell`, anchor: anchorAwayFrom(t, [x.cell.lon, x.cell.lat]) });
    }
  }
  if (x.ring) f.push({ type: "Feature", properties: { kind: "ring" }, geometry: x.ring });
  for (const o of x.others ?? []) f.push({ type: "Feature", properties: { kind: "station", name: o.name, id: o.id, aged: o.aged, minor: o.minor ?? true }, geometry: { type: "Point", coordinates: [o.lon, o.lat] } });
  for (const s of x.stations ?? []) {
    f.push({ type: "Feature", properties: { kind: "station", name: s.name, id: s.id ?? s.name, far: s.km > STATION_FAR_KM, aged: !!s.aged, focus: !!s.focus }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } });
    thread("station", [s.lon, s.lat], { far: s.km > STATION_FAR_KM });
  }
  if (x.pixel) thread("pixel", [x.pixel.lon, x.pixel.lat]);
  for (const o of x.obs ?? []) {
    f.push({ type: "Feature", properties: { kind: "obs", name: o.name, what: o.what }, geometry: { type: "Point", coordinates: [o.lon, o.lat] } });
    thread("obs", [o.lon, o.lat]);
  }
  const lim = nearestLatLimit(x.official, x.target.lat);
  const dLat = lim ? Math.abs(lim.lat - x.target.lat) : Infinity;
  if (lim && dLat > 0.002 && dLat <= LIMIT_RULER_DEG) thread("limit", [x.target.lon, lim.lat], { label: lim.label });
  return { type: "FeatureCollection", features: f };
}

// ---------------------------------------------------------------- the harbour view (carto T3)

/** "6.6 km" below ten, "12 km" above. */
export const kmWords = (km: number) => `${km < 10 ? km.toFixed(1) : Math.round(km)} km`;

export const GAP_NOTE = "No C-HARM value here: nearshore cells not modelled";

/**
 * The nearshore gap at harbour zoom: when the map's centre lies in the C-HARM domain in a cell
 * without a value, a note at the centre says so and names the distance to the nearest cell with a
 * value, and a dashed thread runs from the centre to that cell's centre, carrying the distance.
 * Null when the centre's own cell has a value (nothing to explain) or the point is outside the
 * domain (the readout says "outside"). Features: "note" (Point, text), "thread" (LineString, label),
 * "end" (Point at the cell centre, label). The note is a statement about the model, never a value.
 */
export function gapFeatures(target: { lat: number; lon: number }, gap: Gap): GeoJSON.FeatureCollection | null {
  if (!gap.inDomain || gap.here) return null;
  const t: [number, number] = [+target.lon.toFixed(5), +target.lat.toFixed(5)];
  const f: GeoJSON.Feature[] = [];
  if (gap.nearest && gap.nearestKm != null) {
    const km = kmWords(gap.nearestKm);
    const end: [number, number] = [+gap.nearest.lon.toFixed(5), +gap.nearest.lat.toFixed(5)];
    f.push({ type: "Feature", properties: { kind: "note", text: `${GAP_NOTE} · nearest cell ${km}`, km: +gap.nearestKm.toFixed(1) }, geometry: { type: "Point", coordinates: t } });
    f.push({ type: "Feature", properties: { kind: "thread", label: km }, geometry: { type: "LineString", coordinates: [t, end] } });
    // the end label sits on the side of the cell away from the harbour (Crescent City: 2.4 km)
    f.push({ type: "Feature", properties: { kind: "end", anchor: anchorAwayFrom(t, end), label: `${km} to the nearest modelled cell: ${Math.round(gap.nearest.value * 100)} %` }, geometry: { type: "Point", coordinates: end } });
  } else {
    f.push({ type: "Feature", properties: { kind: "note", text: `${GAP_NOTE} · no modelled cell within 35 km` }, geometry: { type: "Point", coordinates: t } });
  }
  return { type: "FeatureCollection", features: f };
}

/** The HAB measurements whose sample dates say when a shore station was last sampled (lib/hab MEASURED_VARIABLES). */
export const HAB_SAMPLE_VARIABLES = ["pDA", "pn_seriata", "pn_delicatissima"] as const;
/** A station sampled within this many days is drawn filled; older, as a ring. */
export const STATION_RECENT_DAYS = 30;

type StationPoint = { station_id: string; name: string; lat: number; lon: number; status?: string; latest: { variable: string; date: string }[] };

/** The newest HAB sample date (YYYY-MM-DD) across a station's latest values, or null. */
export function lastHabSample(st: Pick<StationPoint, "latest">): string | null {
  let best: string | null = null;
  for (const l of st.latest) {
    if (!(HAB_SAMPLE_VARIABLES as readonly string[]).includes(l.variable) || !l.date) continue;
    const d = l.date.slice(0, 10);
    if (!best || d > best) best = d;
  }
  return best;
}

/**
 * The HAB shore stations as map points for harbour zoom (carto T3): each with its name, the date
 * of its newest HAB sample and whether that was within the last 30 days (`recent`: drawn filled,
 * otherwise a ring), and a label "<name> · last HAB sample <Mon D>" (with the year when it is not
 * this year). Stations whose last update failed are left out, as the inspector leaves them out.
 * `now` null: no station is called recent.
 */
export function stationFeatures(stations: StationPoint[] | null | undefined, now: Date | null, today: string | null = null): GeoJSON.FeatureCollection<GeoJSON.Point> {
  const ref = now ? now.getTime() : null;
  const year = today?.slice(0, 4) ?? null;
  const features = (stations ?? [])
    .filter((s) => s.status !== "failed" && Number.isFinite(s.lat) && Number.isFinite(s.lon))
    .map((s): GeoJSON.Feature<GeoJSON.Point> => {
      const last = lastHabSample(s);
      const age = last && ref != null ? (ref - Date.parse(`${last}T12:00:00Z`)) / 86_400_000 : null;
      const recent = age != null && age <= STATION_RECENT_DAYS;
      const when = last ? sampleDateWords(last, year) : null;
      return {
        type: "Feature",
        properties: { kind: "station", id: s.station_id, name: s.name, last, recent, label: `${s.name} · ${when ? `last HAB sample ${when}` : "no HAB sample published"}` },
        geometry: { type: "Point", coordinates: [s.lon, s.lat] },
      };
    });
  return { type: "FeatureCollection", features };
}

/** "Oct 7" this year, "Aug 27, 2025" another year (no time-zone shifting). */
function sampleDateWords(date: string, year: string | null): string {
  const d = new Date(`${date}T12:00:00Z`);
  const sameYear = year == null || date.slice(0, 4) === year;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", ...(sameYear ? {} : { year: "numeric" }), timeZone: "UTC" });
}
