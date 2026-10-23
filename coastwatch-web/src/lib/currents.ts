import type { LayerArtifact, Manifest, ValueGrid } from "@/generated/schema";
import { distanceKm, loadGrid } from "@/lib/grid";
import { artifactUrl } from "@/lib/layers";

/**
 * Observed surface currents (HF radar): helpers shared by the map, the dock and the
 * inspector. Values are the published u/v grids decoded as-is: nothing is interpolated
 * between cells or hours, and cells without a value stay empty.
 */
export const CURRENTS_GROUP = "currents";
export const KNOTS_PER_MS = 1.943844;
/** Arrow thinning levels (cells between arrows), coarse to fine: statewide, only every 16th cell (about 32 km) has an arrow. */
export const ARROW_LEVELS = [16, 8, 4, 2, 1] as const;

export function currentsHourly(m: Manifest): LayerArtifact[] {
  return m.layers
    .filter((l) => l.group_id === CURRENTS_GROUP && l.layer_id.startsWith("hfr2km_currents_2") && l.vectors)
    .sort((a, b) => (a.time.valid_time ?? "").localeCompare(b.time.valid_time ?? ""));
}

/**
 * An hour whose radar coverage collapsed (a feed still filling in, or a site down) is published like any
 * other; it stays selectable but is not the default view. "Usable": the hour's published coverage is at
 * least half the median of the published hours (no manifest field is added: the pipeline's own per-layer
 * coverage). An hour without a coverage figure counts as usable.
 */
export const USABLE_COVERAGE_SHARE = 0.5;

/** The observed radar cells of an hour, from its published coverage (null when not published). */
export function observedCells(l: LayerArtifact): number | null {
  const c = l.coverage;
  if (!c || c.domain_reference_cells == null) return null;
  return Math.round(c.domain_observed_fraction * c.domain_reference_cells);
}

/** Whether each hourly layer has usable coverage (lib/currents rule above), by layer id. */
export function usableHours(m: Manifest): Map<string, boolean> {
  const hours = currentsHourly(m);
  const fr = hours.map((l) => l.coverage?.domain_observed_fraction).filter((f): f is number => f != null).sort((a, b) => a - b);
  const median = fr.length ? (fr.length % 2 ? fr[(fr.length - 1) / 2] : (fr[fr.length / 2 - 1] + fr[fr.length / 2]) / 2) : null;
  const out = new Map<string, boolean>();
  for (const l of hours) {
    const f = l.coverage?.domain_observed_fraction;
    out.set(l.layer_id, f == null || median == null || median === 0 || f >= median * USABLE_COVERAGE_SHARE);
  }
  return out;
}

const median = (xs: number[]): number | null => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

/** A region's published radar coverage in an hour (coverage.regions[]), or null when the hour does not publish one. */
export function regionFraction(l: LayerArtifact, regionId: string): number | null {
  const r = l.coverage?.regions?.find((x) => x.region_id === regionId);
  return r ? r.observed_fraction : null;
}

/**
 * Whether each hourly layer saw a region, by layer id, from the published per-region coverage: an hour
 * is usable there when it observed some of the region and at least half the region's median over the
 * published hours (the same share as the domain rule). Null when no hour publishes a figure for the
 * region (then the domain rule applies).
 */
export function usableHoursIn(m: Manifest, regionId: string): Map<string, boolean> | null {
  const hours = currentsHourly(m);
  const fr = hours.map((l) => regionFraction(l, regionId)).filter((f): f is number => f != null);
  if (!fr.length) return null;
  const med = median(fr);
  const out = new Map<string, boolean>();
  for (const l of hours) {
    const f = regionFraction(l, regionId);
    out.set(l.layer_id, f == null ? false : f > 0 && (med == null || med === 0 || f >= med * USABLE_COVERAGE_SHARE));
  }
  return out;
}

/**
 * The hour the map opens on (no hour chosen): the newest hour with usable coverage of the region in view
 * (or selected), from the published `coverage.regions[]`; without a region, or for a region no hour
 * publishes a figure for, the newest hour with usable domain coverage; else the newest hour.
 * `newestMissed`: the newest hour was passed over because it saw little or none of that region (the
 * dock says so, and which hour is shown).
 */
export type HourChoice = { layer: LayerArtifact | null; newest: LayerArtifact | null; regionId: string | null; regionLabel: string | null; newestMissed: boolean; newestFraction: number | null };
export function chooseHour(m: Manifest, regionId: string | null = null): HourChoice {
  const hours = currentsHourly(m);
  const newest = hours[hours.length - 1] ?? null;
  const inRegion = regionId ? usableHoursIn(m, regionId) : null;
  const ok = inRegion ?? usableHours(m);
  const layer = [...hours].reverse().find((l) => ok.get(l.layer_id)) ?? newest;
  const label = inRegion && regionId ? (hours.flatMap((l) => l.coverage?.regions ?? []).find((r) => r.region_id === regionId)?.label ?? null) : null;
  return {
    layer,
    newest,
    regionId: inRegion ? regionId : null,
    regionLabel: label,
    newestMissed: !!inRegion && !!layer && !!newest && layer.layer_id !== newest.layer_id,
    newestFraction: inRegion && newest && regionId ? regionFraction(newest, regionId) : null,
  };
}

/** The hour the map opens on: see `chooseHour` (the region in view, else the whole domain). */
export function defaultHour(m: Manifest, regionId: string | null = null): LayerArtifact | null {
  return chooseHour(m, regionId).layer;
}

export function currentsMean(m: Manifest): LayerArtifact | null {
  return m.layers.find((l) => l.layer_id === "hfr2km_currents_mean24h" && l.vectors) ?? null;
}

/** "20261008T12Z" from a valid time, as used in layer ids and URLs. */
export function hourStamp(t: string): string {
  return `${t.slice(0, 4)}${t.slice(5, 7)}${t.slice(8, 10)}T${t.slice(11, 13)}Z`;
}

/** Direction the water moves toward, degrees clockwise from north. */
export function directionDeg(u: number, v: number): number {
  return ((Math.atan2(u, v) * 180) / Math.PI + 360) % 360;
}

const POINTS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
export function compass(deg: number): string {
  return POINTS[Math.round(deg / 22.5) % 16];
}

export type CurrentField = { g: ValueGrid; u: Float32Array; v: Float32Array };

function decode(g: ValueGrid, codes: Uint16Array): Float32Array {
  const nd = g.nodata ?? 65535;
  const out = new Float32Array(codes.length);
  for (let i = 0; i < codes.length; i++) out[i] = codes[i] === nd ? NaN : codes[i] * g.scale_factor + g.add_offset;
  return out;
}

export async function loadField(baseUrl: string, l: LayerArtifact): Promise<CurrentField> {
  const vf = l.vectors!;
  const [cu, cv] = await Promise.all([loadGrid(artifactUrl(baseUrl, vf.u_grid.url)), loadGrid(artifactUrl(baseUrl, vf.v_grid.url))]);
  return { g: vf.u_grid, u: decode(vf.u_grid, cu), v: decode(vf.v_grid, cv) };
}

/**
 * Decoded hours, newest-used last (a few only: each hour decodes to about 1.8 MB). Playback asks for the
 * next real hour while the current one shows, so a step cuts to data that has already arrived.
 */
const FIELD_CACHE = 4;
const fields = new Map<string, Promise<CurrentField>>();
export function loadFieldCached(baseUrl: string, l: LayerArtifact): Promise<CurrentField> {
  const key = `${baseUrl}|${l.layer_id}|${l.vectors?.u_grid.url ?? ""}`;
  let p = fields.get(key);
  if (p) fields.delete(key);
  else {
    p = loadField(baseUrl, l);
    p.catch(() => fields.delete(key));
  }
  fields.set(key, p);
  while (fields.size > FIELD_CACHE) fields.delete(fields.keys().next().value!);
  return p;
}

/** One point per valid cell centre with speed, direction and thinning level. */
export function fieldFeatures(f: CurrentField): GeoJSON.FeatureCollection<GeoJSON.Point> {
  const { g } = f;
  const features: GeoJSON.Feature<GeoJSON.Point>[] = [];
  for (let r = 0; r < g.height; r++) {
    for (let c = 0; c < g.width; c++) {
      const k = r * g.width + c;
      const u = f.u[k];
      const v = f.v[k];
      if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
      const level = ARROW_LEVELS.find((n) => r % n === 0 && c % n === 0) ?? 1;
      features.push({
        type: "Feature",
        geometry: { type: "Point", coordinates: [g.lon_first + c * g.lon_step, g.lat_first + r * g.lat_step] },
        properties: { speed: Math.hypot(u, v), dir: directionDeg(u, v), level },
      });
    }
  }
  return { type: "FeatureCollection", features };
}

export type CurrentSample = { u: number; v: number; speed: number; dir: number; lat: number; lon: number; distanceKm: number };

/** The cell containing the point, or the nearest valid cell within `search` cells. */
export function sampleField(f: CurrentField, lat: number, lon: number, search = 1): CurrentSample | null {
  const { g } = f;
  const r0 = Math.round((lat - g.lat_first) / g.lat_step);
  const c0 = Math.round((lon - g.lon_first) / g.lon_step);
  let best: CurrentSample | null = null;
  for (let dr = -search; dr <= search; dr++) {
    for (let dc = -search; dc <= search; dc++) {
      const r = r0 + dr;
      const c = c0 + dc;
      if (r < 0 || c < 0 || r >= g.height || c >= g.width) continue;
      const k = r * g.width + c;
      const u = f.u[k];
      const v = f.v[k];
      if (!Number.isFinite(u) || !Number.isFinite(v)) continue;
      const cl = g.lat_first + r * g.lat_step;
      const cn = g.lon_first + c * g.lon_step;
      const d = dr === 0 && dc === 0 ? 0 : distanceKm(lat, lon, cl, cn);
      if (!best || d < best.distanceKm) best = { u, v, speed: Math.hypot(u, v), dir: directionDeg(u, v), lat: cl, lon: cn, distanceKm: d };
    }
  }
  return best;
}

/** Hours between a valid time and now, rounded down. */
export function hoursAgo(validTime: string, now: Date): number {
  return Math.max(0, Math.floor((now.getTime() - Date.parse(validTime)) / 3_600_000));
}

/**
 * HF-radar hours are published about 4-15 h after observation, so an hour older than this means
 * the feed stopped (the pipeline's health check alerts at the same age, HFR_MAX_AGE_HOURS). Every
 * view that shows an hour says "Stale" past it; the hour's own time is always shown.
 */
export const HFR_STALE_HOURS = 18;
export const currentsStale = (validTime: string | null | undefined, now: Date | null | undefined): boolean => !!validTime && !!now && hoursAgo(validTime, now) > HFR_STALE_HOURS;
