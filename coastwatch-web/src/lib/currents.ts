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
