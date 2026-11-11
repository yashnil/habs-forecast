/**
 * Reading the water at a tapped point, DOM-free. Truth rules: the value is the cell under the point, never a
 * neighbour's and never interpolated (T19); no value is said as such ("No value is not a low value", T20); a
 * latest view's pixel carries its own date (T18); a transect breaks at every gap; the nearest current is the
 * nearest measured cell with its distance.
 */
import { cellCentre, cellOf, decodeValue, type FieldGrid } from "@/nx/chart/layers/grid";
import type { ExploreFrame, ExploreStation, ProductId } from "./model";

const KM_PER_DEG = 111.32;
export function distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const dy = (lat2 - lat1) * KM_PER_DEG;
  const dx = (lon2 - lon1) * KM_PER_DEG * Math.cos(((lat1 + lat2) / 2) * (Math.PI / 180));
  return Math.hypot(dx, dy);
}

export type PointRead = { value: number | null; row: number; col: number; lat: number; lon: number } | null;

/** The cell under a point and its value (null: no value there), or null outside the grid. */
export function readPoint(g: FieldGrid, codes: Uint16Array, lat: number, lon: number): PointRead {
  const c = cellOf(g, lat, lon);
  if (!c) return null;
  const { lat: clat, lon: clon } = cellCentre(g, c.row, c.col);
  return { value: decodeValue(g, codes[c.row * g.width + c.col]), row: c.row, col: c.col, lat: clat, lon: clon };
}

/** A latest view's pixel date: the reference date minus the pixel's age in days. */
export function pixelDate(referenceDate: string, ageDays: number): string {
  return new Date(Date.parse(`${referenceDate}T12:00:00Z`) - ageDays * 86_400_000).toISOString().slice(0, 10);
}

export type TransectPoint = { lon: number; km: number; value: number | null };

/**
 * The values along the tapped point's row (its latitude), east-west, `halfKm` either side: one point per cell
 * centre, `km` signed (negative west of the point). Cells without a value are kept as null, so the line breaks.
 */
export function transect(g: FieldGrid, codes: Uint16Array, lat: number, lon: number, halfKm = 40): TransectPoint[] {
  const c = cellOf(g, lat, lon);
  if (!c) return [];
  const kmPerDegLon = KM_PER_DEG * Math.cos((lat * Math.PI) / 180);
  const span = Math.ceil(halfKm / (Math.abs(g.lon_step) * kmPerDegLon));
  const out: TransectPoint[] = [];
  for (let col = Math.max(0, c.col - span); col <= Math.min(g.width - 1, c.col + span); col++) {
    const clon = g.lon_first + col * g.lon_step;
    out.push({ lon: clon, km: (clon - lon) * kmPerDegLon, value: decodeValue(g, codes[c.row * g.width + col]) });
  }
  return out.sort((a, b) => a.km - b.km);
}

/** Runs of consecutive values (a transect's drawable pieces): a gap of one cell breaks the line. */
export function runs(points: TransectPoint[]): TransectPoint[][] {
  const out: TransectPoint[][] = [];
  let run: TransectPoint[] = [];
  for (const p of points) {
    if (p.value === null) {
      if (run.length) out.push(run);
      run = [];
    } else run.push(p);
  }
  if (run.length) out.push(run);
  return out;
}

export type NearestCurrent = { speed: number; dirTo: number; km: number } | null;

/** The nearest measured current within `maxKm` (the cell's own u/v, m/s; direction the water moves toward, degrees). */
export function nearestCurrent(g: FieldGrid, u: Float32Array, v: Float32Array, lat: number, lon: number, maxKm = 6): NearestCurrent {
  const c = cellOf(g, lat, lon);
  const r0 = c?.row ?? Math.round((lat - g.lat_first) / g.lat_step);
  const c0 = c?.col ?? Math.round((lon - g.lon_first) / g.lon_step);
  let best: NearestCurrent = null;
  const reach = Math.ceil(maxKm / (Math.abs(g.lat_step) * KM_PER_DEG)) + 1;
  for (let r = r0 - reach; r <= r0 + reach; r++)
    for (let k = c0 - reach; k <= c0 + reach; k++) {
      if (r < 0 || k < 0 || r >= g.height || k >= g.width) continue;
      const i = r * g.width + k;
      if (!Number.isFinite(u[i]) || !Number.isFinite(v[i])) continue;
      const p = cellCentre(g, r, k);
      const km = distanceKm(lat, lon, p.lat, p.lon);
      if (km > maxKm || (best && km >= best.km)) continue;
      best = { speed: Math.hypot(u[i], v[i]), dirTo: ((Math.atan2(u[i], v[i]) * 180) / Math.PI + 360) % 360, km };
    }
  return best;
}

const COMPASS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"];
export const compassWord = (deg: number) => COMPASS[Math.round(((deg % 360) + 360) % 360 / 45) % 8];
export const KNOTS = 1.943844;

/** The nearest monitoring station (CalHABMAP shore samples) and how far it is. */
export function nearestStation(stations: ExploreStation[], lat: number, lon: number): (ExploreStation & { km: number }) | null {
  let best: (ExploreStation & { km: number }) | null = null;
  for (const s of stations) {
    const km = distanceKm(lat, lon, s.lat, s.lon);
    if (!best || km < best.km) best = { ...s, km };
  }
  return best;
}

/**
 * The Research URL contract (agreed with the Research feature):
 *   /research?station=<CalHABMAP station_id>&from=<YYYY-MM-DD>&to=<YYYY-MM-DD>&var=<variable id>[&place=<port key>]
 * `var` is an observations variable id (chl_extracted, temp, pDA, pn_seriata, …): the measurement nearest in
 * kind to the Explore layer. The window ends on the frame's (or pixel's) date and starts 60 days before it.
 */
export const RESEARCH_VAR: Record<ProductId, string> = { chl: "chl_extracted", outlook: "chl_extracted", sst: "temp", charm: "pDA" };
export const RESEARCH_WINDOW_DAYS = 60;

export function researchHref(o: { station: string; to: string; product: ProductId; place?: string | null }): string {
  const from = new Date(Date.parse(`${o.to}T12:00:00Z`) - RESEARCH_WINDOW_DAYS * 86_400_000).toISOString().slice(0, 10);
  const q = new URLSearchParams({ station: o.station, from, to: o.to, var: RESEARCH_VAR[o.product] });
  if (o.place) q.set("place", o.place);
  return `/research?${q.toString()}`;
}

/** The date a reading at a point belongs to: the pixel's own date in a latest view, else the frame's end. */
export function readingDate(frame: Pick<ExploreFrame, "end">, pixel: string | null): string {
  return pixel ?? frame.end;
}
