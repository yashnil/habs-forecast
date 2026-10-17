import type { LayerArtifact, Manifest, ValueGrid } from "@/generated/schema";
import { cellAt, chunkedBlock, decodeCode, type Cell } from "@/lib/grid";
import { currentsHourly, currentsMean, directionDeg, loadField, type CurrentField } from "@/lib/currents";
import { multiSensorMembers, multiSensorPick, observedDate } from "@/lib/multisensor";
import { satelliteDays, satelliteLatest, type SatProduct } from "@/lib/layers";
import { transectWith, type TransectCell } from "./transect";

/**
 * Observation grids in Research (satellite chlorophyll and HF-radar currents): every value is one
 * published pixel or radar cell read exactly, with that pixel's own observation date (a latest-clear
 * view dates each pixel from its age grid) or the radar hour. A pixel without a value is "no
 * observation" and stays empty: nothing is interpolated, filled or carried across a gap, and no value
 * is ever taken from a neighbouring pixel.
 */

export type Lattice = Pick<ValueGrid, "lat_first" | "lat_step" | "lon_first" | "lon_step" | "width" | "height">;
/** west, south, east, north */
export type BBox = [number, number, number, number];
/** Raw codes of a grid by row and column (nodata where nothing was published). */
export type Codes = (row: number, col: number) => number;

export type ObsPixel = {
  row: number;
  col: number;
  lat: number;
  lon: number;
  /** chlorophyll-a (mg/m³) or current speed (m/s); null = no observation */
  value: number | null;
  /** the pixel's own observation date (satellite), or the radar hour (ISO); null without an observation */
  date: string | null;
  /** multi-sensor display: 1 = the primary member (Sentinel-3), 2 = the secondary (VIIRS); null otherwise */
  sensor: 1 | 2 | null;
  /** the contributing sensor's own native cell (multi-sensor: VIIRS values are shown on the 300 m lattice) */
  native: { row: number; col: number } | null;
  u?: number;
  v?: number;
  /** direction the water moves toward, degrees clockwise from north */
  dir?: number;
};

export type Sampler = {
  kind: "satellite" | "radar";
  /** the layer chosen (multi-sensor: the display layer, which has no grid of its own) */
  layer: LayerArtifact;
  /** the lattice whose cells are listed (multi-sensor: the primary member's 300 m lattice) */
  g: Lattice;
  /** the layers whose values appear, primary first */
  sources: LayerArtifact[];
  at(row: number, col: number): ObsPixel;
  /** radar: the decoded field (for the map's arrows) */
  field?: CurrentField;
};

const NODATA = 65535;

/** Indices k (0..n-1) whose lattice coordinate first + k·step lies in [lo, hi]. */
export function rangeIn(first: number, step: number, n: number, lo: number, hi: number): [number, number] {
  const a = (lo - first) / step;
  const b = (hi - first) / step;
  const k0 = Math.max(0, Math.ceil(Math.min(a, b) - 1e-9));
  const k1 = Math.min(n - 1, Math.floor(Math.max(a, b) + 1e-9));
  return [k0, k1];
}

const centre = (g: Lattice, row: number, col: number) => ({ lat: g.lat_first + row * g.lat_step, lon: g.lon_first + col * g.lon_step });

// ---------------------------------------------------------------- samplers (synchronous, over loaded codes)

/** Observation date of one pixel: from its age (latest clear view) or the layer's own day. */
function dateOf(l: LayerArtifact, age: Codes | null, row: number, col: number): string | null {
  if (!l.composite) return l.time.observed_date ?? null;
  const a = age ? age(row, col) : NODATA;
  return a === (l.composite.age_grid?.nodata ?? NODATA) ? null : observedDate(l.composite.reference_date, a);
}

export type Member = { layer: LayerArtifact; code: Codes; age: Codes | null };

function memberAt(m: Member, row: number, col: number): { value: number; date: string } | null {
  const g = m.layer.grid!;
  const v = decodeCode(g, m.code(row, col));
  if (v == null) return null;
  const d = dateOf(m.layer, m.age, row, col);
  return d ? { value: v, date: d } : null;
}

/** One satellite layer (a single day or a latest clear view). */
export function satSampler(m: Member): Sampler {
  const g = m.layer.grid!;
  return {
    kind: "satellite",
    layer: m.layer,
    g,
    sources: [m.layer],
    at(row, col) {
      const o = memberAt(m, row, col);
      return { row, col, ...centre(g, row, col), value: o?.value ?? null, date: o?.date ?? null, sensor: null, native: o ? { row, col } : null };
    },
  };
}

/**
 * The multi-sensor display, pixel by pixel on the primary lattice, by the pipeline's own rule
 * (lib/multisensor `multiSensorPick`): each pixel is one sensor's own value and date, never a blend.
 */
export function multiSampler(layer: LayerArtifact, p: Member, s: Member): Sampler {
  const g = p.layer.grid!;
  const sg = s.layer.grid!;
  const tol = layer.multisensor?.prefer_primary_within_days ?? 2;
  return {
    kind: "satellite",
    layer,
    g,
    sources: [p.layer, s.layer],
    at(row, col) {
      const c = centre(g, row, col);
      const a = memberAt(p, row, col);
      const sc = cellAt(sg as ValueGrid, c.lat, c.lon);
      const b = sc ? memberAt(s, sc.row, sc.col) : null;
      const pick = multiSensorPick(a, b, tol);
      const o = pick === 1 ? a : pick === 2 ? b : null;
      return { row, col, ...c, value: o?.value ?? null, date: o?.date ?? null, sensor: pick === 0 ? null : pick, native: pick === 1 ? { row, col } : pick === 2 && sc ? { row: sc.row, col: sc.col } : null };
    },
  };
}

/** One radar hour (or the 24-hour mean): speed and direction of each valid cell. */
export function radarSampler(layer: LayerArtifact, f: CurrentField): Sampler {
  const g = f.g;
  const when = layer.time.valid_time ?? null;
  return {
    kind: "radar",
    layer,
    g,
    sources: [layer],
    field: f,
    at(row, col) {
      const k = row * g.width + col;
      const ok = row >= 0 && col >= 0 && row < g.height && col < g.width && Number.isFinite(f.u[k]) && Number.isFinite(f.v[k]);
      const base = { row, col, ...centre(g, row, col), sensor: null, native: ok ? { row, col } : null };
      if (!ok) return { ...base, value: null, date: null };
      const u = f.u[k];
      const v = f.v[k];
      return { ...base, value: Math.hypot(u, v), date: when, u, v, dir: directionDeg(u, v) };
    },
  };
}

// ---------------------------------------------------------------- point, transect, area

/** The pixel containing the point, exactly; null outside the product's grid. */
export function obsPoint(s: Sampler, lat: number, lon: number): ObsPixel | null {
  const c = cellAt(s.g as ValueGrid, lat, lon);
  return c ? s.at(c.row, c.col) : null;
}

export type ObsTransectCell = TransectCell & Pick<ObsPixel, "date" | "sensor" | "u" | "v" | "dir">;

/** Every pixel the line crosses, in order (the exact walk of `transect`); gaps stay null. */
export function obsTransect(s: Sampler, a: { lat: number; lon: number }, b: { lat: number; lon: number }): ObsTransectCell[] {
  return transectWith(s.g, (c: Cell) => s.at(c.row, c.col).value, a, b).map((c) => {
    if (!c.inDomain) return { ...c, date: null, sensor: null };
    const p = s.at(c.row, c.col);
    return { ...c, date: p.date, sensor: p.sensor, u: p.u, v: p.v, dir: p.dir };
  });
}

export type Spread = { min: number | null; median: number | null; max: number | null };

/** Lowest, median (the lower middle value when the count is even: always a published value) and highest. */
export function spread(values: number[]): Spread {
  const v = [...values].sort((a, b) => a - b);
  if (!v.length) return { min: null, median: null, max: null };
  return { min: v[0], median: v[Math.floor((v.length - 1) / 2)], max: v[v.length - 1] };
}

export type Counted<K> = { key: K; n: number }[];

function countBy<K>(xs: K[]): Counted<K> {
  const m = new Map<K, number>();
  for (const x of xs) m.set(x, (m.get(x) ?? 0) + 1);
  return [...m.entries()].map(([key, n]) => ({ key, n }));
}

export type ObsSummary = Spread & {
  /** pixels of the product's grid listed (inside the box, or crossed by the line) */
  inGrid: number;
  /** of those, with an observation */
  observed: number;
  /** observation dates (or radar hours), newest first, with pixel counts */
  dates: Counted<string>;
  /** multi-sensor: display pixels per sensor */
  sensors: Counted<1 | 2>;
  /** multi-sensor: distinct native VIIRS cells behind the VIIRS display pixels */
  nativeSecondary: number;
};

export function summarise(cells: (Pick<ObsPixel, "value" | "date" | "sensor"> & { native?: ObsPixel["native"]; inDomain?: boolean })[]): ObsSummary {
  const inGrid = cells.filter((c) => c.inDomain !== false);
  const obs = inGrid.filter((c) => c.value != null);
  const dates = countBy(obs.map((c) => c.date ?? "")).filter((d) => d.key).sort((a, b) => b.key.localeCompare(a.key));
  const sensors = countBy(obs.flatMap((c) => (c.sensor ? [c.sensor] : []))).sort((a, b) => a.key - b.key);
  const nativeSecondary = new Set(obs.filter((c) => c.sensor === 2 && c.native).map((c) => `${c.native!.row},${c.native!.col}`)).size;
  return { inGrid: inGrid.length, observed: obs.length, ...spread(obs.map((c) => c.value!)), dates, sensors, nativeSecondary };
}

/** Every pixel whose centre lies in the box, row by row, with its value or null (never filled). */
export function obsArea(s: Sampler, bbox: BBox): ObsPixel[] {
  const [w, so, e, n] = bbox;
  const [r0, r1] = rangeIn(s.g.lat_first, s.g.lat_step, s.g.height, so, n);
  const [c0, c1] = rangeIn(s.g.lon_first, s.g.lon_step, s.g.width, w, e);
  const out: ObsPixel[] = [];
  for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) out.push(s.at(r, c));
  return out;
}

/**
 * Coverage by the pipeline's own rule (the layers' `coverage.reference`): of the C-HARM ocean cells
 * (cells with a published Pseudo-nitzschia value) whose centre lies in the box, how many hold at
 * least one observed pixel. Separates "no observation over water" from land inside the box.
 */
export function oceanCoverage(ref: { g: ValueGrid; codes: Uint16Array } | null, bbox: BBox, s: Sampler): { cells: number; observed: number } | null {
  if (!ref) return null;
  const { g, codes } = ref;
  const [w, so, e, n] = bbox;
  const [r0, r1] = rangeIn(g.lat_first, g.lat_step, g.height, so, n);
  const [c0, c1] = rangeIn(g.lon_first, g.lon_step, g.width, w, e);
  let cells = 0;
  let observed = 0;
  const hy = Math.abs(g.lat_step) / 2;
  const hx = Math.abs(g.lon_step) / 2;
  for (let r = r0; r <= r1; r++)
    for (let c = c0; c <= c1; c++) {
      if (codes[r * g.width + c] === (g.nodata ?? NODATA)) continue;
      cells++;
      const { lat, lon } = centre(g, r, c);
      const [a0, a1] = rangeIn(s.g.lat_first, s.g.lat_step, s.g.height, lat - hy, lat + hy);
      const [b0, b1] = rangeIn(s.g.lon_first, s.g.lon_step, s.g.width, lon - hx, lon + hx);
      let hit = false;
      for (let i = a0; i <= a1 && !hit; i++) for (let j = b0; j <= b1 && !hit; j++) hit = s.at(i, j).value != null;
      if (hit) observed++;
    }
  return { cells, observed };
}

// ---------------------------------------------------------------- footprints (drawn on the map)

const r5 = (x: number) => +x.toFixed(5);

/** The square a pixel or cell covers (centre ± half a step). */
export function pixelPolygon(g: Lattice, row: number, col: number): GeoJSON.Polygon {
  const { lat, lon } = centre(g, row, col);
  const dy = Math.abs(g.lat_step) / 2;
  const dx = Math.abs(g.lon_step) / 2;
  return { type: "Polygon", coordinates: [[[r5(lon - dx), r5(lat - dy)], [r5(lon + dx), r5(lat - dy)], [r5(lon + dx), r5(lat + dy)], [r5(lon - dx), r5(lat + dy)], [r5(lon - dx), r5(lat - dy)]]] };
}

/**
 * The footprint of the observation behind a pixel: its own pixel, or for a VIIRS value in the
 * multi-sensor display the native 750 m VIIRS pixel it comes from (larger than the display pixel).
 */
export function footprintOf(s: Sampler, px: ObsPixel): { g: Lattice; row: number; col: number; lat: number; lon: number } {
  const src = px.sensor ? s.sources[px.sensor - 1] : null;
  const g = px.sensor === 2 && src?.grid && px.native ? src.grid : s.g;
  const row = g === s.g ? px.row : px.native!.row;
  const col = g === s.g ? px.col : px.native!.col;
  return { g, row, col, ...centre(g, row, col) };
}

/** The outline of a set of pixels: every pixel edge with an observed pixel on one side only. */
export function footprintOutline(g: Lattice, cells: { row: number; col: number }[]): GeoJSON.MultiLineString {
  const on = new Set(cells.map((c) => `${c.row},${c.col}`));
  const lines: number[][][] = [];
  const latE = (r: number, d: number) => r5(g.lat_first + (r + d / 2) * g.lat_step);
  const lonE = (c: number, d: number) => r5(g.lon_first + (c + d / 2) * g.lon_step);
  for (const { row: r, col: c } of cells) {
    for (const d of [-1, 1]) {
      if (!on.has(`${r + d},${c}`)) lines.push([[lonE(c, -1), latE(r, d)], [lonE(c, 1), latE(r, d)]]);
      if (!on.has(`${r},${c + d}`)) lines.push([[lonE(c, d), latE(r, -1)], [lonE(c, d), latE(r, 1)]]);
    }
  }
  return { type: "MultiLineString", coordinates: lines };
}

// ---------------------------------------------------------------- inter-sensor consistency (multi-sensor)

export type SameDayPair = { row: number; col: number; lat: number; lon: number; date: string; viirs: number; s3: number; nS3: number; log10Ratio: number; s3From: "day" | "latest" };

/** The primary sensor's observations on one date: its single-day grid, or its latest clear view's pixels of that date. */
export type S3Day = { g: Lattice; value: (row: number, col: number) => number | null; from: "day" | "latest" };

export type Consistency = {
  /** VIIRS native cells in the box with an observation */
  viirsCells: number;
  /** of those, cells whose VIIRS date has Sentinel-3 data published for the same day */
  sameDayCandidates: number;
  /** days of the VIIRS cells for which no Sentinel-3 observation of that day is published at all */
  noS3Days: string[];
  pairs: SameDayPair[];
  /** median of log10(Sentinel-3 / VIIRS) over the pairs; null without pairs */
  medianLog10: number | null;
  minPixels: number;
};

/**
 * Same-day pairs by the pipeline's published method (`multisensor.agreement_method`): for each VIIRS
 * 750 m cell observed on a day Sentinel-3 also observed, the geometric mean of the valid Sentinel-3
 * 300 m pixels of that same day inside it (at least `minPixels`) against the VIIRS value, in log10.
 * Only values published for that cell and that date are used. VIIRS values come from its latest clear
 * view (its single days are not published), so a VIIRS cell observed again later cannot be paired for
 * the earlier day. A consistency check between two sensors; not a validation of either.
 */
export function sameDayPairs(viirs: Sampler, s3For: (date: string) => S3Day | null, bbox: BBox, minPixels = 5): Consistency {
  const cells = obsArea(viirs, bbox).filter((c) => c.value != null && c.date);
  const pairs: SameDayPair[] = [];
  const noS3 = new Set<string>();
  let candidates = 0;
  const hy = Math.abs(viirs.g.lat_step) / 2;
  const hx = Math.abs(viirs.g.lon_step) / 2;
  for (const c of cells) {
    const d = s3For(c.date!);
    if (!d) {
      noS3.add(c.date!);
      continue;
    }
    candidates++;
    // half-open: a pixel centre on the far edge belongs to the next cell
    const [a0, a1] = rangeIn(d.g.lat_first, d.g.lat_step, d.g.height, c.lat - hy, c.lat + hy);
    const [b0, b1] = rangeIn(d.g.lon_first, d.g.lon_step, d.g.width, c.lon - hx, c.lon + hx);
    const logs: number[] = [];
    for (let i = a0; i <= a1; i++) {
      const plat = d.g.lat_first + i * d.g.lat_step;
      if (!(plat >= c.lat - hy && plat < c.lat + hy)) continue;
      for (let j = b0; j <= b1; j++) {
        const plon = d.g.lon_first + j * d.g.lon_step;
        if (!(plon >= c.lon - hx && plon < c.lon + hx)) continue;
        const v = d.value(i, j);
        if (v != null && v > 0) logs.push(Math.log10(v));
      }
    }
    if (logs.length < minPixels || !(c.value! > 0)) continue;
    const s3log = logs.reduce((a, b) => a + b, 0) / logs.length;
    pairs.push({ row: c.row, col: c.col, lat: c.lat, lon: c.lon, date: c.date!, viirs: c.value!, s3: 10 ** s3log, nS3: logs.length, log10Ratio: s3log - Math.log10(c.value!), s3From: d.from });
  }
  const r = pairs.map((p) => p.log10Ratio).sort((a, b) => a - b);
  const medianLog10 = r.length ? (r.length % 2 ? r[(r.length - 1) / 2] : (r[r.length / 2 - 1] + r[r.length / 2]) / 2) : null;
  return { viirsCells: cells.length, sameDayCandidates: candidates, noS3Days: [...noS3].sort(), pairs, medianLog10, minPixels };
}

// ---------------------------------------------------------------- choices and loading

export type SatChoice = { product: SatProduct; day: string | null };
export type RadarChoice = { hour: string | null; mean: boolean };

/** The satellite layer for a choice: a published single day with a grid, or the latest clear view. */
export function satLayerFor(m: Manifest, c: SatChoice): LayerArtifact | null {
  if (c.day) return satelliteDays(m, c.product).find((d) => d.time.observed_date === c.day && d.grid) ?? null;
  return satelliteLatest(m, c.product);
}

export function radarLayerFor(m: Manifest, c: RadarChoice): LayerArtifact | null {
  if (c.mean) return currentsMean(m);
  const hours = currentsHourly(m);
  return (c.hour ? hours.find((h) => h.time.valid_time === c.hour) : null) ?? hours[hours.length - 1] ?? null;
}

/** Row and column range of a grid covering a box, one pixel of margin. */
function blockOf(g: Lattice, bbox: BBox): [number, number, number, number] {
  const [w, s, e, n] = bbox;
  const [r0, r1] = rangeIn(g.lat_first, g.lat_step, g.height, s - Math.abs(g.lat_step), n + Math.abs(g.lat_step));
  const [c0, c1] = rangeIn(g.lon_first, g.lon_step, g.width, w - Math.abs(g.lon_step), e + Math.abs(g.lon_step));
  return [r0, r1, c0, c1];
}

export async function loadMember(baseUrl: string, l: LayerArtifact, bbox: BBox): Promise<Member> {
  const g = l.grid!;
  const [r0, r1, c0, c1] = blockOf(g, bbox);
  const age = l.composite?.age_grid?.chunks ? l.composite.age_grid : null;
  const [code, ageCode] = await Promise.all([chunkedBlock(baseUrl, g, r0, r1, c0, c1), age ? chunkedBlock(baseUrl, age, r0, r1, c0, c1) : Promise.resolve(null)]);
  return { layer: l, code, age: ageCode };
}

/** A sampler for the chosen satellite layer over a box (only the chunks the box touches are fetched). */
export async function loadSatSampler(m: Manifest, baseUrl: string, l: LayerArtifact, bbox: BBox): Promise<Sampler | null> {
  if (l.multisensor) {
    const [p, s] = multiSensorMembers(m, l);
    if (!p?.grid?.chunks || !s?.grid?.chunks) return null;
    const [a, b] = await Promise.all([loadMember(baseUrl, p, bbox), loadMember(baseUrl, s, bbox)]);
    return multiSampler(l, a, b);
  }
  if (!l.grid?.chunks) return null;
  return satSampler(await loadMember(baseUrl, l, bbox));
}

export async function loadRadarSampler(baseUrl: string, l: LayerArtifact): Promise<Sampler> {
  return radarSampler(l, await loadField(baseUrl, l));
}

/**
 * The inter-sensor check over a box: VIIRS from its latest clear view; Sentinel-3 for each VIIRS date
 * from its single-day grid when published (a day published without a clear pixel has no Sentinel-3
 * observation), else from the latest clear view's pixels of that date.
 */
export async function loadConsistency(m: Manifest, baseUrl: string, multi: LayerArtifact, bbox: BBox, margin = 0.01): Promise<Consistency | null> {
  const [p, s] = multiSensorMembers(m, multi);
  if (!p?.grid?.chunks || !s?.grid?.chunks) return null;
  // read a margin around the box, so every 300 m pixel inside a VIIRS cell at the box edge is there
  const load: BBox = [bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin];
  const [pm, sm] = await Promise.all([loadMember(baseUrl, p, load), loadMember(baseUrl, s, load)]);
  const viirs = satSampler(sm);
  const dates = [...new Set(obsArea(viirs, bbox).flatMap((c) => (c.date ? [c.date] : [])))];
  const product = p.layer_id.replace(/_chl_latest$/, "") as SatProduct;
  const days = satelliteDays(m, product);
  const s3: Map<string, S3Day | null> = new Map();
  await Promise.all(
    dates.map(async (d) => {
      const day = days.find((x) => x.time.observed_date === d);
      if (day && !day.grid) return void s3.set(d, null);
      if (day?.grid?.chunks) {
        const mem = await loadMember(baseUrl, day, load);
        return void s3.set(d, { g: day.grid, value: (r, c) => decodeCode(day.grid!, mem.code(r, c)), from: "day" });
      }
      // no single day in the manifest: the latest clear view's pixels observed that day
      const ok = (r: number, c: number) => dateOf(p, pm.age, r, c) === d;
      const has = p.composite?.days.some((x) => x.date === d && x.pixels_used > 0);
      s3.set(d, has ? { g: p.grid!, value: (r, c) => (ok(r, c) ? decodeCode(p.grid!, pm.code(r, c)) : null), from: "latest" } : null);
    }),
  );
  return sameDayPairs(viirs, (d) => s3.get(d) ?? null, bbox);
}

/** Value of each published OLCI single day at a point's pixel (days published without a grid: no clear pixel). */
export async function satDayStrip(m: Manifest, baseUrl: string, product: SatProduct, lat: number, lon: number): Promise<{ date: string; layer: LayerArtifact; value: number | null; published: boolean }[]> {
  const days = satelliteDays(m, product);
  return Promise.all(
    days.map(async (d) => {
      const date = d.time.observed_date ?? "";
      if (!d.grid?.chunks) return { date, layer: d, value: null, published: false };
      const c = cellAt(d.grid, lat, lon);
      if (!c) return { date, layer: d, value: null, published: true };
      const code = await chunkedBlock(baseUrl, d.grid, c.row, c.row, c.col, c.col);
      return { date, layer: d, value: decodeCode(d.grid, code(c.row, c.col)), published: true };
    }),
  );
}

/** Speed and direction at one radar cell for every published hour (and the 24-hour mean). */
export async function radarSeries(m: Manifest, baseUrl: string, lat: number, lon: number): Promise<{ layer: LayerArtifact; time: string; mean: boolean; px: ObsPixel | null }[]> {
  const hours = currentsHourly(m);
  const mean = currentsMean(m);
  const all = [...hours, ...(mean ? [mean] : [])];
  return Promise.all(
    all.map(async (l) => {
      const s = await loadRadarSampler(baseUrl, l);
      return { layer: l, time: l.time.valid_time ?? "", mean: l === mean, px: obsPoint(s, lat, lon) };
    }),
  );
}
