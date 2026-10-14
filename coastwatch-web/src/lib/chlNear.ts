import type { LayerArtifact } from "@/generated/schema";
import { cellAt, chunkedBlock, decodeCode, distanceKm } from "@/lib/grid";
import { observedDate } from "@/lib/multisensor";
import { loadWaterFallback, rasterWater, type WaterGeometry } from "@/lib/waterFallback";

/**
 * Satellite chlorophyll near a place, read with the same quality rules everywhere a place gets a
 * number (the map's place inspector and My Coast). Nothing published is changed or removed: the
 * map still draws every pixel. The rules decide only which pixels may become a place's headline.
 *
 * - Q1 water: a pixel counts only if its centre is on the sea, per the app's own bundled water
 *   (lib/waterFallback, the OpenMapTiles ocean the map clips rasters with; lakes and lagoons such
 *   as Lake Earl are not sea). If that geometry cannot be loaded, Q1 and Q2 are skipped and the
 *   result says so: a missing file never hides data.
 * - Q2 shoreline: a sea pixel whose centre is within one native pixel of land (OLCI 300 m, VIIRS
 *   750 m) is set aside. Land, rocks and surf brighten the pixel in the near infrared and the
 *   atmospheric correction then overestimates chlorophyll.
 * - Q3 isolated extreme: a value of at least 20 mg/m³ and more than 10 times the median of the
 *   other sea pixels within ±1 km (at least 5 of them) is set aside. A real bloom spans several
 *   pixels; a single pixel 30 to 100 times its neighbours is not a bloom.
 *
 * There is no fixed ceiling: real blooms on this coast have exceeded 300 mg/m³ in situ.
 * Headline: the pixel at the point if it passes; otherwise the median of the pixels that pass
 * within the radius, with their count, range and newest date. Never a fill or an interpolation.
 */
export const CHL_QC = {
  /** Q3: values at least this high can be isolated extremes (mg/m³) */
  extremeMin: 20,
  /** Q3: an isolated extreme is more than this many times its neighbours' median */
  spikeRatio: 10,
  /** Q3: the neighbourhood half-width (km) and the neighbours it needs */
  spikeWindowKm: 1,
  spikeMinNeighbours: 5,
  /** a set-aside pixel is mentioned only if it is at least this many times the headline */
  asideRatio: 3,
} as const;

/** Where the sea is: the bundled water geometry rasterised on demand, cached per tile. */
export type WaterProbe = {
  /** the point is on the sea (true outside the geometry's area, where it has no coastline) */
  isWater(lat: number, lon: number): boolean;
  /** land lies within `metres` of the point */
  nearLand(lat: number, lon: number, metres: number): boolean;
};

const PROBE_Z = 12;
const PROBE_SIZE = 512; // about 14 m per pixel on this coast

export function waterProbe(g: WaterGeometry): WaterProbe {
  const tiles = new Map<number, Uint8ClampedArray>();
  const world = 2 ** PROBE_Z * PROBE_SIZE;
  const toPx = (lat: number, lon: number): [number, number] => [((lon + 180) / 360) * world, (0.5 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / (2 * Math.PI)) * world];
  const wet = (px: number, py: number) => {
    const tx = Math.floor(px / PROBE_SIZE);
    const ty = Math.floor(py / PROBE_SIZE);
    const key = tx * 2 ** PROBE_Z + ty;
    let t = tiles.get(key);
    if (!t) tiles.set(key, (t = rasterWater(g, PROBE_Z, tx, ty, PROBE_SIZE)));
    return t[(py - ty * PROBE_SIZE) * PROBE_SIZE + (px - tx * PROBE_SIZE)] >= 128;
  };
  return {
    isWater(lat, lon) {
      const [x, y] = toPx(lat, lon);
      return wet(Math.floor(x), Math.floor(y));
    },
    nearLand(lat, lon, metres) {
      const [x, y] = toPx(lat, lon);
      const mPerPx = (40_075_016.686 * Math.cos((lat * Math.PI) / 180)) / world;
      const r = metres / mPerPx;
      const step = Math.max(1, Math.floor(r / 16)); // at most ~800 probes per disk
      const [cx, cy] = [Math.floor(x), Math.floor(y)];
      const n = Math.ceil(r / step);
      for (let i = -n; i <= n; i++)
        for (let j = -n; j <= n; j++) {
          const dx = i * step;
          const dy = j * step;
          if (dx * dx + dy * dy > r * r) continue;
          if (!wet(cx + dx, cy + dy)) return true;
        }
      return false;
    },
  };
}

let probe: Promise<WaterProbe | null> | null = null;
/** The water probe from the bundled geometry, or null if it cannot be loaded (retried next time). */
export function loadWaterProbe(): Promise<WaterProbe | null> {
  probe ??= loadWaterFallback().then((g) => {
    if (!g) probe = null;
    return g ? waterProbe(g) : null;
  });
  return probe;
}

export type ChlPixel = { lat: number; lon: number; value: number; date: string | null; km: number };

/** The highest pixel set aside by Q2 or Q3, mentioned because it is far above the headline. */
export type ChlAside = { value: number; reason: "shore" | "spike" };

type Common = {
  radiusKm: number;
  /** whether the water rules (Q1, Q2) were applied; false if the water geometry could not be loaded */
  water: boolean;
  /** the pixel at the point: none (cloud, no pass), used, or set aside (land, shoreline, isolated extreme) */
  atPoint: "none" | "used" | "set-aside";
};

export type ChlNear = Common &
  (
    | { kind: "at"; value: number; date: string | null; pixel: ChlPixel; aside: ChlAside | null }
    | { kind: "median"; value: number; date: string | null; n: number; min: number; max: number; nearest: ChlPixel; aside: ChlAside | null }
    /** pixels exist within the radius, but none on open water that passes the rules */
    | { kind: "flagged"; n: number }
    | { kind: "none" }
  );

const median = (vs: number[]) => {
  const s = [...vs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

type Px = ChlPixel & { code: number; dLatKm: number; dLonKm: number; sea: boolean; shore: boolean; spike: boolean; here: boolean };

/**
 * Read chlorophyll near (lat, lon) from a chunked satellite layer with the rules above.
 * `water`: pass a probe (or null for "unavailable") to skip loading the bundled geometry.
 */
export async function readChlNear(baseUrl: string, layer: LayerArtifact, lat: number, lon: number, opts: { radiusKm: number; water?: WaterProbe | null }): Promise<ChlNear> {
  const { radiusKm } = opts;
  const g = layer.grid;
  const water = opts.water === undefined ? await loadWaterProbe() : opts.water;
  const empty: Common = { radiusKm, water: !!water, atPoint: "none" };
  if (!g?.chunks) return { ...empty, kind: "none" };
  const c = cellAt(g, lat, lon);
  if (!c) return { ...empty, kind: "none" };
  const kmLat = Math.abs(g.lat_step) * 111.32;
  const kmLon = Math.abs(g.lon_step) * 111.32 * Math.cos((lat * Math.PI) / 180);
  const reach = radiusKm + CHL_QC.spikeWindowKm;
  const dr = Math.ceil(reach / kmLat);
  const dc = Math.ceil(reach / kmLon);
  const age = layer.composite?.age_grid ?? null;
  const [codeAt, ageAt] = await Promise.all([
    chunkedBlock(baseUrl, g, c.row - dr, c.row + dr, c.col - dc, c.col + dc),
    age?.chunks ? chunkedBlock(baseUrl, age, c.row - dr, c.row + dr, c.col - dc, c.col + dc) : null,
  ]);
  const nodata = g.nodata ?? 65535;
  const dateOf = (row: number, col: number): string | null => {
    if (!layer.composite) return layer.time.observed_date ?? null;
    const a = ageAt?.(row, col);
    return a == null || a === (age?.nodata ?? 65535) ? null : observedDate(layer.composite.reference_date, a);
  };
  const native = layer.native_resolution_m ?? (layer.resolution_deg ? layer.resolution_deg * 111_000 : 300);

  const all: Px[] = [];
  for (let r = c.row - dr; r <= c.row + dr; r++)
    for (let k = c.col - dc; k <= c.col + dc; k++) {
      const code = codeAt(r, k);
      if (code === nodata) continue;
      const plat = g.lat_first + r * g.lat_step;
      const plon = g.lon_first + k * g.lon_step;
      const here = r === c.row && k === c.col;
      const value = decodeCode(g, code)!;
      all.push({ lat: plat, lon: plon, value, code, date: null, km: here ? 0 : distanceKm(lat, lon, plat, plon), dLatKm: (r - c.row) * kmLat, dLonKm: (k - c.col) * kmLon, sea: water ? water.isWater(plat, plon) : true, shore: false, spike: false, here });
    }
  const inRadius = all.filter((p) => p.here || p.km <= radiusKm);
  const sea = all.filter((p) => p.sea);
  for (const p of inRadius) {
    if (!p.sea) continue;
    p.date = dateOf(Math.round((p.lat - g.lat_first) / g.lat_step), Math.round((p.lon - g.lon_first) / g.lon_step));
    p.shore = !!water && water.nearLand(p.lat, p.lon, native);
    if (p.value >= CHL_QC.extremeMin) {
      // the other sea pixels within ±1 km; a value duplicated from the same native pixel is not a neighbour
      const nb = sea.filter((q) => q !== p && q.code !== p.code && Math.abs(q.dLatKm - p.dLatKm) <= CHL_QC.spikeWindowKm && Math.abs(q.dLonKm - p.dLonKm) <= CHL_QC.spikeWindowKm).map((q) => q.value);
      p.spike = nb.length >= CHL_QC.spikeMinNeighbours && p.value > CHL_QC.spikeRatio * median(nb);
    }
  }
  const ok = inRadius.filter((p) => p.sea && !p.shore && !p.spike);
  const setAside = inRadius.filter((p) => p.sea && (p.shore || p.spike));
  const pixel = (p: Px): ChlPixel => ({ lat: p.lat, lon: p.lon, value: p.value, date: p.date, km: p.km });
  const asideFor = (headline: number): ChlAside | null => {
    const high = setAside.filter((p) => p.value >= CHL_QC.asideRatio * headline);
    if (!high.length) return null;
    const top = high.reduce((a, b) => (b.value > a.value ? b : a));
    return { value: top.value, reason: top.shore ? "shore" : "spike" };
  };

  const at = ok.find((p) => p.here);
  const common: Common = { ...empty, atPoint: at ? "used" : inRadius.some((p) => p.here) ? "set-aside" : "none" };
  if (at) return { ...common, kind: "at", value: at.value, date: at.date, pixel: pixel(at), aside: asideFor(at.value) };
  if (ok.length) {
    const vs = ok.map((p) => p.value);
    const value = median(vs);
    const nearest = ok.reduce((a, b) => (b.km < a.km ? b : a));
    const dates = ok.map((p) => p.date).filter((d): d is string => !!d);
    const date = dates.length ? dates.reduce((a, b) => (b > a ? b : a)) : null;
    return { ...common, kind: "median", value, date, n: ok.length, min: Math.min(...vs), max: Math.max(...vs), nearest: pixel(nearest), aside: asideFor(value) };
  }
  if (inRadius.length) return { ...common, kind: "flagged", n: inRadius.length };
  return { ...common, kind: "none" };
}

/** Chlorophyll as the readouts print it. */
export const fmtChlValue = (v: number) => (v >= 10 ? v.toFixed(0) : v.toPrecision(2));
