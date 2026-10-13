import type { Grid } from "../data/store";
import type { PaletteStop } from "../data/types";
import { km } from "../lib/geo";

/**
 * The model "field": C-HARM probabilities drawn as their own 3 km cells, with exact edges, in Web
 * Mercator so a cell's edge on screen is the cell's edge in the data. Three compositing modes —
 * none of them changes a value, a class or a cell boundary; they only change how loudly the field
 * is drawn relative to the coast:
 *   veil   all cells, each class colour mixed toward the sea colour (default 62 % class) so the
 *          coastline, isobaths and labels — drawn above the field — stay legible. The default.
 * Every valued cell is drawn fully opaque (pre-mixed with the sea), so the "no value" hatch under
 * the field can only ever appear in cells that have no value.
 *   lens   cells whose centre lies within a radius of the selected water point at full opacity,
 *          all other cells at a low opacity. The lens edge follows cell edges (a staircase), so
 *          no cell is ever partly shown.
 *   edges  class boundaries only: lines along the cell edges where the 10 % class changes, over a
 *          faint veil — "isobaths of probability". No contouring between cells, no smoothing.
 */
export type FieldMode = "veil" | "lens" | "edges";
export interface FieldOptions { mode: FieldMode; veil?: number; lens?: { lon: number; lat: number; km: number } | null; outside?: number }

const SEA_RGB = [11, 29, 51]; // style.ts SEA
const PX = 4; // canvas pixels per cell (x); y follows Mercator
const mercY = (lat: number) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
const latFromY = (y: number) => (360 / Math.PI) * Math.atan(Math.exp(y)) - 90;

function hex(c: string): [number, number, number] {
  const n = parseInt(c.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
export function classOf(v: number) { return Math.max(0, Math.min(9, Math.floor(v * 10 + 1e-9))); }

export function fieldExtent(grid: Grid) {
  const s = grid.spec;
  const hx = Math.abs(s.lon_step) / 2, hy = Math.abs(s.lat_step) / 2;
  const lons = [s.lon_first, s.lon_first + (s.width - 1) * s.lon_step];
  const lats = [s.lat_first, s.lat_first + (s.height - 1) * s.lat_step];
  const west = Math.min(...lons) - hx, east = Math.max(...lons) + hx;
  const south = Math.min(...lats) - hy, north = Math.max(...lats) + hy;
  return { west, east, south, north, corners: [[west, north], [east, north], [east, south], [west, south]] as [[number, number], [number, number], [number, number], [number, number]] };
}

/** Renders the field into `canvas` (resized as needed). Returns the image corners for MapLibre. */
export function renderField(canvas: HTMLCanvasElement, grid: Grid, stops: PaletteStop[], opt: FieldOptions) {
  const s = grid.spec;
  const ext = fieldExtent(grid);
  const W = s.width * PX;
  const yN = mercY(ext.north), yS = mercY(ext.south);
  const H = Math.round(((yN - yS) / (((ext.east - ext.west) * Math.PI) / 180)) * W);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const g = canvas.getContext("2d", { willReadFrequently: false })!;
  const img = g.createImageData(W, H);
  const colours = stops.map((st) => hex(st.color));
  const veil = opt.veil ?? 0.6;
  const outside = opt.outside ?? 0.2;
  const edges = opt.mode === "edges";
  // per-cell alpha (lens membership by cell centre)
  const cellAlpha = new Float32Array(s.width * s.height);
  for (let r = 0; r < s.height; r++) for (let c = 0; c < s.width; c++) {
    let a = edges ? 0.24 : veil;
    if (opt.mode === "lens" && opt.lens) {
      const d = km([opt.lens.lon, opt.lens.lat], [s.lon_first + c * s.lon_step, s.lat_first + r * s.lat_step]);
      a = d <= opt.lens.km ? 1 : outside;
    }
    cellAlpha[r * s.width + c] = a;
  }
  const rowOfPixel = new Int32Array(H);
  for (let y = 0; y < H; y++) {
    const lat = latFromY(yN - ((y + 0.5) / H) * (yN - yS));
    rowOfPixel[y] = Math.round((lat - s.lat_first) / s.lat_step);
  }
  const d = img.data;
  for (let y = 0; y < H; y++) {
    const row = rowOfPixel[y];
    if (row < 0 || row >= s.height) continue;
    for (let x = 0; x < W; x++) {
      const col = Math.floor(x / PX);
      const i = row * s.width + (s.lon_step > 0 ? col : s.width - 1 - col);
      const v = grid.values[i];
      if (Number.isNaN(v)) continue;
      const [cr, cg, cb] = colours[classOf(v)];
      const a = cellAlpha[i];
      const o = (y * W + x) * 4;
      d[o] = Math.round(SEA_RGB[0] + (cr - SEA_RGB[0]) * a); d[o + 1] = Math.round(SEA_RGB[1] + (cg - SEA_RGB[1]) * a); d[o + 2] = Math.round(SEA_RGB[2] + (cb - SEA_RGB[2]) * a); d[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return ext.corners;
}

/**
 * Class boundaries along cell edges, as GeoJSON lines (one feature per run of edge segments with
 * the same pair of classes). Each line carries `hi` (the higher class) for styling and labelling.
 */
export function classEdges(grid: Grid): GeoJSON.FeatureCollection<GeoJSON.LineString, { hi: number; lo: number }> {
  const s = grid.spec;
  const hx = s.lon_step / 2, hy = s.lat_step / 2;
  const at = (r: number, c: number) => (r < 0 || c < 0 || r >= s.height || c >= s.width ? NaN : grid.values[r * s.width + c]);
  const feats: GeoJSON.Feature<GeoJSON.LineString, { hi: number; lo: number }>[] = [];
  // vertical edges: between (r,c) and (r,c+1), merged down rows
  for (let c = 0; c < s.width - 1; c++) {
    let run: { r0: number; hi: number; lo: number } | null = null;
    const flush = (rEnd: number) => {
      if (!run) return;
      const lon = s.lon_first + c * s.lon_step + hx;
      feats.push({ type: "Feature", properties: { hi: run.hi, lo: run.lo }, geometry: { type: "LineString", coordinates: [[lon, s.lat_first + run.r0 * s.lat_step - hy], [lon, s.lat_first + rEnd * s.lat_step + hy]] } });
      run = null;
    };
    for (let r = 0; r < s.height; r++) {
      const a = at(r, c), b = at(r, c + 1);
      const ok = !Number.isNaN(a) && !Number.isNaN(b) && classOf(a) !== classOf(b);
      const hi = ok ? Math.max(classOf(a), classOf(b)) : -1, lo = ok ? Math.min(classOf(a), classOf(b)) : -1;
      if (run && (!ok || run.hi !== hi || run.lo !== lo)) flush(r - 1);
      if (ok && !run) run = { r0: r, hi, lo };
    }
    flush(s.height - 1);
  }
  // horizontal edges: between (r,c) and (r+1,c), merged along columns
  for (let r = 0; r < s.height - 1; r++) {
    let run: { c0: number; hi: number; lo: number } | null = null;
    const flush = (cEnd: number) => {
      if (!run) return;
      const lat = s.lat_first + r * s.lat_step + hy;
      feats.push({ type: "Feature", properties: { hi: run.hi, lo: run.lo }, geometry: { type: "LineString", coordinates: [[s.lon_first + run.c0 * s.lon_step - hx, lat], [s.lon_first + cEnd * s.lon_step + hx, lat]] } });
      run = null;
    };
    for (let c = 0; c < s.width; c++) {
      const a = at(r, c), b = at(r + 1, c);
      const ok = !Number.isNaN(a) && !Number.isNaN(b) && classOf(a) !== classOf(b);
      const hi = ok ? Math.max(classOf(a), classOf(b)) : -1, lo = ok ? Math.min(classOf(a), classOf(b)) : -1;
      if (run && (!ok || run.hi !== hi || run.lo !== lo)) flush(c - 1);
      if (ok && !run) run = { c0: c, hi, lo };
    }
    flush(s.width - 1);
  }
  return { type: "FeatureCollection", features: feats };
}

/** The outline of the cells that have a value (the model's footprint), as cell-edge lines. */
export function footprint(grid: Grid): GeoJSON.FeatureCollection<GeoJSON.LineString> {
  const s = grid.spec;
  const hx = s.lon_step / 2, hy = s.lat_step / 2;
  const has = (r: number, c: number) => r >= 0 && c >= 0 && r < s.height && c < s.width && !Number.isNaN(grid.values[r * s.width + c]);
  const lines: GeoJSON.Position[][] = [];
  for (let r = 0; r < s.height; r++) for (let c = 0; c < s.width; c++) {
    if (!has(r, c)) continue;
    const lo = s.lon_first + c * s.lon_step, la = s.lat_first + r * s.lat_step;
    if (!has(r, c - 1)) lines.push([[lo - hx, la - hy], [lo - hx, la + hy]]);
    if (!has(r, c + 1)) lines.push([[lo + hx, la - hy], [lo + hx, la + hy]]);
    if (!has(r - 1, c)) lines.push([[lo - hx, la - hy], [lo + hx, la - hy]]);
    if (!has(r + 1, c)) lines.push([[lo - hx, la + hy], [lo + hx, la + hy]]);
  }
  return { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "MultiLineString", coordinates: lines } as unknown as GeoJSON.LineString }] };
}
