import type { LayerArtifact, Palette, ValueGrid } from "@/generated/schema";

/**
 * C-HARM drawn from its published value grid, tile by tile (exact georeferencing).
 *
 * The published PNG is a faithful nearest-neighbour image, but two things move its cell edges
 * on screen (docs/coastwatch/25-charm-georeferencing.md):
 * - its rows are ~1/5 of a cell tall, so a north-south cell edge lands on the nearest row
 *   (mean 164 m, at most 355 m off);
 * - MapLibre places an image source by rounding its corners to a grid inside one tile (8,192
 *   steps of a zoom-4 tile here: 306 m), which shifts and stretches the whole image by up to
 *   ~77 m.
 * Tiles drawn here have neither: every tile pixel takes the grid cell containing the pixel's
 * centre (nearest neighbour, the same rule as the pipeline), at the zoom being viewed, so a cell
 * edge is within half a screen pixel of its true position at every zoom. Values come from the
 * same published grid the readout uses, coloured with the layer's own published palette using
 * the pipeline's step rule; cells without a value stay transparent. Nothing is interpolated.
 * Each tile is then clipped to the basemap's water (lib/waterMask), so a cell that straddles the
 * shore stops at the drawn coastline.
 */

export const GRID_TILE_SIZE = 512;

const hexRgb = (h: string): [number, number, number] => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
/** RGBA packed for a little-endian Uint32Array view (bytes R, G, B, A in memory). */
const pack = (r: number, g: number, b: number, a: number) => ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
const littleEndian = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** Only a linear step palette is drawn here (C-HARM); anything else keeps the published image. */
export function supportsGridTiles(l: Pick<LayerArtifact, "grid" | "palette">): boolean {
  const p = l.palette;
  return !!l.grid && !l.grid.chunks && (l.grid.transform ?? "none") === "none" && !!p && p.interpolation === "step" && (p.scale ?? "linear") === "linear" && p.stops.length > 0 && littleEndian;
}

/**
 * Colour of every possible grid code, by the pipeline's step rule (palette.py apply_palette):
 * value = code x scale + offset, clipped to the palette domain; class = the last stop whose value
 * is <= the value (numpy searchsorted side="right" - 1); the no-data code is transparent.
 */
export function codeColours(g: Pick<ValueGrid, "scale_factor" | "add_offset" | "nodata">, p: Palette): Uint32Array {
  const xs = p.stops.map((s) => s.value);
  const cols = p.stops.map((s) => pack(...hexRgb(s.color), 255));
  const [lo, hi] = p.domain;
  const lut = new Uint32Array(65536);
  for (let code = 0; code < 65536; code++) {
    if (code === (g.nodata ?? 65535)) continue; // 0 = transparent
    const v = Math.min(hi, Math.max(lo, code * g.scale_factor + g.add_offset));
    let i = 0;
    while (i + 1 < xs.length && xs[i + 1] <= v) i++;
    lut[code] = cols[i];
  }
  return lut;
}

/** Index of the grid row (or column) containing each pixel centre, or -1 outside the grid. */
function axisIndex(n: number, first: number, step: number, count: number, coord: (k: number) => number): Int32Array {
  const out = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    const i = Math.floor((coord(k) - first) / step + 0.5);
    out[k] = i >= 0 && i < count ? i : -1;
  }
  return out;
}

/**
 * How a tile marks what was not observed or not modelled at that water (v5), in screen pixels so the
 * texture keeps its size at every zoom:
 * - `tint`: every cell of the grid without a value is drawn in one calm neutral colour (C-HARM:
 *   "NOAA publishes no toxin probability here"). No value is drawn or implied: NOAA's nearest value is
 *   never painted onto water it publishes nothing for (it showed no skill at the shore against pier samples).
 * - `stipple`: valued pixels get a fine stipple (`dot`) over a lightened colour (`lut`): an
 *   estimate, not an observation.
 * - `hatch`: valued pixels get fine diagonal lines (`line`): an outlook, not an observation.
 * - `strip` (v6, with `stipple`): estimate pixels whose class (`cls`, the class grid's codes, cell
 *   for cell with the values) is 2, extrapolated shoreward and never seen by the satellite, are
 *   paler and carry a denser stipple of their own (`stripAt`), so the shoreline strip reads apart
 *   from the fill between clear pixels.
 */
export type TileMarks = {
  tint?: number;
  stipple?: { lut: Uint32Array; dot: Uint32Array };
  hatch?: { line: Uint32Array };
  strip?: { lut: Uint32Array; dot: Uint32Array; cls: Uint16Array };
};

/** The stipple: one dot every 4 px, rows staggered by 2 px (an even, quiet texture). */
export const stippleAt = (px: number, py: number) => (py & 3) === 0 ? (px & 3) === 0 : (py & 3) === 2 && (px & 3) === 2;
/** The shoreline strip's stipple: a dot every 3 px on every row (denser than the fill's, so the strip reads apart). */
export const stripAt = (px: number, py: number) => px % 3 === 0 && py % 3 === 0;
/** The outlook hatch: a 1 px line every 7 px, rising to the right. */
export const hatchAt = (px: number, py: number) => (px + py) % 7 === 0;

/**
 * RGBA pixels of Web Mercator tile z/x/y (size x size), each pixel the grid cell containing its
 * centre. Pure: tested against the published grid and the pipeline's palette rule.
 */
export function renderGridTile(g: ValueGrid, codes: Uint16Array, lut: Uint32Array, z: number, x: number, y: number, size = GRID_TILE_SIZE, marks: TileMarks = {}): Uint8ClampedArray {
  const n = 2 ** z;
  const lonFirst = g.lon_first > 180 ? g.lon_first - 360 : g.lon_first;
  const cols = axisIndex(size, lonFirst, g.lon_step, g.width, (px) => ((x + (px + 0.5) / size) / n) * 360 - 180);
  const rows = axisIndex(size, g.lat_first, g.lat_step, g.height, (py) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + (py + 0.5) / size)) / n))) * 180) / Math.PI);
  const out = new Uint32Array(size * size);
  const nodata = g.nodata ?? 65535;
  const { tint, stipple, hatch, strip } = marks;
  for (let py = 0; py < size; py++) {
    const r = rows[py];
    if (r < 0) continue;
    const base = r * g.width;
    const o = py * size;
    for (let px = 0; px < size; px++) {
      const c = cols[px];
      if (c < 0) continue;
      const code = codes[base + c];
      if (code === nodata) {
        if (tint) out[o + px] = tint;
        continue;
      }
      if (strip && strip.cls[base + c] === 2) out[o + px] = stripAt(px, py) ? strip.dot[code] : strip.lut[code];
      else if (stipple) out[o + px] = stippleAt(px, py) ? stipple.dot[code] : stipple.lut[code];
      else if (hatch && hatchAt(px, py)) out[o + px] = hatch.line[code];
      else out[o + px] = lut[code];
    }
  }
  return new Uint8ClampedArray(out.buffer);
}

const unpack = (c: number): [number, number, number, number] => [c & 255, (c >>> 8) & 255, (c >>> 16) & 255, c >>> 24];

/** A LUT with every colour moved `f` of the way to `to` (an RGB triple); transparent stays transparent. */
export function mixLut(lut: Uint32Array, to: [number, number, number], f: number, alpha = 255): Uint32Array {
  const out = new Uint32Array(lut.length);
  const memo = new Map<number, number>();
  for (let i = 0; i < lut.length; i++) {
    const c = lut[i];
    if (!c) continue;
    let m = memo.get(c);
    if (m === undefined) {
      const [r, g, b] = unpack(c);
      m = pack(Math.round(r + (to[0] - r) * f), Math.round(g + (to[1] - g) * f), Math.round(b + (to[2] - b) * f), alpha);
      memo.set(c, m);
    }
    out[i] = m;
  }
  return out;
}

/** The light, cool grey every "not observed / not modelled here" colour is moved towards. */
export const VEIL: [number, number, number] = [201, 211, 223];
/** The stipple's dot colour: the basemap's sea, so the dots read as texture, never as a value. */
export const DOT: [number, number, number] = [11, 29, 51];

/** "NOAA publishes no toxin probability here": a calm slate, apart from every probability class (MapCanvas
 *  draws the forecast at 0.72 opacity, so it reads as a quiet grey-blue over the sea). */
export const OUTSIDE_TINT_RGB: [number, number, number] = [126, 140, 156];
export const OUTSIDE_TINT = pack(...OUTSIDE_TINT_RGB, 255);

/**
 * Colour of every possible code of a log10-stored grid (chlorophyll) on a linear ramp in log10
 * units (the satellite palette, `palette.scale` "log10"): code x scale + offset is already log10,
 * clipped to the palette's domain and interpolated between its stops; no-data is transparent.
 */
export function rampColours(g: Pick<ValueGrid, "scale_factor" | "add_offset" | "nodata" | "transform">, p: Palette): Uint32Array {
  const xs = p.stops.map((s) => s.value);
  const cs = p.stops.map((s) => hexRgb(s.color));
  const [lo, hi] = p.domain;
  const lut = new Uint32Array(65536);
  for (let code = 0; code < 65536; code++) {
    if (code === (g.nodata ?? 65535)) continue;
    const q0 = code * g.scale_factor + g.add_offset;
    const q = Math.min(hi, Math.max(lo, g.transform === "log10" || p.scale !== "log10" ? q0 : Math.log10(Math.max(q0, 1e-12))));
    let i = 0;
    while (i < xs.length - 2 && xs[i + 1] < q) i++;
    const f = xs.length < 2 ? 0 : Math.min(1, Math.max(0, (q - xs[i]) / Math.max(1e-9, xs[i + 1] - xs[i])));
    const a = cs[i];
    const b = cs[Math.min(i + 1, cs.length - 1)];
    lut[code] = pack(Math.round(a[0] + (b[0] - a[0]) * f), Math.round(a[1] + (b[1] - a[1]) * f), Math.round(a[2] + (b[2] - a[2]) * f), 255);
  }
  return lut;
}

/** The rows and columns of grid `g` that tile z/x/y touches (inclusive), or null if none. */
export function tileBlock(g: Pick<ValueGrid, "lat_first" | "lat_step" | "lon_first" | "lon_step" | "width" | "height">, z: number, x: number, y: number, size = GRID_TILE_SIZE): { r0: number; r1: number; c0: number; c1: number } | null {
  const n = 2 ** z;
  const lonFirst = g.lon_first > 180 ? g.lon_first - 360 : g.lon_first;
  const cols = axisIndex(size, lonFirst, g.lon_step, g.width, (px) => ((x + (px + 0.5) / size) / n) * 360 - 180);
  const rows = axisIndex(size, g.lat_first, g.lat_step, g.height, (py) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + (py + 0.5) / size)) / n))) * 180) / Math.PI);
  let r0 = Infinity, r1 = -1, c0 = Infinity, c1 = -1;
  for (const r of rows) {
    if (r < 0) continue;
    r0 = Math.min(r0, r);
    r1 = Math.max(r1, r);
  }
  for (const c of cols) {
    if (c < 0) continue;
    c0 = Math.min(c0, c);
    c1 = Math.max(c1, c);
  }
  return r1 < 0 || c1 < 0 ? null : { r0, r1, c0, c1 };
}
