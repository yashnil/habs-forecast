import { addProtocol } from "maplibre-gl";
import type { LayerArtifact, Palette, ValueGrid } from "@/generated/schema";
import { loadGrid } from "@/lib/grid";
import { clipToWater, waterMask } from "@/lib/waterMask";

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
const PROTOCOL = "cwgrid";
type Entry = { grid: ValueGrid; url: string; lut: Promise<Uint32Array> };
const entries = new Map<string, Entry>();

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
 * RGBA pixels of Web Mercator tile z/x/y (size x size), each pixel the grid cell containing its
 * centre. Pure: tested against the published grid and the pipeline's palette rule.
 */
export function renderGridTile(g: ValueGrid, codes: Uint16Array, lut: Uint32Array, z: number, x: number, y: number, size = GRID_TILE_SIZE): Uint8ClampedArray {
  const n = 2 ** z;
  const lonFirst = g.lon_first > 180 ? g.lon_first - 360 : g.lon_first;
  const cols = axisIndex(size, lonFirst, g.lon_step, g.width, (px) => ((x + (px + 0.5) / size) / n) * 360 - 180);
  const rows = axisIndex(size, g.lat_first, g.lat_step, g.height, (py) => (Math.atan(Math.sinh(Math.PI * (1 - (2 * (y + (py + 0.5) / size)) / n))) * 180) / Math.PI);
  const out = new Uint32Array(size * size);
  for (let py = 0; py < size; py++) {
    const r = rows[py];
    if (r < 0) continue;
    const base = r * g.width;
    const o = py * size;
    for (let px = 0; px < size; px++) {
      const c = cols[px];
      if (c >= 0) out[o + px] = lut[codes[base + c]];
    }
  }
  return new Uint8ClampedArray(out.buffer);
}

let registered = false;
function register() {
  if (registered) return;
  registered = true;
  addProtocol(PROTOCOL, async (params) => {
    const m = /^cwgrid:\/\/([^/]+)\/(\d+)\/(\d+)\/(\d+)$/.exec(params.url);
    const e = m && entries.get(decodeURIComponent(m[1]));
    if (!m || !e) throw new Error(`unknown grid tile ${params.url}`);
    const [z, x, y] = [Number(m[2]), Number(m[3]), Number(m[4])];
    const [codes, lut, water] = await Promise.all([loadGrid(e.url), e.lut, waterMask(z, x, y, GRID_TILE_SIZE)]);
    const px = renderGridTile(e.grid, codes, lut, z, x, y);
    // ocean only, at the basemap's shoreline (lib/waterMask); unclipped if the basemap tile is unreadable
    if (water) clipToWater(px, water);
    const bitmap = await createImageBitmap(new ImageData(px as Uint8ClampedArray<ArrayBuffer>, GRID_TILE_SIZE, GRID_TILE_SIZE));
    return { data: bitmap };
  });
}

/** The tile URL template for a layer's grid, and the grid's outer cell edges [w, s, e, n]. */
export function gridTiles(layer: LayerArtifact, gridUrl: string): { template: string; bounds: [number, number, number, number] } {
  register();
  const g = layer.grid!;
  const key = `${layer.layer_id}@${gridUrl}`;
  if (!entries.has(key)) entries.set(key, { grid: g, url: gridUrl, lut: Promise.resolve().then(() => codeColours(g, layer.palette!)) });
  const lonFirst = g.lon_first > 180 ? g.lon_first - 360 : g.lon_first;
  const lon0 = lonFirst - Math.abs(g.lon_step) / 2;
  const lat0 = Math.min(g.lat_first, g.lat_first + g.lat_step * (g.height - 1)) - Math.abs(g.lat_step) / 2;
  return {
    template: `${PROTOCOL}://${encodeURIComponent(key)}/{z}/{x}/{y}`,
    bounds: [lon0, lat0, lon0 + Math.abs(g.lon_step) * g.width, lat0 + Math.abs(g.lat_step) * g.height],
  };
}
