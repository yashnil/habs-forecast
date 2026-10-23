import { addProtocol } from "maplibre-gl";
import type { LayerArtifact, Palette, ValueGrid } from "@/generated/schema";
import { chunkedBlock, loadGrid } from "@/lib/grid";
import { codeColours, DOT, GRID_TILE_SIZE, mixLut, OUTSIDE_TINT, rampColours, renderGridTile, tileBlock, VEIL, type TileMarks } from "@/lib/gridTiles";
import { clearBays } from "@/lib/bays";
import { clipToWater, waterMask } from "@/lib/waterMask";

/**
 * The `cwgrid://` tile protocol (see lib/gridTiles). Kept apart from the pure helpers so that
 * only the lazily loaded map canvas imports MapLibre.
 *
 * Two kinds of grid are drawn:
 * - a full C-HARM grid (step palette), optionally with its no-value water tinted as "NOAA publishes
 *   no toxin probability here" (lib/gridTilesContext);
 * - a chunked chlorophyll grid of a derived product (lib/derived: the estimate and the outlook),
 *   on the satellite ramp, marked as estimated (stipple) or as an outlook (hatch). Only the chunks a
 *   tile touches are fetched. v6: the estimate's shoreline strip (class 2 of its class grid, when
 *   published) is marked apart; the outlook is never drawn over bay water (lib/bays).
 */
const PROTOCOL = "cwgrid";
type Entry =
  | { kind: "full"; grid: ValueGrid; url: string; lut: Promise<Uint32Array>; context: boolean }
  | { kind: "chunked"; grid: ValueGrid; baseUrl: string; lut: Promise<Uint32Array>; marks: Promise<TileMarks>; classGrid?: ValueGrid | null; noBays?: boolean };
const entries = new Map<string, Entry>();

let registered = false;
function register() {
  if (registered) return;
  registered = true;
  addProtocol(PROTOCOL, async (params) => {
    const m = /^cwgrid:\/\/([^/]+)\/(\d+)\/(\d+)\/(\d+)$/.exec(params.url);
    const e = m && entries.get(decodeURIComponent(m[1]));
    if (!m || !e) throw new Error(`unknown grid tile ${params.url}`);
    const [z, x, y] = [Number(m[2]), Number(m[3]), Number(m[4])];
    const px = e.kind === "full" ? await fullTile(e, z, x, y) : await chunkedTile(e, z, x, y);
    const bitmap = await createImageBitmap(new ImageData(px as Uint8ClampedArray<ArrayBuffer>, GRID_TILE_SIZE, GRID_TILE_SIZE));
    return { data: bitmap };
  });
}

async function fullTile(e: Extract<Entry, { kind: "full" }>, z: number, x: number, y: number) {
  const [codes, lut, water] = await Promise.all([loadGrid(e.url), e.lut, waterMask(z, x, y, GRID_TILE_SIZE)]);
  const px = renderGridTile(e.grid, codes, lut, z, x, y, GRID_TILE_SIZE, e.context ? { tint: OUTSIDE_TINT } : {});
  // ocean only, at the basemap's shoreline, or the bundled water if the basemap is unavailable
  // or slow (lib/waterMask); unclipped only if neither can be read
  if (water) clipToWater(px, water);
  return px;
}

async function chunkedTile(e: Extract<Entry, { kind: "chunked" }>, z: number, x: number, y: number) {
  const g = e.grid;
  const b = tileBlock(g, z, x, y);
  if (!b) return new Uint8ClampedArray(GRID_TILE_SIZE * GRID_TILE_SIZE * 4);
  const cg = e.classGrid ?? null;
  const [get, lut, marks0, water, getCls] = await Promise.all([
    chunkedBlock(e.baseUrl, g, b.r0, b.r1, b.c0, b.c1),
    e.lut,
    e.marks,
    waterMask(z, x, y, GRID_TILE_SIZE),
    // the class grid has the value grid's shape: the same block, read only where marks need it
    cg ? chunkedBlock(e.baseUrl, cg, b.r0, b.r1, b.c0, b.c1).catch(() => null) : Promise.resolve(null),
  ]);
  // the block as a small grid of its own, so the one renderer draws it (exact cell edges, nearest)
  const width = b.c1 - b.c0 + 1;
  const height = b.r1 - b.r0 + 1;
  const codes = new Uint16Array(width * height);
  for (let r = 0; r < height; r++) for (let c = 0; c < width; c++) codes[r * width + c] = get(b.r0 + r, b.c0 + c);
  let marks = marks0;
  if (getCls && marks0.strip && cg) {
    const cls = new Uint16Array(width * height);
    const scale = cg.scale_factor || 1;
    const off = cg.add_offset || 0;
    const nod = cg.nodata ?? 65535;
    for (let r = 0; r < height; r++)
      for (let c = 0; c < width; c++) {
        const k = getCls(b.r0 + r, b.c0 + c);
        cls[r * width + c] = k === nod ? 0 : Math.round(k * scale + off);
      }
    marks = { ...marks0, strip: { ...marks0.strip, cls } };
  } else if (marks0.strip) marks = { ...marks0, strip: undefined };
  const lonFirst = g.lon_first > 180 ? g.lon_first - 360 : g.lon_first;
  const sub: ValueGrid = { ...g, chunks: undefined, width, height, lat_first: g.lat_first + b.r0 * g.lat_step, lon_first: lonFirst + b.c0 * g.lon_step };
  const px = renderGridTile(sub, codes, lut, z, x, y, GRID_TILE_SIZE, marks);
  if (water) clipToWater(px, water);
  // the outlook in a bay: never drawn (turbid water; MODIS chlorophyll unreliable there)
  if (e.noBays) clearBays(px, z, x, y, GRID_TILE_SIZE);
  return px;
}

/** The tile URL template for a layer's grid, and the grid's outer cell edges [w, s, e, n]. */
export function gridTiles(layer: LayerArtifact, gridUrl: string, opts: { context?: boolean } = {}): { template: string; bounds: [number, number, number, number] } {
  register();
  const g = layer.grid!;
  const key = `${layer.layer_id}@${gridUrl}${opts.context ? "#context" : ""}`;
  if (!entries.has(key)) entries.set(key, { kind: "full", grid: g, url: gridUrl, lut: Promise.resolve().then(() => codeColours(g, layer.palette!)), context: !!opts.context });
  return { template: `${PROTOCOL}://${encodeURIComponent(key)}/{z}/{x}/{y}`, bounds: boundsOf(g) };
}

/**
 * Tiles of a chunked chlorophyll grid of a derived product, on `palette` (the satellite ramp):
 * "estimate" pixels are lightened and stippled (the shoreline strip of `classGrid`, class 2, paler
 * and more densely stippled), "outlook" pixels carry a fine hatch and are never drawn in a bay.
 */
export function derivedTiles(id: string, g: ValueGrid, baseUrl: string, palette: Palette, style: "estimate" | "outlook" | "observed", classGrid: ValueGrid | null = null): { template: string; bounds: [number, number, number, number] } {
  register();
  const key = `derived:${id}@${baseUrl}/${g.url}#${style}${classGrid ? `+${classGrid.url}` : ""}`;
  if (!entries.has(key)) {
    const lut = Promise.resolve().then(() => rampColours(g, palette));
    const marks = lut.then((l): TileMarks =>
      style === "estimate"
        ? { stipple: { lut: mixLut(l, VEIL, 0.35), dot: mixLut(l, DOT, 0.65) }, ...(classGrid ? { strip: { lut: mixLut(l, VEIL, 0.55), dot: mixLut(l, DOT, 0.55), cls: new Uint16Array(0) } } : {}) }
        : style === "outlook"
          ? { hatch: { line: mixLut(l, DOT, 0.55) } }
          : {},
    );
    entries.set(key, { kind: "chunked", grid: g, baseUrl, lut, marks, classGrid, noBays: style === "outlook" });
  }
  return { template: `${PROTOCOL}://${encodeURIComponent(key)}/{z}/{x}/{y}`, bounds: boundsOf(g) };
}

function boundsOf(g: ValueGrid): [number, number, number, number] {
  const lonFirst = g.lon_first > 180 ? g.lon_first - 360 : g.lon_first;
  const lon0 = Math.min(lonFirst, lonFirst + g.lon_step * (g.width - 1)) - Math.abs(g.lon_step) / 2;
  const lat0 = Math.min(g.lat_first, g.lat_first + g.lat_step * (g.height - 1)) - Math.abs(g.lat_step) / 2;
  return [lon0, lat0, lon0 + Math.abs(g.lon_step) * g.width, lat0 + Math.abs(g.lat_step) * g.height];
}
