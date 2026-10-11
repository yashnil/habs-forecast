import { addProtocol } from "maplibre-gl";
import type { LayerArtifact, ValueGrid } from "@/generated/schema";
import { loadGrid } from "@/lib/grid";
import { codeColours, GRID_TILE_SIZE, renderGridTile } from "@/lib/gridTiles";
import { clipToWater, waterMask } from "@/lib/waterMask";

/**
 * The `cwgrid://` tile protocol (see lib/gridTiles). Kept apart from the pure helpers so that
 * only the lazily loaded map canvas imports MapLibre.
 */
const PROTOCOL = "cwgrid";
type Entry = { grid: ValueGrid; url: string; lut: Promise<Uint32Array> };
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
