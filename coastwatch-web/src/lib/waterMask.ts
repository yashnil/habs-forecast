import { VectorTile } from "@mapbox/vector-tile";
import { PbfReader } from "pbf";
import { loadWaterFallback, rasterWater } from "@/lib/waterFallback";

/**
 * Ocean-only clipping for data rasters, at the basemap's own shoreline.
 *
 * Data rasters (C-HARM at 0.03°, satellite products) are whole cells, so a cell that straddles
 * the shore would paint over land. For each data tile we read the basemap's vector tile for the
 * same z/x/y (OpenFreeMap, OpenMapTiles schema; the browser has usually fetched it already for
 * the basemap), rasterise its `water` polygons at the data tile's resolution, and keep data only
 * where there is water: a pixel's opacity is the fraction of it that is water, so the edge is
 * anti-aliased exactly where the basemap draws the shore. Values are never changed; cells with no
 * value stay transparent and keep their no-data hatch (which is drawn on the same water).
 *
 * If the basemap tile cannot be read (tile server down or offline, a bad tile, no OffscreenCanvas)
 * or has not arrived after BASEMAP_WAIT_MS (a slow network), the same water comes from the bundled
 * geometry (lib/waterFallback): the basemap's OpenMapTiles water, simplified to 8 m near the
 * forecast, so data stay off land. Only if that cannot be loaded either is the tile drawn
 * unclipped (data are never hidden because of a basemap failure).
 */
const TILEJSON = "https://tiles.openfreemap.org/planet";
const MAX_Z = 14; // the vector tiles' maximum zoom
let template: Promise<string | null> | null = null;
const masks = new Map<string, Promise<Uint8ClampedArray | null>>();
const fallbacks = new Map<string, Uint8ClampedArray>();
const MAX_CACHED = 256;
/** How long a data tile waits for the basemap's water before using the bundled water. */
export const BASEMAP_WAIT_MS = 3000;

function tileTemplate(): Promise<string | null> {
  template ??= fetch(TILEJSON)
    .then((r) => (r.ok ? r.json() : null))
    .then((j: { tiles?: string[] } | null) => j?.tiles?.[0] ?? null)
    .catch(() => null)
    .then((t) => {
      if (!t) template = null; // asked again for the next tile
      return t;
    });
  return template;
}

function remember<T>(cache: Map<string, T>, key: string, v: T): T {
  cache.set(key, v);
  if (cache.size > MAX_CACHED) cache.delete(cache.keys().next().value!);
  return v;
}

/**
 * Water coverage of tile z/x/y as a size x size alpha array (255 = all water): the basemap's own
 * water, or the bundled water when the basemap's is unavailable or slow; null only if neither can
 * be read.
 */
export async function waterMask(z: number, x: number, y: number, size: number): Promise<Uint8ClampedArray | null> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const late = Symbol();
  const slow = new Promise<typeof late>((res) => (timer = setTimeout(() => res(late), BASEMAP_WAIT_MS)));
  const m = await Promise.race([basemapWater(z, x, y, size), slow]);
  clearTimeout(timer);
  if (m === late) slowListeners.forEach((f) => f());
  return m && m !== late ? m : fallbackWater(z, x, y, size);
}

const slowListeners = new Set<() => void>();
/** Called whenever a data tile stops waiting for the basemap's water (MapCanvas then draws the
 *  bundled sea until the basemap's tiles arrive). Returns the unsubscribe function. */
export function onBasemapSlow(f: () => void): () => void {
  slowListeners.add(f);
  return () => slowListeners.delete(f);
}

/** The bundled water for tile z/x/y, or null if it cannot be loaded. */
export async function fallbackWater(z: number, x: number, y: number, size: number): Promise<Uint8ClampedArray | null> {
  const key = `${z}/${x}/${y}@${size}`;
  const hit = fallbacks.get(key);
  if (hit) return hit;
  const g = await loadWaterFallback();
  return g ? remember(fallbacks, key, rasterWater(g, z, x, y, size)) : null;
}

/** The basemap's water for tile z/x/y (cached), or null if its vector tile cannot be read. */
export function basemapWater(z: number, x: number, y: number, size: number): Promise<Uint8ClampedArray | null> {
  const key = `${z}/${x}/${y}@${size}`;
  return masks.get(key) ?? remember(masks, key, build(z, x, y, size).catch(() => null));
}

async function build(z: number, x: number, y: number, size: number): Promise<Uint8ClampedArray | null> {
  if (typeof OffscreenCanvas === "undefined") return null;
  const t = await tileTemplate();
  if (!t) return null;
  // above the vector tiles' maximum zoom, use the ancestor tile and the matching quarter of it
  const dz = Math.max(0, z - MAX_Z);
  const [tz, tx, ty] = [z - dz, x >> dz, y >> dz];
  const r = await fetch(t.replace("{z}", String(tz)).replace("{x}", String(tx)).replace("{y}", String(ty)));
  if (!r.ok) return null;
  const vt = new VectorTile(new PbfReader(new Uint8Array(await r.arrayBuffer())));
  const c = new OffscreenCanvas(size, size);
  const g = c.getContext("2d");
  if (!g) return null;
  const water = vt.layers.water;
  if (water) {
    const k = 2 ** dz;
    const ox = (x - (tx << dz)) * (water.extent / k);
    const oy = (y - (ty << dz)) * (water.extent / k);
    const s = (size * k) / water.extent;
    g.fillStyle = "#fff";
    for (let i = 0; i < water.length; i++) {
      const f = water.feature(i);
      if (f.type !== 3) continue; // polygons only
      g.beginPath();
      for (const ring of f.loadGeometry()) {
        ring.forEach((pt, j) => (j ? g.lineTo((pt.x - ox) * s, (pt.y - oy) * s) : g.moveTo((pt.x - ox) * s, (pt.y - oy) * s)));
        g.closePath();
      }
      // vector-tile rings: exterior and holes wind in opposite directions
      g.fill("nonzero");
    }
  }
  const d = g.getImageData(0, 0, size, size).data;
  const a = new Uint8ClampedArray(size * size);
  for (let i = 0; i < a.length; i++) a[i] = d[i * 4 + 3];
  return a;
}

/** Keep RGBA pixels only over water: alpha x water coverage. In place. */
export function clipToWater(rgba: Uint8ClampedArray, mask: Uint8ClampedArray): Uint8ClampedArray {
  for (let i = 0, j = 3; i < mask.length; i++, j += 4) if (mask[i] !== 255 && rgba[j] !== 0) rgba[j] = (rgba[j] * mask[i] + 127) / 255;
  return rgba;
}
