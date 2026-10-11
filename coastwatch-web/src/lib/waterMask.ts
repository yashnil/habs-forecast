import { VectorTile } from "@mapbox/vector-tile";
import { PbfReader } from "pbf";

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
 * If the basemap tile cannot be read, the data tile is drawn unclipped (data is never hidden
 * because of a basemap failure).
 */
const TILEJSON = "https://tiles.openfreemap.org/planet";
const MAX_Z = 14; // the vector tiles' maximum zoom
let template: Promise<string | null> | null = null;
const masks = new Map<string, Promise<Uint8ClampedArray | null>>();
const MAX_CACHED = 256;

function tileTemplate(): Promise<string | null> {
  template ??= fetch(TILEJSON)
    .then((r) => (r.ok ? r.json() : null))
    .then((j: { tiles?: string[] } | null) => j?.tiles?.[0] ?? null)
    .catch(() => null);
  return template;
}

/** Water coverage of tile z/x/y as a size x size alpha array (255 = all water), or null. */
export function waterMask(z: number, x: number, y: number, size: number): Promise<Uint8ClampedArray | null> {
  const key = `${z}/${x}/${y}@${size}`;
  let p = masks.get(key);
  if (!p) {
    p = build(z, x, y, size).catch(() => null);
    masks.set(key, p);
    if (masks.size > MAX_CACHED) masks.delete(masks.keys().next().value!);
  }
  return p;
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
