/**
 * The bundled water geometry (public/basemap/water-fallback.json, built by
 * pipeline/scripts/build_water_fallback.py): the basemap's own OpenMapTiles water over the C-HARM
 * domain, simplified to 8 m within one cell of a C-HARM ocean cell and 40 m elsewhere.
 *
 * It stands in for the basemap's vector tiles when they cannot be read (lib/waterMask), so data
 * rasters are still drawn over water only, and it fills the sea while the basemap is unavailable.
 * Pure: decoding and rasterising here have no browser dependencies, so they are deterministic and
 * unit tested.
 */
export const WATER_FALLBACK_URL = "/basemap/water-fallback.json";

type FallbackFile = { quant: number; domain_lnglat: [number, number, number, number]; polygons: number[][][] };

/** Rings in normalised Web Mercator (0..1, y down), each with its bounding box. */
export type WaterGeometry = {
  rings: Float64Array[];
  /** per ring: minX, minY, maxX, maxY */
  bbox: Float64Array;
  /** ring indices of each polygon: outer ring first, then holes */
  polygons: number[][];
  domain: [number, number, number, number];
};

export function decodeWaterFallback(f: FallbackFile): WaterGeometry {
  const rings: Float64Array[] = [];
  const polygons: number[][] = [];
  for (const poly of f.polygons) {
    const ids: number[] = [];
    for (const enc of poly) {
      const r = new Float64Array(enc.length);
      let x = 0, y = 0;
      for (let i = 0; i < enc.length; i += 2) {
        x += enc[i];
        y += enc[i + 1];
        r[i] = x / f.quant;
        r[i + 1] = y / f.quant;
      }
      ids.push(rings.length);
      rings.push(r);
    }
    polygons.push(ids);
  }
  const bbox = new Float64Array(rings.length * 4);
  rings.forEach((r, k) => {
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (let i = 0; i < r.length; i += 2) {
      x0 = Math.min(x0, r[i]);
      x1 = Math.max(x1, r[i]);
      y0 = Math.min(y0, r[i + 1]);
      y1 = Math.max(y1, r[i + 1]);
    }
    bbox.set([x0, y0, x1, y1], k * 4);
  });
  return { rings, bbox, polygons, domain: f.domain_lnglat };
}

let loading: Promise<WaterGeometry | null> | null = null;

/** The bundled water, fetched once on first need; null if it cannot be loaded (retried next time). */
export function loadWaterFallback(): Promise<WaterGeometry | null> {
  loading ??= fetch(WATER_FALLBACK_URL)
    .then((r) => (r.ok ? r.json() : null))
    .then((j: FallbackFile | null) => (j ? decodeWaterFallback(j) : null))
    .catch(() => null)
    .then((g) => {
      if (!g) loading = null;
      return g;
    });
  return loading;
}

const SUB = 4; // sample lines per pixel row; horizontal coverage is exact

/**
 * Water coverage of tile z/x/y as a size x size alpha array (255 = all water), from the bundled
 * geometry: even-odd scanline fill with 4 sample lines per pixel row and exact horizontal
 * coverage, so the shoreline is anti-aliased like the basemap's. Outside the geometry's domain
 * (the map's pannable area) the result is water (255): unknown, so data there are left unclipped.
 */
export function rasterWater(g: WaterGeometry, z: number, x: number, y: number, size: number): Uint8ClampedArray {
  const k = 2 ** z;
  // tile bounds in normalised Mercator, and the scale to tile pixels
  const tx0 = x / k, ty0 = y / k, tx1 = (x + 1) / k, ty1 = (y + 1) / k;
  const s = k * size;
  // edges as (yTop, yBottom, x at yTop, dx/dy) in tile pixels, only those crossing the tile's rows
  const ey0: number[] = [], ey1: number[] = [], ex: number[] = [], edx: number[] = [];
  for (let r = 0; r < g.rings.length; r++) {
    const b = r * 4;
    // a ring wholly beside the tile crosses each pixel row an even number of times on that
    // side, so it cannot change the even-odd parity inside the tile
    if (g.bbox[b + 1] > ty1 || g.bbox[b + 3] < ty0 || g.bbox[b] > tx1 || g.bbox[b + 2] < tx0) continue;
    const ring = g.rings[r];
    const n = ring.length;
    for (let i = 0; i < n; i += 2) {
      const j = (i + 2) % n;
      let ax = (ring[i] - tx0) * s, ay = (ring[i + 1] - ty0) * s, bx = (ring[j] - tx0) * s, by = (ring[j + 1] - ty0) * s;
      if (ay === by) continue;
      if (ay > by) [ax, ay, bx, by] = [bx, by, ax, ay];
      if (by <= 0 || ay >= size) continue;
      ey0.push(ay);
      ey1.push(by);
      ex.push(ax);
      edx.push((bx - ax) / (by - ay));
    }
  }
  const out = new Uint8ClampedArray(size * size);
  const n = ey0.length;
  if (!n) return outsideDomainIsWater(g, out, tx0, ty0, s, size);
  // bucket edges by the first sample line they cover; sample line l is at y = (l + 0.5) / SUB
  const first = (y0: number) => Math.max(0, Math.ceil(y0 * SUB - 0.5));
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => ey0[a] - ey0[b]);
  const active: number[] = [];
  const xs: number[] = [];
  const cover = new Float64Array(size + 1);
  const full = new Float64Array(size + 1);
  let next = 0;
  for (let row = 0; row < size; row++) {
    cover.fill(0);
    full.fill(0);
    for (let sub = 0; sub < SUB; sub++) {
      const l = row * SUB + sub;
      const yl = (l + 0.5) / SUB;
      while (next < n && first(ey0[order[next]]) <= l) active.push(order[next++]);
      xs.length = 0;
      for (let i = active.length - 1; i >= 0; i--) {
        const e = active[i];
        if (ey1[e] <= yl) {
          active[i] = active[active.length - 1];
          active.pop();
        } else if (ey0[e] <= yl) xs.push(ex[e] + (yl - ey0[e]) * edx[e]);
      }
      xs.sort((a, b) => a - b);
      for (let i = 0; i + 1 < xs.length; i += 2) {
        const a = Math.max(0, xs[i]), b = Math.min(size, xs[i + 1]);
        if (b <= a) continue;
        const ia = Math.floor(a), ib = Math.floor(b);
        if (ia === ib) cover[ia] += b - a;
        else {
          cover[ia] += ia + 1 - a;
          full[ia + 1] += 1;
          full[ib] -= 1;
          if (ib < size) cover[ib] += b - ib;
        }
      }
    }
    let run = 0;
    for (let c = 0; c < size; c++) {
      run += full[c];
      out[row * size + c] = Math.round(((run + cover[c]) / SUB) * 255);
    }
    if (next >= n && !active.length) break;
  }
  outsideDomainIsWater(g, out, tx0, ty0, s, size);
  return out;
}

function outsideDomainIsWater(g: WaterGeometry, out: Uint8ClampedArray, tx0: number, ty0: number, s: number, size: number) {
  const [w, so, e, n] = g.domain;
  const px = (lng: number) => ((lng + 180) / 360 - tx0) * s;
  const py = (lat: number) => (0.5 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) / (2 * Math.PI) - ty0) * s;
  const [c0, c1, r0, r1] = [px(w), px(e), py(n), py(so)];
  if (c0 <= 0 && c1 >= size && r0 <= 0 && r1 >= size) return out;
  for (let r = 0; r < size; r++)
    for (let c = 0; c < size; c++) if (c + 0.5 < c0 || c + 0.5 > c1 || r + 0.5 < r0 || r + 0.5 > r1) out[r * size + c] = 255;
  return out;
}

/** The bundled water as GeoJSON (longitude, latitude), for the sea drawn while the basemap is unavailable. */
export function waterGeoJSON(g: WaterGeometry): GeoJSON.FeatureCollection {
  const lng = (x: number) => x * 360 - 180;
  const lat = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI;
  return {
    type: "FeatureCollection",
    features: [
      {
        type: "Feature",
        properties: {},
        geometry: {
          type: "MultiPolygon",
          coordinates: g.polygons.map((ids) =>
            ids.map((id) => {
              const r = g.rings[id];
              const pts: number[][] = [];
              for (let i = 0; i < r.length; i += 2) pts.push([lng(r[i]), lat(r[i + 1])]);
              pts.push(pts[0]);
              return pts;
            }),
          ),
        },
      },
    ],
  };
}
