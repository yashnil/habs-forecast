import { fetchResource } from "./net";
import { gunzipIfNeeded } from "./store";
import type { GridSpec } from "./types";

/** Point reads from large chunked grids (satellite 300 m): fetches only the 512×512 chunk needed. */
const chunkCache = new Map<string, Promise<{ w: number; h: number; u16: Uint16Array } | null>>();

function chunk(spec: GridSpec, r: number, c: number) {
  const key = `${spec.url}|${r}_${c}`;
  let p = chunkCache.get(key);
  if (!p) {
    const ch = spec.chunks!;
    if (!ch.present.includes(`${r}_${c}`)) p = Promise.resolve(null);
    else {
      const url = `/data/v1/${ch.url_template.replace("{row}", String(r)).replace("{col}", String(c))}`;
      p = fetchResource("satellite", url).then(async (res) => {
        const buf = await gunzipIfNeeded(await res.arrayBuffer());
        const h = Math.min(ch.rows, spec.height - r * ch.rows);
        const w = Math.min(ch.cols, spec.width - c * ch.cols);
        return { w, h, u16: new Uint16Array(buf) };
      });
    }
    chunkCache.set(key, p);
  }
  return p;
}

export interface PointRead { inside: boolean; value: number | null; row: number; col: number; center: [number, number]; bounds: [number, number, number, number] }

export async function readPoint(spec: GridSpec, lon: number, lat: number): Promise<PointRead> {
  const col = Math.round((lon - spec.lon_first) / spec.lon_step);
  const row = Math.round((lat - spec.lat_first) / spec.lat_step);
  const clon = spec.lon_first + col * spec.lon_step, clat = spec.lat_first + row * spec.lat_step;
  const hx = Math.abs(spec.lon_step) / 2, hy = Math.abs(spec.lat_step) / 2;
  const base = { row, col, center: [clon, clat] as [number, number], bounds: [clon - hx, clat - hy, clon + hx, clat + hy] as [number, number, number, number] };
  if (col < 0 || row < 0 || col >= spec.width || row >= spec.height) return { inside: false, value: null, ...base };
  const ch = spec.chunks!;
  const c = await chunk(spec, Math.floor(row / ch.rows), Math.floor(col / ch.cols));
  if (!c) return { inside: true, value: null, ...base };
  const raw = c.u16[(row % ch.rows) * c.w + (col % ch.cols)];
  if (raw === spec.nodata) return { inside: true, value: null, ...base };
  const q = raw * spec.scale_factor + spec.add_offset;
  return { inside: true, value: spec.transform === "log10" ? 10 ** q : q, ...base };
}

/** Nearest valued pixel within `maxPx` pixels (square rings), for "nearest clear view". */
export async function nearestPoint(spec: GridSpec, lon: number, lat: number, maxPx = 60): Promise<PointRead | null> {
  const p0 = await readPoint(spec, lon, lat);
  if (p0.value != null || !p0.inside) return p0.value != null ? p0 : null;
  for (let k = 2; k <= maxPx; k += 2) {
    const hits: PointRead[] = [];
    const reads: Promise<PointRead>[] = [];
    for (let d = -k; d <= k; d += 2) {
      for (const [dr, dc] of [[-k, d], [k, d], [d, -k], [d, k]]) {
        reads.push(readPoint(spec, spec.lon_first + (p0.col + dc) * spec.lon_step, spec.lat_first + (p0.row + dr) * spec.lat_step));
      }
    }
    for (const r of await Promise.all(reads)) if (r.value != null) hits.push(r);
    if (hits.length) {
      const cos = Math.cos((lat * Math.PI) / 180);
      hits.sort((a, b) => ((a.center[0] - lon) * cos) ** 2 + (a.center[1] - lat) ** 2 - (((b.center[0] - lon) * cos) ** 2 + (b.center[1] - lat) ** 2));
      return hits[0];
    }
  }
  return null;
}
