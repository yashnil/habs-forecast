import type { ValueGrid } from "@/generated/schema";

/** Decode a published value grid (uint16 little-endian, gzip). */
export async function decodeGrid(gz: ArrayBuffer): Promise<Uint16Array> {
  const stream = new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"));
  const buf = await new Response(stream).arrayBuffer();
  const view = new DataView(buf);
  const out = new Uint16Array(buf.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = view.getUint16(i * 2, true);
  return out;
}

const cache = new Map<string, Promise<Uint16Array>>();

export function loadGrid(url: string): Promise<Uint16Array> {
  let p = cache.get(url);
  if (!p) {
    p = fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
        return r.arrayBuffer();
      })
      .then(decodeGrid);
    p.catch(() => cache.delete(url));
    cache.set(url, p);
  }
  return p;
}

export type Cell = { row: number; col: number; lat: number; lon: number };

export function cellAt(g: ValueGrid, lat: number, lon: number): Cell | null {
  const row = Math.floor((lat - g.lat_first) / g.lat_step + 0.5);
  const col = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
  if (row < 0 || row >= g.height || col < 0 || col >= g.width) return null;
  return { row, col, lat: g.lat_first + row * g.lat_step, lon: g.lon_first + col * g.lon_step };
}

export function valueOf(g: ValueGrid, codes: Uint16Array, c: Cell): number | null {
  const code = codes[c.row * g.width + c.col];
  if (code === g.nodata) return null;
  return code * g.scale_factor + g.add_offset;
}

const KM_PER_DEG = 111.32;

export function distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const dy = (lat2 - lat1) * KM_PER_DEG;
  const dx = (lon2 - lon1) * KM_PER_DEG * Math.cos(((lat1 + lat2) / 2) * (Math.PI / 180));
  return Math.hypot(dx, dy);
}

export type Sample =
  | { kind: "value"; value: number; cell: Cell; distanceKm: number; nearest: boolean }
  | { kind: "none"; cell: Cell | null };

/**
 * Value at a point. Where the forecast has no value (land, or nearshore cells the
 * producer masks), report the nearest cell with a value within `searchCells` and say
 * how far away it is — never present it as the value at the point itself.
 */
export function sample(g: ValueGrid, codes: Uint16Array, lat: number, lon: number, searchCells = 3): Sample {
  const c = cellAt(g, lat, lon);
  if (!c) return { kind: "none", cell: null };
  const v = valueOf(g, codes, c);
  if (v !== null) return { kind: "value", value: v, cell: c, distanceKm: 0, nearest: false };
  let best: { value: number; cell: Cell; d: number } | null = null;
  for (let dr = -searchCells; dr <= searchCells; dr++) {
    for (let dc = -searchCells; dc <= searchCells; dc++) {
      const r = c.row + dr;
      const k = c.col + dc;
      if (r < 0 || r >= g.height || k < 0 || k >= g.width) continue;
      const cell = { row: r, col: k, lat: g.lat_first + r * g.lat_step, lon: g.lon_first + k * g.lon_step };
      const val = valueOf(g, codes, cell);
      if (val === null) continue;
      const d = distanceKm(lat, lon, cell.lat, cell.lon);
      if (!best || d < best.d) best = { value: val, cell, d };
    }
  }
  if (!best) return { kind: "none", cell: c };
  return { kind: "value", value: best.value, cell: best.cell, distanceKm: best.d, nearest: true };
}
