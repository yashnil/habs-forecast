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

// ---------------------------------------------------------------- chunked grids (satellite)

/** Decoded value of a raw code, honouring the grid's transform (e.g. log10 for chlorophyll). */
export function decodeCode(g: ValueGrid, code: number): number | null {
  if (code === (g.nodata ?? 65535)) return null;
  const q = code * g.scale_factor + g.add_offset;
  return g.transform === "log10" ? 10 ** q : q;
}

function chunkOf(g: ValueGrid, row: number, col: number) {
  const ch = g.chunks!;
  const r = Math.floor(row / ch.rows);
  const c = Math.floor(col / ch.cols);
  const key = `${r}_${c}`;
  const width = Math.min(ch.cols, g.width - c * ch.cols);
  return { key, r, c, width, localRow: row - r * ch.rows, localCol: col - c * ch.cols };
}

/**
 * Value at a cell of a chunked grid. Only the chunk containing the cell is fetched.
 * Chunks that were not written hold no values anywhere (all cloud, land or outside).
 */
export async function chunkedCode(baseUrl: string, g: ValueGrid, row: number, col: number): Promise<number> {
  if (!g.chunks) throw new Error("grid is not chunked");
  const nodata = g.nodata ?? 65535;
  if (row < 0 || row >= g.height || col < 0 || col >= g.width) return nodata;
  const k = chunkOf(g, row, col);
  if (!g.chunks.present.includes(k.key)) return nodata;
  const url = `${baseUrl.replace(/\/$/, "")}/${g.chunks.url_template.replace("{row}", String(k.r)).replace("{col}", String(k.c))}`;
  const codes = await loadGrid(url);
  return codes[k.localRow * k.width + k.localCol];
}

export type ChunkedSample =
  | { kind: "value"; value: number; cell: Cell; distanceKm: number; nearest: boolean; ageDays: number | null }
  | { kind: "none"; cell: Cell | null };

/**
 * Value at a point from a chunked grid (and, for a composite, the age of that pixel).
 * Like `sample`: with no value at the point, the nearest cell with a value within
 * `searchCells` is reported with its distance, never as the value at the point.
 */
export async function sampleChunked(baseUrl: string, g: ValueGrid, lat: number, lon: number, age?: ValueGrid | null, searchCells = 4): Promise<ChunkedSample> {
  const c = cellAt(g, lat, lon);
  if (!c) return { kind: "none", cell: null };
  let best: { code: number; cell: Cell; d: number } | null = null;
  for (let ring = 0; ring <= searchCells && !best; ring++) {
    for (let dr = -ring; dr <= ring; dr++) {
      for (let dc = -ring; dc <= ring; dc++) {
        if (Math.max(Math.abs(dr), Math.abs(dc)) !== ring) continue;
        const r = c.row + dr;
        const k = c.col + dc;
        const code = await chunkedCode(baseUrl, g, r, k);
        if (code === (g.nodata ?? 65535)) continue;
        const cell = { row: r, col: k, lat: g.lat_first + r * g.lat_step, lon: g.lon_first + k * g.lon_step };
        const d = ring === 0 ? 0 : distanceKm(lat, lon, cell.lat, cell.lon);
        if (!best || d < best.d) best = { code, cell, d };
      }
    }
  }
  if (!best) return { kind: "none", cell: c };
  const ageCode = age ? await chunkedCode(baseUrl, age, best.cell.row, best.cell.col) : null;
  return {
    kind: "value",
    value: decodeCode(g, best.code)!,
    cell: best.cell,
    distanceKm: best.d,
    nearest: best.d > 0,
    ageDays: ageCode == null || ageCode === (age?.nodata ?? 65535) ? null : ageCode,
  };
}

/**
 * Cell edges of a full (unchunked) grid, for drawing the model's own cells (M5): every edge
 * that borders at least one cell with a value, merged into runs. Purely the grid's geometry;
 * no value is drawn or changed. Cells are centred on the lattice, so edges fall half a step
 * either side of each centre.
 */
export function cellEdges(g: ValueGrid, codes: Uint16Array): GeoJSON.FeatureCollection<GeoJSON.MultiLineString> {
  const nodata = g.nodata ?? 65535;
  const ok = (r: number, c: number) => r >= 0 && r < g.height && c >= 0 && c < g.width && codes[r * g.width + c] !== nodata;
  const latE = (r: number) => +(g.lat_first + (r - 0.5) * g.lat_step).toFixed(5); // edge above row r
  const lonE = (c: number) => +(g.lon_first + (c - 0.5) * g.lon_step).toFixed(5); // edge left of col c
  const lines: number[][][] = [];
  // horizontal edges: between row r-1 and row r
  for (let r = 0; r <= g.height; r++) {
    let start = -1;
    for (let c = 0; c <= g.width; c++) {
      const on = c < g.width && (ok(r - 1, c) || ok(r, c));
      if (on && start < 0) start = c;
      if (!on && start >= 0) {
        lines.push([[lonE(start), latE(r)], [lonE(c), latE(r)]]);
        start = -1;
      }
    }
  }
  // vertical edges: between col c-1 and col c
  for (let c = 0; c <= g.width; c++) {
    let start = -1;
    for (let r = 0; r <= g.height; r++) {
      const on = r < g.height && (ok(r, c - 1) || ok(r, c));
      if (on && start < 0) start = r;
      if (!on && start >= 0) {
        lines.push([[lonE(c), latE(start)], [lonE(c), latE(r)]]);
        start = -1;
      }
    }
  }
  return { type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "MultiLineString", coordinates: lines } }] };
}
