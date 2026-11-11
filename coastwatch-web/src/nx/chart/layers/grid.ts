/**
 * A published value grid's geometry and decoding, DOM-free (the layers draw from it; tests check it). A grid is
 * a regular lat/lon lattice of cell centres: row 0 at `lat_first`, column 0 at `lon_first`, steps of either
 * sign; a raw uint16 code decodes as `code * scale_factor + add_offset` (then 10^x for a log10 grid), and the
 * `nodata` code is no value, never zero.
 */
export type FieldGrid = {
  width: number;
  height: number;
  lat_first: number;
  lat_step: number;
  lon_first: number;
  lon_step: number;
  scale_factor: number;
  add_offset: number;
  nodata: number;
  transform: "log10" | "none";
};

export const mercX = (lon: number) => (lon + 180) / 360;
export const mercY = (lat: number) => {
  const s = Math.sin((lat * Math.PI) / 180);
  return 0.5 - (0.25 * Math.log((1 + s) / (1 - s))) / Math.PI;
};

/** The grid's outer edges [west, south, east, north]: half a cell beyond the outermost centres. */
export function gridBounds(g: Pick<FieldGrid, "width" | "height" | "lat_first" | "lat_step" | "lon_first" | "lon_step">): [number, number, number, number] {
  const lonA = g.lon_first - g.lon_step / 2, lonB = g.lon_first + (g.width - 0.5) * g.lon_step;
  const latA = g.lat_first - g.lat_step / 2, latB = g.lat_first + (g.height - 0.5) * g.lat_step;
  return [Math.min(lonA, lonB), Math.min(latA, latB), Math.max(lonA, lonB), Math.max(latA, latB)];
}

/** The cell whose centre is nearest a point, or null outside the grid. */
export function cellOf(g: FieldGrid, lat: number, lon: number): { row: number; col: number } | null {
  const row = Math.floor((lat - g.lat_first) / g.lat_step + 0.5);
  const col = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
  if (row < 0 || col < 0 || row >= g.height || col >= g.width) return null;
  return { row, col };
}

export const cellCentre = (g: FieldGrid, row: number, col: number) => ({ lat: g.lat_first + row * g.lat_step, lon: g.lon_first + col * g.lon_step });

/** A raw code as its value (log10 grids back to linear units), or null for no data. */
export function decodeValue(g: Pick<FieldGrid, "scale_factor" | "add_offset" | "nodata" | "transform">, code: number): number | null {
  if (code === g.nodata) return null;
  const q = code * g.scale_factor + g.add_offset;
  return g.transform === "log10" ? 10 ** q : q;
}

/**
 * The edges between observed and unobserved cells (or the grid's border), as segments [lon0, lat0, lon1, lat1]:
 * the real edge of coverage, drawn as a hairline so "no streaks" reads as "no observation" (D §6.4).
 */
export function coverageEdges(g: Pick<FieldGrid, "width" | "height" | "lat_first" | "lat_step" | "lon_first" | "lon_step">, observed: (k: number) => boolean): [number, number, number, number][] {
  const out: [number, number, number, number][] = [];
  const on = (r: number, c: number) => r >= 0 && c >= 0 && r < g.height && c < g.width && observed(r * g.width + c);
  const hx = g.lon_step / 2, hy = g.lat_step / 2;
  for (let r = 0; r < g.height; r++)
    for (let c = 0; c < g.width; c++) {
      if (!on(r, c)) continue;
      const lat = g.lat_first + r * g.lat_step, lon = g.lon_first + c * g.lon_step;
      if (!on(r, c - 1)) out.push([lon - hx, lat - hy, lon - hx, lat + hy]);
      if (!on(r, c + 1)) out.push([lon + hx, lat - hy, lon + hx, lat + hy]);
      if (!on(r - 1, c)) out.push([lon - hx, lat - hy, lon + hx, lat - hy]);
      if (!on(r + 1, c)) out.push([lon - hx, lat + hy, lon + hx, lat + hy]);
    }
  return out;
}

/**
 * Each cell's share of the hours it was measured in (0..1), from the hours' observed masks: how steadily the
 * radar saw it. Streaks are drawn fainter where a cell was measured in fewer hours (D §6.3: opacity from data
 * coverage, the quality signal the published data carries).
 */
export function persistence(masks: Uint8Array[], n: number): Float32Array {
  const out = new Float32Array(n);
  if (!masks.length) return out.fill(1);
  for (const m of masks) for (let k = 0; k < n; k++) out[k] += m[k] ? 1 : 0;
  for (let k = 0; k < n; k++) out[k] /= masks.length;
  return out;
}
