import type { ValueGrid } from "@/generated/schema";

export type AreaCell = { row: number; col: number; lat: number; lon: number; value: number | null };

/**
 * Every grid cell whose centre lies in the box (west, south, east, north), row by row, with its
 * published value or null. Cells without a value are listed and counted, never filled.
 */
export function areaCells(read: { grid: ValueGrid; codes: Uint16Array } | null, bbox: [number, number, number, number]): AreaCell[] | null {
  if (!read) return null;
  const g = read.grid;
  const [w, s, e, n] = bbox;
  const out: AreaCell[] = [];
  for (let row = 0; row < g.height; row++) {
    const lat = g.lat_first + row * g.lat_step;
    if (lat < s || lat > n) continue;
    for (let col = 0; col < g.width; col++) {
      const lon = g.lon_first + col * g.lon_step;
      if (lon < w || lon > e) continue;
      const c = read.codes[row * g.width + col];
      out.push({ row, col, lat, lon, value: c === g.nodata ? null : c * g.scale_factor + g.add_offset });
    }
  }
  return out;
}

/** Counts and the spread (lowest, median, highest) of the published values in the box. */
export function areaStats(read: { grid: ValueGrid; codes: Uint16Array } | null, bbox: [number, number, number, number]) {
  const cells = areaCells(read, bbox);
  if (!cells) return null;
  const vals = cells.flatMap((c) => (c.value == null ? [] : [c.value])).sort((a, b) => a - b);
  const q = (p: number) => vals[Math.min(vals.length - 1, Math.floor(p * (vals.length - 1)))];
  return { domain: cells.length, valued: vals.length, min: vals[0] ?? null, median: vals.length ? q(0.5) : null, max: vals[vals.length - 1] ?? null };
}
