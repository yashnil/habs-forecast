import type { ValueGrid } from "@/generated/schema";
import { cellAt, distanceKm, valueOf, type Cell } from "@/lib/grid";

export type TransectCell = { km: number; row: number; col: number; lat: number; lon: number; value: number | null; inDomain: boolean };

/**
 * The published cells a straight line crosses, in order, with the distance along the line to each
 * cell's first crossing. The walk is exact: the line is cut at every cell edge it crosses (edges lie
 * half a step either side of each centre, as `cellAt` assigns them), so a cell the line clips only at a
 * corner is listed too; a line that merely touches a corner point crosses nothing there. Nothing is
 * interpolated: a cell without a value (land, nearshore, masked) stays null and is drawn as a gap; a
 * stretch outside the grid is one entry marked outside the domain.
 */
export function transect(g: ValueGrid, codes: Uint16Array, a: { lat: number; lon: number }, b: { lat: number; lon: number }): TransectCell[] {
  return transectWith(g, (c) => valueOf(g, codes, c), a, b);
}

/** The same exact walk over any lattice, with the value of each crossed cell read by `valueAt`
 *  (satellite pixels from their chunks, radar cells from the decoded field). */
export function transectWith(g: Pick<ValueGrid, "lat_first" | "lat_step" | "lon_first" | "lon_step" | "width" | "height">, valueAt: (c: Cell) => number | null, a: { lat: number; lon: number }, b: { lat: number; lon: number }): TransectCell[] {
  const total = distanceKm(a.lat, a.lon, b.lat, b.lon);
  // positions in cell units, shifted so that cell k spans [k, k + 1)
  const u = (lat: number) => (lat - g.lat_first) / g.lat_step + 0.5;
  const v = (lon: number) => (lon - g.lon_first) / g.lon_step + 0.5;
  const cuts = new Set<number>([0, 1]);
  for (const [p, q] of [
    [u(a.lat), u(b.lat)],
    [v(a.lon), v(b.lon)],
  ]) {
    if (p === q) continue;
    for (let k = Math.ceil(Math.min(p, q)); k <= Math.floor(Math.max(p, q)); k++) {
      const f = (k - p) / (q - p);
      if (f > 0 && f < 1) cuts.add(f);
    }
  }
  const fs = [...cuts].sort((x, y) => x - y);
  const out: TransectCell[] = [];
  for (let i = 0; i < fs.length - 1; i++) {
    const f0 = fs[i];
    const f1 = fs[i + 1];
    if (f1 - f0 < 1e-12) continue;
    // the middle of each piece lies strictly inside one cell
    const fm = (f0 + f1) / 2;
    const c = cellAt(g as ValueGrid, a.lat + (b.lat - a.lat) * fm, a.lon + (b.lon - a.lon) * fm);
    const last = out[out.length - 1];
    if (c && last && last.row === c.row && last.col === c.col) continue;
    if (!c && last && !last.inDomain) continue;
    const lat = a.lat + (b.lat - a.lat) * f0;
    const lon = a.lon + (b.lon - a.lon) * f0;
    out.push(c ? { km: total * f0, row: c.row, col: c.col, lat: c.lat, lon: c.lon, value: valueAt(c), inDomain: true } : { km: total * f0, row: -1, col: -1, lat, lon, value: null, inDomain: false });
  }
  return out;
}
