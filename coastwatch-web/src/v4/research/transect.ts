import type { ValueGrid } from "@/generated/schema";
import { cellAt, distanceKm, valueOf } from "@/lib/grid";

export type TransectCell = { km: number; row: number; col: number; lat: number; lon: number; value: number | null; inDomain: boolean };

/**
 * The published cells a straight line crosses, in order, with the distance along the line to each
 * cell's first crossing. Nothing is interpolated: a cell without a value (land, nearshore, masked)
 * stays null and is drawn as a gap; a point outside the grid is marked outside the domain.
 */
export function transect(g: ValueGrid, codes: Uint16Array, a: { lat: number; lon: number }, b: { lat: number; lon: number }): TransectCell[] {
  const total = distanceKm(a.lat, a.lon, b.lat, b.lon);
  const stepKm = Math.max(0.2, (Math.abs(g.lat_step) * 111.32) / 6);
  const n = Math.max(2, Math.ceil(total / stepKm));
  const out: TransectCell[] = [];
  for (let i = 0; i <= n; i++) {
    const f = i / n;
    const lat = a.lat + (b.lat - a.lat) * f;
    const lon = a.lon + (b.lon - a.lon) * f;
    const c = cellAt(g, lat, lon);
    const last = out[out.length - 1];
    if (c && last && last.row === c.row && last.col === c.col) continue;
    if (!c && last && !last.inDomain) continue;
    out.push(c ? { km: total * f, row: c.row, col: c.col, lat: c.lat, lon: c.lon, value: valueOf(g, codes, c), inDomain: true } : { km: total * f, row: -1, col: -1, lat, lon, value: null, inDomain: false });
  }
  return out;
}
