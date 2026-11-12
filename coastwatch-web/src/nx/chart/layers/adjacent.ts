/**
 * The band beside land that the pipeline leaves out of the 300 m chlorophyll (QC check `land_adjacency`), marked
 * for drawing. Those cells have no value, as cloud has none, but they are not cloud: the layer draws them with
 * their own mark (FieldLayer `u_adj`), matching the legend's "Left out: beside land" key.
 *
 * PURE. The published codes are never changed: a COPY is returned in which every cell that is both without a
 * value and in the left-out reference carries a marker code. A cell with a value is never marked (the reference
 * says where the rule applies, the data says whether a value was published). If the marker code is already used
 * by the data, nothing is marked: the band is then drawn as "no view", the weaker true statement.
 */
import type { FieldGrid } from "./grid";

export type AdjacentMask = {
  lat_first: number;
  lon_first: number;
  /** degrees per cell; rows run north to south, columns west to east */
  step: number;
  height: number;
  width: number;
  /** per row, runs of left-out cells: [first column, length] */
  rows_rle: [number, number][][];
};

export type Marked = { codes: Uint16Array; adjacentCode: number | null };

export function markAdjacent(g: FieldGrid, codes: Uint16Array, mask: AdjacentMask): Marked {
  const marker = g.nodata > 0 ? g.nodata - 1 : g.nodata + 1;
  for (let i = 0; i < codes.length; i++) if (codes[i] === marker) return { codes, adjacentCode: null };
  let out: Uint16Array | null = null;
  // only the mask's rows and columns that fall inside the grid
  const latA = g.lat_first, latB = g.lat_first + (g.height - 1) * g.lat_step;
  const north = Math.max(latA, latB) + Math.abs(g.lat_step) / 2, south = Math.min(latA, latB) - Math.abs(g.lat_step) / 2;
  const r0 = Math.max(0, Math.ceil((mask.lat_first - north) / mask.step)), r1 = Math.min(mask.height - 1, Math.floor((mask.lat_first - south) / mask.step));
  for (let r = r0; r <= r1; r++) {
    const runs = mask.rows_rle[r];
    if (!runs?.length) continue;
    const row = Math.floor((mask.lat_first - r * mask.step - g.lat_first) / g.lat_step + 0.5);
    if (row < 0 || row >= g.height) continue;
    for (const [start, len] of runs) {
      for (let c = start; c < start + len; c++) {
        const col = Math.floor((mask.lon_first + c * mask.step - g.lon_first) / g.lon_step + 0.5);
        if (col < 0 || col >= g.width) continue;
        const k = row * g.width + col;
        if (codes[k] !== g.nodata) continue;
        out ??= codes.slice();
        out[k] = marker;
      }
    }
  }
  return out ? { codes: out, adjacentCode: marker } : { codes, adjacentCode: null };
}
