import type { GridSpec } from "./model";

/** A chunked grid smaller than this is fetched whole; a larger one only by the chunks in view. */
export const WHOLE_LIMIT = 4_000_000;

/** The chunk keys of a chunked grid that cover a box [w, s, e, n] (all present chunks without one). */
export function chunksFor(g: GridSpec, box: [number, number, number, number] | null): string[] {
  const ch = g.chunks;
  if (!ch) return [];
  if (!box || g.width * g.height <= WHOLE_LIMIT) return ch.present;
  const [w, s, e, n] = box;
  const colOf = (lon: number) => Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
  const rowOf = (lat: number) => Math.floor((lat - g.lat_first) / g.lat_step + 0.5);
  const [c0, c1] = [colOf(w), colOf(e)].sort((a, b) => a - b);
  const [r0, r1] = [rowOf(s), rowOf(n)].sort((a, b) => a - b);
  const out: string[] = [];
  for (let r = Math.max(0, Math.floor(r0 / ch.rows)); r <= Math.min(Math.ceil(g.height / ch.rows) - 1, Math.floor(r1 / ch.rows)); r++)
    for (let c = Math.max(0, Math.floor(c0 / ch.cols)); c <= Math.min(Math.ceil(g.width / ch.cols) - 1, Math.floor(c1 / ch.cols)); c++) {
      const key = `${r}_${c}`;
      if (ch.present.includes(key)) out.push(key);
    }
  return out;
}

/** Every file a grid needs to cover `box` (relative to the data base): the single file, or the chunks in it. */
export function gridFiles(g: GridSpec | null | undefined, box: [number, number, number, number] | null): string[] {
  if (!g) return [];
  if (g.url) return [g.url];
  return chunksFor(g, box).map((key) => g.chunks!.url_template.replace("{row}", key.split("_")[0]).replace("{col}", key.split("_")[1]));
}
