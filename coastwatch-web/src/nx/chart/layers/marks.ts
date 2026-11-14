/**
 * v15.1: an observed layer's cells without a value are three different things, and each is drawn its own way
 * (FieldLayer), matching the legend:
 * - "no view": inside the area the sensor saw this week, cloud, fog or no pass (the hatch);
 * - "outside this layer's area": west of the westernmost pixel the week's latest-view composite holds in that row,
 *   or past the grid (a plain quiet tint), so the open ocean beyond the coastal product is never drawn as cloud;
 * - "not loaded yet": a chunk still on its way (nothing drawn; the page shows its progress instead).
 *
 * PURE. The published codes are never changed: a COPY carries marker codes for drawing only; readings read the
 * published codes. A marker code already used by the data is not used (those cells stay as they were).
 */
import type { FieldGrid } from "./grid";

export type DisplayMarks = { codes: Uint16Array; outCode: number | null; loadingCode: number | null };

type Chunked = FieldGrid & { chunks?: { rows: number; cols: number; present: string[] } | null };

/** Per row, the westernmost column with a value (-1: none in the row). From a fully loaded latest-view composite. */
export function westmostPerRow(g: FieldGrid, codes: Uint16Array, skip: number[] = []): Int32Array {
  const out = new Int32Array(g.height).fill(-1);
  for (let r = 0; r < g.height; r++) {
    const base = r * g.width;
    for (let c = 0; c < g.width; c++) {
      const v = codes[base + c];
      if (v !== g.nodata && !skip.includes(v)) {
        out[r] = c;
        break;
      }
    }
  }
  return out;
}

export function markForDisplay(g: Chunked, codes: Uint16Array, o: { westmost?: Int32Array | null; loaded?: Set<string> | null }): DisplayMarks {
  const outM = g.nodata - 2, loadM = g.nodata - 3;
  let used = false;
  for (let i = 0; i < codes.length; i++) if (codes[i] === outM || codes[i] === loadM) { used = true; break; }
  if (used) return { codes, outCode: null, loadingCode: null };
  const west = o.westmost && o.westmost.length === g.height ? o.westmost : null;
  const ch = g.chunks;
  const loaded = ch && o.loaded ? o.loaded : null;
  if (!west && !loaded) return { codes, outCode: null, loadingCode: null };
  const out = codes.slice();
  // which chunks are published but not here yet (one flag per chunk, not per cell)
  const ncol = ch ? Math.ceil(g.width / ch.cols) : 1, nrow = ch ? Math.ceil(g.height / ch.rows) : 1;
  const waiting = new Uint8Array(ncol * nrow);
  if (loaded && ch) for (const key of ch.present) if (!loaded.has(key)) {
    const [cr, cc] = key.split("_").map(Number);
    if (cr < nrow && cc < ncol) waiting[cr * ncol + cc] = 1;
  }
  for (let r = 0; r < g.height; r++) {
    const base = r * g.width;
    const wm = west ? west[r] : 0;
    const cr = ch ? Math.floor(r / ch.rows) : 0;
    for (let c = 0; c < g.width; c++) {
      const k = base + c;
      if (out[k] !== g.nodata) continue;
      if (loaded && ch && waiting[cr * ncol + Math.floor(c / ch.cols)]) {
        out[k] = loadM;
        continue;
      }
      if (west && (wm < 0 || c < wm)) out[k] = outM;
    }
  }
  return { codes: out, outCode: west ? outM : null, loadingCode: loaded ? loadM : null };
}
