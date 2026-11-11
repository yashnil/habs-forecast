"use client";

import { loadGrid } from "@/lib/grid";
import { gridBounds } from "@/nx/chart/layers/grid";
import type { GridSpec } from "./model";

/**
 * A grid's raw codes in the browser, assembled from the published file or chunks, fetched once (lib/grid caches
 * each file). A chunked grid is filled in as its chunks arrive: chunks in view first, the rest only when the
 * whole grid is small enough to hold (so a statewide 300 m grid is never fetched whole for a harbour view).
 * Cells of chunks not yet fetched, or not written (nothing there), read as no data.
 */
export type Codes = { codes: Uint16Array; complete: boolean };

const WHOLE_LIMIT = 4_000_000;
const join = (base: string, rel: string) => `${base.replace(/\/$/, "")}/${rel}`;

type Assembly = { codes: Uint16Array; loaded: Set<string>; pending: Map<string, Promise<void>> };
const assemblies = new Map<string, Assembly>();

function keyOf(base: string, g: GridSpec) {
  return `${base}|${g.url ?? g.chunks?.url_template}`;
}

function assembly(base: string, g: GridSpec): Assembly {
  const k = keyOf(base, g);
  let a = assemblies.get(k);
  if (!a) {
    a = { codes: new Uint16Array(g.width * g.height).fill(g.nodata), loaded: new Set(), pending: new Map() };
    assemblies.set(k, a);
    // a few grids at most are held (each frame of a large product is its own grid)
    while (assemblies.size > 8) assemblies.delete(assemblies.keys().next().value!);
  }
  return a;
}

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

function placeChunk(a: Assembly, g: GridSpec, key: string, codes: Uint16Array) {
  const ch = g.chunks!;
  const [r, c] = key.split("_").map(Number);
  const w = Math.min(ch.cols, g.width - c * ch.cols);
  const h = Math.min(ch.rows, g.height - r * ch.rows);
  for (let y = 0; y < h; y++) a.codes.set(codes.subarray(y * w, y * w + w), (r * ch.rows + y) * g.width + c * ch.cols);
}

/**
 * The grid's codes covering `box` (the whole grid when it is small, or `box` is null). `onPart` is called as
 * each chunk lands, with the same (growing) array, so a layer can draw what has arrived.
 */
export async function loadCodes(base: string, g: GridSpec, box: [number, number, number, number] | null = null, onPart?: (c: Codes) => void): Promise<Codes> {
  const a = assembly(base, g);
  if (g.url) {
    if (!a.loaded.has("all")) {
      const codes = await loadGrid(join(base, g.url));
      a.codes.set(codes.subarray(0, a.codes.length));
      a.loaded.add("all");
    }
    return { codes: a.codes, complete: true };
  }
  const want = chunksFor(g, box);
  await Promise.all(
    want.map((key) => {
      if (a.loaded.has(key)) return Promise.resolve();
      let p = a.pending.get(key);
      if (!p) {
        p = loadGrid(join(base, g.chunks!.url_template.replace("{row}", key.split("_")[0]).replace("{col}", key.split("_")[1]))).then((codes) => {
          placeChunk(a, g, key, codes);
          a.loaded.add(key);
          a.pending.delete(key);
          onPart?.({ codes: a.codes, complete: false });
        });
        p.catch(() => a.pending.delete(key));
        a.pending.set(key, p);
      }
      return p;
    }),
  );
  return { codes: a.codes, complete: (g.chunks?.present ?? []).every((k) => a.loaded.has(k)) };
}

/** A small box around a point, for reading one cell or one row. */
export function boxAround(lon: number, lat: number, dLon: number, dLat = dLon): [number, number, number, number] {
  return [lon - dLon, lat - dLat, lon + dLon, lat + dLat];
}

export { gridBounds };
