import type { LayerArtifact, ValueGrid } from "@/generated/schema";
import { artifactUrl } from "@/lib/layers";
import { cellAt, decodeCode, distanceKm, loadGrid } from "@/lib/grid";

/**
 * Nearshore context for the C-HARM layer (v5): NOAA's 3 km model publishes no value for most water
 * within a few kilometres of the shore, and none inside the bays. So that the coast is not an empty
 * band, water in the grid without a value is drawn in one calm neutral tint, "outside NOAA's model
 * near shore" (lib/gridTiles OUTSIDE_TINT). Its readout gives the nearest NOAA cell within
 * CONTEXT_KM as text only, with the pipeline's pier-sample check: against CalHABMAP samples the
 * nearest cell showed no skill at the shore, so its probability is never painted onto that water.
 */
export const CONTEXT_KM = 6;

/**
 * For every cell of a full (unchunked) grid without a value: the index of the nearest cell with a
 * value whose centre is within `maxKm` of its centre (ties: the first found, scanning from the
 * north-west), or -1. Cells with a value hold -1 (they need no context). Pure.
 */
export function contextIndex(g: ValueGrid, codes: Uint16Array, maxKm = CONTEXT_KM): Int32Array {
  const nodata = g.nodata ?? 65535;
  const out = new Int32Array(g.width * g.height).fill(-1);
  const kmLat = Math.abs(g.lat_step) * 111.32;
  for (let r = 0; r < g.height; r++) {
    const lat = g.lat_first + r * g.lat_step;
    const kmLon = Math.abs(g.lon_step) * 111.32 * Math.cos((lat * Math.PI) / 180);
    const dr = Math.floor(maxKm / kmLat);
    const dc = Math.floor(maxKm / Math.max(1e-6, kmLon));
    for (let c = 0; c < g.width; c++) {
      if (codes[r * g.width + c] !== nodata) continue;
      let best = -1;
      let bestD = maxKm + 1e-9;
      for (let i = Math.max(0, r - dr); i <= Math.min(g.height - 1, r + dr); i++) {
        const dy = (i - r) * kmLat;
        for (let k = Math.max(0, c - dc); k <= Math.min(g.width - 1, c + dc); k++) {
          const j = i * g.width + k;
          if (codes[j] === nodata) continue;
          const d = Math.hypot(dy, (k - c) * kmLon);
          if (d < bestD) {
            bestD = d;
            best = j;
          }
        }
      }
      out[r * g.width + c] = best;
    }
  }
  return out;
}

const cache = new Map<string, Promise<{ codes: Uint16Array; index: Int32Array }>>();

/** The decoded grid and its context index, computed once per published grid (and shared by the tiles and the readouts). */
export function loadContext(gridUrl: string, g: ValueGrid): Promise<{ codes: Uint16Array; index: Int32Array }> {
  let p = cache.get(gridUrl);
  if (!p) {
    p = loadGrid(gridUrl).then((codes) => ({ codes, index: contextIndex(g, codes) }));
    p.catch(() => cache.delete(gridUrl));
    cache.set(gridUrl, p);
  }
  return p;
}

export type ForecastContext =
  /** the point's own cell has a published value: no context is needed */
  | { kind: "value" }
  /** no value here; the nearest NOAA cell within CONTEXT_KM, its value and its distance from the point */
  | { kind: "context"; value: number; km: number; cell: { lat: number; lon: number } }
  /** no value here and none within CONTEXT_KM (a bay, an estuary, outside the grid): outside NOAA's model */
  | { kind: "outside" };

/** What the map draws for a C-HARM layer at a point (lib/gridTilesContext): the cell's own value, context, or outside. */
export async function forecastContext(baseUrl: string, l: LayerArtifact, lat: number, lon: number): Promise<ForecastContext> {
  const g = l.grid;
  if (!g) return { kind: "outside" };
  const c = cellAt(g, lat, lon);
  if (!c) return { kind: "outside" };
  const { codes, index } = await loadContext(artifactUrl(baseUrl, g.url), g);
  const i = c.row * g.width + c.col;
  if (codes[i] !== (g.nodata ?? 65535)) return { kind: "value" };
  const j = index[i];
  if (j < 0) return { kind: "outside" };
  const row = Math.floor(j / g.width);
  const col = j % g.width;
  const cell = { lat: g.lat_first + row * g.lat_step, lon: (g.lon_first > 180 ? g.lon_first - 360 : g.lon_first) + col * g.lon_step };
  return { kind: "context", value: decodeCode(g, codes[j])!, km: distanceKm(lat, lon, cell.lat, cell.lon), cell };
}

/**
 * The readout's words where NOAA's model does not reach (v5, after the pier-sample check showed the
 * nearest cell has no skill at the shore): the nearest NOAA value as text only, never as this water's.
 */
export function contextWords(x: { value: number; km: number }): string {
  const km = x.km < 10 ? x.km.toFixed(1) : Math.round(x.km).toString();
  return `NOAA's model does not reach this water. The nearest NOAA cell, ${km} km away, reads ${Math.round(x.value * 100)}%: text only, not a forecast for this water.`;
}

export const OUTSIDE_WORDS = "NOAA's model does not reach this water: C-HARM publishes no value here or within 6 km (bays, harbours and estuaries are not modelled). Not a forecast of zero.";
