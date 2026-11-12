/**
 * Land adjacency (docs/coastwatch/v13/research/coastal-adjacency.md). In the Sentinel-3 OLCI 300 m chlorophyll
 * the water pixel that touches the product's land mask reads land and water together, ten to fifty times the open
 * water, and the next one several times; the pipeline leaves those two pixels out (satellite-processing-5, QC
 * check `land_adjacency`). They have no value, and that is not cloud: the readout and the legend say "left out:
 * beside land".
 *
 * The cells are a static reference on the OLCI lattice (public/reference/land-adjacency-v1.json, written by
 * pipeline/scripts/build_land_adjacency_mask.py beside data/reference/olci_land_adjacency.json), read once, only
 * when a reading of an OLCI frame that applied the rule has no value.
 */
export type AdjacencyMask = {
  lat_first: number;
  lon_first: number;
  /** degrees per cell (rows run north to south) */
  step: number;
  height: number;
  width: number;
  /** per row, runs of left-out cells: [first column, length], west to east */
  rows_rle: [number, number][][];
};

export const ADJACENCY_URL = "/reference/land-adjacency-v1.json";

/** The QC check a layer carries when the pipeline left the ring out of it. */
export const ADJACENCY_CHECK = "land_adjacency";

/** True when the layer's own QC says the rule was applied (layers made before it, and VIIRS, do not carry it). */
export function appliesAdjacency(layer: { qc?: { checks?: { name: string }[] | null } | null } | null | undefined): boolean {
  return !!layer?.qc?.checks?.some((c) => c.name === ADJACENCY_CHECK);
}

/** Is the cell that holds this point one of the left-out cells? Nearest cell of the mask's lattice; false outside it. */
export function isLandAdjacent(m: AdjacencyMask, lat: number, lon: number): boolean {
  const row = Math.round((m.lat_first - lat) / m.step);
  const col = Math.round((lon - m.lon_first) / m.step);
  if (row < 0 || row >= m.height || col < 0 || col >= m.width) return false;
  const runs = m.rows_rle[row] ?? [];
  let lo = 0;
  let hi = runs.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const [s, n] = runs[mid];
    if (col < s) hi = mid - 1;
    else if (col >= s + n) lo = mid + 1;
    else return true;
  }
  return false;
}

let cached: Promise<AdjacencyMask | null> | null = null;

/** The reference, read once. Null when it cannot be read: the reading then says "no view", the weaker true statement. */
export function loadAdjacency(fetcher: typeof fetch = fetch): Promise<AdjacencyMask | null> {
  cached ??= fetcher(ADJACENCY_URL)
    .then((r) => (r.ok ? (r.json() as Promise<AdjacencyMask>) : null))
    .then((m) => (m && Array.isArray(m.rows_rle) && Number.isFinite(m.step) ? m : null))
    .catch(() => null);
  return cached;
}

/** For tests. */
export function resetAdjacency() {
  cached = null;
}

/** What the readout says for a left-out cell (the lead's words), and the legend's key for it. */
export const ADJACENT_WORDS = "Left out: beside land, where the satellite reads land and water together. No value is not a low value.";
export const ADJACENT_KEY = "Left out: beside land (two pixels, about 600 m)";
