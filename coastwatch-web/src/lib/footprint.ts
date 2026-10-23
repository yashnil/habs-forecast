import type { ValueGrid } from "@/generated/schema";

/**
 * The true footprint of what a selected point is read from: the published cell or pixel that contains
 * it, at its own native size (a 3 km C-HARM cell, a 300 m OLCI or 750 m VIIRS pixel, a 2 km radar cell).
 * Drawn as an outline on the map, and kept in the URL (`fp=product:row:col`) so the selection survives a
 * reload and a trip to My Coast or Research and back. Geometry only: no value is read or changed here.
 */
export type FootprintProduct = "charm" | "olci300" | "viirs750" | "hfr2km";
export const FOOTPRINT_PRODUCTS: readonly FootprintProduct[] = ["charm", "olci300", "viirs750", "hfr2km"];
export type FootprintRef = { product: FootprintProduct; row: number; col: number };
export type Lattice = Pick<ValueGrid, "lat_first" | "lat_step" | "lon_first" | "lon_step" | "width" | "height">;

export const FOOTPRINT_WORDS: Record<FootprintProduct, string> = {
  charm: "C-HARM model cell",
  olci300: "Sentinel-3 OLCI pixel",
  viirs750: "VIIRS pixel",
  hfr2km: "HF-radar cell",
};

/** The row and column of the cell containing a point (centred lattice, as lib/grid cellAt), or null outside the grid. */
export function footprintCell(g: Lattice, lat: number, lon: number): { row: number; col: number } | null {
  const row = Math.floor((lat - g.lat_first) / g.lat_step + 0.5);
  const col = Math.floor((lon - g.lon_first) / g.lon_step + 0.5);
  if (row < 0 || col < 0 || row >= g.height || col >= g.width) return null;
  return { row, col };
}

/** The cell's centre. */
export function footprintCentre(g: Lattice, row: number, col: number): { lat: number; lon: number } {
  return { lat: g.lat_first + row * g.lat_step, lon: g.lon_first + col * g.lon_step };
}

const r6 = (n: number) => Math.round(n * 1e6) / 1e6;

/** The cell's square, closed ring (centre ± half a step on each axis), counter-clockwise from the south-west corner. */
export function footprintPolygon(g: Lattice, row: number, col: number): GeoJSON.Polygon {
  const { lat, lon } = footprintCentre(g, row, col);
  const dy = Math.abs(g.lat_step) / 2;
  const dx = Math.abs(g.lon_step) / 2;
  const [w, e, s, n] = [r6(lon - dx), r6(lon + dx), r6(lat - dy), r6(lat + dy)];
  return { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] };
}

/** The cell's size in km (north-south, east-west) at its own latitude. */
export function footprintKm(g: Lattice, row: number): { ns: number; ew: number } {
  const lat = g.lat_first + row * g.lat_step;
  return { ns: Math.abs(g.lat_step) * 111.32, ew: Math.abs(g.lon_step) * 111.32 * Math.cos((lat * Math.PI) / 180) };
}

export function encodeFootprint(f: FootprintRef | null): string | null {
  return f ? `${f.product}:${f.row}:${f.col}` : null;
}

export function decodeFootprint(s: string | null | undefined): FootprintRef | null {
  const m = /^([a-z0-9]+):(\d{1,5}):(\d{1,5})$/.exec(s ?? "");
  if (!m || !(FOOTPRINT_PRODUCTS as readonly string[]).includes(m[1])) return null;
  return { product: m[1] as FootprintProduct, row: Number(m[2]), col: Number(m[3]) };
}

/** The footprint product of a published layer id (olci300_…, viirs750_…, hfr2km_…, charm_…), or null. */
export function productOf(layerId: string): FootprintProduct | null {
  const p = layerId.split("_")[0];
  return (FOOTPRINT_PRODUCTS as readonly string[]).includes(p) ? (p as FootprintProduct) : null;
}
