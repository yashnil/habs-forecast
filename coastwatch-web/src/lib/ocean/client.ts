import type { ValueGrid } from "@/generated/schema";
import type { CurrentField } from "@/lib/currents";
import { loadGrid } from "@/lib/grid";
import type { ChlEstimateIndex, OceanGrid, WcofsIndex, WcofsStep } from "@/lib/ocean/contract";

/**
 * Browser loaders for the ocean model routes (lib/ocean/contract). Grids come back in the app's own encoding, so the
 * existing FieldLayer (codes + grid) and FlowLayer (CurrentField) draw them unchanged.
 */
export const WCOFS_INDEX_URL = "/explore/model/wcofs";
export const CHL_ESTIMATE_INDEX_URL = "/explore/model/chlorophyll";

async function json<T>(url: string): Promise<T> {
  const r = await fetch(url);
  return (await r.json()) as T; // unavailable answers are JSON bodies with status "unavailable"
}

export const loadWcofsIndex = () => json<WcofsIndex>(WCOFS_INDEX_URL);
export const loadChlEstimateIndex = () => json<ChlEstimateIndex>(CHL_ESTIMATE_INDEX_URL);

/** An OceanGrid as the published ValueGrid type (they share the geometry and decoding fields). */
export const asValueGrid = (g: OceanGrid): ValueGrid => g as unknown as ValueGrid;

/** A grid's raw codes (uint16; `nodata` = no value). */
export const loadOceanCodes = (g: OceanGrid): Promise<Uint16Array> => loadGrid(g.url);

function decode(g: OceanGrid, codes: Uint16Array): Float32Array {
  const out = new Float32Array(codes.length);
  for (let i = 0; i < codes.length; i++) out[i] = codes[i] === g.nodata ? NaN : codes[i] * g.scale_factor + g.add_offset;
  return out;
}

/** A WCOFS step's predicted currents as the FlowLayer's field (m/s east and north; NaN on land). */
export async function loadWcofsCurrents(step: WcofsStep): Promise<CurrentField> {
  const [u, v] = await Promise.all([loadGrid(step.grids.u.url), loadGrid(step.grids.v.url)]);
  return { g: asValueGrid(step.grids.u), u: decode(step.grids.u, u), v: decode(step.grids.v, v) };
}
