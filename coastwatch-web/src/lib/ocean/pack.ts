import { gzipSync } from "node:zlib";
import { MASK_LAND, MASK_NEAR_SHORE, MASK_WATER, type OceanGrid } from "@/lib/ocean/contract";

/**
 * Packing decoded model fields into the app's grid encoding (uint16 little-endian, gzip; lib/grid decodeGrid).
 * Upstream fill values (WCOFS −99999, DINEOF −999), NaN and values outside the stated valid range become `nodata`,
 * never zero. Server only.
 */
export const NODATA = 65535;

export type Packing = { scale_factor: number; add_offset: number; transform: "none" | "log10"; valid: [number, number] };

/** Currents in m/s, 1 mm/s steps over ±32 m/s. */
export const PACK_VELOCITY: Packing = { scale_factor: 0.001, add_offset: -32, transform: "none", valid: [-10, 10] };
/** Temperature in °C, 0.001 °C steps from −5 °C. */
export const PACK_TEMPERATURE: Packing = { scale_factor: 0.001, add_offset: -5, transform: "none", valid: [-3, 40] };
/** Chlorophyll in mg/m³ as log10, 0.0001 decades from 10^-3.5. */
export const PACK_CHL_LOG10: Packing = { scale_factor: 0.0001, add_offset: -3.5, transform: "log10", valid: [0.001, 100] };
export const PACK_MASK: Packing = { scale_factor: 1, add_offset: 0, transform: "none", valid: [0, 2] };

export function isFill(x: number, fill: number): boolean {
  return !Number.isFinite(x) || x === fill || Math.abs(x - fill) < 1e-3 * Math.max(1, Math.abs(fill));
}

/** Quantize values to codes; fill/NaN/out-of-range → NODATA. */
export function packCodes(values: ArrayLike<number>, p: Packing, fill: number): Uint16Array {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const x = values[i];
    if (isFill(x, fill) || x < p.valid[0] || x > p.valid[1]) { out[i] = NODATA; continue; }
    const q = p.transform === "log10" ? Math.log10(x) : x;
    out[i] = Math.min(NODATA - 1, Math.max(0, Math.round((q - p.add_offset) / p.scale_factor)));
  }
  return out;
}

/** A code back to its value (the same rule as lib/grid valueOf), for tests and server-side sampling. */
export function unpackCode(code: number, p: Pick<Packing, "scale_factor" | "add_offset" | "transform">): number | null {
  if (code === NODATA) return null;
  const q = code * p.scale_factor + p.add_offset;
  return p.transform === "log10" ? 10 ** q : q;
}

export function gzipCodes(codes: Uint16Array): Buffer {
  const b = Buffer.alloc(codes.length * 2);
  for (let i = 0; i < codes.length; i++) b.writeUInt16LE(codes[i], i * 2);
  return gzipSync(b, { level: 6 });
}

type Geometry = Pick<OceanGrid, "width" | "height" | "lat_first" | "lat_step" | "lon_first" | "lon_step">;

/**
 * Land/water/near-shore codes from which cells have a value: a water cell whose centre is within `km` of the edge of a
 * land (no-value) cell (equirectangular) is MASK_NEAR_SHORE, so a cell touching land always is. Cells beyond the grid's edge are not known to be land.
 */
export function shoreMask(wet: (k: number) => boolean, g: Geometry, km: number): Uint16Array {
  const out = new Uint16Array(g.width * g.height);
  const dyKm = Math.abs(g.lat_step) * 111.32;
  const ry = Math.ceil(km / dyKm + 0.5);
  for (let r = 0; r < g.height; r++) {
    const lat = g.lat_first + r * g.lat_step;
    const dxKm = Math.abs(g.lon_step) * 111.32 * Math.cos((lat * Math.PI) / 180);
    const rx = Math.ceil(km / dxKm + 0.5);
    for (let c = 0; c < g.width; c++) {
      const k = r * g.width + c;
      if (!wet(k)) { out[k] = MASK_LAND; continue; }
      let near = false;
      for (let dr = -ry; dr <= ry && !near; dr++) {
        const rr = r + dr;
        if (rr < 0 || rr >= g.height) continue;
        for (let dc = -rx; dc <= rx; dc++) {
          const cc = c + dc;
          if (cc < 0 || cc >= g.width || (dr === 0 && dc === 0)) continue;
          const d = Math.hypot(Math.max(0, (Math.abs(dr) - 0.5) * dyKm), Math.max(0, (Math.abs(dc) - 0.5) * dxKm));
          if (d <= km && !wet(rr * g.width + cc)) { near = true; break; }
        }
      }
      out[k] = near ? MASK_NEAR_SHORE : MASK_WATER;
    }
  }
  return out;
}

/** An OceanGrid record for a packing and geometry. */
export function gridRecord(g: Geometry, p: Packing, url: string): OceanGrid {
  return {
    ...g,
    scale_factor: p.scale_factor,
    add_offset: p.add_offset,
    nodata: NODATA,
    transform: p.transform,
    max_quantization_error: p.scale_factor / 2,
    encoding: "uint16le+gzip",
    url,
  };
}
