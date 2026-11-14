import { gunzipSync } from "node:zlib";
import { describe, expect, it } from "vitest";
import { MASK_LAND, MASK_NEAR_SHORE, MASK_WATER } from "@/lib/ocean/contract";
import { gzipCodes, NODATA, packCodes, PACK_CHL_LOG10, PACK_TEMPERATURE, PACK_VELOCITY, shoreMask, unpackCode } from "@/lib/ocean/pack";

describe("packing", () => {
  it("turns the fill value, NaN and out-of-range values into nodata, never zero", () => {
    const c = packCodes([19.71, -99999, NaN, 99, 0], PACK_TEMPERATURE, -99999);
    expect(unpackCode(c[0], PACK_TEMPERATURE)).toBeCloseTo(19.71, 3);
    expect([c[1], c[2], c[3]]).toEqual([NODATA, NODATA, NODATA]);
    expect(unpackCode(c[4], PACK_TEMPERATURE)).toBeCloseTo(0, 6);
  });

  it("keeps currents within half a millimetre per second, signs intact", () => {
    for (const x of [0.049, -0.731, 1.234, 0]) expect(Math.abs(unpackCode(packCodes([x], PACK_VELOCITY, -99999)[0], PACK_VELOCITY)! - x)).toBeLessThanOrEqual(0.0005 + 1e-9);
  });

  it("stores chlorophyll as log10 with a small relative error", () => {
    for (const x of [0.01, 0.33, 4.2, 60]) {
      const back = unpackCode(packCodes([x], PACK_CHL_LOG10, -999)[0], PACK_CHL_LOG10)!;
      expect(Math.abs(back / x - 1)).toBeLessThan(2e-4);
    }
    expect(packCodes([-999, 0, 200], PACK_CHL_LOG10, -999)).toEqual(new Uint16Array([NODATA, NODATA, NODATA]));
  });

  it("writes uint16 little-endian under gzip (the app's grid encoding)", () => {
    const raw = gunzipSync(gzipCodes(new Uint16Array([1, 258, NODATA])));
    expect([...raw]).toEqual([1, 0, 2, 1, 255, 255]);
  });

  it("flags water within the near-shore distance of a land cell's edge", () => {
    // one row at 36.7°N, 0.04° cells (≈3.57 km across): land in column 0
    const g = { width: 6, height: 1, lat_first: 36.7, lat_step: 0.04, lon_first: -122.3, lon_step: 0.04 };
    const m = shoreMask((k) => k !== 0, g, 6);
    expect(m[0]).toBe(MASK_LAND);
    expect(m[1]).toBe(MASK_NEAR_SHORE); // 1.8 km from the land cell's edge
    expect(m[2]).toBe(MASK_NEAR_SHORE); // 5.4 km
    expect(m[3]).toBe(MASK_WATER); // 8.9 km
  });
});
