import { describe, expect, it } from "vitest";
import type { FieldGrid } from "@/nx/chart/layers/grid";
import { markAdjacent, type AdjacentMask } from "@/nx/chart/layers/adjacent";
import { FIELD_FS } from "@/nx/chart/layers/FieldLayer";

/**
 * The band left out beside land (QC `land_adjacency`) gets its own mark on the chart, never the cloud hatch.
 * A 4 x 6 grid on the mask's own lattice; N = no value.
 */
const N = 65535;
const g: FieldGrid = { width: 6, height: 4, lat_first: 37.0, lat_step: -0.0025, lon_first: -122.0, lon_step: 0.0025, scale_factor: 0.001, add_offset: -1, nodata: N, transform: "log10" };
// the mask covers more than the grid: one row north of it and columns either side
const mask: AdjacentMask = {
  lat_first: 37.0025,
  lon_first: -122.005,
  step: 0.0025,
  height: 6,
  width: 12,
  rows_rle: [[[0, 12]], [[6, 2]], [[5, 2]], [], [[2, 1], [6, 1]], [[0, 12]]],
};
// grid row r is mask row r + 1; grid column c is mask column c + 2
const codes = Uint16Array.from([
  1000, 1000, 1000, 1000, N, N, // mask row 1: columns 6,7 -> grid columns 4,5
  1000, 1000, 1000, N, 900, N, // mask row 2: columns 5,6 -> grid columns 3,4 (4 has a value)
  1000, N, 1000, 1000, 1000, 1000, // mask row 3: none (the gap at column 1 is cloud)
  N, 1000, 1000, 1000, N, 1000, // mask row 4: columns 2,6 -> grid columns 0,4
]);

describe("marking the cells left out beside land", () => {
  it("marks only cells that are both without a value and in the reference; cloud stays cloud; values are never touched", () => {
    const m = markAdjacent(g, codes, mask);
    expect(m.adjacentCode).toBe(N - 1);
    const A = N - 1;
    expect([...m.codes]).toEqual([
      1000, 1000, 1000, 1000, A, A,
      1000, 1000, 1000, A, 900, N,
      1000, N, 1000, 1000, 1000, 1000,
      A, 1000, 1000, 1000, A, 1000,
    ]);
  });
  it("returns a copy: the published codes a readout reads are unchanged", () => {
    const before = [...codes];
    const m = markAdjacent(g, codes, mask);
    expect(m.codes).not.toBe(codes);
    expect([...codes]).toEqual(before);
  });
  it("degrades to nothing: a reference with no cells here, or a grid away from it, marks nothing and keeps the same array", () => {
    expect(markAdjacent(g, codes, { ...mask, rows_rle: mask.rows_rle.map(() => []) })).toEqual({ codes, adjacentCode: null });
    expect(markAdjacent({ ...g, lat_first: 33 }, codes, mask)).toEqual({ codes, adjacentCode: null });
  });
  it("never reuses a code the data holds: the band then stays 'no view'", () => {
    const used = codes.slice();
    used[0] = N - 1;
    expect(markAdjacent(g, used, mask)).toEqual({ codes: used, adjacentCode: null });
  });
  it("the layer draws the marker as a flat band before it decodes a value, with no hatch, on observed layers only", () => {
    const i = FIELD_FS.indexOf("u_adj >= 0.0");
    expect(i).toBeGreaterThan(0);
    const branch = FIELD_FS.slice(FIELD_FS.lastIndexOf("if (", i), FIELD_FS.indexOf("return;", i));
    expect(branch).toMatch(/u_variant == 0/);
    expect(branch).not.toMatch(/hatch\(|rules\(|ramp\(/);
    expect(i).toBeLessThan(FIELD_FS.indexOf("float v = within ? q(codeAt(c), u_dec)"));
  });
});
