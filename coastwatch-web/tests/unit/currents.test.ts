import { describe, expect, it } from "vitest";
import { compass, directionDeg, fieldFeatures, hourStamp, hoursAgo, sampleField, type CurrentField } from "@/lib/currents";
import type { ValueGrid } from "@/generated/schema";

const g = { url: "", width: 9, height: 9, lat_first: 36.5, lat_step: 0.018, lon_first: -122.2, lon_step: 0.0208, scale_factor: 1, add_offset: 0, max_quantization_error: 0 } as unknown as ValueGrid;
function field(fill: (r: number, c: number) => [number, number] | null): CurrentField {
  const u = new Float32Array(81).fill(NaN);
  const v = new Float32Array(81).fill(NaN);
  for (let r = 0; r < 9; r++)
    for (let c = 0; c < 9; c++) {
      const x = fill(r, c);
      if (x) [u[r * 9 + c], v[r * 9 + c]] = x;
    }
  return { g, u, v };
}

describe("currents helpers", () => {
  it("direction is toward, clockwise from north (same as the pipeline)", () => {
    expect([directionDeg(0, 1), directionDeg(1, 0), directionDeg(0, -1), directionDeg(-1, 0)]).toEqual([0, 90, 180, 270]);
    expect(compass(0)).toBe("N");
    expect(compass(338)).toBe("NNW");
    expect(compass(359)).toBe("N");
  });

  it("one point per valid cell; thinning levels are fixed by grid position", () => {
    const f = field((r, c) => (r === 4 && c === 4 ? null : [0.3, 0.4]));
    const fc = fieldFeatures(f);
    expect(fc.features).toHaveLength(80);
    const at = (r: number, c: number) => fc.features.find((x) => x.geometry.coordinates[0] === g.lon_first + c * g.lon_step && x.geometry.coordinates[1] === g.lat_first + r * g.lat_step)!;
    expect(at(0, 0).properties!.level).toBe(8);
    expect(at(0, 4).properties!.level).toBe(4);
    expect(at(2, 6).properties!.level).toBe(2);
    expect(at(1, 3).properties!.level).toBe(1);
    expect(at(0, 0).properties!.speed).toBeCloseTo(0.5, 6);
  });

  it("sampling never interpolates: the containing cell, or the nearest valid one within a cell", () => {
    const f = field((r, c) => (r === 5 && c === 5 ? [0.1, 0] : null));
    expect(sampleField(f, g.lat_first + 5 * g.lat_step, g.lon_first + 5 * g.lon_step)?.u).toBeCloseTo(0.1, 6);
    const near = sampleField(f, g.lat_first + 4 * g.lat_step, g.lon_first + 5 * g.lon_step, 1)!;
    expect(near.u).toBeCloseTo(0.1, 6);
    expect(near.distanceKm).toBeGreaterThan(1.5);
    expect(sampleField(f, g.lat_first + 2 * g.lat_step, g.lon_first + 2 * g.lon_step, 1)).toBeNull();
  });

  it("hour stamps and ages", () => {
    expect(hourStamp("2026-10-08T12:00:00Z")).toBe("20261008T12Z");
    expect(hoursAgo("2026-10-08T12:00:00Z", new Date("2026-10-08T20:30:00Z"))).toBe(8);
  });
});
