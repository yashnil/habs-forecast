import { describe, expect, it } from "vitest";
import { breakEvenSteps, describePoint, linearTicks, nearest, onAxis, runs, stepIndex, windowAround, yDomainOf, type PlotPoint } from "@/nx/ds/plot/model";

const DAY = 86_400_000;
const t0 = Date.UTC(2025, 0, 1, 20);
const pts: PlotPoint[] = [
  { t: t0, v: 2 },
  { t: t0 + 7 * DAY, v: null, kind: "reported-zero", raw: "0" },
  { t: t0 + 14 * DAY, v: 4 },
  { t: t0 + 21 * DAY, v: 0.5, kind: "below-detection", raw: "<0.5" },
  { t: t0 + 120 * DAY, v: 8 },
  { t: t0 + 127 * DAY, v: 6, kind: "forecast" },
];

describe("plot model", () => {
  it("places only real values on the axis: a reported zero has no place, a detection limit does", () => {
    expect(pts.map(onAxis)).toEqual([true, false, true, true, true, true]);
    const [lo, hi] = yDomainOf(pts, "linear");
    expect(lo).toBe(0);
    expect(hi).toBeCloseTo(8.6, 6);
    expect(yDomainOf(pts, "log")).toEqual([0.1, 10]);
    expect(yDomainOf([], "linear")).toEqual([0, 1]);
  });

  it("never joins a line across a zero, a censored value or a gap", () => {
    const r = runs(pts, 45 * DAY);
    expect(r.map((x) => x.map((p) => p.v))).toEqual([[2], [4], [8, 6]]);
  });

  it("finds the nearest sample and steps without leaving the list", () => {
    expect(nearest(pts, t0 + 8 * DAY)).toBe(1);
    expect(nearest([], 0)).toBe(-1);
    expect(stepIndex(pts, -1, "next")).toBe(0);
    expect(stepIndex(pts, -1, "prev")).toBe(5);
    expect(stepIndex(pts, 0, "prev")).toBe(0);
    expect(stepIndex(pts, 5, "next")).toBe(5);
    expect(stepIndex(pts, 2, "first")).toBe(0);
    expect(stepIndex(pts, 2, "last")).toBe(5);
    // a page is a month on a time axis, and always moves at least one sample
    expect(stepIndex(pts, 0, "pageNext")).toBe(3);
    expect(stepIndex(pts, 4, "pageNext")).toBe(5);
    expect(stepIndex([], 0, "next")).toBe(-1);
  });

  it("says a sample in words, with what the source printed for a censored value", () => {
    expect(describePoint(pts[0], { unit: "ng/mL" })).toBe("Jan 1, 2025 · 2 ng/mL · measured");
    expect(describePoint(pts[1], { unit: "ng/mL" })).toBe("Jan 8, 2025 · 0 · reported 0");
    expect(describePoint(pts[3], { unit: "ng/mL" })).toBe("Jan 22, 2025 · <0.5 · below detection");
    expect(describePoint({ t: t0, v: null, kind: "not-collected" })).toBe("Jan 1, 2025 · not collected");
  });

  it("makes round ticks", () => {
    expect(linearTicks([0, 100], 5)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(linearTicks([-400, 3600], 4)).toEqual([0, 1000, 2000, 3000]);
  });

  it("moves a window to hold a moment, keeping its width inside the record", () => {
    const extent: [number, number] = [0, 1000];
    expect(windowAround(extent, [800, 900], 300)).toEqual([250, 350]);
    expect(windowAround(extent, [800, 900], 990)).toEqual([900, 1000]);
    expect(windowAround(extent, [800, 900], 10)).toEqual([0, 100]);
    expect(windowAround(extent, [800, 900], 500, 400)).toEqual([300, 700]);
  });

  it("finds the break-even step of each row", () => {
    expect(breakEvenSteps([[-5, -1, 3], [-2, 0, 4], [-9, -8, -1], [1, 2, 3], [null, 2, 3]])).toEqual([2, 1, null, 0, null]);
  });
});
