import { describe, expect, it } from "vitest";
import { minutesOf, tideDay, tideNow } from "@/lib/tideCurve";
import type { Tide } from "@/lib/conditions";

const T: Tide[] = [
  { at: "2026-10-22 03:10", ft: 5.2, type: "H" },
  { at: "2026-10-22 09:40", ft: 0.4, type: "L" },
  { at: "2026-10-22 15:50", ft: 4.6, type: "H" },
  { at: "2026-10-22 21:30", ft: 1.9, type: "L" },
  { at: "2026-10-23 04:00", ft: 5.0, type: "H" },
];

describe("tide curve (drawn between NOAA's predicted highs and lows)", () => {
  it("reads CO-OPS local times as minutes after the day's midnight, across days", () => {
    expect(minutesOf("2026-10-22 03:10", "2026-10-22")).toBe(190);
    expect(minutesOf("2026-10-23 04:00", "2026-10-22")).toBe(1440 + 240);
    expect(minutesOf("2026-10-21 23:00", "2026-10-22")).toBe(-60);
    expect(minutesOf("not a time", "2026-10-22")).toBeNaN();
  });

  it("passes exactly through every turning point and never beyond the predicted range inside the day", () => {
    const d = tideDay(T, "2026-10-22", 10)!;
    expect(d.turns.map((t) => t.type)).toEqual(["H", "L", "H", "L"]);
    const at = (min: number) => d.points.find((p) => p.min === min)!.ft;
    expect(at(190)).toBeCloseTo(5.2, 6);
    expect(at(580)).toBeCloseTo(0.4, 6);
    expect(at(950)).toBeCloseTo(4.6, 6);
    expect(d.lo).toBeGreaterThanOrEqual(0.4 - 1e-9);
    expect(d.hi).toBeLessThanOrEqual(5.2 + 1e-9);
    expect(d.points[0].min).toBe(0);
    expect(d.points[d.points.length - 1].min).toBe(1440);
  });

  it("marks the mirrored stretch before the first prediction", () => {
    const d = tideDay(T, "2026-10-22")!;
    expect(d.edgeBefore).toBe(190);
    expect(d.edgeAfter).toBe(1440);
  });

  it("draws nothing for a day the predictions do not reach", () => {
    expect(tideDay(T, "2026-10-25")).toBeNull();
    expect(tideDay(T, "2026-10-19")).toBeNull();
  });

  it("needs two turning points", () => {
    expect(tideDay(T.slice(0, 1), "2026-10-22")).toBeNull();
    expect(tideDay([], "2026-10-22")).toBeNull();
  });

  it("says rising before a high and falling before a low, with the next turn", () => {
    expect(tideNow(T, "2026-10-22", 600)).toMatchObject({ rising: true, next: { type: "H", ft: 4.6 } });
    expect(tideNow(T, "2026-10-22", 1000)).toMatchObject({ rising: false, next: { type: "L", ft: 1.9 } });
    expect(tideNow(T, "2026-10-22", 5000)).toBeNull();
  });
});
