import { describe, expect, it } from "vitest";
import { LABEL_CHAR_W, nowLine, placeLabel, minutesOf, tideDay, tideNow } from "@/lib/tideCurve";
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

/** v8 fix pass: on a phone the low-water label "0.8 ft · 2:13 AM" ran through the now line at about 6 AM. */
describe("placeLabel", () => {
  const box = (p: { lx: number; anchor: string }, w: number) => (p.anchor === "start" ? [p.lx, p.lx + w] : p.anchor === "end" ? [p.lx - w, p.lx] : [p.lx - w / 2, p.lx + w / 2]);
  const w = "0.8 ft · 2:13 AM".length * LABEL_CHAR_W;

  it("centres a label on its turn when nothing is in the way", () => {
    expect(placeLabel(200, w, 640, null)).toEqual({ lx: 200, anchor: "middle" });
  });

  it("keeps a label inside the strip at either end", () => {
    expect(placeLabel(30, w, 330, null).anchor).toBe("start");
    expect(placeLabel(320, w, 330, null).anchor).toBe("end");
  });

  it("moves a label off the now line, onto its turn's own side, when it fits there", () => {
    // desktop: 2:13 AM at x 59 of 640, now at 5:45 AM (x 153)
    const p = placeLabel(59, w, 640, 153);
    const [a, b] = box(p, w);
    expect(b < 153 - 5 || a > 153 + 5).toBe(true);
    expect(a).toBeGreaterThanOrEqual(2);
    // a turn right of the line goes right of it
    const q = placeLabel(170, w, 640, 153);
    expect(box(q, w)[0]).toBeGreaterThan(153);
  });

  it("never carries a label across the line, away from its turn: where it cannot clear, it stays put", () => {
    // phone: 2:13 AM at x 30 of 330, now at x 79; neither fits left of the line
    const p = placeLabel(30, w, 330, 79);
    expect(p.anchor).toBe("start");
    expect(p.lx).toBeLessThan(79);
  });
});

describe("nowLine", () => {
  it("is one stretch when no label sits across it", () => {
    expect(nowLine(80, 8, 62, [{ box: { l: 100, r: 200, top: 40, bottom: 53 } }])).toEqual([[8, 62]]);
  });
  it("stops short of a label it crosses and resumes below it", () => {
    expect(nowLine(80, 8, 62, [{ box: { l: 25, r: 125, top: 40, bottom: 53 } }])).toEqual([[8, 38], [55, 62]]);
  });
  it("ends at a label that reaches the bottom", () => {
    expect(nowLine(80, 8, 62, [{ box: { l: 25, r: 125, top: 50, bottom: 63 } }])).toEqual([[8, 48]]);
  });
});
