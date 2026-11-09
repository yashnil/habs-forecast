import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Band, ChartFrame, placeLabels, scaleLinear, timeTicks } from "@/v4/charts/kit";

const el = (C: unknown, props: object, ...kids: unknown[]) => createElement(C as never, props as never, ...(kids as never[]));

describe("chart kit", () => {
  it("ticks stay inside the domain at round steps", () => {
    const t0 = Date.UTC(2026, 9, 4, 3);
    const t1 = Date.UTC(2026, 9, 12, 21);
    const ticks = timeTicks([t0, t1], 6);
    expect(ticks.length).toBeGreaterThan(0);
    expect(ticks.length).toBeLessThanOrEqual(6);
    expect(ticks.every((t) => t >= t0 && t <= t1)).toBe(true);
    expect(timeTicks([5, 5])).toEqual([]);
  });
  it("direct labels never collide and hold to the frame", () => {
    const placed = placeLabels([100, 100, 104, 300], 14, [0, 320]);
    const sorted = [...placed].sort((a, b) => a - b);
    for (let i = 1; i < sorted.length; i++) expect(sorted[i] - sorted[i - 1]).toBeGreaterThanOrEqual(14);
    expect(placed[3]).toBe(300);
  });
  it("a scale maps its domain to its range", () => {
    const x = scaleLinear([0, 10], [100, 200]);
    expect(x(0)).toBe(100);
    expect(x(10)).toBe(200);
    expect(x(5)).toBe(150);
  });
  it("every chart states its text alternative", () => {
    const html = renderToStaticMarkup(el(ChartFrame, { width: 100, height: 40, title: "Toxin results", desc: "5.5 of 30 ppm" }, createElement("rect")));
    expect(html).toContain('role="img"');
    expect(html).toContain("<title");
    expect(html).toContain("Toxin results");
    expect(html).toContain("<desc");
    expect(html).toContain("5.5 of 30 ppm");
  });
  it("a band draws only across its span", () => {
    const html = renderToStaticMarkup(createElement("svg", null, createElement(Band, { x0: 10, x1: 40, y0: 0, y1: 20, label: "Advisory" })));
    expect(html).toContain('width="30"');
    expect(html).toContain("Advisory");
  });
});
