import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TICK_LABEL_PX, monthTicks, tickLabel } from "@/v4/charts/SeriesChart";
import { Icon, iconClass } from "@/v4/ui/Icon";
import { ageWords, savedSnapshot } from "@/v4/offline/snapshot";
import type { PackRecord } from "@/v4/offline/packs";
import { homeHref, parseHome } from "@/v4/coast/home";

const D = (s: string) => Date.parse(`${s}T00:00:00Z`);

describe("icons are always sized", () => {
  it("a colour-only class gets the default 20 px and never shrinks; a caller's size is kept, alone", () => {
    expect(iconClass("text-n-ink-3")).toBe("shrink-0 h-5 w-5 text-n-ink-3");
    expect(iconClass("h-4 w-4 text-n-accent")).toBe("shrink-0 h-4 w-4 text-n-accent");
    expect(iconClass("size-6")).toBe("shrink-0 size-6");
    expect(iconClass("sm:h-6 sm:w-6")).toBe("shrink-0 sm:h-6 sm:w-6");
    expect(iconClass()).toBe("shrink-0 h-5 w-5");
    const svg = renderToStaticMarkup(createElement(Icon, { name: "chevronRight", className: "text-n-ink-3" }));
    expect(svg).toContain('class="shrink-0 h-5 w-5 text-n-ink-3"');
  });
});

describe("time-axis ticks fit their pixels", () => {
  // the plot is the chart's width less its 60 px of margins (SeriesChart, Research's model panel)
  const gaps = (from: number, to: number, W: number) => {
    const t = monthTicks(from, to, W);
    const px = (x: number) => ((x - from) / (to - from)) * (W - 60);
    return t.slice(1).map((x, i) => px(x) - px(t[i]));
  };
  const cases: [string, string, number][] = [
    ["2026-04-10", "2026-10-10", 700],
    ["2026-04-10", "2026-10-10", 280],
    ["2025-10-10", "2026-10-10", 320],
    ["2025-10-10", "2026-10-10", 1100],
    ["2023-10-10", "2026-10-10", 320],
    ["2014-01-01", "2026-10-10", 320],
    ["2014-01-01", "2026-10-10", 1100],
    ["2000-01-01", "2026-10-10", 300],
  ];
  for (const [a, b, W] of cases)
    it(`${a} to ${b} at ${W} px: labels at least ${TICK_LABEL_PX} px apart, and at least two of them`, () => {
      const g = gaps(D(a), D(b), W);
      expect(g.length).toBeGreaterThanOrEqual(1);
      expect(Math.min(...g)).toBeGreaterThanOrEqual(TICK_LABEL_PX);
    });

  it("months for a short window, calendar-aligned; years alone beyond about three years", () => {
    expect(monthTicks(D("2026-04-10"), D("2026-10-10"), 700).map((t) => new Date(t).toISOString().slice(0, 7))).toEqual(["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]);
    expect(tickLabel(D("2026-05-01"), D("2026-04-10"), D("2026-10-10"))).toBe("May 26");
    const long = monthTicks(D("2014-01-01"), D("2026-10-10"), 1100);
    expect(long.every((t) => new Date(t).getUTCMonth() === 0)).toBe(true);
    expect(tickLabel(long[0], D("2014-01-01"), D("2026-10-10"))).toMatch(/^\d{4}$/);
  });
});

describe("a saved copy is dated, never 'now'", () => {
  const rec = (savedAt: string, complete = true) => ({ id: "monterey", version: 2, savedAt, complete }) as unknown as PackRecord;
  it("offline with a complete pack: the saved time and its age; online or incomplete: none", () => {
    const now = new Date("2026-10-22T22:07:00Z");
    expect(savedSnapshot(rec("2026-10-19T22:07:32Z"), false, now)).toMatchObject({ savedAt: "2026-10-19T22:07:32Z", outdated: true });
    expect(savedSnapshot(rec("2026-10-19T22:07:32Z"), true, now)).toBeNull();
    expect(savedSnapshot(rec("2026-10-19T22:07:32Z", false), false, now)).toBeNull();
    expect(savedSnapshot(undefined, false, now)).toBeNull();
  });
  it("the age is said once the copy is outdated", () => {
    expect(ageWords({ ageHours: 5, outdated: false })).toBeNull();
    expect(ageWords({ ageHours: 30, outdated: true })).toBe("1 day old");
    expect(ageWords({ ageHours: 72.5, outdated: true })).toBe("3 days old");
  });
});

describe("the home port keeps its trip, and reads older values", () => {
  it("an object with place, activity and species", () => {
    const h = parseHome(JSON.stringify({ place: "port-550", for: "commercial", species: "dungeness" }));
    expect(h).toEqual({ place: "port-550", for: "commercial", species: "dungeness" });
    expect(homeHref(h!)).toBe("/coast?place=port-550&for=commercial&species=dungeness");
  });
  it("an older plain place key is that place with no trip; junk is nothing", () => {
    expect(parseHome("port-550")).toEqual({ place: "port-550", for: "all", species: "any" });
    expect(parseHome("point-36.72,-121.95")?.place).toBe("point-36.72,-121.95");
    expect(parseHome("{not json")).toBeNull();
    expect(parseHome(JSON.stringify({ for: "commercial" }))).toBeNull();
    expect(parseHome(null)).toBeNull();
  });
});
