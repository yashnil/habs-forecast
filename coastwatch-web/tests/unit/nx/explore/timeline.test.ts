import { describe, expect, it } from "vitest";
import type { ExploreFrame, ExploreProduct } from "@/nx/explore/model";
import { axisLength, axisX, dayFraction, nearestFrame, nextOf, nowX, playFrames, playHours } from "@/nx/explore/timeline";
import { exploreBounds, CA_COAST, EXPLORE_ZOOM } from "@/nx/explore/camera";

const f = (id: string, start: string, end = start, kind: ExploreFrame["kind"] = "measured"): ExploreFrame => ({ id, product: "chl", track: "olci", label: id, stamp: id, start, end, kind, past: false, volume: 0.3, fresh: "current", grid: null, sensor: "x" });
const prod = (kind: ExploreProduct["kind"], frames: ExploreFrame[]): ExploreProduct => ({ id: "chl", name: "", unit: "", kind, source: "", sourceUrl: null, frames, tracks: [], defaultFrame: frames[0].id });
const axis: [string, string] = ["2026-10-01", "2026-10-10"];

describe("the time rail (T26, T27)", () => {
  it("places days on a Pacific day axis", () => {
    expect(axisLength(axis)).toBe(10);
    expect(axisX(axis, "2026-10-01")).toBe(0);
    expect(axisX(axis, "2026-10-10", 1)).toBe(1);
  });
  it("puts the now line inside today's column", () => {
    expect(dayFraction("2026-10-13T06:00:00Z")).toBeCloseTo(23 / 24, 3);
    const x = nowX(["2026-10-01", "2026-10-20"], "2026-10-13T06:00:00Z");
    expect(x).toBeGreaterThan(axisX(["2026-10-01", "2026-10-20"], "2026-10-12"));
    expect(x).toBeLessThan(axisX(["2026-10-01", "2026-10-20"], "2026-10-13"));
  });
  it("a tap picks the frame whose span holds it, else the nearest", () => {
    const fr = [f("latest", "2026-10-01", "2026-10-06"), f("d8", "2026-10-08")];
    expect(nearestFrame(axis, fr, axisX(axis, "2026-10-03", 0.5))?.id).toBe("latest");
    expect(nearestFrame(axis, fr, axisX(axis, "2026-10-09", 0.5))?.id).toBe("d8");
  });
  it("plays only single-day measured frames, in order, never a latest view or a model run", () => {
    const p = prod("measured", [f("olci:latest", "2026-10-01", "2026-10-06"), f("d5", "2026-10-05"), f("d2", "2026-10-02")]);
    expect(playFrames(p).map((x) => x.id)).toEqual(["d2", "d5"]);
    expect(playFrames(prod("model", [f("a", "2026-10-02", "2026-10-02", "model"), f("b", "2026-10-03", "2026-10-03", "model")]))).toEqual([]);
    expect(playFrames(prod("measured", [f("only", "2026-10-02")]))).toEqual([]);
  });
  it("plays hours without the 24 h mean and wraps", () => {
    const h = (id: string, t: string, mean = false) => ({ id, label: id, stamp: id, date: t.slice(0, 10), time: t, mean, volume: 1, fresh: "current" as const, u: null as never, v: null as never });
    const hours = playHours([h("b", "2026-10-08T02:00:00Z"), h("m", "2026-10-08T12:00:00Z", true), h("a", "2026-10-08T01:00:00Z")]);
    expect(hours.map((x) => x.id)).toEqual(["a", "b"]);
    expect(nextOf(hours, "b")?.id).toBe("a");
  });
});

describe("the opening view", () => {
  const extent: [number, number, number, number] = [-122.6, 36.4, -121.75, 37.2];
  it("without a port, frames the data", () => {
    expect(exploreBounds(null, extent)).toEqual([
      [-122.6, 36.4],
      [-121.75, 37.2],
    ]);
    expect(exploreBounds(null, null)).toEqual(CA_COAST);
  });
  it("with a port, leans seaward and stays within a sliver of the data (v11 opened on empty sea)", () => {
    const [[w, s], [e, n]] = exploreBounds({ lon: -121.79, lat: 36.8 }, extent);
    expect(w).toBeGreaterThanOrEqual(-122.6 - 0.03);
    expect(e).toBeGreaterThan(-121.79);
    expect(n - s).toBeGreaterThan(0.5);
    expect((-121.79 - w) / (e - w)).toBeGreaterThan(0.7);
  });
  it("clamps the zoom between the state and a harbour", () => {
    expect(EXPLORE_ZOOM[0]).toBeGreaterThan(4);
    expect(EXPLORE_ZOOM[1]).toBeLessThan(13.5);
  });
});
