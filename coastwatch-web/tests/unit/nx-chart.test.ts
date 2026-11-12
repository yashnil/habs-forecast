import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { coverCamera, fitBounds, formatDegrees, latY, lngX, project, scaleBar, screenRect, ticks, unproject, xLng, yLat, type Bounds } from "@/nx/chart/mercator";
import { boxAt, overlaps, placeAll, type PlaceItem } from "@/nx/chart/placement";
import { bloomRedirect, coastRedirect, hrefForPort, rootRedirect } from "@/nx/shell/redirects";
import { ENTRY_SCRIPT } from "@/nx/shell/entry";
import { activeDestination, withPlace } from "@/nx/shell/destinations";

const MONTEREY: Bounds = [
  [-122.3, 36.5],
  [-121.7, 37.0],
];

describe("chart projection (Web Mercator, MapLibre's 512 px world)", () => {
  it("round-trips lon/lat", () => {
    expect(xLng(lngX(-121.9))).toBeCloseTo(-121.9, 9);
    expect(yLat(latY(36.8))).toBeCloseTo(36.8, 9);
    const cam = { center: [-122, 36.8] as [number, number], zoom: 10.3 };
    const vp = { width: 1000, height: 700 };
    const p = project(cam, vp, -121.79, 36.81);
    const back = unproject(cam, vp, p.x, p.y);
    expect(back[0]).toBeCloseTo(-121.79, 8);
    expect(back[1]).toBeCloseTo(36.81, 8);
    expect(project(cam, vp, -122, 36.8)).toEqual({ x: 500, y: 350 });
  });

  it("fits a box inside the free area, centred in it", () => {
    const vp = { width: 1000, height: 800 };
    const insets = { top: 40, right: 40, bottom: 300, left: 40 };
    const cam = fitBounds(MONTEREY, vp, insets);
    const a = project(cam, vp, MONTEREY[0][0], MONTEREY[1][1]);
    const b = project(cam, vp, MONTEREY[1][0], MONTEREY[0][1]);
    // fully inside the free area, touching it on the limiting axis
    expect(a.x).toBeGreaterThanOrEqual(39.9);
    expect(b.x).toBeLessThanOrEqual(960.1);
    expect(a.y).toBeGreaterThanOrEqual(39.9);
    expect(b.y).toBeLessThanOrEqual(500.1);
    expect(Math.min(Math.abs(a.y - 40), Math.abs(b.x - 960))).toBeLessThan(0.5);
    // centred
    expect((a.x + b.x) / 2).toBeCloseTo(500, 3);
    expect((a.y + b.y) / 2).toBeCloseTo(270, 3);
  });

  it("covers a viewport exactly as CSS object-fit: cover does (the opening's still registers to the chart)", () => {
    for (const vp of [{ width: 1080, height: 820 }, { width: 380, height: 420 }, { width: 500, height: 300 }]) {
      const cam = coverCamera(MONTEREY, vp);
      const r = screenRect(cam, vp, MONTEREY);
      // covers, overflowing on one axis only, centred (object-position: 50% 50%)
      expect(r.left).toBeLessThanOrEqual(0.01);
      expect(r.top).toBeLessThanOrEqual(0.01);
      expect(r.left + r.width).toBeGreaterThanOrEqual(vp.width - 0.01);
      expect(r.top + r.height).toBeGreaterThanOrEqual(vp.height - 0.01);
      expect(Math.min(Math.abs(r.left), Math.abs(r.top))).toBeLessThan(0.01);
      expect(r.left + r.width / 2).toBeCloseTo(vp.width / 2, 4);
      expect(r.top + r.height / 2).toBeCloseTo(vp.height / 2, 4);
    }
  });

  it("gives the neatline round ticks and labels in degrees and minutes", () => {
    const t = ticks(-122.3, -121.7, 500);
    expect(t.stepMin).toBe(10);
    expect(t.major.map((v) => formatDegrees(v, "lon"))).toEqual(["122°10′ W", "122°00′ W", "121°50′ W"]);
    expect(t.minor.length).toBeGreaterThan(0);
    expect(formatDegrees(36.8333333, "lat")).toBe("36°50′ N");
    expect(formatDegrees(-121.75, "lon")).toBe("121°45′ W");
  });

  it("sizes a scale bar to a round distance no longer than asked", () => {
    const s = scaleBar(36.8, 10, 240);
    expect(s.px).toBeLessThanOrEqual(240);
    expect(s.label).toMatch(/^(5|10|20) km$/);
  });
});

describe("chart placement (soundings, notes, names)", () => {
  const area = { x0: 0, y0: 0, x1: 400, y1: 300 };
  const item = (id: string, x: number, y: number, priority: number, extra: Partial<PlaceItem> = {}): PlaceItem => ({ id, x, y, width: 80, height: 20, priority, ...extra });

  it("places by priority, never overlapping, and drops what does not fit", () => {
    const out = placeAll([item("low", 200, 150, 1, { anchors: ["right"] }), item("high", 200, 150, 9, { anchors: ["right"] })], area);
    const high = out.find((p) => p.id === "high")!;
    const low = out.find((p) => p.id === "low")!;
    expect(high.visible).toBe(true);
    expect(low.visible).toBe(false);
  });

  it("tries other anchors before dropping, and keeps the reader's port whatever happens", () => {
    const out = placeAll([item("a", 200, 150, 9, { anchors: ["right"] }), item("b", 200, 150, 5, { anchors: ["right", "left"] }), item("port", 200, 150, 1, { anchors: ["right"], keep: true })], area);
    expect(out.find((p) => p.id === "b")!.anchor).toBe("left");
    expect(out.find((p) => p.id === "port")!.visible).toBe(true);
  });

  it("keeps a label's previous anchor while the camera moves (no jumping)", () => {
    const prev = new Map([["a", "left" as const]]);
    expect(placeAll([item("a", 200, 150, 1)], area, [], prev)[0].anchor).toBe("left");
  });

  it("stays inside the free area and clear of obstacles", () => {
    const out = placeAll([item("edge", 390, 150, 1, { anchors: ["right", "left"] })], area, [{ x0: 280, y0: 100, x1: 330, y1: 200 }]);
    expect(out[0].visible).toBe(false);
    const r = boxAt(item("x", 100, 100, 1), "right");
    expect(overlaps(r, { x0: 150, y0: 95, x1: 160, y1: 105 })).toBe(true);
  });

  it("does not place a mark whose point is off the sheet", () => {
    expect(placeAll([item("off", 900, 150, 1)], area)[0].visible).toBe(false);
    expect(placeAll([item("off", 900, 150, 1, { insist: true })], area)[0].visible).toBe(false);
  });

  it("lets an insisting figure cover a lower place's dot rather than drop, but never a placed box or an obstacle", () => {
    // a neighbouring port's dot sits right of the figure's only anchor
    const dot = item("town", 270, 150, 1, { pointRadius: 6, width: 40, anchors: ["right"] });
    const fig = (extra: Partial<PlaceItem>) => item("fig", 200, 150, 9, { anchors: ["right"], ...extra });
    expect(placeAll([fig({}), dot], area).find((p) => p.id === "fig")!.visible).toBe(false);
    const placed = placeAll([fig({ insist: true }), dot], area);
    expect(placed.find((p) => p.id === "fig")).toMatchObject({ visible: true, anchor: "right" });
    // an obstacle (a panel, the scale) still wins
    expect(placeAll([fig({ insist: true }), dot], area, [{ x0: 250, y0: 100, x1: 260, y1: 200 }])[0].visible).toBe(false);
  });
});

describe("one entry: redirects of old addresses (journeys.md)", () => {
  it("sends Explore's old addresses to /explore with their parameters", () => {
    expect(rootRedirect({ view: "map" })).toBe("/explore");
    expect(rootRedirect({ sel: "harbor:592", layer: "forecast" })).toBe("/explore?sel=harbor%3A592&layer=forecast");
    expect(rootRedirect({ view: "map", at: "36.8,-121.9,9" })).toBe("/explore?at=36.8%2C-121.9%2C9");
  });
  it("keeps a bare front door, a campaign tag or the picker at /", () => {
    expect(rootRedirect({})).toBeNull();
    expect(rootRedirect({ utm_source: "x", fbclid: "y" })).toBeNull();
    expect(rootRedirect({ pick: "1" })).toBeNull();
  });
  it("sends /bloom to Research, renaming var and turning a year into a window", () => {
    expect(bloomRedirect({})).toBe("/research");
    expect(bloomRedirect({ station: "SCW", var: "pda", range: "1y" })).toBe("/research?station=SCW&variable=pda&range=1y");
    expect(bloomRedirect({ year: "2024" })).toBe("/research?start=2024-01-01&end=2024-12-31");
  });
  it("sends My Coast's research mode to Research", () => {
    expect(coastRedirect({ mode: "research", place: "port-592" })).toBe("/research?place=port-592");
    expect(coastRedirect({ place: "port-592" })).toBeNull();
  });
});

describe("the shell", () => {
  it("marks Today active at / only for a reader with a saved port", () => {
    expect(activeDestination("/", "port-592")).toBe("/today");
    expect(activeDestination("/", null)).toBeNull();
    expect(activeDestination("/explore", null)).toBe("/explore");
    expect(activeDestination("/about", "port-592")).toBeNull();
  });
  it("carries the port on every tab and on a port change", () => {
    expect(withPlace("/coast", "port-550")).toBe("/coast?place=port-550");
    expect(withPlace("/coast", null)).toBe("/coast");
    expect(hrefForPort("/research", "port-550")).toBe("/research?place=port-550");
    expect(hrefForPort("/", "port-550")).toBe("/today?place=port-550");
  });
});

describe("the pre-paint entry script", () => {
  // a tiny DOM stand-in: enough for the script's reads and writes
  function run(storage: Record<string, string>, opts: { path?: string; search?: string; reduced?: boolean; dark?: boolean } = {}) {
    const attrs: Record<string, string> = {};
    const store = { ...storage };
    const win = {
      localStorage: { getItem: (k: string) => store[k] ?? null, setItem: (k: string, v: string) => (store[k] = v) },
      WebGL2RenderingContext: function () {},
    };
    const doc = { documentElement: { setAttribute: (k: string, v: string) => (attrs[k] = v), getAttribute: (k: string) => attrs[k] ?? null } };
    const loc = { pathname: opts.path ?? "/", search: opts.search ?? "" };
    const mm = (q: string) => ({ matches: q.includes("reduced-motion") ? !!opts.reduced : q.includes("dark") ? !!opts.dark : false });
    const noop = () => 0;
    new Function("document", "localStorage", "location", "matchMedia", "window", "self", "navigator", "addEventListener", "removeEventListener", "setTimeout", ENTRY_SCRIPT)(doc, win.localStorage, loc, mm, win, win, { onLine: true }, noop, noop, noop);
    return { attrs, store };
  }

  it("opens Today at / for a saved port, from nx-port-v1 or the keys it migrates", () => {
    expect(run({ "nx-port-v1": JSON.stringify({ v: 1, lastPort: "port-592" }) }).attrs["data-nx-entry"]).toBe("today");
    expect(run({ "nx-today-place-v1": "port-550" }).attrs["data-nx-port"]).toBe("port-550");
    expect(run({ "cw-home-port": JSON.stringify({ place: "port-593", for: "all" }) }).attrs["data-nx-port"]).toBe("port-593");
    expect(run({ "cw-home-port": "site-12" }).attrs["data-nx-entry"]).toBe("home");
  });

  it("plays the opening once, for a first visit to a bare /", () => {
    const first = run({});
    expect(first.attrs["data-nx-opening"]).toBe("on");
    expect(first.store["nx-opening-seen-v1"]).toBe("1");
    expect(run(first.store).attrs["data-nx-opening"]).toBeUndefined();
  });

  it("never plays it under reduced motion, on a deep link or on another page", () => {
    expect(run({}, { reduced: true }).attrs["data-nx-opening"]).toBeUndefined();
    expect(run({}, { search: "?sel=harbor:1" }).attrs["data-nx-opening"]).toBeUndefined();
    expect(run({}, { search: "?utm_source=x" }).attrs["data-nx-opening"]).toBe("on");
    expect(run({}, { path: "/today" }).attrs["data-nx-opening"]).toBeUndefined();
  });

  it("sets the chart mode before the first paint: the saved one, else Dusk for a dark system; Night only when chosen", () => {
    expect(run({}).attrs["data-chart-mode"]).toBe("day");
    expect(run({}, { dark: true }).attrs["data-chart-mode"]).toBe("dusk");
    expect(run({ "nx-chart-mode-v1": "night" }, { path: "/explore" }).attrs["data-chart-mode"]).toBe("night");
  });
});

describe("chart assets", () => {
  const root = path.resolve(__dirname, "../..");
  it("stay small enough to ship (gzip-friendly geometry, compressed images)", () => {
    const meta = JSON.parse(readFileSync(path.join(root, "public/chart/chart.json"), "utf8"));
    expect(meta.isobaths.index_m).toEqual([200, 500, 1000, 2000, 3000]);
    expect(meta.images["opening-canyon.webp"].exaggeration).toBe(4);
    for (const f of ["depth.geojson", "isobaths.geojson", "land.geojson"]) {
      expect(readFileSync(path.join(root, "public/chart", f)).length, f).toBeLessThan(700_000);
    }
    expect(readFileSync(path.join(root, "public/chart/opening-canyon.webp")).length).toBeLessThan(80_000);
  });
});
