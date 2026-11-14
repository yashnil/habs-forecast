import { describe, expect, it } from "vitest";
import type { Manifest } from "@/generated/schema";
import { parseDerived } from "@/lib/derived";
import manifest from "../../../fixture-data/v1/manifest.json";
import derivedRaw from "../../../fixture-derived/v1/derived/index.json";
import { buildExploreModel } from "@/nx/explore/model";
import { COAST_ZOOM, holdsCoast, MEXICO_BORDER, OREGON_BORDER, STATE_BOUNDS } from "@/nx/explore/camera";
import { cameraFor } from "@/nx/chart/ChartHost";
import { unproject } from "@/nx/chart/mercator";
import { cellKmOf, chipWords, kmPerPixel, layerMetas, notPublished, resolutionNote, speedStats, speedWords } from "@/nx/explore/layers";
import { backLink, exploreHref, readFrom, readView } from "@/nx/explore/links";
import { derivedEnabled } from "@/nx/explore/load.server";
import { activeDestination, destinationHref, DESTINATIONS } from "@/nx/shell/destinations";

/** v15.1 Explore California Waters: camera bounds, the layer switch's metadata, missing data, the address. */
const now = new Date("2026-10-13T06:00:00Z");
const buoys = [{ id: "46042", name: "MONTEREY - 27NM WNW of Monterey, CA", lat: 36.785, lon: -122.398, owner: "NDBC", type: "buoy" }];
const stations = [{ station_id: "HABs-SantaCruzWharf", name: "Santa Cruz Wharf", lat: 36.958, lon: -122.017, status: "updated", source_url: "", latest: [{ variable: "pDA", label: "Particulate domoic acid", units: "ng/mL", kind: "toxin", value: 0.12, date: "2026-10-06", qualifier: null, n_last_365d: 40 }] }];
const full = buildExploreModel({ manifest: manifest as unknown as Manifest, derived: parseDerived(derivedRaw), stations, baseUrl: "/data/fixture/v1", now, buoys });
const gated = buildExploreModel({ manifest: manifest as unknown as Manifest, derived: parseDerived(derivedRaw), stations, baseUrl: "https://x/v1", now, buoys, derivedOn: false });

describe("statewide camera", () => {
  it("the statewide box holds the Oregon border and the Mexican border", () => {
    expect(STATE_BOUNDS[0][1]).toBeLessThan(MEXICO_BORDER);
    expect(STATE_BOUNDS[1][1]).toBeGreaterThan(OREGON_BORDER);
    expect(holdsCoast([-125, 32.4, -117, 42.1])).toBe(true);
    expect(holdsCoast([-125, 33, -117, 42.1])).toBe(false);
  });
  for (const [w, h, cover] of [
    [1440, 900, { top: 0, right: 0, bottom: 0, left: 400 }],
    [1280, 800, { top: 0, right: 0, bottom: 0, left: 400 }],
    [1100, 800, { top: 0, right: 0, bottom: 0, left: 400 }],
    [390, 844, { top: 0, right: 0, bottom: 420, left: 0 }],
    [320, 640, { top: 0, right: 0, bottom: 320, left: 0 }],
  ] as const) {
    it(`fits Oregon to Mexico in the free part of a ${w}×${h} sheet`, () => {
      const cam = cameraFor({ id: "explore:state", view: { bounds: STATE_BOUNDS }, zoomRange: COAST_ZOOM }, { width: w, height: h }, cover);
      // the free part of the sheet: under the bar, above the phone sheet, right of a desk panel
      const n = unproject(cam, { width: w, height: h }, (cover.left + w) / 2, cover.top)[1];
      const s = unproject(cam, { width: w, height: h }, (cover.left + w) / 2, h - cover.bottom)[1];
      expect(s).toBeLessThanOrEqual(MEXICO_BORDER);
      expect(n).toBeGreaterThanOrEqual(OREGON_BORDER);
      expect(cam.zoom).toBeGreaterThanOrEqual(COAST_ZOOM[0]);
    });
  }
});

describe("layer metadata", () => {
  const metas = layerMetas(full);
  const m = (id: string) => metas.find((x) => x.id === id)!;
  it("every live layer says observation or model, its units and its dates", () => {
    expect(m("chl")).toMatchObject({ productClass: "observation", units: "mg/m³", role: "surface" });
    expect(m("chl").when).toMatch(/Oct/);
    expect(m("charm")).toMatchObject({ productClass: "model", units: "probability (%)" });
    expect(m("charm").when).toMatch(/^run of .+, valid .+\(past run\)$/);
    expect(m("charm").note).toMatch(/not a measured toxin/);
    expect(m("currents")).toMatchObject({ productClass: "observation", units: "kn · cm/s", role: "overlay" });
    expect(m("currents").when).toMatch(/\d+ hours to /);
    expect(m("stations").when).toMatch(/1 shore stations, newest sample Oct 6; 1 buoys read live/);
    expect(chipWords(m("charm"))).toMatch(/^Model · run of/);
    expect(chipWords(m("chl"))).toMatch(/^Observation · /);
  });
  it("chlorophyll is never called toxin", () => {
    expect(m("chl").note).toMatch(/Algae, not toxin/);
  });
});

describe("missing and unpublished data", () => {
  it("deployed data reads no derived layers until they are published; a local fixture does", () => {
    expect(derivedEnabled("https://raw.githubusercontent.com/x/coastwatch-data-staging/v1", undefined)).toBe(false);
    expect(derivedEnabled("https://raw.githubusercontent.com/x/coastwatch-data/v1", "1")).toBe(true);
    expect(derivedEnabled("/data/fixture/v1", undefined)).toBe(true);
    expect(derivedEnabled("/data/fixture/v1", "0")).toBe(false);
  });
  it("gated: SST and the outlook are not live; they are named as not published yet", () => {
    expect(gated.products.map((p) => p.id)).toEqual(["chl", "charm"]);
    expect(gated.missing.filter((x) => x.id === "sst" || x.id === "outlook").every((x) => x.why === "Not published yet")).toBe(true);
    expect(notPublished(gated)).toEqual(["sea temperature", "the 8-day chlorophyll outlook"]);
    expect(layerMetas(gated).some((x) => x.id === "sst" || x.id === "outlook")).toBe(false);
    expect(notPublished(full)).toEqual([]);
  });
  it("speeds come only from cells with both components; an empty hour has none", () => {
    expect(speedStats([NaN, NaN], [NaN, 1])).toBeNull();
    const st = speedStats([0.3, NaN, 0, 0.6], [0.4, 0.2, 0.1, 0.8])!;
    expect(st.cells).toBe(3);
    expect(st.max).toBeCloseTo(1);
    expect(st.median).toBeCloseTo(0.5);
    expect(speedWords(0.5)).toBe("0.97 kn (50 cm/s)");
  });
  it("the resolution line appears when a pixel spans many cells, and goes once the cells show", () => {
    const cell = cellKmOf({ lat_step: -0.0025 })!;
    expect(cell).toBeCloseTo(0.278, 2);
    expect(kmPerPixel(4.5, 37)).toBeGreaterThan(2);
    expect(resolutionNote(4.5, 37, cell)).toMatch(/one screen pixel spans about .* km, about \d+ cells of 278 m/);
    expect(resolutionNote(11, 37, cell)).toBeNull();
  });
});

describe("the address and the way back", () => {
  it("builds and reads the URL contract", () => {
    expect(exploreHref({ view: "coast", place: "port-592" })).toBe("/explore?view=coast&place=port-592");
    expect(exploreHref({ view: "coast", place: "port-592", layer: "charm", stations: true, from: "trip", trip: "t-abc-123" })).toBe("/explore?view=coast&place=port-592&layer=charm&st=1&from=trip&trip=t-abc-123");
    const sp = new URLSearchParams("view=coast&from=trip&trip=t-abc-123");
    expect(readView(sp)).toBe("coast");
    expect(readView(new URLSearchParams(""))).toBe("port");
    expect(readFrom(sp)).toEqual({ kind: "trip", id: "t-abc-123" });
    expect(readFrom(new URLSearchParams("from=trip"))).toEqual({ kind: "today" });
    expect(readFrom(new URLSearchParams("from=trip&trip=<x>"))).toEqual({ kind: "today" });
  });
  it("back goes to the trip, Research or Today, carrying the port", () => {
    expect(backLink({ kind: "trip", id: "t-1" }, "port-592")).toEqual({ href: "/coast/trips?id=t-1", label: "Back to your trip" });
    expect(backLink({ kind: "today" }, "port-592")).toEqual({ href: "/today?place=port-592", label: "Back to Today" });
    expect(backLink({ kind: "research" }, null)).toEqual({ href: "/research", label: "Back to Research" });
  });
  it("Explore is a section of its own; its link opens the whole coast with the port", () => {
    expect(DESTINATIONS.map((d) => d.label)).toEqual(["Today", "Explore", "My Coast", "Research", "Alerts"]);
    expect(activeDestination("/explore", null)).toBe("/explore");
    expect(activeDestination("/notices", null)).toBe("/today");
    expect(destinationHref("/explore", "port-592")).toBe("/explore?view=coast&place=port-592");
    expect(destinationHref("/coast", "port-592")).toBe("/coast?place=port-592");
  });
});

import { markForDisplay, westmostPerRow } from "@/nx/chart/layers/marks";
import { axisLabels } from "@/nx/explore/timeline";

describe("three meanings of no value, drawn apart (review j9)", () => {
  const N = 65535;
  // 2 rows x 4 cols, chunks of 1 row x 2 cols: row 0 has a value at col 2; row 1 none
  const g = { width: 4, height: 2, lat_first: 1, lat_step: -1, lon_first: 0, lon_step: 1, scale_factor: 1, add_offset: 0, nodata: N, transform: "none" as const, chunks: { rows: 1, cols: 2, present: ["0_0", "0_1", "1_0", "1_1"] } };
  const codes = Uint16Array.from([N, N, 5, N, N, N, N, N]);
  it("west of the week's westernmost pixel is outside; a gap east of it stays cloud; the published codes are untouched", () => {
    const west = westmostPerRow(g, codes);
    expect([...west]).toEqual([2, -1]);
    const m = markForDisplay(g, codes, { westmost: west });
    expect([...m.codes.slice(0, 4)]).toEqual([m.outCode, m.outCode, 5, N]);
    expect([...m.codes.slice(4)]).toEqual([m.outCode, m.outCode, m.outCode, m.outCode]);
    expect(codes[0]).toBe(N);
  });
  it("cells of a chunk not loaded yet are marked loading, never cloud", () => {
    const m = markForDisplay(g, codes, { loaded: new Set(["0_0", "0_1"]) });
    expect([...m.codes.slice(4)]).toEqual([m.loadingCode, m.loadingCode, m.loadingCode, m.loadingCode]);
    expect([...m.codes.slice(0, 4)]).toEqual([N, N, 5, N]);
  });
});

describe("the day axis thins on a regular beat (review: labels dropped at 1100)", () => {
  it("every day when there is room, else every second day, never a ragged gap", () => {
    const wide = axisLabels(["2026-11-02", "2026-11-14"], 1200, 2000);
    expect(wide.length).toBe(13);
    const narrow = axisLabels(["2026-11-02", "2026-11-14"], 260, 2000);
    const idx = narrow.map((l) => Number(l.date.slice(8)) - 2);
    const gaps = new Set(idx.slice(1).map((v, i) => v - idx[i]));
    expect(gaps.size).toBe(1);
  });
});
