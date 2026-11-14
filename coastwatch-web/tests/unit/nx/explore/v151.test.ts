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
    // an inferred issue date is never shown: the run is dated by its valid days (round 2, C)
    expect(m("charm").when).toMatch(/^valid .+ \(expired\)$/);
    expect(m("charm").note).toMatch(/not a measured toxin/);
    expect(m("currents")).toMatchObject({ productClass: "observation", units: "kn · cm/s", role: "overlay" });
    expect(m("currents").when).toMatch(/\d+ hours to /);
    expect(m("stations").when).toMatch(/1 shore stations, newest sample Oct 6; 1 buoys read live/);
    expect(chipWords(m("charm"))).toMatch(/^Model · valid /);
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
    expect(notPublished(gated)).toEqual(["sea temperature", "the experimental chlorophyll outlook (not validated)"]);
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

import { outsideObservedArea } from "@/nx/explore/read";

describe("land and the open ocean read as outside the layer's area, never cloud (review j10)", () => {
  const shoreLon = (lat: number) => (lat > 32 && lat < 42.1 ? -122 + (lat - 37) * -0.5 : null);
  it("a cell in the Sierra the week never saw is outside", () => {
    expect(outsideObservedArea({ lat: 37.46, lon: -118.7, row: 10, col: 900, compositeEmpty: true, westmost: null, shoreLon })).toBe(true);
  });
  it("a cell at sea the week never saw stays cloud (no view)", () => {
    expect(outsideObservedArea({ lat: 37, lon: -123, row: 10, col: 100, compositeEmpty: true, westmost: null, shoreLon })).toBe(false);
  });
  it("west of the week's westernmost pixel is outside; east of it a gap is cloud", () => {
    const westmost = Int32Array.from([50]);
    expect(outsideObservedArea({ lat: 37, lon: -125, row: 0, col: 10, compositeEmpty: false, westmost, shoreLon })).toBe(true);
    expect(outsideObservedArea({ lat: 37, lon: -123, row: 0, col: 60, compositeEmpty: false, westmost, shoreLon })).toBe(false);
  });
  it("north or south of California, an unseen cell is outside", () => {
    expect(outsideObservedArea({ lat: 43, lon: -124.5, row: 0, col: 10, compositeEmpty: true, westmost: null, shoreLon })).toBe(true);
  });
});

import { DEFAULT_PREFS, parsePrefs } from "@/nx/explore/prefs";
import { expiredWords, familyOf } from "@/nx/explore/TimeRail";

describe("round 2 (A): Explore's choices kept on the device", () => {
  it("a version-1 record is read; anything else gives the defaults", () => {
    expect(parsePrefs(JSON.stringify({ v: 1, base: "chart", opacity: 0.5, depth: false, regulation: true, currents: true, stations: true, rail: true }))).toEqual({ v: 1, base: "chart", opacity: 0.5, depth: false, regulation: true, currents: true, stations: true, rail: true });
    expect(parsePrefs(null)).toEqual(DEFAULT_PREFS);
    expect(parsePrefs("{not json")).toEqual(DEFAULT_PREFS);
    expect(parsePrefs(JSON.stringify({ v: 2, base: "chart" }))).toEqual(DEFAULT_PREFS);
  });
  it("odd values are clamped or dropped, never trusted", () => {
    const p = parsePrefs(JSON.stringify({ v: 1, base: "<script>", opacity: 7, depth: "yes" }));
    expect(p.base).toBe("chl");
    expect(p.opacity).toBe(1);
    expect(p.depth).toBe(true);
    expect(parsePrefs(JSON.stringify({ v: 1, opacity: 0.01 })).opacity).toBe(0.2);
  });
});

describe("round 2 (E): the time rail's families", () => {
  it("measured layers are observations, analyses estimates, models and outlooks forecasts", () => {
    expect(familyOf({ kind: "measured" })).toBe("obs");
    expect(familyOf({ kind: "analysis" })).toBe("est");
    expect(familyOf({ kind: "model" })).toBe("fc");
    expect(familyOf({ kind: "outlook" })).toBe("fc");
  });
  it("an expired run says its last run, its valid interval and that it is not current", () => {
    const w = expiredWords({ runIssued: "2026-11-03", frames: [{ start: "2026-11-02", end: "2026-11-02" }, { start: "2026-11-05", end: "2026-11-05" }] as never });
    expect(w).toMatch(/^Last run Nov 3, valid Nov 2–5\. Expired: .*not a current forecast\.$/);
  });
});

import { chlEstimateStateWords, chlEstimateUsable, maskedCodes, maskedVectors, wcofsStateWords, wcofsTracks, wcofsUsable } from "@/nx/explore/ocean";

describe("round 2 (B, D): the live model and estimate layers", () => {
  it("near shore and land are left out of a model grid and its vectors (mask: 0 land, 1 water, 2 near shore)", () => {
    const mask = Uint16Array.from([1, 2, 0, 1]);
    expect([...maskedCodes({ nodata: 65535 }, Uint16Array.from([5, 6, 7, 8]), mask)]).toEqual([5, 65535, 65535, 8]);
    const m = maskedVectors(Float32Array.from([1, 1, 1, 1]), Float32Array.from([2, 2, 2, 2]), mask);
    expect([...m.u].map((x) => (Number.isNaN(x) ? null : x))).toEqual([1, null, null, 1]);
  });
  const step = (id: string, kind: "nowcast" | "forecast", t: string) => ({ id, kind, track: kind === "nowcast" ? "estimates" : "forecasts", lead_h: 0, valid_time: t, grids: {} }) as never;
  const ix = { status: "current", reason: null, source: {}, cycle: { id: "2026111403", issued: "", steps_available: 2, steps_expected: 2 }, valid: { start: "2026-11-13T06:00:00Z", end: "2026-11-17T03:00:00Z" }, fallback: false, default_step: "f006", steps: [step("f006", "forecast", "2026-11-14T09:00:00Z"), step("n024", "nowcast", "2026-11-14T03:00:00Z")], families: {}, near_shore_km: 6, caveats: [], generated_at: "" } as never;
  it("nowcast steps go to Analysis / Estimates, forecast steps to Forecasts, each in time order", () => {
    const t = wcofsTracks(ix);
    expect(t.nowcast.map((x) => x.id)).toEqual(["n024"]);
    expect(t.forecast.map((x) => x.id)).toEqual(["f006"]);
    expect(wcofsUsable(ix)).toBe(true);
    expect(wcofsStateWords(ix, true)).toMatch(/^NOAA WCOFS model guidance \(developmental\) · cycle Nov 14 03z/);
  });
  it("an unavailable source says so and is never offered", () => {
    const u = { status: "unavailable", reason: "upstream did not answer", cycle: null, steps: [] } as never;
    expect(wcofsUsable(u)).toBe(false);
    expect(wcofsStateWords(u, true)).toBe("NOAA WCOFS is unavailable: upstream did not answer.");
    expect(chlEstimateUsable({ status: "unavailable", grids: null, date: null } as never)).toBe(false);
  });
  it("a stale estimate is offered, dated and said stale", () => {
    const c = { status: "stale", date: "2026-11-02", age_days: 12, grids: { chl: {}, mask: {} }, reason: null } as never;
    expect(chlEstimateUsable(c)).toBe(true);
    expect(chlEstimateStateWords(c, true)).toMatch(/^Gap-filled estimate \(NOAA DINEOF\), Nov 2, stale: 12 days old, no newer day published\. Reconstructed .* not observed\.$/);
  });
});

import { extendAxis, pickServedStep, playableSteps, stepsAxis } from "@/nx/explore/ocean";

describe("round 2: one shared, true axis; served steps only", () => {
  it("the axis widens to the latest model time and the estimate's day, never narrows", () => {
    expect(extendAxis(["2026-10-29", "2026-11-14"], ["2026-11-17T03:00:00Z"], ["2026-11-02"])).toEqual(["2026-10-29", "2026-11-16"]);
    expect(extendAxis(["2026-10-29", "2026-11-14"], [], ["2026-10-20"])).toEqual(["2026-10-20", "2026-11-14"]);
    expect(stepsAxis([{ valid_time: "2026-11-17T03:00:00Z" }, { valid_time: "2026-11-13T06:00:00Z" }])).toEqual(["2026-11-12", "2026-11-16"]);
  });
  it("opens on the served step nearest to now, trying nearest first", async () => {
    const steps = [{ id: "n003", valid_time: "2026-11-13T06:00:00Z" }, { id: "n024", valid_time: "2026-11-14T03:00:00Z" }, { id: "f006", valid_time: "2026-11-14T09:00:00Z" }];
    const tried: string[] = [];
    const st = await pickServedStep(steps, "2026-11-14T08:00:00Z", async (s) => (tried.push(s.id), s.id === "n024"));
    expect(st?.id).toBe("n024");
    expect(tried).toEqual(["f006", "n024"]);
    expect(await pickServedStep(steps, "2026-11-14T08:00:00Z", async () => false)).toBeNull();
  });
  it("only forecast steps play, in order, without the unserved ones", () => {
    const s = [{ id: "f006", kind: "forecast" as const, valid_time: "2026-11-14T09:00:00Z" }, { id: "n024", kind: "nowcast" as const, valid_time: "2026-11-14T03:00:00Z" }, { id: "f003", kind: "forecast" as const, valid_time: "2026-11-14T06:00:00Z" }];
    expect(playableSteps(s, new Set()).map((x) => x.id)).toEqual(["f003", "f006"]);
    expect(playableSteps(s, new Set(["f003"])).map((x) => x.id)).toEqual(["f006"]);
  });
});

import { isolineLabels, traceIsolines } from "@/nx/chart/layers/contours";

describe("isotherms traced on the CPU: nothing along a mask or the grid's edge, no tiny rings", () => {
  const N = 65535;
  const grid = (w: number, h: number) => ({ width: w, height: h, lat_first: 36, lat_step: 0.04, lon_first: -123, lon_step: 0.04, scale_factor: 1, add_offset: 0, nodata: N, transform: "none" as const });
  it("a cell with a missing corner emits no segment", () => {
    // a west-east gradient 0..4 with the middle column masked: no line may cross the masked column
    const w = 5, h = 5;
    const codes = new Uint16Array(w * h);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) codes[r * w + c] = c === 2 ? N : c;
    const lines = traceIsolines(grid(w, h), codes, { step: 1, index: 2, toValue: (k) => k });
    for (const l of lines) for (const [lon] of l.coords) expect(Math.abs(lon - (-123 + 2 * 0.04))).toBeGreaterThan(0.039);
    expect(lines.some((l) => l.level === 0.5 || l.level === 1)).toBe(true);
  });
  it("a tiny closed ring is dropped below the minimum length", () => {
    const w = 6, h = 6;
    const codes = new Uint16Array(w * h).fill(10);
    codes[2 * w + 2] = 20; // one warm cell: a ring of about one cell
    expect(traceIsolines(grid(w, h), codes, { step: 5, index: 2, toValue: (k) => k }).length).toBeGreaterThan(0);
    expect(traceIsolines(grid(w, h), codes, { step: 5, index: 2, toValue: (k) => k, minKm: 12 }).length).toBe(0);
  });
  it("labels the longest line of a few major levels in degrees", () => {
    const w = 8, h = 8;
    const codes = new Uint16Array(w * h);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) codes[r * w + c] = 60 + c;
    const labels = isolineLabels(traceIsolines(grid(w, h), codes, { step: 1, index: 2, toValue: (k) => k }));
    expect(labels.length).toBeGreaterThan(0);
    expect(labels.every((l) => /^\d+°$/.test(l.text) && Number(l.text.replace("°", "")) % 2 === 0)).toBe(true);
  });
});

import { expiredWords as railExpired } from "@/nx/explore/TimeRail";
import { leadSentence, legendCaption, layerKindWord, shortLine } from "@/nx/explore/words";

describe("final pass: nothing expired, stale or old ever reads as current", () => {
  const m = layerMetas(full);
  const charm = full.products.find((p) => p.id === "charm")!;
  const chl = full.products.find((p) => p.id === "chl")!;
  it("an expired C-HARM run says expired or past run in its chip, kind word, lead, legend and rail", () => {
    const meta = m.find((x) => x.id === "charm")!;
    expect(meta.when).toMatch(/\(expired\)$/);
    expect(meta.fresh).toBe("historical");
    expect(chipWords(meta)).toMatch(/^Model · .*\(expired\)$/);
    expect(layerKindWord(charm)).toMatch(/^past run/);
    expect(leadSentence(charm, charm.frames[0])).toMatch(/expired/);
    expect(shortLine(charm)).toMatch(/expired/);
    expect(legendCaption(charm, charm.frames[0])).toMatch(/Past run: every day it covers is before today/);
    expect(railExpired(charm)).toMatch(/expired .*not a current forecast/);
  });
  it("old satellite days are marked (old) wherever their dates are shown", () => {
    const meta = m.find((x) => x.id === "chl")!;
    expect(meta.when).toMatch(/\(old\)$/);
    expect(legendCaption(chl, chl.frames[0])).toMatch(/\(old\)/);
  });
  it("the stale estimate says stale and its age", () => {
    const c = { status: "stale", date: "2026-11-02", age_days: 12, grids: { chl: {}, mask: {} }, reason: null } as never;
    expect(chlEstimateStateWords(c, true)).toMatch(/stale: 12 days old/);
  });
});
