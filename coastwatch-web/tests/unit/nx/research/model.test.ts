import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { ObservationDataset, ObsStation } from "@/generated/observations";
import {
  buildStrips,
  chooseStation,
  clampWin,
  DAY,
  extentOf,
  figure,
  findStation,
  linearTicksFor,
  located,
  logDomainOf,
  logRows,
  logTicksFor,
  mapVar,
  modelVarFor,
  northToSouth,
  openingWindow,
  parseQuery,
  pointsOf,
  presetOf,
  presetWindow,
  queryFor,
  readingAt,
  runsOf,
  tickLabel,
  valueWords,
  windowStats,
  type RecPoint,
  type Strip,
} from "@/nx/research/model";

const dir = path.resolve(__dirname, "../../../fixture-data/v1");
const ds = JSON.parse(readFileSync(path.join(dir, readdirSync(dir).find((f) => f.startsWith("observations-"))!), "utf8")) as ObservationDataset;
const stations = northToSouth(located(ds));
const scw = findStation(stations, "HABs-SantaCruzWharf")!;
const q = (o: Record<string, string>) => new URLSearchParams(o);
const D = (s: string) => Date.parse(`${s}T12:00:00Z`);

describe("the address", () => {
  it("reads v12's station, from, to and var", () => {
    const r = parseQuery(q({ station: "HABs-MontereyWharf", from: "2025-01-01", to: "2025-06-30", var: "pn_seriata" }));
    expect(r).toMatchObject({ station: "HABs-MontereyWharf", v: "pn_seriata", from: D("2025-01-01"), to: D("2025-06-30"), back: null });
  });
  it("maps v11's parameters: variable, start/end, range, day, from=/path, #r- sections", () => {
    const r = parseQuery(q({ variable: "particulate_domoic", start: "2024-01-01", end: "2024-12-31", range: "1y", day: "2024-06-04", from: "/coast?place=port-592" }), "#r-provenance");
    expect(r.v).toBe("pDA");
    expect([r.from, r.to]).toEqual([D("2024-01-01"), D("2024-12-31")]);
    expect(r.range).toBe("1y");
    expect(r.day).toBe(D("2024-06-04"));
    expect(r.back).toBe("/coast?place=port-592");
    expect(r.section).toBe("checks");
    expect(parseQuery(q({ from: "//evil.example" })).back).toBeNull();
  });
  it("reads Explore's layer names as the nearest thing a station measures, and says when it cannot", () => {
    expect(mapVar("pseudo_nitzschia")).toBe("pn_seriata");
    expect(mapVar("olci300")).toBe("chl_extracted");
    expect(mapVar("sst")).toBe("temp");
    expect(mapVar("currents")).toBeNull();
    expect(parseQuery(q({ var: "currents" })).askedVar).toBe("currents");
    expect(modelVarFor("pn_delicatissima")).toBe("pseudo_nitzschia");
    expect(modelVarFor("chl_extracted")).toBe("particulate_domoic");
  });
  it("writes the canonical address", () => {
    const s = queryFor({ station: "HABs-SantaCruzWharf", v: "pDA", win: [D("2024-01-01"), D("2025-01-01")], place: "port-592" });
    expect(Object.fromEntries(new URLSearchParams(s))).toEqual({ place: "port-592", station: "HABs-SantaCruzWharf", var: "pDA", from: "2024-01-01", to: "2025-01-01" });
  });
});

describe("the station", () => {
  it("lists stations north to south", () => {
    expect(stations[0].lat).toBeGreaterThan(stations.at(-1)!.lat);
  });
  it("finds a station by id, short id or name", () => {
    expect(findStation(stations, "SantaCruzWharf")?.station_id).toBe("HABs-SantaCruzWharf");
    expect(findStation(stations, "monterey wharf")?.station_id).toBe("HABs-MontereyWharf");
    expect(findStation(stations, "nowhere")).toBeNull();
  });
  it("opens the link's station; else the place's nearest, with the distance; else Santa Cruz Wharf", () => {
    expect(chooseStation(stations, { station: "HABs-ScrippsPier" }, null, null).station.station_id).toBe("HABs-ScrippsPier");
    const moss = { key: "port-592", name: "Moss Landing", lat: 36.805, lon: -121.786 };
    const c = chooseStation(stations, { station: null }, moss, null);
    expect(c.arrival).toMatchObject({ kind: "place", name: "Moss Landing", near: true });
    expect(chooseStation(stations, { station: null }, null, null).station.station_id).toBe("HABs-SantaCruzWharf");
  });
  it("says a far place has no station near it", () => {
    const off = { key: "point-36.0,-125.5", name: "Offshore", lat: 36, lon: -125.5 };
    expect(chooseStation(stations, { station: null }, off, null).arrival?.near).toBe(false);
  });
});

describe("the record", () => {
  it("keeps a reported 0 as its own class and leaves visits that did not measure out (never zero)", () => {
    const p = pointsOf(scw, "pDA");
    expect(p.some((x) => x.cls === "zero" && x.v === 0)).toBe(true);
    const series = scw.series.find((s) => s.variable === "pDA")!;
    expect(p.length).toBe(series.values.filter((v) => v != null).length);
  });
  it("classifies rejected and flagged samples", () => {
    const st = { sample_times: ["2025-01-01T00:00:00Z", "2025-01-08T00:00:00Z", "2025-01-15T00:00:00Z"], series: [{ variable: "pDA", values: [null, 50, 0.1], qualifiers: { "0": "rejected_negative", "1": "flag_high" } }] } as unknown as ObsStation;
    const p = pointsOf(st, "pDA");
    expect(p.map((x) => x.cls)).toEqual(["rejected", "value", "value"]);
    expect(p[1].flagged).toBe(true);
    expect(p[0].v).toBeNull();
  });
  it("builds one strip per measured quantity, toxin first, then the model strip that follows the focus", () => {
    const s = buildStrips(scw, "pDA");
    expect(s.map((x) => x.id)).toEqual(["pDA", "pn_seriata", "chl_extracted", "temp", "model"]);
    expect(s.find((x) => x.id === "model")?.label).toContain("pDA > 500 ng/L");
    expect(buildStrips(scw, "pn_seriata").find((x) => x.id === "model")?.label).toContain("10,000 cells/L");
    expect(s.find((x) => x.id === "temp")?.scale).toBe("linear");
    expect(s.find((x) => x.id === "pDA")?.scale).toBe("log");
  });
  it("gives a station without model history no model strip", () => {
    expect(buildStrips(findStation(stations, "HABs-ScrippsPier")!, "pDA").some((x) => x.kind === "model")).toBe(false);
  });
  it("spans log axes over whole decades, at least two", () => {
    expect(logDomainOf([0.0013, 11])).toEqual([0.001, 100]);
    expect(logDomainOf([3, 4])).toEqual([1, 100]);
    expect(logDomainOf([])).toEqual([0.01, 1]);
  });
  it("thins decade labels so they never collide", () => {
    const t = logTicksFor([1e-4, 1e2], 60);
    expect(t).toHaveLength(7);
    expect(t.filter((x) => x.label).length).toBeLessThan(7);
    expect(t.at(-1)!.label).toBe(true);
    expect(linearTicksFor([0, 100], 40).map((x) => x.v)).toEqual([0, 50, 100]);
    expect(tickLabel(0.001)).toBe("0.001");
    expect(tickLabel(10_000)).toBe("10K");
    expect(tickLabel(0)).toBe("0");
  });
});

describe("gaps and runs", () => {
  const pt = (t: string, v: number | null, cls: RecPoint["cls"] = "value"): RecPoint => ({ t: D(t), time: t, i: 0, v, cls, flagged: false });
  const strip = (points: RecPoint[], extra: Partial<Strip> = {}): Strip => ({ id: "pDA", v: "pDA", kind: "measured", label: "", unit: "", unitShort: "ng/mL", scale: "log", domain: [1e-4, 1e2], points, joinDays: 21, ...extra });
  it("breaks a line at a reported 0 and across a gap of more than three weeks", () => {
    const s = strip([pt("2025-01-01", 0.1), pt("2025-01-08", 0.2), pt("2025-01-15", 0, "zero"), pt("2025-01-22", 0.3), pt("2025-03-01", 0.4), pt("2025-03-08", 0.5)]);
    expect(runsOf(s, [D("2024-12-01"), D("2025-04-01")]).map((r) => r.length)).toEqual([2, 1, 2]);
  });
  it("never joins a partial model day", () => {
    const pts = [pt("2026-05-01", 70), pt("2026-05-02", 72), pt("2026-05-03", 60), pt("2026-05-04", 71)];
    const s = strip(pts, { kind: "model", scale: "linear", domain: [0, 100], joinDays: 1.5, partial: new Set([D("2026-05-03")]) });
    expect(runsOf(s, [D("2026-04-01"), D("2026-06-01")]).map((r) => r.length)).toEqual([2, 1]);
  });
  it("reads a strip at a visit only on that day", () => {
    const s = strip([pt("2025-01-01", 0.1)]);
    expect(readingAt(s, D("2025-01-01") + 6 * 3600_000)?.v).toBe(0.1);
    expect(readingAt(s, D("2025-01-04"))).toBeNull();
  });
});

describe("the window", () => {
  const strips = buildStrips(scw, "pDA");
  const now = Date.parse("2026-10-13T06:00:00Z");
  const ext = extentOf(strips, now);
  it("runs from the first sample to today", () => {
    expect(ext[1]).toBe(now);
    expect(ext[0]).toBeLessThan(D("2014-01-09"));
  });
  it("opens on the last three years by default, and on a link's window when given", () => {
    const w = openingWindow({ extent: ext, from: null, to: null, range: null, day: null, focusTimes: [D("2026-09-30")] });
    expect(Math.round((w[1] - w[0]) / DAY)).toBe(3 * 365);
    const l = openingWindow({ extent: ext, from: D("2020-01-01"), to: D("2020-12-31"), range: null, day: null, focusTimes: [] });
    expect(l).toEqual([D("2020-01-01"), D("2020-12-31")]);
  });
  it("turns a single day into 90 days around it, clamps to the record, and reaches back for a pinned day", () => {
    const one = openingWindow({ extent: ext, from: D("2020-06-01"), to: D("2020-06-01"), range: null, day: null, focusTimes: [] });
    expect(Math.round((one[1] - one[0]) / DAY)).toBe(90);
    expect(clampWin([D("2000-01-01"), D("2001-01-01")], ext)[0]).toBe(ext[0]);
    const pinned = openingWindow({ extent: ext, from: null, to: null, range: null, day: D("2016-07-01"), focusTimes: [] });
    expect(pinned[0]).toBeLessThan(D("2016-07-01"));
    expect(pinned[1]).toBeGreaterThan(D("2016-07-01"));
  });
  it("opens on the whole record when the focused quantity was not measured lately", () => {
    expect(openingWindow({ extent: ext, from: null, to: null, range: null, day: null, focusTimes: [D("2018-01-01")] })).toEqual(ext);
  });
  it("knows which preset a window is", () => {
    expect(presetOf(presetWindow("1y", ext), ext)).toBe("1y");
    expect(presetOf([D("2020-01-01"), D("2020-06-01")], ext)).toBeNull();
  });
});

describe("reading the window", () => {
  const s = buildStrips(scw, "pDA").find((x) => x.id === "pDA")!;
  const win: [number, number] = [D("2025-10-01"), D("2026-10-12")];
  it("prints the highest with its date beside the median (T8) and counts reported 0 apart", () => {
    const st = windowStats(s.points, win);
    expect(st.max).not.toBeNull();
    expect(st.median).not.toBeNull();
    expect(st.max!.v!).toBeGreaterThanOrEqual(st.median!);
    expect(st.n + st.zeros + st.rejected).toBe(s.points.filter((p) => p.t >= win[0] && p.t <= win[1]).length);
  });
  it("sorts the log by date or value, a reported 0 below every measured value", () => {
    const byDate = logRows(s.points, win, "date", "desc");
    expect(byDate[0].t).toBeGreaterThanOrEqual(byDate.at(-1)!.t);
    const byValue = logRows(s.points, win, "value", "desc");
    const firstZero = byValue.findIndex((p) => p.cls === "zero");
    expect(byValue.slice(firstZero).every((p) => p.cls !== "value")).toBe(true);
    expect(byValue[0].v).toBe(windowStats(s.points, win).max!.v);
  });
  it("writes values at their honest precision and a reported 0 in words (T7, T15)", () => {
    expect(valueWords(s, { t: 0, time: "", i: 0, v: 0.20209, cls: "value", flagged: false })).toBe("0.20 ng/mL");
    expect(valueWords(s, { t: 0, time: "", i: 0, v: 0, cls: "zero", flagged: false })).toBe("reported 0");
    expect(figure({ kind: "model", v: null }, 74.6)).toBe("75");
    expect(figure({ kind: "measured", v: "temp" }, 17.06)).toBe("17.1");
  });
});
