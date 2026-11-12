import { readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { ObservationDataset, ObsStation } from "@/generated/observations";
import { slimStations } from "@/lib/data";
import { lastMeasured, sourceRowUrl, stationsMeasuring, variableRecord } from "@/lib/observations";
import {
  availabilityAll,
  availabilityOf,
  chooseFocus,
  copyStart,
  defaultCursor,
  emptyRange,
  exportCounts,
  inspectSample,
  latestWindow,
  LATEST_TAIL_DAYS,
  lastWithValue,
  legendFor,
  measuredInstead,
  measuredVariables,
  modelCollapsed,
  nearestWithRecent,
  stationLatest,
  stepSample,
  visitTicks,
} from "@/nx/research/availability";
import { buildStrips, DAY, extentOf, findStation, inWin, located, northToSouth, parseQuery, pointsOf, queryFor } from "@/nx/research/model";
import { ageWords, modelLine, plotPoints, sayPoint, tickPaths } from "@/nx/research/plotData";

const dir = path.resolve(__dirname, "../../../fixture-data/v1");
const ds = JSON.parse(readFileSync(path.join(dir, readdirSync(dir).find((f) => f.startsWith("observations-"))!), "utf8")) as ObservationDataset;
const stations = northToSouth(located(ds));
const mw = findStation(stations, "HABs-MontereyWharf")!;
const scw = findStation(stations, "HABs-SantaCruzWharf")!;
const D = (s: string) => Date.parse(`${s}T12:00:00Z`);
/** the fixture clock (playwright.config.ts CW_FIXED_NOW) */
const NOW = Date.parse("2026-10-13T06:00:00Z");
const YEAR: [number, number] = [NOW - 365 * DAY, NOW];
const NINETY: [number, number] = [NOW - 90 * DAY, NOW];

/** a station with hand-written samples: [time, pDA, temp]; qualifiers by index for pDA */
function synthetic(rows: [string, number | null, number | null][], qualifiers: Record<string, string> = {}): ObsStation {
  return {
    ...mw,
    station_id: "HABs-Synthetic",
    sample_times: rows.map((r) => r[0]),
    depths_m: rows.map(() => 1),
    series: [
      { variable: "pDA", values: rows.map((r) => r[1]), qualifiers },
      { variable: "temp", values: rows.map((r) => r[2]), qualifiers: {} },
    ],
  } as ObsStation;
}

describe("Monterey Wharf, as CalHABMAP publishes it (checked at SCCOOS ERDDAP 2026-11-12)", () => {
  it("a year of weekly visits holds no pDA, no Pseudo-nitzschia and no chlorophyll: temperature only", () => {
    const pda = availabilityOf(mw, "pDA", YEAR);
    expect(pda.visits).toBeGreaterThanOrEqual(45);
    expect([pda.quantified, pda.zeros, pda.rejected, pda.withValue]).toEqual([0, 0, 0, 0]);
    expect(pda.absent).toBe(pda.visits); // every visit is a visit without pDA: counted as such, never as a sample of it
    const temp = availabilityOf(mw, "temp", YEAR);
    expect(temp.quantified).toBe(temp.visits - temp.absent);
    expect(temp.quantified).toBeGreaterThanOrEqual(45);
    for (const v of ["pn_seriata", "pn_delicatissima", "chl_extracted"] as const) expect(availabilityOf(mw, v, YEAR).withValue).toBe(0);
    expect(availabilityOf(mw, "pDA", NINETY).withValue).toBe(0);
    expect(availabilityOf(mw, "temp", NINETY).quantified).toBeGreaterThanOrEqual(11);
  });

  it("the last measured value and date are per variable, not the station's newest visit", () => {
    const all = availabilityAll(mw, YEAR);
    const last = Object.fromEntries(all.map((a) => [a.v, a.last?.time.slice(0, 10)]));
    expect(last).toMatchObject({ pDA: "2022-08-31", pn_seriata: "2025-08-27", pn_delicatissima: "2025-08-27", chl_extracted: "2024-09-11" });
    expect(last.temp).toBe(mw.sample_times.at(-1)!.slice(0, 10));
    expect(all.find((a) => a.v === "pDA")!.last!.v).toBe(0.05220236); // exactly as published
    expect(lastMeasured(mw, "pDA")).toMatchObject({ date: "2022-08-31", value: 0.05220236, qualifier: null });
    expect(lastMeasured(mw, "dDA")).toBeNull(); // never published here
    // dDA and tDA have never been published at this pier: they are not rows of the rail
    expect(measuredVariables(mw)).toEqual(["pDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"]);
  });

  it("an empty range says why, what the visits did record, and where the record is", () => {
    const e = emptyRange(mw, "pDA", YEAR)!;
    expect(e.kind).toBe("after");
    expect(e.last!.time.slice(0, 10)).toBe("2022-08-31");
    expect(e.instead).toEqual(["temp"]);
    expect(e.visits).toBeGreaterThanOrEqual(45);
    expect(emptyRange(mw, "temp", YEAR)).toBeNull();
    expect(emptyRange(mw, "dDA", YEAR)).toMatchObject({ kind: "never", last: null, next: null });
    // a window before the first pDA sample and one between two samples
    const first = pointsOf(mw, "pDA")[0];
    expect(emptyRange(mw, "pDA", [first.t - 200 * DAY, first.t - 100 * DAY])).toMatchObject({ kind: "before", last: null });
    expect(emptyRange(mw, "pDA", [D("2023-01-01"), D("2024-01-01")])).toMatchObject({ kind: "after" });
  });

  it("jump to latest: a window that holds the newest pDA sample, with its day pinned", () => {
    const strips = buildStrips(mw, "pDA");
    const extent = extentOf(strips, NOW);
    const j = latestWindow(pointsOf(mw, "pDA"), extent)!;
    expect(j.point.time).toBe("2022-08-31T14:45:00Z");
    expect(j.day).toBe(j.point.t);
    expect(inWin(j.point.t, j.win)).toBe(true);
    expect(Math.round((j.win[1] - j.point.t) / DAY)).toBe(LATEST_TAIL_DAYS);
    expect(Math.round((j.win[1] - j.win[0]) / DAY)).toBe(365);
    expect(availabilityOf(mw, "pDA", j.win).withValue).toBeGreaterThan(0);
    expect(latestWindow(pointsOf(mw, "dDA"), extent)).toBeNull(); // nothing to jump to
    // the newest sample at the record's end: the window is clamped to the record, the sample still inside
    const t = latestWindow(pointsOf(mw, "temp"), extent)!;
    expect(t.win[1]).toBeLessThanOrEqual(extent[1]);
    expect(inWin(t.point.t, t.win)).toBe(true);
  });

  it("the nearest station with a recent pDA sample is computed: Santa Cruz Wharf, about 41 km", () => {
    const n = nearestWithRecent(stations, mw, "pDA", NOW)!;
    expect(n.station.station_id).toBe("HABs-SantaCruzWharf");
    expect(Math.round(n.km)).toBe(41);
    expect(n.ageDays).toBeLessThanOrEqual(90);
    expect(n.last.time).toBe(lastMeasured(scw, "pDA")!.time);
    expect(nearestWithRecent(stations, mw, "dDA", NOW)).toBeNull(); // no station publishes dissolved domoic acid
    expect(nearestWithRecent(stations, mw, "pDA", NOW, 0)).toBeNull();
    // never the station itself
    expect(nearestWithRecent(stations, scw, "pDA", NOW)!.station.station_id).not.toBe("HABs-SantaCruzWharf");
  });

  it("the crosshair rests on the selected variable's samples, never on a visit that measured something else", () => {
    expect(defaultCursor(pointsOf(mw, "pDA"), YEAR)).toBeNull(); // no pDA in the window: no crosshair
    expect(defaultCursor(pointsOf(mw, "temp"), YEAR)).toBe(Date.parse(mw.sample_times.at(-1)!));
    const w: [number, number] = [D("2022-01-01"), D("2022-12-31")];
    expect(defaultCursor(pointsOf(mw, "pDA"), w)).toBe(Date.parse("2022-08-31T14:45:00Z")); // not the newest visit of 2022
    expect(defaultCursor(pointsOf(mw, "pDA"), w, D("2022-08-31"))).toBe(Date.parse("2022-08-31T14:45:00Z")); // a pinned day snaps to its sample
    expect(defaultCursor(pointsOf(mw, "pDA"), w, D("2022-03-02"))).toBe(D("2022-03-02")); // a pinned day without one stays as asked
  });

  it("the model panel is collapsed when the window has no measured value of the selected variable", () => {
    expect(modelCollapsed(pointsOf(mw, "pDA"), YEAR)).toBe(true);
    expect(modelCollapsed(pointsOf(mw, "pDA"), [D("2022-01-01"), D("2022-12-31")])).toBe(false);
    expect(modelCollapsed(pointsOf(scw, "pDA"), YEAR)).toBe(false);
    // a rejected sample alone is not a measured value
    const st = synthetic([["2026-01-05T16:00:00Z", null, 12]], { "0": "rejected_negative" });
    expect(modelCollapsed(pointsOf(st, "pDA"), [D("2026-01-01"), D("2026-02-01")])).toBe(true);
  });

  it("the legend names only what is drawn", () => {
    const strips = buildStrips(mw, "pDA");
    const folded = new Set(strips.filter((x) => !x.points.some((p) => inWin(p.t, YEAR))).map((x) => x.id));
    folded.add("model"); // collapsed: no pDA in the window
    const ids = legendFor(strips, YEAR, (x) => !folded.has(x.id)).map((l) => l.id);
    expect(ids).toEqual(["measured", "now"]); // temperature only: no threshold (its strips are folded), no model
    const wide: [number, number] = [D("2018-01-01"), D("2022-12-31")];
    const all = legendFor(strips, wide).map((l) => l.id);
    expect(all).toEqual(expect.arrayContaining(["measured", "zero", "threshold", "now"]));
    expect(legendFor(strips, wide).find((l) => l.id === "threshold")!.words).toContain("Pseudo-nitzschia");
    expect(legendFor(strips, YEAR, () => false)).toEqual([]);
  });

  it("the export line counts rows by variable, never '49 samples' of every quantity", () => {
    const c = exportCounts(buildStrips(mw, "pDA"), YEAR);
    expect(c.by.find((x) => x.v === "pDA")!.n).toBe(0);
    expect(c.by.find((x) => x.v === "temp")!.n).toBe(c.rows);
    expect(c.by.map((x) => x.v)).not.toContain(null); // the model strip is not a sample
  });
});

describe("zero, below detection, rejected and not measured are four different things", () => {
  const st = synthetic(
    [
      ["2026-01-05T16:00:00Z", 0.2, 12.1],
      ["2026-01-12T16:00:00Z", 0, 12.4],
      ["2026-01-19T16:00:00Z", null, 12.0], // rejected (the pipeline nulls a negative and qualifies it)
      ["2026-01-26T16:00:00Z", null, 11.8], // not measured
      ["2026-02-02T16:00:00Z", 900, null], // flagged high; no temperature
    ],
    { "1": "reported_zero", "2": "rejected_negative", "4": "flag_high" },
  );
  const win: [number, number] = [D("2026-01-01"), D("2026-03-01")];

  it("counts each state once and they add up to the visits", () => {
    const a = availabilityOf(st, "pDA", win);
    expect(a).toMatchObject({ visits: 5, quantified: 2, zeros: 1, rejected: 1, absent: 1, flagged: 1, withValue: 3, total: 3 });
    expect(a.quantified + a.zeros + a.rejected + a.absent).toBe(a.visits);
    expect(visitTicks(st, "pDA", win).map((k) => k.state)).toEqual(["quantified", "zero", "rejected", "absent", "quantified"]);
    expect(visitTicks(st, "temp").map((k) => k.state)).toEqual(["quantified", "quantified", "quantified", "quantified", "absent"]);
    expect(measuredInstead(availabilityAll(st, win), "pDA")).toEqual(["temp"]);
  });

  it("a reported 0 is a sample with a value; a rejected one is not", () => {
    const pts = pointsOf(st, "pDA");
    expect(lastWithValue(pts.slice(0, 3))!.cls).toBe("zero");
    expect(lastMeasured(synthetic([["2026-01-05T16:00:00Z", 0, 1], ["2026-01-12T16:00:00Z", null, 1]], { "0": "reported_zero", "1": "rejected_negative" }), "pDA")).toMatchObject({ value: 0, qualifier: "reported_zero", date: "2026-01-05" });
    // a window whose only pDA entry was rejected is empty of values
    expect(emptyRange(st, "pDA", [D("2026-01-15"), D("2026-01-28")])).toMatchObject({ kind: "gap", visits: 2, instead: ["temp"] });
  });

  it("Previous and Next step through the selected variable's samples, not the visits", () => {
    const pts = pointsOf(st, "pDA");
    const times = (p: { time: string } | null) => p?.time.slice(0, 10) ?? null;
    expect(times(stepSample(pts, null, 1))).toBe("2026-01-05");
    expect(times(stepSample(pts, null, -1))).toBe("2026-02-02");
    expect(times(stepSample(pts, pts[2].t, 1))).toBe("2026-02-02"); // skips the visit of Jan 26 that did not measure pDA
    expect(times(stepSample(pts, pts[0].t, -1))).toBeNull();
    expect(times(stepSample(pts, pts.at(-1)!.t, 1))).toBeNull();
    expect(times(stepSample(pts, null, -1, [D("2026-01-01"), D("2026-01-15")]))).toBe("2026-01-12");
  });

  it("inspection: the exact published value, its state in words, and what else that visit recorded", () => {
    const zero = inspectSample(ds, st, "pDA", 1)!;
    expect(zero).toMatchObject({ value: 0, valueText: "0", state: "reported_zero", qualifier: "reported_zero", words: "reported 0" });
    expect(zero.stateWords).toContain("Not quantified, not absent");
    const rej = inspectSample(ds, st, "pDA", 2)!;
    expect(rej).toMatchObject({ value: null, valueText: "", state: "rejected", qualifier: "rejected_negative" });
    const hi = inspectSample(ds, st, "pDA", 4)!;
    expect(hi).toMatchObject({ value: 900, state: "flagged_high", qualifier: "flag_high" });
    expect(hi.others).toEqual([{ v: "temp", label: "Water temperature", unit: "°C", state: "absent", value: null, words: "not measured in this sample" }]);
    expect(inspectSample(ds, st, "pDA", 3)).toBeNull(); // a visit that did not measure it is not a sample of it
  });
});

describe("inspecting a real sample", () => {
  it("Monterey Wharf's last pDA: value, unit, both clocks, depth, place, source row, citation, checks", () => {
    const i = pointsOf(mw, "pDA").at(-1)!.i;
    const r = inspectSample(ds, mw, "pDA", i)!;
    expect(r).toMatchObject({
      time: "2022-08-31T14:45:00Z",
      utc: "2022-08-31 14:45 UTC",
      value: 0.05220236,
      valueText: "0.05220236",
      words: "0.052 ng/mL",
      state: "measured",
      qualifier: null,
      sampleId: null,
      index: i,
    });
    expect(r.local).toMatch(/^Aug 31, 2022, 7:45 AM (PDT|GMT-7)$/);
    expect(r.station).toMatchObject({ id: "HABs-MontereyWharf", name: "Monterey Wharf", locationCode: "HAB_MWII", nearestPort: "Monterey" });
    expect(r.station.lat).toBeCloseTo(36.6037, 3);
    expect(r.variable).toMatchObject({ id: "pDA", label: "Particulate domoic acid", unit: "ng/mL", sourceColumn: "pDA" });
    expect(r.variable.detectionLimitNote).toContain("No detection or quantification limit is published");
    expect(r.depthM).toBe(mw.depths_m[i]);
    expect(r.sourceRowUrl).toBe(
      "https://erddap.sccoos.org/erddap/tabledap/HABs-MontereyWharf.htmlTable?time,latitude,longitude,depth,Location_Code,SampleID,pDA,dDA,tDA,Pseudo_nitzschia_seriata_group,Pseudo_nitzschia_delicatissima_group,Avg_Chloro,Temp,DA_Volume_Filtered,Chl_Volume_Filtered,Volume_Settled_for_Counting&time=2022-08-31T14%3A45%3A00Z",
    );
    expect(r.sourceUrl).toBe("https://erddap.sccoos.org/erddap/tabledap/HABs-MontereyWharf.html");
    expect(r.citation).toContain("CalHABMAP");
    expect(r.citation).toContain("Monterey Wharf (HABs-MontereyWharf), particulate domoic acid, sample of 2022-08-31 14:45 UTC");
    expect(r.qc.length).toBe(mw.qc.length);
    // the co-measured list holds every other variable of the rail, each with its own state
    expect(r.others.map((o) => o.v)).toEqual(["pn_seriata", "pn_delicatissima", "chl_extracted", "temp"]);
    for (const o of r.others) expect(o.state === "absent" ? o.words : o.state).not.toBe("");
  });

  it("a source row link is only ever built for a real station id and sample time", () => {
    expect(sourceRowUrl({ station_id: "HABs-MontereyWharf" }, "2022-08-31")).toBeNull();
    expect(sourceRowUrl({ station_id: "evil/../x" }, "2022-08-31T14:45:00Z")).toBeNull();
  });
});

describe("the address and the stations", () => {
  it("honours `var` whenever the station has ever measured it, whatever the window holds", () => {
    const q = parseQuery(new URLSearchParams({ station: "HABs-MontereyWharf", var: "chl_extracted", from: "2026-08-01", to: "2026-10-01" }));
    expect(chooseFocus(mw, q.v)).toMatchObject({ focus: "chl_extracted", askedMissing: null }); // none in that window: still the focus
    expect(chooseFocus(mw, parseQuery(new URLSearchParams({ var: "temp" })).v).focus).toBe("temp");
    expect(chooseFocus(mw, parseQuery(new URLSearchParams({ var: "sst" })).v).focus).toBe("temp"); // an Explore layer name
    expect(chooseFocus(mw, null)).toMatchObject({ focus: "pDA", askedMissing: null });
    expect(chooseFocus(mw, "dDA")).toMatchObject({ focus: "pDA", askedMissing: "dDA" }); // said, not silently swapped
  });

  it("stationsMeasuring: by variable, and by how recent that variable's own newest sample is", () => {
    const ids = (xs: { station_id: string }[]) => xs.map((s) => s.station_id);
    const ever = ids(stationsMeasuring(ds.stations, "pDA"));
    expect(ever).toContain("HABs-MontereyWharf");
    expect(ever).toContain("HABs-SantaCruzWharf");
    expect(stationsMeasuring(ds.stations, "dDA")).toEqual([]);
    const recent = ids(stationsMeasuring(ds.stations, "pDA", { withinDays: 90, now: NOW }));
    expect(recent).toContain("HABs-SantaCruzWharf");
    expect(recent).not.toContain("HABs-MontereyWharf"); // visited last week; pDA last measured in 2022
    expect(ids(stationsMeasuring(ds.stations, "temp", { withinDays: 90, now: NOW }))).toContain("HABs-MontereyWharf");
    // the slim stations Explore carries give the same answer for the variables they hold…
    const slim = slimStations(ds);
    expect(ids(stationsMeasuring(slim, "pDA")).sort()).toEqual([...ever].sort());
    expect(ids(stationsMeasuring(slim, "pDA", { withinDays: 90, now: NOW }))).not.toContain("HABs-MontereyWharf");
    // …and say nothing about one they do not (unknown is never "measures")
    expect(variableRecord(slim[0], "temp").known).toBe(false);
    expect(stationsMeasuring(slim, "temp")).toEqual([]);
  });

  it("the station list dates each station by the selected variable, with the visit date apart", () => {
    const rows = stationLatest(stations, "pDA");
    const m = rows.find((r) => r.station.station_id === "HABs-MontereyWharf")!;
    expect(m.last!.date).toBe("2022-08-31");
    expect(m.lastVisit!.slice(0, 4)).toBe("2026");
    expect(stationLatest(stations, "dDA").every((r) => r.last == null)).toBe(true);
  });

  it("says where Neritic's copy starts from the data, with either request window", () => {
    const now = copyStart(ds, mw);
    expect(now.windowStart).toBe(ds.window_start);
    expect(now.firstSample).toBe(mw.sample_times[0].slice(0, 10));
    expect(now.sourceUrl).toBe(mw.source_url);
    // the 2005 request: Monterey's first sample (Jun 2005) is the source's own first, months after the request start
    expect(copyStart({ window_start: "2005-01-01" }, { sample_times: ["2005-06-03T17:10:00Z"], source_url: "u" }).mayBeCut).toBe(false);
    // the 2014 request (what staging holds): the first sample sits on the request start, so earlier ones may exist
    expect(copyStart({ window_start: "2014-01-01" }, { sample_times: ["2014-01-08T16:45:00Z"], source_url: "u" })).toMatchObject({ mayBeCut: true, firstSample: "2014-01-08" });
    expect(copyStart({ window_start: "2014-01-01" }, { sample_times: [], source_url: "u" })).toMatchObject({ mayBeCut: false, firstSample: null });
  });
});

describe("the plot kit's points (plotData)", () => {
  it("a reported 0 and a rejected sample have no place on the y axis; a visit without the variable is not a point", () => {
    const st = { ...mw, sample_times: ["2026-01-05T16:00:00Z", "2026-01-12T16:00:00Z", "2026-01-19T16:00:00Z", "2026-01-26T16:00:00Z"], depths_m: [1, 1, 1, 1], series: [{ variable: "pDA", values: [0.2, 0, null, null], qualifiers: { "1": "reported_zero", "2": "rejected_negative" } }] } as unknown as ObsStation;
    const strip = buildStrips(st, "pDA").find((x) => x.v === "pDA")!;
    const pts = plotPoints(strip, [-121.9, 36.6]);
    expect(pts.map((p) => [p.kind, p.v, p.raw])).toEqual([
      ["measured", 0.2, undefined],
      ["reported-zero", null, "reported 0"],
      ["missing", null, "rejected by QC"],
    ]);
    expect(pts[0]).toMatchObject({ id: "pDA-0", place: [-121.9, 36.6] });
    expect(pts[0].meta.i).toBe(0);
    expect(sayPoint(strip, strip.points[0])).toBe("Jan 5, 2026 · 0.20 ng/mL · measured");
    expect(sayPoint(strip, strip.points[1])).toBe("Jan 12, 2026 · reported 0 · at or below an unpublished limit, not absent");
    expect(sayPoint(strip, strip.points[2])).toBe("Jan 19, 2026 · rejected by QC · no value");
  });

  it("the model is its own kind, and a day with few cells breaks its line", () => {
    const model = buildStrips(scw, "pDA").find((x) => x.kind === "model")!;
    const pts = plotPoints(model);
    expect(pts.every((p) => p.kind === "forecast" && p.v != null && p.v >= 0 && p.v <= 100)).toBe(true);
    expect(sayPoint(model, model.points[0])).toMatch(/ · \d+ % · model, not measured/);
    const partial = new Set([pts[1].t]);
    const line = modelLine(pts, partial);
    expect(line[1]).toMatchObject({ v: null, kind: "missing" });
    expect(line[0]).toBe(pts[0]);
  });

  it("the rail's ticks: one stroke per visit by state, inside the 0–1000 box", () => {
    const ticks = visitTicks(mw, "pDA");
    const paths = tickPaths(ticks, extentOf(buildStrips(mw, "pDA"), NOW));
    const n = (d: string) => (d.match(/M/g) ?? []).length;
    expect(n(paths.quantified) + n(paths.zero) + n(paths.rejected) + n(paths.absent)).toBe(ticks.length);
    expect(n(paths.quantified)).toBe(ticks.filter((k) => k.state === "quantified").length);
    expect(n(paths.absent)).toBeGreaterThan(40); // the recent temperature-only visits are drawn, as visits without pDA
    for (const m of paths.quantified.matchAll(/M([\d.]+) /g)) expect(Number(m[1])).toBeLessThanOrEqual(1000);
  });

  it("ages a sample in the unit that fits", () => {
    expect([0, 1, 12, 150, 800, 1500].map(ageWords)).toEqual(["today", "yesterday", "12 days ago", "5 months ago", "2.2 years ago", "4.1 years ago"]);
  });

  it("the address carries a comparison variable, never the selected one twice", () => {
    expect(parseQuery(new URLSearchParams({ var: "pDA", cmp: "temp" })).cmp).toBe("temp");
    const w: [number, number] = [D("2026-01-01"), D("2026-06-01")];
    expect(queryFor({ station: "HABs-MontereyWharf", v: "pDA", win: w, cmp: "temp" })).toContain("cmp=temp");
    expect(queryFor({ station: "HABs-MontereyWharf", v: "pDA", win: w, cmp: "pDA" })).not.toContain("cmp=");
  });
});
