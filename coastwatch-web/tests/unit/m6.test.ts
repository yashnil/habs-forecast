import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import type { ObsVariableSummary } from "@/generated/observations";
import type { FreshnessPolicy } from "@/generated/schema";
import {
  AXES,
  axisPosition,
  SELECTOR_VARS,
  binOf,
  decadeTicks,
  headline,
  heatmap,
  highestInView,
  modelOnAxis,
  modelSegments,
  periodOf,
  previewPoints,
  samplingStrip,
  segments,
  viewSummary,
} from "@/lib/bloom";
import { DAY, samples, type Sample } from "@/lib/observations";
import manifestJson from "../fixture-data/v1/manifest.json";

/** Bloom Intelligence (M6) helpers: fixed axes, freshness strip, selector headlines, chart,
 *  heatmap and model band. Synthetic series unless a fixture is named. */
const T = (d: string) => Date.parse(`${d}T19:00:00Z`); // noon in California
const S = (d: string, value: number | null, q: Sample["q"] = null, index = 0): Sample => ({ t: T(d), time: `${d}T19:00:00Z`, value, q, index });
const policy = (manifestJson as { sources: { source_id: string; freshness: FreshnessPolicy }[] }).sources.find((s) => s.source_id === "calhabmap")!.freshness;

describe("fixed axes (spec §5.2): never rescaled to the data", () => {
  it("whole decades per quantity, temperature linear 5–30 °C", () => {
    expect(AXES.pDA.domain).toEqual([1e-4, 1e2]);
    expect(AXES.pn_seriata.domain).toEqual([1, 1e7]);
    expect(AXES.pn_delicatissima.domain).toEqual([1, 1e7]);
    expect(AXES.chl_extracted.domain).toEqual([1e-2, 1e3]);
    expect(AXES.temp).toMatchObject({ scale: "linear", domain: [5, 30] });
    expect(decadeTicks(AXES.pDA.domain)).toEqual([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]);
  });
  it("the selector holds the spec's five quantities, pDA first", () => {
    expect(SELECTOR_VARS).toEqual(["pDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"]);
  });
  it("log heatmap bins are decades (display intervals, not risk levels)", () => {
    for (const id of ["pDA", "pn_seriata", "chl_extracted"] as const) {
      const e = AXES[id].edges;
      for (let i = 1; i < e.length; i++) expect(e[i] / e[i - 1]).toBeCloseTo(10, 9);
    }
    expect([0.005, 0.01, 0.5, 10, 99].map((v) => binOf(v, AXES.pDA.edges))).toEqual([0, 1, 2, 4, 4]);
  });
});

describe("sampling strip (freshness card)", () => {
  it("ticks every visit in the last 120 days and counts Pacific days since the last", () => {
    const now = new Date("2026-10-11T06:30:00Z"); // still Oct 10 in California
    const times = ["2026-05-01T19:00:00Z", "2026-07-01T19:00:00Z", "2026-09-30T19:00:00Z", "2026-10-07T19:00:00Z"];
    const s = samplingStrip(times, now);
    expect(s.visits).toHaveLength(3);
    expect(s.daysSinceLast).toBe(3);
    expect(s.t1 - s.t0).toBe(120 * DAY);
  });
  it("a station with no recent visit shows an empty strip but still knows its last visit", () => {
    const s = samplingStrip(["2022-08-31T19:00:00Z"], new Date("2026-10-10T19:00:00Z"));
    expect(s.visits).toEqual([]);
    expect(s.last).toBe(Date.parse("2022-08-31T19:00:00Z"));
    expect(s.daysSinceLast).toBeGreaterThan(1400);
  });
});

describe("selector headlines: an old value never looks current (rule 8)", () => {
  const sm = (last_date: string | null, n = 10): ObsVariableSummary =>
    ({ variable: "pDA", first_date: "2014-01-08", last_date, last_value: 0.2, last_qualifier: null, n_measured: n, n_last_365d: 0, n_reported_zero: 0, n_rejected: 0, max_last_365d: null }) as unknown as ObsVariableSummary;
  const now = new Date("2026-10-10T19:00:00Z");
  it("recent value: headline with freshness", () => {
    const h = headline(sm("2026-10-05"), policy, now);
    expect(h.kind).toBe("latest");
  });
  it("Monterey Wharf case: pDA last measured Aug 2022 is not a headline", () => {
    expect(headline(sm("2022-08-31"), policy, now)).toMatchObject({ kind: "old", lastDate: "2022-08-31", value: 0.2 });
  });
  it("never measured: says so", () => {
    expect(headline(sm(null, 0), policy, now).kind).toBe("never");
    expect(headline(null, policy, now).kind).toBe("never");
  });
});

describe("previews and the primary chart", () => {
  const ss = [
    S("2026-01-05", 0.2, null, 0),
    S("2026-01-12", 0, "reported_zero", 1),
    S("2026-01-19", null, null, 2), // visit, pDA not measured
    S("2026-01-26", 0.5, null, 3),
    S("2026-03-30", 1.5, "flag_high", 4), // 63 days later: line breaks
    S("2026-04-06", -0.1, "rejected_negative", 5),
    S("2026-04-13", 300, null, 6), // beyond the axis
  ];
  const x0 = T("2026-01-01");
  const x1 = T("2026-05-01");
  it("previews keep zeros on the baseline, leave gaps empty, clamp and mark values beyond the axis", () => {
    const p = previewPoints(ss, AXES.pDA, x0, x1);
    expect(p).toHaveLength(5); // not the unmeasured or rejected visits
    expect(p.filter((x) => x.zero)).toHaveLength(1);
    expect(p.at(-1)).toMatchObject({ v: 100, clipped: true });
  });
  it("a reported zero never goes on the log axis and breaks the line; lines break at gaps over 21 days", () => {
    const seg = segments(ss, AXES.pDA, x0, x1);
    // v10 (guardrail A3): the reported 0 on Jan 12 breaks the line, so Jan 5 and Jan 26 are never joined
    // across it (a visit that did not measure pDA, Jan 19, does not break it). Index 6 (300 ng/mL) is above
    // the 100 ng/mL axis top: an edge marker, not part of the line
    expect(seg.map((s) => s.map((x) => x.index))).toEqual([[0], [3], [4]]);
  });
  it("a value below the axis floor is a marker, not a point on the line, and breaks the line (real case: 4.6e-5 ng/mL)", () => {
    const low = [S("2016-11-02", 0.002, null, 0), S("2016-11-09", 0.003, null, 1), S("2016-11-16", 4.6e-5, null, 2), S("2016-11-23", 0.004, null, 3)];
    expect(axisPosition(4.6e-5, AXES.pDA)).toBe("below");
    expect(axisPosition(150, AXES.pDA)).toBe("above");
    expect(segments(low, AXES.pDA, T("2016-11-01"), T("2016-12-01")).map((sg) => sg.map((x) => x.index))).toEqual([[0, 1], [3]]);
  });
  it("highest in view and the summary line", () => {
    expect(highestInView(ss, AXES.pDA, x0, x1)?.index).toBe(6);
    expect(highestInView(ss, AXES.pDA, x0, T("2026-02-01"))?.index).toBe(3);
    expect(viewSummary(ss, x0, x1)).toEqual({ visits: 7, measured: 5, zeros: 1, rejected: 1, flagged: 1 });
  });
  it("the range changes only the x window: the axis is the same for every range", () => {
    const a = previewPoints(ss, AXES.pDA, x0, x1).filter((p) => !p.zero);
    const b = previewPoints(ss, AXES.pDA, T("2026-03-01"), x1).filter((p) => !p.zero);
    expect(b.every((p) => a.some((q) => q.t === p.t && q.v === p.v))).toBe(true);
  });
});

describe("season heatmap", () => {
  it("highest value per week, zero-only and unsampled weeks kept apart, Pacific dates", () => {
    const ss = [S("2025-01-06", 0.02), S("2025-01-07", 0.5), S("2025-01-20", 0, "reported_zero"), S("2025-02-03", null), S("2026-07-01", 12)];
    const h = heatmap(ss, AXES.pDA, "week");
    expect(h.years).toEqual([2025, 2026]);
    expect(h.cols).toBe(53);
    expect(h.cells.get("2025:0")).toMatchObject({ kind: "bin", bin: 2, max: 0.5, n: 2 });
    expect(h.cells.get("2025:2")).toMatchObject({ kind: "zero" });
    expect(h.cells.get("2025:4")).toBeUndefined(); // sampled, pDA not measured: unsampled for pDA
    expect(h.cells.get(`2026:${periodOf(T("2026-07-01"), "week").i}`)).toMatchObject({ bin: 4 });
  });
  it("a late-evening California sample stays on its California date", () => {
    expect(periodOf(Date.parse("2026-01-01T07:30:00Z"), "month")).toEqual({ year: 2025, i: 11 }); // Dec 31, 11:30 pm PT
  });
  it("on the real fixture station every bin is a whole decade of the fixed axis", () => {
    const dir = path.resolve(__dirname, "../fixture-data/v1");
    const file = readdirSync(dir).find((f) => f.startsWith("observations-"))!;
    const obs = JSON.parse(readFileSync(path.join(dir, file), "utf8")) as { stations: Parameters<typeof samples>[0][] };
    const st = obs.stations.find((s) => s.station_id === "HABs-SantaCruzWharf")!;
    const h = heatmap(samples(st, "pDA"), AXES.pDA, "month");
    for (const c of h.cells.values()) if (c.kind === "bin") expect(c.bin).toBe(binOf(c.max, AXES.pDA.edges));
    expect(h.cells.size).toBeGreaterThan(0);
  });
});

describe("model band (§5.4): never interpolated", () => {
  const hist = [
    { date: "2026-09-01", value: 0.4, n: 25 },
    { date: "2026-09-02", value: 0.45, n: 26 },
    { date: "2026-09-03", value: null, n: 0 },
    { date: "2026-09-04", value: 0.5, n: 25 },
    { date: "2026-09-05", value: 0.9, n: 8 }, // partial coverage
    { date: "2026-09-06", value: 0.55, n: 25 },
  ];
  it("drops empty days, marks partial-coverage days and breaks the line at both", () => {
    const m = modelOnAxis(hist, T("2026-08-01"), T("2026-10-01"));
    expect(m.points.map((p) => p.date)).toEqual(["2026-09-01", "2026-09-02", "2026-09-04", "2026-09-05", "2026-09-06"]);
    expect(m.points.find((p) => p.date === "2026-09-05")?.partial).toBe(true);
    expect(modelSegments(m.points).map((s) => s.map((p) => p.date))).toEqual([["2026-09-01", "2026-09-02"], ["2026-09-04"], ["2026-09-06"]]);
    expect(m.historyStart).toBe(Date.parse("2026-09-01T12:00:00Z"));
  });
  it("an empty history (NOAA outage) yields nothing to draw and no start", () => {
    expect(modelOnAxis([], 0, Date.now())).toEqual({ points: [], historyStart: null });
    expect(modelOnAxis(undefined, 0, Date.now()).points).toEqual([]);
  });
});
