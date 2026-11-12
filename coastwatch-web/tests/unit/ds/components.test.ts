import { readFileSync } from "node:fs";
import path from "node:path";
import { createElement as h } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

// CSS modules are not processed under vitest (the app's PostCSS config is for Next); class names echo their key
const css = vi.hoisted(() => ({ default: new Proxy({}, { get: (_t, k) => String(k) }) }));
vi.mock("@/nx/ds/Agenda.module.css", () => css);
vi.mock("@/nx/ds/BigDate.module.css", () => css);
vi.mock("@/nx/ds/Button.module.css", () => css);
vi.mock("@/nx/ds/Margin.module.css", () => css);
vi.mock("@/nx/ds/MarginSheet.module.css", () => css);
vi.mock("@/nx/ds/Mark.module.css", () => css);
vi.mock("@/nx/ds/Overlay.module.css", () => css);
vi.mock("@/nx/ds/PortFinder.module.css", () => css);
vi.mock("@/nx/ds/Root.module.css", () => css);
vi.mock("@/nx/ds/Stamp.module.css", () => css);
vi.mock("@/nx/ds/Statement.module.css", () => css);
vi.mock("@/nx/ds/States.module.css", () => css);
vi.mock("@/nx/ds/chart/chart.module.css", () => css);
vi.mock("@/nx/ds/chart/legend.module.css", () => css);
vi.mock("@/nx/ds/chart/readout.module.css", () => css);
vi.mock("@/nx/ds/series/series.module.css", () => css);
vi.mock("@/nx/ds/worksheet/worksheet.module.css", () => css);
import {
  Agenda,
  BigDate,
  Button,
  Waterfall,
  Legend,
  Mark,
  RampLegend,
  SeriesChart,
  Statement,
  agendaPhase,
  censoredWords,
  clampWindow,
  labelledTicks,
  classify,
  day,
  daysBetween,
  fixed,
  layoutWaterfall,
  logDomain,
  logTickLabel,
  logTicks,
  markFor,
  range,
  rampPosition,
  readoutText,
  resolveChartMode,
  segments,
  sig,
  stampText,
  usd,
  type SeriesPoint,
} from "@/nx/ds";

const NOW = "2026-11-10T15:50:00Z";

describe("format: absolute dates in Pacific time, honest figures", () => {
  it("prints ranges the way the spec writes them", () => {
    expect(range("2026-12-15", "2026-12-18")).toBe("Dec 15–18");
    expect(range("2026-11-28", "2026-12-02")).toBe("Nov 28–Dec 2");
    expect(range("2027-01-13", "2027-01-15", { ref: NOW })).toBe("Jan 13–15, 2027");
    expect(range("2026-12-30", "2027-01-02")).toBe("Dec 30, 2026–Jan 2, 2027");
    expect(day("2027-01-01", { ref: NOW })).toBe("Jan 1, 2027");
  });
  it("a plain date is a Pacific calendar day, and an evening UTC time can be the day before in PT", () => {
    expect(day("2026-11-10")).toBe("Nov 10");
    expect(day("2026-11-10T07:21:05Z")).toBe("Nov 9");
    expect(daysBetween("2026-09-28", NOW)).toBe(43);
  });
  it("precision follows error (T15) and money and minus signs are typeset", () => {
    expect(sig(0.6912, 2)).toBe("0.69");
    expect(sig(9.8, 2)).toBe("9.8");
    expect(sig(19.21, 3)).toBe("19.2");
    expect(sig(1058, 2)).toBe("1,058");
    expect(usd(-190)).toBe("−$190");
    expect(fixed(-0.39)).toBe("−0.4");
  });
});

describe("kinds, modes and stamps", () => {
  it("a RAMP assessment date reads as assessed", () => {
    expect(stampText({ kind: "official", title: "Whale measures", source: "CDFW RAMP", date: "2026-11-06", dateRole: "assessed" }, NOW)).toBe("Official · CDFW RAMP · assessed Nov 6 · not verified");
  });
  it("buttons pass data- and aria- attributes through", () => {
    const html = renderToStaticMarkup(h(Button, { variant: "primary", "data-testid": "go-find", "aria-describedby": "x", children: "Find your port" }));
    expect(html).toContain('data-testid="go-find"');
    expect(html).toContain('aria-describedby="x"');
  });
  it("only measured readings are solid; official keeps its own ring", () => {
    expect(markFor("measured")).toBe("measured");
    expect(markFor("analysis")).toBe("model");
    expect(markFor("outlook")).toBe("model");
    expect(markFor("official")).toBe("official");
  });
  it("a stored mode wins, the system picks Day or Dusk, Night is never automatic", () => {
    expect(resolveChartMode("night", false)).toBe("night");
    expect(resolveChartMode(null, true)).toBe("dusk");
    expect(resolveChartMode("bogus", false)).toBe("day");
  });
  it("an official stamp always says not verified (T38)", () => {
    const t = stampText({ kind: "official", title: "Season", source: "CDFW", date: "2026-11-08", dateRole: "checked" });
    expect(t).toBe("Official · CDFW · checked Nov 8 · not verified");
    expect(stampText({ kind: "measured", title: "Waves", source: "NDBC buoy 46236", date: "2026-11-10T15:00Z" })).toBe("Measured · NDBC buoy 46236 · Nov 10, 7:00 AM PT");
  });
});

describe("series: censored marks and gaps (T7, T19, T20)", () => {
  const pts: SeriesPoint[] = [
    { t: "2026-01-01", v: 0.1 },
    { t: "2026-01-08", v: 0.2 },
    { t: "2026-01-15", v: 0, flag: "reported0", raw: "0" },
    { t: "2026-01-22", v: 0.3 },
    { t: "2026-01-29", v: null },
    { t: "2026-05-01", v: 0.4 },
    { t: "2026-05-08", v: null, flag: "censored", raw: "<2.5" },
  ];
  it("classifies: a flag is censored, null is missing, zero with no flag is a value", () => {
    expect(pts.map(classify)).toEqual(["value", "value", "censored", "value", "missing", "value", "censored"]);
    expect(classify({ t: "2026-01-01", v: 0 })).toBe("value");
  });
  it("breaks the line at censored samples and at long gaps; missing samples are skipped", () => {
    expect(segments(pts, 45).map((r) => r.map((p) => p.t))).toEqual([["2026-01-01", "2026-01-08"], ["2026-01-22"], ["2026-05-01"]]);
  });
  it("prints censored values in words, never as 0 or the limit", () => {
    expect(censoredWords(pts[2])).toBe("reported 0");
    expect(censoredWords(pts[6])).toBe("<2.5, below the limit");
    expect(censoredWords({ t: "x", v: null, flag: "nd" })).toBe("ND, not detected");
    expect(readoutText(pts[6], "ppm", "measured")).toBe("May 8, 2026 · <2.5, below the limit · measured");
  });
  it("log axes in whole decades with plain labels", () => {
    expect(logDomain([0.0013, 11.2])).toEqual([0.001, 100]);
    expect(logTicks([0.001, 100]).map(logTickLabel)).toEqual(["0.001", "0.01", "0.1", "1", "10", "100"]);
    expect(logTickLabel(1e6)).toBe("1M");
  });
  it("the chart draws censored samples in their own strip, with its words", () => {
    const html = renderToStaticMarkup(h(SeriesChart, { points: pts, label: "pDA", unit: "ng/mL" }));
    expect(html).toContain("Below limit");
    expect((html.match(/ptCensored/g) ?? []).length).toBe(2);
    expect(html).toContain("6 samples, Jan 1, 2026 to May 8, 2026; 2 below the limit or reported 0");
  });
});

describe("strips: low-cell model days and thinned tick labels", () => {
  it("a model day from few cells is never joined into the line and says so", () => {
    const pts: SeriesPoint[] = [
      { t: "2026-10-01", v: 0.4 },
      { t: "2026-10-02", v: 0.5, low: true },
      { t: "2026-10-03", v: 0.6 },
      { t: "2026-10-04", v: 0.7 },
    ];
    expect(segments(pts).map((r) => r.map((p) => p.t))).toEqual([["2026-10-01"], ["2026-10-03", "2026-10-04"]]);
    expect(readoutText(pts[1], "%", "model")).toContain("from few model cells");
  });
  it("labels only as many ticks as fit, always the top one", () => {
    expect(labelledTicks([1, 10, 100, 1000, 10000], 200)).toEqual([true, true, true, true, true]);
    expect(labelledTicks([1, 10, 100, 1000, 10000], 40)).toEqual([true, false, true, false, true]);
  });
});

describe("waterfall layout (E3) and brush", () => {
  it("costs step down from where the last bar ended; net sits on zero", () => {
    const l = layoutWaterfall([
      { id: "g", label: "Landed", kind: "gross", amount: 3800 },
      { id: "f", label: "Fees", kind: "cost", amount: 190 },
      { id: "n", label: "Net", kind: "net", amount: 3610 },
    ]);
    expect(l.bars.map((b) => [b.from, b.to])).toEqual([[0, 3800], [3610, 3800], [0, 3610]]);
    expect(l.domain).toEqual([0, 3800]);
  });
  it("when costs exceed gross the domain's low end is the running minimum", () => {
    const l = layoutWaterfall([
      { id: "g", label: "Landed", kind: "gross", amount: 420 },
      { id: "f", label: "Fuel", kind: "cost", amount: 552 },
      { id: "n", label: "Net", kind: "net", amount: -132 },
    ]);
    expect(l.domain[0]).toBe(-132);
    expect(l.pos(-132)).toBe(0);
  });
  it("a missing figure waits: no bar after it", () => {
    const l = layoutWaterfall([
      { id: "g", label: "Landed", kind: "gross", amount: null },
      { id: "f", label: "Fuel", kind: "cost", amount: 552 },
    ]);
    expect(l.bars.every((b) => b.waiting)).toBe(true);
  });
  it("a waterfall on a scale held from outside keeps that scale, and a blank figure waits in words", () => {
    const steps = [
      { id: "gross", label: "Landed", kind: "gross" as const, amount: 1000 },
      { id: "fuel", label: "Fuel", kind: "cost" as const, amount: 250 },
      { id: "net", label: "Net", kind: "net" as const, amount: null, waiting: "waiting for crew share" },
    ];
    const html = renderToStaticMarkup(h(Waterfall, { steps, label: "Trip", domain: [0, 2000], testIdPrefix: "ledger" }));
    expect(html).toContain('data-testid="ledger-gross"');
    expect(html).toContain("width:50%");
    expect(html).toContain('colSpan="2"');
    expect(html).toContain("waiting for crew share");
  });
  it("the brush window stays inside the record and never narrower than its minimum", () => {
    expect(clampWindow([-50, 50], [0, 1000], 10)).toEqual([0, 100]);
    expect(clampWindow([990, 1100], [0, 1000], 10)).toEqual([890, 1000]);
    expect(clampWindow([500, 502], [0, 1000], 10)).toEqual([500, 510]);
  });
});

describe("agenda, legend and type render accessibly", () => {
  it("phases by Pacific day", () => {
    expect(agendaPhase({ start: "2026-10-30" }, NOW)).toBe("past");
    expect(agendaPhase({ start: "2026-11-10" }, NOW)).toBe("now");
    expect(agendaPhase({ start: "2026-12-15", end: "2026-12-18" }, NOW)).toBe("future");
  });
  it("places the now line between past and coming items and words each phase", () => {
    const html = renderToStaticMarkup(
      h(Agenda, {
        label: "Dates",
        now: NOW,
        items: [
          { id: "a", start: "2026-12-15", end: "2026-12-18", title: "Assessment", tentative: true },
          { id: "b", start: "2026-10-30", title: "Delayed" },
        ],
      }),
    );
    expect(html.indexOf("Delayed")).toBeLessThan(html.indexOf("Today, Nov 10"));
    expect(html.indexOf("Today, Nov 10")).toBeLessThan(html.indexOf("Assessment"));
    expect(html).toContain("Past. ");
    expect(html).toContain("Coming. ");
    expect(html).toContain("Subject to change.");
  });
  it("an item under way today is an item, and the now line above it does not repeat its date", () => {
    const html = renderToStaticMarkup(
      h(Agenda, {
        label: "Dates",
        now: NOW,
        items: [
          { id: "b", start: "2026-10-30", title: "Delayed" },
          { id: "w", start: "2026-11-10", title: "Wind forecast" },
        ],
      }),
    );
    const item = html.slice(html.indexOf('data-phase="now"') - 40, html.indexOf("Wind forecast"));
    expect(item).toContain("underway");
    expect(item).not.toMatch(/class="now"/);
    // the now line keeps its spoken date but prints only "Today"
    const line = html.slice(html.indexOf("Today, Nov 10"), html.indexOf('data-phase="now"'));
    expect(line).not.toContain("nowWhat");
    expect(html.match(/>Nov 10</g)?.length).toBe(1);
  });
  it("the rail mark is the item's kind, as the legend draws it, never its time (T2)", () => {
    const html = renderToStaticMarkup(
      h(Agenda, {
        label: "Dates",
        now: NOW,
        items: [
          { id: "t", start: "2026-10-30", title: "Crab tested", kind: "measured" },
          { id: "w", start: "2026-11-10", title: "NWS tonight", kind: "model" },
          { id: "a", start: "2026-12-15", title: "Assessment", kind: "official" },
        ],
      }),
    );
    const markOf = (title: string) => {
      const at = html.indexOf(title);
      const before = html.slice(0, at);
      return /data-kind="(\w+)"/.exec(before.slice(before.lastIndexOf("<li")))?.[1];
    };
    // an NWS forecast for today is under way, yet it is drawn as a prediction, not as a solid measurement
    expect(markOf("NWS tonight")).toBe("model");
    expect(markOf("Crab tested")).toBe("measured");
    expect(markOf("Assessment")).toBe("official");
  });
  it("official marks carry their words; the no-view mark is named", () => {
    expect(renderToStaticMarkup(h(Mark, { kind: "official" }))).toContain("not verified");
    expect(renderToStaticMarkup(h(Mark, { kind: "noview" }))).toContain('aria-label="no view"');
    expect(renderToStaticMarkup(h(Legend, { kinds: ["measured", "noview"] }))).toContain("No view: no data, not a low value");
  });
  it("each mode's tokens say which end of a ramp is more, and the legend words follow them (README §4)", () => {
    const css = readFileSync(path.join(__dirname, "../../../src/nx/ds/tokens.css"), "utf8");
    const block = (mode: string) => css.slice(css.indexOf(`[data-chart-mode="${mode}"]`), css.indexOf("}", css.indexOf(`[data-chart-mode="${mode}"]`)));
    expect(block("day")).toMatch(/--ramp-says-darker: inline; --ramp-says-lighter: none;/);
    for (const m of ["dusk", "night"]) expect(block(m)).toMatch(/--ramp-says-darker: none; --ramp-says-lighter: inline;/);
    const html = renderToStaticMarkup(h(RampLegend, { title: "Chlorophyll", unit: "mg/m³", steps: ["--chl-0", "--chl-1"], domain: [0.05, 50], ticks: [0.1, 1, 10] }));
    expect(html).toContain("Darker is more.");
    expect(html).toContain("Lighter is more (dark chart).");
  });
  it("the ramp legend places log ticks and says no view", () => {
    expect(rampPosition(1, [0.05, 50])).toBeCloseTo(0.4337, 3);
    const html = renderToStaticMarkup(h(RampLegend, { title: "Chlorophyll", unit: "mg/m³", steps: ["--chl-0", "--chl-1"], domain: [0.05, 50], ticks: [0.1, 1, 10] }));
    expect(html).toContain("No view");
    expect(html).toContain("logarithmic");
  });
  it("statement and big date render their parts, with a machine-readable date", () => {
    const st = renderToStaticMarkup(h(Statement, { title: "Season delayed", tone: "regulation", rule: true, scope: "Moss Landing" }));
    expect(st).toContain("<h1");
    expect(st).toContain('data-tone="regulation"');
    const bd = renderToStaticMarkup(h(BigDate, { label: "Next assessment", start: "2026-12-15", end: "2026-12-18", then: [{ start: "2027-01-13", end: "2027-01-15" }], now: NOW }));
    expect(bd).toContain("Dec 15–18");
    expect(bd).toContain("then Jan 13–15, 2027");
    expect(bd).toContain('dateTime="2026-12-15T12:00:00.000Z"');
  });
});
