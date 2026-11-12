import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

// CSS modules are not processed under vitest; class names echo their key
const css = vi.hoisted(() => ({ default: new Proxy({}, { get: (_t, k) => String(k) }) }));
vi.mock("@/nx/ds/Button.module.css", () => css);
vi.mock("@/nx/ds/controls.module.css", () => css);
vi.mock("@/nx/ds/layout/Desk.module.css", () => css);
vi.mock("@/nx/ds/plot/plot.module.css", () => css);
vi.mock("@/nx/ds/worksheet/worksheet.module.css", () => css);
vi.mock("@/nx/ds/series/series.module.css", () => css);
vi.mock("@/nx/ds/Root.module.css", () => css);
vi.mock("@/nx/ds/gates.module.css", () => css);
vi.mock("@/nx/ds/Overlay.module.css", () => css);
vi.mock("@/nx/ds/Stamp.module.css", () => css);
vi.mock("@/nx/ds/Mark.module.css", () => css);
vi.mock("@/nx/ds/chart/chart.module.css", () => css);
vi.mock("@/nx/ds/chart/readout.module.css", () => css);
import { Advisories, GateList } from "@/nx/ds/GateList";
import { EditableText, Facts, Inspector, OptionList } from "@/nx/ds/Panels";
import { Readout } from "@/nx/ds/chart/Readout";
import { Sounding } from "@/nx/ds/chart/Sounding";
import { Brush } from "@/nx/ds/series/Brush";
import { markFor } from "@/nx/ds/kinds";
import { lineObstacle } from "@/nx/chart/cover";
import { tallBoundsAround } from "@/nx/chart/scene";
import { cameraFor } from "@/nx/chart/ChartHost";
import { Button } from "@/nx/ds/Button";
import { ButtonGroup } from "@/nx/ds/ButtonGroup";
import { IconButton } from "@/nx/ds/IconButton";
import { Disclosure, Hint, RangeControl, ScrubNumber, ToggleRow } from "@/nx/ds/Controls";
import { Availability, DataState, DataStateLegend, DATA_STATE_KINDS, DATA_STATE_WORDS, Skeleton } from "@/nx/ds/DataState";
import { Figure } from "@/nx/ds/Figure";
import { Desk, DeskAside, DeskBody, DeskFoot, DeskHeader, DeskMain, DeskSticky, PageLayout } from "@/nx/ds/layout/Desk";
import { getLayout, PANE_DETENTS, setPageLayout, setPaneDetent } from "@/nx/ds/layout/pane";
import { AxisX, AxisY, Crosshair, GridY, Line, NoData, NowRule, Plot, PlotSync as PlotSyncAny, Points, Selection, Threshold, Tip } from "@/nx/ds/plot/Plot";
import { PlotState, PlotTable } from "@/nx/ds/plot/Extras";
import { BreakEvenMarker, WaterfallChart } from "@/nx/ds/plot/Waterfall";
import { SensitivityGrid } from "@/nx/ds/plot/SensitivityGrid";
import { Segmented } from "@/nx/ds/worksheet/Inputs";
import * as motion from "@/nx/ds/motion";

// children are passed as arguments here, which createElement's typing does not count toward a required `children`
const h = createElement as unknown as (type: unknown, props?: Record<string, unknown> | null, ...kids: unknown[]) => ReactElement;
const html = (el: ReactElement) => renderToStaticMarkup(el);
const DAY = 86_400_000;
const t0 = Date.UTC(2025, 0, 1, 20);

describe("v13 layout", () => {
  it("a page's mode is a marker in its own HTML, so the stylesheet lays it out before any script", () => {
    expect(html(h(PageLayout, { mode: "desk", paneKey: "research" }))).toBe('<span hidden="" data-nx-layout="desk"></span>');
  });
  it("the desk renders its header (context, one h1, status, actions), body, aside and foot", () => {
    const out = html(
      h(Desk, null, h(DeskHeader, { context: "Moss Landing · Dungeness crab · Hypothetical", title: "A trip", titleProps: { "data-testid": "t" }, status: "Status", pinned: "Dungeness crab", actions: h("button", null, "Save") }), h(DeskBody, { layout: "main-aside", aside: "start" }, h(DeskAside, { label: "Trip inputs" }, "in"), h(DeskMain, null, "main")), h(DeskFoot, null, "foot"), h(DeskSticky, null, "net")),
    );
    expect(out).toContain("data-nx-desk");
    expect(out.match(/<h1/g)).toHaveLength(1);
    expect(out).toContain('data-testid="t"');
    expect(out).toContain("Hypothetical");
    expect(out).toContain("Dungeness crab");
    expect(out).toContain('aria-label="Trip inputs"');
    expect(out).toContain('data-layout="main-aside"');
    expect(out).toContain('data-aside="start"');
    expect(out).toContain("data-nx-desk-sticky");
  });
  it("the pane has four sizes and a page's default until the reader chooses", () => {
    expect(PANE_DETENTS).toEqual(["rail", "context", "split", "chart"]);
    setPageLayout("desk", null, "split");
    expect(getLayout()).toMatchObject({ mode: "desk", detent: "split", open: false });
    setPaneDetent("rail");
    expect(getLayout().detent).toBe("rail");
    setPageLayout("chart", null, "context");
    expect(getLayout().mode).toBe("chart");
  });
});

describe("v13 controls", () => {
  it("buttons: three kinds (text is quiet's old name), two sizes, an end icon", () => {
    expect(html(h(Button, { variant: "primary", size: "l" }, "Save"))).toMatch(/class="btn primary large"/);
    expect(html(h(Button, { variant: "text" }, "More"))).toMatch(/class="btn quiet"/);
    expect(html(h(Button, { variant: "quiet", iconEnd: "chevron" }, "More"))).toContain("<svg");
    expect(html(h(ButtonGroup, { align: "end" }, h(Button, null, "A")))).toContain('role="group"');
  });
  it("an icon button always has a name", () => {
    const out = html(h(IconButton, { icon: "close", label: "Close the chart" }));
    expect(out).toContain('aria-label="Close the chart"');
    expect(out).toContain('title="Close the chart"');
  });
  it("a segmented control with nothing chosen says so and shows no segment as chosen", () => {
    const opts = [
      { value: "shared", label: "Shared, off the top" },
      { value: "boat", label: "The boat" },
    ];
    const unset = html(h(Segmented, { label: "Who pays fuel", options: opts, value: null, onChange: () => {} }));
    expect(unset).toContain("Choose one");
    expect(unset).toContain('data-unset="true"');
    expect(unset).not.toContain('aria-checked="true"');
    const set = html(h(Segmented, { label: "Who pays fuel", options: opts, value: "boat", onChange: () => {} }));
    expect(set).not.toContain("Choose one");
    expect(set.match(/aria-checked="true"/g)).toHaveLength(1);
    expect(set).toContain("segOn");
  });
  it("a range control names a custom window and offers the latest sample", () => {
    const presets = [
      { value: "90d", label: "90 d" },
      { value: "all", label: "All" },
    ];
    const out = html(h(RangeControl, { label: "Window", presets, value: null, onChange: () => {}, onJumpLatest: () => {}, latestLabel: "Latest pDA sample: Mar 3, 2024" }));
    expect(out).toContain("Custom");
    expect(out).toContain("Latest pDA sample: Mar 3, 2024");
    expect(out).not.toContain('aria-checked="true"');
    expect(html(h(RangeControl, { label: "Window", presets, value: "all", onChange: () => {} }))).not.toContain("Custom");
  });
  it("a scrub number is a labelled native input that stays empty when empty", () => {
    const out = html(h(ScrubNumber, { label: "Catch", value: "", onChange: () => {}, unit: "lb", hint: "Your own number" }));
    expect(out).toMatch(/<label for="[^"]+"[^>]*>Catch<\/label>/);
    expect(out).toContain('value=""');
    expect(out).toContain('inputMode="numeric"');
    expect(html(h(ScrubNumber, { label: "Price", value: "9.00", onChange: () => {}, step: 0.25 }))).toContain('inputMode="decimal"');
  });
  it("a disclosure says whether it is open and keeps its value on the row", () => {
    const shut = html(h(Disclosure, { summary: "Station", value: "600 m from Monterey" }, "detail"));
    expect(shut).toContain('aria-expanded="false"');
    expect(shut).toContain("600 m from Monterey");
    expect(shut).not.toContain("detail");
    expect(html(h(Disclosure, { summary: "Station", defaultOpen: true }, "detail"))).toContain("detail");
  });
  it("a toggle row is a switch that says its state in words; a hint is a note", () => {
    const out = html(h(ToggleRow, { label: "Season changes", hint: "Checked when you open Neritic", checked: false, onChange: () => {} }));
    expect(out).toContain('role="switch"');
    expect(out).toContain('aria-checked="false"');
    expect(out).toContain(">Off<");
    expect(html(h(Hint, { tone: "caution" }, "No text messages are sent."))).toContain('role="note"');
  });
  it("a figure shows its exact value to assistive tech, to the cent on request, and never 0 for an unknown", () => {
    const out = html(h(Figure, { label: "Net to boat", value: 2602.24, format: "money", cents: true, size: "xl" }));
    expect(out).toContain("$2,602.24");
    expect(out).toContain('data-size="xl"');
    expect(html(h(Figure, { label: "Net to boat", value: 2602.24, format: "money" }))).toContain("$2,602");
    const none = html(h(Figure, { label: "Net to boat", value: null, waiting: "waiting for price" }));
    expect(none).toContain("waiting for price");
    expect(none).not.toMatch(/>0</);
    expect(html(h(Figure, { label: "Break-even catch", value: 39, unit: "lb", kind: "estimated" }))).toContain("estimated");
  });
});

describe("v13 states", () => {
  it("every state has a word, a legend sentence and a mark", () => {
    expect(DATA_STATE_KINDS).toHaveLength(15);
    expect(DATA_STATE_KINDS).toContain("rejected");
    for (const k of DATA_STATE_KINDS) {
      expect(DATA_STATE_WORDS[k].word.length).toBeGreaterThan(2);
      expect(html(h(DataState, { kind: k }))).toContain(`data-state-mark="${k}"`);
    }
    expect(html(h(DataStateLegend, { kinds: ["measured", "below-detection"] }))).toContain("not zero");
  });
  it("a block state gives the reason, the date and its age, and loading is announced", () => {
    const out = html(h(DataState, { kind: "not-collected", variant: "block", what: "pDA", reason: "0 of 12 samples in this window measured pDA", date: "2024-03-03", now: "2026-10-13T06:00:00Z", source: "CalHABMAP" }));
    expect(out).toContain("Not collected");
    expect(out).toContain("0 of 12 samples");
    expect(out).toContain("Mar 3, 2024");
    expect(out).toContain("days ago");
    expect(html(h(DataState, { kind: "loading", what: "tests" }))).toContain('role="status"');
    expect(html(h(Skeleton, { label: "Reading the tests", figure: true, plot: 200 }))).toContain("Reading the tests");
  });
  it("availability names each variable's latest observation, or says there is none", () => {
    const out = html(
      h(Availability, {
        label: "What was measured, and when",
        domain: [t0, t0 + 400 * DAY],
        now: "2026-10-13T06:00:00Z",
        variables: [
          { id: "pDA", label: "pDA", spans: [{ from: t0, to: t0 + 30 * DAY, kind: "measured" }], latest: "2025-01-31", summary: "5 of 49 samples" },
          { id: "chl", label: "Chlorophyll", spans: [], latest: null },
        ],
        onPick: () => {},
        onJumpLatest: () => {},
        selected: "pDA",
      }),
    );
    expect(out).toContain("Latest: Jan 31, 2025");
    expect(out).toContain("5 of 49 samples");
    expect(out).toContain("No observation in the record");
    expect(out).toContain('aria-pressed="true"');
  });
});

describe("v13 plot kit", () => {
  const data = [
    { t: t0, v: 2 },
    { t: t0 + 7 * DAY, v: null, kind: "reported-zero" as const, raw: "0" },
    { t: t0 + 14 * DAY, v: 4 },
  ];
  it("a plot is one labelled, focusable chart with a summary, a live region and marks by state", () => {
    const out = html(
      h(
        Plot,
        { label: "pDA at Monterey Wharf", data, x: { domain: [t0 - DAY, t0 + 20 * DAY] }, y: { unit: "ng/mL" }, "data-testid": "p" },
        h(GridY),
        h(AxisX),
        h(AxisY, { title: "pDA, ng/mL" }),
        h(NoData, { spans: [[t0 + 15 * DAY, t0 + 20 * DAY]] }),
        h(Threshold, { value: 3, label: "Reference 3 ng/mL" }),
        h(Line, { data }),
        h(Points, { data }),
        h(NowRule, { t: t0 + 10 * DAY }),
        h(Crosshair),
        h(Selection),
        h(Tip),
      ),
    );
    expect(out).toContain('role="group"');
    expect(out).toContain('aria-roledescription="chart"');
    expect(out).toContain('tabindex="0"');
    expect(out).toContain("pDA at Monterey Wharf: 3 samples, Jan 1, 2025 to Jan 15, 2025.");
    expect(out).toContain('aria-live="polite"');
    // measured: solid circles; the reported zero: a hollow square on the baseline; never a line through it
    expect(out.match(/class="ptFill"/g)).toHaveLength(2);
    expect(out.match(/<rect[^>]*class="ptHollow"/g)).toHaveLength(1);
    expect(out).not.toContain('class="line"');
    expect(out).toContain("Reference 3 ng/mL");
    expect(out).toContain("pDA, ng/mL");
  });
  it("a plot with nothing to draw says why and what to do; the table holds the same samples", () => {
    expect(html(h(PlotState, { kind: "not-collected", what: "pDA at Monterey Wharf", reason: "0 of 12 samples in this window measured pDA", action: h(Button, null, "Latest sample") }))).toContain("0 of 12 samples");
    const table = html(
      h(PlotTable, {
        caption: "pDA samples",
        rows: data,
        rowKey: (r: (typeof data)[number]) => String(r.t),
        selected: String(t0),
        columns: [
          { id: "v", head: "ng/mL", numeric: true, cell: (r: (typeof data)[number]) => r.raw ?? String(r.v) },
        ],
      }),
    );
    expect(table.match(/<tr/g)).toHaveLength(4);
    expect(table).toContain('aria-selected="true"');
  });
  it("the waterfall says every step with how it was worked out, and never draws an unknown as 0", () => {
    const steps = [
      { id: "gross", label: "Gross", amount: 3600, kind: "gross" as const, formula: "400 lb × $9.00" },
      { id: "fuel", label: "Fuel", amount: 247.2, kind: "cost" as const, formula: "30 gal × $8.24" },
      { id: "crew", label: "Crew", amount: null, kind: "cost" as const, waiting: "waiting for who pays fuel" },
      { id: "net", label: "Net to boat", amount: null, kind: "net" as const, waiting: "waiting for crew" },
    ];
    const out = html(h(WaterfallChart, { steps, label: "Gross to net to boat, hypothetical", cents: true }, h(BreakEvenMarker)));
    expect(out).toContain("Gross: $3,600.00. 400 lb × $9.00.");
    expect(out).toContain("Fuel: −$247.20. 30 gal × $8.24.");
    expect(out).toContain("Crew: waiting for who pays fuel.");
    // a bar that cannot be drawn says its short word inside its own column (clipped there); the kit adds no sentence
    expect(out.match(/<text[^>]*clip-path="url\(#wf[^"]+\)"[^>]*>waiting/g)).toHaveLength(2);
    expect(out).not.toContain("waterfall-waiting");
    expect(out).toContain("Break-even");
    expect(out.match(/class="wfSolid"/g)).toHaveLength(1);
  });
  it("the sensitivity grid is a keyboard grid with the break-even contour between the cells it separates", () => {
    const rows = [200, 400].map((v) => ({ label: `${v} lb`, value: v }));
    const cols = [0.5, 2, 9].map((v) => ({ label: `$${v}`, value: v }));
    const out = html(h(SensitivityGrid, { caption: "Net to boat", rows, cols, rowAxis: "Catch", colAxis: "Price", value: (r: number, c: number) => rows[r].value * cols[c].value - 500, selected: { row: 1, col: 2 }, base: { row: 1, col: 2 } }));
    expect(out).toContain('role="grid"');
    expect(out.match(/tabindex="0"/g)).toHaveLength(1);
    expect(out.match(/aria-selected="true"/g)).toHaveLength(1);
    // 200 lb breaks even at $9 (col 2); 400 lb at $2 (col 1)
    expect(out).toMatch(/data-be-left="true"[^>]*data-testid="sens-0-2"/);
    expect(out).toMatch(/data-be-left="true"[^>]*data-testid="sens-1-1"/);
    expect(out).toContain("−$400");
    expect(out).toContain("yours");
  });
});

describe("v13 motion", () => {
  it("has the added presets, and a figure's spring cannot overshoot", () => {
    for (const k of ["reflow", "pane", "shared", "select", "reveal", "value", "swap", "scrollCondense"]) expect(motion).toHaveProperty(k);
    const v = motion.value as { stiffness: number; damping: number };
    // critical damping for unit mass is 2√k: at or above it the spring never passes its target
    expect(v.damping).toBeGreaterThanOrEqual(2 * Math.sqrt(v.stiffness));
    expect((motion.swapIn as { delay: number }).delay).toBe((motion.swapOut as { duration: number }).duration);
  });
});

describe("v13 block 1: page parts", () => {
  it("gates stay separate: each has its authority, its state in words, advisories set first, and how it relates to the others", () => {
    const out = html(
      h(GateList, {
        label: "What decides a trip from Moss Landing",
        gates: [
          { id: "whale", title: "Whale measures", authority: "CDFW RAMP", state: "ok", tone: "regulation", headline: "Season delayed to at least Jan 1, 2027", relation: "A crab test says nothing about this delay." },
          { id: "domoic", title: "Domoic acid", authority: "CDPH tests", state: "old", headline: "Highest 9.8 of 30 ppm", date: "Sampled Sep 28, 45 days ago" },
          { id: "sea", title: "Sea", authority: "NWS and NDBC", state: "ok", headline: "Small craft advisory until 3 AM", facts: [{ id: "a", role: "advisory", text: "Small craft advisory until 3 AM" }, { id: "b", role: "measured", text: "Waves 4.9 ft at 17 s", old: true }], actions: [{ id: "nws", label: "NWS marine forecast", href: "https://example.org", external: true }] },
        ],
      }),
    );
    expect(out.match(/data-gate="/g)).toHaveLength(3);
    expect(out).toContain('data-tone="regulation"');
    expect(out).toContain('data-state="old"');
    expect(out).toMatch(/data-state-mark="stale"[\s\S]*old/);
    expect(out).toContain('data-role="advisory"');
    expect(out).toContain("(old)");
    expect(out).toContain("A crab test says nothing about this delay.");
    expect(out).toContain('target="_blank"');
  });
  it("advisories: shown with their source; none in force is said only with the source that was read; unknown renders nothing", () => {
    const some = html(h(Advisories, { label: "In force now", items: [{ id: "sca", text: "Small craft advisory until 3 AM Tuesday", source: "NWS, issued Oct 12, 8:10 PM PT", area: "Monterey Bay" }] }));
    expect(some).toContain("Small craft advisory until 3 AM Tuesday");
    expect(some).toContain("NWS, issued Oct 12, 8:10 PM PT");
    expect(html(h(Advisories, { label: "In force now", items: [], none: "No marine advisory in force (NWS, read 6 AM)" }))).toContain("No marine advisory in force");
    expect(html(h(Advisories, { label: "In force now", items: [] }))).toBe("");
  });
  it("an advisory is an official kind; a stale sounding is marked and says (old)", () => {
    expect(markFor("advisory")).toBe("official");
    const out = html(h(Sounding, { x: 0, y: 0, value: "9.8", unit: "of 30 ppm", kind: "measured", stale: true, placement: "box" }));
    expect(out).toContain('data-stale="true"');
    expect(out).toContain("staleFig");
  });
  it("a readout outside a model's footprint has its own mark and words, a subtitle and its links in every state", () => {
    const out = html(h(Readout, { title: "C-HARM here", subtitle: "Cell 3 km, C-HARM grid", lat: 36.8, lon: -122.05, state: "outside", actions: h("a", { href: "/research" }, "See the record") }));
    expect(out).toContain('data-state-mark="outside-coverage"');
    expect(out).toContain("not a low value");
    expect(out).toContain("Cell 3 km, C-HARM grid");
    expect(out).toContain("See the record");
    expect(out).not.toContain('aria-label="no view"');
  });
  it("an editable name reads as text with a named rename control", () => {
    const out = html(h(EditableText, { label: "Trip name", value: "Crab, 400 lb at $9" }));
    expect(out).toContain("Crab, 400 lb at $9");
    expect(out).toContain('aria-label="Rename: Trip name"');
    expect(html(h(EditableText, { label: "Trip name", value: "" }))).toContain("Untitled");
  });
  it("an option list is a radio group whose empty choices stay listed, disabled, saying why", () => {
    const out = html(
      h(OptionList, {
        label: "Variable",
        value: "pDA",
        onChange: () => {},
        options: [
          { value: "pDA", label: "Particulate domoic acid", meta: "165 samples · latest Mar 3, 2024" },
          { value: "chl", label: "Chlorophyll", meta: "Not collected here", disabled: true },
        ],
      }),
    );
    expect(out).toContain('role="radiogroup"');
    expect(out.match(/aria-checked="true"/g)).toHaveLength(1);
    expect(out).toContain('aria-disabled="true"');
    expect(out).toContain("Not collected here");
  });
  it("an inspector is a live, labelled region with its facts as a definition list", () => {
    const out = html(h(Inspector, { title: "Sample, Mar 3, 2024", subtitle: "Monterey Wharf · pDA", onClose: () => {} }, h(Facts, { items: [{ label: "Value", value: "6.2 ng/mL" }, { label: "QC", value: "passed" }] })));
    expect(out).toContain('aria-label="Sample, Mar 3, 2024"');
    expect(out).toContain('aria-live="polite"');
    expect(out.match(/<dt/g)).toHaveLength(2);
  });
  it("the waterfall takes any number of steps: rows when columns would be too narrow, with the same words", () => {
    const steps = ["Gross", "Fees", "Fuel", "Bait and ice", "Observer", "Moorage", "Crew", "Net to boat"].map((l, i, a) => ({ id: `s${i}`, label: l, amount: i === 0 ? 3600 : i === a.length - 1 ? 2000 : 200, kind: (i === 0 ? "gross" : i === a.length - 1 ? "net" : "cost") as "gross" | "cost" | "net", formula: i === 2 ? "30 gal × $8.24" : undefined }));
    // 640 px wide in a server render: eight steps have 71 px each, so rows
    const out = html(h(WaterfallChart, { steps, label: "Gross to net" }));
    expect(out).toContain('data-orientation="rows"');
    expect(out.match(/data-step="/g)).toHaveLength(8);
    expect(out).toContain("Fuel: −$200. 30 gal × $8.24.");
    expect(html(h(WaterfallChart, { steps: steps.slice(0, 3), label: "Gross to net" }))).not.toContain('data-orientation="rows"');
    expect(html(h(WaterfallChart, { steps: steps.slice(0, 3), label: "Gross to net", orientation: "rows" }))).toContain('data-orientation="rows"');
  });
  it("a one-variable sweep is a plot on a linear axis with its unit, and break-even is a named rule where the result crosses zero", () => {
    const pts = [0, 20, 40, 60, 80].map((lb) => ({ t: lb, v: lb * 9 - 351, kind: "estimated" as const }));
    const out = html(h(Plot, { label: "Net to boat as catch changes, hypothetical", data: pts, x: { type: "linear", domain: [0, 80], unit: "lb" }, y: { unit: "$" } }, h(AxisX), h(AxisY), h(Line, { data: pts, kind: "estimated" }), h(BreakEvenMarker, { at: 39, label: "Break-even 39 lb" })));
    expect(out).toContain("5 points, 0 lb to 80 lb.");
    expect(out).toContain("Break-even 39 lb");
    expect(out).toContain("lineModel");
    // off the axis: no vertical rule is drawn for a break-even the sweep never reaches
    expect(html(h(Plot, { label: "x", data: pts, x: { type: "linear", domain: [0, 30] }, y: { domain: [-400, -50] } }, h(BreakEvenMarker, { at: 39 })))).not.toContain("data-break-even");
  });
  it("a narrow brush frame puts its handles outside it, facing outward, so they never share a hit area", () => {
    const DAYms = 86_400_000;
    const extent: [number, number] = [Date.UTC(2005, 0, 1), Date.UTC(2026, 0, 1)];
    const narrow = html(h(Brush, { extent, value: [Date.UTC(2024, 0, 1), Date.UTC(2025, 0, 1)], onChange: () => {}, label: "every strip" }));
    expect(narrow.match(/data-outward="true"/g)).toHaveLength(2);
    const lefts = [...narrow.matchAll(/aria-label="Window (start|end)"[^>]*|style="left:([\d.]+)px;width:44px/g)];
    expect(lefts.length).toBeGreaterThan(0);
    const wide = html(h(Brush, { extent, value: [extent[0], extent[0] + 3000 * DAYms], onChange: () => {}, label: "every strip" }));
    expect(wide).not.toContain("data-outward");
  });
});

describe("v13 block 1: chart", () => {
  it("a line obstacle covers a transect in padded boxes", () => {
    const r = lineObstacle({ x: 0, y: 100 }, { x: 480, y: 100 }, 8, 48);
    expect(r).toHaveLength(10);
    expect(r[0]).toEqual({ x0: -8, y0: 92, x1: 56, y1: 108 });
    expect(r[9].x1).toBe(488);
  });
  it("a scene's tall box is used in a tall, narrow free area and its own box in a wide one", () => {
    const at: [number, number] = [-121.79, 36.8];
    const scene = { id: "t", view: { bounds: [[-122.3, 36.5], [-121.6, 37.1]] as [[number, number], [number, number]], tall: tallBoundsAround(at, 60) } };
    const none = { top: 0, right: 0, bottom: 0, left: 0 };
    const wide = cameraFor(scene, { width: 1000, height: 800 }, none);
    const tall = cameraFor(scene, { width: 360, height: 800 }, none);
    const plain = cameraFor({ id: "p", view: { bounds: scene.view.bounds } }, { width: 360, height: 800 }, none);
    // the tall box runs up and down the coast from the place: centred on it, and closer in than the wide box squeezed into the pane
    expect(Math.abs(tall.center[1] - at[1])).toBeLessThan(0.02);
    expect(tall.zoom).toBeGreaterThan(plain.zoom);
    expect(wide.zoom).toBeCloseTo(cameraFor({ id: "p", view: { bounds: scene.view.bounds } }, { width: 1000, height: 800 }, none).zoom, 6);
  });
});

describe("v13 block 4", () => {
  it("a reported zero's default sentence claims no limit and no absence; a page may give its source's own words", () => {
    expect(DATA_STATE_WORDS["reported-zero"].legend).not.toMatch(/none|not detected|absent\b(?!e)/i);
    expect(DATA_STATE_WORDS["reported-zero"].legend).toContain("no detection limit is given");
    const out = html(h(DataStateLegend, { kinds: ["reported-zero", "rejected"], words: { "reported-zero": "Reported 0: at or below a limit CalHABMAP does not publish" } }));
    expect(out).toContain("at or below a limit CalHABMAP does not publish");
    expect(out).toContain("Rejected by quality control");
    expect(html(h(DataState, { kind: "reported-zero", word: "At or below an unpublished limit" }))).toContain("At or below an unpublished limit");
  });
  it("a value quality control rejected is never drawn on the axis", () => {
    const pts = [{ t: t0, v: 5 }, { t: t0 + DAY, v: 900, kind: "rejected" as const }, { t: t0 + 2 * DAY, v: 6 }];
    const out = html(h(Plot, { label: "x", data: pts, x: { domain: [t0 - DAY, t0 + 3 * DAY] } }, h(Line, { data: pts }), h(Points, { data: pts })));
    // the y domain ignores it, and the line breaks at it rather than passing through 900
    expect(out).not.toContain('class="line"');
    expect(out.match(/class="ptFill"/g)).toHaveLength(2);
  });
  it("a page can pin a sample from outside by its position alone, and every plot inside shows it", () => {
    const pts = [{ t: t0, v: 5 }, { t: t0 + DAY, v: 6 }];
    const out = html(
      h(PlotSyncAny, { value: { hoverT: null, pinned: { t: t0 + DAY } } }, h(Plot, { label: "a", data: pts, x: { domain: [t0 - DAY, t0 + 2 * DAY] }, "data-testid": "a" }, h(Selection)), h(Plot, { label: "b", data: pts, x: { domain: [t0 - DAY, t0 + 2 * DAY] }, "data-testid": "b" }, h(Selection))),
    );
    expect(out.match(new RegExp(`data-pinned-t="${t0 + DAY}"`, "g"))).toHaveLength(2);
    expect(out.match(/data-selection/g)).toHaveLength(2);
  });
  it("with little room above the plot the y title takes the top tick's place", () => {
    const pts = [{ t: t0, v: 0 }, { t: t0 + DAY, v: 100 }];
    const tight = html(h(Plot, { label: "x", data: pts, x: { domain: [t0, t0 + DAY] }, y: { domain: [0, 100] }, height: 200, margin: { top: 8 } }, h(AxisY, { title: "pDA, ng/mL" })));
    expect(tight).toContain("pDA, ng/mL");
    expect(tight).not.toMatch(/>100</);
    expect(html(h(Plot, { label: "x", data: pts, x: { domain: [t0, t0 + DAY] }, y: { domain: [0, 100] }, height: 200 }, h(AxisY, { title: "pDA, ng/mL" })))).toMatch(/>100</);
  });
  it("an option can carry a full-width strip under its words", () => {
    const out = html(h(OptionList, { label: "Variable", value: "a", onChange: () => {}, options: [{ value: "a", label: "pDA", strip: h("i", { "data-strip": "" }) }] }));
    expect(out).toMatch(/optStrip[^>]*aria-hidden="true"[^>]*><i data-strip/);
  });
  it("the waterfall's amount column is as wide as its widest figure", () => {
    const steps = [
      { id: "g", label: "Gross", amount: 125000, kind: "gross" as const },
      { id: "c", label: "Fuel", amount: 2500, kind: "cost" as const },
      { id: "n", label: "Net to boat", amount: 122500, kind: "net" as const },
    ];
    const out = html(h(WaterfallChart, { steps, label: "x", orientation: "rows" }));
    // every bar ends before the amounts' column begins (640 px wide in a server render)
    const ends = [...out.matchAll(/<rect y="[\d.]+" height="[\d.]+"[^>]*class="wf(?:Solid|Cost)"/g)].length;
    expect(ends).toBe(3);
    expect(out).toContain("$122,500");
  });
});

describe("v13 review round", () => {
  it("the waterfall's rows keep the worked sum's place whether or not a bar is chosen", () => {
    const steps = [
      { id: "g", label: "Gross", amount: 3600, kind: "gross" as const },
      { id: "n", label: "Net to boat", amount: 3600, kind: "net" as const },
    ];
    const out = html(h(WaterfallChart, { steps, label: "x", orientation: "rows" }));
    expect(out).toContain('data-testid="waterfall-callout-slot"');
    expect(out).toContain("Choose a bar to see how it was worked out.");
  });
  it("an editable name can be held to one line, with its full text as the title", () => {
    const out = html(h(EditableText, { label: "Trip name", value: "Dungeness crab, 400 lb at $9.00 a pound, Tuesday", truncate: true }));
    expect(out).toContain('data-truncate="true"');
    expect(out).toContain('title="Dungeness crab, 400 lb at $9.00 a pound, Tuesday"');
    expect(html(h(EditableText, { label: "Trip name", value: "x" }))).not.toContain("data-truncate");
  });
});
