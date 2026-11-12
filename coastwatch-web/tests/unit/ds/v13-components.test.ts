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
import { Button } from "@/nx/ds/Button";
import { ButtonGroup } from "@/nx/ds/ButtonGroup";
import { IconButton } from "@/nx/ds/IconButton";
import { Disclosure, Hint, RangeControl, ScrubNumber, ToggleRow } from "@/nx/ds/Controls";
import { Availability, DataState, DataStateLegend, DATA_STATE_KINDS, DATA_STATE_WORDS, Skeleton } from "@/nx/ds/DataState";
import { Figure } from "@/nx/ds/Figure";
import { Desk, DeskAside, DeskBody, DeskFoot, DeskHeader, DeskMain, DeskSticky, PageLayout } from "@/nx/ds/layout/Desk";
import { getLayout, PANE_DETENTS, setPageLayout, setPaneDetent } from "@/nx/ds/layout/pane";
import { AxisX, AxisY, Crosshair, GridY, Line, NoData, NowRule, Plot, Points, Selection, Threshold, Tip } from "@/nx/ds/plot/Plot";
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
    expect(DATA_STATE_KINDS).toHaveLength(14);
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
