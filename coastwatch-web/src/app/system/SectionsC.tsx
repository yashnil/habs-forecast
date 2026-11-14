"use client";
import {
  Advisories,
  Availability,
  AxisX,
  AxisY,
  BreakEvenMarker,
  Brush,
  Button,
  ButtonGroup,
  Crosshair,
  DATA_STATE_KINDS,
  DataState,
  DataStateLegend,
  Desk,
  DeskAside,
  DeskBody,
  DeskFoot,
  DeskHeader,
  DeskMain,
  Disclosure,
  Drawer,
  EditableText,
  Facts,
  Figure,
  GateList,
  GridY,
  Hint,
  IconButton,
  Inspector,
  Line,
  NoData,
  NowRule,
  OptionList,
  Plot,
  PlotState,
  PlotSync,
  PlotTable,
  Points,
  RangeControl,
  Recovered,
  ScrubNumber,
  Segmented,
  Selection,
  SensitivityGrid,
  Skeleton,
  Threshold,
  Tip,
  ToggleRow,
  WaterfallChart,
  day,
  describePoint,
  usd,
  type BrushHandle,
  type PlotPoint,
  type WaterfallStep,
} from "@/nx/ds";
import { useMemo, useRef, useState } from "react";
import { NOW, record, station } from "./data";
import { Section, Specimen } from "./parts";

const DAY = 86_400_000;

/** The station's pDA record as plot points: a reported zero is its own state, never a value. */
const PDA: PlotPoint[] = record.pDA.points.map((p) => ({ t: Date.parse(p.t), v: p.flag ? null : p.v, kind: p.flag === "reported0" ? ("reported-zero" as const) : p.flag ? ("below-detection" as const) : ("measured" as const), raw: p.flag === "reported0" ? "0" : p.raw }));
const TEMP: PlotPoint[] = record.temp.points.filter((p) => p.v !== null).map((p) => ({ t: Date.parse(p.t), v: p.v }));
const LAST = PDA[PDA.length - 1].t;
const EXTENT: [number, number] = [PDA[0].t, LAST];

/** v13 parts (docs/coastwatch/v13/design-system/README.md): layout, controls, states, the plot kit, page parts. */
export function SectionsC() {
  return (
    <>
      <DeskPlate />
      <Controls />
      <Figures />
      <DataStates />
      <PlotKit />
      <Money />
      <PageParts />
    </>
  );
}

function DeskPlate() {
  return (
    <Section id="desk" name="Desk" purpose="A desk page's anatomy: a header that keeps what the page is about, the primary visual beside its inputs, and a foot for provenance. The chart is a pane beside it, sized by the reader.">
      <Specimen state="Header, main and aside" wide note="the header condenses after 64 px of scroll in a real page">
        <Desk>
          <DeskHeader context="Moss Landing · Dungeness crab · Hypothetical" title="A trip from Moss Landing" status="Your own numbers; nothing is filled in." actions={<Button variant="secondary">Saved trips</Button>} sticky={false} />
          <DeskBody layout="main-aside">
            <DeskMain>
              <Figure label="Net to boat" value={2602.24} format="money" cents size="xl" note="Before insurance, maintenance and the skipper's own pay." />
            </DeskMain>
            <DeskAside label="Trip inputs">
              <ScrubNumber label="Catch" value="400" onChange={() => {}} unit="lb" />
              <ScrubNumber label="Price" value="9.00" onChange={() => {}} prefix="$" unit="/lb" step={0.25} />
            </DeskAside>
          </DeskBody>
          <DeskFoot>Not a prediction of catch or price.</DeskFoot>
        </Desk>
      </Specimen>
    </Section>
  );
}

function Controls() {
  const [payer, setPayer] = useState<"shared" | "boat" | null>(null);
  const [range, setRange] = useState<"90d" | "1y" | "3y" | "all" | null>("1y");
  const [catchLb, setCatch] = useState("400");
  const [on, setOn] = useState(false);
  const [name, setName] = useState("Crab, 400 lb at $9");
  const [drawer, setDrawer] = useState(false);
  const [variable, setVariable] = useState<"pDA" | "temp" | "dDA">("pDA");
  return (
    <Section id="controls" name="Controls (v13)" purpose="Buttons in three kinds and two sizes, and the inputs a desk page is built from. A choice that has not been made never looks made.">
      <Specimen state="Button kinds and sizes" wide>
        <ButtonGroup>
          <Button variant="primary" size="l">
            Save new
          </Button>
          <Button variant="secondary" size="l">
            Update
          </Button>
          <Button variant="secondary">Duplicate</Button>
          <Button variant="quiet" iconEnd="chevron">
            More
          </Button>
          <IconButton icon="close" label="Close" ruled />
        </ButtonGroup>
      </Specimen>
      <Specimen state="Segmented, nothing chosen then chosen" note="choose one: the block slides">
        <Segmented label="Who pays fuel, bait and ice" options={[{ value: "shared", label: "Shared, off the top" }, { value: "boat", label: "The boat" }]} value={payer} onChange={setPayer} fit="equal" />
      </Specimen>
      <Specimen state="RangeControl with the latest action">
        <RangeControl label="Window" presets={[{ value: "90d", label: "90 d" }, { value: "1y", label: "1 y" }, { value: "3y", label: "3 y" }, { value: "all", label: "All" }]} value={range} onChange={setRange} onJumpLatest={() => setRange(null)} latestLabel={`Latest pDA sample: ${day(LAST, { year: true })}`} />
      </Specimen>
      <Specimen state="ScrubNumber" note="type, Up and Down, or drag the label">
        <ScrubNumber label="Catch" value={catchLb} onChange={setCatch} unit="lb" step={10} hint="Your own number." data-testid="sys-scrub" />
      </Specimen>
      <Specimen state="ToggleRow and Hint">
        <ToggleRow label="Season changes at Moss Landing" hint="Checked when you open Neritic on this device." checked={on} onChange={setOn} />
        <Hint tone="caution">No text message or email is sent.</Hint>
      </Specimen>
      <Specimen state="Disclosure">
        <Disclosure summary="Station" value="600 m from Monterey">
          <p>Sampled weekly from the wharf by CalHABMAP.</p>
        </Disclosure>
        <Disclosure summary="Checks" value="6 of 6 passed" defaultOpen>
          <p>Units, duplicates, dates, ranges, flags and gaps.</p>
        </Disclosure>
      </Specimen>
      <Specimen state="EditableText and Drawer">
        <EditableText label="Trip name" value={name} onChange={setName} />
        <p>
          <Button variant="secondary" onClick={() => setDrawer(true)}>
            Saved trips
          </Button>
        </p>
        <Drawer title="Saved trips" description="Kept on this device." open={drawer} onOpenChange={setDrawer}>
          <p>{name}</p>
        </Drawer>
      </Specimen>
      <Specimen state="OptionList" note="a variable with nothing behind it stays listed">
        <OptionList
          label="Variable"
          value={variable}
          onChange={setVariable}
          options={[
            { value: "pDA", label: "Particulate domoic acid", meta: `${PDA.length} samples · latest ${day(LAST, { year: true })}` },
            { value: "temp", label: "Water temperature", meta: `${TEMP.length} samples` },
            { value: "dDA", label: "Dissolved domoic acid", meta: "Not collected at this station", disabled: true },
          ]}
        />
      </Specimen>
    </Section>
  );
}

function Figures() {
  const [v, setV] = useState(2602.24);
  return (
    <Section id="figure" name="Figure" purpose="A result figure. A change moves from the old value to the new one; it never counts up from zero, never passes its target, and an unknown is words, never 0.">
      <Specimen state="The one result, to the cent" note="press to change the inputs">
        <Figure label="Net to boat" value={v} format="money" cents size="xl" data-testid="sys-figure" />
        <ButtonGroup>
          <Button onClick={() => setV(v > 2000 ? 143.5 : 2602.24)} data-testid="sys-figure-change">
            Another scenario
          </Button>
        </ButtonGroup>
      </Specimen>
      <Specimen state="Secondary and ledger sizes">
        <Figure label="Break-even catch" value={39} unit="lb" kind="estimated" size="l" />
        <Figure label="Crew pay" value={650.56} format="money" cents size="m" layout="inline" />
      </Specimen>
      <Specimen state="Not worked out">
        <Figure label="Net to boat" value={null} waiting="waiting for who pays fuel" size="l" />
      </Specimen>
    </Section>
  );
}

function DataStates() {
  return (
    <Section id="datastate" name="DataState, Availability, Skeleton" purpose="Every state a reading can be in, said the same way everywhere: one mark, one word, one legend sentence. A missing value is never drawn as zero.">
      <Specimen state="All fourteen, inline" wide>
        <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "grid", gap: 4 }}>
          {DATA_STATE_KINDS.map((k) => (
            <li key={k}>
              <DataState kind={k} what="pDA" />
            </li>
          ))}
        </ul>
      </Specimen>
      <Specimen state="Block, with the reason and what to do">
        <DataState kind="not-collected" variant="block" what="pDA in this window" reason="0 of 12 samples in the last 90 days measured pDA." source="CalHABMAP" date={new Date(LAST).toISOString()} now={NOW} action={<Button variant="secondary">Latest pDA sample: {day(LAST, { year: true })}</Button>} />
      </Specimen>
      <Specimen state="Legend for a view">
        <DataStateLegend kinds={["measured", "reported-zero", "below-detection", "forecast", "not-collected"]} />
      </Specimen>
      <Specimen state="Availability by variable and date" wide>
        <Availability
          label="What was measured, and when"
          domain={EXTENT}
          window={[LAST - 365 * DAY, LAST]}
          now={NOW}
          selected="pDA"
          onPick={() => {}}
          onJumpLatest={() => {}}
          variables={[
            { id: "pDA", label: "Particulate domoic acid", spans: spansOf(PDA), latest: new Date(LAST).toISOString(), summary: `${PDA.filter((p) => p.kind === "measured").length} measured, ${PDA.filter((p) => p.kind === "reported-zero").length} reported 0` },
            { id: "temp", label: "Water temperature", spans: spansOf(TEMP), latest: new Date(TEMP[TEMP.length - 1].t).toISOString() },
            { id: "dDA", label: "Dissolved domoic acid", spans: [], latest: null },
          ]}
        />
      </Specimen>
      <Specimen state="Loading and recovered">
        <Skeleton label="Reading the tests" figure lines={2} />
        <Recovered at={NOW} now={NOW} />
      </Specimen>
    </Section>
  );
}

/** Runs of samples no more than 45 days apart, as availability spans. */
function spansOf(pts: PlotPoint[]) {
  const out: Array<{ from: number; to: number; kind: "measured" }> = [];
  for (const p of pts) {
    const last = out[out.length - 1];
    if (last && p.t - last.to <= 45 * DAY) last.to = p.t;
    else out.push({ from: p.t, to: p.t, kind: "measured" });
  }
  return out;
}

function PlotKit() {
  const [win, setWin] = useState<[number, number]>([LAST - 2 * 365 * DAY, LAST]);
  const [pinned, setPinned] = useState<PlotPoint | null>(null);
  const [table, setTable] = useState(false);
  const brush = useRef<BrushHandle>(null);
  const shown = useMemo(() => PDA.filter((p) => p.t >= win[0] && p.t <= win[1]), [win]);
  const gaps = useMemo(() => {
    const g: Array<[number, number]> = [];
    for (let i = 1; i < PDA.length; i++) if (PDA[i].t - PDA[i - 1].t > 120 * DAY) g.push([PDA[i - 1].t + 7 * DAY, PDA[i].t - 7 * DAY]);
    return g;
  }, []);
  return (
    <Section id="plot" name="Plot kit" purpose="One interactive plot: one tab stop, the crosshair snaps to real samples, arrows step them, Enter pins one, and a live region reads each. Data never tweens or smooths.">
      <Specimen state="Primary variable, a secondary, one brush" wide note={`${station.name}, CalHABMAP · hover, click, or focus and use the arrow keys`}>
        <RangeControl
          label="Window"
          presets={[{ value: "1y", label: "1 y" }, { value: "2y", label: "2 y" }, { value: "all", label: "All" }]}
          value={win[1] === LAST && Math.abs(win[1] - win[0] - 365 * DAY) < DAY ? "1y" : win[1] === LAST && Math.abs(win[1] - win[0] - 730 * DAY) < DAY ? "2y" : win[0] === EXTENT[0] && win[1] === LAST ? "all" : null}
          onChange={(r) => setWin(r === "all" ? EXTENT : [LAST - (r === "1y" ? 365 : 730) * DAY, LAST])}
          onJumpLatest={() => brush.current?.jumpTo(LAST, 180 * DAY)}
          latestLabel={`Latest pDA sample: ${day(LAST, { year: true })}`}
          action={
            <Button variant="quiet" onClick={() => setTable((t) => !t)} aria-pressed={table}>
              {table ? "Show as plot" : "Show as table"}
            </Button>
          }
        />
        <PlotSync>
          {shown.length ? (
            <Plot label={`${record.pDA.label} at ${station.name}`} data={shown} x={{ domain: win }} y={{ type: "log", unit: record.pDA.units }} height={{ desktop: 340, phone: 240 }} onSelect={setPinned} data-testid="sys-plot">
              <GridY />
              <AxisX />
              <AxisY title={`pDA, ${record.pDA.units}`} />
              <NoData spans={gaps} />
              <Line data={shown} />
              <Points data={shown} />
              <NowRule t={Date.parse(NOW)} />
              <Crosshair />
              <Selection />
              <Tip />
            </Plot>
          ) : (
            <PlotState kind="not-collected" what={`pDA at ${station.name}`} reason="No sample in this window." height={340} action={<Button onClick={() => brush.current?.jumpTo(LAST, 180 * DAY)}>Latest pDA sample</Button>} />
          )}
          <Plot label={`Water temperature at ${station.name}`} data={TEMP} x={{ domain: win }} y={{ unit: "°C" }} height={96} margin={{ bottom: 8, top: 24 }}>
            <AxisY title="Temperature, °C" ticks={2} />
            <Line data={TEMP} />
            <Crosshair />
          </Plot>
        </PlotSync>
        <Brush extent={EXTENT} value={win} onChange={setWin} handle={brush} inset={{ left: 64, right: 12 }} label="both plots" scale="log" points={record.pDA.points} />
        {pinned ? (
          <Inspector title={`Sample, ${day(pinned.t, { year: true })}`} subtitle={`${station.name} · pDA`} onClose={() => setPinned(null)} data-testid="sys-inspector">
            <Facts
              items={[
                { label: "Value", value: describePoint(pinned, { unit: record.pDA.units }).split(" · ")[1] },
                { label: "State", value: <DataState kind={pinned.kind ?? "measured"} /> },
                { label: "Source", value: "CalHABMAP via SCCOOS ERDDAP" },
                { label: "Place", value: station.name },
              ]}
            />
          </Inspector>
        ) : null}
        {table ? (
          <PlotTable
            caption={`pDA samples at ${station.name}, ${day(win[0], { year: true })} to ${day(win[1], { year: true })}`}
            rows={shown.slice(-12)}
            rowKey={(r) => String(r.t)}
            selected={pinned ? String(pinned.t) : null}
            onSelect={setPinned}
            columns={[
              { id: "d", head: "Sampled", cell: (r) => day(r.t, { year: true }) },
              { id: "v", head: record.pDA.units, numeric: true, cell: (r) => r.raw ?? (r.v === null ? "" : String(r.v)) },
              { id: "k", head: "State", cell: (r) => <DataState kind={r.kind ?? "measured"} /> },
            ]}
          />
        ) : null}
      </Specimen>
      <Specimen state="Nothing to draw" note="the reason and what to do, never an empty axis">
        <PlotState kind="outside-coverage" what="C-HARM at this station" reason="The model's grid does not reach inside the harbour." height={160} />
      </Specimen>
      <Specimen state="Threshold, named">
        <Plot label="Crab viscera domoic acid, Monterey Bay" data={[{ t: LAST - 60 * DAY, v: 4.1 }, { t: LAST - 30 * DAY, v: 9.8 }]} x={{ domain: [LAST - 90 * DAY, LAST] }} y={{ domain: [0, 40], unit: "ppm" }} height={160}>
          <GridY />
          <AxisX />
          <AxisY title="ppm" />
          <Threshold value={30} label="Action level 30 ppm" />
          <Points data={[{ t: LAST - 60 * DAY, v: 4.1 }, { t: LAST - 30 * DAY, v: 9.8 }]} />
          <Crosshair />
          <Tip />
        </Plot>
      </Specimen>
    </Section>
  );
}

const TRIP = { lb: 400, price: 9, fuel: 247.2, bait: 100, share: 0.2 };
function tripSteps(lb: number, price: number): WaterfallStep[] {
  const gross = lb * price;
  const crew = Math.max(0, (gross - TRIP.fuel - TRIP.bait) * TRIP.share);
  return [
    { id: "gross", label: "Gross", amount: gross, kind: "gross", formula: `${lb.toLocaleString("en-US")} lb × ${usd(price, { cents: true })}` },
    { id: "fuel", label: "Fuel", amount: TRIP.fuel, kind: "cost", formula: "30 gal × $8.24" },
    { id: "bait", label: "Bait and ice", amount: TRIP.bait, kind: "cost", formula: "As entered" },
    { id: "crew", label: "Crew", amount: crew, kind: "cost", formula: `20 % × (${usd(gross)} − ${usd(TRIP.fuel + TRIP.bait, { cents: true })})` },
    { id: "net", label: "Net to boat", amount: gross - TRIP.fuel - TRIP.bait - crew, kind: "net", formula: "Gross less fuel, bait and ice, and crew" },
  ];
}
const netOf = (lb: number, price: number) => tripSteps(lb, price)[4].amount as number;

function Money() {
  const rows = [200, 300, 400, 500].map((v) => ({ label: `${v} lb`, value: v }));
  const cols = [1, 3, 6, 9, 12].map((v) => ({ label: `$${v}`, value: v }));
  const [cell, setCell] = useState<{ row: number; col: number } | null>(null);
  const [sel, setSel] = useState({ row: 2, col: 3 });
  const steps = tripSteps(rows[sel.row].value, cols[sel.col].value);
  const sweep: PlotPoint[] = Array.from({ length: 17 }, (_, i) => ({ t: i * 50, v: netOf(i * 50, TRIP.price), kind: "estimated" as const }));
  const breakEven = Math.ceil((TRIP.fuel + TRIP.bait) / TRIP.price);
  return (
    <Section id="money" name="WaterfallChart, SensitivityGrid, sweep" purpose="A trip's arithmetic, large: gross to net as bars that each say how they were worked out, a grid of scenarios with the break-even line drawn as a contour, and one input swept against the result. Hypothetical, from the reader's own numbers.">
      <Specimen state="Gross to net, with a previewed scenario" wide note="hover or focus a bar; hover a cell below">
        <WaterfallChart steps={steps} preview={cell ? tripSteps(rows[cell.row].value, cols[cell.col].value) : null} domain={[-400, 6000]} label="Gross to net to boat, hypothetical, from your numbers" cents data-testid="sys-waterfall">
          <BreakEvenMarker />
        </WaterfallChart>
      </Specimen>
      <Specimen state="Any number of steps, as rows" note="rows whenever a column would be under 88 px">
        <WaterfallChart steps={[...steps.slice(0, 3), { id: "obs", label: "Observer", amount: 60, kind: "cost" }, { id: "moor", label: "Moorage", amount: 35, kind: "cost" }, { id: "wait", label: "Fees", amount: null, kind: "cost", waiting: "waiting for fees" }, ...steps.slice(3)]} orientation="rows" label="Gross to net to boat, eight steps" />
      </Specimen>
      <Specimen state="Catch down, price across" wide>
        <SensitivityGrid caption="Net to boat as catch and price move, hypothetical" rows={rows} cols={cols} rowAxis="Catch" colAxis="Price per lb" value={(r, c) => netOf(rows[r].value, cols[c].value)} selected={sel} onSelect={setSel} onPreview={setCell} base={{ row: 2, col: 3 }} />
      </Specimen>
      <Specimen state="One input swept" wide note="net to boat as catch changes, at $9.00 a pound">
        <Plot label="Net to boat as catch changes, hypothetical" data={sweep} x={{ type: "linear", domain: [0, 800], unit: "lb" }} y={{ unit: "", format: (v) => usd(Math.round(v)) }} height={{ desktop: 260, phone: 200 }}>
          <GridY />
          <AxisX title="Catch, lb" />
          <AxisY title="Net to boat" />
          <Line data={sweep} kind="estimated" />
          <BreakEvenMarker at={breakEven} label={`Break-even ${breakEven} lb`} />
          <Crosshair />
          <Tip />
        </Plot>
      </Specimen>
    </Section>
  );
}

function PageParts() {
  return (
    <Section id="gates" name="GateList and Advisories" purpose="What decides a trip, as separate gates: who decides each, its state in one line, its date, and how it relates to the others. Nothing combines them into a verdict.">
      <Specimen state="Advisory in force, then none, with the source read">
        <Advisories label="In force now" items={[{ id: "sca", text: "Small craft advisory until 3 AM Tuesday", area: "Monterey Bay", source: "NWS Monterey · issued Oct 12, 8:10 PM PT" }]} />
        <Advisories label="In force now" items={[]} none="No marine advisory in force (NWS Monterey, read Oct 12, 8:10 PM PT)." />
      </Specimen>
      <Specimen state="Three gates" wide>
        <GateList
          label="What decides a trip from Moss Landing"
          gates={[
            { id: "whale", title: "Whale measures", authority: "CDFW RAMP", state: "ok", tone: "regulation", headline: "Season delayed to at least Jan 1, 2027", date: "Official · CDFW · checked Oct 12 · not verified", relation: "A crab test says nothing about this delay: it is a whale measure." },
            { id: "domoic", title: "Domoic acid", authority: "CDPH tests", state: "old", headline: "Highest 9.8 of 30 ppm in 6 samples", date: "Sampled Sep 28, 45 days ago", relation: "An old test is a record of that day, not of the crab now." },
            { id: "sea", title: "Sea", authority: "NWS and NDBC", state: "ok", headline: "Small craft advisory until 3 AM Tuesday", facts: [{ id: "a", role: "advisory", text: "Small craft advisory until 3 AM Tuesday", stamp: "NWS · issued Oct 12, 8:10 PM PT" }, { id: "b", role: "measured", text: "Waves 4.9 ft at 17 s", stamp: "NDBC 46236 · Oct 12, 10:30 PM PT" }], actions: [{ id: "nws", label: "NWS marine forecast", href: "https://www.weather.gov/mtr/", external: true }] },
            { id: "none", title: "Buoy", authority: "NDBC", state: "unavailable", headline: "The buoy could not be read" },
          ]}
        />
      </Specimen>
    </Section>
  );
}
