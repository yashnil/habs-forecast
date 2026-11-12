"use client";

import { useMemo, useState, type ReactNode } from "react";
import { AxisX, AxisY, BreakEvenMarker, Button, Crosshair, Figure, GridY, Hint, Line, NowRule, Plot, Points, PlotState, Segmented, SensitivityGrid, Tip, useMedia, WaterfallChart, type PlotPoint, type WaterfallStep } from "@/nx/ds";
import { HYPOTHETICAL, type TripFlag } from "./context";
import type { CoastFishery } from "./fisheries";
import {
  assumptions,
  breakEvenCatchText,
  breakEvenFuelText,
  breakEvenPriceText,
  gridValue,
  GRID_COLS,
  GRID_ROWS,
  money,
  moneyExtent,
  sweep,
  waitingWords,
  type OutputId,
  type Result,
  type Scale,
  type SweepVariable,
  type TripResult,
  type Workspace,
} from "./workspace";
import s from "./trip.module.css";

const ROWS = GRID_ROWS.map((r) => ({ label: r.label, value: r.factor }));
const COLS = GRID_COLS.map((c) => ({ label: c.label, value: c.factor }));
const BASE = { row: 1, col: 1 };
const pctWord = (f: number) => (f === 1 ? "as entered" : `${f > 1 ? "+" : "−"}${Math.round(Math.abs(f - 1) * 100)} %`);

export type Cell = { row: number; col: number };

/**
 * The grid scenario being looked at: pointed at (hover, focus) or held (pressed, tapped). Kept above the results
 * so the phone's sticky figure shows the same scenario as the hero.
 */
export function useTripPreview(result: TripResult, preview: (scale: Scale) => TripResult) {
  const [hover, setHover] = useState<Cell | null>(null);
  const [pinned, setPinned] = useState<Cell | null>(null);
  // a cell exists only while the grid does: a gap opening under a held scenario returns to the reader's numbers
  // a finger has no hover: a touch leaves a pointerenter behind with no leave to clear it, so on a touch screen
  // only the held cell counts (a tap holds, a second tap returns)
  const touch = useMedia("(hover: none)");
  const cell = result.grid ? (touch ? pinned : (hover ?? pinned)) : null;
  const moved = cell && !(cell.row === BASE.row && cell.col === BASE.col) ? cell : null;
  const shown = useMemo(() => (moved ? preview({ catch: ROWS[moved.row].value, price: COLS[moved.col].value }) : result), [moved, preview, result]);
  const words = moved ? `Showing catch ${pctWord(ROWS[moved.row].value)}, price ${pctWord(COLS[moved.col].value)}.` : "";
  return { shown, moved, pinned: result.grid ? pinned : null, setHover, setPinned, words };
}
export type TripPreview = ReturnType<typeof useTripPreview>;
const SWEEPS: Array<{ value: SweepVariable; label: string }> = [
  { value: "catch", label: "Catch" },
  { value: "price", label: "Price" },
  { value: "fuelPrice", label: "Fuel price" },
];
const VIEWS: Array<{ value: "grid" | "sweep"; label: string }> = [
  { value: "grid", label: "Catch × price" },
  { value: "sweep", label: "One at a time" },
];

/** The step ids of the waterfall and the output each one's worked sum comes from. */
const STEP_OUTPUT: Record<string, OutputId> = { gross: "gross", fees: "fees", fuel: "fuel", crew: "crewPay", net: "net" };

/**
 * The trip's results: the one figure the view exists for (net to boat, or a recreational trip's cost), what it
 * is made of, the waterfall from gross down to it, and what happens to it when the catch, the price or fuel
 * move. Every figure is the reader's own numbers through `evaluateTrip`; a figure that cannot be worked out
 * says which fields it waits for, never 0. A previewed grid cell moves the hero, the figures and the waterfall
 * together, so the scenario is read where the result is.
 */
export function TripResults({ ws, result, look, fishery, flag, strip }: { ws: Workspace; result: TripResult; look: TripPreview; fishery: CoastFishery | null; flag: TripFlag | null; /** the scenario strip, set between the figures and the waterfall */ strip?: ReactNode }) {
  const unit = fishery?.unit ?? "lb";
  const rec = result.audience === "recreational";
  const { shown, moved, pinned, setHover, setPinned } = look;
  const [view, setView] = useState<"grid" | "sweep">("grid");
  const [variable, setVariable] = useState<SweepVariable>("catch");
  // one money axis for the reader's numbers and the scenario beside them, held only while a scenario is shown:
  // on their own the bars take the whole height
  const domain = useMemo(() => (moved ? moneyExtent(ws, fishery) : undefined), [moved, ws, fishery]);

  const steps = (r: TripResult): WaterfallStep[] =>
    r.steps.map((st) => {
      const o = STEP_OUTPUT[st.id] ? r.out[rec && st.id === "net" ? "operating" : STEP_OUTPUT[st.id]] : null;
      return { ...st, formula: o && o.status === "value" ? o.formula : undefined };
    });

  const fig = (r: Result) => ({ value: r.status === "value" ? r.value : null, waiting: r.status === "waiting" ? (waitingWords(r, ws, fishery) ?? undefined) : r.status === "none" ? r.reason : "does not apply" });
  const main = rec ? shown.out.operating : shown.out.net;
  const share = shown.share;
  const perCrew = shown.out.perCrew;
  const crewNote = perCrew.status === "value" && share && share.crew > 1 ? `${money(perCrew.value)} each of ${share.crew}` : perCrew.status === "value" && !share ? `${money(perCrew.value)} each` : perCrew.status === "na" ? "Nobody else is paid" : undefined;

  return (
    <div className={s.results} data-testid="trip-results">
      <section className={s.hero} aria-label={rec ? "Trip cost, hypothetical, from your numbers" : "Net to boat, hypothetical, from your numbers"} data-testid="trip-hero">
        <p className={s.hypo}>
          {HYPOTHETICAL}{" "}
          <span aria-live="polite" data-testid="trip-showing">
            {look.words}
          </span>
          {pinned && moved ? (
            <Button variant="quiet" className={s.inlineButton} onClick={() => setPinned(null)}>
              Back to your numbers
            </Button>
          ) : null}
        </p>
        <Figure
          size="xl"
          label={rec ? "Trip cost" : "Net to boat"}
          format="money"
          cents
          {...fig(main)}
          note={rec ? "What the trip costs. A recreational catch is not sold: there is no price, gross or net." : "Before insurance, permits, maintenance, loan payments and your own pay. Not profit or take-home pay."}
          data-testid="trip-net"
        />
      </section>

      {flag || result.warnings.length ? (
        <div className={s.notices}>
        {flag ? (
          <Hint tone="caution" className={s.flag}>
            <span data-testid="trip-flag">{flag.words}</span>
          </Hint>
        ) : null}
        {result.warnings.map((w) => (
          <Hint key={w} tone="caution">
            <span data-testid="trip-warning">{w}</span>
          </Hint>
        ))}
        </div>
      ) : null}

      <div className={s.figures} data-testid="trip-figures">
        {rec ? (
          <>
            <Figure size="l" label="Each person" format="money" cents {...fig(shown.out.perPerson)} data-testid="trip-per-person" />
            <Figure size="l" label="Fuel" format="money" cents {...fig(shown.out.fuel)} data-testid="trip-fuel" />
            <Figure size="l" label="Other costs" format="money" cents {...fig(shown.out.otherCosts)} data-testid="trip-other" />
          </>
        ) : (
          <>
            <Figure size="l" label="Gross" format="money" cents {...fig(shown.out.gross)} data-testid="trip-gross" />
            <Figure size="l" label="Operating costs" format="money" cents {...fig(shown.out.operating)} note="Fuel and the trip's cost lines" data-testid="trip-operating" />
            <Figure size="l" label="Crew pay" format="money" cents {...fig(shown.out.crewPay)} note={crewNote} data-testid="trip-crew" />
            <Figure size="l" label="Break-even catch" format={(v) => breakEvenCatchText(v, unit)} {...fig(shown.out.breakEvenCatch)} note="Rounded up" data-testid="trip-be-catch" />
            <Figure size="l" label="Break-even price" format={(v) => breakEvenPriceText(v, unit)} {...fig(shown.out.breakEvenPrice)} note="Rounded up" data-testid="trip-be-price" />
            {shown.out.fees.status === "value" && shown.out.fees.value > 0 ? <Figure size="m" label="Fees" format="money" cents {...fig(shown.out.fees)} data-testid="trip-fees" /> : null}
            {shown.out.breakEvenFuel.status === "value" ? <Figure size="m" label="Fuel could cost up to" format={breakEvenFuelText} {...fig(shown.out.breakEvenFuel)} data-testid="trip-be-fuel" /> : null}
          </>
        )}
      </div>

      {strip}

      <section aria-labelledby="trip-waterfall">
        <h3 id="trip-waterfall" className={s.h3}>
          {rec ? "What the trip costs" : "From gross to net to boat"}
        </h3>
        <WaterfallChart
          steps={steps(result)}
          preview={moved ? steps(shown) : null}
          domain={domain}
          cents
          height={{ desktop: 320, phone: 240 }}
          label={rec ? "Fuel and costs up to the trip cost, hypothetical, from your numbers" : "Gross to net to boat, hypothetical, from your numbers"}
          data-testid="trip-waterfall"
        >
          {rec ? null : <BreakEvenMarker label="Net to boat $0" />}
        </WaterfallChart>
        <p className={s.small}>Point at a bar, or use the arrow keys, to see how it was worked out.</p>
      </section>

      {rec ? (
        result.fuelSwing ? (
          <p className={s.swing} data-testid="trip-fuel-swing">
            If fuel cost 20 % less the trip would cost {money(result.fuelSwing.down)}; 20 % more, {money(result.fuelSwing.up)}.
          </p>
        ) : null
      ) : (
        <section aria-labelledby="trip-sens" className={s.sens}>
          <div className={s.sensHead}>
            <h3 id="trip-sens" className={s.h3}>
              If things move
            </h3>
            <Segmented label="What moves" hideLabel options={VIEWS} value={view} onChange={setView} />
          </div>
          {!result.grid ? (
            <PlotState kind="missing" what="Net to boat" reason={result.out.net.status === "waiting" ? `Still ${waitingWords(result.out.net, ws, fishery)}.` : "Net to boat has no value for these numbers."} height={160} />
          ) : view === "grid" ? (
            <div className={s.gridScroll}>
            <SensitivityGrid
              caption={`Net to boat, price across and catch down. Trip costs${ws.crewKind === "dayRate" ? " and day pay" : ""} stay as entered; fees and shares move with gross. Point at a figure to see it above; press it to hold it there.`}
              rows={ROWS}
              cols={COLS}
              rowAxis="Catch"
              colAxis="Price"
              value={(r, c) => gridValue(result.grid, ROWS[r].value, COLS[c].value)}
              base={BASE}
              selected={pinned}
              onPreview={setHover}
              onSelect={(c) => setPinned((was) => (was && was.row === c.row && was.col === c.col ? null : c.row === BASE.row && c.col === BASE.col ? null : c))}
            />
            </div>
          ) : (
            <SweepPlot ws={ws} fishery={fishery} variable={variable} setVariable={setVariable} />
          )}
          {result.fuelSwing && view === "grid" ? (
            <p className={s.swing} data-testid="trip-fuel-swing">
              With fuel 20 % cheaper net to boat would be {money(result.fuelSwing.down)}; 20 % dearer, {money(result.fuelSwing.up)}.
            </p>
          ) : null}
        </section>
      )}

      <section aria-label="What these numbers leave out">
        <h3 className={s.h3}>Read it with</h3>
        <ul className={s.assume}>
          {assumptions(ws, fishery).map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}

/** Axis money: whole dollars with a true minus. */
const wholeDollars = (v: number) => `${v < -0.5 ? "−" : ""}$${Math.abs(Math.round(v)).toLocaleString("en-US")}`;

/** Net to boat as one input runs from 0 to twice what was entered: the engine's own line, the break-even marked. */
function SweepPlot({ ws, fishery, variable, setVariable }: { ws: Workspace; fishery: CoastFishery | null; variable: SweepVariable; setVariable: (v: SweepVariable) => void }) {
  const sw = useMemo(() => sweep(ws, fishery, variable), [ws, fishery, variable]);
  const unit = fishery?.unit ?? "lb";
  const x = (v: number) => (variable === "catch" ? `${Math.round(v * 10) / 10} ${unit}` : `$${v.toFixed(2)}${variable === "price" ? `/${unit}` : "/gal"}`);
  // worked from the reader's inputs, never measured: every point says so
  const data: PlotPoint[] = (sw?.points ?? []).map((p, i) => ({ t: p.x, v: p.net, id: String(i), kind: "estimated" }));
  const yours = data.filter((p) => sw && Math.abs(p.t - sw.entered) < 1e-9);
  const what = variable === "catch" ? "catch" : variable === "price" ? "price" : "fuel price";
  const be = sw?.breakEven ?? null;
  return (
    <div data-testid="trip-sweep">
      <Segmented label="Net to boat against" options={SWEEPS} value={variable} onChange={setVariable} />
      {sw ? (
        <>
          <Plot
            label={`Net to boat against ${what}, everything else as entered`}
            data={data}
            x={{ type: "linear", domain: [0, sw.entered * 2] }}
            y={{ unit: "$", format: wholeDollars }}
            height={{ desktop: 280, phone: 220 }}
            margin={{ left: 72 }}
            describe={() => `Net to boat as ${what} runs from 0 to ${x(sw.entered * 2)}; ${x(sw.entered)} entered.`}
            describePoint={(p) => `${x(p.t)}: net to boat ${money(p.v ?? 0)}`}
          >
            <GridY />
            <AxisX title={variable === "catch" ? `Catch, ${unit}` : variable === "price" ? `Price, $/${unit}` : "Fuel price, $/gal"} format={(v) => (variable === "catch" ? Math.round(v).toLocaleString("en-US") : `$${v.toFixed(2)}`)} />
            <AxisY title="Net to boat" />
            <Line data={data} kind="estimated" />
            <Points data={yours} />
            <NowRule t={sw.entered} label="As entered" />
            <BreakEvenMarker at={be ?? undefined} label={be !== null ? `Break-even ${variable === "catch" ? breakEvenCatchText(be, unit) : variable === "price" ? breakEvenPriceText(be, unit) : breakEvenFuelText(be)}` : "Net to boat $0"} />
            <Crosshair />
            <Tip render={(p) => <span>{`${x(p.t)}: ${money(p.v ?? 0)}`}</span>} />
          </Plot>
          <p className={s.small} data-testid="trip-sweep-words">
            {be !== null
              ? variable === "fuelPrice"
                ? `Net to boat falls to $0 once fuel costs more than ${breakEvenFuelText(be)}.`
                : `Net to boat reaches $0 at ${variable === "catch" ? breakEvenCatchText(be, unit) : breakEvenPriceText(be, unit)} (rounded up).`
              : variable === "fuelPrice"
                ? "Net to boat stays above $0 across this range of fuel prices."
                : "Net to boat does not cross $0 in this range."}{" "}
            Each point is your trip worked again with only the {what} changed. It is arithmetic, not a forecast of catch or price.
          </p>
        </>
      ) : (
        <PlotState kind="missing" what={`Net to boat against ${what}`} reason={`Enter a ${what} above 0 to see how net to boat moves with it.`} height={160} />
      )}
    </div>
  );
}
