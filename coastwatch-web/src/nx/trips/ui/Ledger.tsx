"use client";

import { useCallback, useMemo, useState } from "react";
import { dieselSnapshot, priceRef, priceSnapshot, type RefSnapshot } from "@/content/referenceValues";
import type { Draft } from "@/lib/tripEconomicsStore";
import { Button, Dialog, Micro, RollingNumber, day } from "@/nx/ds";
import { tripFishery, type CoastFishery } from "@/nx/coast/fisheries";
import { breakEvenGross, passSteps } from "@/nx/coast/pass";
import { TripWaterfall } from "@/nx/coast/TripFigures";
import { TripEquation, TripMore, TripTerms } from "@/nx/coast/TripInputs";
import { breakEvenCatchText, breakEvenPriceText, emptyWorkspace, evaluateWorkspace, fromDraft, hasEntry, money, NET_IS_NOT, read, toDraft, waitingWords, type Step, type TripResult, type Workspace } from "@/nx/coast/workspace";
import type { Trip } from "../types";
import s from "./economics.module.css";

const num = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 2 });

/** The references a worksheet's taken values came from, dated (what the coast planner keeps with a saved trip). */
export function refsFor(ws: Workspace, f: CoastFishery | null): RefSnapshot[] {
  const out: RefSnapshot[] = [];
  if (ws.taken.fuelPrice) out.push(dieselSnapshot());
  const pr = priceRef(f?.priceKey);
  if (ws.taken.price && pr) out.push(priceSnapshot(pr));
  return out;
}

/** Each figure's own words for its waterfall column ("400 lb × $9.00"), as the planner writes them. */
function basisFor(ws: Workspace, f: CoastFishery | null, result: TripResult) {
  const { n } = read(ws, f);
  const unit = f?.unit ?? "lb";
  return (st: Step): string => {
    const c = n.catch[0];
    if (st.id === "gross") return ws.catch.length === 1 && c?.amount != null && c.price != null ? `${num(c.amount)} ${unit} × ${money(c.price)}` : "catch × price";
    if (st.id === "fuel") return result.out.gallons.status === "value" && n.fuelPrice != null ? `${num(result.out.gallons.value)} gal × ${money(n.fuelPrice)}` : "gal × $/gal";
    if (st.id === "fees") return n.feesPct != null ? `${num(n.feesPct)}% of gross` : "% of gross";
    if (st.id === "crew") return n.crew === 0 ? "none paid" : ws.crewKind === "dayRate" ? "day rate" : n.totalPct != null ? `${num(n.totalPct)}%` : "share";
    return "";
  };
}

/** Where the reader's figures came from: their own, or a dated reference they took into the worksheet. */
function Provenance({ trip, ws }: { trip: Trip; ws: Workspace }) {
  const refs = trip.refs;
  const own = Object.keys(ws.taken).length === 0;
  return (
    <ul className={s.prov} aria-label="Where these figures come from" data-testid="ledger-provenance">
      <li>
        <i className={s.provOwn} aria-hidden="true" />
        {own ? "Every figure is your own" : "Your own figures, except"}
      </li>
      {refs.map((r) => (
        <li key={r.key} title={r.source}>
          <i className={s.provRef} aria-hidden="true" />
          {r.key.startsWith("eia:") ? "Fuel price" : "Price"} from {r.key.startsWith("eia:") ? "EIA" : "CDFW"}, {r.period}
          {r.fetchedAt ? <span className={s.provWhen}> · read {day(r.fetchedAt, { year: true })}</span> : null}
        </li>
      ))}
    </ul>
  );
}

export type LedgerProps = {
  trip: Trip;
  /** defaults to the trip's own fishery */
  fishery?: CoastFishery | null;
  /** the worksheet saved as the trip's updated plan (the store keeps the previous economics in the trip's history) */
  onSave: (economics: Draft, refs: RefSnapshot[]) => void;
  /** a clock for the references' age (null: not shown) */
  now?: Date | null;
  /** read-only: a newer version wrote the store, or the trip is completed */
  readOnly?: boolean;
};

/**
 * The trip's economics in the Trip Workspace (v15): net to boat as the one figure (a recreational trip: what it costs),
 * the planner's waterfall from gross to net, the break-even catch and price, and where each figure came from. "Open the
 * worksheet" opens the full interactive worksheet (the coast planner's own equation, terms and inputs) in a sheet; its
 * Save hands the new economics to `onSave` as the updated plan. Every figure is the engine's (evaluateWorkspace).
 */
export function Ledger({ trip, fishery: given, onSave, now = null, readOnly }: LedgerProps) {
  const fishery = given ?? tripFishery(trip.fishery);
  const saved = useMemo(() => (trip.economics ? fromDraft(trip.economics, trip.fishery) : emptyWorkspace(fishery)), [trip.economics, trip.fishery, fishery]);
  const result = useMemo(() => evaluateWorkspace(saved, fishery), [saved, fishery]);
  const [open, setOpen] = useState(false);
  const rec = result.audience === "recreational";
  const main = rec ? result.out.operating : result.out.net;
  const label = rec ? "Trip cost" : "Net to boat";
  const steps = useMemo(() => passSteps(result.steps), [result.steps]);
  const be = useMemo(() => breakEvenGross(saved, fishery, result), [saved, fishery, result]);
  const unit = fishery?.unit ?? "lb";
  const beC = result.out.breakEvenCatch;
  const beP = result.out.breakEvenPrice;
  const modelled = !!trip.economics && hasEntry(saved);

  return (
    <section className={s.ledger} aria-label={`Economics: ${label.toLowerCase()}, hypothetical, from your numbers`} data-testid="ledger">
      <div className={s.ledgerHead}>
        <Micro glyph="hypothetical">Hypothetical · {rec ? "trip cost" : "net to boat"}</Micro>
        {modelled ? (
          <p className={s.figure} data-testid="ledger-net">
            <span className={s.vh}>{label}: </span>
            {main.status === "value" ? (
              <>
                <span className={s.vh}>{money(main.value)}</span>
                <span aria-hidden="true" className={s.figureNum} data-loss={(!rec && main.value < 0) || undefined}>
                  <RollingNumber value={main.value} format={money} cents />
                </span>
              </>
            ) : (
              <span className={s.figureSlot}>{main.status === "waiting" ? waitingWords(main, saved, fishery) : main.status === "none" ? main.reason : "does not apply"}</span>
            )}
          </p>
        ) : (
          <p className={s.empty}>No costs or catch modelled for this trip yet.</p>
        )}
        {modelled && !rec ? (
          <dl className={s.be} data-testid="ledger-breakeven">
            <div>
              <dt>Break-even catch</dt>
              <dd>{beC.status === "value" ? breakEvenCatchText(beC.value, unit) : "—"}</dd>
            </div>
            <div>
              <dt>or price</dt>
              <dd>{beP.status === "value" ? breakEvenPriceText(beP.value, unit) : "—"}</dd>
            </div>
          </dl>
        ) : null}
      </div>
      {modelled ? (
        <div className={s.wfBox}>
          <TripWaterfall steps={steps} basis={basisFor(saved, fishery, result)} rec={rec} breakEven={be} label={rec ? "Fuel and costs up to the trip cost, hypothetical, from your numbers" : "Gross to net to boat, hypothetical, from your numbers"} />
        </div>
      ) : null}
      <div className={s.ledgerFoot}>
        {modelled ? <Provenance trip={trip} ws={saved} /> : null}
        {readOnly ? null : (
          <Button variant="secondary" glyph="→" onClick={() => setOpen(true)} data-testid="ledger-open">
            {modelled ? "Open the worksheet" : "Model the costs"}
          </Button>
        )}
      </div>
      {modelled && !rec ? <p className={s.note}>{NET_IS_NOT}</p> : null}
      {open ? <Worksheet trip={trip} fishery={fishery} start={saved} now={now} onClose={() => setOpen(false)} onSave={(ws) => (onSave(toDraft(ws, fishery), refsFor(ws, fishery)), setOpen(false))} /> : null}
    </section>
  );
}

/** The full worksheet in a sheet: the planner's own equation, terms and inputs, the result live, Save as the updated plan. */
function Worksheet({ trip, fishery, start, now, onClose, onSave }: { trip: Trip; fishery: CoastFishery | null; start: Workspace; now: Date | null; onClose: () => void; onSave: (ws: Workspace) => void }) {
  const [ws, setWs] = useState(start);
  const edit = useCallback((change: (w: Workspace) => Workspace) => setWs((w) => change(w)), []);
  const result = useMemo(() => evaluateWorkspace(ws, fishery), [ws, fishery]);
  const rec = result.audience === "recreational";
  const main = rec ? result.out.operating : result.out.net;
  const steps = useMemo(() => passSteps(result.steps), [result.steps]);
  const be = useMemo(() => breakEvenGross(ws, fishery, result), [ws, fishery, result]);
  const changed = JSON.stringify(toDraft(ws, fishery)) !== JSON.stringify(toDraft(start, fishery));
  return (
    <Dialog
      open
      onOpenChange={(o) => (o ? null : onClose())}
      placement="side"
      title={trip.name}
      description={rec ? "What the trip costs, from your numbers. A recreational catch is not sold." : "Your plan's numbers. A blank is a gap, never 0."}
      className={s.sheet}
      footer={
        <div className={s.sheetFoot}>
          <p className={s.sheetFig} aria-live="polite" data-testid="worksheet-net">
            <span className={s.sheetFigLabel}>{rec ? "Trip cost" : "Net to boat"}</span> <strong>{main.status === "value" ? money(main.value) : "waiting"}</strong>
          </p>
          <Button variant="quiet" onClick={onClose}>
            Discard
          </Button>
          <Button variant="primary" glyph="↓" disabled={!changed} onClick={() => onSave(ws)} data-testid="worksheet-save">
            Save as updated plan
          </Button>
        </div>
      }
    >
      <div className={s.sheetBody} data-testid="worksheet">
        <div className={s.sheetWf}>
          <TripWaterfall steps={steps} basis={basisFor(ws, fishery, result)} rec={rec} breakEven={be} label={rec ? "Fuel and costs up to the trip cost, hypothetical" : "Gross to net to boat, hypothetical"} />
        </div>
        <TripEquation ws={ws} edit={edit} result={result} fishery={fishery} pops={{}} />
        <TripTerms ws={ws} edit={edit} fishery={fishery} />
        <TripMore ws={ws} edit={edit} result={result} fishery={fishery} now={now} />
      </div>
    </Dialog>
  );
}
