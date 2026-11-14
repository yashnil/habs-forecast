"use client";

import { motion, useAnimationControls } from "motion/react";
import { useEffect, useMemo, useRef, type KeyboardEvent, type ReactNode } from "react";
import { Button, IconButton, Micro, ScrubNumber, Segmented, cx } from "@/nx/ds";
import { fieldPop } from "@/nx/ds/motion";
import { audienceOf, fuelReference, priceReference, shareReference, type CoastFishery, type CoastReference } from "./fisheries";
import { fieldDomId } from "./pass";
import {
  addCatch,
  addCost,
  MAX_CATCH_ROWS,
  MAX_COST_ROWS,
  money,
  payersMatter,
  read,
  removeCatch,
  removeCost,
  setCatch,
  setCost,
  setField,
  setShareBasis,
  takeReference,
  type CrewKind,
  type FieldId,
  type FuelRoute,
  type Payer,
  type ShareBasis,
  type TripResult,
  type Workspace,
} from "./workspace";
import s from "./trip.module.css";

const CREW_KINDS: Array<{ value: CrewKind; label: string }> = [
  { value: "afterExpenses", label: "After costs" },
  { value: "ofGross", label: "Of gross" },
  { value: "dayRate", label: "Day rate" },
];
/** What each way of paying crew means, said once it is chosen. */
const CREW_WORDS: Record<CrewKind, string> = {
  afterExpenses: "A share of what is left after fees and the shared costs.",
  ofGross: "A share of gross, before any cost comes off.",
  dayRate: "A day rate for each person, whatever the catch.",
};
const CREW_GROUP: Record<CrewKind, string> = { afterExpenses: "Crew, after costs", ofGross: "Crew, of gross", dayRate: "Crew, day rate" };
const PAYERS: Array<{ value: Payer; label: string }> = [
  { value: "shared", label: "Shared" },
  { value: "boat", label: "Boat" },
];
const ROUTES: Array<{ value: FuelRoute; label: string }> = [
  { value: "gallons", label: "Gallons" },
  { value: "distance", label: "Distance" },
];
const BASES: Array<{ value: ShareBasis; label: string }> = [
  { value: "total", label: "Total" },
  { value: "each", label: "Each" },
];

type Edit = (change: (ws: Workspace) => Workspace) => void;
type Props = { ws: Workspace; edit: Edit; result: TripResult; fishery: CoastFishery | null };

/**
 * One blank of the equation (v14 synthesis): the figure in a soft field with its unit beside it; a blank is drawn as a
 * dotted outline (a gap, never 0), the next blank the trip waits for dashed in ultramarine. Arrow keys step it; a value
 * set from elsewhere (a what-if end) pops it once.
 */
function EqField({ id, label, value, onChange, prefix, unit, step, numeric, next, error, pop }: { id: FieldId; label: string; value: string; onChange: (v: string) => void; prefix?: string; unit?: string; step: number; numeric?: boolean; next?: boolean; error?: string; pop?: number }) {
  const controls = useAnimationControls();
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    void controls.start({ scale: [1.06, 1], transition: fieldPop });
  }, [pop, controls]);
  // what was typed is what is shown: a reformatted value under the caret would fight the hand
  const text = value;
  const empty = value.trim() === "";
  const errId = `${fieldDomId(id)}-err`;
  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    e.preventDefault();
    const v = Number(value) || 0;
    const next = Math.max(0, Math.round((v + (e.key === "ArrowUp" ? step : -step)) * 100) / 100);
    onChange(String(next));
  };
  return (
    <motion.label className={s.f} data-empty={empty || undefined} data-next={(empty && next) || undefined} data-error={error ? "" : undefined} animate={controls}>
      {prefix ? <i aria-hidden="true">{prefix}</i> : null}
      <input
        id={fieldDomId(id)}
        aria-label={label}
        inputMode={numeric ? "numeric" : "decimal"}
        autoComplete="off"
        placeholder="—"
        value={text}
        style={{ width: `${Math.max(1.5, (text || "—").length + 0.3)}ch` }}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKey}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? errId : undefined}
      />
      {unit ? <i aria-hidden="true">{unit}</i> : null}
      {error ? (
        <span id={errId} className={s.fErr} role="alert">
          {error}
        </span>
      ) : null}
    </motion.label>
  );
}

const Grp = ({ label, children, className }: { label: string; children: ReactNode; className?: string }) => (
  <div className={cx(s.grp, className)} role="group" aria-label={label}>
    <Micro className={s.grpLabel} aria-hidden="true">
      {label}
    </Micro>
    <span className={s.expr}>{children}</span>
  </div>
);
const Op = ({ c, last }: { c: string; last?: boolean }) => (
  <span className={cx(s.op, last && s.opEq)} aria-hidden="true">
    {c}
  </span>
);
const Times = ({ w = "×" }: { w?: string }) => (
  <em className={s.times} aria-hidden="true">
    {w}
  </em>
);

/**
 * The trip as an equation that runs left to right into the stub (v14 synthesis, from Chartroom): catch × price − fuel
 * − bait and ice − fees − crew =. A recreational trip is fuel + costs ÷ the people sharing them. Every edit goes
 * straight to the workspace; the next blank the trip waits for is marked. Lines beyond the first (another species,
 * another cost) and the fuel worked out from distance live under the pass's fold (`TripMore`).
 */
export function TripEquation({ ws, edit, result, fishery, pops }: Props & { pops: Record<string, number> }) {
  const unit = fishery?.unit ?? "lb";
  const rec = audienceOf(fishery) === "recreational";
  const { n } = useMemo(() => read(ws, fishery), [ws, fishery]);
  const next = result.waiting[0] ?? null;
  const scalar = (id: Parameters<typeof setField>[1]) => (v: string) => edit((w) => setField(w, id, v));
  const err = (id: FieldId) => result.errors[id];
  const f = (id: FieldId, o: Omit<Parameters<typeof EqField>[0], "id" | "next" | "error" | "pop">) => <EqField key={id} id={id} next={next === id} error={err(id)} pop={pops[id]} {...o} />;
  const c0 = ws.catch[0];
  const many = ws.catch.length > 1;
  const cname = c0?.label || "Catch 1";
  const cost = ws.costs[0];
  const costName = cost ? cost.label || "Cost 1" : null;
  const gal = ws.fuelRoute === "gallons";
  const galWords = result.out.gallons.status === "value" ? `${Math.round(result.out.gallons.value * 10) / 10}` : "—";
  const fuel = (
    <Grp label="Fuel">
      {gal ? f("fuelGal", { label: "Fuel", value: ws.fuelGal, onChange: scalar("fuelGal"), unit: "gal", step: 5 }) : <span className={s.derivedGal}>{galWords} gal</span>}
      {n.gallons === 0 ? null : (
        <>
          <Times />
          {f("fuelPrice", { label: "Fuel price", value: ws.fuelPrice, onChange: scalar("fuelPrice"), prefix: "$", unit: "/gal", step: 0.05 })}
        </>
      )}
    </Grp>
  );
  const costs = cost && costName ? (
    <Grp label={costName === "Bait and ice" ? "Bait, ice" : costName}>{f("cost.0.usd", { label: costName, value: cost.usd, onChange: (v) => edit((w) => setCost(w, 0, { usd: v })), prefix: "$", step: 10 })}</Grp>
  ) : null;

  if (rec)
    return (
      <div className={s.eq}>
        {fuel}
        {costs ? (
          <>
            <Op c="+" />
            {costs}
          </>
        ) : null}
        <Op c="÷" />
        <Grp label="People sharing">{f("anglers", { label: "People sharing the cost", value: ws.anglers, onChange: scalar("anglers"), step: 1, numeric: true })}</Grp>
        <Op c="=" last />
      </div>
    );

  const crewKind = ws.crewKind;
  const crewLabel = n.crew === 0 ? "Crew, none paid" : crewKind ? CREW_GROUP[crewKind] : "Crew";
  return (
    <div className={s.eq}>
      <Grp label={many ? `${cname}, and ${ws.catch.length - 1} more` : "Catch"}>
        {f("catch.0.amount", { label: many ? `${cname} catch` : "Catch", value: c0?.amount ?? "", onChange: (v) => edit((w) => setCatch(w, 0, { amount: v })), unit, step: unit === "ton" ? 1 : 10 })}
        <Times />
        {f("catch.0.price", { label: many ? `${cname} price` : "Price", value: c0?.price ?? "", onChange: (v) => edit((w) => setCatch(w, 0, { price: v })), prefix: "$", unit: `/${unit}`, step: unit === "ton" ? 10 : 0.05 })}
      </Grp>
      <Op c="−" />
      {fuel}
      {costs ? (
        <>
          <Op c="−" />
          {costs}
        </>
      ) : null}
      <Op c="−" />
      <Grp label="Fees, of gross">{f("feesPct", { label: "Fees", value: ws.feesPct, onChange: scalar("feesPct"), unit: "%", step: 0.5 })}</Grp>
      <Op c="−" />
      <Grp label={crewLabel}>
        {f("crew", { label: "Crew", value: ws.crew, onChange: scalar("crew"), step: 1, numeric: true })}
        {n.crew === 0 ? null : crewKind === "dayRate" ? (
          <>
            <Times w="at" />
            {f("dayRate", { label: "Day rate", value: ws.dayRate, onChange: scalar("dayRate"), prefix: "$", unit: "/day", step: 25 })}
            <Times />
            {f("days", { label: "Days at sea", value: ws.days, onChange: scalar("days"), unit: "days", step: 1, numeric: true })}
          </>
        ) : (
          <>
            <Times w="at" />
            {f("share", { label: "Crew share", value: ws.share, onChange: scalar("share"), unit: ws.shareBasis === "each" && (n.crew ?? 0) > 1 ? "% each" : "%", step: 1 })}
          </>
        )}
      </Grp>
      <Op c="=" last />
    </div>
  );
}

/**
 * The equation's terms (under it, only where they move the money): how crew is paid, whether a share is the crew's
 * total or each person's, and who pays fuel and each cost when crew takes a share after costs. Nothing is chosen for
 * the reader: an unset choice says "Choose one".
 */
export function TripTerms({ ws, edit, fishery, fold }: Omit<Props, "result"> & { fold?: ReactNode }) {
  const { n } = useMemo(() => read(ws, fishery), [ws, fishery]);
  const rec = audienceOf(fishery) === "recreational";
  const payers = payersMatter(ws, n, "commercial");
  const paid = n.crew !== null && n.crew > 0;
  const shareKind = paid && (ws.crewKind === "afterExpenses" || ws.crewKind === "ofGross");
  const items: ReactNode[] = [];
  if (!rec && n.crew !== 0) items.push(<Segmented key="kind" size="s" label="How crew is paid" options={CREW_KINDS} value={ws.crewKind} onChange={(v) => edit((w) => ({ ...w, crewKind: v }))} className={s.term} />);
  if (!rec && shareKind && n.crew! > 1) items.push(<Segmented key="basis" size="s" label="Crew share is" options={BASES} value={ws.shareBasis} onChange={(v) => edit((w) => setShareBasis(w, fishery, v))} className={s.term} />);
  if (!rec && payers && n.gallons !== 0) items.push(<Segmented key="fuel" size="s" label="Who pays fuel" options={PAYERS} value={ws.fuelPayer} onChange={(v) => edit((w) => ({ ...w, fuelPayer: v }))} className={s.term} />);
  if (!rec && payers)
    ws.costs.forEach((c, i) => {
      if (n.costs[i] === 0) return;
      const name = c.label || `Cost ${i + 1}`;
      items.push(<Segmented key={`cost-${i}`} size="s" label={`Who pays ${name.toLowerCase()}`} options={PAYERS} value={c.payer} onChange={(v) => edit((w) => setCost(w, i, { payer: v }))} className={s.term} />);
    });
  if (!items.length && !fold) return null;
  return (
    <div className={s.terms} role="group" aria-label="Terms of the trip">
      {items}
      {fold}
    </div>
  );
}

/**
 * Under the pass's fold: the rest of the trip's inputs. Fuel from distance and burn, more species, more costs, a fee
 * per pound, the sourced references a reader may take (T47: they enter only on a tap, then stay labelled), and what
 * the chosen terms mean.
 */
export function TripMore({ ws, edit, result, fishery, now }: Props & { now: Date | null }) {
  const unit = fishery?.unit ?? "lb";
  const rec = audienceOf(fishery) === "recreational";
  const refs = useMemo(() => ({ price: priceReference(fishery), fuelPrice: fuelReference(now), share: shareReference(fishery) }), [fishery, now]);
  const { n } = useMemo(() => read(ws, fishery), [ws, fishery]);
  const payers = payersMatter(ws, n, audienceOf(fishery));
  const paid = n.crew !== null && n.crew > 0;
  const note = (id: FieldId, hint?: ReactNode): { error?: string; hint?: ReactNode } => ({
    error: result.errors[id],
    hint: result.clamped.includes(id) ? "Taken as the most this worksheet accepts." : hint,
  });
  const offer = (id: "price" | "fuelPrice" | "share", ref: CoastReference | null) =>
    ref && !ws.taken[id] ? (
      <Button variant="quiet" className={s.offer} onClick={() => edit((w) => takeReference(w, id, ref.value))}>
        Use {ref.words} ({ref.source})
      </Button>
    ) : null;
  const taken = (id: "price" | "fuelPrice" | "share", ref: CoastReference | null) => (ws.taken[id] && ref ? <p className={s.derived}>Your entry, from {ref.taken}.</p> : null);
  const scalar = (id: Parameters<typeof setField>[1]) => (v: string) => edit((w) => setField(w, id, v));
  const many = ws.catch.length > 1;

  return (
    <section className={s.more} aria-labelledby="trip-more">
      <h3 id="trip-more" className={s.h3}>
        More about this trip
      </h3>
      <div className={s.moreGrid}>
        <div className={s.moreCol}>
          <Segmented label="Fuel entered as" options={ROUTES} value={ws.fuelRoute} onChange={(v) => edit((w) => ({ ...w, fuelRoute: v }))} />
          {ws.fuelRoute === "distance" ? (
            <>
              <ScrubNumber label="Round trip" value={ws.roundTripNm} onChange={scalar("roundTripNm")} unit="nm" step={1} {...note("roundTripNm", "0 if you do not steam.")} />
              {n.roundTripNm === 0 ? null : (
                <>
                  <ScrubNumber label="Cruising speed" value={ws.cruiseKt} onChange={scalar("cruiseKt")} unit="kt" step={0.5} {...note("cruiseKt")} />
                  <ScrubNumber label="Steaming burn" value={ws.steamGph} onChange={scalar("steamGph")} unit="gal/h" step={0.5} {...note("steamGph")} />
                </>
              )}
              <ScrubNumber label="Hours fishing" value={ws.fishingHours} onChange={scalar("fishingHours")} unit="h" step={0.5} {...note("fishingHours", "0 if the engine is off while you fish.")} />
              {n.fishingHours === 0 ? null : <ScrubNumber label="Fishing burn" value={ws.fishingGph} onChange={scalar("fishingGph")} unit="gal/h" step={0.5} {...note("fishingGph")} />}
            </>
          ) : null}
          <p className={s.derived} data-testid="trip-gallons">
            {result.out.gallons.status === "value" ? `${Math.round(result.out.gallons.value * 10) / 10} gal for the trip.` : "Gallons are worked out once these are filled."}
          </p>
          {n.gallons === 0 ? null : (
            <>
              {offer("fuelPrice", refs.fuelPrice)}
              {taken("fuelPrice", refs.fuelPrice)}
            </>
          )}
          {!rec ? (
            <>
              {offer("price", refs.price)}
              {taken("price", refs.price)}
              {!refs.price ? <p className={s.derived}>No sourced average price for this species: ask your buyer.</p> : null}
            </>
          ) : null}
        </div>

        <div className={s.moreCol}>
          {rec ? null : (
            <>
              {ws.catch.map((c, i) => {
                if (!many) return null;
                const name = c.label || `Catch ${i + 1}`;
                return (
                  <div key={i} className={s.line} data-testid={`trip-catch-${i}`}>
                    <div className={s.lineHead}>
                      <input className={s.nameInput} value={c.label} onChange={(e) => edit((w) => setCatch(w, i, { label: e.target.value.slice(0, 60) }))} aria-label={`Name of catch line ${i + 1}`} placeholder={`Species ${i + 1}`} />
                      {i > 0 ? <IconButton icon="close" label={`Remove ${name}`} size={44} onClick={() => edit((w) => removeCatch(w, i))} /> : null}
                    </div>
                    {i > 0 ? (
                      <>
                        <ScrubNumber label={`${name} catch`} value={c.amount} onChange={(v) => edit((w) => setCatch(w, i, { amount: v }))} unit={unit} step={unit === "ton" ? 1 : 10} decimals={unit === "ton" ? 1 : undefined} size={7} {...note(`catch.${i}.amount`)} />
                        <ScrubNumber label={`${name} price`} value={c.price} onChange={(v) => edit((w) => setCatch(w, i, { price: v }))} prefix="$" unit={`/${unit}`} step={unit === "ton" ? 10 : 0.05} decimals={unit === "ton" ? 0 : 2} {...note(`catch.${i}.price`)} />
                      </>
                    ) : null}
                  </div>
                );
              })}
              {ws.catch.length < MAX_CATCH_ROWS ? (
                <Button variant="quiet" icon="plus" className={s.compact} onClick={() => edit(addCatch)}>
                  Add a species
                </Button>
              ) : null}
            </>
          )}
          {ws.costs.map((c, i) => {
            if (i === 0 && ws.costs.length === 1) return null;
            const name = c.label || `Cost ${i + 1}`;
            return (
              <div key={i} className={s.line} data-testid={`trip-cost-${i}`}>
                <div className={s.lineHead}>
                  <input className={s.nameInput} value={c.label} onChange={(e) => edit((w) => setCost(w, i, { label: e.target.value.slice(0, 60) }))} aria-label={`Name of cost line ${i + 1}`} placeholder={`Cost ${i + 1}`} />
                  <IconButton icon="close" label={`Remove ${name}`} size={44} onClick={() => edit((w) => removeCost(w, i))} />
                </div>
                {i > 0 ? <ScrubNumber label={name} value={c.usd} onChange={(v) => edit((w) => setCost(w, i, { usd: v }))} prefix="$" step={10} {...note(`cost.${i}.usd`)} /> : null}
              </div>
            );
          })}
          {ws.costs.length < MAX_COST_ROWS ? (
            <Button variant="quiet" icon="plus" className={s.compact} onClick={() => edit(addCost)}>
              Add a cost
            </Button>
          ) : null}
          {rec ? null : ws.feesPerUnit === null ? (
            <Button variant="quiet" icon="plus" className={s.compact} onClick={() => edit((w) => ({ ...w, feesPerUnit: "" }))}>
              Add a fee per {unit === "ton" ? "ton" : "pound"}
            </Button>
          ) : (
            <div className={s.lineHead}>
              <ScrubNumber label={`Fee per ${unit === "ton" ? "ton" : "pound"}`} value={ws.feesPerUnit} onChange={(v) => edit((w) => ({ ...w, feesPerUnit: v }))} prefix="$" unit={`/${unit}`} step={unit === "ton" ? 1 : 0.01} decimals={2} {...note("feesPerUnit")} />
              <IconButton icon="close" label={`Remove the fee per ${unit === "ton" ? "ton" : "pound"}`} size={44} onClick={() => edit((w) => ({ ...w, feesPerUnit: null }))} />
            </div>
          )}
        </div>

        {rec ? null : (
          <div className={s.moreCol}>
            {paid && ws.crewKind ? (
              <p className={s.derived} data-testid="trip-crew-words">
                {CREW_WORDS[ws.crewKind]}
              </p>
            ) : null}
            {payers ? (
              <p className={s.derived} data-testid="trip-payer-words">
                Shared: taken off the top, before the crew&apos;s share is worked out. Boat: paid from the boat&apos;s own share.
              </p>
            ) : null}
            {result.share ? (
              <p className={s.derived} data-testid="trip-share-note">
                {result.share.words}
                {result.share.eachUsd !== null && result.share.crew > 1 ? ` ${money(result.share.eachUsd)} each.` : ""}
              </p>
            ) : null}
            {paid && ws.crewKind === "afterExpenses" && ws.shareBasis === "total" ? (
              <>
                {offer("share", refs.share)}
                {taken("share", refs.share)}
              </>
            ) : null}
            <p className={s.derived}>Crew counts the people paid, not you; 0 if none are paid. Fees are landing and buyer fees; 0 if none.</p>
          </div>
        )}
      </div>
    </section>
  );
}
