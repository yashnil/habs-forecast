"use client";

import { useMemo, type ReactNode } from "react";
import { Button, IconButton, ScrubNumber, Segmented } from "@/nx/ds";
import { audienceOf, fuelReference, priceReference, shareReference, type CoastFishery, type CoastReference } from "./fisheries";
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
  { value: "afterExpenses", label: "Share after costs" },
  { value: "ofGross", label: "Share of gross" },
  { value: "dayRate", label: "Day rate" },
];
const PAYERS: Array<{ value: Payer; label: string }> = [
  { value: "shared", label: "Shared, off the top" },
  { value: "boat", label: "The boat" },
];
const ROUTES: Array<{ value: FuelRoute; label: string }> = [
  { value: "gallons", label: "Gallons" },
  { value: "distance", label: "Distance and burn" },
];
const BASES: Array<{ value: ShareBasis; label: string }> = [
  { value: "total", label: "Crew's total" },
  { value: "each", label: "Each person" },
];

/**
 * The trip's inputs, in three compact groups: catch, trip costs, crew (a recreational trip: costs and the
 * people sharing them). Nothing is filled in. A control appears only where it moves the money: who pays a cost
 * is asked only when crew is paid a share after costs, a fuel price only when fuel is burned, the total/each
 * choice only for a share. Every edit goes straight to the workspace, so the results beside it follow the hand.
 */
export function TripInputs({ ws, edit, result, fishery, now }: { ws: Workspace; edit: (change: (ws: Workspace) => Workspace) => void; result: TripResult; fishery: CoastFishery | null; now: Date | null }) {
  const unit = fishery?.unit ?? "lb";
  const recreational = audienceOf(fishery) === "recreational";
  const refs = useMemo(() => ({ price: priceReference(fishery), fuelPrice: fuelReference(now), share: shareReference(fishery) }), [fishery, now]);
  const { n } = useMemo(() => read(ws, fishery), [ws, fishery]);
  const payers = payersMatter(ws, n, audienceOf(fishery));
  const paid = n.crew !== null && n.crew > 0;
  const shareKind = paid && (ws.crewKind === "afterExpenses" || ws.crewKind === "ofGross");

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
  const taken = (id: "price" | "fuelPrice" | "share", ref: CoastReference | null) => (ws.taken[id] && ref ? `Your entry, from ${ref.taken}.` : undefined);
  const scalar = (id: Parameters<typeof setField>[1]) => (v: string) => edit((w) => setField(w, id, v));

  return (
    <div className={s.inputs} data-testid="trip-inputs">
      {recreational ? null : (
        <section className={s.group} aria-labelledby="trip-in-catch">
          <h3 id="trip-in-catch" className={s.groupHead}>
            Catch and price
          </h3>
          {ws.catch.map((c, i) => {
            const many = ws.catch.length > 1;
            const name = c.label || `Catch ${i + 1}`;
            return (
              <div key={i} className={s.line} data-testid={`trip-catch-${i}`}>
                {many ? (
                  <div className={s.lineHead}>
                    <input className={s.nameInput} value={c.label} onChange={(e) => edit((w) => setCatch(w, i, { label: e.target.value.slice(0, 60) }))} aria-label={`Name of catch line ${i + 1}`} placeholder={`Species ${i + 1}`} />
                    {i > 0 ? <IconButton icon="close" label={`Remove ${name}`} size={44} onClick={() => edit((w) => removeCatch(w, i))} /> : null}
                  </div>
                ) : null}
                <ScrubNumber label={many ? `${name} catch` : "Catch"} value={c.amount} onChange={(v) => edit((w) => setCatch(w, i, { amount: v }))} unit={unit} step={unit === "ton" ? 1 : 10} decimals={unit === "ton" ? 1 : undefined} size={7} {...note(`catch.${i}.amount`)} />
                <ScrubNumber
                  label={many ? `${name} price` : "Price"}
                  value={c.price}
                  onChange={(v) => edit((w) => setCatch(w, i, { price: v }))}
                  prefix="$"
                  unit={`/${unit}`}
                  step={unit === "ton" ? 10 : 0.05}
                  decimals={unit === "ton" ? 0 : 2}
                  {...note(`catch.${i}.price`, i === 0 ? (taken("price", refs.price) ?? (refs.price ? undefined : "No sourced average for this species: ask your buyer.")) : undefined)}
                />
                {i === 0 ? offer("price", refs.price) : null}
              </div>
            );
          })}
          {ws.catch.length < MAX_CATCH_ROWS ? (
            <Button variant="quiet" icon="plus" onClick={() => edit(addCatch)}>
              Add a species
            </Button>
          ) : null}
        </section>
      )}

      <section className={s.group} aria-labelledby="trip-in-costs">
        <h3 id="trip-in-costs" className={s.groupHead}>
          Trip costs
        </h3>
        <Segmented label="Fuel entered as" options={ROUTES} value={ws.fuelRoute} onChange={(v) => edit((w) => ({ ...w, fuelRoute: v }))} fit="equal" />
        {ws.fuelRoute === "gallons" ? (
          <ScrubNumber label="Fuel" value={ws.fuelGal} onChange={scalar("fuelGal")} unit="gal" step={5} {...note("fuelGal")} />
        ) : (
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
            <p className={s.derived} data-testid="trip-gallons">
              {result.out.gallons.status === "value" ? `${Math.round(result.out.gallons.value * 10) / 10} gal for the trip.` : "Gallons are worked out once these are filled."}
            </p>
          </>
        )}
        {n.gallons === 0 ? null : (
          <>
            <ScrubNumber label="Fuel price" value={ws.fuelPrice} onChange={scalar("fuelPrice")} prefix="$" unit="/gal" step={0.05} decimals={2} {...note("fuelPrice", taken("fuelPrice", refs.fuelPrice))} />
            {offer("fuelPrice", refs.fuelPrice)}
          </>
        )}
        {payers && n.gallons !== 0 ? <Segmented label="Who pays fuel" options={PAYERS} value={ws.fuelPayer} onChange={(v) => edit((w) => ({ ...w, fuelPayer: v }))} fit="equal" /> : null}

        {ws.costs.map((c, i) => {
          const name = c.label || `Cost ${i + 1}`;
          return (
            <div key={i} className={s.line} data-testid={`trip-cost-${i}`}>
              {i > 0 || ws.costs.length > 1 ? (
                <div className={s.lineHead}>
                  <input className={s.nameInput} value={c.label} onChange={(e) => edit((w) => setCost(w, i, { label: e.target.value.slice(0, 60) }))} aria-label={`Name of cost line ${i + 1}`} placeholder={`Cost ${i + 1}`} />
                  <IconButton icon="close" label={`Remove ${name}`} size={44} onClick={() => edit((w) => removeCost(w, i))} />
                </div>
              ) : null}
              <ScrubNumber label={name} value={c.usd} onChange={(v) => edit((w) => setCost(w, i, { usd: v }))} prefix="$" step={10} {...note(`cost.${i}.usd`)} />
              {payers && n.costs[i] !== 0 ? <Segmented label={`Who pays ${name.toLowerCase()}`} options={PAYERS} value={c.payer} onChange={(v) => edit((w) => setCost(w, i, { payer: v }))} fit="equal" /> : null}
            </div>
          );
        })}
        {ws.costs.length < MAX_COST_ROWS ? (
          <Button variant="quiet" icon="plus" onClick={() => edit(addCost)}>
            Add a cost
          </Button>
        ) : null}

        {recreational ? null : (
          <>
            <ScrubNumber label="Fees" value={ws.feesPct} onChange={scalar("feesPct")} unit="% of gross" step={0.5} decimals={1} size={4} {...note("feesPct", "Landing fees and buyer deductions; 0 if none.")} />
            {ws.feesPerUnit === null ? (
              <Button variant="quiet" icon="plus" onClick={() => edit((w) => ({ ...w, feesPerUnit: "" }))}>
                Add a fee per {unit === "ton" ? "ton" : "pound"}
              </Button>
            ) : (
              <div className={s.lineHead}>
                <ScrubNumber label={`Fee per ${unit === "ton" ? "ton" : "pound"}`} value={ws.feesPerUnit} onChange={(v) => edit((w) => ({ ...w, feesPerUnit: v }))} prefix="$" unit={`/${unit}`} step={unit === "ton" ? 1 : 0.01} decimals={2} {...note("feesPerUnit")} />
                <IconButton icon="close" label={`Remove the fee per ${unit === "ton" ? "ton" : "pound"}`} size={44} onClick={() => edit((w) => ({ ...w, feesPerUnit: null }))} />
              </div>
            )}
          </>
        )}
      </section>

      {recreational ? (
        <section className={s.group} aria-labelledby="trip-in-people">
          <h3 id="trip-in-people" className={s.groupHead}>
            People
          </h3>
          <ScrubNumber label="People sharing the cost" value={ws.anglers} onChange={scalar("anglers")} unit="people" step={1} size={3} {...note("anglers", "Counting you.")} />
        </section>
      ) : (
        <section className={s.group} aria-labelledby="trip-in-crew">
          <h3 id="trip-in-crew" className={s.groupHead}>
            Crew pay
          </h3>
          <ScrubNumber label="Crew" value={ws.crew} onChange={scalar("crew")} unit="people" step={1} size={3} {...note("crew", "Not counting you; 0 if nobody else is paid.")} />
          {n.crew === 0 ? null : <Segmented label="How crew is paid" options={CREW_KINDS} value={ws.crewKind} onChange={(v) => edit((w) => ({ ...w, crewKind: v }))} fit="equal" />}
          {shareKind ? (
            <>
              {n.crew! > 1 ? <Segmented label="Crew share is" options={BASES} value={ws.shareBasis} onChange={(v) => edit((w) => setShareBasis(w, fishery, v))} fit="equal" /> : null}
              <ScrubNumber
                label="Crew share"
                value={ws.share}
                onChange={scalar("share")}
                unit={`%${ws.shareBasis === "each" && n.crew! > 1 ? " each" : ""}${ws.crewKind === "ofGross" ? " of gross" : " after costs"}`}
                step={1}
                decimals={1}
                size={4}
                {...note("share", taken("share", refs.share))}
              />
              {ws.crewKind === "afterExpenses" && ws.shareBasis === "total" ? offer("share", refs.share) : null}
              {result.share ? (
                <p className={s.derived} data-testid="trip-share-note">
                  {result.share.words}
                  {result.share.eachUsd !== null && result.share.crew > 1 ? ` ${money(result.share.eachUsd)} each.` : ""}
                </p>
              ) : null}
            </>
          ) : null}
          {paid && ws.crewKind === "dayRate" ? (
            <>
              <ScrubNumber label="Day rate" value={ws.dayRate} onChange={scalar("dayRate")} prefix="$" unit="a day each" step={25} {...note("dayRate")} />
              <ScrubNumber label="Days at sea" value={ws.days} onChange={scalar("days")} unit="days" step={1} size={3} {...note("days")} />
            </>
          ) : null}
        </section>
      )}
    </div>
  );
}
