"use client";

import { motion, useReducedMotion } from "motion/react";
import { useMemo } from "react";
import { Micro, RollingNumber, settle } from "@/nx/ds";
import { money } from "@/nx/coast/workspace";
import { planVsActual, signedMoney, type Fig, type Part, type PlanVsActual } from "../outcome";
import type { Trip } from "../types";
import s from "./economics.module.css";

/**
 * Planned against actual (v15), drawn: the difference as the one figure; the planned and the actual result as two bars
 * on one axis; then where the difference came from, one bar per part on either side of a centre line (raised the
 * result / lowered it), each with its amount. A part the reader did not record is a dotted slot that says so, never a
 * zero bar. A recreational trip compares the trip cost, fuel and supplies only. Never "profit".
 */
export function PlannedVsActual({ trip, model }: { trip: Trip; /** precomputed (planVsActual(trip)) */ model?: PlanVsActual }) {
  const pv = useMemo(() => model ?? planVsActual(trip), [model, trip]);
  const reduce = useReducedMotion();
  const t = reduce ? { duration: 0 } : settle;
  const rec = pv.measure === "tripCost";
  const what = rec ? "trip cost" : "net to boat";

  if (!pv.hasOutcome) return null;
  const p = pv.planned.result.value;
  const a = pv.actual.result.value;
  const top = Math.max(1, Math.abs(p ?? 0), Math.abs(a ?? 0));
  const effects = pv.parts.map((x) => Math.abs(x.effect ?? 0));
  const span = Math.max(1, ...effects);
  /** for the reader, + is good news for net to boat and bad news for a trip cost */
  const tone = (v: number) => (Math.abs(v) < 0.005 ? "even" : (v > 0) !== rec ? "up" : "down");

  const bar = (label: string, f: Fig, kind: "plan" | "actual") => (
    <div className={s.pvRow} data-testid={`pv-${kind}`}>
      <span className={s.pvLabel}>{label}</span>
      <span className={s.pvTrack}>
        {f.value === null ? (
          <span className={s.pvSlot}>{f.why}</span>
        ) : (
          <motion.span className={s.pvBar} data-kind={kind} data-loss={(!rec && f.value < 0) || undefined} initial={{ width: 0 }} animate={{ width: `${(Math.abs(f.value) / top) * 100}%` }} transition={t} />
        )}
      </span>
      <span className={s.pvVal}>{f.value === null ? "—" : money(f.value)}</span>
    </div>
  );

  const part = (x: Part) => {
    const v = x.effect;
    return (
      <li key={x.id} className={s.attrRow} data-testid={`attr-${x.id}`}>
        <span className={s.attrLabel}>{x.label}</span>
        <span className={s.attrTrack}>
          <span className={s.attrAxis} aria-hidden="true" />
          {v === null ? (
            <span className={s.attrSlot}>{x.why}</span>
          ) : (
            <motion.span
              className={s.attrBar}
              data-tone={tone(v)}
              data-side={v >= 0 ? "right" : "left"}
              initial={{ width: 0 }}
              animate={{ width: `${(Math.abs(v) / span) * 50}%` }}
              transition={t}
              aria-hidden="true"
            />
          )}
        </span>
        <span className={s.attrVal} data-tone={v === null ? undefined : tone(v)}>
          {v === null ? "—" : signedMoney(v)}
        </span>
      </li>
    );
  };

  return (
    <section className={s.pv} aria-label={`Planned against actual, ${what}`} data-testid="planned-vs-actual">
      <Micro>Actual against plan · {what}</Micro>
      <p className={s.pvFigure} data-testid="pv-difference">
        {pv.difference === null ? (
          <span className={s.figureSlot}>{pv.hasPlan ? "Not every figure is recorded yet" : "No plan to compare with"}</span>
        ) : (
          <>
            <span className={s.vh}>{`${signedMoney(pv.difference)} ${rec ? "trip cost" : "net to boat"} against plan`}</span>
            <span aria-hidden="true" className={s.figureNum} data-tone={tone(pv.difference)}>
              <RollingNumber value={pv.difference} format={signedMoney} cents />
            </span>
          </>
        )}
      </p>
      <div className={s.pvBars}>
        {bar("Planned", pv.planned.result, "plan")}
        {bar("Actual", pv.actual.result, "actual")}
      </div>
      {pv.parts.length ? (
        <>
          <div className={s.attrHead}>
            <Micro>Where the difference came from</Micro>
            <span className={s.attrKey} aria-hidden="true">
              <i data-tone={rec ? "up" : "down"} /> {rec ? "cost less" : "lowered"} <i data-tone={rec ? "down" : "up"} /> {rec ? "cost more" : "raised"}
            </span>
          </div>
          <ul className={s.attr} aria-label="Where the difference came from">
            {pv.parts.map(part)}
          </ul>
          {!pv.explained && pv.difference !== null ? <p className={s.note}>Some parts are not recorded, so they do not add up to the whole difference.</p> : null}
        </>
      ) : null}
      <details className={s.pvMore}>
        <summary>How this is worked out</summary>
        <ul>
          {!rec ? <li>Catch landed: the change in pounds at the planned price. Price received: the change in price on what you landed. Together they are the whole change in gross.</li> : null}
          {pv.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      </details>
    </section>
  );
}
