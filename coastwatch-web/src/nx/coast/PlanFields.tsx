"use client";

import { useId } from "react";
import { Disclosure, Field } from "@/nx/ds";
import type { PlanDraft } from "./plan";
import p from "./plan.module.css";

/**
 * The planner's when, limits and area (v15). `full`: the date and an optional time, then "Your limits and area" behind
 * a disclosure (wind in knots and seas in feet, labelled as the reader's own, never as safe limits; a few words for
 * the area, never a position). `date`: the date alone, for the worksheet's strip.
 */
export function PlanFields({ plan, set, look = "full" }: { plan: PlanDraft; set: (patch: Partial<PlanDraft>) => void; look?: "full" | "date" }) {
  const id = useId();
  const date = (
    <span className={p.when} data-look={look}>
      <label htmlFor={`${id}-d`} className={look === "date" ? p.vh : p.label}>
        {look === "date" ? "Departure date" : "Date"}
      </label>
      <input id={`${id}-d`} type="date" className={p.date} value={plan.date} onChange={(e) => set({ date: e.target.value })} data-testid={look === "date" ? "plan-date-strip" : "plan-date"} />
    </span>
  );
  if (look === "date") return date;
  const limits = [plan.windKt && `${plan.windKt} kt`, plan.seasFt && `${plan.seasFt} ft`].filter(Boolean).join(" · ");
  return (
    <div className={p.fields} data-testid="plan-fields">
      <div className={p.row}>
        {date}
        <span className={p.when}>
          <label htmlFor={`${id}-t`} className={p.label}>
            Time <span className={p.opt}>optional</span>
          </label>
          <input id={`${id}-t`} type="time" className={p.time} value={plan.time} onChange={(e) => set({ time: e.target.value })} data-testid="plan-time" />
        </span>
      </div>
      <Disclosure summary="Your limits and area" value={limits || <span className={p.opt}>optional</span>} className={p.more} data-testid="plan-more">
        <p className={p.note}>Your own thresholds, to read the forecast against. Personal: not a rule or a standard.</p>
        <div className={p.row}>
          <Field label="Wind" suffix="kt" value={plan.windKt} onChange={(v) => set({ windKt: v })} size={3} />
          <Field label="Seas" suffix="ft" value={plan.seasFt} onChange={(v) => set({ seasFt: v })} size={3} />
        </div>
        <label htmlFor={`${id}-a`} className={p.label}>
          Area <span className={p.opt}>approximate, optional</span>
        </label>
        <input id={`${id}-a`} className={p.area} value={plan.area} maxLength={120} placeholder="Outside the bay, 20–40 fm" onChange={(e) => set({ area: e.target.value })} data-testid="plan-area" />
      </Disclosure>
    </div>
  );
}
