"use client";

import { useId, useMemo, useState, type ReactNode } from "react";
import { parseField, type LimitKind } from "@/lib/tripEconomics";
import { Button, Dialog, IconButton, Micro, dayTime } from "@/nx/ds";
import { tripFishery } from "@/nx/coast/fisheries";
import type { Outcome, Trip } from "../types";
import s from "./economics.module.css";

/**
 * What happened (v15): the reader records the trip after it. Every field is optional and nothing is filled in with a
 * number (the plan's catch lines and cost lines lend only their names); a blank is "not recorded", never 0. A recorded
 * outcome opens again with its figures, and saving it again keeps when it was first recorded and stamps the edit.
 * A recreational trip has no price or crew pay: its catch is not sold.
 */

type Line = { label: string; lb: string; price: string };
type Cost = { label: string; usd: string };
type Form = { departedOn: string; returnedOn: string; landed: Line[]; fuelGal: string; fuelPrice: string; costs: Cost[]; crew: string; landingRef: string; conditions: string; notes: string };

const txt = (v: number | null | undefined) => (typeof v === "number" && Number.isFinite(v) ? String(v) : "");

/** The form for a trip: its recorded outcome, or blanks named after the plan's own lines. */
export function formFor(trip: Trip): Form {
  const o = trip.outcome;
  if (o) {
    return {
      departedOn: o.departedOn ?? "",
      returnedOn: o.returnedOn ?? "",
      landed: o.landed.length ? o.landed.map((l) => ({ label: l.label, lb: txt(l.lb), price: txt(l.usdPerLb) })) : [{ label: "", lb: "", price: "" }],
      fuelGal: txt(o.fuelGal),
      fuelPrice: txt(o.fuelUsdPerGal),
      costs: o.costs.length ? o.costs.map((c) => ({ label: c.label, usd: txt(c.usd) })) : [{ label: "Bait and ice", usd: "" }],
      crew: txt(o.crewPaidUsd),
      landingRef: o.landingRef ?? "",
      conditions: o.conditions,
      notes: o.notes,
    };
  }
  const f = tripFishery(trip.fishery);
  const planned = trip.economics?.inputs;
  const catchNames = (planned?.catch ?? []).map((c) => c.species).filter(Boolean);
  const costNames = (planned?.costs ?? []).map((c) => c.label).filter(Boolean);
  return {
    departedOn: trip.depart?.date ?? "",
    returnedOn: "",
    landed: (catchNames.length ? catchNames : [f?.label ?? "Catch"]).map((label) => ({ label, lb: "", price: "" })),
    fuelGal: "",
    fuelPrice: "",
    costs: (costNames.length ? costNames : [f?.supplies ?? "Bait and ice"]).map((label) => ({ label, usd: "" })),
    crew: "",
    landingRef: "",
    conditions: "",
    notes: "",
  };
}

type Parsed = { outcome: Omit<Outcome, "recordedAt" | "updatedAt">; errors: Record<string, string> };

/** The form read back: numbers parsed like the worksheet's (blank is null), per-field errors, lines with nothing in them dropped. */
export function readForm(form: Form, rec: boolean): Parsed {
  const errors: Record<string, string> = {};
  const n = (key: string, raw: string, kind: LimitKind) => {
    const p = parseField(raw, kind);
    if (p.error) errors[key] = p.error;
    return p.value;
  };
  const landed = form.landed
    .map((l, i) => ({ label: l.label.trim().slice(0, 60), lb: n(`landed.${i}.lb`, l.lb, "lb"), usdPerLb: rec ? null : n(`landed.${i}.price`, l.price, "usdPerLb") }))
    .filter((l, i) => l.lb !== null || l.usdPerLb !== null || form.landed[i].lb.trim() || form.landed[i].price.trim());
  const costs = form.costs.map((c, i) => ({ label: c.label.trim().slice(0, 60), usd: n(`cost.${i}`, c.usd, "usd") })).filter((c, i) => c.usd !== null || form.costs[i].usd.trim());
  if (form.departedOn && form.returnedOn && form.returnedOn < form.departedOn) errors.returnedOn = "Before the departure date";
  return {
    errors,
    outcome: {
      departedOn: form.departedOn || null,
      returnedOn: form.returnedOn || null,
      landed,
      fuelGal: n("fuelGal", form.fuelGal, "gal"),
      fuelUsdPerGal: n("fuelPrice", form.fuelPrice, "usdPerGal"),
      costs,
      crewPaidUsd: rec ? null : n("crew", form.crew, "usd"),
      landingRef: form.landingRef.trim().slice(0, 60) || null,
      conditions: form.conditions.trim().slice(0, 500),
      notes: form.notes.trim().slice(0, 2000),
    },
  };
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className={s.ofGroup}>
      <legend>
        <Micro>{title}</Micro>
      </legend>
      {children}
    </fieldset>
  );
}

/**
 * One field of the form, in the v14 equation's soft filled style (a figure with its unit beside it). Built here because
 * ds Field is the v12 ruled ledger line; a candidate for the design system.
 */
function TextField({ label, value, onChange, type = "text", numeric, prefix, suffix, multiline, error, width }: { label: string; value: string; onChange: (v: string) => void; type?: "text" | "date"; numeric?: boolean; prefix?: string; suffix?: string; multiline?: boolean; error?: string; width?: "figure" | "wide" }) {
  const id = useId();
  return (
    <div className={s.ofText} data-width={width ?? (numeric ? "figure" : "wide")}>
      <label htmlFor={id}>{label}</label>
      {multiline ? (
        <textarea id={id} value={value} rows={3} onChange={(e) => onChange(e.target.value)} />
      ) : (
        <span className={s.ofBox} data-error={error ? "" : undefined}>
          {prefix ? <i aria-hidden="true">{prefix}</i> : null}
          <input id={id} type={type} inputMode={numeric ? "decimal" : undefined} autoComplete="off" placeholder={numeric ? "—" : undefined} value={value} onChange={(e) => onChange(e.target.value)} aria-invalid={error ? true : undefined} aria-describedby={error ? `${id}-e` : undefined} />
          {suffix ? <i aria-hidden="true">{suffix}</i> : null}
        </span>
      )}
      {error ? (
        <span id={`${id}-e`} className={s.ofErr} role="alert">
          {error}
        </span>
      ) : null}
    </div>
  );
}

export type OutcomeFormProps = {
  trip: Trip;
  /** the outcome to keep: recordedAt kept from an earlier record, updatedAt now */
  onSave: (outcome: Outcome) => void;
  onCancel?: () => void;
  now?: () => Date;
};

/** The outcome form itself (inline). `OutcomeSheet` puts it in a sheet. */
export function OutcomeForm({ trip, onSave, onCancel, now = () => new Date() }: OutcomeFormProps) {
  const rec = trip.activity === "recreational";
  const [form, setForm] = useState<Form>(() => formFor(trip));
  const parsed = useMemo(() => readForm(form, rec), [form, rec]);
  // a figure that is not a number says so as it is typed (the worksheet's own wording)
  const err = (k: string) => parsed.errors[k];
  const set = (patch: Partial<Form>) => setForm((f) => ({ ...f, ...patch }));
  const setLine = (i: number, patch: Partial<Line>) => setForm((f) => ({ ...f, landed: f.landed.map((l, k) => (k === i ? { ...l, ...patch } : l)) }));
  const setCost = (i: number, patch: Partial<Cost>) => setForm((f) => ({ ...f, costs: f.costs.map((c, k) => (k === i ? { ...c, ...patch } : c)) }));
  const save = () => {
    if (Object.keys(parsed.errors).length) return;
    const at = now().toISOString();
    onSave({ ...parsed.outcome, recordedAt: trip.outcome?.recordedAt ?? at, updatedAt: at });
  };
  return (
    <form
      className={s.of}
      onSubmit={(e) => {
        e.preventDefault();
        save();
      }}
      data-testid="outcome-form"
      noValidate
    >
      {trip.outcome ? (
        <p className={s.ofStamp} data-testid="outcome-stamp">
          Recorded {dayTime(trip.outcome.recordedAt)}
          {trip.outcome.updatedAt !== trip.outcome.recordedAt ? ` · edited ${dayTime(trip.outcome.updatedAt)}` : ""}
        </p>
      ) : (
        <p className={s.ofStamp}>Everything is optional. A blank stays blank, never 0. It stays on this device.</p>
      )}

      <Group title={rec ? "Catch kept" : "Landed"}>
        {form.landed.map((l, i) => (
          <div className={s.ofLine} key={i}>
            <TextField label={i === 0 ? "Species" : `Species, line ${i + 1}`} value={l.label} onChange={(v) => setLine(i, { label: v })} />
            <TextField numeric label="Weight" suffix="lb" value={l.lb} onChange={(v) => setLine(i, { lb: v })} error={err(`landed.${i}.lb`)} />
            {rec ? null : <TextField numeric label="Price received" prefix="$" suffix="/lb" value={l.price} onChange={(v) => setLine(i, { price: v })} error={err(`landed.${i}.price`)} />}
            {i > 0 ? <IconButton icon="close" size={44} label={`Remove line ${i + 1}`} onClick={() => setForm((f) => ({ ...f, landed: f.landed.filter((_, k) => k !== i) }))} /> : null}
          </div>
        ))}
        {form.landed.length < 10 ? (
          <Button variant="quiet" icon="plus" onClick={() => set({ landed: [...form.landed, { label: "", lb: "", price: "" }] })}>
            Another species
          </Button>
        ) : null}
      </Group>

      <Group title="Fuel">
        <div className={s.ofLine}>
          <TextField numeric label="Fuel used" suffix="gal" value={form.fuelGal} onChange={(v) => set({ fuelGal: v })} error={err("fuelGal")} />
          <TextField numeric label="Fuel price" prefix="$" suffix="/gal" value={form.fuelPrice} onChange={(v) => set({ fuelPrice: v })} error={err("fuelPrice")} />
        </div>
      </Group>

      <Group title="Bait, ice and other">
        {form.costs.map((c, i) => (
          <div className={s.ofLine} key={i}>
            <TextField label={i === 0 ? "What" : `What, line ${i + 1}`} value={c.label} onChange={(v) => setCost(i, { label: v })} />
            <TextField numeric label="Paid" prefix="$" value={c.usd} onChange={(v) => setCost(i, { usd: v })} error={err(`cost.${i}`)} />
            {i > 0 ? <IconButton icon="close" size={44} label={`Remove cost ${i + 1}`} onClick={() => setForm((f) => ({ ...f, costs: f.costs.filter((_, k) => k !== i) }))} /> : null}
          </div>
        ))}
        {form.costs.length < 20 ? (
          <Button variant="quiet" icon="plus" onClick={() => set({ costs: [...form.costs, { label: "", usd: "" }] })}>
            Another cost
          </Button>
        ) : null}
      </Group>

      {rec ? null : (
        <Group title="Crew">
          <TextField numeric label="Crew paid, in total" prefix="$" value={form.crew} onChange={(v) => set({ crew: v })} error={err("crew")} />
        </Group>
      )}

      <Group title="The trip">
        <div className={s.ofLine}>
          <TextField label="Departed" type="date" value={form.departedOn} onChange={(v) => set({ departedOn: v })} />
          <TextField label="Returned" type="date" value={form.returnedOn} onChange={(v) => set({ returnedOn: v })} error={err("returnedOn")} />
        </div>
        <TextField label="Landing receipt or fish ticket number" value={form.landingRef} onChange={(v) => set({ landingRef: v })} />
        <TextField label="Conditions, in your words" value={form.conditions} onChange={(v) => set({ conditions: v })} />
        <TextField label="Notes" value={form.notes} onChange={(v) => set({ notes: v })} multiline />
      </Group>

      <div className={s.ofFoot}>
        {onCancel ? (
          <Button variant="quiet" onClick={onCancel}>
            Cancel
          </Button>
        ) : null}
        <Button variant="primary" glyph="✓" type="submit" data-testid="outcome-save">
          {trip.outcome ? "Save changes" : "Record outcome"}
        </Button>
      </div>
    </form>
  );
}

/** The outcome form in a sheet (from the side on a wide screen, from the bottom on a phone). */
export function OutcomeSheet({ open, onOpenChange, ...props }: OutcomeFormProps & { open: boolean; onOpenChange: (open: boolean) => void }) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange} placement="side" title={props.trip.outcome ? "What happened" : "Record what happened"} description={props.trip.name} className={s.sheet}>
      {open ? <OutcomeForm {...props} onCancel={() => onOpenChange(false)} /> : null}
    </Dialog>
  );
}
