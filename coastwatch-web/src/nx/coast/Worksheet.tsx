"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { dieselSnapshot, priceRef, priceSnapshot, type RefSnapshot } from "@/content/referenceValues";
import { Button, Field, Segmented, SensitivityTable } from "@/nx/ds";
import { day, usd } from "@/nx/ds/format";
import { cx } from "@/nx/ds/util";
import { notifyTrip, tripStore, useTripData } from "@/v4/coast/economics/useTripData";
import { fuelReference, priceReference, shareReference, type CoastFishery, type CoastReference } from "./fisheries";
import { Ledger } from "./Ledger";
import {
  assumptions,
  EMPTY_FORM,
  evaluateForm,
  fromDraft,
  gridValue,
  GRID_COLS,
  GRID_ROWS,
  hasEntry,
  scenarioDomain,
  toDraft,
  tripName,
  waitingWords,
  withFuel,
  type CoastForm,
  type CrewKind,
  type FuelPayer,
  type Result,
  type TextField,
} from "./trip";
import s from "./coast.module.css";

const CREW_KINDS: Array<{ value: CrewKind; label: string }> = [
  { value: "afterExpenses", label: "Share after costs" },
  { value: "ofGross", label: "Share of gross" },
  { value: "dayRate", label: "Day rate" },
];
const PAYERS: Array<{ value: FuelPayer; label: string }> = [
  { value: "shared", label: "Shared, off the top" },
  { value: "boat", label: "The boat" },
];

const cap = (t: string) => t.charAt(0).toUpperCase() + t.slice(1);

/** "−20 %" for the grid's own labels; "As entered" stays words. */
const pctWord = (f: number) => (f === 1 ? "as entered" : `${f > 1 ? "+" : "−"}${Math.round(Math.abs(f - 1) * 100)} %`);

/**
 * The trip worksheet: fields with units (nothing filled in), a live waterfall from gross to net to boat,
 * break-evens, the ±20/30 % grid that previews a scenario in the waterfall, the assumptions in plain words,
 * and saving the trip on this device. Every figure here is the reader's own numbers worked through
 * `evaluateTrip` (via ./worksheet), labelled hypothetical (T45); nothing is forecast (T46).
 */
export function Worksheet({ fishery, placeKey, onFishery, now }: { fishery: CoastFishery | null; placeKey: string; onFishery: (id: string) => void; now: Date | null }) {
  const [form, setForm] = useState<CoastForm>(EMPTY_FORM);
  const [cell, setCell] = useState<[number, number] | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [said, setSaid] = useState<string | null>(null);
  const prevFishery = useRef(fishery);
  /** a saved trip brings its own fishery and numbers: nothing is cleared for it */
  const opening = useRef(false);

  // another fishery: a value taken from the last one's reference described that fishery, and a change of
  // unit (squid is sold by the ton) makes the catch and price mean something else, so those are cleared
  useEffect(() => {
    const prev = prevFishery.current;
    prevFishery.current = fishery;
    if (prev?.id === fishery?.id) return;
    if (opening.current) {
      opening.current = false;
      return;
    }
    setForm((f) => {
      const next = { ...f, taken: {} as CoastForm["taken"] };
      if (f.taken.price || (prev?.unit ?? "lb") !== (fishery?.unit ?? "lb")) {
        next.price = "";
        if ((prev?.unit ?? "lb") !== (fishery?.unit ?? "lb")) next.catch = "";
      }
      if (f.taken.share) next.share = "";
      if (f.taken.fuelPrice) next.taken.fuelPrice = true;
      return next;
    });
    setCell(null);
  }, [fishery]);

  const base = useMemo(() => evaluateForm(form, fishery), [form, fishery]);
  const shown = useMemo(() => (cell ? evaluateForm(form, fishery, { catch: GRID_ROWS[cell[0]].factor, price: GRID_COLS[cell[1]].factor }) : base), [form, fishery, base, cell]);
  const domain = useMemo(() => scenarioDomain(form, fishery), [form, fishery]);
  const refs = useMemo(() => ({ price: priceReference(fishery), fuelPrice: fuelReference(now), share: shareReference(fishery) }), [fishery, now]);
  const unit = fishery?.unit ?? "lb";
  const supplies = fishery?.supplies ?? "Bait and ice";

  const set = (id: TextField) => (v: string) => {
    setSaid(null);
    setForm((f) => {
      const next: CoastForm = { ...f, [id]: v, taken: { ...f.taken } };
      const ref = id === "price" || id === "fuelPrice" || id === "share" ? refs[id] : null;
      if (ref && (id === "price" || id === "fuelPrice" || id === "share")) {
        if (f[id] === "" && v === ref.value) next.taken[id] = true;
        else if (v !== ref.value) delete next.taken[id];
      }
      return next;
    });
  };

  const field = (id: TextField, label: string, o: { prefix?: string; suffix?: string; size?: number; hint?: ReactNode; numeric?: boolean; reference?: CoastReference | null } = {}) => {
    const ref = o.reference ?? null;
    const taken = (id === "price" || id === "fuelPrice" || id === "share") && form.taken[id] && ref ? ref.taken : undefined;
    const clamped = base.clamped.includes(id) ? `Taken as the most this worksheet accepts.` : undefined;
    return (
      <Field
        label={label}
        value={form[id]}
        onChange={set(id)}
        prefix={o.prefix}
        suffix={o.suffix}
        inputMode={o.numeric ? "numeric" : "decimal"}
        size={o.size ?? 6}
        layout="inline"
        className={s.field}
        error={base.errors[id]}
        reference={ref ? { value: ref.value, words: ref.words, source: ref.source } : undefined}
        fromReference={taken}
        hint={clamped ?? o.hint}
      />
    );
  };

  const crewNone = form.crew.trim() !== "" && Number(form.crew) === 0;
  const share = !crewNone && (form.crewKind === "afterExpenses" || form.crewKind === "ofGross");

  // the grid on a touch screen: a tap shows a scenario and a second tap (or "Back to your numbers") returns;
  // a pointer previews on hover and returns when it leaves the grid
  const hoverless = useRef(false);
  useEffect(() => {
    hoverless.current = typeof window !== "undefined" && window.matchMedia("(hover: none)").matches;
  }, []);
  const burst = useRef<{ first: [number, number] | null; t: ReturnType<typeof setTimeout> | null }>({ first: null, t: null });
  const onGrid = (c: [number, number] | null) => {
    if (!hoverless.current) return setCell(c);
    const b = burst.current;
    if (c && !b.first) b.first = c;
    if (b.t) return;
    b.t = setTimeout(() => {
      const first = b.first;
      b.first = null;
      b.t = null;
      if (first) setCell((prev) => (prev && prev[0] === first[0] && prev[1] === first[1] ? null : first));
    }, 80);
  };

  const label = (r: Result, fmt: (v: number) => string) => (r.status === "value" ? fmt(r.value) : r.status === "none" ? r.reason : r.status === "na" ? "no hired crew" : waitingWords(r, fishery));
  const perUnit = (v: number) => (unit === "ton" ? `$${Math.round(v).toLocaleString("en-US")}/ton` : `$${v.toFixed(2)}/lb`);
  const amount = (v: number) => (unit === "ton" ? `${v.toLocaleString("en-US", { maximumFractionDigits: 1 })} ton` : `${Math.round(v).toLocaleString("en-US")} lb`);

  // ---------------------------------------------------------------- saving on this device
  const data = useTripData();
  const saved = (data?.scenarios ?? []).filter((x) => x.placeKey === placeKey).slice(0, 6);
  const save = () => {
    const snaps: RefSnapshot[] = [];
    if (form.taken.fuelPrice) snaps.push(dieselSnapshot());
    const pr = priceRef(fishery?.priceKey);
    if (form.taken.price && pr) snaps.push(priceSnapshot(pr));
    const r = tripStore().saveScenario({ id: openId ?? undefined, name: tripName(form, fishery), placeKey, species: fishery?.id ?? "", draft: toDraft(form, fishery), refs: snaps });
    if (r.ok) {
      setOpenId(r.id ?? null);
      setSaid("Saved on this device.");
      notifyTrip();
    } else setSaid(r.reason === "full" ? "Twenty trips are saved here; delete one first." : r.reason === "readOnly" ? "A newer Neritic saved trips here; nothing was changed." : "This browser is not keeping data here (a private window?). Nothing was saved.");
  };
  const open = (id: string) => {
    const rec = data?.scenarios.find((x) => x.id === id);
    if (!rec) return;
    if (rec.species && rec.species !== fishery?.id) {
      opening.current = true;
      onFishery(rec.species);
    }
    setForm(fromDraft(rec.draft, rec.species || null));
    setCell(null);
    setOpenId(rec.id);
    setSaid(`Opened “${rec.name}”.`);
  };
  const remove = (id: string) => {
    tripStore().deleteScenario(id);
    notifyTrip();
    if (openId === id) setOpenId(null);
  };
  const clear = () => {
    setForm(EMPTY_FORM);
    setCell(null);
    setOpenId(null);
    setSaid(null);
  };

  const net = base.out.net;
  return (
    <div className={s.sheet} data-testid="coast-worksheet">
      <div className={s.group}>
        {field("catch", "Catch", { suffix: unit, size: 7 })}
        {field("price", "Price", { prefix: "$", suffix: `/${unit}`, reference: refs.price, hint: refs.price ? undefined : "No sourced average for this species: ask your buyer." })}
      </div>
      <div className={s.group}>
        {field("fuelGal", "Fuel", { suffix: "gal" })}
        {field("fuelPrice", "Fuel price", { prefix: "$", suffix: "/gal", reference: refs.fuelPrice })}
        {field("supplies", supplies, { prefix: "$" })}
        {field("fees", "Fees", { suffix: "% of gross", size: 4, hint: "Landing fees and buyer deductions; 0 if none." })}
      </div>
      <div className={s.group}>
        {field("crew", "Crew", { suffix: "people", size: 3, numeric: true, hint: "Not counting you; 0 if nobody else is paid." })}
        {!crewNone ? <Segmented label="How crew is paid" options={CREW_KINDS} value={form.crewKind} onChange={(v) => setForm((f) => ({ ...f, crewKind: v }))} className={s.seg3} /> : null}
        {share ? field("share", "Crew share", { suffix: form.crewKind === "ofGross" ? "% of gross" : "%", size: 4, reference: form.crewKind === "afterExpenses" ? refs.share : null }) : null}
        {!crewNone && form.crewKind === "afterExpenses" ? (
          <Segmented label={`Who pays ${withFuel(fishery)}`} options={PAYERS} value={form.fuelPayer} onChange={(v) => setForm((f) => ({ ...f, fuelPayer: v }))} className={s.seg2} />
        ) : null}
        {!crewNone && form.crewKind === "dayRate" ? (
          <>
            {field("dayRate", "Day rate", { prefix: "$", suffix: "a day each" })}
            {field("days", "Days at sea", { suffix: "days", size: 3 })}
          </>
        ) : null}
      </div>

      <section className={s.results} aria-label="Results, hypothetical, from your numbers">
        <p className={s.resultsHead}>
          <span className={s.hypo}>Hypothetical, from your numbers.</span>{" "}
          <span aria-live="polite" className={s.showing}>
            {cell ? (
              <>
                Showing catch {pctWord(GRID_ROWS[cell[0]].factor)}, price {pctWord(GRID_COLS[cell[1]].factor)}.{" "}
                <button type="button" className={s.textButton} onClick={() => setCell(null)}>
                  Back to your numbers
                </button>
              </>
            ) : null}
          </span>
        </p>
        <Ledger steps={shown.steps} domain={domain} label="Gross to net to boat, hypothetical, from your numbers" />
        <div className={s.grid} role="group" aria-label="If catch and price move">
          <h3 className={s.h3}>If catch and price move</h3>
          {base.grid ? (
            <SensitivityTable
              rows={GRID_ROWS}
              cols={GRID_COLS}
              value={(r, c) => gridValue(base.grid, r, c)}
              selected={cell}
              onSelect={onGrid}
              corner="Catch"
              caption={`Net to boat, price across and catch down. ${form.crewKind === "dayRate" ? `${cap(withFuel(fishery))} and day pay` : cap(withFuel(fishery))} stay as entered; fees and shares move with gross. Hover or tap a figure to see it above.`}
              className={s.sens}
            />
          ) : (
            <p className={s.waitingText}>{net.status === "waiting" ? `${cap(waitingWords(net, fishery) ?? "")} before the grid can be worked out.` : "Net to boat has no value for these numbers."}</p>
          )}
        </div>
        <dl className={s.outs}>
          <div>
            <dt>Trip costs</dt>
            <dd className={cx(shown.out.tripCost.status !== "value" && s.waitingText)}>{label(shown.out.tripCost, (v) => usd(v))}</dd>
          </div>
          <div>
            <dt>Per crew</dt>
            <dd className={cx(shown.out.perCrew.status !== "value" && s.waitingText)}>{label(shown.out.perCrew, (v) => usd(v))}</dd>
          </div>
          <div>
            <dt>Break-even price</dt>
            <dd className={cx(shown.out.breakEvenPrice.status !== "value" && s.waitingText)}>{label(shown.out.breakEvenPrice, perUnit)}</dd>
          </div>
          <div>
            <dt>Break-even catch</dt>
            <dd className={cx(shown.out.breakEvenCatch.status !== "value" && s.waitingText)}>{label(shown.out.breakEvenCatch, amount)}</dd>
          </div>
        </dl>
      </section>

      <section aria-label="What these numbers leave out">
        <h3 className={s.h3}>Read it with</h3>
        <ul className={s.assume}>
          {assumptions(form, fishery).map((a) => (
            <li key={a}>{a}</li>
          ))}
        </ul>
      </section>

      <section className={s.save} aria-label="Save this trip">
        <div className={s.actions}>
          <Button variant="primary" onClick={save} disabled={!hasEntry(form)}>
            {openId ? "Save changes" : "Save this trip"}
          </Button>
          {hasEntry(form) ? (
            <Button variant="text" onClick={clear}>
              Clear
            </Button>
          ) : null}
        </div>
        <p className={s.small} aria-live="polite">
          {said ?? "Kept on this device only. Nothing is sent to Neritic."}
        </p>
        {saved.length ? (
          <ul className={s.saved} aria-label="Trips saved on this device">
            {saved.map((x) => (
              <li key={x.id} data-testid="coast-saved-trip">
                <span className={s.savedName}>
                  {x.name}
                  <span className={s.savedWhen}> · saved {day(x.updatedAt)}</span>
                </span>
                <span className={s.savedActions}>
                  <Button variant="text" onClick={() => open(x.id)} aria-label={`Open ${x.name}`}>
                    Open
                  </Button>
                  <Button variant="text" onClick={() => remove(x.id)} aria-label={`Delete ${x.name}`}>
                    Delete
                  </Button>
                </span>
              </li>
            ))}
          </ul>
        ) : null}
      </section>
    </div>
  );
}
