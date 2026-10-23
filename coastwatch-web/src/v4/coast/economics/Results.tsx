"use client";

import type { ReactNode } from "react";
import { num } from "@/lib/tripEconomics";
import type { FieldKey, Output, TripEvaluation } from "@/lib/tripEvaluation";
import type { Draft } from "@/lib/tripEconomicsStore";
import { Caps } from "@/v4/coast/ui/Instrument";
import { gapsOf, money, needLabel, valueOf } from "./model";

/**
 * The worksheet's results (v10): each result appears the moment its own inputs exist; one that cannot
 * says exactly what it waits for, as links that focus those fields. Known terms are drawn solid, missing
 * ones dashed (never as a zero). Numbers change on every keystroke and never animate.
 */

export type Focus = (k: FieldKey) => void;

/** The weighing unit shown: lb, kg or (market squid) short tons; `w` scales pounds, `p` scales $/lb. */
export type WU = { unit: "lb" | "kg" | "ton"; w: number; p: number };

/** "Needs fuel used · how crew is paid", each a link to its field. */
export function Needs({ fields, draft, onFocus, lead = "Needs", testId }: { fields: FieldKey[]; draft: Draft | null; onFocus: Focus; lead?: string; testId?: string }) {
  if (!fields.length) return null;
  return (
    <span className="text-[13.5px] leading-snug text-n-ink-2" data-testid={testId}>
      {lead}{" "}
      {fields.map((k, i) => (
        <span key={k}>
          {i > 0 && <span aria-hidden> · </span>}
          <button type="button" onClick={() => onFocus(k)} className="nx-focus inline-flex min-h-11 items-center rounded-[2px] font-medium text-n-ink underline decoration-dashed decoration-n-ink-3 underline-offset-[3px] hover:decoration-n-ink" data-testid="econ-need" data-field={k}>
            {needLabel(k, draft)}
          </button>
        </span>
      ))}
    </span>
  );
}

/** One headline figure: its value, or "Needs …" in its place (never a number it was not given). */
function Figure({ label, o, fmt, draft, onFocus, testId, strong = false, note }: { label: string; o: Output; fmt: (v: number) => string; draft: Draft; onFocus: Focus; testId: string; strong?: boolean; note?: ReactNode }) {
  // a phone reads it as a row (label and what it waits for on the left, the figure on the right); wider, a column
  const size = strong ? "text-[30px] sm:text-[34px]" : "text-[26px] sm:text-[28px]";
  return (
    <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 gap-y-1 py-3 sm:flex sm:flex-col sm:gap-1.5 sm:py-0">
      <dt className="sm:order-1">
        <Caps>{label}</Caps>
      </dt>
      <dd className="row-span-2 text-right sm:order-2 sm:text-left">
        {o.status === "value" ? (
          <span className={`nx-value ${size} ${o.value < 0 ? "text-n-down" : ""}`} data-testid={testId}>
            {fmt(o.value)}
          </span>
        ) : o.status === "unreachable" ? (
          <span className="text-[15px] font-medium leading-snug text-n-ink" data-testid={testId}>
            None
          </span>
        ) : (
          <span className={`nx-value ${size} text-n-ink-3`} data-testid={testId} aria-label={`${label}: not worked out`}>
            —
          </span>
        )}
      </dd>
      <dd className="min-w-0 sm:order-3">{o.status === "missing" ? <Needs fields={o.missing} draft={draft} onFocus={onFocus} /> : o.status === "unreachable" ? <span className="text-[13px] leading-snug text-n-ink-2">{o.reason}</span> : note}</dd>
    </div>
  );
}

export function Headline({ ev, draft, onFocus, wu }: { ev: TripEvaluation; draft: Draft; onFocus: Focus; wu: WU }) {
  const rec = ev.audience === "recreational";
  return (
    <dl className="grid grid-cols-1 divide-y divide-n-line sm:grid-cols-3 sm:gap-5 sm:divide-y-0" data-testid="econ-summary" aria-live="polite" aria-atomic="true">
      {rec ? (
        <>
          <Figure label="Cost of the trip" o={ev.tripCost} fmt={(v) => money(v)} draft={draft} onFocus={onFocus} testId="econ-sum-cost" strong note={<span className="text-[13px] text-n-ink-2">fuel and other costs</span>} />
          <Figure label="Per person" o={ev.perPerson} fmt={(v) => money(v)} draft={draft} onFocus={onFocus} testId="econ-sum-person" note={<span className="text-[13px] text-n-ink-2">optional split</span>} />
          <Figure label="Fuel used" o={ev.fuelGallons} fmt={(v) => `${num(v, 1)} gal`} draft={draft} onFocus={onFocus} testId="econ-sum-gal" />
        </>
      ) : (
        <>
          <Figure label="Net to boat" o={ev.netToBoat} fmt={(v) => money(v, true)} draft={draft} onFocus={onFocus} testId="econ-sum-net" strong note={<span className="text-[13px] text-n-ink-2">after fuel, costs, fees and crew</span>} />
          <Figure label="Trip cost" o={ev.tripCost} fmt={(v) => money(v)} draft={draft} onFocus={onFocus} testId="econ-sum-cost" note={<span className="text-[13px] text-n-ink-2">fuel and other costs</span>} />
          <Figure label="Break-even catch" o={ev.breakEvenCatch} fmt={(v) => `${num(v * wu.w, wu.unit === "ton" ? 2 : 0)} ${wu.unit}`} draft={draft} onFocus={onFocus} testId="econ-sum-breakeven" note={<span className="text-[13px] text-n-ink-2">where net reaches $0</span>} />
        </>
      )}
    </dl>
  );
}

// ---------------------------------------------------------------- waterfall

type Row = { key: string; label: string; o: Output; sign: 1 | -1 | 0; testId: string };

/**
 * Gross, less each deduction, to net: a waterfall in rows. A known term is a solid bar placed where the
 * money stands after the terms before it; a missing term is a dashed box across the track saying what it
 * waits for, and the running position after it is unknown (later known terms are drawn at their size,
 * hollow, because where they fall is not known).
 */
export function Waterfall({ ev, draft }: { ev: TripEvaluation; draft: Draft; onFocus?: Focus }) {
  const rows: Row[] = [
    { key: "gross", label: "Gross", o: ev.gross, sign: 1, testId: "econ-r-gross" },
    ...(ev.fees.status === "value" && ev.fees.value > 0 ? [{ key: "fees", label: "Fees", o: ev.fees, sign: -1 as const, testId: "econ-r-fees" }] : []),
    { key: "fuel", label: "Fuel", o: ev.fuelCost, sign: -1, testId: "econ-r-fuel" },
    { key: "other", label: "Other costs", o: ev.otherCosts, sign: -1, testId: "econ-r-other" },
    { key: "crew", label: "Crew pay", o: ev.crewPay, sign: -1, testId: "econ-r-crew" },
    { key: "net", label: "Net to boat", o: ev.netToBoat, sign: 0, testId: "econ-r-net" },
  ];
  const g = valueOf(ev.gross);
  const known = rows.filter((r) => r.sign === -1).map((r) => valueOf(r.o) ?? 0);
  const net = valueOf(ev.netToBoat);
  const hi = Math.max(g ?? 0, known.reduce((a, b) => a + b, 0), 1);
  const lo = Math.min(0, net ?? 0);
  const span = hi - lo;
  const pos = (v: number) => `${((v - lo) / span) * 100}%`;
  const width = (a: number, b: number) => `${(Math.abs(b - a) / span) * 100}%`;
  let x: number | null = g;
  return (
    <figure className="flex flex-col gap-2" data-testid="econ-waterfall">
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-3">
        <Caps>From gross to net</Caps>
        <span className="text-[12.5px] text-n-ink-2">solid: worked out · dashed: waits for an input</span>
      </figcaption>
      <ul className="flex flex-col">
        {rows.map((r) => {
          const v = valueOf(r.o);
          let bar: ReactNode;
          if (r.o.status !== "value") {
            bar = (
              <span className="absolute inset-y-1 left-0 right-0 flex items-center rounded-[2px] border border-dashed border-n-ink-3 px-2">
                <span className="truncate text-[12px] text-n-ink-2">{r.o.status === "missing" ? `waits for ${r.o.missing.map((k) => needLabel(k, draft)).join(", ")}` : r.o.status === "unreachable" ? r.o.reason : ""}</span>
              </span>
            );
            if (r.sign === -1) x = null;
          } else if (r.sign === 1) {
            bar = <span className="absolute inset-y-1 rounded-[1px] bg-n-ink/80" style={{ left: pos(0), width: width(0, v!) }} />;
          } else if (r.sign === -1) {
            if (x !== null) {
              bar = <span className="absolute inset-y-1 rounded-[1px] bg-n-ink-3/70" style={{ left: pos(x - v!), width: width(x - v!, x) }} />;
              x -= v!;
            } else {
              // its size is known, where it falls is not: drawn hollow at the right edge
              bar = <span className="absolute inset-y-1 rounded-[1px] border border-n-ink-3" style={{ right: 0, width: width(0, v!) }} title="Its size is known; where it falls waits for an earlier term" />;
            }
          } else {
            bar = <span className={`absolute inset-y-1 rounded-[1px] ${v! < 0 ? "bg-n-down/80" : "bg-n-ink"}`} style={{ left: pos(Math.min(0, v!)), width: width(0, v!) }} />;
          }
          return (
            <li key={r.key} className={`grid grid-cols-[6.5rem_minmax(0,1fr)_5.5rem] items-center gap-3 py-1 ${r.sign === 0 ? "mt-1 border-t border-n-line-strong pt-2" : ""}`} data-row={r.key}>
              <span className={`text-[13.5px] ${r.sign === 0 ? "font-medium text-n-ink" : "text-n-ink-2"}`}>{r.label}</span>
              <span className="relative h-7">
                {lo < 0 && <span aria-hidden className="absolute inset-y-0 w-px bg-n-line-strong" style={{ left: pos(0) }} />}
                {bar}
              </span>
              <span className={`text-right font-data text-[15px] tabular-nums ${v === null ? "text-n-ink-3" : v < 0 && r.sign === 0 ? "text-n-down" : "text-n-ink"}`} data-testid={r.testId}>
                {v === null ? "—" : r.sign === -1 ? (v === 0 ? "$0" : `−${money(v)}`) : money(v, r.sign === 0)}
              </span>
            </li>
          );
        })}
      </ul>
    </figure>
  );
}

// ---------------------------------------------------------------- break-even

export function BreakEven({ ev, draft, onFocus, wu }: { ev: TripEvaluation; draft: Draft; onFocus: Focus; wu: WU }) {
  const be = ev.breakEvenCatch;
  const lb = draft.inputs.catch?.[0]?.lb ?? null;
  const price = draft.inputs.catch?.[0]?.usdPerLb ?? null;
  if (be.status === "missing")
    return (
      <div className="flex flex-col gap-1.5" data-testid="econ-breakeven">
        <Caps>Break-even against your catch</Caps>
        <p className="text-[13.5px] text-n-ink-2">
          The catch that brings net to $0 does not need your catch, only the price, the costs and how crew is paid.{" "}
          <Needs fields={be.missing} draft={draft} onFocus={onFocus} lead="Waits for" />
        </p>
      </div>
    );
  if (be.status !== "value")
    return (
      <div className="flex flex-col gap-1.5" data-testid="econ-breakeven">
        <Caps>Break-even against your catch</Caps>
        <p className="text-[13.5px] text-n-ink-2">{be.status === "unreachable" || be.status === "notApplicable" ? be.reason : ""}</p>
      </div>
    );
  const unit = wu.unit;
  const d = unit === "ton" ? 2 : 0;
  const W = (v: number) => num(v * wu.w, d);
  const P = (v: number) => `$${(v * wu.p).toFixed(unit === "ton" ? 0 : 2)}/${unit}`;
  const b = be.value;
  const max = Math.max(b, lb ?? 0, 1) * 1.35;
  const at = (v: number) => `${Math.min(100, (v / max) * 100)}%`;
  const diff = lb !== null ? lb - b : null;
  const bp = ev.breakEvenPrice;
  const bf = ev.breakEvenFuelPrice;
  return (
    <figure className="flex flex-col gap-2" data-testid="econ-breakeven">
      <figcaption>
        <Caps>Break-even against your catch</Caps>
      </figcaption>
      <div className="relative mt-5 h-6" aria-hidden>
        <span className="absolute inset-y-0 left-0 rounded-l-[2px] border border-n-line-strong bg-[repeating-linear-gradient(135deg,transparent_0_5px,rgba(240,138,116,0.28)_5px_6px)]" style={{ width: at(b) }} />
        <span className="absolute inset-y-0 right-0 rounded-r-[2px] border border-l-0 border-n-line-strong bg-n-surface-3" style={{ left: at(b) }} />
        <span className="absolute -top-5 h-[calc(100%+20px)] w-px bg-n-ink" style={{ left: at(b) }} />
        <span className="absolute -top-5 whitespace-nowrap pl-1.5 font-data text-[12px] text-n-ink" style={{ left: at(b) }}>
          break-even {W(b)} {unit}
        </span>
        {lb !== null && <span className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-n-bg bg-n-ink" style={{ left: at(lb) }} />}
      </div>
      <div className="flex justify-between text-[11.5px] text-n-ink-3" aria-hidden>
        <span>0 {unit} · loss</span>
        <span>gain</span>
      </div>
      <p className="text-[13.5px] leading-snug text-n-ink-2" data-testid="econ-breakeven-text">
        {lb !== null ? (
          <>
            Your {W(lb)} {unit} is <b className="font-medium text-n-ink">{W(Math.abs(diff!))} {unit} {diff! >= 0 ? "above" : "below"}</b> break-even at {price !== null ? P(price) : "your price"}.
          </>
        ) : (
          <>
            Break-even is {W(b)} {unit} at your price. <Needs fields={["catch.0.lb"]} draft={draft} onFocus={onFocus} lead="To place your catch on it:" />
          </>
        )}{" "}
        {bp.status === "value" && lb !== null && <>At {W(lb)} {unit}, net reaches $0 at {P(bp.value)}. </>}
        {bf.status === "value" && <>Fuel would have to reach ${bf.value.toFixed(2)}/gal.</>}
      </p>
    </figure>
  );
}

// ---------------------------------------------------------------- sensitivity

const pctWord = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.round(Math.abs(x) * 100)} %`;

export function Sensitivity({ ev, draft, onFocus, wu }: { ev: TripEvaluation; draft: Draft; onFocus: Focus; wu: WU }) {
  const s = ev.sensitivity;
  const f = ev.fuelSensitivity;
  if (ev.audience === "recreational")
    return (
      <div className="flex flex-col gap-1.5" data-testid="econ-fuel-sens">
        <Caps>If fuel costs 20 % more or less</Caps>
        {f.status === "value" ? (
          <p className="font-data text-[15px] text-n-ink">
            −20 %: {money(f.value.down)} · +20 %: {money(f.value.up)} <span className="font-ui text-[13px] text-n-ink-2">trip cost</span>
          </p>
        ) : (
          <p className="text-[13.5px] text-n-ink-2">
            <Needs fields={gapsOf(f)} draft={draft} onFocus={onFocus} lead="Waits for" />
          </p>
        )}
      </div>
    );
  if (s.status !== "value")
    return (
      <div className="flex flex-col gap-1.5" data-testid="econ-sensitivity">
        <Caps>If price or catch differ</Caps>
        <p className="text-[13.5px] text-n-ink-2">
          A 3 × 3 grid of net to boat at price ±20 % and catch ±30 %, and fuel ±20 %, once net is worked out. <Needs fields={gapsOf(s)} draft={draft} onFocus={onFocus} lead="Waits for" />
        </p>
      </div>
    );
  const cell = (p: number, c: number) => s.value.find((x) => x.priceDelta === p && x.catchDelta === c)!.netUsd;
  const lb = draft.inputs.catch?.[0]?.lb ?? null;
  const price = draft.inputs.catch?.[0]?.usdPerLb ?? null;
  return (
    <div className="flex flex-col gap-3">
      <table className="w-full table-fixed border-collapse text-right" data-testid="econ-sensitivity">
        <caption className="mb-2 text-left">
          <Caps>Net to boat if price (rows) or catch (columns) differ</Caps>
        </caption>
        <thead>
          <tr className="text-[11.5px] text-n-ink-2">
            <th scope="col" className="w-[24%] pb-1.5 text-left font-normal">
              <span className="sr-only">Price change</span>
            </th>
            {[-0.3, 0, 0.3].map((c) => (
              <th key={c} scope="col" className="pb-1.5 pr-2 font-normal">
                <span className="block">catch {pctWord(c)}</span>
                {lb !== null && <span className="block font-data text-n-ink-3">{num(lb * (1 + c) * wu.w, wu.unit === "ton" ? 2 : 0)} {wu.unit}</span>}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-data text-[14.5px] tabular-nums">
          {[0.2, 0, -0.2].map((p) => (
            <tr key={p}>
              <th scope="row" className="py-0.5 text-left font-ui text-[11.5px] font-normal text-n-ink-2">
                <span className="block">price {pctWord(p)}</span>
                {price !== null && <span className="block font-data text-n-ink-3">${(price * (1 + p) * wu.p).toFixed(wu.unit === "ton" ? 0 : 2)}</span>}
              </th>
              {[-0.3, 0, 0.3].map((c) => {
                const v = cell(p, c);
                const centre = p === 0 && c === 0;
                return (
                  <td key={c} className="p-0.5">
                    <span className={`flex min-h-10 items-center justify-end rounded-[2px] px-2 ${centre ? "outline outline-1 outline-n-ink" : ""} ${v < 0 ? "bg-[repeating-linear-gradient(135deg,transparent_0_5px,rgba(240,138,116,0.22)_5px_6px)] text-n-down" : "bg-n-surface-3 text-n-ink"}`} data-testid={`econ-sens-${p}-${c}`}>
                      {money(v, true)}
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-[13px] leading-snug text-n-ink-2" data-testid="econ-fuel-sens">
        Outlined: your numbers. Hatched: the owner loses money. {f.status === "value" && <>Fuel price −20 %: <span className="font-data text-n-ink">{money(f.value.down, true)}</span>; +20 %: <span className="font-data text-n-ink">{money(f.value.up, true)}</span> net.</>}
      </p>
    </div>
  );
}
