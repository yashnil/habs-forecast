"use client";

import { useLayoutEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { Dumbbell, Waterfall, type CascadeStep } from "@/nx/ds";
import type { WhatIfRow } from "./pass";
import { money, type Step } from "./workspace";
import s from "./trip.module.css";

type Size = { w: number; h: number };
function useSize<T extends HTMLElement>(): [React.RefObject<T | null>, Size] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState<Size>({ w: 0, h: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => setSize((o) => (o.w === el.clientWidth && o.h === el.clientHeight ? o : { w: el.clientWidth, h: el.clientHeight }));
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

/** The equation's names, shortened the way the prototype writes them under a column. */
const NAME: Record<string, string> = { "Bait and ice": "Bait, ice", "Ice and bait": "Ice, bait", "Crew pay": "Crew" };

export type BreakEven = { gross: number; words: string; beyond: boolean } | null;

/**
 * The trip's waterfall: the kit's `Waterfall` (v14 README §3: cascades gross → net, re-tweens on every input, a null
 * amount is a dotted slot and the levels after it wait, the break-even caliper on the gross column), fed in the
 * equation's order. The net column says its share of gross (the net dollars live in the stub). A recreational trip is
 * its total carved into fuel and each cost. Over the chart, one focusable column per step gives its worked sum on
 * hover, focus or the arrow keys (the kit's column geometry).
 */
export function TripWaterfall({ steps, basis, rec, breakEven, label, description }: { steps: Step[]; basis: (st: Step) => string; rec: boolean; breakEven: BreakEven; label: string; description?: string | null }) {
  const [box, { w, h }] = useSize<HTMLDivElement>();
  const ordered = useMemo(() => (rec ? [...steps.filter((x) => x.kind === "net"), ...steps.filter((x) => x.kind !== "net")] : steps), [steps, rec]);
  const gross = steps.find((x) => x.id === "gross")?.amount ?? null;
  const cascade: CascadeStep[] = ordered.map((st) => {
    const name = NAME[st.label] ?? st.label;
    if (rec) return { id: st.id, label: st.kind === "net" ? "Trip cost" : name, basis: basis(st) || undefined, amount: st.amount, kind: st.kind === "net" ? "start" : "cost" };
    if (st.kind === "gross") return { id: st.id, label: name, basis: basis(st), amount: st.amount, kind: "start" };
    if (st.kind === "net") {
      const pct = st.amount !== null && gross ? Math.round((st.amount / gross) * 100) : null;
      return { id: st.id, label: name, basis: st.amount === null ? "needs every input" : undefined, amount: st.amount, kind: "end", valueText: pct === null ? undefined : `${pct < 0 ? "−" : ""}${Math.abs(pct)}%${w < 420 ? "" : " of gross"}` };
    }
    return { id: st.id, label: name, basis: basis(st) || undefined, amount: st.amount, kind: st.id === "crew" ? "share" : "cost" };
  });
  const height = Math.max(120, h);
  const [focus, setFocus] = useState<number | null>(null);
  const cols = useRef<Array<HTMLDivElement | null>>([]);
  // the kit's column geometry (synth/charts.tsx Waterfall)
  const n = cascade.length;
  const gap = w < 420 ? 8 : 14;
  const cw = n ? Math.max(20, Math.min(110, (w - gap * (n - 1)) / n)) : 0;
  const shown = focus !== null ? ordered[focus] : null;
  const onKey = (i: number) => (e: KeyboardEvent<HTMLDivElement>) => {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!d) return;
    e.preventDefault();
    cols.current[(i + d + n) % n]?.focus();
  };
  const signed = (st: Step) => (st.kind === "gross" || st.kind === "net" || rec ? money(st.amount ?? 0) : `−${money(st.amount ?? 0)}`);
  const beParts = breakEven ? breakEven.words.split(", ") : [];
  return (
    <div className={s.wf} ref={box} data-testid="trip-waterfall">
      {w > 0 ? (
        <Waterfall
          label={label}
          steps={cascade}
          format={money}
          height={height}
          breakEven={
            breakEven && !breakEven.beyond
              ? {
                  at: breakEven.gross,
                  label: (
                    <>
                      <tspan>Break-even </tspan>
                      <tspan className={s.beStrong}>{beParts[0]}</tspan>
                      {beParts.length > 1 ? <tspan>, {beParts.slice(1).join(", ")}</tspan> : null}
                    </>
                  ),
                }
              : null
          }
        />
      ) : null}
      <div className={s.wfCols}>
        {ordered.map((st, i) => (
          <div
            key={st.id}
            ref={(el) => {
              cols.current[i] = el;
            }}
            className={s.wfCol}
            style={{ left: i * (cw + gap) - gap / 2, width: cw + gap }}
            data-step={st.id}
            tabIndex={i === 0 ? 0 : -1}
            role="img"
            aria-label={st.amount === null ? `${st.label}: not drawn yet, ${st.waiting ?? "waiting"}` : `${st.label} ${signed(st)}${st.formula ? `: ${st.formula}` : ""}`}
            onMouseEnter={() => setFocus(i)}
            onMouseLeave={() => setFocus((f) => (f === i ? null : f))}
            onFocus={() => setFocus(i)}
            onBlur={() => setFocus((f) => (f === i ? null : f))}
            onKeyDown={onKey(i)}
          />
        ))}
      </div>
      {description ? (
        <p className={s.vh} data-testid="trip-waterfall-waiting">
          {description}
        </p>
      ) : null}
      {shown && shown.amount !== null ? (
        <div className={s.callout} data-testid="waterfall-callout" style={{ left: Math.max(0, Math.min(w - 260, (focus ?? 0) * (cw + gap) + cw / 2 - 130)) }} role="status">
          <b>
            {shown.label} {signed(shown)}
          </b>
          {shown.formula ? <span>{shown.formula}</span> : null}
        </div>
      ) : null}
    </div>
  );
}

/**
 * What if: the kit's `Dumbbell` (v14 README §3). Catch and price, each row the trip worked with that input lower and
 * higher; each end is a 44 px button that sets that input, the trip as entered is the caliper, the ends slide on change.
 */
export function TripWhatIf({ rows, now, onApply }: { rows: WhatIfRow[]; now: number; onApply: (k: "catch" | "price", v: number) => void }) {
  const [box, { h }] = useSize<HTMLDivElement>();
  const rowHeight = Math.max(44, Math.min(58, Math.floor(((h || 138) - 28) / rows.length)));
  return (
    <div className={s.wi} ref={box} data-testid="trip-whatif">
      <Dumbbell
        label={`What if, net to boat: ${rows.map((r) => `${r.label} ${r.lo.label} ${money(r.lo.net)}, ${r.hi.label} ${money(r.hi.net)}`).join("; ")}; as entered ${money(now)}`}
        heading="What if"
        caliper={{ value: now, label: "as entered" }}
        rowHeight={rowHeight}
        rows={rows.map((r) => ({ id: r.k, label: r.label, lo: end(r, r.lo, onApply), hi: end(r, r.hi, onApply) }))}
      />
    </div>
  );
}

const end = (r: WhatIfRow, e: WhatIfRow["lo"], onApply: (k: "catch" | "price", v: number) => void) => ({
  value: e.net,
  label: e.label,
  text: money(e.net),
  mark: "filled" as const,
  tone: e.net < 0 ? ("coral" as const) : ("ultra" as const),
  onPress: () => onApply(r.k, e.v),
  pressLabel: `What if ${r.k === "catch" ? "the catch" : "the price"} is ${e.label}: net to boat ${money(e.net)}. Set ${r.k === "catch" ? "the catch" : "the price"} to it.`,
});
