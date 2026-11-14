"use client";
import { animate, motion, useMotionValue, useReducedMotion, useTransform } from "motion/react";
import { useEffect, useRef, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { usd } from "../format";
import { settle } from "../motion";
import { cx } from "../util";
import { layoutWaterfall, type WaterfallStep } from "./waterfallLayout";
import s from "./worksheet.module.css";

export interface WaterfallProps {
  steps: WaterfallStep[];
  /** Accessible name: "Trip, gross to net to boat, hypothetical". */
  label: string;
  /** Left column content per step (a Field, words); defaults to the label. */
  cells?: Record<string, ReactNode>;
  /**
   * A money scale held from outside, so previewed scenarios move the bars against one fixed axis (a bigger
   * catch visibly lengthens gross instead of rescaling everything to it). Values outside it are clamped.
   * Without it the scale follows the steps' own running low and high (E3).
   */
  domain?: [number, number];
  /** Each row gets `data-testid="<prefix>-<step id>"`. */
  testIdPrefix?: string;
  className?: string;
}

/**
 * A money figure that moves to its new value with `settle` when the reader's numbers or the previewed
 * scenario change (spec section 8: figures changing). The first value is drawn as it is; reduced motion cuts.
 */
export function SettlingMoney({ value, className }: { value: number; className?: string }) {
  const reduce = useReducedMotion();
  const mv = useMotionValue(value);
  const text = useTransform(mv, (v) => usd(Math.round(v)));
  useEffect(() => {
    if (reduce) {
      mv.set(value);
      return;
    }
    const c = animate(mv, value, settle);
    return () => c.stop();
  }, [value, reduce, mv]);
  return <motion.span className={className}>{text}</motion.span>;
}

/**
 * Gross to net as a ruled ledger with a live waterfall: gross solid ink, each cost a lighter solid from where
 * the last ended, net solid ink on a double rule. Never hatched: hatch is the model mark (README, semantic marks). Bars and figures move with `settle` springs when a figure or scenario
 * changes. A row that cannot be worked out says what it waits for across its figure and bar, never 0.
 */
export function Waterfall({ steps, label, cells, domain: fixed, testIdPrefix, className }: WaterfallProps) {
  const reduce = useReducedMotion();
  const t = reduce ? { duration: 0 } : settle;
  const own = layoutWaterfall(steps);
  const { bars } = own;
  const domain = fixed ?? own.domain;
  const [lo, hi] = domain;
  const span = hi - lo || 1;
  const pos = (v: number) => (Math.min(hi, Math.max(lo, v)) - lo) / span;
  return (
    <table className={cx(s.ledger, className)} aria-label={label}>
      <tbody>
        {steps.map((st, i) => {
          const b = bars[i];
          const amount = st.amount === null ? null : st.kind === "cost" ? -st.amount : st.amount;
          return (
            <tr key={st.id} className={cx(s.lrow, st.kind === "net" && s.sum)} data-kind={st.kind} data-testid={testIdPrefix ? `${testIdPrefix}-${st.id}` : undefined}>
              <th scope="row" className={s.lab}>
                {cells?.[st.id] ?? st.label}
              </th>
              {amount === null ? (
                <td colSpan={2} className={s.waiting}>
                  {st.waiting ?? "waiting"}
                </td>
              ) : (
                <>
                  <td className={s.amt}>
                    <SettlingMoney value={amount} />
                  </td>
                  <td className={s.barCell} aria-hidden>
                    <span className={s.barTrack}>
                      {lo < 0 ? <span className={s.zero} style={{ left: `${pos(0) * 100}%` }} /> : null}
                      {!b.waiting ? (
                        <>
                          {b.joinAt !== null ? <motion.span className={s.join} initial={false} animate={{ left: `${pos(b.joinAt) * 100}%` }} transition={t} /> : null}
                          <motion.span
                            className={cx(s.bar, st.kind === "cost" ? s.barCost : s.barSolid)}
                            initial={false}
                            animate={{ left: `${pos(b.from) * 100}%`, width: `${Math.max(0.4, (pos(b.to) - pos(b.from)) * 100)}%` }}
                            transition={t}
                          />
                        </>
                      ) : null}
                    </span>
                  </td>
                </>
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export interface SensitivityTableProps {
  /** Row factors (catch), e.g. +30 %, as entered, −30 %. */
  rows: Array<{ label: string; factor: number }>;
  /** Column factors (price), e.g. −20 %, as entered, +20 %. */
  cols: Array<{ label: string; factor: number }>;
  /** Net for a scenario; null while inputs are missing. */
  value: (rowFactor: number, colFactor: number) => number | null;
  /** The scenario the waterfall shows; hovering or focusing a cell previews it. */
  selected: [number, number] | null;
  onSelect: (cell: [number, number] | null) => void;
  /** What the rows vary ("Catch"): said in the corner, pointing down the row labels. */
  rowAxis: string;
  /** What the columns vary ("Price"): said over the column labels, pointing across. */
  colAxis: string;
  /** What is held fixed and that it is hypothetical (E4, T45). */
  caption: ReactNode;
  className?: string;
}

/**
 * The fixed ±20/30 % grid (T45). With a mouse or pen, hovering or focusing a cell previews that scenario and
 * leaving the grid returns to the reader's numbers. On touch a tap is a choice: it shows the scenario and keeps
 * it until the same cell is tapped again (or the page offers a way back). A touch sends pointerenter,
 * pointerleave and then click; only the click counts, so the scenario does not appear and clear at once.
 */
export function SensitivityTable({ rows, cols, value, selected, onSelect, rowAxis, colAxis, caption, className }: SensitivityTableProps) {
  // the pointer type of the gesture now under way (focus that follows a tap is not a hover preview either)
  const touch = useRef(false);
  const fine = (e: ReactPointerEvent) => {
    touch.current = e.pointerType === "touch";
    return !touch.current;
  };
  return (
    // scrolls inside itself when the figures are wider than a 320 px screen (critic 12)
    <div className={cx(s.sens, className)}>
      <table className={s.sensTable}>
        <thead>
          <tr className={s.axisRow}>
            <td />
            <th scope="colgroup" colSpan={cols.length} className={s.colAxis}>
              {colAxis} <span aria-hidden>→</span>
            </th>
          </tr>
          <tr>
            <th scope="col" className={s.corner}>
              {rowAxis} <span aria-hidden>↓</span>
            </th>
            {cols.map((c) => (
              <th key={c.label} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody onPointerLeave={(e) => fine(e) && onSelect(null)}>
          {rows.map((r, ri) => (
            <tr key={r.label}>
              <th scope="row">{r.label}</th>
              {cols.map((c, ci) => {
                const v = value(r.factor, c.factor);
                const on = selected?.[0] === ri && selected?.[1] === ci;
                const base = r.factor === 1 && c.factor === 1;
                return (
                  <td key={c.label} className={cx(on && s.on, base && s.base)}>
                    <button
                      type="button"
                      className={s.cellBtn}
                      aria-pressed={on}
                      aria-label={`${r.label} catch, ${c.label} price: ${v === null ? "waiting for your numbers" : usd(v)}`}
                      onPointerDown={(e) => void fine(e)}
                      onPointerEnter={(e) => fine(e) && onSelect([ri, ci])}
                      onFocus={() => !touch.current && onSelect([ri, ci])}
                      onClick={() => onSelect(on && touch.current ? null : [ri, ci])}
                      disabled={v === null}
                    >
                      {v === null ? "–" : usd(v)}
                    </button>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className={s.sensCap}>{caption}</p>
    </div>
  );
}
