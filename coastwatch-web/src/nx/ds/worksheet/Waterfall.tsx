"use client";
import { motion } from "motion/react";
import { useId, type ReactNode } from "react";
import { usd } from "../format";
import { settle } from "../motion";
import { cx, svgId } from "../util";
import { layoutWaterfall, type WaterfallStep } from "./waterfallLayout";
import s from "./worksheet.module.css";

export interface WaterfallProps {
  steps: WaterfallStep[];
  /** Accessible name: "Trip, gross to net to boat, hypothetical". */
  label: string;
  /** Left column content per step (a Field, words); defaults to the label. */
  cells?: Record<string, ReactNode>;
  className?: string;
}

function Hatch({ id }: { id: string }) {
  return (
    <svg className={s.hatchSvg} aria-hidden focusable="false">
      <defs>
        <pattern id={id} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="4" className={s.hatchLine} />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill={`url(#${id})`} />
    </svg>
  );
}

/**
 * Gross to net as a ruled ledger with a live waterfall: gross solid, each cost hatched from where the last
 * ended, net solid on a double rule. Bars move with `settle` springs when a figure or scenario changes.
 */
export function Waterfall({ steps, label, cells, className }: WaterfallProps) {
  const base = svgId(useId());
  const { bars, domain, pos } = layoutWaterfall(steps);
  const zero = pos(0);
  return (
    <table className={cx(s.ledger, className)} aria-label={label}>
      <tbody>
        {steps.map((st, i) => {
          const b = bars[i];
          const left = pos(b.from) * 100;
          const width = Math.max(0.4, (pos(b.to) - pos(b.from)) * 100);
          const amount = st.amount === null ? null : st.kind === "cost" ? -st.amount : st.amount;
          return (
            <tr key={st.id} className={cx(s.lrow, st.kind === "net" && s.sum)} data-kind={st.kind}>
              <th scope="row" className={s.lab}>
                {cells?.[st.id] ?? st.label}
              </th>
              <td className={s.amt}>
                {amount === null ? <span className={s.waiting}>{st.waiting ?? "waiting"}</span> : usd(amount)}
              </td>
              <td className={s.barCell} aria-hidden>
                <span className={s.barTrack}>
                {domain[0] < 0 ? <span className={s.zero} style={{ left: `${zero * 100}%` }} /> : null}
                {!b.waiting ? (
                  <>
                    {b.joinAt !== null ? <motion.span className={s.join} initial={false} animate={{ left: `${pos(b.joinAt) * 100}%` }} transition={settle} /> : null}
                    <motion.span
                      className={cx(s.bar, st.kind === "cost" ? s.barCost : s.barSolid)}
                      initial={false}
                      animate={{ left: `${left}%`, width: `${width}%` }}
                      transition={settle}
                    >
                      {st.kind === "cost" ? <Hatch id={`${base}-${i}`} /> : null}
                    </motion.span>
                  </>
                ) : null}
                </span>
              </td>
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
  corner: string;
  /** What is held fixed and that it is hypothetical (E4, T45). */
  caption: ReactNode;
  className?: string;
}

/** The fixed ±20/30 % grid (T45). Hover or focus a cell to see that scenario in the waterfall. */
export function SensitivityTable({ rows, cols, value, selected, onSelect, corner, caption, className }: SensitivityTableProps) {
  return (
    <div className={cx(s.sens, className)}>
      <table className={s.sensTable}>
        <thead>
          <tr>
            <th scope="col" className={s.corner}>
              {corner}
            </th>
            {cols.map((c) => (
              <th key={c.label} scope="col">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody onPointerLeave={() => onSelect(null)}>
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
                      onPointerEnter={() => onSelect([ri, ci])}
                      onFocus={() => onSelect([ri, ci])}
                      onClick={() => onSelect(on ? null : [ri, ci])}
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
