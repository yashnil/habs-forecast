"use client";
import { motion, useTransform } from "motion/react";
import type { ReactNode } from "react";
import { fixed, usd } from "./format";
import { KIND_WORD, type Kind } from "./kinds";
import { useSettledNumber } from "./useSettledNumber";
import { cx } from "./util";
import s from "./controls.module.css";

export type FigureFormat = "money" | "int" | "dec1" | "dec2" | ((v: number) => string);

export interface FigureProps {
  /** The number; null when it cannot be worked out (then `waiting` is shown, never 0). */
  value: number | null;
  /** What it is: "Net to boat". */
  label: ReactNode;
  /** xl: the one result of a view; l: a secondary result; m: a ledger value. */
  size?: "xl" | "l" | "m";
  format?: FigureFormat;
  /** Money to the cent ($2,602.24) instead of whole dollars. */
  cents?: boolean;
  unit?: string;
  /** The kind of reading, said in words beside the label ("measured", "estimated"). */
  kind?: Kind;
  /** Words when the value is null: "waiting for price". */
  waiting?: string;
  /** A line under the figure (what it leaves out, its date). */
  note?: ReactNode;
  /** Label above the figure (default) or beside it on one line. */
  layout?: "stacked" | "inline";
  className?: string;
  "data-testid"?: string;
}

function formatter(format: FigureFormat, cents: boolean): (v: number) => string {
  if (typeof format === "function") return format;
  if (format === "money") return (v) => usd(cents ? Math.round(v * 100) / 100 : Math.round(v), { cents });
  if (format === "int") return (v) => Math.round(v).toLocaleString("en-US");
  return (v) => fixed(v, format === "dec2" ? 2 : 1);
}

function Moving({ value, fmt }: { value: number; fmt: (v: number) => string }) {
  const mv = useSettledNumber(value);
  const text = useTransform(mv, fmt);
  return <motion.span>{text}</motion.span>;
}

/**
 * A result figure: Public Sans 600, tabular, with its label and unit. A change of value moves from the old
 * figure to the new one (`useSettledNumber`); the exact value is always in the accessible text, so a screen
 * reader never hears a number in flight.
 */
export function Figure({ value, label, size = "l", format = "int", cents = false, unit, kind, waiting = "not worked out yet", note, layout = "stacked", className, ...rest }: FigureProps) {
  const fmt = formatter(format, cents);
  return (
    <div className={cx(s.figure, className)} data-size={size} data-layout={layout} data-testid={rest["data-testid"]}>
      <span className={s.figLabel}>
        {label}
        {kind ? <span className={s.figKind}> · {KIND_WORD[kind]}</span> : null}
      </span>
      {value === null || !Number.isFinite(value) ? (
        <span className={s.figWaiting}>{waiting}</span>
      ) : (
        <span className={s.figValue}>
          <span className={s.vh} data-figure-exact>
            {fmt(value)}
            {unit ? ` ${unit}` : ""}
          </span>
          <span aria-hidden="true">
            <Moving value={value} fmt={fmt} />
            {unit ? <span className={s.figUnit}>{unit}</span> : null}
          </span>
        </span>
      )}
      {note ? <span className={s.figNote}>{note}</span> : null}
    </div>
  );
}
