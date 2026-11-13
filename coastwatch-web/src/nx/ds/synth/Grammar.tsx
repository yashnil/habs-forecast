import type { ComponentPropsWithoutRef, ReactNode } from "react";
import { cx } from "../util";
import s from "./synth.module.css";

/**
 * The data-state grammar (v14 README §1, SYSTEM.md): marks, not sentences. One glyph per state, used everywhere a
 * value or a record is shown, on the chart and off it.
 *
 * - `measured` ● filled · `old` ◐ half-filled: measured, but older than 30 days
 * - `unverified` ○ ring: an official record not verified by a person (`unverified-coral` for regulation)
 * - `zero` □ "Reported 0": at or below the lab's limit, never a hero numeral
 * - `predicted` ┄ dashed: predicted or modelled (NWS seas, the next whale look, C-HARM)
 * - `hypothetical` ▨ hatch: hypothetical (the trip); `illustrative` the coral hatch (a sample change, an advisory area)
 * - `none` ⋯ dotted slot: no data (no test site, a blank input, not measured)
 * - `unread` / `read`: an alert's state
 */
export type GlyphKind = "measured" | "old" | "unverified" | "unverified-coral" | "zero" | "predicted" | "hypothetical" | "illustrative" | "none" | "unread" | "read";

export const GLYPH_WORDS: Record<GlyphKind, string> = {
  measured: "Measured",
  old: "Measured, more than 30 days ago",
  unverified: "Official record, not verified by a person",
  "unverified-coral": "Official regulation, not verified by a person",
  zero: "Reported 0: at or below the lab's limit",
  predicted: "Predicted or modelled",
  hypothetical: "Hypothetical",
  illustrative: "Illustrative",
  none: "No data",
  unread: "Unread",
  read: "Read",
};

export function Glyph({ kind, size, label, className }: { kind: GlyphKind; size?: number; label?: string; className?: string }) {
  return (
    <i
      className={cx(s.g, s[`g-${kind}`], className)}
      style={size ? { width: size, height: size } : undefined}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    />
  );
}

/** A micro-label: 11 px capitals, tracked +0.08em ("FORT BRAGG · DUNGENESS CRAB, COMMERCIAL"). Write it in sentence case. */
export function Micro({ as: As = "span", glyph, className, children, ...rest }: { as?: "span" | "p" | "h2" | "h3" | "div" | "dt"; glyph?: GlyphKind; children: ReactNode } & Omit<ComponentPropsWithoutRef<"span">, "children">) {
  return (
    <As className={cx(s.micro, className)} {...rest}>
      {glyph ? <Glyph kind={glyph} className={s.microG} /> : null}
      {children}
    </As>
  );
}

/** A provenance line with its glyph: "○ CDFW · checked Nov 8 · not verified". */
export function StampLine({ glyph = "unverified", className, children, ...rest }: { glyph?: GlyphKind; children: ReactNode } & Omit<ComponentPropsWithoutRef<"p">, "children">) {
  return (
    <p className={cx(s.stamp, className)} {...rest}>
      <Glyph kind={glyph} />
      <span>{children}</span>
    </p>
  );
}

/** A dotted slot: where a value would be and is not ("No test site for this port"). */
export function Slot({ className, children, ...rest }: ComponentPropsWithoutRef<"span">) {
  return (
    <span className={cx(s.slot, className)} {...rest}>
      {children}
    </span>
  );
}

/** NWS's pennants: one for a small craft advisory, two for a gale warning. */
export function Pennant({ n = 1, className }: { n?: 1 | 2; className?: string }) {
  return (
    <svg className={cx(s.pennant, className)} viewBox="0 0 22 18" aria-hidden="true">
      <path d="M2 1v16" className={s.pennantStaff} />
      {n === 1 ? <path d="M3 2l13 3.5L3 9z" className={s.pennantFlag} /> : (
        <>
          <path d="M3 1.5l11 3L3 7.5z" className={s.pennantFlag} />
          <path d="M3 9l11 3L3 15z" className={s.pennantFlag} />
        </>
      )}
    </svg>
  );
}

/** A wind arrow pointing where the wind blows to (from `fromDeg`). */
export function WindArrow({ fromDeg, className }: { fromDeg: number; className?: string }) {
  return (
    <svg className={cx(s.wind, className)} viewBox="0 0 14 14" style={{ transform: `rotate(${fromDeg + 180}deg)` }} aria-hidden="true">
      <path d="M7 13V2M3 6l4-4 4 4" />
    </svg>
  );
}

/** The gauge under a test: filled to the value, a caliper at the action level ("9.8 of 30 ppm"). */
export function LimitGauge({ value, limit, label, className }: { value: number; limit: number; label: string; className?: string }) {
  const pct = Math.max(0, Math.min(100, (100 * value) / limit));
  return (
    <span className={cx(s.lim, className)} role="img" aria-label={label}>
      <i style={{ width: `${pct.toFixed(1)}%` }} />
    </span>
  );
}
