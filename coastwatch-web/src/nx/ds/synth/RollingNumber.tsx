"use client";

import { animate, useReducedMotion } from "motion/react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { roll, rollSlow } from "../motion";
import { cx } from "../util";
import s from "./synth.module.css";

export interface RollingNumberProps {
  value: number | null;
  /** The figure as it is written ("$2,602.24", "11"). */
  format: (v: number) => string;
  /** Written when value is null (never 0). */
  empty?: string;
  /** Text after the figure for screen readers ("ng/mL"). */
  srSuffix?: string;
  /** Split the cents into a smaller part, as the trip's net ("$2,602.24" → "$2,602" + ".24"). */
  cents?: boolean;
  /** "fast" (260 ms, the trip) or "slow" (360 ms, a sample's value). */
  pace?: "fast" | "slow";
  className?: string;
  "data-testid"?: string;
}

/**
 * A figure that rolls per digit (v14 README §3): when its value changes, each digit turns in a clipped column from the
 * old digit to the new one for 260 ms, then the figure settles back to its plain string (no tabular gaps). The first
 * value is written as it is; it never counts up from zero. The exact value is always the accessible text.
 */
export function RollingNumber({ value, format, empty = "—", srSuffix, cents, pace = "fast", className, "data-testid": testid }: RollingNumberProps) {
  const text = value == null ? empty : format(value);
  const prev = useRef<string | null>(null);
  const [rolling, setRolling] = useState<{ from: string; to: string } | null>(null);
  const reduce = useReducedMotion();
  const box = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    const from = prev.current;
    prev.current = text;
    // only a figure of the same shape rolls ($2,602.24 → $2,739.74); a new shape is written as it is
    if (from == null || from === text || reduce || value == null || from.replace(/\d/g, "d") !== text.replace(/\d/g, "d")) return;
    setRolling({ from, to: text });
  }, [text, reduce, value]);
  useEffect(() => {
    if (!rolling || !box.current) return;
    const cols = [...box.current.querySelectorAll<HTMLElement>("[data-col]")];
    const t = pace === "slow" ? rollSlow : roll;
    const anims = cols.map((c) => animate(c, { y: [`${-Number(c.dataset.from)}em`, `${-Number(c.dataset.to)}em`] }, t));
    const done = setTimeout(() => setRolling(null), ((t as { duration: number }).duration) * 1000 + 40);
    return () => {
      clearTimeout(done);
      anims.forEach((a) => a.stop());
    };
  }, [rolling, pace]);
  const split = (str: string) => {
    if (!cents) return [str, ""];
    const m = /^(.*?)(\.\d+)$/.exec(str);
    return m ? [m[1], m[2]] : [str, ""];
  };
  const render = (str: string, from: string | null) => {
    // align the old string to the new one from the right, so units roll into units
    const pad = from ? from.padStart(str.length, " ").slice(-str.length) : null;
    return [...str].map((ch, i) => {
      if (!/\d/.test(ch) || !pad) return <span key={i}>{ch}</span>;
      const a = /\d/.test(pad[i]) ? Number(pad[i]) : 0;
      return (
        <span key={i} className={s.rd}>
          <span className={s.rc} data-col="" data-from={a} data-to={ch}>
            {"0123456789".split("").map((d) => (
              <span key={d}>{d}</span>
            ))}
          </span>
        </span>
      );
    });
  };
  const [whole, frac] = split(text);
  const r = rolling ? split(rolling.to) : null;
  const rf = rolling ? split(rolling.from) : null;
  return (
    <span className={cx(s.roll, rolling && s.rolling, className)} data-testid={testid}>
      <span ref={box} aria-hidden="true">
        {rolling && r && rf ? (
          <>
            {render(r[0], rf[0])}
            {r[1] ? <span className={s.cents}>{render(r[1], rf[1] || null)}</span> : null}
          </>
        ) : (
          <>
            {whole}
            {frac ? <span className={s.cents}>{frac}</span> : null}
          </>
        )}
      </span>
      <span className={s.vh}>
        {text}
        {srSuffix ? ` ${srSuffix}` : ""}
      </span>
    </span>
  );
}
