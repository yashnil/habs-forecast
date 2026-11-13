"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Button } from "@/nx/ds";
import { reveal } from "@/nx/ds/motion";
import { NWS_TZ, periodLabel } from "../forecast";
import { cellFacts, compareWindows, type WindowCell, type WindowField } from "../windows";
import s from "./departure.module.css";

export type WindowCompareProps = {
  a: WindowCell;
  b: WindowCell;
  timeZone?: string;
  onClear?: () => void;
  className?: string;
};

const SAME_LABEL: Partial<Record<WindowField, string>> = {
  windDir: "direction",
  wind: "wind",
  gust: "gusts",
  seas: "seas",
  waves: "wave detail",
  advisories: "advisories",
  tides: "tide",
};

/**
 * Two selected periods side by side: only what NWS (and the tide table, and the reader's own limits) report
 * differently, each difference on its own line; what is the same is one quiet line underneath. "not stated" is its
 * own value and differs from any number. No winner is drawn.
 */
export function WindowCompare({ a, b, timeZone = NWS_TZ, onClear, className }: WindowCompareProps) {
  const reduce = useReducedMotion();
  const [first, second] = (a.start ?? a.key) <= (b.start ?? b.key) ? [a, b] : [b, a];
  const diffs = compareWindows(first, second);
  const fa = cellFacts(first);
  const same = (Object.keys(SAME_LABEL) as WindowField[]).filter((k) => !diffs.some((d) => d.field === k) && fa[k] !== "—");
  const name = (c: WindowCell) => (c.period && c.state === "reported" ? periodLabel(c.period, timeZone) : `${c.label}${c.state === "expired" ? " (over)" : " (no forecast)"}`);
  return (
    <motion.section
      className={[s.compare, className].filter(Boolean).join(" ")}
      aria-label="Two departure periods compared"
      data-testid="window-compare"
      initial={reduce ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={reduce ? { duration: 0 } : reveal}
    >
      <div className={s.cmpHead}>
        <p className={s.cmpName}>{name(first)}</p>
        <span className={s.cmpVs} aria-hidden>
          ⇄
        </span>
        <p className={s.cmpName}>{name(second)}</p>
      </div>
      {diffs.length ? (
        <ul className={s.cmpRows}>
          <AnimatePresence initial={false}>
            {diffs.map((d) => (
              <motion.li
                key={d.field}
                className={s.cmpRow}
                data-field={d.field}
                layout={!reduce}
                initial={reduce ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={reduce ? { duration: 0 } : reveal}
              >
                <span className={s.cmpLabel}>{d.label}</span>
                <span className={s.cmpA}>{d.a}</span>
                <span className={s.cmpB}>{d.b}</span>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      ) : (
        <p className={s.cmpSame}>NWS reports the same conditions for both periods.</p>
      )}
      {same.length && diffs.length ? <p className={s.cmpSame}>Same in both: {same.map((k) => `${SAME_LABEL[k]} ${fa[k]}`).join(" · ")}</p> : null}
      {onClear ? (
        <Button variant="text" onClick={onClear} className={s.cmpClear}>
          Clear the comparison
        </Button>
      ) : null}
    </motion.section>
  );
}
