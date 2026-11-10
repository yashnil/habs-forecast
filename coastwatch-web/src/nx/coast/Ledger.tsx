"use client";

import { animate, motion, useMotionValue, useReducedMotion, useTransform } from "motion/react";
import { useEffect } from "react";
import { hatchSegments } from "@/nx/ds/Mark";
import { usd } from "@/nx/ds/format";
import { settle } from "@/nx/ds/motion";
import { cx } from "@/nx/ds/util";
import { layoutWaterfall, type WaterfallStep } from "@/nx/ds/worksheet/waterfallLayout";
import s from "./coast.module.css";

/**
 * A money figure that moves to its new value with the `settle` spring when the reader's numbers or the
 * previewed scenario change (spec §8: figures changing). Under reduced motion it cuts. Never on arrival:
 * the first value is drawn as it is.
 */
export function Money({ value, className }: { value: number; className?: string }) {
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

const HATCH = hatchSegments(0, 0, 600, 40, 5);

/**
 * Gross to net to boat as a ruled ledger with a live waterfall (README §9 Worksheet): gross solid, each
 * cost hatched from where the last ended, net solid on a double rule. Unlike the library's Waterfall it
 * takes the money scale from outside (`domain`), so a previewed scenario moves the bars against one fixed
 * axis: a bigger catch visibly lengthens gross instead of rescaling everything to it. Bars and figures
 * move with `settle`. A row that cannot be worked out says what it waits for, across its figure and bar.
 */
export function Ledger({ steps, domain, label }: { steps: WaterfallStep[]; domain: [number, number]; label: string }) {
  const { bars } = layoutWaterfall(steps);
  const [lo, hi] = domain;
  const pos = (v: number) => (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo);
  const reduce = useReducedMotion();
  const t = reduce ? { duration: 0 } : settle;
  return (
    <table className={s.ledger} aria-label={label}>
      <tbody>
        {steps.map((st, i) => {
          const b = bars[i];
          const signed = st.amount === null ? null : st.kind === "cost" ? -st.amount : st.amount;
          return (
            <tr key={st.id} className={cx(s.lrow, st.kind === "net" && s.lsum)} data-kind={st.kind} data-testid={`ledger-${st.id}`}>
              <th scope="row" className={s.llab}>
                {st.label}
              </th>
              {signed === null ? (
                <td colSpan={2} className={s.lwait}>
                  {st.waiting}
                </td>
              ) : (
                <>
                  <td className={s.lamt}>
                    <Money value={signed} />
                  </td>
                  <td className={s.lbar} aria-hidden>
                    <span className={s.ltrack}>
                      {lo < 0 ? <span className={s.lzero} style={{ left: `${pos(0) * 100}%` }} /> : null}
                      {!b.waiting ? (
                        <>
                          {b.joinAt !== null ? <motion.span className={s.ljoin} initial={false} animate={{ left: `${pos(b.joinAt) * 100}%` }} transition={t} /> : null}
                          <motion.span
                            className={cx(s.lbarFill, st.kind === "cost" ? s.lcost : s.lsolid)}
                            initial={false}
                            animate={{ left: `${pos(b.from) * 100}%`, width: `${Math.max(0.5, (pos(b.to) - pos(b.from)) * 100)}%` }}
                            transition={t}
                          >
                            {st.kind === "cost" ? (
                              <svg className={s.lhatch} focusable="false">
                                <path d={HATCH} />
                              </svg>
                            ) : null}
                          </motion.span>
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
