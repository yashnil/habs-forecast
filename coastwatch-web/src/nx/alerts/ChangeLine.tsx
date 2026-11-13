"use client";

import { animate, useReducedMotion } from "motion/react";
import { useEffect, useId, useState } from "react";
import { alertSlide, day, monthYear, useElementWidth } from "@/nx/ds";
import { daysWords } from "./change";
import s from "./alerts.module.css";

const DAY = 864e5;

/**
 * The alert's before → after on a date axis (the synthesis "dumbbell", SYSTEM.md "Alerts"): the date the record gave
 * (a ring, "was"), the date it gives now (filled; hatched when it comes from sample data), a coral arrow between, and
 * the gap at the end ("+14 days"). On open "now" slides from the old date to the new one with a small overshoot, then
 * the gap appears (alertSlide; reduced motion shows the end state). Weekly ticks; a month's name where it starts.
 */
export function ChangeLine({ was, now, sample }: { was: Date; now: Date; sample?: boolean }) {
  const [ref, w] = useElementWidth<HTMLDivElement>(416);
  const reduce = useReducedMotion();
  const pat = useId().replace(/:/g, "");
  const [e, setE] = useState(reduce ? 1 : 0);
  const a = was.getTime(), b = now.getTime();
  useEffect(() => {
    if (reduce) {
      setE(1);
      return;
    }
    setE(0);
    const c = animate(0, 1, { ...alertSlide, onUpdate: setE });
    return () => c.stop();
  }, [a, b, reduce]);

  const days = Math.round((b - a) / DAY);
  const gap = daysWords(days);
  const narrow = w < 330;
  const right = narrow ? 0 : 92;
  const lo = Math.min(a, b), hi = Math.max(a, b);
  const span = Math.max(14 * DAY, hi - lo);
  const t0 = lo - Math.max(12 * DAY, span * 0.7);
  const t1 = hi + Math.max(6 * DAY, span * 0.25);
  const x = (t: number) => 6 + ((t - t0) / (t1 - t0)) * (w - 12 - right);
  const by = 30;
  const ticks: number[] = [];
  for (let t = Math.ceil(t0 / (7 * DAY)) * 7 * DAY; t <= t1; t += 7 * DAY) ticks.push(t);
  // month names: the axis's first month at its start, each new month where it begins
  const months: { t: number; at: number; start: boolean }[] = [{ t: t0, at: 0, start: true }];
  const m = new Date(t0);
  for (let k = 1; k < 4; k++) {
    const first = Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + k, 1, 12);
    if (first > t1) break;
    months.push({ t: first, at: x(first), start: false });
  }
  const xa = x(a), xb = x(b);
  const xn = xa + (xb - xa) * e;
  const same = days === 0;
  const late = Math.max(0, Math.min(1, e * 2 - 1));
  const label = `Date in the record: was ${day(was, { year: true })}, now ${day(now, { year: true })}${sample ? " (sample data)" : ""}. ${same ? "Same date." : `${gap.figure} ${gap.note}.`}`;

  return (
    <div ref={ref} className={s.line}>
      <svg role="img" aria-label={label} viewBox={`0 0 ${w} 76`} width={w} height={76}>
        <defs>
          <pattern id={pat} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="4" height="4" className={s.lnHatchBg} />
            <line x1="0" y1="0" x2="0" y2="4" className={s.lnHatch} />
          </pattern>
        </defs>
        <line x1={0} x2={w - right} y1={by} y2={by} className={s.lnAxis} />
        {ticks.map((t) => (
          <line key={t} x1={x(t)} x2={x(t)} y1={by - 3} y2={by + 3} className={s.lnTick} />
        ))}
        {months
          .filter((mo, i) => i === 0 || mo.at - 0 > 74)
          .map((mo) => (
            <text key={mo.t} x={mo.start ? 0 : mo.at} y={by - 14} textAnchor={mo.start ? "start" : "middle"} className={s.lnMonth}>
              {monthYear(mo.t)}
            </text>
          ))}
        {!same && Math.abs(xn - xa) > 20 ? (
          <>
            <path d={`M${xa + Math.sign(xn - xa) * 9},${by}H${xn - Math.sign(xn - xa) * 14}`} className={s.lnArrow} />
            <path d={`M${xn - Math.sign(xn - xa) * 21},${by - 6}L${xn - Math.sign(xn - xa) * 12},${by}L${xn - Math.sign(xn - xa) * 21},${by + 6}`} className={s.lnHead} />
          </>
        ) : null}
        {!same ? <circle cx={xa} cy={by} r={6.5} className={s.lnWas} /> : null}
        <circle cx={xn} cy={by} r={8.5} className={s.lnNow} fill={sample ? `url(#${pat})` : undefined} />
        {!same ? (
          <text x={xa} y={by + 26} textAnchor="middle" className={s.lnWasT}>
            was
          </text>
        ) : null}
        <text x={xn} y={by + 26} textAnchor="middle" className={s.lnNowT}>
          now
        </text>
        <text x={xn} y={by + 42} textAnchor="middle" className={s.lnSub} opacity={e}>
          {sample ? "sample" : day(now)}
        </text>
        {!narrow ? (
          <>
            <text x={w} y={by + 6} textAnchor="end" className={s.lnGap} opacity={late}>
              {gap.figure}
            </text>
            <text x={w} y={by + 22} textAnchor="end" className={s.lnSub} opacity={late}>
              {gap.note}
            </text>
          </>
        ) : null}
      </svg>
      {narrow ? (
        <p className={s.lineGap} aria-hidden="true">
          <b>{gap.figure}</b> {gap.note}
        </p>
      ) : null}
    </div>
  );
}
