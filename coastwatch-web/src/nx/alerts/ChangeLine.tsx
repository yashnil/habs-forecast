"use client";

import { Dumbbell, day, monthYear } from "@/nx/ds";
import { daysWords } from "./change";
import s from "./alerts.module.css";

const DAY = 864e5;

/**
 * The alert's before → after on a date axis (the synthesis dumbbell, ds `Dumbbell`, v14 README §3): the date the
 * record gave ("was", a ring) and the date it gives now (coral; hatched when it comes from sample data), weekly ticks
 * with each month named where it starts, and the gap at the end ("+14 days"). On open "now" slides out from "was"
 * with a small overshoot, then the gap appears (`slide`; reduced motion shows the end state).
 */
export function ChangeLine({ was, now, sample }: { was: Date; now: Date; sample?: boolean }) {
  const a = was.getTime(), b = now.getTime();
  const days = Math.round((b - a) / DAY);
  const gap = daysWords(days);
  const lo = Math.min(a, b), hi = Math.max(a, b);
  const span = Math.max(14 * DAY, hi - lo);
  const t0 = lo - Math.max(12 * DAY, span * 0.7);
  const t1 = hi + Math.max(6 * DAY, span * 0.25);
  const ticks: { value: number; label: string }[] = [];
  for (let t = Math.ceil(t0 / (7 * DAY)) * 7 * DAY; t <= t1; t += 7 * DAY) ticks.push({ value: t, label: "" });
  // each month named where it starts
  const m = new Date(t0);
  for (let k = 1; k < 4; k++) {
    const first = Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + k, 1, 12);
    if (first > t1) break;
    ticks.push({ value: first, label: monthYear(first) });
  }
  // the axis's first month at its start, when the next month's name is far enough along not to touch it
  const firstNamed = ticks.find((t) => t.label)?.value ?? Infinity;
  if ((firstNamed - t0) / (t1 - t0) > 0.3) ticks.push({ value: t0 + 4 * DAY, label: monthYear(t0) });
  const same = days === 0;
  const label = `Date in the record: was ${day(was, { year: true })}, now ${day(now, { year: true })}${sample ? " (sample data)" : ""}. ${same ? "Same date." : `${gap.figure} ${gap.note}.`}`;
  return (
    <Dumbbell
      className={s.line}
      label={label}
      slide
      domain={[t0, t1]}
      ticks={ticks}
      delta={{ big: gap.figure, small: gap.note }}
      rows={[
        {
          id: "change",
          lo: { value: a, label: same ? "" : "was", text: same ? undefined : day(was), mark: "hollow", tone: "ink" },
          hi: { value: b, label: "now", text: sample ? "sample" : day(now), mark: sample ? "hatched" : "filled", tone: "coral" },
        },
      ]}
    />
  );
}
