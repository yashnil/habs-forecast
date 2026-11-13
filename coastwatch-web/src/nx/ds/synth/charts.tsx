"use client";

import { animate, useReducedMotion } from "motion/react";
import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { alertSlide, cascadeTween, datesDraw, seaRise, whatIfSlide } from "../motion";
import { useElementWidth } from "../useMedia";
import { cx } from "../util";
import s from "./charts.module.css";

/**
 * The synthesis chart kit (v14 README §3): plain SVG drawn from the page's numbers, animated with Motion, every value
 * written where it belongs. Truth rules hold: data never smooths or invents; a blank is a dotted slot, never 0; a
 * prediction is dashed; a hypothetical is hatched; an unverified record is hollow.
 */

/** A 0 → 1 progress that runs `t` on each change of `key` (instantly under reduced motion); 1 before hydration. */
function useProgress(key: string, t: Parameters<typeof animate>[2] | ((first: boolean) => Parameters<typeof animate>[2])): [number, boolean] {
  const reduce = useReducedMotion();
  const [p, setP] = useState(1);
  const first = useRef(true);
  const [wasFirst, setWasFirst] = useState(true);
  useLayoutEffect(() => {
    const isFirst = first.current;
    first.current = false;
    setWasFirst(isFirst);
    if (reduce) return setP(1);
    setP(0);
    const tr = typeof t === "function" ? t(isFirst) : t;
    const zero: number = 0;
    const a = animate(zero, 1, { ...(tr as object), onUpdate: setP });
    return () => a.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, reduce]);
  return [p, wasFirst];
}

const ease3 = (t: number) => 1 - (1 - t) ** 3;
const clamp01 = (t: number) => Math.max(0, Math.min(1, t));

/* ================================================================== Waterfall */

export type CascadeStepKind = "start" | "cost" | "share" | "end";
export type CascadeStep = {
  id: string;
  /** "Gross", "Fuel", "Crew", "Net to boat" */
  label: string;
  /** the worked sum under the label ("400 lb × $9.00"); short words on a narrow column */
  basis?: string;
  /** start: the total; cost and share: the amount taken off (positive); end: the result. null: not yet entered (a dotted slot) */
  amount: number | null;
  kind: CascadeStepKind;
  /** the column-end text when it is not the amount (the net column says "72% of gross"; the net dollars live in the stub) */
  valueText?: string;
};

export interface WaterfallProps {
  /** what the chart shows, for screen readers ("Hypothetical trip: gross to net to boat") */
  label: string;
  steps: CascadeStep[];
  /** writes an amount ("$3,600.00") */
  format: (v: number) => string;
  /** the result is hypothetical: hatched in ultramarine (the trip). A loss is the coral hatch. */
  hypothetical?: boolean;
  /** a caliper across the start column at the amount that just covers the costs, and its words */
  breakEven?: { at: number; label: ReactNode } | null;
  height?: number;
  className?: string;
}

/**
 * Waterfall (v14 README §3, the trip): one bar carved from gross down to net. Each column's value sits at its end (above
 * the gross and the net, under a cost), dashed connectors carry each level to the next, the break-even caliper stands on
 * the gross column. On arrival the columns cascade gross → net 70 ms apart; on every later change they tween.
 */
export function Waterfall({ label, steps, format, hypothetical = true, breakEven, height = 168, className }: WaterfallProps) {
  const [ref, w] = useElementWidth<HTMLDivElement>(560);
  const sig = steps.map((st) => `${st.kind}:${st.amount == null ? "_" : "n"}`).join(",");
  const prev = useRef<{ sig: string; bars: Array<{ a: number; b: number } | null> } | null>(null);
  const key = JSON.stringify(steps.map((st) => st.amount));
  const [p, first] = useProgress(key, (isFirst) => (isFirst || prev.current?.sig !== sig ? { duration: 0.42 + 0.07 * (steps.length - 1), ease: "linear" } : cascadeTween));
  // levels
  let lvl: number | null = null;
  const bars = steps.map((st) => {
    if (st.amount == null) {
      lvl = null;
      return null;
    }
    if (st.kind === "start") {
      lvl = st.amount;
      return { a: st.amount, b: 0 };
    }
    if (st.kind === "end") {
      const v = st.amount;
      return { a: Math.max(0, v), b: Math.min(0, v) };
    }
    if (lvl == null) return null;
    const top: number = lvl;
    lvl = top - st.amount;
    return { a: top, b: top - st.amount };
  });
  const from = prev.current && prev.current.sig === sig && !first ? prev.current.bars : null;
  useEffect(() => {
    if (p >= 1) prev.current = { sig, bars };
  });
  const narrow = w < 420;
  const n = steps.length;
  const top = 22, bottom = height - (narrow ? 30 : 34), gap = narrow ? 8 : 14;
  const cw = Math.max(20, Math.min(110, (w - gap * (n - 1)) / n));
  const vals = bars.flatMap((b) => (b ? [b.a, b.b] : []));
  const vmax = Math.max(1, ...vals), vmin = Math.min(0, ...vals);
  const y = (v: number) => top + ((vmax - v) / (vmax - vmin)) * (bottom - top);
  const xAt = (i: number) => i * (cw + gap);
  const hatch = useId().replace(/[^a-zA-Z0-9]/g, "");
  let prevEnd: number | null = null;
  const desc = steps.map((st) => `${st.label} ${st.amount == null ? "not entered" : st.valueText ?? (st.kind === "cost" || st.kind === "share" ? `−${format(st.amount)}` : format(st.amount))}${st.basis ? ` (${st.basis})` : ""}`).join(", ");
  return (
    <div ref={ref} className={cx(s.chart, className)} style={{ height }}>
      <svg viewBox={`0 0 ${w} ${height}`} role="img" aria-label={`${label}: ${desc}`}>
        <defs>
          <pattern id={`u${hatch}`} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="7" height="7" className={s.hatchGround} />
            <rect width="2.6" height="7" className={s.hatchStripe} />
          </pattern>
          <pattern id={`c${hatch}`} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="7" height="7" className={s.hatchGroundCoral} />
            <rect width="2.4" height="7" className={s.hatchStripeCoral} />
          </pattern>
        </defs>
        {vmin < 0 ? <line x1={0} x2={w} y1={y(0)} y2={y(0)} className={s.base} /> : null}
        <line x1={0} x2={xAt(n - 1) + cw} y1={bottom + 0.5} y2={bottom + 0.5} className={vmin < 0 ? s.grid : s.base} />
        {steps.map((st, i) => {
          const x0 = xAt(i);
          const bar = bars[i];
          const lt = from ? p : clamp01((p * (0.42 + 0.07 * (n - 1)) - 0.07 * i) / 0.42);
          const e = from ? p : ease3(lt);
          const short: Record<string, string> = { "Bait, ice": "Bait", "Net to boat": "Net" };
          const name = narrow ? short[st.label] ?? st.label : st.label;
          const nameEl = (
            <>
              <text x={x0} y={bottom + 15} className={cx(s.colName, st.kind === "end" && s.colNameNet)}>
                {name}
              </text>
              {!narrow && cw >= 84 && st.basis ? (
                <text x={x0} y={bottom + 29} className={s.colBasis}>
                  {st.basis}
                </text>
              ) : null}
            </>
          );
          if (!bar) {
            prevEnd = null;
            const yt = top + (bottom - top) * 0.45;
            return (
              <g key={st.id}>
                <rect x={x0 + 1} y={yt} width={cw - 2} height={(bottom - top) * 0.55 - 1} rx={7} className={s.blank} />
                <text x={x0 + cw / 2} y={yt - 8} textAnchor="middle" className={s.blankWord}>
                  —
                </text>
                {nameEl}
              </g>
            );
          }
          const old = from?.[i] ?? null;
          let a = bar.a, b = bar.b;
          if (old) {
            a = old.a + (bar.a - old.a) * e;
            b = old.b + (bar.b - old.b) * e;
          } else if (st.kind === "cost" || st.kind === "share") {
            a = bar.a;
            b = bar.a + (bar.b - bar.a) * e;
          } else {
            a = bar.a * e;
            b = bar.b * e;
          }
          const zero = st.amount === 0 && st.kind !== "start";
          const loss = st.kind === "end" && (st.amount ?? 0) < 0;
          const yt = y(Math.max(a, b)), yb = y(Math.min(a, b)), hh = Math.max(zero ? 0 : 1.5, yb - yt);
          const fill = st.kind === "end" ? `url(#${loss ? "c" : "u"}${hatch})` : undefined;
          const cls = st.kind === "start" ? s.gross : st.kind === "share" ? s.share : st.kind === "cost" ? s.cost : hypothetical ? undefined : s.netSolid;
          const conn = prevEnd != null && e > 0.95 ? <line x1={x0 - gap} x2={x0} y1={y(prevEnd)} y2={y(prevEnd)} className={s.conn} /> : null;
          prevEnd = st.kind === "start" ? bar.a : st.kind === "end" ? null : bar.b;
          const costLike = st.kind === "cost" || st.kind === "share";
          const vy = costLike ? Math.min(bottom - 4, yb + 16) : yt - 7;
          const lower = narrow && costLike && i % 2 === 0 ? 13 : 0;
          const text = st.valueText ?? (costLike ? (zero ? format(0) : `−${format(st.amount as number)}`) : format(st.amount as number));
          return (
            <g key={st.id}>
              {conn}
              {zero ? <line x1={x0} x2={x0 + cw} y1={yt} y2={yt} className={s.zero} /> : <rect x={x0} y={yt} width={cw} height={hh} rx={4} className={cls} fill={fill} />}
              <text x={x0 + cw / 2} y={vy + lower} textAnchor="middle" className={cx(s.colVal, st.kind === "end" && s.colValNet, loss && s.colValLoss)} opacity={Math.min(1, e * 1.4)}>
                {text}
              </text>
              {nameEl}
            </g>
          );
        })}
        {breakEven && p >= 1 && breakEven.at <= vmax && bars[0]
          ? (() => {
              const yy = y(breakEven.at);
              const d = `M${-5},${yy}H${cw + 5}M${-5},${yy - 5}v10M${cw + 5},${yy - 5}v10`;
              return (
                <g data-breakeven="">
                  <path d={d} className={s.caliperHalo} />
                  <path d={d} className={s.caliper} />
                  <text x={cw + 12} y={Math.min(bottom - 6, yy + 4)} className={s.beText}>
                    {breakEven.label}
                  </text>
                </g>
              );
            })()
          : null}
      </svg>
    </div>
  );
}

/* ================================================================== Dumbbell */

export type DumbbellEnd = {
  value: number;
  /** above the dot ("300 lb", "was") */
  label: ReactNode;
  /** under the dot ("$1,882.24", "Jan 1") */
  text?: ReactNode;
  /** filled (a what-if result), hollow (an unverified record), hatched (illustrative) */
  mark?: "filled" | "hollow" | "hatched";
  tone?: "ultra" | "coral" | "ink";
  /** a what-if end sets its input: the end is a 44 px button */
  onPress?: () => void;
  /** the button's name ("What if catch is 300 lb: net to boat $1,882.24. Apply.") */
  pressLabel?: string;
};
export type DumbbellRow = { id: string; label?: ReactNode; lo: DumbbellEnd; hi: DumbbellEnd };

export interface DumbbellProps {
  /** what the chart shows, for screen readers */
  label: string;
  rows: DumbbellRow[];
  /** the value as entered: a caliper through every row, named above ("as entered") */
  caliper?: { value: number; label: ReactNode } | null;
  /** a heading over the chart ("What if") */
  heading?: ReactNode;
  /** words at the right end ("+14 days" over "later start"), shown when the slide has landed */
  delta?: { big: ReactNode; small?: ReactNode } | null;
  /** the `hi` end slides out from the `lo` end on arrival (an alert's was → now) */
  slide?: boolean;
  /** domain override (dates as ms) and axis ticks under the line */
  domain?: [number, number];
  ticks?: { value: number; label: ReactNode }[];
  rowHeight?: number;
  className?: string;
}

/**
 * Dumbbell (v14 README §3): two values on one axis joined by a bar. What-if rows (catch ±25 %, price ±$1, each end a
 * button that sets that input, the trip as entered a caliper through them) and an alert's change (was → now, the new
 * end sliding out with a small overshoot, then "+14 days"). The ends slide to new values (360 ms).
 */
export function Dumbbell({ label, rows, caliper, heading, delta, slide, domain, ticks, rowHeight = 58, className }: DumbbellProps) {
  const [ref, w] = useElementWidth<HTMLDivElement>(360);
  const key = JSON.stringify(rows.map((r) => [r.lo.value, r.hi.value]));
  const lastX = useRef<Array<[number, number]> | null>(null);
  const [p, first] = useProgress(key, (isFirst) => (slide && isFirst ? alertSlide : whatIfSlide));
  const hasLabels = rows.some((r) => r.label);
  const L = hasLabels ? 66 : 8, R = delta ? 84 : 50;
  const all = rows.flatMap((r) => [r.lo.value, r.hi.value]).concat(caliper ? [caliper.value] : []);
  const [lo, hi] = domain ?? [Math.min(...all), Math.max(...all)];
  const pad = domain ? 0 : (hi - lo) * 0.04 || 1;
  const x = (v: number) => L + ((v - (lo - pad)) / (hi - lo + 2 * pad || 1)) * Math.max(10, w - L - R);
  const top = heading || caliper ? 22 : 8;
  const rowH = rowHeight;
  const H = top + rowH * rows.length + (ticks ? 22 : 6);
  const prevX = !first ? lastX.current : null;
  const now = rows.map((r) => [x(r.lo.value), x(r.hi.value)] as [number, number]);
  useEffect(() => {
    if (p >= 1) lastX.current = now;
  });
  const markCls = (e: DumbbellEnd) => cx(s.dot, e.mark === "hollow" && s.dotHollow, e.mark === "hatched" && s.dotHatched, e.tone === "coral" && s.dotCoral, e.tone === "ink" && s.dotInk);
  return (
    <div ref={ref} className={cx(s.chart, s.dumbbell, className)} style={{ height: H }}>
      <svg viewBox={`0 0 ${w} ${H}`} role="img" aria-label={label}>
        {heading ? (
          <text x={0} y={11} className={s.cap}>
            {heading}
          </text>
        ) : null}
        {ticks?.map((t, i) => (
          <g key={i}>
            <line x1={x(t.value)} x2={x(t.value)} y1={top + rowH / 2 - 3} y2={top + rowH / 2 + 3} className={s.tick} />
            <text x={x(t.value)} y={H - 4} textAnchor="middle" className={s.tickText}>
              {t.label}
            </text>
          </g>
        ))}
        {rows.map((r, i) => {
          const yy = top + rowH * i + rowH / 2 + 2;
          const [tl, th] = now[i];
          const pl = prevX?.[i];
          const xl = pl ? pl[0] + (tl - pl[0]) * p : tl;
          const xh = slide && first ? tl + (th - tl) * p : pl ? pl[1] + (th - pl[1]) * p : th;
          return (
            <g key={r.id}>
              {r.label ? (
                <text x={0} y={yy + 4} className={s.rowLabel}>
                  {r.label}
                </text>
              ) : null}
              {ticks ? <line x1={L - 4} x2={w - R} y1={yy} y2={yy} className={s.axis} /> : null}
              <line x1={xl} x2={xh} y1={yy} y2={yy} className={cx(s.bar, r.hi.tone === "coral" && s.barCoral)} />
              <circle cx={xl} cy={yy} r={6.5} className={markCls(r.lo)} />
              <circle cx={xh} cy={yy} r={r.hi.mark === "hatched" ? 8.5 : 6.5} className={markCls(r.hi)} />
            </g>
          );
        })}
        {caliper
          ? (() => {
              const xn = x(caliper.value), yt = top + 4, yb = top + rowH * rows.length - 2;
              return (
                <g>
                  <path d={`M${xn},${yt}V${yb}M${xn - 4},${yt}h8M${xn - 4},${yb}h8`} className={s.caliper} />
                  <text x={xn} y={11} textAnchor="middle" className={s.tickText}>
                    {caliper.label}
                  </text>
                </g>
              );
            })()
          : null}
        {delta ? (
          <g opacity={clamp01(p * 2 - 1)}>
            <text x={w} y={top + rowH / 2 + 8} textAnchor="end" className={s.deltaBig}>
              {delta.big}
            </text>
            {delta.small ? (
              <text x={w} y={top + rowH / 2 + 24} textAnchor="end" className={s.tickText}>
                {delta.small}
              </text>
            ) : null}
          </g>
        ) : null}
      </svg>
      {/* the ends' words and buttons, in HTML so each is a real 44 px target */}
      <div className={s.ends}>
        {rows.flatMap((r, i) => {
          const yy = top + rowH * i + rowH / 2 + 2;
          const [tl, th] = now[i];
          const tight = th - tl < 120;
          return ([
            [r.lo, tl, "l"],
            [r.hi, th, "h"],
          ] as const).map(([e, px, side]) => {
            const body = (
              <>
                <span className={s.endLabel}>{e.label}</span>
                {e.text != null ? <span className={cx(s.endText, e.tone === "coral" && s.endTextCoral)}>{e.text}</span> : null}
              </>
            );
            const style = { transform: `translate(${Math.round(px)}px, ${Math.round(yy)}px)` };
            const cls = cx(s.end, tight && (side === "l" ? s.sideL : s.sideR));
            return e.onPress ? (
              <button key={`${r.id}${side}`} type="button" className={cx(cls, s.endBtn)} style={style} onClick={e.onPress} aria-label={e.pressLabel}>
                {body}
              </button>
            ) : (
              <span key={`${r.id}${side}`} className={cls} style={style}>
                {body}
              </span>
            );
          });
        })}
      </div>
    </div>
  );
}

/* ================================================================== DatesLine */

export interface DatesLineProps {
  /** for screen readers: every date in words */
  label: string;
  /** the window drawn, ms */
  domain: [number, number];
  /** the measure on record (a delay): from its effective date to the earliest start; hollow coral (official, not verified) */
  span?: { from: number; to: number; fromLabel?: ReactNode; toLabel?: ReactNode; openEnd?: boolean } | null;
  /** a predicted window (the next whale assessment): a dashed box, subject to change */
  window?: { from: number; to: number } | null;
  /** today: an ink rule and dot, named */
  today?: { t: number; label?: ReactNode } | null;
  /** month ticks under the line */
  ticks?: number[];
  height?: number;
  className?: string;
}

/** Dates on record (v14 README §3, Today): the delay drawn Oct 30 → Jan 1, today, the next assessment dashed. Draws in 700 ms. */
export function DatesLine({ label, domain, span, window: win, today, ticks = [], height = 82, className }: DatesLineProps) {
  const [ref, w] = useElementWidth<HTMLDivElement>(340);
  const [p] = useProgress(JSON.stringify([domain, span?.from, span?.to, win?.from, today?.t]), datesDraw);
  const x = (t: number) => 3 + ((t - domain[0]) / (domain[1] - domain[0])) * (w - 6);
  const by = 34;
  return (
    <div ref={ref} className={cx(s.chart, className)} style={{ height }}>
      <svg viewBox={`0 0 ${w} ${height}`} role="img" aria-label={label}>
        <line x1={0} x2={w} y1={by} y2={by} className={s.axisLine} />
        {ticks.map((t) => (
          <line key={t} x1={x(t)} x2={x(t)} y1={by + 5} y2={by + 9} className={s.tick} />
        ))}
        {span
          ? (() => {
              const xa = x(span.from), xb = x(span.to), xe = xa + (xb - xa) * p;
              return (
                <g>
                  <rect x={xa} y={by - 5} width={Math.max(0, xe - xa)} height={10} rx={5} className={s.delay} />
                  {p >= 1 && span.openEnd !== false ? <line x1={xb + 6} x2={w - 8} y1={by} y2={by} className={s.delayOpen} /> : null}
                  {p >= 1 ? <path d={`M${xb},${by - 11}V${by + 11}`} className={s.delayEnd} /> : null}
                  {span.fromLabel ? (
                    <text x={xa} y={by - 17} className={s.tickText}>
                      {span.fromLabel}
                    </text>
                  ) : null}
                  {span.toLabel ? (
                    <text x={xb} y={by + 30} textAnchor="middle" className={s.delayWord}>
                      {span.toLabel}
                    </text>
                  ) : null}
                </g>
              );
            })()
          : null}
        {win ? <rect x={x(win.from)} y={by - 9} width={Math.max(4, x(win.to) - x(win.from))} height={18} rx={4} className={s.window} opacity={p} /> : null}
        {today ? (
          <g>
            <line x1={x(today.t)} x2={x(today.t)} y1={by - 12} y2={by + 12} className={s.today} />
            <circle cx={x(today.t)} cy={by} r={4} className={s.todayDot} />
            <text x={x(today.t)} y={by + 30} textAnchor="middle" className={s.todayWord}>
              {today.label ?? "Today"}
            </text>
          </g>
        ) : null}
      </svg>
    </div>
  );
}

/* ================================================================== SeaBars */

export type SeaPeriod = { id: string; from: number; to: number; lo: number; hi: number; name: string; text: string; night?: boolean };

export interface SeaBarsProps {
  /** for screen readers */
  label: string;
  domain: [number, number];
  /** NWS periods: a dashed bar each (predicted), its value inside the top; a range's band darker */
  periods: SeaPeriod[];
  /** the one measured mark: a filled dot at its time, named ("Buoy 46014 · now"); its number lives at the buoy on the chart */
  measured?: { t: number; v: number; label: ReactNode; aria: string } | null;
  /** advisories as coral spans over the periods they cover, in NWS's words */
  advisories?: { id: string; from: number; to: number; label: string; level: 1 | 2 }[];
  /** day names under the axis */
  days?: { t: number; label: string; short?: string }[];
  /** the key at the top right ("Seas, ft · NWS forecast") */
  keyText?: string;
  /** feet at the top of the scale (24 by default) */
  max?: number;
  height?: number;
  className?: string;
}

/** Sea bars (v14 README §3, Today's dock): NWS seas per period, the buoy's measured point, advisory spans. Bars rise in 620 ms. */
export function SeaBars({ label, domain, periods, measured, advisories = [], days = [], keyText = "Seas, ft · NWS forecast", max = 24, height = 128, className }: SeaBarsProps) {
  const [ref, w] = useElementWidth<HTMLDivElement>(520);
  const [grow] = useProgress(JSON.stringify(periods.map((q) => [q.lo, q.hi])), seaRise);
  const [hover, setHover] = useState<string | null>(null);
  const narrow = w < 430;
  const x = (t: number) => ((t - domain[0]) / (domain[1] - domain[0])) * w;
  const pt = narrow ? 46 : 40, pb = height - 22;
  const y = (v: number) => pb - (v / max) * (pb - pt) * grow;
  const top = Math.max(...periods.map((q) => q.hi));
  let maxDone = false;
  let capRight = -99;
  const hovered = periods.find((q) => q.id === hover) ?? null;
  return (
    <div ref={ref} className={cx(s.chart, className)} style={{ height }}>
      <svg viewBox={`0 0 ${w} ${height}`} role="img" aria-label={label}>
        {periods
          .filter((q) => q.night)
          .map((q) => (
            <rect key={`n${q.id}`} x={x(q.from)} y={pt - 8} width={x(q.to) - x(q.from)} height={pb - pt + 8} className={s.night} />
          ))}
        {periods.map((q) => {
          const a = x(q.from) + 2, b = x(q.to) - 2, yt = y(q.hi), yl = y(q.lo);
          return (
            <g key={q.id}>
              <rect x={a} y={yt} width={Math.max(0, b - a)} height={Math.max(0, pb - yt)} rx={3} className={s.sea} />
              {q.lo !== q.hi ? <rect x={a} y={yt} width={Math.max(0, b - a)} height={Math.max(0, yl - yt)} rx={3} className={s.seaRange} /> : null}
              <path d={`M${a},${pb}V${yt + 3}q0,-3 3,-3H${b - 3}q3,0 3,3V${pb}`} className={s.seaEdge} />
            </g>
          );
        })}
        <line x1={0} x2={w} y1={pb + 0.5} y2={pb + 0.5} className={s.base} />
        {periods.map((q) => {
          const cxp = (x(q.from) + x(q.to)) / 2, bw = x(q.to) - x(q.from);
          const isMax = q.hi === top && !maxDone;
          if (isMax) maxDone = true;
          const str = q.lo === q.hi ? String(q.hi) : `${q.lo}–${q.hi}`;
          return (
            <text key={`v${q.id}`} x={cxp} y={y(q.hi) + 16} textAnchor="middle" className={cx(s.seaVal, isMax && s.seaValMax, bw < 44 && str.length > 3 && s.seaValSmall)} opacity={grow}>
              {str}
              {isMax && bw > 50 ? (
                <tspan dx={2} className={s.seaUnit}>
                  ft
                </tspan>
              ) : null}
            </text>
          );
        })}
        {advisories.map((a) => {
          const xa = x(a.from), xb = x(a.to);
          const cxa = Math.max(xa, capRight + 10);
          capRight = cxa + a.label.length * 7.5;
          return (
            <g key={a.id}>
              <rect x={xa} y={19} width={Math.max(4, xb - xa - 2) * Math.min(1, grow * 1.4)} height={6} rx={3} className={cx(s.adv, a.level > 1 && s.advStrong)} />
              <text x={cxa} y={12} className={cx(s.cap, s.capCoral)}>
                {a.label}
              </text>
            </g>
          );
        })}
        <g>
          <text x={w} y={w - keyText.length * 7.2 - 26 < capRight + 8 ? 31 : 12} textAnchor="end" className={s.cap}>
            {narrow ? keyText.replace(/^Seas, /, "") : keyText}
          </text>
        </g>
        {days.map((d) => (
          <text key={d.t} x={x(d.t)} y={pb + 16} textAnchor="middle" className={s.tickText}>
            {narrow ? (d.short ?? d.label.slice(0, 3)) : d.label}
          </text>
        ))}
        {periods.map((q) => (
          <rect
            key={`h${q.id}`}
            x={x(q.from)}
            y={pt - 8}
            width={x(q.to) - x(q.from)}
            height={pb - pt + 8}
            className={cx(s.hit, hover === q.id && s.hitOn)}
            tabIndex={0}
            role="img"
            aria-label={`${q.name}: ${q.text}`}
            onPointerEnter={() => setHover(q.id)}
            onPointerLeave={() => setHover(null)}
            onFocus={() => setHover(q.id)}
            onBlur={() => setHover(null)}
          />
        ))}
        {measured
          ? (() => {
              const bx = x(measured.t), byy = pb - (measured.v / max) * (pb - pt);
              const hl = (on: boolean) => document.dispatchEvent(new CustomEvent("nx-buoy-hl", { detail: on }));
              return (
                <g tabIndex={0} role="img" aria-label={measured.aria} onPointerEnter={() => hl(true)} onPointerLeave={() => hl(false)} onFocus={() => hl(true)} onBlur={() => hl(false)} className={s.buoyMark}>
                  <line x1={bx} x2={bx} y1={byy + 7} y2={pb} className={s.buoyStem} />
                  <circle cx={bx} cy={byy} r={5.5} className={s.buoyDot} />
                  <circle cx={bx} cy={byy} r={16} className={s.hit} />
                  <text x={Math.max(0, bx - 4)} y={byy - 10} className={s.buoyWord}>
                    {measured.label}
                  </text>
                </g>
              );
            })()
          : null}
      </svg>
      {hovered ? (
        <div className={s.tip} role="status" style={{ left: Math.max(0, Math.min(w - 260, (x(hovered.from) + x(hovered.to)) / 2 - 130)) }}>
          <b>{hovered.name} · NWS forecast</b>
          {hovered.text}
        </div>
      ) : null}
    </div>
  );
}
