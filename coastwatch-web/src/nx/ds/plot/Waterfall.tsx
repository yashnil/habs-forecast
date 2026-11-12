"use client";
import { motion, useReducedMotion } from "motion/react";
import { createContext, useContext, useState, type KeyboardEvent, type ReactNode } from "react";
import { usd } from "../format";
import { settle } from "../motion";
import { useElementWidth, usePhone } from "../useMedia";
import { cx } from "../util";
import { layoutWaterfall, type WaterfallStep } from "../worksheet/waterfallLayout";
import { linearTicks } from "./model";
import s from "./plot.module.css";

type WfGeo = { left: number; right: number; top: number; bottom: number; y: (v: number) => number; domain: [number, number] };
const Wf = createContext<WfGeo | null>(null);

export interface WaterfallChartProps {
  steps: WaterfallStep[];
  /** Accessible name: "Gross to net to boat, hypothetical, from your numbers". */
  label: string;
  /** A scenario previewed against the same axis (a hovered sensitivity cell): drawn as outlines. */
  preview?: WaterfallStep[] | null;
  /** A money axis held from outside, so a previewed scenario moves bars instead of rescaling them. */
  domain?: [number, number];
  /** Plot height in px, or one for wide screens and one for phones. */
  height?: number | { desktop: number; phone: number };
  /** Money to the cent in the callout. */
  cents?: boolean;
  /** Marks drawn on the same axis (BreakEvenMarker). */
  children?: ReactNode;
  className?: string;
  "data-testid"?: string;
}

/**
 * Gross to net as tall bars on a money axis: gross and net solid ink, each cost a lighter solid hanging from
 * where the last ended (never hatched: hatch is the model mark). One tab stop; Left/Right move bar to bar and
 * each says its amount and how it was worked out in a callout (and a live region). Bars move with `settle` when
 * a figure changes; a step that cannot be worked out says what it waits for and draws nothing, never 0.
 */
export function WaterfallChart({ steps, label, preview, domain: fixed, height = { desktop: 360, phone: 260 }, cents, children, className, ...rest }: WaterfallChartProps) {
  const reduce = useReducedMotion();
  const phone = usePhone();
  const [setEl, width] = useElementWidth<HTMLDivElement>(640);
  const [at, setAt] = useState<number | null>(null);
  const H = typeof height === "number" ? height : phone ? height.phone : height.desktop;
  const own = layoutWaterfall(steps);
  const ghost = preview ? layoutWaterfall(preview) : null;
  const raw = fixed ?? [Math.min(own.domain[0], ghost?.domain[0] ?? 0, 0), Math.max(own.domain[1], ghost?.domain[1] ?? 0, 1)];
  const ticks = linearTicks(raw as [number, number], 4);
  const lo = Math.min(raw[0], ticks[0]), hi = Math.max(raw[1], ticks[ticks.length - 1]);
  const left = 64, right = Math.max(left + 80, width - 8), top = 28, bottom = top + H;
  const y = (v: number) => Math.round((bottom - ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo || 1)) * (bottom - top)) * 100) / 100;
  const band = (right - left) / Math.max(1, steps.length);
  const bw = Math.min(96, band * 0.62);
  const cx0 = (i: number) => left + band * (i + 0.5);
  const t = reduce ? { duration: 0 } : settle;
  const money = (v: number) => usd(cents ? Math.round(v * 100) / 100 : Math.round(v), { cents });
  const amount = (st: WaterfallStep) => (st.amount === null ? null : st.kind === "cost" ? -st.amount : st.amount);
  const say = (i: number) => {
    const st = steps[i];
    const a = amount(st);
    return a === null ? `${st.label}: ${st.waiting ?? "waiting"}.` : `${st.label}: ${money(a)}.${st.formula ? ` ${st.formula}.` : ""}`;
  };
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (d) {
      e.preventDefault();
      setAt((v) => Math.min(steps.length - 1, Math.max(0, (v ?? (d > 0 ? -1 : steps.length)) + d)));
    } else if (e.key === "Home") {
      e.preventDefault();
      setAt(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setAt(steps.length - 1);
    } else if (e.key === "Escape" && at !== null) {
      e.preventDefault();
      setAt(null);
    }
  };
  const focusStep = at !== null ? steps[at] : null;
  return (
    <Wf.Provider value={{ left, right, top, bottom, y, domain: [lo, hi] }}>
      <div ref={setEl} className={cx(s.plot, s.waterfall, className)} role="group" aria-roledescription="chart" aria-label={label} tabIndex={0} onKeyDown={key} onBlur={() => setAt(null)} data-testid={rest["data-testid"]}>
        <svg className={s.svg} width={width} height={bottom + 56} viewBox={`0 0 ${width} ${bottom + 56}`} aria-hidden="true" focusable="false">
          {ticks.map((v) => (
            <g key={v}>
              <line x1={left} x2={right} y1={y(v)} y2={y(v)} className={v === 0 ? s.axisLine : s.grid} />
              <text x={left - 8} y={y(v)} dy="0.32em" textAnchor="end" className={s.tickLabel}>
                {usd(v)}
              </text>
            </g>
          ))}
          {steps.map((st, i) => {
            const b = own.bars[i];
            const g = ghost?.bars[i];
            const x = cx0(i) - bw / 2;
            const a = amount(st);
            return (
              <g key={st.id} data-step={st.id} data-kind={st.kind} data-on={at === i || undefined} onPointerEnter={() => setAt(i)} onPointerLeave={() => setAt((v) => (v === i ? null : v))}>
                <rect x={left + band * i} y={top - 24} width={band} height={bottom - top + 80} className={s.wfHit} />
                {!b.waiting && b.joinAt !== null && i > 0 ? <motion.line x1={cx0(i - 1) + bw / 2} x2={x} initial={false} animate={{ y1: y(b.joinAt), y2: y(b.joinAt) }} transition={t} className={s.wfJoin} /> : null}
                {!b.waiting ? (
                  <motion.rect x={x} width={bw} initial={false} animate={{ y: y(b.to), height: Math.max(1.5, y(b.from) - y(b.to)) }} transition={t} className={st.kind === "cost" ? s.wfCost : s.wfSolid} />
                ) : (
                  <text x={cx0(i)} y={(top + bottom) / 2} textAnchor="middle" className={s.wfWaiting}>
                    {st.waiting ?? "waiting"}
                  </text>
                )}
                {g && !g.waiting ? <motion.rect x={x - 3} width={bw + 6} initial={false} animate={{ y: y(g.to), height: Math.max(1.5, y(g.from) - y(g.to)) }} transition={t} className={s.wfGhost} /> : null}
                {a !== null && !b.waiting ? (
                  <motion.text x={cx0(i)} textAnchor="middle" initial={false} animate={{ y: y(b.to) - 8 }} transition={t} className={cx(s.wfAmount, st.kind === "net" && s.wfAmountNet)}>
                    {usd(Math.round(a))}
                  </motion.text>
                ) : null}
                {at === i ? <rect x={x - 6} y={top - 4} width={bw + 12} height={bottom - top + 8} className={s.wfFocus} /> : null}
                <text x={cx0(i)} y={bottom + 20} textAnchor="middle" className={cx(s.wfLabel, st.kind === "net" && s.wfLabelNet)}>
                  {st.label}
                </text>
              </g>
            );
          })}
          {children}
        </svg>
        {focusStep ? (
          <div className={s.wfCallout} style={at! > steps.length / 2 ? { right: Math.max(0, width - cx0(at!) + bw / 2 + 12) } : { left: cx0(at!) + bw / 2 + 12 }} aria-hidden="true" data-testid="waterfall-callout">
            <span className={s.tipValue}>
              {focusStep.label}
              {amount(focusStep) !== null ? ` ${money(amount(focusStep)!)}` : ""}
            </span>
            <span className={s.tipMeta}>{focusStep.formula ?? focusStep.waiting ?? (focusStep.kind === "cost" ? "Taken off" : focusStep.kind === "net" ? "What is left" : "Catch × price")}</span>
          </div>
        ) : null}
        <p className={s.vh} role="status" aria-live="polite">
          {at !== null ? say(at) : ""}
        </p>
        <ol className={s.vh}>
          {steps.map((_, i) => (
            <li key={steps[i].id}>{say(i)}</li>
          ))}
        </ol>
      </div>
    </Wf.Provider>
  );
}

/** The break-even rule on a waterfall: where net to boat is zero, named. Drawn only when zero is on the axis. */
export function BreakEvenMarker({ label = "Break-even" }: { label?: string }) {
  const g = useContext(Wf);
  if (!g || g.domain[0] > 0 || g.domain[1] < 0) return null;
  const y = g.y(0);
  return (
    <g data-break-even="" pointerEvents="none">
      <line x1={g.left} x2={g.right} y1={y} y2={y} className={s.threshold} />
      <text x={g.right} y={y - 6} textAnchor="end" className={s.thresholdLabel}>
        {label}
      </text>
    </g>
  );
}
