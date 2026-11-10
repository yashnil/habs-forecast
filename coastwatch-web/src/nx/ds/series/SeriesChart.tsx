"use client";
import { scaleLinear, scaleLog, scaleUtc } from "d3-scale";
import { line as d3line } from "d3-shape";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useId, useMemo, useState, type KeyboardEvent, type PointerEvent, type ReactNode } from "react";
import { ms as toMs, type SeriesPoint, classify, describeSeries, logDomain, logTickLabel, logTicks, nearestIndex, readoutText, segments } from "./series";
import { KIND_WORD, type Kind } from "../kinds";
import { ink } from "../motion";
import { useElementWidth } from "../useMedia";
import { cx, svgId } from "../util";
import s from "./series.module.css";

/** Coordinates rounded to 0.01 px, so server and client render the same attribute strings. */
const r2 = (v: number) => Math.round(v * 100) / 100;

export interface SeriesChartProps {
  points: SeriesPoint[];
  /** What is plotted: "Particulate domoic acid". */
  label: string;
  unit: string;
  kind?: Kind;
  scale?: "log" | "linear";
  yDomain?: [number, number];
  /** Time window in ms; defaults to the data's span. */
  xDomain?: [number, number];
  /** Plot height in px (the censored strip and axis are added). */
  height?: number;
  marks?: "points" | "line" | "both";
  /** Samples further apart than this break the line. */
  gapDays?: number;
  /** A labelled reference line ("Action level 30 ppm"). Only on the scale it belongs to (T6). */
  threshold?: { value: number; label: string };
  /** Draws the now divider; samples after it are on the future side (T2). */
  now?: string;
  /** Words for the censored strip; defaults from the flags present. */
  censoredLabel?: string;
  /** Hover shared across small multiples (ms), and its setter. */
  hoverT?: number | null;
  onHoverT?: (t: number | null) => void;
  /** The crosshair readout line above the plot. */
  showReadout?: boolean;
  xAxis?: boolean;
  /** Small-multiple heading (label and unit). */
  title?: ReactNode;
  /** Draws the line in with `ink` the first time it appears. */
  inkIn?: boolean;
  className?: string;
}

const M = { left: 84, right: 12, top: 10 };
const STRIP = 26;
const AXIS = 26;
const utcYear = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", year: "numeric" });
const utcMonth = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short" });
const utcDay = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", month: "short", day: "numeric" });

function xTickLabel(d: Date, spanDays: number): string {
  if (spanDays > 3 * 365) return utcYear.format(d);
  if (spanDays <= 120) return utcDay.format(d);
  return d.getUTCMonth() === 0 ? utcYear.format(d) : utcMonth.format(d);
}

/**
 * A measured record as SVG (d3-scale/d3-shape maths, our own marks). Censored samples sit in their own
 * strip under the axis, never at 0 or at the limit (T7); long gaps break the line (T19). The crosshair
 * reads one sample at a time by pointer or arrow keys.
 */
export function SeriesChart({
  points,
  label,
  unit,
  kind = "measured",
  scale = "log",
  yDomain,
  xDomain,
  height = 220,
  marks = "points",
  gapDays = 45,
  threshold,
  now,
  censoredLabel,
  hoverT,
  onHoverT,
  showReadout = true,
  xAxis = true,
  title,
  inkIn = true,
  className,
}: SeriesChartProps) {
  const [ref, width] = useElementWidth<HTMLDivElement>(640);
  const reduce = useReducedMotion();
  const descId = svgId(useId());
  const clipId = svgId(useId());
  const [localHover, setLocalHover] = useState<number | null>(null);
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const hollow = kind !== "measured";
  const kindWord = KIND_WORD[kind];

  const samples = useMemo(() => points.filter((p) => classify(p) !== "missing").sort((a, b) => toMs(a.t) - toMs(b.t)), [points]);
  const span: [number, number] = xDomain ?? (samples.length ? [toMs(samples[0].t), toMs(samples[samples.length - 1].t)] : [0, 1]);
  const visible = samples.filter((p) => toMs(p.t) >= span[0] && toMs(p.t) <= span[1]);
  const values = visible.filter((p) => classify(p) === "value").map((p) => p.v as number);
  const censored = visible.filter((p) => classify(p) === "censored");
  const hasStrip = samples.some((p) => classify(p) === "censored");

  const yd: [number, number] = yDomain ?? (scale === "log" ? logDomain(values) : [Math.min(0, ...values), Math.max(1, ...values) * 1.08]);
  const plotW = Math.max(120, width - M.left - M.right);
  const plotBottom = M.top + height;
  const stripY = plotBottom + STRIP / 2 + 2;
  const totalH = plotBottom + (hasStrip ? STRIP : 0) + (xAxis ? AXIS : 6);

  const xs = scaleUtc().domain([new Date(span[0]), new Date(span[1])]).range([M.left, M.left + plotW]);
  const x = Object.assign((d: Date) => r2(xs(d)), { invert: (px: number) => xs.invert(px), ticks: (n: number) => xs.ticks(n) });
  const ys = scale === "log" ? scaleLog().domain(yd).range([plotBottom, M.top]).clamp(true) : scaleLinear().domain(yd).range([plotBottom, M.top]).nice();
  const y = (v: number) => r2(ys(v));
  const yTicks = scale === "log" ? logTicks(yd) : (ys as ReturnType<typeof scaleLinear<number, number>>).ticks(5);
  const spanDays = (span[1] - span[0]) / 86_400_000;
  const xTicks = x.ticks(spanDays > 3 * 365 ? Math.max(2, Math.min(Math.round(spanDays / 365), Math.floor(plotW / 64))) : Math.max(2, Math.round(plotW / 110)));

  const gen = d3line<SeriesPoint>()
    .x((p) => x(new Date(toMs(p.t))))
    .y((p) => y(p.v as number));
  const runs = segments(visible, gapDays);

  const activeT = hoverT !== undefined ? hoverT : localHover;
  const activeIdx = activeT === null || activeT === undefined ? -1 : nearestIndex(visible, activeT);
  const active = activeIdx >= 0 ? visible[activeIdx] : null;
  const setT = (t: number | null) => {
    setLocalHover(t);
    onHoverT?.(t);
  };

  const fromPointer = (e: PointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    if (px < M.left - 8 || !visible.length) return setT(null);
    const t = x.invert(px).getTime();
    const i = nearestIndex(visible, t);
    setT(i >= 0 ? toMs(visible[i].t) : null);
  };
  const onKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!visible.length) return;
    const i = activeIdx < 0 ? (e.key === "ArrowLeft" || e.key === "End" ? visible.length : -1) : activeIdx;
    let n = i;
    if (e.key === "ArrowRight") n = Math.min(visible.length - 1, i + 1);
    else if (e.key === "ArrowLeft") n = Math.max(0, i - 1);
    else if (e.key === "Home") n = 0;
    else if (e.key === "End") n = visible.length - 1;
    else if (e.key === "Escape") return setT(null);
    else return;
    e.preventDefault();
    setT(toMs(visible[n].t));
  };

  const nowX = now ? x(new Date(toMs(now))) : null;
  // the initial frame is the same on server and client; reduced motion only shortens the draw to nothing
  const animate = inkIn && !mounted;
  const pointMarks = marks !== "line";
  const lineMarks = marks !== "points";
  const stripWords = censoredLabel ?? (censored.some((p) => p.flag !== "reported0") || samples.some((p) => p.flag && p.flag !== "reported0") ? "Below limit" : "Reported 0");
  const readout = active ? readoutText(active, unit, kindWord) : null;

  return (
    <figure className={cx(s.series, className)}>
      {title ? <figcaption className={s.head}>{title}</figcaption> : null}
      {showReadout ? (
        <p className={s.readout} aria-live="polite">
          {readout ?? <span className={s.hint}>{visible.length ? "Point at a mark to read it." : "Nothing to read in this window."}</span>}
        </p>
      ) : null}
      <div
        ref={ref}
        className={s.plot}
        tabIndex={0}
        role="group"
        aria-label={`${label}, ${unit}. ${showReadout ? "Arrow keys read each sample." : readout ?? "Arrow keys read each sample."}`}
        aria-describedby={descId}
        onPointerMove={fromPointer}
        onPointerDown={fromPointer}
        onPointerLeave={(e) => (e.pointerType === "mouse" ? setT(null) : undefined)}
        onKeyDown={onKey}
        onBlur={() => setT(null)}
      >
        <svg width={width} height={totalH} className={s.svg} aria-hidden focusable="false">
          <defs>
            <clipPath id={clipId}>
              <rect x={M.left} y={M.top - 6} width={plotW} height={height + 12} />
            </clipPath>
          </defs>
          {/* grid */}
          {yTicks.map((v) => (
            <g key={`y${v}`}>
              <line x1={M.left} x2={M.left + plotW} y1={y(v)} y2={y(v)} className={s.grid} />
              <text x={M.left - 8} y={y(v)} className={s.yTick} dy="0.34em">
                {scale === "log" ? logTickLabel(v) : v.toLocaleString("en-US")}
              </text>
            </g>
          ))}
          {xTicks.map((d) => (
            <line key={`x${d.getTime()}`} x1={x(d)} x2={x(d)} y1={M.top} y2={plotBottom} className={s.grid} />
          ))}
          <line x1={M.left} x2={M.left + plotW} y1={plotBottom} y2={plotBottom} className={s.baseline} />
          {threshold ? (
            <g>
              <line x1={M.left} x2={M.left + plotW} y1={y(threshold.value)} y2={y(threshold.value)} className={s.threshold} />
              <text x={M.left + plotW} y={y(threshold.value) - 6} className={s.thresholdLabel}>
                {threshold.label}
              </text>
            </g>
          ) : null}
          {nowX !== null && nowX >= M.left && nowX <= M.left + plotW ? (
            <g>
              <line x1={nowX} x2={nowX} y1={M.top - 4} y2={plotBottom + (hasStrip ? STRIP : 0)} className={s.now} />
              <text x={nowX > M.left + plotW - 40 ? nowX - 4 : nowX + 4} y={M.top + 10} className={s.nowLabel} textAnchor={nowX > M.left + plotW - 40 ? "end" : "start"}>
                Now
              </text>
            </g>
          ) : null}
          {/* the record */}
          <g clipPath={`url(#${clipId})`}>
            {lineMarks
              ? runs.map((run) =>
                  run.length > 1 ? (
                    <motion.path
                      key={`${run[0].t}-${run.length}`}
                      d={gen(run) ?? undefined}
                      className={cx(s.line, hollow && s.lineModel)}
                      initial={animate ? { pathLength: 0 } : false}
                      animate={{ pathLength: 1 }}
                      transition={reduce ? { duration: 0 } : ink(1.4)}
                    />
                  ) : null,
                )
              : null}
            {pointMarks || lineMarks
              ? visible
                  .filter((p) => classify(p) === "value" && (pointMarks || runs.some((r) => r.length === 1 && r[0] === p)))
                  .map((p) => {
                    const cx_ = x(new Date(toMs(p.t)));
                    const cy = y(p.v as number);
                    return hollow ? (
                      <rect key={p.t} x={cx_ - 3} y={cy - 3} width={6} height={6} className={s.ptHollow} />
                    ) : (
                      <rect key={p.t} x={cx_ - 2.75} y={cy - 2.75} width={5.5} height={5.5} className={s.pt} />
                    );
                  })
              : null}
          </g>
          {visible.length === 0 ? (
            <text x={M.left + plotW / 2} y={M.top + height / 2} className={s.emptyText}>
              No samples in this window
            </text>
          ) : null}
          {/* censored strip: its own mark, its own words */}
          {hasStrip ? (
            <g>
              <text x={M.left - 8} y={stripY} className={s.yTick} dy="0.34em">
                {stripWords}
              </text>
              <line x1={M.left} x2={M.left + plotW} y1={plotBottom + STRIP} y2={plotBottom + STRIP} className={s.grid} />
              {censored.map((p) => (
                <rect key={`c${p.t}`} x={x(new Date(toMs(p.t))) - 2.75} y={stripY - 2.75} width={5.5} height={5.5} className={s.ptCensored} />
              ))}
            </g>
          ) : null}
          {xAxis
            ? xTicks.map((d) => (
                <text key={`xl${d.getTime()}`} x={x(d)} y={totalH - 6} className={s.xTick}>
                  {xTickLabel(d, spanDays)}
                </text>
              ))
            : null}
          {/* crosshair */}
          {active ? (
            <g className={s.cross}>
              <line x1={x(new Date(toMs(active.t)))} x2={x(new Date(toMs(active.t)))} y1={M.top - 4} y2={plotBottom + (hasStrip ? STRIP : 0)} className={s.crossLine} />
              {classify(active) === "value" ? (
                <rect x={x(new Date(toMs(active.t))) - 5} y={y(active.v as number) - 5} width={10} height={10} className={hollow ? s.crossHollow : s.crossMark} />
              ) : (
                <rect x={x(new Date(toMs(active.t))) - 5} y={stripY - 5} width={10} height={10} className={s.crossHollow} />
              )}
            </g>
          ) : null}
        </svg>
      </div>
      <p id={descId} className={s.vh}>
        {describeSeries(visible, label)}
      </p>
    </figure>
  );
}
