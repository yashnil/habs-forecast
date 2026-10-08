"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { formatDate } from "@/lib/time";

export type Point = { date: string; value: number | null; n?: number };

type Props = {
  points: Point[];
  /** y domain; for log scale both bounds must be > 0 */
  domain: [number, number];
  scale?: "linear" | "log";
  format: (v: number) => string;
  unit: string;
  color: string;
  /** consecutive points further apart than this (days) are not connected */
  maxGapDays?: number;
  height?: number;
  label: string;
  /** show a marker for the period a highlighted date represents (e.g. current valid day) */
  highlightDate?: string | null;
  ticks?: number[];
};

const DAY = 86_400_000;
const parse = (d: string) => Date.parse(`${d}T00:00:00Z`);

/**
 * Single-series time chart: thin 2px line, recessive grid, direct label on the last value,
 * gaps where data are missing (never interpolated), hover tooltip.
 */
export function TimeSeriesChart({ points, domain, scale = "linear", format, unit, color, maxGapDays = 1, height = 132, label, highlightDate, ticks }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(320);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(200, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const pad = { l: 40, r: 44, t: 10, b: 22 };
  const valid = points.filter((p) => p.value != null);
  const t0 = points.length ? parse(points[0].date) : 0;
  const t1 = points.length ? parse(points[points.length - 1].date) : 1;
  const x = (d: string) => pad.l + ((parse(d) - t0) / Math.max(DAY, t1 - t0)) * (width - pad.l - pad.r);
  const [lo, hi] = domain;
  const y = (v: number) => {
    const f = scale === "log" ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);
    return pad.t + (1 - Math.min(1, Math.max(0, f))) * (height - pad.t - pad.b);
  };
  const yTicks = ticks ?? (scale === "log" ? [0.1, 1, 10] : [lo, (lo + hi) / 2, hi]);

  const paths = useMemo(() => {
    const segs: string[] = [];
    let cur = "";
    let prev: Point | null = null;
    for (const p of points) {
      if (p.value == null) {
        if (cur) segs.push(cur);
        cur = "";
        prev = null;
        continue;
      }
      const gap = prev ? (parse(p.date) - parse(prev.date)) / DAY : 0;
      if (prev && gap > maxGapDays) {
        segs.push(cur);
        cur = "";
      }
      cur += `${cur ? "L" : "M"}${x(p.date).toFixed(1)},${y(p.value).toFixed(1)}`;
      prev = p;
    }
    if (cur) segs.push(cur);
    return segs;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [points, width, height, lo, hi, scale, maxGapDays]);

  if (!valid.length) {
    return (
      <div ref={ref} className="flex h-[92px] items-center justify-center rounded-md border border-dashed border-hairline-strong text-[12px] text-ink-3" data-testid="chart-empty">
        No values in this period
      </div>
    );
  }

  const last = valid[valid.length - 1];
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left;
    let best = 0;
    let bd = Infinity;
    points.forEach((p, i) => {
      const d = Math.abs(x(p.date) - px);
      if (d < bd) {
        bd = d;
        best = i;
      }
    });
    setHover(best);
  };
  const hp = hover != null ? points[hover] : null;
  const missing = points.length - valid.length;

  return (
    <figure ref={ref} className="relative" aria-label={label} data-testid="timeseries">
      <svg width={width} height={height} onPointerMove={onMove} onPointerLeave={() => setHover(null)} role="img" aria-label={`${label}. Latest ${format(last.value!)} on ${last.date}.`}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={width - pad.r} y1={y(t)} y2={y(t)} stroke="var(--cw-hairline)" />
            <text x={pad.l - 6} y={y(t) + 3.5} textAnchor="end" className="fill-[var(--cw-ink-3)] text-[10px] tabular">
              {format(t)}
            </text>
          </g>
        ))}
        {highlightDate && parse(highlightDate) >= t0 && parse(highlightDate) <= t1 && (
          <line x1={x(highlightDate)} x2={x(highlightDate)} y1={pad.t} y2={height - pad.b} stroke="var(--cw-hairline-strong)" strokeDasharray="2 3" />
        )}
        {paths.map((d, i) => (
          <path key={i} d={d} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        ))}
        {valid.length < 40 &&
          valid.map((p) => <circle key={p.date} cx={x(p.date)} cy={y(p.value!)} r={1.8} fill={color} />)}
        <circle cx={x(last.date)} cy={y(last.value!)} r={3.5} fill={color} stroke="var(--cw-surface)" strokeWidth={2} />
        <text x={x(last.date) + 7} y={y(last.value!) + 3.5} className="fill-[var(--cw-ink)] text-[11px] font-semibold tabular">
          {format(last.value!)}
        </text>
        {[points[0], points[points.length - 1]].map((p, i) => (
          <text key={i} x={x(p.date)} y={height - 6} textAnchor={i ? "end" : "start"} className="fill-[var(--cw-ink-3)] text-[10px]">
            {formatDate(p.date).replace(/^\w+, /, "")}
          </text>
        ))}
        {hp && (
          <line x1={x(hp.date)} x2={x(hp.date)} y1={pad.t} y2={height - pad.b} stroke="var(--cw-ink-3)" strokeWidth={1} />
        )}
      </svg>
      {hp && (
        <div
          className="pointer-events-none absolute top-0 z-10 rounded-md border border-hairline-strong bg-surface-3 px-2 py-1 text-[11px] text-ink shadow-lg"
          style={{ left: Math.min(width - 150, Math.max(0, x(hp.date) - 70)) }}
        >
          <div className="text-ink-3">{formatDate(hp.date, { year: true })}</div>
          <div className="font-semibold tabular">{hp.value == null ? "no value (missing)" : `${format(hp.value)} ${unit}`}</div>
          {hp.n != null && <div className="text-ink-3">{hp.n} cells</div>}
        </div>
      )}
      <figcaption className="mt-0.5 flex justify-between text-[10.5px] text-ink-3">
        <span>{unit}</span>
        <span>{missing ? `${missing} day${missing > 1 ? "s" : ""} without a value` : ""}</span>
      </figcaption>
    </figure>
  );
}
