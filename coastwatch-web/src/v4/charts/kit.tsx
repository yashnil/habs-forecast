"use client";

import { useId, useRef, type ReactNode } from "react";
import { useDrawOnce } from "@/v4/motion/draw";

/**
 * The chart kit (design.md §5, §7.5): the pieces every hand-drawn SVG chart uses, so charts read as one
 * family. Time runs past-left, future-right with now marked; ticks in mono ink-3; gridlines only where asked;
 * no chart borders. Direct labels, not legends. Colours keep their meaning: official amber, model violet,
 * measured teal, conditions sky.
 *
 * Everything takes coordinates in the chart's own px space (see `scaleLinear`). A chart draws in once
 * through ChartFrame; numbers and labels never animate.
 */
export type Family = "official" | "model" | "measured" | "conditions" | "ink";
export const FAMILY: Record<Family, string> = {
  official: "var(--nx-k-official)",
  model: "var(--nx-k-model)",
  measured: "var(--nx-k-measured)",
  conditions: "var(--nx-k-conditions)",
  ink: "var(--nx-ink-2)",
};

/* ---- pure helpers ---- */

export type Scale = (v: number) => number;
export function scaleLinear([d0, d1]: readonly [number, number], [r0, r1]: readonly [number, number]): Scale {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

const DAY = 86_400_000;
const STEPS = [DAY / 4, DAY / 2, DAY, 2 * DAY, 7 * DAY, 14 * DAY, 30 * DAY, 91 * DAY, 182 * DAY, 365 * DAY, 730 * DAY, 3652 * DAY];

/** Tick times inside [t0, t1], at round UTC multiples, at most `max`. Never invents a tick outside the domain. */
export function timeTicks([t0, t1]: readonly [number, number], max = 6): number[] {
  if (!(t1 > t0)) return [];
  const span = t1 - t0;
  const step = STEPS.find((s) => span / s <= max) ?? STEPS[STEPS.length - 1];
  const out: number[] = [];
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) out.push(t);
  return out;
}

/**
 * Where to put direct labels at the ends of series: each label as near its own y as it can be, none closer than `gap`.
 * Returns ys in the input's order; a pure pass (two sweeps), so the same data places the same way.
 */
export function placeLabels(ys: readonly number[], gap = 14, [lo, hi]: readonly [number, number] = [-Infinity, Infinity]): number[] {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y);
  const out = order.map((o) => o.y);
  for (let k = 1; k < out.length; k++) out[k] = Math.max(out[k], out[k - 1] + gap);
  if (out.length) out[out.length - 1] = Math.min(out[out.length - 1], hi);
  for (let k = out.length - 2; k >= 0; k--) out[k] = Math.min(out[k], out[k + 1] - gap);
  if (out.length) out[0] = Math.max(out[0], lo);
  for (let k = 1; k < out.length; k++) out[k] = Math.max(out[k], out[k - 1] + gap);
  const res = new Array<number>(ys.length);
  order.forEach((o, k) => (res[o.i] = out[k]));
  return res;
}

const shortDay = (t: number) => new Date(t).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });

/* ---- the frame ---- */

/**
 * The chart's root: an SVG with its text alternative (`title` and `desc` are required, not decoration) and a
 * once-only draw-in. Children use ChartLine, ChartArea, ChartBars and ChartMarks (or the .nx-draw-* classes).
 */
export function ChartFrame({ width, height, title, desc, draw = true, className = "", testid, children }: { width: number; height: number; title: string; desc: string; draw?: boolean; className?: string; testid?: string; children: ReactNode }) {
  const ref = useRef<SVGSVGElement | null>(null);
  const phase = useDrawOnce(ref);
  const id = useId();
  return (
    <svg ref={ref} viewBox={`0 0 ${width} ${height}`} className={`nx-k ${className}`} role="img" aria-labelledby={`${id}-t ${id}-d`} data-draw={draw && phase !== "idle" ? phase : undefined} data-testid={testid}>
      <title id={`${id}-t`}>{title}</title>
      <desc id={`${id}-d`}>{desc}</desc>
      {children}
    </svg>
  );
}

/* ---- axis, now, band, readout ---- */

/**
 * The shared time axis: past left of now, future right. Ticks in mono ink-3. `gridTo` draws a light vertical line
 * from each tick up to that y (only where a grid is asked for: decades, say).
 */
export function TimeAxis({ x, domain, y, ticks, format = shortDay, gridTo, max = 6 }: { x: Scale; domain: readonly [number, number]; y: number; ticks?: readonly { t: number; label: string }[]; format?: (t: number) => string; gridTo?: number; max?: number }) {
  const list = ticks ?? timeTicks(domain, max).map((t) => ({ t, label: format(t) }));
  return (
    <g className="nx-k-axis" aria-hidden>
      {list.map(({ t, label }) => (
        <g key={t}>
          {gridTo != null ? <line className="nx-k-grid" x1={x(t)} x2={x(t)} y1={gridTo} y2={y} /> : null}
          <text x={x(t)} y={y + 14} textAnchor="middle">
            {label}
          </text>
        </g>
      ))}
    </g>
  );
}

/** "Now" on a time axis: a thin line from `y0` to `y1`, labelled at the top. */
export function NowLine({ x, y0, y1, label = "now" }: { x: number; y0: number; y1: number; label?: string }) {
  return (
    <g className="nx-k-now" aria-hidden>
      <line x1={x} x2={x} y1={y0} y2={y1} />
      <text x={x} y={y0 - 4} textAnchor="middle">
        {label}
      </text>
    </g>
  );
}

/**
 * A span of time or a level across its valid span (an advisory, an action level), labelled directly.
 * It draws only where it is valid. `line` (a level): a single line at y0 instead of a filled span.
 */
export function Band({ x0, x1, y0, y1, family = "official", label, line = false, labelX, labelY, anchor = "start" }: { x0: number; x1: number; y0: number; y1: number; family?: Family; label?: ReactNode; line?: boolean; labelX?: number; labelY?: number; anchor?: "start" | "middle" | "end" }) {
  const c = FAMILY[family];
  return (
    <g className="nx-k-band" aria-hidden>
      {line ? <line x1={x0} x2={x1} y1={y0} y2={y0} stroke={c} className="nx-k-level" /> : <rect x={x0} y={Math.min(y0, y1)} width={Math.max(0, x1 - x0)} height={Math.abs(y1 - y0)} fill={c} className="nx-k-span" />}
      {label != null ? (
        <text x={labelX ?? x0 + 4} y={labelY ?? Math.min(y0, y1) + 12} textAnchor={anchor} fill={c}>
          {label}
        </text>
      ) : null}
    </g>
  );
}

/**
 * The crosshair's reading, one style everywhere: a thin vertical at `x`, with the reading beside it. It flips to
 * the left of the crosshair when it would run off `width`. Values are the chart's own, never interpolated.
 */
export function Readout({ x, y0, y1, width, lines, boxWidth = 132 }: { x: number; y0: number; y1: number; width: number; lines: readonly { text: string; family?: Family }[]; boxWidth?: number }) {
  const flip = x + boxWidth + 10 > width;
  const bx = flip ? x - boxWidth - 8 : x + 8;
  return (
    <g className="nx-k-read" aria-hidden>
      <line x1={x} x2={x} y1={y0} y2={y1} />
      <rect x={bx} y={y0} width={boxWidth} height={lines.length * 16 + 8} rx={6} />
      {lines.map((l, i) => (
        <text key={i} x={bx + 8} y={y0 + 16 + i * 16} fill={l.family ? FAMILY[l.family] : undefined}>
          {l.text}
        </text>
      ))}
    </g>
  );
}

/* ---- the marks (each draws in once inside a ChartFrame) ---- */

/** A series line. Solid lines draw left to right; `dash` (an outlook or a model) wipes in, keeping its pattern. */
export function ChartLine({ d, family, dash, width }: { d: string; family: Family; dash?: string; width?: number }) {
  const style = { stroke: FAMILY[family], strokeWidth: width ?? "var(--nx-k-stroke)" };
  if (dash) return <path d={d} className="nx-draw-wipe nx-k-line" style={style} strokeDasharray={dash} />;
  return <path d={d} pathLength={1} className="nx-draw-line nx-k-line" style={style} />;
}
/** An area under a line; grows from the baseline. */
export function ChartArea({ d, family, opacity = 0.14 }: { d: string; family: Family; opacity?: number }) {
  return <path d={d} className="nx-draw-area" style={{ fill: FAMILY[family], fillOpacity: opacity }} />;
}
/** Bars; each grows from `baseline`. */
export function ChartBars({ bars, family, baseline, width }: { bars: readonly { x: number; y: number }[]; family: Family; baseline: number; width: number }) {
  return (
    <g>
      {bars.map((b, i) => (
        <rect key={i} className="nx-draw-bar" x={b.x - width / 2} y={b.y} width={width} height={Math.max(0, baseline - b.y)} rx={1} style={{ fill: FAMILY[family] }} />
      ))}
    </g>
  );
}
/** Points, flags and end labels that appear after the line. */
export function ChartMarks({ children }: { children: ReactNode }) {
  return <g className="nx-draw-marks">{children}</g>;
}
/** A label set directly at the data instead of a legend. */
export function DirectLabel({ x, y, family = "ink", anchor = "start", children }: { x: number; y: number; family?: Family; anchor?: "start" | "middle" | "end"; children: ReactNode }) {
  return (
    <text className="nx-k-direct" x={x} y={y} textAnchor={anchor} fill={FAMILY[family]}>
      {children}
    </text>
  );
}
