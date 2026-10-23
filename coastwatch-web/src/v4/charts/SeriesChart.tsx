"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { AXES, axisPosition, decadeTicks, onAxis, segments, type Axis } from "@/lib/bloom";
import { unitLabel, type ObsVariableId, type Sample } from "@/lib/observations";
import { dayLabel } from "@/v4/ui/time";

/**
 * A measured series as it was sampled: points at sample dates, lines only between samples less than
 * 21 days apart (a gap is a gap), fixed whole-decade axes that never rescale to the data (a quiet year
 * cannot look like a bloom), reported zeros as a mark on the floor, values beyond the axis as edge
 * markers. Pointer and arrow keys move a readout with each sample's own date and value.
 * Per-sample QC is visible: a value flagged high is ringed (kept as published), a sample rejected by
 * QC is never a value: it is a cross on the floor labelled "rejected (QC)".
 *
 * `cursorT`/`onCursor` make the cursor shared (Research): the chart then shows the sample taken on
 * the cursor's UTC day, or none, and reports the sample the pointer or arrow keys move to.
 */
export function SeriesChart({
  samples,
  variable,
  units,
  label,
  from,
  to,
  height = 168,
  tone = "measured",
  refLine,
  axisTitle,
  cursorT,
  onCursor,
  hideReadout = false,
  between,
}: {
  samples: Sample[];
  variable: ObsVariableId;
  units: string;
  label: string;
  from: number;
  to: number;
  height?: number;
  tone?: "measured" | "model";
  /** a labelled reference value (e.g. C-HARM's own 500 ng/L threshold): context, not a safety line */
  refLine?: { value: number; label: string };
  /** a y-axis title drawn above the plot, e.g. "pDA (ng/mL, log scale)" */
  axisTitle?: string;
  /** a shared cursor (ms); undefined = the chart keeps its own */
  cursorT?: number | null;
  onCursor?: (t: number) => void;
  /** the parent shows the readout (shared cursor) */
  hideReadout?: boolean;
  /** drawn directly under the plot, before the legend and the table (Research: the model on the same time axis) */
  between?: React.ReactNode;
}) {
  const unit = unitLabel(units);
  const axis: Axis = AXES[variable];
  // drawn at its real width, so its text is real 11–12 px type, never a scaled-down picture
  const box = useRef<HTMLDivElement | null>(null);
  const [W, setW] = useState(360);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setW(Math.max(240, Math.round(el.getBoundingClientRect().width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const H = height;
  const pad = { l: 50, r: 10, t: axisTitle ? 26 : 10, b: 24 };
  const iw = W - pad.l - pad.r;
  const ih = H - pad.t - pad.b;
  const x = (t: number) => pad.l + ((t - from) / Math.max(1, to - from)) * iw;
  const y = (v: number) => {
    const [lo, hi] = axis.domain;
    const f = axis.scale === "log" ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);
    return pad.t + ih * (1 - Math.min(1, Math.max(0, f)));
  };
  const inWin = useMemo(() => samples.filter((s) => s.t >= from && s.t <= to), [samples, from, to]);
  const segs = useMemo(() => segments(samples, axis, from, to), [samples, axis, from, to]);
  const ticks = axis.scale === "log" ? decadeTicks(axis.domain).filter((_, i, a) => a.length < 6 || i % 2 === 0) : [axis.domain[0], (axis.domain[0] + axis.domain[1]) / 2, axis.domain[1]];
  const months = useMemo(() => monthTicks(from, to, W), [from, to, W]);
  const [own, setOwn] = useState<number | null>(inWin.length ? inWin.length - 1 : null);
  const controlled = cursorT !== undefined;
  const cur = controlled ? (cursorT == null ? null : sampleOnDay(inWin, cursorT)) : own;
  const pick = (i: number) => (controlled ? onCursor?.(inWin[i].t) : setOwn(i));
  /** the next sample after (dir 1) or before (dir -1) the cursor */
  const step = (dir: 1 | -1) => {
    const base = cur != null ? inWin[cur]?.t : controlled ? cursorT : null;
    if (base == null) return inWin.length && pick(dir > 0 ? 0 : inWin.length - 1);
    const i = dir > 0 ? inWin.findIndex((s) => s.t > base) : inWin.findLastIndex((s) => s.t < base);
    if (i >= 0) pick(i);
  };
  const rejected = inWin.filter((s) => s.q === "rejected_negative");
  const flagged = inWin.filter((s) => s.q === "flag_high" && s.value != null);
  const zeros = inWin.some((s) => s.q === "reported_zero" || s.value === 0);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const draw = useDrawOnce(svgRef);
  const id = useId();
  const color = tone === "model" ? "var(--nx-model)" : "var(--nx-measured)";
  const sel = cur != null ? inWin[cur] : null;
  const fmt = (v: number) => (v >= 1000 ? Math.round(v).toLocaleString("en-US") : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toPrecision(2));
  const tickFmt = (v: number) => (v >= 1000 ? `${v / 1000 >= 1000 ? `${v / 1e6}M` : `${v / 1000}k`}` : String(v));

  const nearest = (clientX: number) => {
    const r = svgRef.current?.getBoundingClientRect();
    if (!r || !inWin.length) return;
    const t = from + ((clientX - r.left) / r.width * W - pad.l) / iw * (to - from);
    let best = 0;
    inWin.forEach((s, i) => Math.abs(s.t - t) < Math.abs(inWin[best].t - t) && (best = i));
    pick(best);
  };

  // a window whose visits never measured this variable is empty too: nothing would be drawn, so say so,
  // and when the newest measurement was
  if (!inWin.some((s) => s.value != null || s.q === "rejected_negative")) {
    const newest = samples.findLast((s) => s.value != null);
    return (
      <p className="rounded-lg border border-dashed border-n-line-strong px-3 py-4 text-[14px] text-n-ink-2" data-testid="series-empty">
        No {label.toLowerCase()} measured in this window
        {newest ? `; the newest measurement is from ${new Date(newest.t).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" })}` : ""}. No sample is not a clean result.
      </p>
    );
  }

  return (
    <figure ref={box} className="flex flex-col gap-2" data-testid={`series-${variable}`}>
      {!hideReadout && (
        <div className="flex min-h-[44px] items-baseline justify-between gap-3" aria-live="polite">
          {sel ? (
            <p className="text-[14px] text-n-ink" data-testid="series-readout">
              <SampleValue s={sel} unit={unit} fmt={fmt} />
              <span className="text-n-ink-3"> · sampled {dayLabel(sel.time, true)}</span>
            </p>
          ) : (
            <span />
          )}
          <span className="text-[12.5px] text-n-ink-3">{inWin.length} samples</span>
        </div>
      )}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        className={`block touch-pan-y select-none ${draw === "wait" ? "nx-chart-wait" : draw === "draw" ? "nx-chart-draw" : ""}`}
        role="img"
        aria-labelledby={`${id}-t`}
        tabIndex={0}
        onPointerMove={(e) => nearest(e.clientX)}
        onPointerDown={(e) => nearest(e.clientX)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") step(-1);
          else if (e.key === "ArrowRight") step(1);
          else return;
          e.preventDefault();
        }}
        data-testid="series-svg"
      >
        <title id={`${id}-t`}>{`${label}, ${unit}, ${axis.scale === "log" ? "log scale" : "linear scale"}: ${inWin.length} samples from ${dayLabel(inWin[0].time)} to ${dayLabel(inWin[inWin.length - 1].time)}. Use the left and right arrow keys to read each sample.`}</title>
        {axisTitle && (
          <text x={4} y={13} fontSize="11.5" fontWeight={600} fill="var(--nx-ink-2)" data-testid="series-axis-title">
            {axisTitle}
          </text>
        )}
        {ticks.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--nx-line)" strokeWidth={1} />
            <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--nx-ink-3)" className="nx-data">
              {tickFmt(v)}
            </text>
          </g>
        ))}
        {months.filter((t) => x(t) <= W - pad.r - 18).map((t) => (
          <text key={t} x={x(t)} y={H - 6} fontSize="11.5" fill="var(--nx-ink-3)" textAnchor="middle">
            {tickLabel(t, from, to)}
          </text>
        ))}
        {refLine && axisPosition(refLine.value, axis) === "in" && (
          <g data-testid="series-ref">
            <line x1={pad.l} x2={W - pad.r} y1={y(refLine.value)} y2={y(refLine.value)} stroke="var(--nx-model)" strokeDasharray="5 4" strokeWidth={1.2} />
            <text x={W - pad.r - 4} y={y(refLine.value) - 5} textAnchor="end" fontSize="11.5" fill="var(--nx-model)">
              {refLine.label}
            </text>
          </g>
        )}
        {segs.map((sg, i) => (
          <polyline key={i} className="nx-chart-line" pathLength={1} points={sg.map((s) => `${x(s.t)},${y(s.value!)}`).join(" ")} fill="none" stroke={color} strokeWidth={1.8} strokeLinejoin="round" />
        ))}
        <g className="nx-chart-marks">
        {inWin.map((s, i) => {
          if (s.q === "rejected_negative") {
            const cx = x(s.t);
            const cy = pad.t + ih - 8;
            return (
              <g key={i} data-testid="series-rejected">
                <title>{`${dayLabel(s.time, true)}: rejected (QC), not a measurement`}</title>
                <path d={`M${cx - 3.5},${cy - 3.5} l7,7 m0,-7 l-7,7`} stroke="var(--nx-down)" strokeWidth={i === cur ? 2.4 : 1.6} strokeLinecap="round" />
                {rejected.length <= 4 && (
                  <text x={cx} y={cy - 7} textAnchor="middle" fontSize="10.5" fill="var(--nx-down)">
                    rejected (QC)
                  </text>
                )}
              </g>
            );
          }
          const zero = s.value === 0 || s.q === "reported_zero";
          if (zero) return <rect key={i} x={x(s.t) - 3} y={pad.t + ih - 3} width={6} height={3} fill="var(--nx-ink-3)" />;
          if (!onAxis(s, axis)) return null;
          const pos = axisPosition(s.value!, axis);
          if (pos !== "in") return <path key={i} d={pos === "above" ? `M${x(s.t) - 4},${pad.t + 6} l4,-6 l4,6z` : `M${x(s.t) - 4},${pad.t + ih - 6} l4,6 l4,-6z`} fill={color} />;
          const dot = <circle cx={x(s.t)} cy={y(s.value!)} r={i === cur ? 5 : 2.6} fill={i === cur ? "var(--nx-surface)" : color} stroke={color} strokeWidth={i === cur ? 2.2 : 0} />;
          if (s.q !== "flag_high") return <g key={i}>{dot}</g>;
          return (
            <g key={i} data-testid="series-flagged">
              <title>{`${dayLabel(s.time, true)}: flagged high by QC, kept as published`}</title>
              {dot}
              <circle cx={x(s.t)} cy={y(s.value!)} r={7.5} fill="none" stroke="var(--nx-down)" strokeWidth={1.4} />
            </g>
          );
        })}
        </g>
        {(sel || (controlled && cursorT != null && cursorT >= from && cursorT <= to)) && (
          <line x1={x(sel?.t ?? cursorT!)} x2={x(sel?.t ?? cursorT!)} y1={pad.t} y2={pad.t + ih} stroke="var(--nx-ink-3)" strokeDasharray="2 3" data-testid="series-cursor" />
        )}
      </svg>
      {between}
      {(zeros || flagged.length > 0 || rejected.length > 0) && (
        <p className="flex flex-wrap gap-x-4 gap-y-1 text-[12.5px] text-n-ink-3" data-testid="series-legend">
          {zeros && <span>Bar on the floor: reported 0 (not quantified)</span>}
          {flagged.length > 0 && <span>Red ring: flagged high by QC, value kept as published ({flagged.length})</span>}
          {rejected.length > 0 && <span>Red cross on the floor: rejected (QC), not a measurement ({rejected.length})</span>}
        </p>
      )}
      <details className="text-[13px] text-n-ink-2">
        <summary className="nx-focus min-h-11 cursor-pointer py-2 font-semibold text-n-accent">Show the samples as a table</summary>
        <div className="max-h-48 overflow-auto">
          <table className="nx-data w-full text-left text-[12.5px]">
            <thead>
              <tr className="text-n-ink-3">
                <th className="py-1 pr-3 font-semibold">Date</th>
                <th className="py-1 pr-3 font-semibold">{unit}</th>
                <th className="py-1 font-semibold">QC</th>
              </tr>
            </thead>
            <tbody>
              {[...inWin].reverse().map((s) => (
                <tr key={s.index} className="border-t border-n-line">
                  <td className="py-1 pr-3">{s.time.slice(0, 10)}</td>
                  <td className="py-1 pr-3">{s.q === "rejected_negative" ? "—" : s.value == null ? "—" : s.q === "reported_zero" ? "0 (reported)" : s.value}</td>
                  <td className="py-1">{s.q === "rejected_negative" ? "rejected (QC)" : s.q === "flag_high" ? "flagged high" : s.q === "reported_zero" ? "reported zero" : s.value == null ? "not measured" : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

/** What one sample says, as the readout's value: a rejected sample is never a number. */
export function SampleValue({ s, unit, fmt, size = "md" }: { s: Sample; unit: string; fmt: (v: number) => string; size?: "md" | "lg" }) {
  // "lg": a sounding (Research's readout), the interface face with tabular figures
  const big = size === "lg" ? "nx-sounding-v" : "nx-data text-[20px] font-semibold";
  if (s.q === "rejected_negative")
    return (
      <>
        <span className={`${big} text-n-down`}>rejected (QC)</span> <span className="text-n-ink-2">not a measurement</span>
      </>
    );
  if (s.value == null) return <span className="text-n-ink-2">a visit, but this was not measured</span>;
  return (
    <>
      <span className={big}>{s.q === "reported_zero" || s.value === 0 ? "0" : fmt(s.value)}</span> <span className="text-n-ink-2">{unit}</span>
      {s.q === "reported_zero" && <span className="text-n-ink-3"> · reported as zero</span>}
      {s.q === "flag_high" && <span className="text-n-down"> · flagged high by QC (kept as published)</span>}
    </>
  );
}

const dayKey = (t: number) => new Date(t).toISOString().slice(0, 10);

/** Index of the sample taken on the same UTC day as `t` (the latest that day), or null. */
export function sampleOnDay(ss: Sample[], t: number): number | null {
  const k = dayKey(t);
  for (let i = ss.length - 1; i >= 0; i--) if (dayKey(ss[i].t) === k) return i;
  return null;
}

/** The plot's own width for a chart drawn `W` px wide (the same 50 px + 10 px margins as SeriesChart). */
const plotWidth = (W: number) => Math.max(1, W - 60);
/** Room one tick label needs: "Sep 25" at 11.5 px is about 38 px, plus a clear gap between labels. */
export const TICK_LABEL_PX = 56;
/** Beyond about three years the ticks are whole years and the labels say only the year. */
const YEARS_FROM_DAYS = 3 * 365;

/**
 * Time-axis ticks inside [from, to], chosen from the pixels available: the smallest step of 1, 2, 3,
 * 4 or 6 months (or 1, 2, 5, 10 years beyond ~3 years) whose ticks stand at least TICK_LABEL_PX apart,
 * so labels never run into each other on a phone or over a long record.
 */
export function monthTicks(from: number, to: number, W: number): number[] {
  const span = Math.max(1, to - from);
  const px = plotWidth(W);
  const years = span / 86_400_000 > YEARS_FROM_DAYS;
  const steps = years ? [12, 24, 60, 120, 240] : [1, 2, 3, 4, 6, 12];
  // a month is ~30.44 days: the smallest step whose shortest gap still leaves room for a label
  const every = steps.find((m) => ((m * 28 * 86_400_000) / span) * px >= TICK_LABEL_PX) ?? steps[steps.length - 1];
  const d = new Date(from);
  d.setUTCDate(1);
  d.setUTCHours(0, 0, 0, 0);
  if (years) d.setUTCMonth(0);
  // steps align to the calendar (Jan, Apr, Jul, Oct for 3 months; whole decades for 10 years)
  if (years) d.setUTCFullYear(Math.floor(d.getUTCFullYear() / (every / 12)) * (every / 12));
  else d.setUTCMonth(Math.floor(d.getUTCMonth() / every) * every);
  const out: number[] = [];
  while (d.getTime() <= to) {
    if (d.getTime() >= from) out.push(d.getTime());
    d.setUTCMonth(d.getUTCMonth() + every);
  }
  return out;
}

/** A tick's label: "Sep 25" (month and two-digit year), or "2025" alone when the axis is in years. */
export function tickLabel(t: number, from: number, to: number): string {
  const years = (to - from) / 86_400_000 > YEARS_FROM_DAYS;
  return new Date(t).toLocaleDateString("en-US", years ? { year: "numeric", timeZone: "UTC" } : { month: "short", year: "2-digit", timeZone: "UTC" });
}

/** StationLite.recent pairs → samples */
export function recentSamples(pairs: [string, number | null][] | undefined): Sample[] {
  return (pairs ?? []).map(([d, v], i) => ({ t: Date.parse(`${d}T12:00:00Z`), time: d, value: v, q: v === 0 ? "reported_zero" : null, index: i }));
}

/**
 * A chart's line draws itself once, the first time it comes into view (a chart already on the screen
 * when the page opens is left as drawn). Never again: not on a new window, a hover or new data.
 */
function useDrawOnce(ref: React.RefObject<SVGSVGElement | null>): "idle" | "wait" | "draw" | "done" {
  const [phase, setPhase] = useState<"idle" | "wait" | "draw" | "done">("idle");
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.top < window.innerHeight && r.bottom > 0) return;
    let t = 0;
    const io = new IntersectionObserver(
      (es) => {
        if (!es.some((e) => e.isIntersecting)) return setPhase("wait");
        io.disconnect();
        setPhase("draw");
        t = window.setTimeout(() => setPhase("done"), 700);
      },
      { threshold: 0.25 },
    );
    io.observe(el);
    return () => {
      io.disconnect();
      window.clearTimeout(t);
    };
    // once per chart
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return phase;
}
