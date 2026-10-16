"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { AXES, axisPosition, decadeTicks, onAxis, segments, type Axis } from "@/lib/bloom";
import type { ObsVariableId, Sample } from "@/lib/observations";
import { dayLabel } from "@/v4/ui/time";

/**
 * A measured series as it was sampled: points at sample dates, lines only between samples less than
 * 21 days apart (a gap is a gap), fixed whole-decade axes that never rescale to the data (a quiet year
 * cannot look like a bloom), reported zeros as a mark on the floor, values beyond the axis as edge
 * markers. Pointer and arrow keys move a readout with each sample's own date and value.
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
}) {
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
  const pad = { l: 44, r: 10, t: 10, b: 24 };
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
  const months = useMemo(() => {
    const out: number[] = [];
    const d = new Date(from);
    d.setUTCDate(1);
    d.setUTCHours(0, 0, 0, 0);
    while (d.getTime() <= to) {
      if (d.getTime() >= from) out.push(d.getTime());
      d.setUTCMonth(d.getUTCMonth() + (W < 420 ? 3 : 2));
    }
    return out;
  }, [from, to, W]);
  const [cur, setCur] = useState<number | null>(inWin.length ? inWin.length - 1 : null);
  const svgRef = useRef<SVGSVGElement | null>(null);
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
    setCur(best);
  };

  if (!inWin.length)
    return (
      <p className="rounded-lg border border-dashed border-n-line-strong px-3 py-4 text-[14px] text-n-ink-2" data-testid="series-empty">
        No {label.toLowerCase()} samples in this window. No sample is not a clean result.
      </p>
    );

  return (
    <figure ref={box} className="flex flex-col gap-2" data-testid={`series-${variable}`}>
      <div className="flex min-h-[44px] items-baseline justify-between gap-3" aria-live="polite">
        {sel ? (
          <p className="text-[14px] text-n-ink">
            <span className="nx-data text-[20px] font-semibold">{sel.value == null ? "—" : sel.q === "reported_zero" || sel.value === 0 ? "0" : fmt(sel.value)}</span>{" "}
            <span className="text-n-ink-2">{units}</span>
            <span className="text-n-ink-3"> · sampled {dayLabel(sel.time, true)}{sel.q === "reported_zero" ? " · reported as zero" : ""}</span>
          </p>
        ) : (
          <span />
        )}
        <span className="text-[12.5px] text-n-ink-3">{inWin.length} samples</span>
      </div>
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        width={W}
        height={H}
        className="block touch-pan-y select-none"
        role="img"
        aria-labelledby={`${id}-t`}
        tabIndex={0}
        onPointerMove={(e) => nearest(e.clientX)}
        onPointerDown={(e) => nearest(e.clientX)}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") setCur((c) => Math.max(0, (c ?? 0) - 1));
          else if (e.key === "ArrowRight") setCur((c) => Math.min(inWin.length - 1, (c ?? 0) + 1));
          else return;
          e.preventDefault();
        }}
        data-testid="series-svg"
      >
        <title id={`${id}-t`}>{`${label}, ${units}, ${axis.scale === "log" ? "log scale" : "linear scale"}: ${inWin.length} samples from ${dayLabel(inWin[0].time)} to ${dayLabel(inWin[inWin.length - 1].time)}. Use the left and right arrow keys to read each sample.`}</title>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--nx-line)" strokeWidth={1} />
            <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--nx-ink-3)" className="nx-data">
              {tickFmt(v)}
            </text>
          </g>
        ))}
        {months.map((t) => (
          <text key={t} x={x(t)} y={H - 6} fontSize="11.5" fill="var(--nx-ink-3)" textAnchor="middle">
            {new Date(t).toLocaleDateString("en-US", { month: "short", year: "2-digit", timeZone: "UTC" })}
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
          <polyline key={i} points={sg.map((s) => `${x(s.t)},${y(s.value!)}`).join(" ")} fill="none" stroke={color} strokeWidth={1.8} strokeLinejoin="round" />
        ))}
        {inWin.map((s, i) => {
          const zero = s.value === 0 || s.q === "reported_zero";
          if (zero) return <rect key={i} x={x(s.t) - 3} y={pad.t + ih - 3} width={6} height={3} fill="var(--nx-ink-3)" />;
          if (!onAxis(s, axis)) return null;
          const pos = axisPosition(s.value!, axis);
          if (pos !== "in") return <path key={i} d={pos === "above" ? `M${x(s.t) - 4},${pad.t + 6} l4,-6 l4,6z` : `M${x(s.t) - 4},${pad.t + ih - 6} l4,6 l4,-6z`} fill={color} />;
          return <circle key={i} cx={x(s.t)} cy={y(s.value!)} r={i === cur ? 5 : 2.6} fill={i === cur ? "var(--nx-surface)" : color} stroke={color} strokeWidth={i === cur ? 2.2 : 0} />;
        })}
        {sel && <line x1={x(sel.t)} x2={x(sel.t)} y1={pad.t} y2={pad.t + ih} stroke="var(--nx-ink-3)" strokeDasharray="2 3" />}
      </svg>
      <details className="text-[13px] text-n-ink-2">
        <summary className="nx-focus min-h-11 cursor-pointer py-2 font-semibold text-n-accent">Show the samples as a table</summary>
        <div className="max-h-48 overflow-auto">
          <table className="nx-data w-full text-left text-[12.5px]">
            <thead>
              <tr className="text-n-ink-3">
                <th className="py-1 pr-3 font-semibold">Date</th>
                <th className="py-1 font-semibold">{units}</th>
              </tr>
            </thead>
            <tbody>
              {[...inWin].reverse().map((s) => (
                <tr key={s.index} className="border-t border-n-line">
                  <td className="py-1 pr-3">{s.time.slice(0, 10)}</td>
                  <td className="py-1">{s.value == null ? "—" : s.q === "reported_zero" ? "0 (reported)" : s.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}

/** StationLite.recent pairs → samples */
export function recentSamples(pairs: [string, number | null][] | undefined): Sample[] {
  return (pairs ?? []).map(([d, v], i) => ({ t: Date.parse(`${d}T12:00:00Z`), time: d, value: v, q: v === 0 ? "reported_zero" : null, index: i }));
}
