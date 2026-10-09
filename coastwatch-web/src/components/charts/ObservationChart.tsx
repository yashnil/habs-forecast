"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { ObsVariable } from "@/generated/observations";
import { DAY, coverage, formatObs, formatTick, unitLabel, type Sample } from "@/lib/observations";

const PAD = { l: 48, r: 14, t: 10 };
const LANE = 14; // reported-zero / rejected lane under the plot
const STRIP = 8; // sampling strip
const AXIS = 18;
const CONNECT_DAYS = 21; // samples further apart are not joined by a line
/** Round plot coordinates so server and browser math (e.g. Math.log10) render identical markup. */
const r1 = (v: number) => Math.round(v * 10) / 10;

function useWidth<T extends HTMLElement>(): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(300);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export function timeTicks(x0: number, x1: number): { t: number; label: string }[] {
  const spanDays = (x1 - x0) / DAY;
  const step = spanDays <= 420 ? 2 : spanDays <= 1300 ? 6 : 12;
  const d = new Date(x0);
  const out: { t: number; label: string }[] = [];
  for (let k = 1; ; k++) {
    const t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + k, 1); // Date.UTC rolls months over
    if (t > x1) break;
    const dt = new Date(t);
    const mm = dt.getUTCMonth();
    if (mm % step !== 0) continue;
    const yy = dt.getUTCFullYear();
    const mon = dt.toLocaleDateString("en-US", { month: "short", timeZone: "UTC" });
    out.push({ t, label: step === 12 || mm === 0 ? String(yy) : step === 6 ? `${mon} ${String(yy).slice(2)}` : mon });
  }
  return out;
}

export function fmtDay(t: number, year = true): string {
  return new Date(t).toLocaleDateString("en-US", { month: "short", day: "numeric", ...(year ? { year: "numeric" } : {}), timeZone: "UTC" });
}

type XProps = { x0: number; x1: number; hoverT: number | null; onHoverT: (t: number | null) => void };

type Props = XProps & {
  variable: ObsVariable;
  samples: Sample[];
  scale: "log" | "linear";
  yDomain: [number, number];
  color: string;
  height?: number;
  testid?: string;
  title: ReactNode;
};

/**
 * One measured quantity over time. Each dot is one laboratory value. Reported zeros and
 * rejected values sit in a separate lane under the axis (a log axis has no zero, and a
 * reported 0 is "not quantified", not a low value). The strip shows every sampling visit;
 * faint ticks are visits where this quantity was not measured.
 */
export function ObservationChart({ variable, samples, x0, x1, scale, yDomain, color, hoverT, onHoverT, height = 150, testid, title }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [tableOpen, setTableOpen] = useState(false);
  const win = useMemo(() => samples.filter((s) => s.t >= x0 && s.t <= x1), [samples, x0, x1]);
  const measured = win.filter((s) => s.value != null);
  const cov = coverage(samples, x0, x1);
  const plotH = height - PAD.t - LANE - STRIP - AXIS;
  const laneY = PAD.t + plotH + LANE / 2 + 1;
  const stripY = PAD.t + plotH + LANE + 2;
  const [lo, hi] = yDomain;
  const x = (t: number) => r1(PAD.l + ((t - x0) / Math.max(DAY, x1 - x0)) * (width - PAD.l - PAD.r));
  const y = (v: number) => {
    const f = scale === "log" ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);
    return r1(PAD.t + (1 - Math.min(1, Math.max(0, f))) * plotH);
  };
  const onAxis = (s: Sample) => s.value != null && s.q !== "reported_zero" && (scale === "linear" || s.value > 0);
  const yTicks = scale === "log" ? decades(lo, hi) : [lo, (lo + hi) / 2, hi];

  const paths = useMemo(() => {
    const segs: string[] = [];
    let cur = "";
    let prev: Sample | null = null;
    for (const s of win) {
      if (s.value == null) continue; // a visit without this measurement does not break or join the line
      if (!onAxis(s)) {
        if (cur) segs.push(cur);
        cur = "";
        prev = null;
        continue;
      }
      if (prev && (s.t - prev.t) / DAY > CONNECT_DAYS) {
        segs.push(cur);
        cur = "";
      }
      cur += `${cur ? "L" : "M"}${x(s.t).toFixed(1)},${y(s.value).toFixed(1)}`;
      prev = s;
    }
    if (cur) segs.push(cur);
    return segs;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [win, width, lo, hi, scale, x0, x1]);

  // the sample of this chart closest to the shared hover time (within 4 days)
  const hovered = useMemo(() => {
    if (hoverT == null || !win.length) return null;
    let best: Sample | null = null;
    for (const s of win) if (!best || Math.abs(s.t - hoverT) < Math.abs(best.t - hoverT)) best = s;
    return best && Math.abs(best.t - hoverT) <= 4 * DAY ? best : null;
  }, [hoverT, win]);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const t = x0 + ((e.clientX - r.left - PAD.l) / (width - PAD.l - PAD.r)) * (x1 - x0);
    if (t < x0 || t > x1) return onHoverT(null);
    let best: Sample | null = null;
    for (const s of win) if (!best || Math.abs(s.t - t) < Math.abs(best.t - t)) best = s;
    onHoverT(best ? best.t : t);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (!win.length) return;
    const i = hovered ? win.indexOf(hovered) : -1;
    let next: Sample | null = null;
    if (e.key === "ArrowRight") next = win[Math.min(win.length - 1, i + 1)];
    else if (e.key === "ArrowLeft") next = win[Math.max(0, i < 0 ? win.length - 1 : i - 1)];
    else if (e.key === "Home") next = win[0];
    else if (e.key === "End") next = win[win.length - 1];
    else if (e.key === "Escape") return onHoverT(null);
    else return;
    e.preventDefault();
    onHoverT(next.t);
  };

  const last = [...measured].reverse()[0] ?? null;
  const hasZero = measured.some((s) => s.q === "reported_zero");
  const hasRejected = win.some((s) => s.q === "rejected_negative");
  const hasFlag = measured.some((s) => s.q === "flag_high");
  const summary = last
    ? `${variable.label}: ${cov.measured} measurements between ${fmtDay(x0)} and ${fmtDay(x1)}; latest ${formatObs(variable, last.value, last.q)} on ${fmtDay(last.t)}.`
    : `${variable.label}: not measured between ${fmtDay(x0)} and ${fmtDay(x1)}.`;

  return (
    <figure ref={ref} className="min-w-0 space-y-1" data-testid={testid} data-measured={cov.measured} data-visits={cov.visits}>
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-3 text-[12.5px]">
        <span className="font-semibold text-ink">{title}</span>
        <span className="text-[11px] text-ink-3">{unitLabel(variable.units)} · {scale === "log" ? "log scale" : "linear scale"}</span>
      </figcaption>
      <div className="relative">
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={summary}
          tabIndex={0}
          onKeyDown={onKey}
          onPointerMove={onMove}
          onPointerLeave={() => onHoverT(null)}
          onBlur={() => onHoverT(null)}
          className="rounded-sm"
          data-testid={testid ? `${testid}-svg` : undefined}
        >
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={width - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--cw-hairline)" />
              <text x={PAD.l - 6} y={y(t) + 3.5} textAnchor="end" className="fill-[var(--cw-ink-3)] text-[10px] tabular">
                {scale === "log" ? formatTick(t) : t.toFixed(0)}
              </text>
            </g>
          ))}
          {/* reported-zero lane and sampling strip labels */}
          <text x={PAD.l - 6} y={laneY + 3.5} textAnchor="end" className="fill-[var(--cw-ink-3)] text-[9.5px]">
            0*
          </text>
          <line x1={PAD.l} x2={width - PAD.r} y1={PAD.t + plotH + 1} y2={PAD.t + plotH + 1} stroke="var(--cw-hairline-strong)" />
          {timeTicks(x0, x1).map((tk) => (
            <g key={tk.t}>
              <line x1={x(tk.t)} x2={x(tk.t)} y1={PAD.t} y2={stripY + STRIP} stroke="var(--cw-hairline)" />
              <text x={x(tk.t)} y={height - 4} textAnchor="middle" className="fill-[var(--cw-ink-3)] text-[10px]">
                {tk.label}
              </text>
            </g>
          ))}
          {paths.map((d, i) => (
            <path key={i} d={d} fill="none" stroke={color} strokeWidth={1.5} strokeOpacity={0.75} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {measured.filter(onAxis).map((s) => (
            <circle
              key={s.index}
              cx={x(s.t)}
              cy={y(s.value!)}
              r={win.length > 260 ? 1.9 : 2.6}
              fill={color}
              stroke={s.q === "flag_high" ? "var(--cw-warning)" : "none"}
              strokeWidth={s.q === "flag_high" ? 1.5 : 0}
            />
          ))}
          {measured
            .filter((s) => s.q === "reported_zero")
            .map((s) => (
              <circle key={s.index} cx={x(s.t)} cy={laneY} r={2.4} fill="none" stroke={color} strokeWidth={1.2} data-q="reported_zero" />
            ))}
          {win
            .filter((s) => s.q === "rejected_negative")
            .map((s) => (
              <path key={s.index} d={`M${x(s.t) - 2.5},${laneY - 2.5}l5,5m0,-5l-5,5`} stroke="var(--cw-ink-3)" strokeWidth={1.3} data-q="rejected" />
            ))}
          {/* every sampling visit: strong = measured, faint = not measured in that visit */}
          {win.map((s) => (
            <line
              key={s.index}
              x1={x(s.t)}
              x2={x(s.t)}
              y1={stripY}
              y2={stripY + STRIP}
              stroke={s.value != null ? "var(--cw-ink-2)" : "var(--cw-hairline-strong)"}
              strokeWidth={1}
            />
          ))}
          {hoverT != null && hoverT >= x0 && hoverT <= x1 && (
            <line x1={x(hoverT)} x2={x(hoverT)} y1={PAD.t} y2={stripY + STRIP} stroke="var(--cw-ink-3)" strokeWidth={1} />
          )}
          {hovered && onAxis(hovered) && <circle cx={x(hovered.t)} cy={y(hovered.value!)} r={4.5} fill={color} stroke="var(--cw-surface)" strokeWidth={2} />}
          {last && onAxis(last) && !hovered && (
            <circle cx={x(last.t)} cy={y(last.value!)} r={4} fill={color} stroke="var(--cw-surface)" strokeWidth={2} />
          )}
        </svg>
        {hovered && (
          <div
            className="pointer-events-none absolute top-0 z-10 rounded-md border border-hairline-strong bg-surface-3 px-2 py-1 text-[11px] text-ink shadow-lg"
            style={{ left: Math.min(width - 190, Math.max(0, x(hovered.t) - 90)) }}
            data-testid={testid ? `${testid}-tooltip` : undefined}
          >
            <div className="text-ink-3">{fmtDay(hovered.t)}</div>
            <div className="font-semibold tabular">{formatObs(variable, hovered.value, hovered.q)}</div>
            {hovered.q === "flag_high" && <div className="text-warning">Above plausibility check — kept, flagged for review</div>}
          </div>
        )}
      </div>
      <p className="sr-only" aria-live="polite">
        {hovered ? `${fmtDay(hovered.t)}: ${formatObs(variable, hovered.value, hovered.q)}` : ""}
      </p>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10.5px] text-ink-3" data-testid={testid ? `${testid}-legend` : undefined}>
        <span className="inline-flex items-center gap-1">
          <svg width="8" height="8" aria-hidden>
            <circle cx="4" cy="4" r="3" fill={color} />
          </svg>
          measured
        </span>
        {hasZero && (
          <span className="inline-flex items-center gap-1">
            <svg width="8" height="8" aria-hidden>
              <circle cx="4" cy="4" r="2.8" fill="none" stroke={color} strokeWidth="1.2" />
            </svg>
            0* reported 0 (not quantified, not absent)
          </span>
        )}
        {hasRejected && <span>× rejected by QC (negative)</span>}
        {hasFlag && <span className="text-warning">○ above plausibility check</span>}
        <span>
          ticks: sampling visits{cov.visits > cov.measured ? " (faint = not measured that visit)" : ""}
        </span>
      </div>
      <p className="text-[11px] text-ink-2" data-testid={testid ? `${testid}-coverage` : undefined}>
        {cov.measured === 0 ? (
          <span className="font-medium text-ink">Not measured in this period. No measurement is not the same as no toxin or no cells.</span>
        ) : (
          <>
            Measured in <span className="tabular text-ink">{cov.measured}</span> of <span className="tabular">{cov.visits}</span> sampling visits
            {cov.medianGapDays != null && <> · typical interval {Math.round(cov.medianGapDays)} days</>}
            {cov.longestGapDays != null && cov.longestGapDays > 28 && <> · longest gap {Math.round(cov.longestGapDays)} days</>}
            {cov.zeros > 0 && <> · {cov.zeros} reported 0</>}
            {cov.rejected > 0 && <> · {cov.rejected} rejected</>}
          </>
        )}
      </p>
      <details className="text-[11px] text-ink-3" onToggle={(e) => setTableOpen((e.target as HTMLDetailsElement).open)}>
        <summary className="cursor-pointer">Data table</summary>
        {tableOpen && (
          <div className="mt-1 max-h-56 overflow-y-auto rounded border border-hairline">
            <table className="w-full text-left text-[11px]">
              <caption className="sr-only">{variable.label} samples in the selected period</caption>
              <thead className="sticky top-0 bg-surface-2 text-ink-3">
                <tr>
                  <th className="px-2 py-1 font-medium">Sample time (UTC)</th>
                  <th className="px-2 py-1 font-medium">{variable.label}</th>
                </tr>
              </thead>
              <tbody>
                {[...win].reverse().map((s) => (
                  <tr key={s.index} className="border-t border-hairline">
                    <td className="px-2 py-0.5 tabular">{s.time.replace("T", " ").replace("Z", "")}</td>
                    <td className="px-2 py-0.5 tabular text-ink-2">{formatObs(variable, s.value, s.q)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </details>
    </figure>
  );
}

function decades(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let e = Math.round(Math.log10(lo)); e <= Math.round(Math.log10(hi)); e++) out.push(10 ** e);
  // keep labels readable on short axes
  return out.length > 6 ? out.filter((_, i) => i % 2 === 0) : out;
}

type TrackPoint = { date: string; value: number | null; n: number };

/**
 * C-HARM nowcast probability near the station on the same time axis as the measurements.
 * A model probability, a different quantity: it has its own axis and is never plotted
 * on, or numerically compared with, a measurement axis.
 */
export function ModelTrack({ points, x0, x1, hoverT, onHoverT, color, height = 112, testid, title }: XProps & { points: TrackPoint[]; color: string; height?: number; testid?: string; title: ReactNode }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const plotH = height - PAD.t - AXIS;
  const x = (t: number) => r1(PAD.l + ((t - x0) / Math.max(DAY, x1 - x0)) * (width - PAD.l - PAD.r));
  const y = (v: number) => r1(PAD.t + (1 - Math.min(1, Math.max(0, v))) * plotH);
  const pts = points.map((p) => ({ ...p, t: Date.parse(`${p.date}T12:00:00Z`) })).filter((p) => p.t >= x0 && p.t <= x1);
  const first = points.length ? Date.parse(`${points[0].date}T12:00:00Z`) : null;
  const segs: string[] = [];
  let cur = "";
  let prev: number | null = null;
  for (const p of pts) {
    if (p.value == null) {
      if (cur) segs.push(cur);
      cur = "";
      prev = null;
      continue;
    }
    if (prev != null && (p.t - prev) / DAY > 1.5) {
      segs.push(cur);
      cur = "";
    }
    cur += `${cur ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.value).toFixed(1)}`;
    prev = p.t;
  }
  if (cur) segs.push(cur);
  const hovered = hoverT == null ? null : (pts.reduce<(typeof pts)[number] | null>((b, p) => (!b || Math.abs(p.t - hoverT) < Math.abs(b.t - hoverT) ? p : b), null) ?? null);
  const hv = hovered && hoverT != null && Math.abs(hovered.t - hoverT) <= 1.5 * DAY ? hovered : null;
  const valid = pts.filter((p) => p.value != null);
  const lastV = valid.at(-1) ?? null;
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const t = x0 + ((e.clientX - r.left - PAD.l) / (width - PAD.l - PAD.r)) * (x1 - x0);
    onHoverT(t < x0 || t > x1 ? null : t);
  };
  const missingDays = pts.filter((p) => p.value == null).length;
  const label = lastV
    ? `Model probability near the station, ${valid.length} daily nowcasts; latest ${Math.round(lastV.value! * 100)}% on ${lastV.date}.`
    : "No model values near this station in this period.";

  return (
    <figure ref={ref} className="min-w-0 space-y-1" data-testid={testid}>
      <figcaption className="flex flex-wrap items-baseline justify-between gap-x-3 text-[12.5px]">
        <span className="font-semibold text-ink">{title}</span>
        <span className="text-[11px] text-ink-3">probability 0–100% · model, not a measurement</span>
      </figcaption>
      <div className="relative">
        <svg width={width} height={height} role="img" aria-label={label} onPointerMove={onMove} onPointerLeave={() => onHoverT(null)}>
          {first != null && first > x0 && (
            <g>
              <rect x={PAD.l} y={PAD.t} width={Math.max(0, x(first) - PAD.l)} height={plotH} fill="url(#cw-hatch)" />
              {x(first) - PAD.l > 120 && (
                <text x={(PAD.l + x(first)) / 2} y={PAD.t + plotH / 2 + 3} textAnchor="middle" className="fill-[var(--cw-ink-3)] text-[10.5px]">
                  model history not kept before {fmtDay(first, false)}
                </text>
              )}
            </g>
          )}
          <defs>
            <pattern id="cw-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--cw-hairline)" strokeWidth="2" />
            </pattern>
          </defs>
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={width - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--cw-hairline)" />
              <text x={PAD.l - 6} y={y(t) + 3.5} textAnchor="end" className="fill-[var(--cw-ink-3)] text-[10px] tabular">
                {t * 100}%
              </text>
            </g>
          ))}
          {timeTicks(x0, x1).map((tk) => (
            <g key={tk.t}>
              <line x1={x(tk.t)} x2={x(tk.t)} y1={PAD.t} y2={PAD.t + plotH} stroke="var(--cw-hairline)" />
              <text x={x(tk.t)} y={height - 4} textAnchor="middle" className="fill-[var(--cw-ink-3)] text-[10px]">
                {tk.label}
              </text>
            </g>
          ))}
          {segs.map((d, i) => (
            <path key={i} d={d} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          ))}
          {hoverT != null && hoverT >= x0 && hoverT <= x1 && <line x1={x(hoverT)} x2={x(hoverT)} y1={PAD.t} y2={PAD.t + plotH} stroke="var(--cw-ink-3)" />}
          {hv?.value != null && <circle cx={x(hv.t)} cy={y(hv.value)} r={4} fill={color} stroke="var(--cw-surface)" strokeWidth={2} />}
        </svg>
        {hv && (
          <div
            className="pointer-events-none absolute top-0 z-10 rounded-md border border-hairline-strong bg-surface-3 px-2 py-1 text-[11px] text-ink shadow-lg"
            style={{ left: Math.min(width - 190, Math.max(0, x(hv.t) - 90)) }}
          >
            <div className="text-ink-3">Nowcast valid {fmtDay(hv.t)}</div>
            <div className="font-semibold tabular">{hv.value == null ? "no model value (missing run)" : `${Math.round(hv.value * 100)}% probability`}</div>
            {hv.value != null && <div className="text-ink-3">median of {hv.n} model cells</div>}
          </div>
        )}
      </div>
      <p className="text-[11px] text-ink-3">
        {valid.length} daily nowcasts{missingDays ? ` · ${missingDays} day${missingDays > 1 ? "s" : ""} without a published run (left blank)` : ""}
      </p>
    </figure>
  );
}
