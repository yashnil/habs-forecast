"use client";

import { useMemo, useState, type ReactNode } from "react";
import type { ObsVariable, SeriesPoint } from "@/generated/observations";
import { axisPosition, decadeTicks, highestInView, inView, modelOnAxis, modelSegments, onAxis, segments, viewSummary, type Axis, type ModelPoint } from "@/lib/bloom";
import { DAY, formatObs, formatTick, unitLabel, type Sample } from "@/lib/observations";
import { useWidth } from "@/lib/useWidth";
import { Mark } from "@/components/ui/Mark";

/**
 * The page's one primary chart (design spec §3.2 item 6, M6 §0): the measured record of one
 * quantity at one pier, with the C-HARM model band directly beneath it on the same time axis.
 *
 * - Fixed whole-decade axes (§5.2); the range changes the x window only.
 * - Reported zeros live in their own lane under the plot, labelled with their count; they are
 *   "not quantified", never a low value, and never on the log axis.
 * - Every sampling visit has a tick; a faint tick is a visit where this quantity was not measured.
 * - Lines join samples at most 21 days apart. Nothing is smoothed, interpolated or averaged.
 * - The model band is a separate panel with its own 0–100 % axis. It shares only the time axis and
 *   the crosshair: a probability is a different quantity and is never compared numerically.
 */

const PAD = { l: 52, r: 18 };
const PLOT_T = 26; // room for the "highest in view" label
const LANE = 22;
const STRIP = 12;
const AXIS = 22;
const BAND = 104;
const r1 = (v: number) => Math.round(v * 10) / 10;
const PT = "America/Los_Angeles";

// one formatter per option set, reused: toLocaleDateString builds a new one on every call
const DATE_FMT = new Map<string, Intl.DateTimeFormat>();
export const fmtDate = (t: number, opts: Intl.DateTimeFormatOptions = { month: "short", day: "numeric", year: "numeric" }) => {
  const key = JSON.stringify(opts);
  let f = DATE_FMT.get(key);
  if (!f) DATE_FMT.set(key, (f = new Intl.DateTimeFormat("en-US", { ...opts, timeZone: PT })));
  return f.format(new Date(t));
};

export function timeTicks(x0: number, x1: number): { t: number; label: string }[] {
  const spanDays = (x1 - x0) / DAY;
  const step = spanDays <= 120 ? 1 : spanDays <= 420 ? 2 : spanDays <= 1300 ? 6 : 12;
  const d = new Date(x0);
  const out: { t: number; label: string }[] = [];
  for (let k = 1; ; k++) {
    const t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + k, 1);
    if (t > x1) break;
    const dt = new Date(t);
    const mm = dt.getUTCMonth();
    if (mm % step !== 0) continue;
    const yy = dt.getUTCFullYear();
    const mon = dt.toLocaleDateString("en-US", { month: "short", timeZone: "UTC" });
    out.push({ t, label: step === 12 || mm === 0 ? String(yy) : step === 6 ? `${mon} ’${String(yy).slice(2)}` : mon });
  }
  return out;
}

export type ModelInput = {
  variable: string;
  label: string;
  title: string;
  history: SeriesPoint[] | undefined;
  error: string | null | undefined;
  radiusKm: number;
  nearestCellKm: number | null | undefined;
  controls: ReactNode;
};

type Props = {
  variable: ObsVariable;
  samples: Sample[];
  /** every sampling visit at the station (for the tick strip) */
  visits: number[];
  axis: Axis;
  x0: number;
  x1: number;
  title: ReactNode;
  note?: ReactNode;
  controls: ReactNode;
  model: ModelInput;
  plotHeight?: number;
  /** shown over the plot when nothing was measured in view (e.g. a button to widen the range) */
  empty?: ReactNode;
  /** official notices for waters near this pier, drawn as amber rules at their effective dates */
  notices?: ChartNotice[];
};

/** An official notice on the time axis: the agency's own effective date, never inferred. */
export type ChartNotice = { id: string; t: number; label: string };

export function StationChart({ variable, samples, visits, axis, x0, x1, title, note, controls, model, plotHeight = 300, empty, notices = [] }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hoverT, setHoverT] = useState<number | null>(null);
  const [tableOpen, setTableOpen] = useState(false);
  const narrow = width < 560;
  const plotH = narrow ? Math.round(plotHeight * 0.75) : plotHeight;
  const laneY = PLOT_T + plotH + LANE / 2 + 2;
  const stripY = PLOT_T + plotH + LANE + 4;
  const H = stripY + STRIP + AXIS;
  const iw = width - PAD.l - PAD.r;
  const x = (t: number) => r1(PAD.l + ((t - x0) / Math.max(DAY, x1 - x0)) * iw);
  const [lo, hi] = axis.domain;
  const y = (v: number) => {
    const f = axis.scale === "log" ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);
    return r1(PLOT_T + (1 - Math.min(1, Math.max(0, f))) * plotH);
  };
  const yTicks = axis.scale === "log" ? decadeTicks(axis.domain) : [5, 10, 15, 20, 25, 30];

  const win = useMemo(() => samples.filter((s) => inView(s, x0, x1)), [samples, x0, x1]);
  const segs = useMemo(() => segments(samples, axis, x0, x1), [samples, axis, x0, x1]);
  const top = useMemo(() => highestInView(samples, axis, x0, x1), [samples, axis, x0, x1]);
  const sum = useMemo(() => viewSummary(samples, x0, x1), [samples, x0, x1]);
  const m = useMemo(() => modelOnAxis(model.history, x0, x1), [model.history, x0, x1]);
  const mSegs = useMemo(() => modelSegments(m.points), [m.points]);
  const visitsIn = useMemo(() => visits.filter((t) => t >= x0 && t <= x1), [visits, x0, x1]);
  const measuredAt = useMemo(() => new Set(win.filter((s) => s.value != null).map((s) => s.t)), [win]);

  // the crosshair snaps to the nearest sampling visit of this quantity (within 6 days), else to a model day
  const hovered = useMemo(() => {
    if (hoverT == null) return null;
    let best: Sample | null = null;
    for (const s of win) if (!best || Math.abs(s.t - hoverT) < Math.abs(best.t - hoverT)) best = s;
    return best && Math.abs(best.t - hoverT) <= 6 * DAY ? best : null;
  }, [hoverT, win]);
  const hoveredModel = useMemo(() => {
    if (hoverT == null) return null;
    const t = hovered?.t ?? hoverT;
    let best: ModelPoint | null = null;
    for (const p of m.points) if (!best || Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
    return best && Math.abs(best.t - t) <= 0.75 * DAY ? best : null;
  }, [hoverT, hovered, m.points]);
  const crossT = hovered?.t ?? hoveredModel?.t ?? null;

  const tFromEvent = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    return x0 + ((e.clientX - r.left - PAD.l) / iw) * (x1 - x0);
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const t = tFromEvent(e);
    setHoverT(t < x0 || t > x1 ? null : t);
  };
  // a mouse leaving clears the readout; a finger lifting keeps it, so a tap reads a value on touch screens
  const onLeave = (e: React.PointerEvent) => {
    if (e.pointerType === "mouse") setHoverT(null);
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (!win.length) return;
    const i = hovered ? win.indexOf(hovered) : -1;
    let next: Sample | null = null;
    if (e.key === "ArrowRight") next = win[Math.min(win.length - 1, i + 1)];
    else if (e.key === "ArrowLeft") next = win[Math.max(0, i < 0 ? win.length - 1 : i - 1)];
    else if (e.key === "Home") next = win[0];
    else if (e.key === "End") next = win[win.length - 1];
    else if (e.key === "Escape") return setHoverT(null);
    else return;
    e.preventDefault();
    setHoverT(next.t);
  };

  const ticks = timeTicks(x0, x1);
  const unit = unitLabel(variable.units);
  const notMeasured = sum.visits - sum.measured - sum.rejected;
  // "Highest in view" sits beside its point but always inside the chart (about 6.6 px per character)
  const topLabel = top ? `Highest in view: ${formatObs(variable, top.value, top.q)}${top.q === "flag_high" ? " (flagged for review)" : ""} · ${fmtDate(top.t)}` : "";
  // rings shrink as visits crowd the lane, so a long record of zeros stays a lane, not a band
  const zeroR = sum.visits > 240 ? 1.7 : sum.visits > 120 ? 2.1 : 2.6;
  const offAxis = win.filter((s2) => onAxis(s2, axis) && axisPosition(s2.value!, axis) !== "in");
  const labelW = topLabel.length * 6.6;
  const labelX = top ? Math.round(Math.max(4, Math.min(x(top.t) + (x(top.t) > PAD.l + iw * 0.62 ? -labelW - 6 : 6), width - labelW - 4))) : 0;
  const noticesIn = notices.filter((n) => n.t >= x0 && n.t <= x1);
  const desc =
    (top
      ? `${variable.label} at this pier from ${fmtDate(x0)} to ${fmtDate(x1)}: ${sum.measured} measured values in ${sum.visits} sampling visits, ${sum.zeros} reported 0. Highest ${formatObs(variable, top.value, top.q)} on ${fmtDate(top.t)}.`
      : `${variable.label} at this pier from ${fmtDate(x0)} to ${fmtDate(x1)}: no value on the axis in view (${sum.zeros} reported 0 in ${sum.visits} visits).`) +
    noticesIn.map((n) => ` ${n.label} from ${fmtDate(n.t)}.`).join("");

  // ---------------------------------------------------------------- model band geometry
  const yM = (v: number) => r1(10 + (1 - v) * (BAND - 28));
  const histStart = m.historyStart;
  const noHistory = !model.history || !Object.keys(model.history).length || !(model.history ?? []).length;

  return (
    <figure ref={ref} className="min-w-0 overflow-x-clip" data-testid="primary-chart" data-variable={variable.id} data-measured={sum.measured} data-visits={sum.visits} data-zeros={sum.zeros} data-domain={axis.domain.join(",")}>
      <figcaption className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
        <div className="min-w-0 space-y-1">
          <h2 className="text-[18px] font-semibold leading-tight text-ink">
            {title} <span className="text-[13px] font-normal text-ink-3">{unit} · {axis.scale === "log" ? "log scale" : "linear scale"}</span>
          </h2>
          <p className="text-[13px] text-ink-2" data-testid="chart-summary">
            {sum.visits === 0 ? (
              "No sampling visits in this period."
            ) : (
              <>
                <span className="whitespace-nowrap">{sum.visits} sampling visits in view:</span>{" "}
                <span className="whitespace-nowrap">
                  <b className="font-semibold text-ink">{sum.measured - sum.zeros}</b> measured value{sum.measured - sum.zeros === 1 ? "" : "s"}
                </span>
                {sum.zeros > 0 && (
                  <>
                    , <span className="whitespace-nowrap"><b className="font-semibold text-ink">{sum.zeros}</b> reported 0</span>
                  </>
                )}
                {notMeasured > 0 && <>, <span className="whitespace-nowrap">{notMeasured} not measured</span></>}
                {sum.rejected > 0 && <>, <span className="whitespace-nowrap">{sum.rejected} rejected in QC</span></>}
              </>
            )}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">{controls}</div>
      </figcaption>

      <div className="relative mt-3">
        <svg
          width={width}
          height={H}
          role="img"
          aria-label={desc}
          tabIndex={0}
          onKeyDown={onKey}
          onPointerMove={onMove}
          onPointerDown={onMove}
          onPointerLeave={onLeave}
          style={{ maxWidth: "100%" }}
          className="block touch-pan-y outline-none focus-visible:ring-2 focus-visible:ring-accent"
          data-testid="chart-measured"
        >
          {/* decade gridlines */}
          {yTicks.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={width - PAD.r} y1={y(t)} y2={y(t)} stroke="var(--cw-hairline)" />
              <text x={PAD.l - 8} y={y(t) + 4} textAnchor="end" className="fill-[var(--cw-ink-3)] text-[12px] tabular">
                {axis.scale === "log" ? formatTick(t) : t}
              </text>
            </g>
          ))}
          {/* the zero lane: reported 0 is "not quantified", never a point on the log axis */}
          {axis.scale === "log" && (
            <g data-testid="zero-lane">
              <line x1={PAD.l} x2={width - PAD.r} y1={laneY} y2={laneY} stroke="var(--cw-hairline)" strokeDasharray="2 3" />
              <text x={PAD.l - 8} y={laneY + 4} textAnchor="end" className="fill-[var(--cw-ink-3)] text-[12px]">
                0*
              </text>
            </g>
          )}
          {/* month ticks */}
          {ticks.map((tk) => (
            <g key={tk.t}>
              <line x1={x(tk.t)} x2={x(tk.t)} y1={PLOT_T} y2={stripY + STRIP} stroke="var(--cw-hairline)" strokeOpacity={0.6} />
              <text x={x(tk.t)} y={H - 6} textAnchor="middle" className="fill-[var(--cw-ink-3)] text-[12px]">
                {tk.label}
              </text>
            </g>
          ))}
          {/* every sampling visit; faint = this quantity not measured that visit */}
          <g data-testid="visit-ticks">
            {visitsIn.map((t) => (
              <line key={t} x1={x(t)} x2={x(t)} y1={stripY} y2={stripY + STRIP - 2} stroke="var(--cw-ink-3)" strokeOpacity={measuredAt.has(t) ? 0.85 : 0.25} strokeWidth={1} />
            ))}
          </g>
          {/* official notices: an amber rule at each effective date (the official mark is a line) */}
          <g data-testid="chart-notices">
            {noticesIn.map((n, i) => (
              <g key={n.id} data-notice={n.id}>
                <line x1={x(n.t)} x2={x(n.t)} y1={PLOT_T - 4} y2={PLOT_T + plotH} stroke="var(--cw-official-line)" strokeWidth={1.5} />
                {i < 3 && !narrow && (
                  // near the right edge the label sits on the rule's left, so it is never cut off
                  <text x={x(n.t) + (x(n.t) > width - PAD.r - 150 ? -4 : 4)} y={PLOT_T + 8 + i * 14} textAnchor={x(n.t) > width - PAD.r - 150 ? "end" : "start"} className="fill-[var(--cw-official-ink)] text-[12px]">
                    {n.label}
                  </text>
                )}
              </g>
            ))}
          </g>
          {/* joined runs of samples ≤ 21 days apart */}
          {segs.map((sg, i) => (
            <path key={i} d={sg.map((s, k) => `${k ? "L" : "M"}${x(s.t)},${y(s.value!)}`).join("")} fill="none" stroke="var(--cw-chl)" strokeWidth={1.6} strokeLinejoin="round" />
          ))}
          {win.map((s) =>
            onAxis(s, axis) && axisPosition(s.value!, axis) !== "in" ? (
              // beyond the fixed axis: a marker at the edge, exact value in the readout
              <path
                key={s.index}
                d={
                  axisPosition(s.value!, axis) === "below"
                    ? `M${x(s.t) - 4},${PLOT_T + plotH - 7}h8l-4,7z`
                    : `M${x(s.t) - 4},${PLOT_T + 7}h8l-4,-7z`
                }
                fill="var(--cw-surface)"
                stroke="var(--cw-chl)"
                strokeWidth={1.3}
                data-q={axisPosition(s.value!, axis) === "below" ? "below-axis" : "above-axis"}
              />
            ) : onAxis(s, axis) ? (
              <circle
                key={s.index}
                cx={x(s.t)}
                cy={y(s.value!)}
                r={s === hovered ? 4.5 : 2.8}
                fill={s.q === "flag_high" ? "var(--cw-surface)" : "var(--cw-chl)"}
                stroke={s.q === "flag_high" ? "var(--cw-warning)" : s === hovered ? "var(--cw-surface)" : "none"}
                strokeWidth={s.q === "flag_high" ? 1.6 : 1.5}
                data-q={s.q ?? "value"}
              />
            ) : s.value != null && (s.q === "reported_zero" || (axis.scale === "log" && s.value <= 0)) ? (
              <circle key={s.index} cx={x(s.t)} cy={laneY} r={s === hovered ? 4 : zeroR} fill="var(--cw-surface)" stroke="var(--cw-chl)" strokeWidth={1.3} data-q="reported_zero" />
            ) : s.q === "rejected_negative" ? (
              <path key={s.index} d={`M${x(s.t) - 2.6},${laneY - 2.6}l5.2,5.2m0,-5.2l-5.2,5.2`} stroke="var(--cw-ink-3)" strokeWidth={1.3} data-q="rejected" />
            ) : null,
          )}
          {/* highest in view */}
          {top && (
            <g data-testid="highest-in-view">
              <line x1={x(top.t)} x2={x(top.t)} y1={14} y2={y(top.value!) - 6} stroke="var(--cw-ink-2)" strokeWidth={1} />
              <text x={labelX} y={12} textAnchor="start" className="fill-[var(--cw-ink)] text-[12.5px] font-semibold">
                Highest in view: {formatObs(variable, top.value, top.q)}
                {top.q === "flag_high" ? " (flagged for review)" : ""}
                <tspan className="fill-[var(--cw-ink-3)] font-normal"> · {fmtDate(top.t)}</tspan>
              </text>
            </g>
          )}
          {crossT != null && <line x1={x(crossT)} x2={x(crossT)} y1={PLOT_T} y2={stripY + STRIP} stroke="var(--cw-ink)" strokeOpacity={0.35} strokeWidth={1} pointerEvents="none" />}
        </svg>

      <div className="mb-3 mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-ink-2" data-testid="chart-legend">
          {noticesIn.length > 0 && (
            <span className="inline-flex items-center gap-1.5">
              <svg width="10" height="12" aria-hidden>
                <line x1="5" x2="5" y1="0" y2="12" stroke="var(--cw-official-line)" strokeWidth="1.5" />
              </svg>
              Official notice in effect from
            </span>
          )}
          <span className="inline-flex items-center gap-1.5">
            <svg width="10" height="10" aria-hidden>
              <circle cx="5" cy="5" r="3.2" fill="var(--cw-chl)" />
            </svg>
            measured
          </span>
          {axis.scale === "log" && (
            <span className="inline-flex items-center gap-1.5">
              <svg width="10" height="10" aria-hidden>
                <circle cx="5" cy="5" r="3" fill="none" stroke="var(--cw-chl)" strokeWidth="1.3" />
              </svg>
              0* reported 0 (not quantified, not absent)
            </span>
          )}
          <span className="inline-flex items-center gap-1.5">
            <svg width="6" height="12" aria-hidden>
              <line x1="3" x2="3" y1="1" y2="11" stroke="var(--cw-ink-3)" />
            </svg>
            sampling visit (faint: not measured)
          </span>
          {offAxis.length > 0 && (
          <span className="inline-flex items-center gap-1.5" data-testid="legend-off-axis">
            <svg width="10" height="10" aria-hidden>
              <path d="M1,2h8l-4,7z" fill="var(--cw-surface)" stroke="var(--cw-chl)" strokeWidth="1.2" />
            </svg>
            {offAxis.length} value{offAxis.length > 1 ? "s" : ""} beyond the axis, drawn at its edge (exact value in the readout)
          </span>
        )}
        <span>Gaps are not zeros: lines break where samples are more than 21 days apart.</span>
          {note}
        </div>

          {sum.measured === 0 && empty && (
          <div className="absolute inset-x-0 flex justify-center" style={{ top: PLOT_T + plotH / 2 - 40 }} data-testid="chart-empty">
            <div className="max-w-[420px] rounded-lg border border-hairline bg-surface px-4 py-3 text-center text-[14px] text-ink-2 shadow-sm">{empty}</div>
          </div>
        )}
        {/* the model, directly beneath: same time axis, its own 0–100 % axis, never compared numerically */}
        <section className="mt-2 rounded-lg bg-model-bg/70 outline outline-1 -outline-offset-1 outline-model-line" data-testid="model-section" aria-labelledby="model-h">
          <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 px-3 pt-2.5">
            <div className="min-w-0">
              <p className="cw-kind text-model-ink"><Mark kind="model" />Model, not a measurement · C-HARM v3.1, NOAA</p>
              <h3 id="model-h" className="text-[14px] font-semibold leading-snug text-ink">
                {model.title}
              </h3>
              <p className="text-[13px] text-ink-2">
                Median of model cells within {model.radiusKm} km{model.nearestCellKm != null ? ` (nearest ${model.nearestCellKm} km away)` : ""}. A probability, never compared with the measurements above.
              </p>
            </div>
            <div className="shrink-0">{model.controls}</div>
          </div>
          {noHistory ? (
            <p className="px-3 pb-3 pt-2 text-[13px] text-ink-2" data-testid="model-unavailable">
              {model.nearestCellKm == null
                ? "C-HARM has no model cells near this station, so there is no model history here."
                : model.error && /unavailable|failed|circuit|timed out/i.test(model.error)
                  ? "The model history could not be retrieved from NOAA in the latest update; it returns when NOAA's server answers."
                  : "No model history has been published for this station."}{" "}
              This says nothing about bloom conditions.
            </p>
          ) : (
            <svg width={width} height={BAND} style={{ maxWidth: "100%" }} className="block" onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={onLeave} data-testid="model-track" aria-label={`${model.title}: ${m.points.length} days in view.`} role="img">
              <defs>
                <pattern id="cw-model-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1="0" y1="0" x2="0" y2="6" stroke="var(--cw-model-line)" strokeWidth="2" />
                </pattern>
              </defs>
              {[0, 0.5, 1].map((v) => (
                <g key={v}>
                  <line x1={PAD.l} x2={width - PAD.r} y1={yM(v)} y2={yM(v)} stroke="var(--cw-model-line)" strokeDasharray={v === 0.5 ? "3 3" : undefined} />
                  <text x={PAD.l - 8} y={yM(v) + 4} textAnchor="end" className="fill-[var(--cw-model-ink)] text-[12px] tabular">
                    {v * 100}%
                  </text>
                </g>
              ))}
              {histStart != null && histStart > x0 && (
                <g data-testid="model-prehistory">
                  <rect x={PAD.l} y={yM(1)} width={Math.max(0, x(Math.min(histStart, x1)) - PAD.l)} height={yM(0) - yM(1)} fill="url(#cw-model-hatch)" opacity={0.7} />
                  {x(Math.min(histStart, x1)) - PAD.l > 150 && (
                    <text x={PAD.l + 8} y={yM(0.5) + 4} className="fill-[var(--cw-model-ink)] text-[12px]">
                      Model history starts {fmtDate(histStart)}
                    </text>
                  )}
                </g>
              )}
              {mSegs.map((sg, i) => (
                <path key={i} d={sg.map((p, k) => `${k ? "L" : "M"}${x(p.t)},${yM(p.v)}`).join("")} fill="none" stroke="var(--cw-forecast)" strokeWidth={1.5} />
              ))}
              {m.points
                .filter((p) => p.partial)
                .map((p) => (
                  <circle key={p.date} cx={x(p.t)} cy={yM(p.v)} r={2.6} fill="var(--cw-surface)" stroke="var(--cw-forecast)" strokeWidth={1.2} data-partial />
                ))}
              {crossT != null && <line x1={x(crossT)} x2={x(crossT)} y1={yM(1)} y2={yM(0)} stroke="var(--cw-ink)" strokeOpacity={0.35} pointerEvents="none" />}
            </svg>
          )}
        </section>

        {crossT != null && (hovered || hoveredModel) && (
          <div
            className="pointer-events-none absolute top-[22px] z-10 w-max max-w-[260px] rounded-md bg-navy-900 px-2.5 py-1.5 text-[13px] leading-snug text-on-navy shadow-lg"
            style={{ left: Math.min(Math.max(x(crossT) + 10, 0), width - 250) }}
            data-testid="chart-tooltip"
            role="status"
          >
            <div className="font-semibold">{fmtDate(crossT)}</div>
            {hovered && (
              <div>
                {hovered.value == null
                  ? "Not measured in this sample"
                  : hovered.q === "reported_zero"
                    ? "Reported 0 (not quantified, not absent)"
                    : formatObs(variable, hovered.value, hovered.q)}
                {hovered.q === "flag_high" && <span className="block text-official-on-navy">Above the plausibility check: kept, flagged for review</span>}
              </div>
            )}
            {hoveredModel && (
              <div className="text-[#c3aef3]">
                Model {Math.round(hoveredModel.v * 100)}% ({hoveredModel.n} cells{hoveredModel.partial ? ", partial coverage" : ""})
              </div>
            )}
          </div>
        )}
      </div>

      <details className="mt-2 text-[13px]" open={tableOpen} onToggle={(e) => setTableOpen((e.target as HTMLDetailsElement).open)}>
          <summary className="cursor-pointer text-ink-2 hover:text-ink">Values in view ({sum.visits} visits)</summary>
          {tableOpen && (
            <div className="mt-2 max-h-72 overflow-auto rounded-md border border-hairline">
              <table className="w-full text-left text-[13px] tabular" data-testid="chart-table">
                <thead className="sticky top-0 bg-surface-2 text-ink-3">
                  <tr>
                    <th className="px-3 py-1.5 font-medium">Sample date</th>
                    <th className="px-3 py-1.5 font-medium">{variable.label}</th>
                  </tr>
                </thead>
                <tbody>
                  {[...win].reverse().map((s) => (
                    <tr key={s.index} className="border-t border-hairline">
                      <td className="px-3 py-1">{fmtDate(s.t)}</td>
                      <td className="px-3 py-1">{formatObs(variable, s.value, s.q)}</td>
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
