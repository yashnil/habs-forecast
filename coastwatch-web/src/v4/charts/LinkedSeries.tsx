"use client";

import { useId, useMemo, useRef, useState, type ReactNode } from "react";
import { AXES, axisPosition, decadeTicks, highestInView, modelSegments, onAxis, type Axis, type ModelPoint } from "@/lib/bloom";
import { DAY, unitLabel, type ObsVariableId, type Sample } from "@/lib/observations";
import { measuredSegments, monthTicks, sampleOnDay, tickLabel } from "./SeriesChart";
import { useWidth } from "./useWidth";

/**
 * The Research instrument (v10): a station's measured series and the C-HARM model probability near it,
 * as two linked panels on one shared time axis, never one value axis (a sample and a probability are
 * not comparable as numbers).
 *
 * The measured panel keeps every rule SeriesChart keeps: fixed whole-decade axes that never rescale to
 * the data, lines only between samples less than 21 days apart (a gap is a gap, and a long one is drawn
 * as a band that says "not measured"), reported zeros as a mark on the floor (not quantified, not
 * absent), a rejected sample as a cross on the floor (never a value), a value flagged high ringed and
 * kept as published, values beyond the axis as edge markers. Every sampling visit is a tick in the rug
 * under the plot, so a visit that did not measure this quantity is visible as a visit without a point.
 *
 * One crosshair crosses both panels. The pointer reads the nearest real sample (or, over the model
 * panel, the nearest model day); a tip gives the exact sample (date, value, units, QC, source) and the
 * model value of the same day when one was kept. The keyboard moves between real samples (arrow keys,
 * Home, End) and Enter pins the reading (Escape lets it go). On a touch screen a drag scrubs and the
 * reading stays pinned where the finger lifts.
 */

export type LinkedModel = {
  points: ModelPoint[];
  /** before this, Neritic kept no model history: hatched, and said */
  historyStart: number | null;
  /** e.g. "Domoic acid (pDA)": what the probability is of */
  what: string;
  radiusKm: number;
};

export type LinkedReading = {
  t: number;
  day: string;
  sample: Sample | null;
  model: ModelPoint | null;
  /** why there is no model value that day */
  modelWhy: string | null;
};

const dayKey = (t: number) => new Date(t).toISOString().slice(0, 10);
const pct = (v: number) => `${Math.round(v * 100)}%`;
const LONG_GAP = 42 * DAY;
const MONO = "var(--nx-font-data)";
const CAPS = "var(--nx-font-caps)";
const SERIF = "var(--nx-font-display)";

/** A value as the chart and its tip print it: tabular, never more precise than published. */
export function fmtValue(v: number): string {
  return v >= 1000 ? Math.round(v).toLocaleString("en-US") : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toPrecision(2);
}

/** The words for one sample's QC state (a blank qualifier is "no QC flag", never "passed"). */
export function qcWords(s: Sample): string {
  if (s.q === "rejected_negative") return "rejected by QC, not a measurement";
  if (s.value == null) return "a visit; this quantity was not measured";
  if (s.q === "reported_zero" || s.value === 0) return "reported 0: not quantified, not absent";
  if (s.q === "flag_high") return "flagged high by QC, kept as published";
  return "no QC flag";
}

/** The samples the crosshair stops at: a value, a reported zero or a rejected sample (not a bare visit). */
export const realSamples = (ss: Sample[]) => ss.filter((s) => s.value != null || s.q === "rejected_negative");

export function readingAt(t: number, inWin: Sample[], model: LinkedModel | null): LinkedReading {
  const day = dayKey(t);
  const i = sampleOnDay(inWin, t);
  const sample = i != null && (inWin[i].value != null || inWin[i].q === "rejected_negative") ? inWin[i] : null;
  const m = model?.points.find((p) => p.date === day) ?? null;
  const modelWhy = !model ? null : m ? null : model.historyStart != null && t < model.historyStart ? "before the model history Neritic keeps" : "no model value kept for that day";
  return { t, day, sample, model: m, modelWhy };
}

export function LinkedSeries({
  samples,
  variable,
  units,
  label,
  axisTitle,
  from,
  to,
  refLine,
  model,
  modelNote,
  cursorT,
  pinnedT,
  onHover,
  onPin,
  source,
  obsHeight = 300,
  modelHeight = 120,
  compact = false,
}: {
  samples: Sample[];
  variable: ObsVariableId;
  units: string;
  label: string;
  axisTitle: string;
  from: number;
  to: number;
  /** a labelled reference value (C-HARM's own threshold): context, not a safety line */
  refLine?: { value: number; label: string };
  /** the model panel; null = this quantity has no model counterpart (modelNote says why) */
  model: LinkedModel | null;
  /** shown in place of the model panel: no counterpart, or no history published */
  modelNote?: ReactNode;
  /** the crosshair (ms) */
  cursorT: number | null;
  /** the pinned reading (ms), if any */
  pinnedT: number | null;
  onHover: (t: number | null) => void;
  onPin: (t: number | null) => void;
  /** the measured series' source, as the tip names it */
  source: string;
  obsHeight?: number;
  modelHeight?: number;
  /** a phone: smaller margins, no floating tip (the readings above the chart follow the crosshair) */
  compact?: boolean;
}) {
  const unit = unitLabel(units);
  const axis: Axis = AXES[variable];
  const [box, W] = useWidth<HTMLDivElement>(compact ? 358 : 1040, 280);
  const id = useId();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [active, setActive] = useState(false);
  const scrub = useRef<{ touch: boolean; moved: boolean; x: number } | null>(null);

  // ---------------------------------------------------------------- geometry
  const pad = { l: compact ? 46 : 58, r: compact ? 10 : 18 };
  const iw = Math.max(40, W - pad.l - pad.r);
  // on a phone the threshold's label is a key above the plot (on the line it ran into the data)
  const refIn = !!refLine && axisPosition(refLine.value, axis) === "in";
  const refKey = compact && refIn;
  const obsTop = refKey ? 36 : compact ? 22 : 26;
  const obsH = obsHeight;
  const obsBot = obsTop + obsH;
  const rugY = obsBot + 9;
  const hasModelPanel = !!model;
  const modelTop = rugY + 34;
  const modelBot = hasModelPanel ? modelTop + modelHeight : rugY;
  const axisY = modelBot + 20;
  const H = axisY + 8;
  const x = (t: number) => pad.l + ((t - from) / Math.max(1, to - from)) * iw;
  const tAt = (px: number) => from + ((px - pad.l) / iw) * (to - from);
  const y = (v: number) => {
    const [lo, hi] = axis.domain;
    const f = axis.scale === "log" ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);
    return obsTop + obsH * (1 - Math.min(1, Math.max(0, f)));
  };
  const my = (v: number) => modelTop + (1 - v) * modelHeight;

  // ---------------------------------------------------------------- data in view
  const inWin = useMemo(() => samples.filter((s) => s.t >= from && s.t <= to), [samples, from, to]);
  const real = useMemo(() => realSamples(inWin), [inWin]);
  const segs = useMemo(() => measuredSegments(samples, axis, from, to), [samples, axis, from, to]);
  const top = useMemo(() => highestInView(samples, axis, from, to), [samples, axis, from, to]);
  const msegs = useMemo(() => (model ? modelSegments(model.points) : []), [model]);
  // long stretches without a measurement: drawn as "not measured", never as a low value
  const gaps = useMemo(() => {
    const measured = inWin.filter((s) => s.value != null && s.q !== "rejected_negative");
    const out: [number, number][] = [];
    let prev = from;
    for (const s of measured) {
      if (s.t - prev > LONG_GAP) out.push([prev, s.t]);
      prev = s.t;
    }
    if (measured.length && to - prev > LONG_GAP) out.push([prev, to]);
    return out;
  }, [inWin, from, to]);
  const ticks = axis.scale === "log" ? decadeTicks(axis.domain).filter((_, i, a) => a.length < 7 || obsH >= 200 || i % 2 === 0) : [5, 10, 15, 20, 25, 30].filter((v) => v >= axis.domain[0] && v <= axis.domain[1]);
  const months = useMemo(() => monthTicks(from, to, iw + 60), [from, to, iw]);
  const tickFmt = (v: number) => (axis.scale === "linear" ? `${v}°` : v >= 1e6 ? `${v / 1e6}M` : v >= 1000 ? `${v / 1000}k` : String(v));
  const empty = real.length === 0;

  const reading = cursorT != null ? readingAt(cursorT, inWin, model) : null;
  const curX = cursorT != null && cursorT >= from && cursorT <= to ? x(cursorT) : null;
  const curSample = reading?.sample ?? null;
  const pinX = pinnedT != null && pinnedT >= from && pinnedT <= to ? x(pinnedT) : null;

  // ---------------------------------------------------------------- pointer and keys
  const nearestSample = (t: number) => {
    let best: Sample | null = null;
    for (const s of real) if (!best || Math.abs(s.t - t) < Math.abs(best.t - t)) best = s;
    return best;
  };
  const snap = (clientX: number, clientY: number): number | null => {
    const r = svgRef.current?.getBoundingClientRect();
    if (!r) return null;
    const px = clientX - r.left;
    const py = clientY - r.top;
    const t = Math.min(to, Math.max(from, tAt(px)));
    // over the model panel the pointer reads the nearest model day; elsewhere the nearest real sample
    if (model && py >= modelTop - 14 && model.points.length) {
      let best = model.points[0];
      for (const p of model.points) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
      // a sample of the same day wins (the crosshair stands on real data)
      return best.t;
    }
    return nearestSample(t)?.t ?? null;
  };
  const step = (dir: 1 | -1 | "home" | "end") => {
    if (!real.length) return;
    let next: Sample | undefined;
    if (dir === "home") next = real[0];
    else if (dir === "end") next = real[real.length - 1];
    else {
      const base = cursorT ?? (dir > 0 ? -Infinity : Infinity);
      // from a model day the crosshair steps to the next sample on another day
      const k = dayKey(base);
      next = dir > 0 ? real.find((s) => s.t > base && dayKey(s.t) !== k) : real.findLast((s) => s.t < base && dayKey(s.t) !== k);
    }
    if (next) onHover(next.t);
  };

  const tipText = reading
    ? `${longDay(reading.day)}: ${reading.sample ? (reading.sample.q === "rejected_negative" ? "rejected by QC, not a measurement" : `${reading.sample.q === "reported_zero" || reading.sample.value === 0 ? "0" : fmtValue(reading.sample.value!)} ${unit}, ${qcWords(reading.sample)}`) : "no sample that day"}${model ? `; C-HARM nowcast ${reading.model ? `${pct(reading.model.v)} model probability` : reading.modelWhy}` : ""}${pinnedT != null && dayKey(pinnedT) === reading.day ? " (pinned)" : ""}`
    : "";
  const showTip = !compact && reading != null && (active || pinnedT != null);
  const tipLeft = curX != null && curX > pad.l + iw * 0.62;

  return (
    <div ref={box} className="relative w-full" data-testid={`series-${variable}`}>
      <div
        role="slider"
        tabIndex={0}
        aria-label={`${label} sample cursor. Arrow keys move between samples, Home and End go to the first and last, Enter pins the reading, Escape releases it.`}
        aria-valuemin={0}
        aria-valuemax={Math.max(0, real.length - 1)}
        aria-valuenow={Math.max(0, curSample ? real.indexOf(curSample) : 0)}
        aria-valuetext={tipText || "no sample in view"}
        aria-describedby={`${id}-d`}
        className="nx-focus relative block rounded-[2px] outline-offset-4 [touch-action:pan-y]"
        onFocus={() => setActive(true)}
        onBlur={() => setActive(false)}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight" || e.key === "ArrowUp") step(1);
          else if (e.key === "ArrowLeft" || e.key === "ArrowDown") step(-1);
          else if (e.key === "Home") step("home");
          else if (e.key === "End") step("end");
          else if (e.key === "Enter" || e.key === " ") onPin(cursorT != null && pinnedT != null && dayKey(pinnedT) === dayKey(cursorT) ? null : cursorT);
          else if (e.key === "Escape") {
            if (pinnedT == null) return;
            onPin(null);
          } else return;
          e.preventDefault();
        }}
        onPointerEnter={() => setActive(true)}
        onPointerLeave={(e) => {
          if (e.pointerType === "touch") return;
          setActive(false);
          onHover(null);
        }}
        onPointerDown={(e) => {
          scrub.current = { touch: e.pointerType !== "mouse", moved: false, x: e.clientX };
          if (e.pointerType !== "mouse") {
            const t = snap(e.clientX, e.clientY);
            if (t != null) onHover(t);
          }
        }}
        onPointerMove={(e) => {
          const s = scrub.current;
          if (s && Math.abs(e.clientX - s.x) > 4) s.moved = true;
          // a touch reads only while the finger is down (a scroll that starts here is the page's)
          if (e.pointerType !== "mouse" && !s) return;
          const t = snap(e.clientX, e.clientY);
          if (t != null) onHover(t);
        }}
        onPointerCancel={() => {
          // the page took the gesture (a scroll): the crosshair returns to where it was
          scrub.current = null;
          onHover(null);
        }}
        onPointerUp={(e) => {
          const s = scrub.current;
          scrub.current = null;
          const t = snap(e.clientX, e.clientY);
          if (t == null) return;
          // a click (or a tap) pins the reading, a second click on the same day lets it go;
          // a touch scrub keeps the reading where the finger lifts
          if (s?.touch && s.moved) onPin(t);
          else onPin(pinnedT != null && dayKey(pinnedT) === dayKey(t) ? null : t);
          if (s?.touch) onHover(null);
        }}
        data-testid="series-svg"
      >
        <svg ref={svgRef} width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block select-none" aria-hidden data-testid="linked-svg">
          <defs>
            <pattern id={`${id}-gap`} width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="7" stroke="var(--nx-line)" strokeWidth="1.2" />
            </pattern>
            <pattern id={`${id}-hatch`} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1="0" y1="0" x2="0" y2="6" stroke="var(--nx-line-strong)" strokeWidth="1" opacity="0.6" />
            </pattern>
          </defs>

          {/* ---- the measured panel */}
          <text x={pad.l} y={12} fontFamily={CAPS} fontSize="10.5" letterSpacing="0.12em" fill="var(--nx-measured)" style={{ textTransform: "uppercase" }} data-testid="series-axis-title">
            {axisTitle}
          </text>
          {!compact && (
            <text x={W - pad.r} y={12} textAnchor="end" fontFamily={SERIF} fontStyle="italic" fontSize="12.5" fill="var(--nx-ink-3)">
              {axis.scale === "log" ? "fixed whole decades, never rescaled to the data" : "fixed scale, never rescaled to the data"}
            </text>
          )}
          {months.map((t) => (
            <line key={`v${t}`} x1={x(t)} x2={x(t)} y1={obsTop} y2={modelBot} stroke="var(--nx-line)" strokeOpacity={0.55} />
          ))}
          {ticks.map((v) => (
            <g key={v}>
              <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--nx-line)" />
              <text x={pad.l - 8} y={y(v) + 3.5} textAnchor="end" fontFamily={MONO} fontSize="10.5" fill="var(--nx-ink-3)">
                {tickFmt(v)}
              </text>
            </g>
          ))}
          {gaps.map(([a, b]) => {
            const x0 = x(a) + 4;
            const x1 = x(b) - 4;
            if (x1 - x0 < 8) return null;
            return (
              <g key={a} data-testid="series-gap">
                <rect x={x0} y={obsTop} width={x1 - x0} height={obsH} fill={`url(#${id}-gap)`} />
                {x1 - x0 > 76 && (
                  <text x={(x0 + x1) / 2} y={obsTop + obsH / 2} textAnchor="middle" fontFamily={SERIF} fontStyle="italic" fontSize="12" fill="var(--nx-ink-3)" stroke="var(--nx-bg)" strokeWidth={4} paintOrder="stroke">
                    not measured
                  </text>
                )}
              </g>
            );
          })}
          {refLine && refIn && (
            <g data-testid="series-ref">
              <line x1={pad.l} x2={W - pad.r} y1={y(refLine.value)} y2={y(refLine.value)} stroke="var(--nx-model)" strokeDasharray="5 4" strokeWidth={1.1} opacity={0.9} />
              {refKey ? (
                <g data-testid="series-ref-key">
                  <line x1={pad.l} x2={pad.l + 18} y1={24} y2={24} stroke="var(--nx-model)" strokeDasharray="5 4" strokeWidth={1.1} />
                  <text x={pad.l + 24} y={28} fontSize="11.5" fill="var(--nx-model)">
                    {refLine.label}
                  </text>
                </g>
              ) : (
                <text x={W - pad.r - 2} y={y(refLine.value) - 6} textAnchor="end" fontSize="11.5" fill="var(--nx-model)" stroke="var(--nx-bg)" strokeWidth={4} strokeLinejoin="round" paintOrder="stroke">
                  {refLine.label}
                </text>
              )}
            </g>
          )}
          {segs.map((sg, i) => (
            <polyline key={i} points={sg.map((s) => `${x(s.t).toFixed(1)},${y(s.value!).toFixed(1)}`).join(" ")} fill="none" stroke="var(--nx-measured)" strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" data-testid="series-line" />
          ))}
          {inWin.map((s, i) => {
            const cx = x(s.t);
            if (s.q === "rejected_negative") {
              const cy = obsBot - 7;
              return (
                <path key={i} d={`M${cx - 3.5},${cy - 3.5} l7,7 m0,-7 l-7,7`} stroke="var(--nx-down)" strokeWidth={1.6} strokeLinecap="round" data-testid="series-rejected" />
              );
            }
            if (s.value == null) return null;
            if (s.value === 0 || s.q === "reported_zero") return <rect key={i} x={cx - 3} y={obsBot - 3} width={6} height={3} fill="var(--nx-ink-3)" data-testid="series-zero" />;
            if (!onAxis(s, axis)) return null;
            const pos = axisPosition(s.value, axis);
            if (pos !== "in") return <path key={i} d={pos === "above" ? `M${cx - 4},${obsTop + 6} l4,-6 l4,6z` : `M${cx - 4},${obsBot - 6} l4,6 l4,-6z`} fill="var(--nx-measured)" />;
            return (
              <g key={i}>
                <circle cx={cx} cy={y(s.value)} r={2.4} fill="var(--nx-measured)" />
                {s.q === "flag_high" && <circle cx={cx} cy={y(s.value)} r={7} fill="none" stroke="var(--nx-down)" strokeWidth={1.3} data-testid="series-flagged" />}
              </g>
            );
          })}
          {top && axisPosition(top.value!, axis) === "in" && real.length > 2 && !compact && (
            <HighestNote x={x(top.t)} y={y(top.value!)} left={x(top.t) > pad.l + iw * 0.7} text={`highest in view · ${fmtValue(top.value!)} ${unit} · ${shortDay(dayKey(top.t))}`} />
          )}
          <line x1={pad.l} x2={W - pad.r} y1={obsBot} y2={obsBot} stroke="var(--nx-line-strong)" />
          {/* the rug: every sampling visit, measured or not */}
          {inWin.map((s, i) => (
            <line key={`r${i}`} x1={x(s.t)} x2={x(s.t)} y1={rugY - 3} y2={rugY + 3} stroke={s.value != null || s.q === "rejected_negative" ? "var(--nx-measured)" : "var(--nx-gap)"} strokeOpacity={s.value != null ? 0.55 : 0.8} />
          ))}
          {empty && (
            <text x={pad.l + iw / 2} y={obsTop + obsH / 2} textAnchor="middle" fontSize="14" fill="var(--nx-ink-2)" data-testid="series-empty">
              No {label.toLowerCase()} measured in this window. No sample is not a clean result.
            </text>
          )}

          {/* ---- the model panel: its own axis (0–100 % probability), never the samples' */}
          {model && (
            <g data-testid="model-panel-svg">
              <text x={pad.l} y={modelTop - 12} fontFamily={CAPS} fontSize="10.5" letterSpacing="0.12em" fill="var(--nx-model)" style={{ textTransform: "uppercase" }} data-testid="model-title">
                {compact ? "C-HARM nowcast, model probability" : `NOAA C-HARM nowcast · probability · ${model.what} · median of cells within ${model.radiusKm} km`}
              </text>
              {!compact && (
                <text x={W - pad.r} y={modelTop - 12} textAnchor="end" fontFamily={SERIF} fontStyle="italic" fontSize="12.5" fill="var(--nx-ink-3)">
                  a model, not comparable with the samples as numbers
                </text>
              )}
              {[0, 0.5, 1].map((v) => (
                <g key={v}>
                  <line x1={pad.l} x2={W - pad.r} y1={my(v)} y2={my(v)} stroke="var(--nx-line)" />
                  <text x={pad.l - 8} y={my(v) + 3.5} textAnchor="end" fontFamily={MONO} fontSize="10.5" fill="var(--nx-ink-3)">
                    {Math.round(v * 100)}%
                  </text>
                </g>
              ))}
              {model.historyStart != null && model.historyStart > from && (
                <g data-testid="model-hatch">
                  <rect x={pad.l} y={modelTop} width={Math.max(0, Math.min(x(model.historyStart), W - pad.r) - pad.l)} height={modelHeight} fill={`url(#${id}-hatch)`} />
                  <HistoryNote x0={pad.l} x1={Math.min(x(model.historyStart), W - pad.r)} y={modelTop + 16} right={W - pad.r} text={`no model history kept before ${shortDay(dayKey(model.historyStart), true)}`} />
                </g>
              )}
              {msegs.map((sg, i) => (
                <polyline key={i} points={sg.map((p) => `${x(p.t).toFixed(1)},${my(p.v).toFixed(1)}`).join(" ")} fill="none" stroke="var(--nx-model)" strokeWidth={1.4} strokeLinejoin="round" />
              ))}
              {model.points.map((p) =>
                p.partial ? <circle key={p.t} cx={x(p.t)} cy={my(p.v)} r={2.6} fill="var(--nx-bg)" stroke="var(--nx-model)" strokeWidth={1.1} /> : model.points.length < 120 ? <circle key={p.t} cx={x(p.t)} cy={my(p.v)} r={1.5} fill="var(--nx-model)" /> : null,
              )}
            </g>
          )}

          {/* ---- the shared time axis */}
          {months.filter((t) => x(t) <= W - pad.r - 16).map((t) => (
            <text key={t} x={x(t)} y={axisY} textAnchor="middle" fontFamily={MONO} fontSize="10.5" fill="var(--nx-ink-3)">
              {tickLabel(t, from, to)}
            </text>
          ))}

          {/* ---- the pin and the crosshair, across both panels */}
          {pinX != null && pinnedT != null && (!reading || dayKey(pinnedT) !== reading.day) && <line x1={pinX} x2={pinX} y1={obsTop} y2={modelBot} stroke="var(--nx-ink-3)" strokeDasharray="2 3" data-testid="series-pin" />}
          {curX != null && (
            <g pointerEvents="none">
              <line x1={curX} x2={curX} y1={obsTop - 2} y2={obsBot} stroke="var(--nx-ink-2)" strokeWidth={1} data-testid="series-cursor" />
              {model && <line x1={curX} x2={curX} y1={modelTop} y2={modelBot} stroke="var(--nx-ink-2)" strokeWidth={1} data-testid="model-cursor" />}
              {pinnedT != null && reading && dayKey(pinnedT) === reading.day && <path d={`M${curX - 4},${obsTop - 8} h8 l-4,5z`} fill="var(--nx-ink)" data-testid="series-pinned" />}
              {curSample?.value != null && onAxis(curSample, axis) && axisPosition(curSample.value, axis) === "in" && (
                <circle cx={curX} cy={y(curSample.value)} r={5} fill="var(--nx-bg)" stroke="var(--nx-measured)" strokeWidth={2} />
              )}
              {curSample && (curSample.value === 0 || curSample.q === "reported_zero") && <rect x={curX - 5} y={obsBot - 5} width={10} height={5} fill="var(--nx-ink)" />}
              {reading?.model && <circle cx={curX} cy={my(reading.model.v)} r={4.5} fill="var(--nx-bg)" stroke="var(--nx-model)" strokeWidth={2} />}
            </g>
          )}
        </svg>
        <span id={`${id}-d`} className="sr-only">
          {axisTitle}. {real.length} samples in view. {model ? "Below it, on its own axis, the C-HARM model probability near the station." : ""}
        </span>
        {showTip && reading && curX != null && (
          <Tip
            reading={reading}
            unit={unit}
            label={label}
            source={source}
            model={model}
            pinned={pinnedT != null && dayKey(pinnedT) === reading.day}
            style={tipLeft ? { right: W - curX + 12, top: obsTop + 6 } : { left: curX + 12, top: obsTop + 6 }}
          />
        )}
      </div>
      {modelNote && !model && <div className="mt-1">{modelNote}</div>}
    </div>
  );
}

function HighestNote({ x, y, left, text }: { x: number; y: number; left: boolean; text: string }) {
  // a short leader up and away from the point, the words at its end (an annotation, not a label on every point)
  const ty = Math.max(30, y - 22);
  const tx = left ? x - 10 : x + 10;
  return (
    <g data-testid="series-highest" pointerEvents="none">
      <path d={`M${x},${y - 6} L${x},${ty + 4} L${tx},${ty + 4}`} fill="none" stroke="var(--nx-ink-3)" strokeWidth={0.8} />
      <text x={left ? tx - 4 : tx + 4} y={ty + 8} textAnchor={left ? "end" : "start"} fontFamily={SERIF} fontStyle="italic" fontSize="12.5" fill="var(--nx-ink-2)" stroke="var(--nx-bg)" strokeWidth={4} paintOrder="stroke">
        {text}
      </text>
    </g>
  );
}

function HistoryNote({ x0, x1, y, right, text }: { x0: number; x1: number; y: number; right: number; text: string }) {
  // inside the hatch when it fits; otherwise just after its edge, pointing back at it
  const inside = x1 - x0 > text.length * 6.2 + 16;
  if (!inside && right - x1 < text.length * 6.2 + 24) return null;
  return (
    <text x={inside ? x0 + 8 : x1 + 8} y={y} fontFamily={SERIF} fontStyle="italic" fontSize="12.5" fill="var(--nx-ink-2)" stroke="var(--nx-bg)" strokeWidth={4} paintOrder="stroke" data-testid="model-history-note">
      {inside ? text : `← ${text}`}
    </text>
  );
}

function Tip({ reading, unit, label, source, model, pinned, style }: { reading: LinkedReading; unit: string; label: string; source: string; model: LinkedModel | null; pinned: boolean; style: React.CSSProperties }) {
  const s = reading.sample;
  return (
    <div className="nx-plate pointer-events-none absolute z-10 w-[248px] px-3 py-2.5 shadow-[var(--nx-shadow)]" style={style} data-testid="chart-tip">
      <p className="nx-caps flex justify-between gap-2">
        <span>{longDay(reading.day)}</span>
        {pinned && <span className="text-n-ink">pinned</span>}
      </p>
      <div className="mt-1.5 flex gap-2.5">
        <span aria-hidden className="nx-tick" data-family="measured" />
        <div className="min-w-0">
          {s ? (
            s.q === "rejected_negative" ? (
              <p className="text-[14px] text-n-down">rejected by QC, not a measurement</p>
            ) : (
              <p className="nx-value !text-[22px]" data-testid="tip-value">
                {s.q === "reported_zero" || s.value === 0 ? "0" : fmtValue(s.value!)}
                <small>{unit}</small>
              </p>
            )
          ) : (
            <p className="text-[13.5px] text-n-ink-2">no sample that day</p>
          )}
          <p className="text-[12.5px] leading-snug text-n-ink-2">{s ? `${label} · ${qcWords(s)}` : label}</p>
          <p className="text-[11.5px] leading-snug text-n-ink-3">{source}</p>
        </div>
      </div>
      {model && (
        <div className="nx-rule mt-2 flex gap-2.5 pt-2">
          <span aria-hidden className="nx-tick" data-family="model" />
          <div className="min-w-0">
            {reading.model ? (
              <p className="text-[13px] leading-snug text-n-ink-2">
                <span className="font-[family-name:var(--nx-font-data)] text-[15px] text-n-model">{pct(reading.model.v)}</span> C-HARM nowcast, same day · median of {reading.model.n} cells within {model.radiusKm} km{reading.model.partial ? " (fewer cells than usual)" : ""}
              </p>
            ) : (
              <p className="text-[12.5px] leading-snug text-n-ink-3">C-HARM: {reading.modelWhy}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

const DTF_LONG = new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
const DTF_SHORT = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
const DTF_SHORT_Y = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });
/** "Wed, Sep 30, 2026": a sample's UTC day, with its year (a tip is read out of context) */
export const longDay = (day: string) => DTF_LONG.format(new Date(`${day}T12:00:00Z`));
const shortDay = (day: string, year = false) => (year ? DTF_SHORT_Y : DTF_SHORT).format(new Date(`${day}T12:00:00Z`));
