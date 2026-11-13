"use client";

import { motion } from "motion/react";
import { useMemo } from "react";
import { AxisY, GridY, Line, NoData, Plot, Points, usePlot, crosshairGlide, lineDraw } from "@/nx/ds";
import { day } from "@/nx/ds/format";
import { onAxis } from "@/nx/ds/plot/model";
import type { RecPoint, Strip } from "./model";
import { modelLine, sayPoint, type SamplePoint } from "./plotData";
import s from "./rx.module.css";

const DAY = 86_400_000;
/** The reported-0 lane under the log axis, and the time axis under it (v14 synthesis, Research scrubber). */
export const LANE = 24;
const AXIS = 22;
/** The plots' axis room, shared with the overview brush so the two line up. */
export const INSET = { left: 58, right: 12 };

/** Where a sample sits: on the axis, or in the lane (a reported 0 or a rejected sample has no place on a log axis). */
function placeOf(g: ReturnType<typeof usePlot>, p: SamplePoint, lane: boolean) {
  const x = g.x(p.t);
  const y = onAxis(p) ? g.y(p.v as number) : lane ? g.bottom + LANE / 2 + 1 : g.bottom;
  return { x, y, inLane: !onAxis(p) };
}

/** Time ticks as the prototype draws them: years as major rules up the plot, months under them, labels that never collide. */
function ticksOf([a, b]: [number, number]): Array<{ t: number; label: string; major: boolean }> {
  const span = (b - a) / DAY;
  const out: Array<{ t: number; label: string; major: boolean }> = [];
  const y0 = new Date(a).getUTCFullYear(),
    y1 = new Date(b).getUTCFullYear();
  if (span > 1500) {
    const every = span > 4000 ? 2 : 1;
    for (let y = y0; y <= y1; y++) if (y % every === 0) out.push({ t: Date.UTC(y, 0, 1), label: String(y), major: true });
  } else {
    const step = span > 500 ? 3 : span > 200 ? 2 : 1;
    for (let y = y0; y <= y1; y++)
      for (let mo = 0; mo < 12; mo += step) {
        const t = Date.UTC(y, mo, 1);
        const name = new Date(t).toLocaleDateString("en-US", { month: "short", timeZone: "UTC" });
        out.push({ t, label: mo === 0 ? String(y) : span <= 120 ? `${name} 1` : name, major: mo === 0 });
        if (span <= 120) out.push({ t: Date.UTC(y, mo, 15), label: `${name} 15`, major: false });
      }
  }
  return out.filter((x) => x.t >= a && x.t <= b);
}

function TimeAxis({ lane }: { lane: boolean }) {
  const g = usePlot();
  const ly = lane ? LANE : 0;
  let lastRight = -99;
  return (
    <g className={s.axis} data-axis="x">
      <line x1={g.left} x2={g.right} y1={g.bottom + 0.5} y2={g.bottom + 0.5} className={s.baseline} />
      {ticksOf(g.xSpec.domain).map(({ t, label, major }) => {
        const x = g.x(t);
        const w = label.length * 6.4;
        const left = major ? x + 5 : x - w / 2;
        const show = left >= lastRight + 8 && left + w <= g.width;
        if (show) lastRight = left + w;
        return (
          <g key={t}>
            <line x1={x} x2={x} y1={major ? g.top : g.bottom} y2={g.bottom + (ly || 5)} className={major ? s.tickMajor : s.tick} />
            {show ? (
              <text x={major ? x + 5 : x} y={g.bottom + ly + 16} textAnchor={major ? "start" : "middle"} className={major ? s.tickYear : s.tickLabel}>
                {label}
              </text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

/** The model's reference level on the measured scale: dashed (a model's line), named in the legend, never a limit. */
function RefLine({ value }: { value: number }) {
  const g = usePlot();
  const [lo, hi] = g.ySpec.domain;
  if (value < lo || value > hi) return null;
  const y = g.y(value);
  return (
    <g data-threshold="" pointerEvents="none">
      <line x1={g.left} x2={g.right} y1={y} y2={y} className={s.refLine} />
    </g>
  );
}

/** Reported 0s (hollow squares) and rejected samples (a crossed ring) in their own lane under the axis, labelled "rep. 0". */
function ZeroLane({ data }: { data: SamplePoint[] }) {
  const g = usePlot();
  const [a, b] = g.xSpec.domain;
  const zy = g.bottom + LANE / 2 + 1;
  const shown = data.filter((p) => !onAxis(p) && p.t >= a && p.t <= b);
  const dense = shown.length > 120;
  const r = dense ? 2 : 3;
  let zeros = "";
  let rejected = "";
  for (const p of shown) {
    const x = g.x(p.t);
    if (p.kind === "rejected") rejected += `M${x - r} ${zy + r}L${x + r} ${zy - r}M${x + r + 0.5} ${zy}a${r + 0.5} ${r + 0.5} 0 1 0 ${-2 * r - 1} 0a${r + 0.5} ${r + 0.5} 0 1 0 ${2 * r + 1} 0`;
    else zeros += `M${x - r} ${zy - r}h${2 * r}v${2 * r}h${-2 * r}Z`;
  }
  return (
    <g data-zero-lane="">
      <text x={g.left - 10} y={zy + 4} textAnchor="end" className={s.tickLabel}>
        rep. 0
      </text>
      <path d={zeros} className={s.zero} />
      <path d={rejected} className={s.rejected} />
    </g>
  );
}

/** Long gaps between samples (over 60 days) drawn as a dotted run in the lane, "not measured" where there is room. */
function Gaps({ data, lane, now }: { data: SamplePoint[]; lane: boolean; now: number }) {
  const g = usePlot();
  const [a, b] = g.xSpec.domain;
  const gy = lane ? g.bottom + LANE / 2 + 1 : g.bottom + 9;
  const out: Array<[number, number]> = [];
  let prev: number | null = null;
  for (const p of data) {
    if (prev != null && p.t >= a && prev <= b && p.t - prev >= 60 * DAY) out.push([prev, p.t]);
    prev = p.t;
  }
  if (prev != null && prev < b && Math.min(b, now) - prev >= 60 * DAY) out.push([prev, Math.min(b, now)]);
  return (
    <g data-gaps="" pointerEvents="none">
      {out.map(([f, t]) => {
        const xa = Math.max(g.left, g.x(f)) + 7;
        const xb = Math.min(g.right, g.x(t)) - 7;
        if (xb - xa < 14) return null;
        return (
          <g key={f}>
            <line x1={xa} x2={xb} y1={gy} y2={gy} className={s.gap} />
            {xb - xa > 110 ? (
              <text x={(xa + xb) / 2} y={gy + 4} textAnchor="middle" className={s.gapWords}>
                not measured
              </text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

/** The crosshair: it glides between samples on a spring (crosshairGlide) instead of jumping; the pinned sample has none. */
function GlideCrosshair({ lane }: { lane: boolean }) {
  const g = usePlot();
  const p = g.focus;
  if (!p || (g.pinned && g.pinned.t === p.t)) return null;
  const { x, y, inLane } = placeOf(g, p as SamplePoint, lane);
  const foot = g.bottom + (lane ? LANE : 0);
  return (
    <g data-crosshair="" pointerEvents="none">
      <motion.g initial={false} animate={{ x }} transition={crosshairGlide}>
        <line x1={0} x2={0} y1={g.top} y2={foot} className={s.cross} />
      </motion.g>
      {!inLane ? <motion.line initial={false} x1={g.left} y1={y} y2={y} animate={{ x2: x, y1: y, y2: y }} transition={crosshairGlide} className={s.crossH} /> : null}
      <motion.circle initial={false} r={6} animate={{ cx: x, cy: y }} transition={crosshairGlide} className={s.crossRing} />
    </g>
  );
}

/** The selected sample: an ultramarine rule, a ring on the sample and its date on a flag above the plot; at the edge when outside the window. */
function PinMark({ lane, pinnedT }: { lane: boolean; pinnedT: number | null }) {
  const g = usePlot();
  const p = g.pinned;
  const foot = g.bottom + (lane ? LANE : 0);
  if (!p) {
    if (pinnedT == null) return null;
    const [a, b] = g.xSpec.domain;
    if (pinnedT >= a && pinnedT <= b) return null;
    const before = pinnedT < a;
    const words = before ? `‹ selected ${day(pinnedT, { year: true })}` : `selected ${day(pinnedT, { year: true })} ›`;
    const w = words.length * 6.3 + 18;
    const x = before ? g.left : g.right - w;
    return (
      <g data-selection-outside="" pointerEvents="none">
        <rect x={x} y={g.top - 24} width={w} height={22} rx={7} className={s.flagOut} />
        <text x={x + w / 2} y={g.top - 9} textAnchor="middle" className={s.flagOutText}>
          {words}
        </text>
      </g>
    );
  }
  const { x, y, inLane } = placeOf(g, p as SamplePoint, lane);
  const words = day(p.t, { year: true });
  const w = words.length * 6.6 + 18;
  const fx = Math.max(g.left, Math.min(g.right - w, x - w / 2));
  return (
    <g data-selection="" pointerEvents="none">
      <line x1={x} x2={x} y1={g.top} y2={foot} className={s.pinRule} />
      {inLane ? (
        <rect data-ring="" x={x - 6} y={y - 6} width={12} height={12} rx={2} className={s.pinSquare} />
      ) : (
        <g data-ring="">
          <circle cx={x} cy={y} r={8.5} className={s.pinRing} />
          <circle cx={x} cy={y} r={3.2} className={s.pinDot} />
        </g>
      )}
      <rect x={fx} y={g.top - 24} width={w} height={22} rx={7} className={s.flag} />
      <text x={fx + w / 2} y={g.top - 9} textAnchor="middle" className={s.flagText}>
        {words}
      </text>
    </g>
  );
}

/**
 * The readout: the sample under the crosshair, pinned to the plot's top-right corner (the prototype's `.rs-read`), so it
 * never covers the sample or the line. Decorative for assistive tech: the plot's live region says the same.
 */
function Readout({ strip }: { strip: Strip }) {
  const g = usePlot();
  const p = g.focus;
  if (!p || (g.pinned && g.pinned.t === p.t)) return null;
  const [when, value] = sayPoint(strip, p.meta as RecPoint).split(" · ");
  return (
    <div className={s.readout} style={{ right: g.width - g.right }} aria-hidden="true" data-testid="plot-tip">
      {when} · {value}
    </div>
  );
}
(Readout as unknown as { plotLayer: string }).plotLayer = "html";

/** The measured line draws itself once, left to right (lineDraw, 1.0 s), when a station or variable arrives. */
function Reveal({ drawKey, children }: { drawKey: string; children: React.ReactNode }) {
  const g = usePlot();
  const id = `${g.id}-reveal`;
  const W = Math.max(0, g.width);
  return (
    <g clipPath={`url(#${id})`}>
      <clipPath id={id}>
        <motion.rect key={drawKey} x={0} y={0} height={g.height + 40} initial={{ width: 0 }} animate={{ width: W }} transition={lineDraw} />
      </clipPath>
      {children}
    </g>
  );
}

export type ScrubPlotProps = {
  strip: Strip;
  data: SamplePoint[];
  win: [number, number];
  now: number;
  label: string;
  /** the plot area's height, px (the axis and lane are added under it) */
  height: number;
  /** the page's selected sample (from the address), so the plot can say where it is when it is outside the window */
  pinnedT: number | null;
  onSelect?: (p: SamplePoint | null) => void;
  drawKey: string;
  testid: string;
};

/**
 * One measured variable as the synthesis draws it: the log axis fixed to the station's whole record, marks only where
 * a sample is, joined only within three weeks, reported 0s in their own lane ("at or below the lab's limit", never on
 * the axis), long gaps said as "not measured", the model's reference level dashed. The kit's Plot carries the
 * interaction (snap within 24 px, tap to pin, arrow keys, Enter, Escape, the live region); the marks are the page's.
 */
export function ScrubPlot({ strip, data, win, now, label, height, pinnedT, onSelect, drawKey, testid }: ScrubPlotProps) {
  const lane = strip.scale === "log" && data.some((p) => !onAxis(p));
  const onAxisData = useMemo(() => data.filter((p) => onAxis(p)), [data]);
  return (
    <Plot
      label={label}
      data={data}
      x={{ domain: win }}
      y={{ type: strip.scale, domain: strip.domain, unit: strip.unitShort }}
      height={Math.max(60, height)}
      margin={{ left: INSET.left, right: INSET.right, top: 26, bottom: (lane ? LANE : 0) + AXIS }}
      describePoint={(p) => sayPoint(strip, p.meta as RecPoint)}
      onSelect={onSelect ? (p) => onSelect((p as SamplePoint | null) ?? null) : undefined}
      className={s.plot}
      data-testid={testid}
    >
      <GridY />
      <AxisY />
      <TimeAxis lane={lane} />
      {strip.threshold ? <RefLine value={strip.threshold.value} /> : null}
      <Gaps data={data} lane={lane} now={now} />
      <Reveal drawKey={drawKey}>
        <Line data={onAxisData} gapDays={strip.joinDays} />
        <Points data={onAxisData} r={onAxisData.length > 300 ? 2 : onAxisData.length > 80 ? 3.2 : 4} />
        {lane ? <ZeroLane data={data} /> : null}
      </Reveal>
      <GlideCrosshair lane={lane} />
      <PinMark lane={lane} pinnedT={pinnedT} />
      <Readout strip={strip} />
    </Plot>
  );
}

/**
 * The model on its own panel and axis (0–100 %): hollow marks, a dashed line joined only day to day, a dotted slot
 * where Neritic kept no history. Never on the measured plot's axis.
 */
export function ModelPanel({ strip, data, win, label, height, testid }: { strip: Strip; data: SamplePoint[]; win: [number, number]; label: string; height: number; testid: string }) {
  const start = strip.historyStart ?? null;
  return (
    <Plot
      label={label}
      data={data}
      x={{ domain: win }}
      y={{ type: "linear", domain: [0, 100], unit: "%" }}
      height={Math.max(40, height)}
      margin={{ left: INSET.left, right: INSET.right, top: 6, bottom: 6 }}
      describePoint={(p) => sayPoint(strip, p.meta as RecPoint)}
      className={s.plot}
      data-testid={testid}
    >
      <GridY ticks={2} />
      <AxisY ticks={2} />
      {start != null ? <NoData spans={[[win[0] - 1, start]]} words={`No model runs kept before ${day(start, { year: true })}`} /> : null}
      <Line data={modelLine(data, strip.partial)} gapDays={strip.joinDays} kind="forecast" />
      <Points data={data} r={2} />
      <GlideCrosshair lane={false} />
      <Readout strip={strip} />
    </Plot>
  );
}
