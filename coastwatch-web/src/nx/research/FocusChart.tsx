"use client";

import { AxisX, AxisY, Crosshair, GridY, Line, NoData, NowRule, Plot, Points, Selection, usePlot } from "@/nx/ds";
import { day } from "@/nx/ds/format";
import type { RecPoint, Strip } from "./model";
import { modelLine, sayPoint, type SamplePoint } from "./plotData";
import s from "./v13.module.css";

/**
 * C-HARM's own threshold on the measured scale, named. Drawn over the data with a paper halo behind its words
 * and set at the plot's right end, so a dense record never overprints the name (the kit's Threshold writes it at
 * the left with no backing). Not drawn off-scale.
 */
function ThresholdRule({ value, label }: { value: number; label: string }) {
  const g = usePlot();
  const [lo, hi] = g.ySpec.domain;
  if (value < lo || value > hi) return null;
  const y = g.y(value);
  return (
    <g data-threshold="" pointerEvents="none">
      <line x1={g.left} x2={g.right} y1={y} y2={y} className={s.thresholdLine} />
      <text x={g.right - 8} y={y - 7} textAnchor="end" className={s.thresholdLabel} data-testid="research-threshold-label">
        {label}
      </text>
    </g>
  );
}

/** A value above Neritic's plausibility ceiling: kept where the source put it, with a ring that says "flagged". */
function FlagRings({ data }: { data: SamplePoint[] }) {
  const g = usePlot();
  const [a, b] = g.xSpec.domain;
  return (
    <g data-flagged="" pointerEvents="none">
      {data
        .filter((p) => p.meta.flagged && p.v != null && p.t >= a && p.t <= b)
        .map((p) => (
          <circle key={p.id} cx={g.x(p.t)} cy={g.y(p.v as number)} r={8} className={s.flagRing} />
        ))}
    </g>
  );
}

/** The room under a plot's baseline for its time axis (the kit's default bottom margin). */
const AXIS_ROOM = 28;

/**
 * The tooltip for the sample under the crosshair. It stands beside the sample on the side with room and, for a
 * sample in the plot's lower half (a reported 0 sits on the baseline), ABOVE it, so it is never cut by the plot's
 * foot or the screen's. The kit's Tip always hangs below the sample. The pinned sample has no tooltip: the
 * inspector reads it. Decorative for assistive tech: the same words go through the plot's live region.
 */
function SampleTip({ strip }: { strip: Strip }) {
  const g = usePlot();
  const p = g.focus;
  if (!p) return null;
  const rec = p.meta as RecPoint;
  const [, value, state] = sayPoint(strip, rec).split(" · ");
  const x = g.x(p.t);
  const y = p.v != null ? g.y(p.v) : g.bottom;
  const side = x > (g.left + g.right) / 2 ? { right: g.width - x + 12 } : { left: x + 12 };
  const stand = y > (g.top + g.bottom) / 2 ? { bottom: g.bottom + AXIS_ROOM - y + 12 } : { top: Math.max(0, y - 20) };
  return (
    <div className={s.tip} style={{ ...side, ...stand }} aria-hidden="true" data-testid="plot-tip">
      <span className={s.tipValue}>{value}</span>
      <span className={s.tipMeta}>
        {day(p.t, { year: true })} · {state}
      </span>
    </div>
  );
}
// the kit lays a child with this mark over the SVG as HTML (as its own Tip is)
(SampleTip as unknown as { plotLayer: string }).plotLayer = "html";

export type FocusPlotProps = {
  strip: Strip;
  data: SamplePoint[];
  win: [number, number];
  now: number;
  label: string;
  height: number | { desktop: number; phone: number };
  onSelect?: (p: SamplePoint | null) => void;
  testid: string;
};

/**
 * One measured variable on its own axis: marks only where a sample is, joined only when samples are within
 * three weeks, a reported 0 and a rejected sample on the baseline (never at zero on the axis), C-HARM's own
 * threshold named where it is on the scale. The axis spans the station's whole record and does not rescale as
 * the window moves, so a quiet year cannot look like a bloom.
 */
export function MeasuredPlot({ strip, data, win, now, label, height, onSelect, testid }: FocusPlotProps) {
  return (
    <Plot
      label={label}
      data={data}
      x={{ domain: win }}
      y={{ type: strip.scale, domain: strip.domain, unit: strip.unitShort }}
      height={height}
      margin={{ top: 14 }}
      describePoint={(p) => sayPoint(strip, p.meta as RecPoint)}
      onSelect={onSelect ? (p) => onSelect((p as SamplePoint | null) ?? null) : undefined}
      data-testid={testid}
    >
      <GridY />
      <AxisX />
      <AxisY />
      <Line data={data} gapDays={strip.joinDays} />
      <Points data={data} />
      {strip.threshold ? <ThresholdRule value={strip.threshold.value} label={strip.threshold.label} /> : null}
      <FlagRings data={data} />
      <NowRule t={now} label="Today" />
      <Crosshair />
      <Selection />
      <SampleTip strip={strip} />
    </Plot>
  );
}

/**
 * The model on its own panel: hollow marks and a dashed line (computed, never measured), joined only day to
 * day, hatched where Neritic kept no history. Never on the measured plot's axis.
 */
export function ModelPlot({ strip, data, win, now, label, height, testid }: FocusPlotProps) {
  const start = strip.historyStart ?? null;
  return (
    <Plot label={label} data={data} x={{ domain: win }} y={{ type: "linear", domain: [0, 100], unit: "%" }} height={height} margin={{ top: 14 }} describePoint={(p) => sayPoint(strip, p.meta as RecPoint)} data-testid={testid}>
      <GridY ticks={2} />
      <AxisX />
      <AxisY ticks={2} />
      {start != null ? <NoData spans={[[win[0] - 1, start]]} words={`No model history kept before ${day(start, { year: true })}`} /> : null}
      <Line data={modelLine(data, strip.partial)} gapDays={strip.joinDays} kind="forecast" />
      <Points data={data} r={2} />
      <NowRule t={now} label="Today" />
      <Crosshair />
      <SampleTip strip={strip} />
    </Plot>
  );
}
