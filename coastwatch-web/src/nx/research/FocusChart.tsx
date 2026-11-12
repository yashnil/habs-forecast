"use client";

import { AxisX, AxisY, Crosshair, GridY, Line, NoData, NowRule, Plot, Points, Threshold, Tip, usePlot, type PlotPoint } from "@/nx/ds";
import { day } from "@/nx/ds/format";
import type { RecPoint, Strip } from "./model";
import { modelLine, sayPoint, type SamplePoint } from "./plotData";
import s from "./v13.module.css";

/** The selected sample: a dashed rule and a square ring that stay while the pointer is elsewhere (the page owns the selection, so a link can carry it). */
function SelectedMark({ p }: { p: PlotPoint | null }) {
  const g = usePlot();
  if (!p || p.t < g.xSpec.domain[0] || p.t > g.xSpec.domain[1]) return null;
  const x = g.x(p.t);
  const y = p.v != null ? g.y(p.v) : g.bottom;
  return (
    <g data-selected-sample="" pointerEvents="none" data-testid="research-selected">
      <line x1={x} x2={x} y1={g.top} y2={g.bottom} className={s.selRule} />
      <rect x={x - 9} y={y - 9} width={18} height={18} className={s.selRing} />
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

function SampleTip({ strip, p }: { strip: Strip; p: PlotPoint }) {
  const rec = p.meta as RecPoint;
  const [, value, state] = sayPoint(strip, rec).split(" · ");
  return (
    <>
      <span className={s.tipValue}>{value}</span>
      <span className={s.tipMeta}>
        {day(p.t, { year: true })} · {state}
      </span>
    </>
  );
}

export type FocusPlotProps = {
  strip: Strip;
  data: SamplePoint[];
  win: [number, number];
  now: number;
  label: string;
  height: number | { desktop: number; phone: number };
  selected?: PlotPoint | null;
  onSelect?: (p: SamplePoint | null) => void;
  testid: string;
};

/**
 * One measured variable on its own axis: marks only where a sample is, joined only when samples are within
 * three weeks, a reported 0 and a rejected sample on the baseline (never at zero on the axis), C-HARM's own
 * threshold named where it is on the scale. The axis spans the station's whole record and does not rescale as
 * the window moves, so a quiet year cannot look like a bloom.
 */
export function MeasuredPlot({ strip, data, win, now, label, height, selected, onSelect, testid }: FocusPlotProps) {
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
      {strip.threshold ? <Threshold value={strip.threshold.value} label={strip.threshold.label} /> : null}
      <Line data={data} gapDays={strip.joinDays} />
      <Points data={data} />
      <FlagRings data={data} />
      <NowRule t={now} label="Today" />
      <Crosshair />
      <SelectedMark p={selected ?? null} />
      <Tip render={(p) => <SampleTip strip={strip} p={p} />} />
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
      <Tip render={(p) => <SampleTip strip={strip} p={p} />} />
    </Plot>
  );
}
