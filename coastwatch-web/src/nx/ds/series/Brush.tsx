"use client";
import { scaleLinear, scaleLog, scaleUtc } from "d3-scale";
import { useRef, type KeyboardEvent, type PointerEvent } from "react";
import { day, monthYear } from "../format";
import { useElementWidth } from "../useMedia";
import { cx } from "../util";
import { classify, logDomain, ms as toMs, type SeriesPoint } from "./series";
import s from "./series.module.css";

/** Coordinates rounded to 0.01 px, so server and client render the same attribute strings. */
const r2 = (v: number) => Math.round(v * 100) / 100;

export interface BrushProps {
  /** The whole record, ms. */
  extent: [number, number];
  /** The window shown above, ms. */
  value: [number, number];
  onChange: (v: [number, number]) => void;
  /** An overview of the record, drawn small. */
  points?: SeriesPoint[];
  scale?: "log" | "linear";
  height?: number;
  /** Narrowest window, in days. */
  minDays?: number;
  /** What the window filters: "every strip above". */
  label: string;
  className?: string;
}

const LEFT = 84;
const RIGHT = 12;
const DAY = 86_400_000;
const utcYear = new Intl.DateTimeFormat("en-US", { timeZone: "UTC", year: "numeric" });

export function clampWindow([a, b]: [number, number], [lo, hi]: [number, number], minMs: number): [number, number] {
  const w = Math.min(hi - lo, Math.max(minMs, b - a));
  const start = Math.min(Math.max(a, lo), hi - w);
  return [start, start + w];
}

/**
 * One brush for a stack of strips: drag the window, drag either edge, or click to centre. Each edge and the
 * window are sliders for the keyboard (arrows: a month; with Shift: a year).
 */
export function Brush({ extent, value, onChange, points = [], scale = "log", height = 56, minDays = 30, label, className }: BrushProps) {
  const [ref, width] = useElementWidth<HTMLDivElement>(640);
  const drag = useRef<{ mode: "a" | "b" | "pan"; x0: number; v0: [number, number] } | null>(null);
  const plotW = Math.max(120, width - LEFT - RIGHT);
  const xs = scaleUtc().domain([new Date(extent[0]), new Date(extent[1])]).range([LEFT, LEFT + plotW]);
  const x = Object.assign((d: Date) => r2(xs(d)), { invert: (px: number) => xs.invert(px), ticks: (n: number) => xs.ticks(n) });
  const vals = points.filter((p) => classify(p) === "value").map((p) => p.v as number);
  const ys = scale === "log" ? scaleLog().domain(logDomain(vals)).range([height - 6, 6]).clamp(true) : scaleLinear().domain([Math.min(0, ...vals), Math.max(1, ...vals)]).range([height - 6, 6]);
  const y = (v: number) => r2(ys(v));
  const minMs = minDays * DAY;
  const msPerPx = (extent[1] - extent[0]) / plotW;
  const xa = x(new Date(value[0]));
  const xb = x(new Date(value[1]));
  const years = x.ticks(Math.min(12, Math.max(2, Math.round(plotW / 90))));

  const set = (v: [number, number]) => onChange(clampWindow(v, extent, minMs));

  const start = (mode: "a" | "b" | "pan") => (e: PointerEvent<HTMLElement>) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { mode, x0: e.clientX, v0: value };
  };
  const move = (e: PointerEvent<HTMLElement>) => {
    const d = drag.current;
    if (!d) return;
    const dt = (e.clientX - d.x0) * msPerPx;
    if (d.mode === "pan") set([d.v0[0] + dt, d.v0[1] + dt]);
    else if (d.mode === "a") set([Math.max(extent[0], Math.min(d.v0[0] + dt, d.v0[1] - minMs)), d.v0[1]]);
    else set([d.v0[0], Math.min(extent[1], Math.max(d.v0[1] + dt, d.v0[0] + minMs))]);
  };
  const end = () => {
    drag.current = null;
  };
  const centre = (e: PointerEvent<HTMLDivElement>) => {
    if (drag.current) return;
    const r = e.currentTarget.getBoundingClientRect();
    const t = x.invert(e.clientX - r.left).getTime();
    const w = value[1] - value[0];
    set([t - w / 2, t + w / 2]);
  };
  const key = (mode: "a" | "b" | "pan") => (e: KeyboardEvent<HTMLButtonElement>) => {
    const step = (e.shiftKey ? 365 : 30) * DAY;
    const w = value[1] - value[0];
    const k = e.key;
    if (!["ArrowRight", "ArrowUp", "ArrowLeft", "ArrowDown", "Home", "End"].includes(k)) return;
    e.preventDefault();
    const d = k === "ArrowRight" || k === "ArrowUp" ? step : k === "ArrowLeft" || k === "ArrowDown" ? -step : 0;
    if (mode === "pan") {
      if (k === "Home") return set([extent[0], extent[0] + w]);
      if (k === "End") return set([extent[1] - w, extent[1]]);
      return set([value[0] + d, value[1] + d]);
    }
    if (mode === "a") {
      const a = k === "Home" ? extent[0] : k === "End" ? value[1] - minMs : value[0] + d;
      return set([Math.max(extent[0], Math.min(a, value[1] - minMs)), value[1]]);
    }
    const b = k === "End" ? extent[1] : k === "Home" ? value[0] + minMs : value[1] + d;
    return set([value[0], Math.min(extent[1], Math.max(b, value[0] + minMs))]);
  };

  const H = height;
  return (
    <div className={cx(s.brush, className)}>
      <div ref={ref} style={{ position: "relative", height: H + 24 }} onPointerDown={centre}>
        <svg width={width} height={H + 24} className={s.brushSvg} aria-hidden focusable="false">
          <line x1={LEFT} x2={LEFT + plotW} y1={H} y2={H} className={s.grid} />
          {points
            .filter((p) => classify(p) === "value")
            .map((p) => (
              <rect key={p.t} x={x(new Date(toMs(p.t))) - 1} y={y(p.v as number) - 1} width={2} height={2} className={s.brushDot} />
            ))}
          <rect x={LEFT} y={0} width={Math.max(0, xa - LEFT)} height={H} className={s.brushShade} />
          <rect x={xb} y={0} width={Math.max(0, LEFT + plotW - xb)} height={H} className={s.brushShade} />
          <rect x={xa} y={1.5} width={Math.max(0, xb - xa)} height={H - 3} className={s.brushWindow} />
          {years.map((d) => (
            <text key={d.getTime()} x={x(d)} y={H + 19} className={s.xTick}>
              {utcYear.format(d)}
            </text>
          ))}
        </svg>
        <button
          type="button"
          className={s.brushPan}
          style={{ left: xa + 12, width: Math.max(0, xb - xa - 24), height: H }}
          role="slider"
          aria-label={`Window for ${label}`}
          aria-valuemin={extent[0]}
          aria-valuemax={extent[1]}
          aria-valuenow={value[0]}
          aria-valuetext={`${monthYear(value[0])} to ${monthYear(value[1])}`}
          onPointerDown={start("pan")}
          onPointerMove={move}
          onPointerUp={end}
          onPointerCancel={end}
          onKeyDown={key("pan")}
        />
        <button
          type="button"
          className={s.brushHandle}
          style={{ left: xa, height: H }}
          role="slider"
          aria-label="Window start"
          aria-valuemin={extent[0]}
          aria-valuemax={value[1]}
          aria-valuenow={value[0]}
          aria-valuetext={day(value[0], { year: true })}
          onPointerDown={start("a")}
          onPointerMove={move}
          onPointerUp={end}
          onPointerCancel={end}
          onKeyDown={key("a")}
        />
        <button
          type="button"
          className={s.brushHandle}
          style={{ left: xb, height: H }}
          role="slider"
          aria-label="Window end"
          aria-valuemin={value[0]}
          aria-valuemax={extent[1]}
          aria-valuenow={value[1]}
          aria-valuetext={day(value[1], { year: true })}
          onPointerDown={start("b")}
          onPointerMove={move}
          onPointerUp={end}
          onPointerCancel={end}
          onKeyDown={key("b")}
        />
      </div>
      <p className={s.brushCaption} aria-live="polite">
        Showing {day(value[0], { year: true })} to {day(value[1], { year: true })}; drag the frame to filter {label}.
      </p>
    </div>
  );
}
