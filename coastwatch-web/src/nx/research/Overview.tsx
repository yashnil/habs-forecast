"use client";

import { useRef, type KeyboardEvent, type PointerEvent } from "react";
import { useElementWidth } from "@/nx/ds";
import { day, monthYear } from "@/nx/ds/format";
import { clampWin, DAY, MIN_WINDOW_DAYS, type RecPoint, type Strip } from "./model";
import s from "./rx.module.css";

const r2 = (v: number) => Math.round(v * 100) / 100;

/**
 * The overview of the whole record under the plot (v14 synthesis, Research's brush): every measured value as a small
 * dash on the station's fixed axis, year rules with their labels inside the track, and the window as an ultramarine
 * frame with two grips. Drag the frame or a grip, or press the track to centre the window there. The keyboard has the
 * same three sliders as the v13 brush: the window (arrows a month, Shift a year, Home and End) and each edge.
 */
export function Overview({
  extent,
  value,
  onChange,
  points,
  strip,
  selectedT,
  label,
  inset,
  height = 34,
}: {
  extent: [number, number];
  value: [number, number];
  onChange: (w: [number, number]) => void;
  points: RecPoint[];
  strip: Pick<Strip, "scale" | "domain">;
  selectedT: number | null;
  label: string;
  inset: { left: number; right: number };
  height?: number;
}) {
  const [ref, width] = useElementWidth<HTMLDivElement>(640);
  const drag = useRef<{ mode: "a" | "b" | "pan"; x0: number; v0: [number, number] } | null>(null);
  const L = inset.left;
  const W = Math.max(60, width - inset.left - inset.right);
  const span = Math.max(1, extent[1] - extent[0]);
  const x = (t: number) => r2(L + ((t - extent[0]) / span) * W);
  const toT = (px: number) => extent[0] + ((px - L) / W) * span;
  const [lo, hi] = strip.domain;
  const f = strip.scale === "log" ? (v: number) => (Math.log10(Math.max(v, lo)) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v: number) => (v - lo) / (hi - lo || 1);
  const H = height;
  const y = (v: number) => r2(H - 4 - Math.max(0, Math.min(1, f(v))) * (H - 12));
  const minMs = MIN_WINDOW_DAYS * DAY;
  const set = (w: [number, number]) => onChange(clampWin(w, extent));
  let d = "";
  for (const p of points) if (p.cls === "value" && p.v != null) d += `M${x(p.t)} ${y(p.v)}h1.4`;
  const xa = Math.max(L, x(value[0]));
  const xb = Math.min(L + W, Math.max(xa + 6, x(value[1])));
  const y0 = new Date(extent[0]).getUTCFullYear() + 1;
  const y1 = new Date(extent[1]).getUTCFullYear();
  const every = W < 500 ? 4 : 2;
  const years: number[] = [];
  for (let yy = y0; yy <= y1; yy++) if (yy % every === 0) years.push(yy);

  const start = (mode: "a" | "b" | "pan") => (e: PointerEvent<HTMLElement>) => {
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    drag.current = { mode, x0: e.clientX, v0: value };
  };
  const move = (e: PointerEvent<HTMLElement>) => {
    const g = drag.current;
    if (!g) return;
    const dt = ((e.clientX - g.x0) / W) * span;
    if (g.mode === "pan") set([g.v0[0] + dt, g.v0[1] + dt]);
    else if (g.mode === "a") set([Math.min(g.v0[0] + dt, g.v0[1] - minMs), g.v0[1]]);
    else set([g.v0[0], Math.max(g.v0[1] + dt, g.v0[0] + minMs)]);
  };
  const end = () => {
    drag.current = null;
  };
  const centre = (e: PointerEvent<HTMLDivElement>) => {
    if (drag.current) return;
    const r = e.currentTarget.getBoundingClientRect();
    const t = toT(e.clientX - r.left);
    const w = value[1] - value[0];
    set([t - w / 2, t + w / 2]);
  };
  const key = (mode: "a" | "b" | "pan") => (e: KeyboardEvent<HTMLButtonElement>) => {
    const step = (e.shiftKey ? 365 : 30) * DAY;
    const w = value[1] - value[0];
    const k = e.key;
    if (!["ArrowRight", "ArrowUp", "ArrowLeft", "ArrowDown", "Home", "End"].includes(k)) return;
    e.preventDefault();
    const dd = k === "ArrowRight" || k === "ArrowUp" ? step : k === "ArrowLeft" || k === "ArrowDown" ? -step : 0;
    if (mode === "pan") {
      if (k === "Home") return set([extent[0], extent[0] + w]);
      if (k === "End") return set([extent[1] - w, extent[1]]);
      return set([value[0] + dd, value[1] + dd]);
    }
    if (mode === "a") {
      const a = k === "Home" ? extent[0] : k === "End" ? value[1] - minMs : value[0] + dd;
      return set([Math.max(extent[0], Math.min(a, value[1] - minMs)), value[1]]);
    }
    const b = k === "End" ? extent[1] : k === "Home" ? value[0] + minMs : value[1] + dd;
    return set([value[0], Math.min(extent[1], Math.max(b, value[0] + minMs))]);
  };
  const grip = 22;
  return (
    <div ref={ref} className={s.overview} style={{ height: H + 4 }} onPointerDown={centre}>
      <svg width={width} height={H + 4} aria-hidden="true" focusable="false" className={s.overviewSvg}>
        <rect x={L} y={2} width={W} height={H} rx={6} className={s.ovTrack} />
        {years.map((yy) => {
          const xx = x(Date.UTC(yy, 0, 1));
          const clash = Math.abs(xx + 4 - xa) < 12 || Math.abs(xx + 18 - xb) < 22 || xx + 34 > L + W;
          return (
            <g key={yy}>
              <line x1={xx} x2={xx} y1={2} y2={H + 2} className={s.ovYear} />
              {clash ? null : (
                <text x={xx + 4} y={13} className={s.ovYearText}>
                  {yy}
                </text>
              )}
            </g>
          );
        })}
        <path d={d} className={s.ovDots} />
        <rect x={xa} y={0} width={Math.max(6, xb - xa)} height={H + 4} rx={6} className={s.ovWindow} />
        {[xa + 4.5, xb - 4.5].map((gx, i) => (
          <line key={i} x1={gx} x2={gx} y1={7} y2={H - 3} className={s.ovGrip} />
        ))}
        {selectedT != null ? <line x1={x(selectedT)} x2={x(selectedT)} y1={0} y2={H + 4} className={s.ovSel} /> : null}
      </svg>
      <button
        type="button"
        className={s.ovPan}
        style={{ left: xa + 8, width: Math.max(10, xb - xa - 16), height: H + 4 }}
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
        className={s.ovGripBtn}
        style={{ left: xa - grip / 2, width: grip, height: H + 4 }}
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
        className={s.ovGripBtn}
        style={{ left: xb - grip / 2, width: grip, height: H + 4 }}
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
  );
}
