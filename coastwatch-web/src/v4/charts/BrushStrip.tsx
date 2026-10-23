"use client";

import { useEffect, useRef, useState } from "react";
import { AXES, axisPosition, onAxis } from "@/lib/bloom";
import { DAY, type ObsVariableId, type Sample } from "@/lib/observations";
import { useWidth } from "./useWidth";

/**
 * The whole record in one strip, with the window the chart shows as a brush (v10). Drag across the
 * strip to draw a new window, drag the window to move it, drag an edge to resize it; the edges are
 * also sliders (arrow keys a week, Shift a month). The chart follows while dragging; the window is
 * committed (and written to the link) when the drag ends.
 *
 * The overview is drawn on the same fixed axis as the chart (never rescaled), reported zeros on the
 * floor, and the span of the model history kept under it.
 */
export const MIN_WINDOW_DAYS = 14;

export function BrushStrip({
  samples,
  variable,
  x0,
  x1,
  from,
  to,
  modelStart,
  modelEnd,
  onBrush,
  onCommit,
}: {
  samples: Sample[];
  variable: ObsVariableId;
  /** the record's extent */
  x0: number;
  x1: number;
  /** the window */
  from: number;
  to: number;
  modelStart: number | null;
  modelEnd: number | null;
  /** live, while dragging */
  onBrush: (from: number, to: number) => void;
  onCommit: (from: number, to: number) => void;
}) {
  const [box, W] = useWidth<HTMLDivElement>(1040, 240);
  const svg = useRef<SVGSVGElement | null>(null);
  const axis = AXES[variable];
  const H = 58;
  const pad = { l: 0, r: 0, t: 6 };
  const plotH = 30;
  const span = Math.max(1, x1 - x0);
  const x = (t: number) => pad.l + ((t - x0) / span) * (W - pad.l - pad.r);
  const tAt = (px: number) => x0 + ((px - pad.l) / (W - pad.l - pad.r)) * span;
  const y = (v: number) => {
    const [lo, hi] = axis.domain;
    const f = axis.scale === "log" ? (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : (v - lo) / (hi - lo);
    return pad.t + plotH * (1 - Math.min(1, Math.max(0, f)));
  };
  const minW = MIN_WINDOW_DAYS * DAY;
  const clamp = (a: number, b: number): [number, number] => {
    let lo = Math.max(x0, Math.min(a, b));
    let hi = Math.min(x1, Math.max(a, b));
    if (hi - lo < minW) {
      if (lo + minW <= x1) hi = lo + minW;
      else lo = Math.max(x0, hi - minW);
    }
    return [lo, hi];
  };
  const drag = useRef<{ mode: "new" | "move" | "a" | "b"; t0: number; from: number; to: number; id: number } | null>(null);
  const [live, setLive] = useState<[number, number] | null>(null);
  const [a, b] = live ?? [Math.max(x0, from), Math.min(x1, to)];
  // a new window from outside (a preset, a link) ends any drag that was in progress
  useEffect(() => {
    if (!drag.current) setLive(null);
  }, [from, to]);

  const years: number[] = [];
  for (let yr = new Date(x0).getUTCFullYear() + 1; yr <= new Date(x1).getUTCFullYear(); yr++) years.push(Date.UTC(yr, 0, 1));
  const yearStep = Math.max(1, Math.ceil(years.length / Math.max(1, Math.floor(W / 64))));

  const down = (e: React.PointerEvent) => {
    const r = svg.current!.getBoundingClientRect();
    const px = e.clientX - r.left;
    const t = tAt(px);
    const ax = x(a);
    const bx = x(b);
    const grip = e.pointerType === "mouse" ? 7 : 14;
    const mode = Math.abs(px - ax) <= grip ? "a" : Math.abs(px - bx) <= grip ? "b" : px > ax && px < bx ? "move" : "new";
    drag.current = { mode, t0: t, from: a, to: b, id: e.pointerId };
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    e.preventDefault();
  };
  const move = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    const r = svg.current!.getBoundingClientRect();
    const t = Math.max(x0, Math.min(x1, tAt(e.clientX - r.left)));
    let next: [number, number];
    if (d.mode === "move") {
      const w = d.to - d.from;
      let lo = d.from + (t - d.t0);
      lo = Math.max(x0, Math.min(x1 - w, lo));
      next = [lo, lo + w];
    } else if (d.mode === "a") next = clamp(Math.min(t, d.to - minW), d.to);
    else if (d.mode === "b") next = clamp(d.from, Math.max(t, d.from + minW));
    else next = clamp(d.t0, t);
    setLive(next);
    onBrush(next[0], next[1]);
  };
  const up = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d || d.id !== e.pointerId) return;
    drag.current = null;
    // a click without a drag on the strip centres the window there, at its width
    if (live == null && d.mode === "new") {
      const w = d.to - d.from;
      const lo = Math.max(x0, Math.min(x1 - w, d.t0 - w / 2));
      onCommit(lo, lo + w);
      return;
    }
    if (live) onCommit(live[0], live[1]);
    setLive(null);
  };
  const key = (edge: "a" | "b") => (e: React.KeyboardEvent) => {
    const stepD = (e.shiftKey ? 30 : 7) * DAY;
    const dir = e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowDown" ? -1 : 0;
    if (!dir) return;
    e.preventDefault();
    const next = edge === "a" ? clamp(Math.min(a + dir * stepD, b - minW), b) : clamp(a, Math.max(b + dir * stepD, a + minW));
    onCommit(next[0], next[1]);
  };
  const ax = x(a);
  const bx = x(b);
  const date = (t: number) => new Date(t).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" });

  return (
    <div ref={box} className="relative w-full select-none" data-testid="brush">
      <svg
        ref={svg}
        width={W}
        height={H}
        className="block cursor-crosshair [touch-action:none]"
        aria-hidden
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerCancel={() => {
          drag.current = null;
          setLive(null);
        }}
        data-testid="brush-svg"
      >
        <rect x={0} y={pad.t} width={W} height={plotH} fill="var(--nx-surface)" />
        {samples.map((s, i) => {
          const cx = x(s.t);
          if (s.value === 0 || s.q === "reported_zero") return <rect key={i} x={cx - 0.75} y={pad.t + plotH - 2} width={1.5} height={2} fill="var(--nx-ink-3)" />;
          if (!onAxis(s, axis) || axisPosition(s.value!, axis) !== "in") return null;
          return <circle key={i} cx={cx} cy={y(s.value!)} r={1.1} fill="var(--nx-measured)" opacity={0.85} />;
        })}
        {modelStart != null && modelEnd != null && (
          <line x1={x(Math.max(x0, modelStart))} x2={x(Math.min(x1, modelEnd))} y1={pad.t + plotH + 4} y2={pad.t + plotH + 4} stroke="var(--nx-model)" strokeWidth={2} strokeLinecap="round" data-testid="brush-model" />
        )}
        {years.filter((_, i) => i % yearStep === 0).map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={pad.t} y2={pad.t + plotH} stroke="var(--nx-line)" />
            <text x={x(t) + 3} y={H - 4} fontFamily="var(--nx-font-data)" fontSize="10" fill="var(--nx-ink-3)">
              {new Date(t).getUTCFullYear()}
            </text>
          </g>
        ))}
        {/* outside the window, dimmed; the window itself, outlined */}
        <rect x={0} y={pad.t} width={Math.max(0, ax)} height={plotH} fill="var(--nx-bg)" opacity={0.62} />
        <rect x={bx} y={pad.t} width={Math.max(0, W - bx)} height={plotH} fill="var(--nx-bg)" opacity={0.62} />
        <rect x={ax} y={pad.t + 0.5} width={Math.max(1, bx - ax)} height={plotH - 1} fill="none" stroke="var(--nx-ink-2)" strokeWidth={1} data-testid="brush-window" data-from={new Date(a).toISOString().slice(0, 10)} data-to={new Date(b).toISOString().slice(0, 10)} />
        <rect x={ax - 2} y={pad.t + 7} width={4} height={plotH - 14} rx={1} fill="var(--nx-ink)" />
        <rect x={bx - 2} y={pad.t + 7} width={4} height={plotH - 14} rx={1} fill="var(--nx-ink)" />
      </svg>
      {/* the edges as sliders, over the drawn grips */}
      <div
        tabIndex={0}
        role="slider"
        aria-label="Window start"
        aria-valuemin={x0}
        aria-valuemax={x1}
        aria-valuenow={Math.round(a)}
        aria-valuetext={date(a)}
        onKeyDown={key("a")}
        className="nx-focus absolute h-[30px] w-3 -translate-x-1/2 rounded-[2px] opacity-0 focus-visible:opacity-100"
        style={{ left: ax, top: pad.t, pointerEvents: "none" }}
        data-testid="brush-start"
      />
      <div
        tabIndex={0}
        role="slider"
        aria-label="Window end"
        aria-valuemin={x0}
        aria-valuemax={x1}
        aria-valuenow={Math.round(b)}
        aria-valuetext={date(b)}
        onKeyDown={key("b")}
        className="nx-focus absolute h-[30px] w-3 -translate-x-1/2 rounded-[2px] opacity-0 focus-visible:opacity-100"
        style={{ left: bx, top: pad.t, pointerEvents: "none" }}
        data-testid="brush-end"
      />
    </div>
  );
}
