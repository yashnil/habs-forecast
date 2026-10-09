"use client";

import { useEffect, useRef, useState } from "react";

export type Bar = { key: string; label: string; value: number | null; note?: string };

/**
 * Single-series vertical bars with a zero baseline, 4px rounded data ends, 2px gaps and a
 * per-bar hover tooltip. Missing values are drawn as an outlined "no value" stub, never as 0.
 */
export function BarChart({ bars, format, color, height = 150, label, testid }: { bars: Bar[]; format: (v: number) => string; color: string; height?: number; label: string; testid?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(300);
  const [hover, setHover] = useState<number | null>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(220, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const pad = { l: 52, r: 6, t: 10, b: 20 };
  const max = Math.max(1, ...bars.map((b) => b.value ?? 0));
  const nice = niceMax(max);
  const plotH = height - pad.t - pad.b;
  const slot = (w - pad.l - pad.r) / Math.max(1, bars.length);
  const bw = Math.max(4, slot - 2);
  const y = (v: number) => Math.round((pad.t + plotH - (v / nice) * plotH) * 10) / 10;
  const hb = hover != null ? bars[hover] : null;
  return (
    <figure ref={ref} className="relative min-w-0" data-testid={testid} aria-label={label}>
      <svg width={w} height={height} role="img" aria-label={`${label}: ${bars.map((b) => `${b.label} ${b.value == null ? "no value" : format(b.value)}`).join("; ")}`} onPointerLeave={() => setHover(null)}>
        {[0, nice / 2, nice].map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={y(t)} y2={y(t)} stroke="var(--cw-hairline)" />
            <text x={pad.l - 6} y={y(t) + 3.5} textAnchor="end" className="fill-[var(--cw-ink-3)] text-[10px] tabular">
              {format(t)}
            </text>
          </g>
        ))}
        {bars.map((b, i) => {
          const x = pad.l + i * slot + 1;
          const top = b.value != null ? y(b.value) : pad.t + plotH - 6;
          const h = pad.t + plotH - top;
          return (
            <g key={b.key} onPointerEnter={() => setHover(i)}>
              <rect x={x - 1} y={pad.t} width={slot} height={plotH} fill="transparent" />
              {b.value != null ? (
                <path d={roundedTop(x, top, bw, h, Math.min(4, bw / 2, h))} fill={color} opacity={hover == null || hover === i ? 1 : 0.55} />
              ) : (
                <rect x={x + 0.5} y={top} width={bw - 1} height={6} fill="none" stroke="var(--cw-ink-3)" strokeDasharray="2 2" />
              )}
              {(bars.length <= 12 || i % 2 === 0) && (
                <text x={x + bw / 2} y={height - 5} textAnchor="middle" className="fill-[var(--cw-ink-3)] text-[10px] tabular">
                  {bars.length > 8 ? `’${b.label.slice(-2)}` : b.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>
      {hb && (
        <div
          className="pointer-events-none absolute top-0 z-10 rounded-md border border-hairline-strong bg-surface-3 px-2 py-1 text-[11px] text-ink shadow-lg"
          style={{ left: Math.min(w - 170, Math.max(0, pad.l + hover! * slot - 60)) }}
        >
          <div className="text-ink-3">{hb.label}</div>
          <div className="font-semibold tabular">{hb.value == null ? "no published value" : format(hb.value)}</div>
          {hb.note && <div className="text-ink-3">{hb.note}</div>}
        </div>
      )}
    </figure>
  );
}

function roundedTop(x: number, y: number, w: number, h: number, r: number): string {
  if (h <= 0) return "";
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

function niceMax(v: number): number {
  const e = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * e >= v) return m * e;
  return 10 * e;
}

export function money(v: number): string {
  if (v >= 1e9) return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6) return `$${(v / 1e6).toFixed(v >= 1e7 ? 0 : 1)}M`;
  if (v >= 1e3) return `$${(v / 1e3).toFixed(0)}k`;
  return `$${Math.round(v)}`;
}
