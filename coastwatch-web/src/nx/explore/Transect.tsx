"use client";

import { scaleLinear, scaleLog } from "d3-scale";
import { line as d3line } from "d3-shape";
import { motion, useReducedMotion } from "motion/react";
import { useMemo, useState, type KeyboardEvent, type PointerEvent } from "react";
import { ink, sig, useElementWidth } from "@/nx/ds";
import { runs, type TransectPoint } from "./read";
import s from "./explore.module.css";

/**
 * The values along the tapped point's latitude, west to east (README §9 Readout: "a linked transect chart").
 * One mark per published cell; a cell without a value breaks the line and, for an observation, is hatched "no
 * view" (T19). The line inks in with the `ink` preset; hovering (or arrow keys) moves a marker on the chart
 * to the same cell, so the chart and the transect read together.
 */
const M = { left: 44, right: 10, top: 10, bottom: 28 };
const H = 132;

export function Transect({ points, unit, scale, label, observed, format, onHover, hoverKm }: { points: TransectPoint[]; unit: string; scale: "log" | "linear"; label: string; observed: boolean; format: (v: number) => string; onHover: (p: TransectPoint | null) => void; hoverKm: number | null }) {
  const [ref, width] = useElementWidth<HTMLDivElement>(300);
  const reduce = useReducedMotion();
  const [focusI, setFocusI] = useState<number | null>(null);
  const vals = points.filter((p) => p.value !== null).map((p) => p.value as number);
  const x0 = points[0]?.km ?? -1, x1 = points[points.length - 1]?.km ?? 1;
  const W = Math.max(160, width);
  const x = scaleLinear().domain([x0, x1]).range([M.left, W - M.right]);
  const lo = vals.length ? Math.min(...vals) : 1, hi = vals.length ? Math.max(...vals) : 2;
  const y =
    scale === "log"
      ? scaleLog()
          .domain([Math.max(0.01, 10 ** Math.floor(Math.log10(Math.max(lo, 0.01)))), 10 ** Math.ceil(Math.log10(Math.max(hi, 0.02)))])
          .range([H - M.bottom, M.top])
          .clamp(true)
      : scaleLinear().domain([lo, hi === lo ? lo + 1 : hi]).nice().range([H - M.bottom, M.top]);
  const yTicks = scale === "log" ? (y.ticks(3) as number[]).filter((t) => Math.abs(Math.log10(t) - Math.round(Math.log10(t))) < 1e-6) : (y.ticks(3) as number[]);
  const line = d3line<TransectPoint>()
    .x((p) => x(p.km))
    .y((p) => y(p.value as number));
  const pieces = useMemo(() => runs(points), [points]);
  const gaps: [number, number][] = [];
  let g0: number | null = null;
  const cell = points.length > 1 ? Math.abs(points[1].km - points[0].km) : 1;
  points.forEach((p, i) => {
    if (p.value === null && g0 === null) g0 = p.km - cell / 2;
    if (p.value !== null && g0 !== null) {
      gaps.push([g0, p.km - cell / 2]);
      g0 = null;
    }
    if (i === points.length - 1 && g0 !== null) gaps.push([g0, p.km + cell / 2]);
  });
  const nearestI = (km: number) => points.reduce((b, p, i) => (Math.abs(p.km - km) < Math.abs(points[b].km - km) ? i : b), 0);
  const move = (e: PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const km = x.invert(e.clientX - r.left);
    onHover(points[nearestI(km)] ?? null);
  };
  const keys = (e: KeyboardEvent<SVGSVGElement>) => {
    const i = focusI ?? nearestI(0);
    const to = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? points.length - 1 : null;
    if (to === null) return;
    e.preventDefault();
    const j = Math.max(0, Math.min(points.length - 1, to));
    setFocusI(j);
    onHover(points[j]);
  };
  const hp = hoverKm === null ? null : points[nearestI(hoverKm)];
  const kmTick = (v: number) => (Math.abs(v) < 0.5 ? "here" : `${Math.round(Math.abs(v))} km ${v < 0 ? "W" : "E"}`);
  // "here" always; an end label only where it clears "here" (tapping near the coast left "12 km E" over it, critic N8)
  const xTicks = [x0 + 2, 0, x1 - 2].filter((v, i, a) => a.indexOf(v) === i && (v === 0 || Math.abs(x(v) - x(0)) > 78));
  const readout = hp ? (hp.value === null ? `${kmTick(hp.km)}: ${observed ? "no view" : "no value"}` : `${kmTick(hp.km)}: ${format(hp.value)} ${unit}`) : null;
  return (
    <figure className={s.transect} ref={ref} data-testid="nx-transect">
      <figcaption className={s.transectHead}>
        <span>{label}</span>
        <span className={s.transectRead} aria-live="polite">
          {readout ?? `${points.length} cells, west to east`}
        </span>
      </figcaption>
      <svg width={W} height={H} className={s.transectSvg} role="img" aria-label={`${label}: ${vals.length} values${points.length - vals.length ? `, ${points.length - vals.length} cells with no value` : ""}. Arrow keys read one cell at a time.`} tabIndex={0} onPointerMove={move} onPointerLeave={() => onHover(null)} onKeyDown={keys} onBlur={() => onHover(null)}>
        <defs>
          <pattern id="nx-transect-hatch" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="5" className={s.hatchLine} />
          </pattern>
        </defs>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} className={s.gridLine} />
            <text x={M.left - 6} y={y(t) + 5} textAnchor="end" className={s.axisText}>
              {sig(t, 2)}
            </text>
          </g>
        ))}
        {observed
          ? gaps.map(([a, b]) => <rect key={a} x={x(a)} y={M.top} width={Math.max(1, x(b) - x(a))} height={H - M.top - M.bottom} fill="url(#nx-transect-hatch)" className={s.gapBand} />)
          : null}
        <line x1={x(0)} x2={x(0)} y1={M.top - 4} y2={H - M.bottom} className={s.hereLine} />
        {pieces.map((run) =>
          run.length > 1 ? (
            <motion.path key={`${run[0].km}`} d={line(run) ?? ""} className={s.transectLine} initial={reduce ? false : { pathLength: 0 }} animate={{ pathLength: 1 }} transition={ink(1.1)} />
          ) : (
            <circle key={`${run[0].km}`} cx={x(run[0].km)} cy={y(run[0].value as number)} r={2} className={s.transectDot} />
          ),
        )}
        {hp && hp.value !== null ? <circle cx={x(hp.km)} cy={y(hp.value)} r={4} className={s.hoverDot} /> : null}
        {hp ? <line x1={x(hp.km)} x2={x(hp.km)} y1={M.top} y2={H - M.bottom} className={s.hoverLine} /> : null}
        {xTicks.map((t) => (
          <text key={t} x={x(t)} y={H - 8} textAnchor={t === xTicks[0] ? "start" : t === xTicks[xTicks.length - 1] ? "end" : "middle"} className={s.axisText}>
            {kmTick(t)}
          </text>
        ))}
      </svg>
    </figure>
  );
}
