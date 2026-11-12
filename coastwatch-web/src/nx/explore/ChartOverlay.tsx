"use client";

import { motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState, type RefObject } from "react";
import { createPortal } from "react-dom";
import { ink } from "@/nx/ds";
import { getFrame, onFrame, type ChartFrame } from "@/nx/chart/runtime";
import { visibleBounds } from "@/nx/chart/mercator";
import { lineObstacle, registerChartObstacle, useChartObstacle } from "@/nx/chart/cover";
import COAST from "./locator-coast.json";
import s from "./explore.module.css";

/**
 * What Explore writes on the chart itself, over the canvas: the tapped point, the transect along its latitude
 * (inked in with `ink`), and the cell the transect chart is hovering. Positions follow the camera every frame
 * through the host's projector, without re-rendering React. On a phone, a small locator shows where the view
 * sits on the California coast, so zooming in never loses the context.
 */
function useTarget() {
  const [el, setEl] = useState<HTMLElement | null>(null);
  useEffect(() => setEl(document.getElementById("nx-chart-marks")), []);
  return el;
}

/** Below this many pixels across, a cell's outline is a speck: the ring around its centre marks it instead. */
const OUTLINE_MIN_PX = 9;

export function ReadingOnChart({ lat, lon, lon0, lon1, hoverLon, outline }: { lat: number; lon: number; lon0: number | null; lon1: number | null; hoverLon: number | null; outline?: [number, number][] | null }) {
  const target = useTarget();
  const reduce = useReducedMotion();
  const lineRef = useRef<SVGPathElement>(null);
  const ptRef = useRef<SVGGElement>(null);
  const hovRef = useRef<SVGCircleElement>(null);
  const cellRef = useRef<SVGPathElement>(null);
  const caseRef = useRef<SVGPathElement>(null);
  const args = useRef({ lat, lon, lon0, lon1, hoverLon, outline });
  args.current = { lat, lon, lon0, lon1, hoverLon, outline };
  useEffect(() => {
    const draw = (f: ChartFrame | null) => {
      if (!f) return;
      const a = args.current;
      const p = f.project(a.lon, a.lat);
      ptRef.current?.setAttribute("transform", `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`);
      if (lineRef.current && a.lon0 !== null && a.lon1 !== null) {
        const q0 = f.project(a.lon0, a.lat), q1 = f.project(a.lon1, a.lat);
        lineRef.current.setAttribute("d", `M${q0.x.toFixed(1)},${q0.y.toFixed(1)}L${q1.x.toFixed(1)},${q1.y.toFixed(1)}`);
      }
      const c = cellRef.current;
      if (c) {
        // the cell that was read, at its true footprint; too small to see, the ring at its centre stands for it
        const pts = (a.outline ?? []).map(([lo, la]) => f.project(lo, la));
        const w = pts.length ? Math.max(...pts.map((q) => q.x)) - Math.min(...pts.map((q) => q.x)) : 0;
        if (pts.length && w >= OUTLINE_MIN_PX) {
          const d = `M${pts.map((q) => `${q.x.toFixed(1)},${q.y.toFixed(1)}`).join("L")}Z`;
          for (const el of [c, caseRef.current]) {
            el?.setAttribute("d", d);
            el?.setAttribute("visibility", "visible");
          }
          ptRef.current?.setAttribute("data-cell", "outlined");
        } else {
          c.setAttribute("visibility", "hidden");
          caseRef.current?.setAttribute("visibility", "hidden");
          ptRef.current?.setAttribute("data-cell", "point");
        }
      }
      const h = hovRef.current;
      if (h) {
        if (a.hoverLon === null) h.setAttribute("visibility", "hidden");
        else {
          const q = f.project(a.hoverLon, a.lat);
          h.setAttribute("cx", q.x.toFixed(1));
          h.setAttribute("cy", q.y.toFixed(1));
          h.setAttribute("visibility", "visible");
        }
      }
    };
    const now = () => draw(getFrame());
    drawRef.current = now;
    now();
    const off = onFrame(draw);
    return off;
  }, [target]);
  const drawRef = useRef<() => void>(() => undefined);
  // the chart's lettering keeps clear of the transect line and of the read cell's mark (src/nx/chart/cover.ts)
  useEffect(
    () =>
      registerChartObstacle("explore-reading", (project) => {
        const a = args.current;
        const p = project(a.lon, a.lat);
        const mark = [{ x0: p.x - 12, y0: p.y - 12, x1: p.x + 12, y1: p.y + 12 }];
        return a.lon0 === null || a.lon1 === null ? mark : [...mark, ...lineObstacle(project(a.lon0, a.lat), project(a.lon1, a.lat), 6)];
      }),
    [],
  );
  useEffect(() => {
    // a new hover, point or cell: redraw at once with the camera where it is
    drawRef.current();
  }, [lat, lon, lon0, lon1, hoverLon, outline]);
  if (!target) return null;
  return createPortal(
    <svg className={s.onChart} aria-hidden="true" data-testid="nx-reading-on-chart">
      {lon0 !== null && lon1 !== null ? <motion.path key={`${lat}-${lon}`} ref={lineRef} className={s.chartTransect} initial={reduce ? false : { pathLength: 0 }} animate={{ pathLength: 1 }} transition={ink(1.1)} /> : null}
      <path ref={caseRef} className={s.chartCellCase} visibility="hidden" />
      <path ref={cellRef} className={s.chartCell} visibility="hidden" data-testid="nx-cell-outline" />
      <circle ref={hovRef} r={5} className={s.chartHover} visibility="hidden" />
      <g ref={ptRef} className={s.chartPoint}>
        <circle r={7} className={s.chartPointRing} />
        <circle r={2.5} className={s.chartPointDot} />
      </g>
    </svg>,
    target,
  );
}

// the locator's own small projection: equirectangular, scaled to the coast's box
const LON0 = -125.2, LON1 = -116.6, LAT0 = 32.2, LAT1 = 42.2;
const LW = 44, LH = Math.round((LW * (LAT1 - LAT0)) / ((LON1 - LON0) * Math.cos((37 * Math.PI) / 180)));
const lx = (lon: number) => ((lon - LON0) / (LON1 - LON0)) * LW;
const ly = (lat: number) => ((LAT1 - lat) / (LAT1 - LAT0)) * LH;
const COAST_D = (COAST as [number, number][]).map(([lon, lat], i) => `${i ? "L" : "M"}${lx(lon).toFixed(1)},${ly(lat).toFixed(1)}`).join("");

export function Locator() {
  const target = useTarget();
  const rectRef = useRef<SVGRectElement>(null);
  // port and place names keep clear of the inset
  const insetRef = useRef<SVGSVGElement>(null);
  useChartObstacle("explore-locator", insetRef as unknown as RefObject<HTMLElement | null>, !!target);
  useEffect(() => {
    const draw = (f: ChartFrame | null) => {
      const r = rectRef.current;
      if (!f || !r) return;
      const [[w, so], [e, n]] = visibleBounds(f.camera, { width: f.viewport.width, height: f.viewport.height - f.insets.bottom });
      const x0 = Math.max(0, lx(w)), x1 = Math.min(LW, lx(e)), y0 = Math.max(0, ly(n)), y1 = Math.min(LH, ly(so));
      r.setAttribute("x", x0.toFixed(1));
      r.setAttribute("y", y0.toFixed(1));
      r.setAttribute("width", Math.max(3, x1 - x0).toFixed(1));
      r.setAttribute("height", Math.max(3, y1 - y0).toFixed(1));
    };
    draw(getFrame());
    return onFrame(draw);
  }, [target]);
  if (!target) return null;
  return createPortal(
    <svg ref={insetRef} className={s.locator} width={LW + 8} height={LH + 8} viewBox={`-4 -4 ${LW + 8} ${LH + 8}`} role="img" aria-label="Where this view is on the California coast" data-testid="nx-locator">
      <rect x={-4} y={-4} width={LW + 8} height={LH + 8} className={s.locatorGround} />
      <path d={COAST_D} className={s.locatorCoast} />
      <rect ref={rectRef} className={s.locatorView} />
    </svg>,
    target,
  );
}
