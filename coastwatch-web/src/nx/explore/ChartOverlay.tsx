"use client";

import { motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ink } from "@/nx/ds";
import { getFrame, onFrame, type ChartFrame } from "@/nx/chart/runtime";
import { visibleBounds } from "@/nx/chart/mercator";
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

export function ReadingOnChart({ lat, lon, lon0, lon1, hoverLon }: { lat: number; lon: number; lon0: number | null; lon1: number | null; hoverLon: number | null }) {
  const target = useTarget();
  const reduce = useReducedMotion();
  const lineRef = useRef<SVGPathElement>(null);
  const ptRef = useRef<SVGGElement>(null);
  const hovRef = useRef<SVGCircleElement>(null);
  const args = useRef({ lat, lon, lon0, lon1, hoverLon });
  args.current = { lat, lon, lon0, lon1, hoverLon };
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
    draw(getFrame());
    const off = onFrame(draw);
    return off;
  }, [target]);
  useEffect(() => {
    // a new hover or point: redraw at once with the camera where it is
    const f = getFrame();
    if (f) {
      const a = args.current;
      const h = hovRef.current;
      if (h && a.hoverLon !== null) {
        const q = f.project(a.hoverLon, a.lat);
        h.setAttribute("cx", q.x.toFixed(1));
        h.setAttribute("cy", q.y.toFixed(1));
        h.setAttribute("visibility", "visible");
      } else h?.setAttribute("visibility", "hidden");
      const p = f.project(a.lon, a.lat);
      ptRef.current?.setAttribute("transform", `translate(${p.x.toFixed(1)},${p.y.toFixed(1)})`);
      if (lineRef.current && a.lon0 !== null && a.lon1 !== null) {
        const q0 = f.project(a.lon0, a.lat), q1 = f.project(a.lon1, a.lat);
        lineRef.current.setAttribute("d", `M${q0.x.toFixed(1)},${q0.y.toFixed(1)}L${q1.x.toFixed(1)},${q1.y.toFixed(1)}`);
      }
    }
  }, [lat, lon, lon0, lon1, hoverLon]);
  if (!target) return null;
  return createPortal(
    <svg className={s.onChart} aria-hidden="true" data-testid="nx-reading-on-chart">
      {lon0 !== null && lon1 !== null ? <motion.path key={`${lat}-${lon}`} ref={lineRef} className={s.chartTransect} initial={reduce ? false : { pathLength: 0 }} animate={{ pathLength: 1 }} transition={ink(1.1)} /> : null}
      <circle ref={hovRef} r={5} className={s.chartHover} visibility="hidden" />
      <g ref={ptRef}>
        <circle r={7} className={s.chartPointRing} />
        <circle r={2.5} className={s.chartPointDot} />
      </g>
    </svg>,
    target,
  );
}

// the locator's own small projection: equirectangular, scaled to the coast's box
const LON0 = -125.2, LON1 = -116.6, LAT0 = 32.2, LAT1 = 42.2;
const LW = 64, LH = Math.round((LW * (LAT1 - LAT0)) / ((LON1 - LON0) * Math.cos((37 * Math.PI) / 180)));
const lx = (lon: number) => ((lon - LON0) / (LON1 - LON0)) * LW;
const ly = (lat: number) => ((LAT1 - lat) / (LAT1 - LAT0)) * LH;
const COAST_D = (COAST as [number, number][]).map(([lon, lat], i) => `${i ? "L" : "M"}${lx(lon).toFixed(1)},${ly(lat).toFixed(1)}`).join("");

export function Locator() {
  const target = useTarget();
  const rectRef = useRef<SVGRectElement>(null);
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
    <svg className={s.locator} width={LW + 8} height={LH + 8} viewBox={`-4 -4 ${LW + 8} ${LH + 8}`} role="img" aria-label="Where this view is on the California coast" data-testid="nx-locator">
      <rect x={-4} y={-4} width={LW + 8} height={LH + 8} className={s.locatorGround} />
      <path d={COAST_D} className={s.locatorCoast} />
      <rect ref={rectRef} className={s.locatorView} />
    </svg>,
    target,
  );
}
