"use client";

import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { onFrame, registerMark, updateMark, type MarkSpec } from "./runtime";
import { useScene } from "./scene";
import { CANYON_AXIS, CANYON_NAME, WATER_NAMES } from "./features";

export type ChartPort = { key: string; name: string; lon: number; lat: number; pilot: boolean };

/**
 * A mark on the chart: content placed beside its lon/lat by the host's collision pass, optionally a symbol
 * drawn exactly at the point. Feature engineers render ds Sounding and ChartNote here:
 *
 *   <ChartMarks marks={[{ id: "buoy-46236", lon: -122.1, lat: 36.76, priority: 60, leader: true,
 *                         content: <Sounding … /> }]} />
 *
 * Priority decides who stays when space is short (the reader's port 100, soundings 50–90, other ports 20–30,
 * water names under 10). The placed box carries `data-anchor` and the CSS variables `--nx-px`/`--nx-py` (the
 * point, relative to the box) for drawing a custom leader.
 */
export type ChartMark = MarkSpec & { content: ReactNode; point?: ReactNode; label?: string };

function Mark({ mark }: { mark: ChartMark }) {
  const ref = useRef<HTMLDivElement>(null);
  const { content, point, label, ...spec } = mark;
  const key = JSON.stringify(spec);
  useLayoutEffect(() => {
    if (!ref.current) return;
    return registerMark(JSON.parse(key) as MarkSpec, ref.current);
    // registered once per element; later changes go through updateMark
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useLayoutEffect(() => {
    updateMark(JSON.parse(key) as MarkSpec);
  }, [key]);
  return (
    <>
      {point ? (
        <div className="n12-point" data-nx-point={mark.id} aria-hidden="true">
          {point}
        </div>
      ) : null}
      <div ref={ref} className="n12-mark" aria-label={label}>
        {content}
      </div>
    </>
  );
}

/** Marks rendered into the chart's overlay from anywhere in the page (a portal; client only). */
export function ChartMarks({ marks }: { marks: ChartMark[] }) {
  const [target, setTarget] = useState<HTMLElement | null>(null);
  useEffect(() => setTarget(document.getElementById("nx-chart-marks")), []);
  if (!target) return null;
  return createPortal(
    <>
      {marks.map((m) => (
        <Mark key={m.id} mark={m} />
      ))}
    </>,
    target,
  );
}

const PortSymbol = ({ focus }: { focus: boolean }) => (
  <svg width={focus ? 22 : 12} height={focus ? 22 : 12} viewBox="-12 -12 24 24" className={focus ? "n12-port-sym n12-port-sym-focus" : "n12-port-sym"}>
    {focus ? <circle r="9.5" className="n12-port-ring" /> : null}
    <circle r={focus ? 3.6 : 4.2} className="n12-port-dot" />
  </svg>
);

/** The chart's own names: water features (Literata italic) and ports (upright sans, circled dot). */
export function ChartLabels({ ports }: { ports: ChartPort[] }) {
  const scene = useScene();
  const focus = scene?.focus?.key ?? null;
  const marks: ChartMark[] = [
    ...WATER_NAMES.map((w) => ({
      id: `water-${w.id}`,
      lon: w.lon,
      lat: w.lat,
      priority: w.priority,
      minZoom: w.minZoom,
      maxZoom: w.maxZoom,
      anchors: ["center" as const],
      content: (
        <span className="n12-water-name" style={{ fontSize: w.size, letterSpacing: `${w.spacing ?? 0.04}em` }}>
          {w.name}
        </span>
      ),
    })),
    ...ports.map((p) => {
      const isFocus = p.key === focus;
      return {
        id: `port-${p.key}`,
        lon: p.lon,
        lat: p.lat,
        priority: isFocus ? 100 : p.pilot ? 30 : 20,
        minZoom: isFocus ? 0 : p.pilot ? 6.4 : 8.6,
        keep: isFocus,
        pointRadius: isFocus ? 11 : 6,
        offset: isFocus ? 16 : 10,
        anchors: ["right" as const, "left" as const, "top-right" as const, "bottom-right" as const, "top-left" as const, "bottom-left" as const],
        point: <PortSymbol focus={isFocus} />,
        content: <span className={isFocus ? "n12-port-name n12-port-name-focus" : "n12-port-name"}>{p.name}</span>,
        label: p.name,
      };
    }),
  ];
  return (
    <div className="n12-labels" aria-hidden="true">
      {marks.map((m) => (
        <Mark key={m.id} mark={m} />
      ))}
    </div>
  );
}

/** "Monterey Canyon", set along the canyon's own axis (west to east, so it reads left to right). */
export function CanyonName() {
  const pathRef = useRef<SVGPathElement>(null);
  const textRef = useRef<SVGTextElement>(null);
  useEffect(
    () =>
      onFrame((f) => {
        const path = pathRef.current;
        const text = textRef.current;
        if (!path || !text) return;
        const show = f.camera.zoom >= CANYON_NAME.minZoom;
        const pts = [...CANYON_AXIS].reverse().map(([lon, lat]) => f.project(lon, lat));
        const onScreen = pts.filter((p) => p.x > 0 && p.y > 0 && p.x < f.viewport.width && p.y < f.viewport.height - f.insets.bottom);
        if (!show || onScreen.length < pts.length * 0.6) {
          text.setAttribute("visibility", "hidden");
          return;
        }
        // a smooth curve through the axis points (Catmull-Rom as cubic Béziers)
        let d = `M${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
        for (let i = 0; i < pts.length - 1; i++) {
          const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
          const c1 = { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 };
          const c2 = { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 };
          d += `C${c1.x.toFixed(1)},${c1.y.toFixed(1)} ${c2.x.toFixed(1)},${c2.y.toFixed(1)} ${p2.x.toFixed(1)},${p2.y.toFixed(1)}`;
        }
        path.setAttribute("d", d);
        text.setAttribute("visibility", "visible");
      }),
    [],
  );
  return (
    <svg className="n12-canyon-name" aria-hidden="true">
      <path id="nx-canyon-axis" ref={pathRef} fill="none" />
      <text ref={textRef} visibility="hidden" className="n12-water-name-svg" style={{ fontSize: CANYON_NAME.size }} dy="-6">
        <textPath href="#nx-canyon-axis" startOffset="58%" textAnchor="middle">
          {CANYON_NAME.name}
        </textPath>
      </text>
    </svg>
  );
}
