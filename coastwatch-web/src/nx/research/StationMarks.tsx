"use client";

import { useEffect, useState } from "react";
import { ChartMarks, type ChartMark } from "@/nx/chart/Labels";
import { getFrame, onFrame } from "@/nx/chart/runtime";
import s from "./rx.module.css";

export type StationSymbol = { id: string; name: string; lat: number; lon: number };

/** Screen pixels per kilometre at a latitude and zoom (Web Mercator, 512 px tiles as MapLibre draws them). */
export function pxPerKm(lat: number, zoom: number): number {
  return (512 * 2 ** zoom) / (40_075.016686 * Math.cos((lat * Math.PI) / 180));
}

/** The chart's zoom, rounded so a pan never re-renders the marks (only a change of scale does). */
function useZoom(): number | null {
  const [z, setZ] = useState<number | null>(() => getFrame()?.camera.zoom ?? null);
  useEffect(() => onFrame((f) => setZ((o) => (o != null && Math.abs(o - f.camera.zoom) < 0.02 ? o : Math.round(f.camera.zoom * 100) / 100))), []);
  return z;
}

/**
 * Shore stations on the chart where they sample (v14 synthesis, Research): squares, the measured mark. The open
 * station is ultramarine with a paper halo and, with the model shown, the dashed circle of the C-HARM cells its model
 * panel is the median of. Neighbours near it are named with their newest sample of the variable ("pDA Oct 28 · 0.0013
 * ng/mL", or "pDA last measured 2022" with an open square when it is old); the rest are quiet pins. Each is a button
 * that opens that station's record.
 */
export function StationMarks({
  stations,
  selected,
  named,
  subline,
  radiusKm,
  onPick,
}: {
  stations: StationSymbol[];
  selected: string;
  named: Set<string>;
  subline: (id: string) => string | null;
  radiusKm: number | null;
  onPick: (id: string) => void;
}) {
  const zoom = useZoom();
  const marks: ChartMark[] = stations.map((st) => {
    const on = st.id === selected;
    const loud = on || named.has(st.id);
    const sub = loud ? subline(st.id) : null;
    const stale = !!sub && / last measured /.test(sub);
    const ring = on && radiusKm && zoom != null ? radiusKm * pxPerKm(st.lat, zoom) : 0;
    const R = Math.max(14, ring + 2);
    const kind = on ? "on" : !loud ? "quiet" : stale ? "stale" : "named";
    const point = (
      <svg width={2 * R} height={2 * R} viewBox={`${-R} ${-R} ${2 * R} ${2 * R}`} className={s.stationSym} data-kind={kind}>
        {on && ring > 14 ? <circle r={ring} className={s.footprint} /> : null}
        {on ? <rect x={-14} y={-14} width={28} height={28} rx={9} className={s.stationHalo} /> : null}
        {kind === "on" ? (
          <rect x={-9} y={-9} width={18} height={18} rx={5} className={s.stationOn} />
        ) : kind === "quiet" ? (
          <rect x={-5} y={-5} width={10} height={10} rx={3} className={s.stationQuiet} />
        ) : (
          <rect x={-7} y={-7} width={14} height={14} rx={4} className={kind === "stale" ? s.stationStale : s.stationNamed} />
        )}
      </svg>
    );
    return {
      id: `station-${st.id}`,
      lon: st.lon,
      lat: st.lat,
      priority: on ? 96 : loud ? 60 : 30,
      keep: on,
      minZoom: on || loud ? 0 : 5.4,
      pointRadius: on ? 14 : loud ? 9 : 6,
      offset: loud ? 14 : 0,
      layer: undefined,
      anchors: loud ? ["left", "top-left", "bottom-left", "right", "top", "bottom"] : ["center"],
      label: st.name,
      point,
      content: loud ? (
        <button type="button" className={s.stationName} data-on={on || undefined} aria-pressed={on} aria-label={`${st.name}${sub ? `, ${sub}` : ""}: open its record`} onClick={() => onPick(st.id)} data-testid={`chart-station-${st.id}`}>
          <b>{st.name}</b>
          {sub ? <small>{sub}</small> : null}
        </button>
      ) : (
        <button type="button" className={s.stationPin} aria-pressed={false} aria-label={`${st.name}: open its record`} title={st.name} onClick={() => onPick(st.id)} data-testid={`chart-station-${st.id}`} />
      ),
    };
  });
  return <ChartMarks marks={marks} />;
}
