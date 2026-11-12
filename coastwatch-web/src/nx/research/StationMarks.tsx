"use client";

import { useEffect, useState } from "react";
import { ChartMarks, type ChartMark } from "@/nx/chart/Labels";
import { getFrame, onFrame } from "@/nx/chart/runtime";
import s from "./research.module.css";

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
 * Shore stations written on the chart where they sample (principle 2): a solid square, the measured mark; the
 * open station inside a ring, with a dashed circle of the model's radius around it (the model strip is the
 * median of C-HARM cells inside it, T11). Each name is a button that opens that station's record.
 */
export function StationMarks({ stations, selected, radiusKm, onPick, leanLeft = false }: { stations: StationSymbol[]; selected: string; radiusKm: number | null; onPick: (id: string) => void; leanLeft?: boolean }) {
  const zoom = useZoom();
  const marks: ChartMark[] = stations.map((st) => {
    const on = st.id === selected;
    const ring = on && radiusKm && zoom != null ? radiusKm * pxPerKm(st.lat, zoom) : 0;
    const R = Math.max(12, ring + 2);
    return {
      id: `station-${st.id}`,
      lon: st.lon,
      lat: st.lat,
      priority: on ? 96 : 44,
      keep: on,
      minZoom: on ? 0 : 6.6,
      pointRadius: on ? 12 : 6,
      offset: on ? 14 : 8,
      // with the record set on the sheet's right, names go to the left of their stations, where the chart is free
      anchors: leanLeft ? ["left", "top-left", "bottom-left", "top", "bottom", "right"] : ["right", "left", "top-right", "bottom-right", "top-left", "bottom-left"],
      label: st.name,
      point: (
        <svg width={2 * R} height={2 * R} viewBox={`${-R} ${-R} ${2 * R} ${2 * R}`} className={s.stationSym} data-on={on || undefined}>
          {on && ring > 14 ? <circle r={ring} className={s.stationRadius} /> : null}
          {on ? <circle r={10} className={s.stationRing} /> : null}
          <rect x={on ? -4.5 : -3.75} y={on ? -4.5 : -3.75} width={on ? 9 : 7.5} height={on ? 9 : 7.5} className={s.stationDot} />
        </svg>
      ),
      content: (
        <button type="button" className={s.stationName} data-on={on || undefined} aria-pressed={on} aria-label={`${st.name}: open its record`} onClick={() => onPick(st.id)} data-testid={`chart-station-${st.id}`}>
          {st.name}
        </button>
      ),
    };
  });
  return <ChartMarks marks={marks} />;
}
