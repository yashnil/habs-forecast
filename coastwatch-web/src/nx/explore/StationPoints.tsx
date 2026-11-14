"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { getFrame, onFrame, type ChartFrame } from "@/nx/chart/runtime";
import type { ExploreBuoy, ExploreStation } from "./model";
import s from "./explore.module.css";

/**
 * The Stations layer: CalHABMAP shore stations (circles) and NDBC buoys (diamonds) as real buttons on the chart,
 * each at its own position, following the camera every frame without re-rendering React. Opening one shows its
 * own latest dated values (the readout); nothing is drawn where no station is.
 */
export type StationPick = { kind: "shore"; id: string } | { kind: "buoy"; id: string };

type Pt = { key: string; kind: "shore" | "buoy"; id: string; name: string; lat: number; lon: number; label: string };

export function StationPoints({ stations, buoys, selected, onSelect }: { stations: ExploreStation[]; buoys: ExploreBuoy[]; selected: StationPick | null; onSelect: (p: StationPick) => void }) {
  const [target, setTarget] = useState<HTMLElement | null>(null);
  useEffect(() => setTarget(document.getElementById("nx-chart-marks")), []);
  const refs = useRef(new Map<string, HTMLButtonElement>());
  const pts: Pt[] = [
    ...buoys.map((b) => ({ key: `buoy:${b.id}`, kind: "buoy" as const, id: b.id, name: b.name, lat: b.lat, lon: b.lon, label: `NDBC station ${b.id}, ${titleCase(b.name)}. Open its latest reading.` })),
    ...stations.map((st) => ({ key: `shore:${st.id}`, kind: "shore" as const, id: st.id, name: st.name, lat: st.lat, lon: st.lon, label: `${st.name}, CalHABMAP shore station. Open its latest samples.` })),
  ];
  const ptsRef = useRef(pts);
  ptsRef.current = pts;
  useEffect(() => {
    if (!target) return;
    const draw = (f: ChartFrame | null) => {
      if (!f) return;
      for (const p of ptsRef.current) {
        const el = refs.current.get(p.key);
        if (!el) continue;
        const q = f.project(p.lon, p.lat);
        const inside = q.x > -20 && q.y > -20 && q.x < f.viewport.width + 20 && q.y < f.viewport.height + 20;
        el.style.transform = `translate(${q.x.toFixed(1)}px, ${q.y.toFixed(1)}px)`;
        el.style.visibility = inside ? "visible" : "hidden";
      }
    };
    draw(getFrame());
    return onFrame(draw);
  }, [target, stations, buoys]);
  if (!target) return null;
  const sel = selected ? `${selected.kind}:${selected.id}` : null;
  return createPortal(
    <div className={s.stations} data-testid="nx-stations">
      {pts.map((p) => (
        // design-lint-allow raw-button: a station is a point on the chart (a positioned target, no words); ds Button is a labelled control
        <button
          key={p.key}
          ref={(el) => {
            if (el) refs.current.set(p.key, el);
            else refs.current.delete(p.key);
          }}
          type="button"
          className={s.stationPt}
          data-kind={p.kind}
          data-on={sel === p.key ? "1" : "0"}
          aria-pressed={sel === p.key}
          aria-label={p.label}
          title={p.kind === "buoy" ? `NDBC ${p.id} · ${titleCase(p.name)}` : p.name}
          data-testid={`nx-station-${p.kind}-${p.id}`}
          onClick={(e) => {
            e.stopPropagation();
            onSelect({ kind: p.kind, id: p.id });
          }}
        >
          <span aria-hidden="true" />
          {sel === p.key ? <b className={s.stationName}>{p.kind === "buoy" ? `NDBC ${p.id}` : p.name}</b> : null}
        </button>
      ))}
    </div>,
    target,
  );
}

/** NDBC names come in capitals ("SANTA MARIA - 21NM NW of Point Arguello, CA"): the place, read as words. */
export function titleCase(name: string): string {
  return name.replace(/\b([A-Z])([A-Z]{2,})\b/g, (_, a: string, b: string) => a + b.toLowerCase());
}
