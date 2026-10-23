"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import { Icon } from "@/v4/ui/Icon";
import type { LadderStation } from "./CoastLadder";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), { ssr: false, loading: () => null });

type Win = { __cwMap?: unknown; __cwLocator?: MlMap };

/** A circle of `km` around a point, as a polygon (64 sides; a drawing of a distance, not a model cell). */
export function circlePolygon(lat: number, lon: number, km: number, n = 64): GeoJSON.Polygon {
  const dLat = km / 111.32;
  const dLon = km / (111.32 * Math.cos((lat * Math.PI) / 180));
  const ring: [number, number][] = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI;
    ring.push([lon + dLon * Math.cos(a), lat + dLat * Math.sin(a)]);
  }
  return { type: "Polygon", coordinates: [ring] };
}

/** The view around a station that holds its whole neighbourhood circle, with a margin. */
export function neighbourhoodBounds(lat: number, lon: number, km: number): [[number, number], [number, number]] {
  const k = km * 1.12;
  const dLat = k / 111.32;
  const dLon = k / (111.32 * Math.cos((lat * Math.PI) / 180));
  return [
    [lon - dLon, lat - dLat],
    [lon + dLon, lat + dLat],
  ];
}

/**
 * The small linked map (v10): the selected station and the circle of model cells whose median the model
 * panel charts, over the C-HARM nowcast it is read from. Every other station is a point that selects it;
 * a station chosen on the ladder moves the map to it. The map is the Explore map (MapCanvas) without
 * its chrome; it is made after the page has settled (it is context, never the first thing to load).
 */
export function LocatorMap({
  stations,
  selected,
  radiusKm,
  forecast,
  onPick,
  height = 196,
  label,
}: {
  stations: LadderStation[];
  selected: LadderStation;
  radiusKm: number;
  forecast: { id: string; url: string; corners: number[][] } | null;
  onPick: (id: string) => void;
  height?: number;
  label: string;
}) {
  const map = useRef<MlMap | null>(null);
  const [ready, setReady] = useState(false);
  // made once the page is idle: the chart and its text come first
  const [on, setOn] = useState(false);
  useEffect(() => {
    const w = window as Window & { requestIdleCallback?: (cb: () => void, o?: { timeout: number }) => number; cancelIdleCallback?: (id: number) => void };
    if (w.requestIdleCallback) {
      const id = w.requestIdleCallback(() => setOn(true), { timeout: 1200 });
      return () => w.cancelIdleCallback?.(id);
    }
    const t = window.setTimeout(() => setOn(true), 400);
    return () => window.clearTimeout(t);
  }, []);
  const features = useMemo<GeoJSON.FeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: stations
        .filter((s) => s.id !== selected.id)
        .map((s) => ({ type: "Feature", geometry: { type: "Point", coordinates: [s.lon, s.lat] }, properties: { kind: "station", id: s.id, name: s.name, recent: false, label: s.name } })),
    }),
    [stations, selected.id],
  );
  const highlight = useMemo<GeoJSON.FeatureCollection>(
    () => ({
      type: "FeatureCollection",
      features: [
        { type: "Feature", geometry: circlePolygon(selected.lat, selected.lon, radiusKm), properties: { kind: "cell" } },
        { type: "Feature", geometry: { type: "Point", coordinates: [selected.lon, selected.lat] }, properties: { kind: "station" } },
      ],
    }),
    [selected, radiusKm],
  );
  const bounds = useMemo(() => neighbourhoodBounds(selected.lat, selected.lon, radiusKm), [selected, radiusKm]);
  // a station chosen elsewhere (the ladder, a link) moves the map to it
  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    m.fitBounds(bounds, { padding: 8, duration: reduce ? 0 : 600 });
  }, [bounds, ready]);
  return (
    <figure className="flex flex-col gap-1.5" data-testid="locator">
      <div className="nx-plate relative overflow-hidden" style={{ height }} data-testid="locator-map" data-station={selected.id} data-mounted={on}>
        {on ? (
          <MapCanvas
            forecast={forecast}
            satellite={null}
            satelliteAge={null}
            imagery={null}
            ports={null}
            showPorts={false}
            officialGeometry={null}
            showOfficial={false}
            selectedPort={null}
            inspect={null}
            chrome="none"
            stations={features}
            clusterStations
            onStationPick={onPick}
            highlight={highlight}
            initialBounds={bounds}
            initialPadding={8}
            onPort={() => undefined}
            onPoint={() => undefined}
            onReady={(m) => {
              // the page's own map (window.__cwMap) is the grid map; this one is the locator
              const w = window as unknown as Win;
              if (w.__cwMap === m) delete w.__cwMap;
              w.__cwLocator = m;
              m.scrollZoom.disable();
              map.current = m;
              setReady(true);
            }}
          />
        ) : (
          <span className="absolute inset-0 grid place-items-center text-[12.5px] text-n-ink-3">Map of {selected.name}</span>
        )}
        <span className="nx-on-map pointer-events-none absolute bottom-1 right-1.5 text-[9.5px]">© OpenStreetMap · not for navigation</span>
      </div>
      <figcaption className="nx-margin text-[12.5px] leading-[17px]">{label}</figcaption>
    </figure>
  );
}

/** The station picker on a phone: a sheet with the coast ladder and the linked map. */
export function StationSheet({ open, onClose, children }: { open: boolean; onClose: () => void; children: React.ReactNode }) {
  const panel = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", key);
    const prev = document.activeElement as HTMLElement | null;
    panel.current?.focus();
    return () => {
      document.removeEventListener("keydown", key);
      prev?.focus?.();
    };
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex flex-col justify-end" data-testid="station-sheet">
      <button type="button" aria-label="Close the station list" className="absolute inset-0 bg-[rgba(2,6,9,0.62)]" onClick={onClose} />
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-labelledby="r-sheet-t"
        tabIndex={-1}
        style={{ animation: "nx-rise var(--nx-t-panel) var(--nx-ease-out) both" }}
        className="nx-plate relative max-h-[86dvh] overflow-y-auto rounded-b-none border-b-0 px-4 pb-[max(16px,env(safe-area-inset-bottom))] pt-3 outline-none"
      >
        <div className="sticky top-0 z-10 -mx-4 mb-2 flex items-center justify-between bg-n-surface px-4 pb-2">
          <h2 id="r-sheet-t" className="nx-caps">
            Shore stations, north to south
          </h2>
          <button type="button" onClick={onClose} className="nx-focus -mr-2 grid h-11 w-11 place-items-center rounded-[4px] text-n-ink-2" aria-label="Close" data-testid="station-sheet-close">
            <Icon name="close" className="h-5 w-5" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}
