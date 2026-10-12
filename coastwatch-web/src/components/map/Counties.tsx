"use client";

import { useEffect, useId, useState } from "react";
import { createPortal } from "react-dom";
import { Layer, Source, useControl } from "react-map-gl/maplibre";
import type { ExpressionSpecification, IControl } from "maplibre-gl";
import { COUNTIES } from "@/content/copy";

/** Bundled county geography (pipeline/scripts/build_counties.py): borders between counties and one label each. */
export const COUNTIES_URL = "/basemap/ca-counties.json";
const PREF_KEY = "cw-counties-v1";
export type CountyPrefs = { lines: boolean; labels: boolean };
const DEFAULT: CountyPrefs = { lines: true, labels: true };

function readPrefs(): CountyPrefs {
  try {
    const v = window.localStorage.getItem(PREF_KEY);
    if (!v) return DEFAULT;
    const p = JSON.parse(v) as Partial<CountyPrefs>;
    return { lines: p.lines !== false, labels: p.labels !== false };
  } catch {
    return DEFAULT;
  }
}

/** The viewer's choice, remembered on this device (a convenience only; the default shows both). */
export function useCountyPrefs(): [CountyPrefs, (p: CountyPrefs) => void] {
  const [prefs, set] = useState<CountyPrefs>(DEFAULT);
  useEffect(() => set(readPrefs()), []);
  return [
    prefs,
    (p) => {
      set(p);
      try {
        window.localStorage.setItem(PREF_KEY, JSON.stringify(p));
      } catch {
        // not remembered
      }
    },
  ];
}

const LABEL_COAST: ExpressionSpecification = ["!=", ["get", "coast"], "inland"];

/**
 * County borders and names, quiet enough not to compete with the data: thin dashed lines between
 * counties (the coastline and state lines stay the basemap's own), and names that grow stronger for
 * the counties that face the ocean or San Francisco Bay. Reference only: county lines do not define
 * closures, crab zones, state waters or how far an advisory reaches.
 */
export function CountyLayers({ prefs, beforeId }: { prefs: CountyPrefs; beforeId?: string }) {
  return (
    <Source id="counties" type="geojson" data={COUNTIES_URL} attribution={COUNTIES.attribution}>
      <Layer
        id="county-borders"
        type="line"
        beforeId={beforeId}
        minzoom={5.2}
        filter={["==", ["get", "kind"], "border"]}
        layout={{ visibility: prefs.lines ? "visible" : "none", "line-join": "round" }}
        paint={{
          "line-color": "#d7e2ee",
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 5.2, 0, 6, 0.22, 9, 0.34],
          "line-width": ["interpolate", ["linear"], ["zoom"], 6, 0.6, 10, 1.2],
          "line-dasharray": [3, 2],
        }}
      />
      <Layer
        id="county-labels"
        type="symbol"
        beforeId={beforeId}
        minzoom={6.4}
        maxzoom={11}
        filter={["==", ["get", "kind"], "label"]}
        layout={{
          visibility: prefs.labels ? "visible" : "none",
          "text-field": ["upcase", ["get", "name"]],
          "text-font": ["Noto Sans Regular"],
          "text-size": ["interpolate", ["linear"], ["zoom"], 6.4, 10, 9, 12],
          "text-letter-spacing": 0.14,
          "text-max-width": 7,
          "text-padding": 6,
          "symbol-sort-key": ["case", LABEL_COAST, 0, 1],
          "text-optional": true,
        }}
        paint={{
          "text-color": ["case", LABEL_COAST, "rgba(232,238,245,0.78)", "rgba(232,238,245,0.48)"],
          "text-halo-color": "rgba(6,17,30,0.85)",
          "text-halo-width": 1.1,
        }}
      />
    </Source>
  );
}

class Holder implements IControl {
  el: HTMLDivElement;
  constructor() {
    this.el = document.createElement("div");
    this.el.className = "maplibregl-ctrl maplibregl-ctrl-group";
  }
  onAdd() {
    return this.el;
  }
  onRemove() {
    this.el.remove();
  }
}

/** A map control (above the zoom buttons): county lines and names on or off, with what they are not. */
export function CountyControl({ prefs, onChange }: { prefs: CountyPrefs; onChange: (p: CountyPrefs) => void }) {
  const ctl = useControl(() => new Holder(), { position: "bottom-right" });
  const [open, setOpen] = useState(false);
  const id = useId();
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);
  return createPortal(
    <div className="relative">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        aria-label="County lines and names"
        title="Counties"
        onClick={() => setOpen(!open)}
        data-testid="counties-toggle"
        className="!grid !h-[29px] !w-[29px] place-items-center"
      >
        <svg viewBox="0 0 20 20" width="18" height="18" aria-hidden fill="none" stroke="currentColor" strokeWidth="1.4" strokeDasharray="2.4 1.6">
          <path d="M3 4.5l5-1.5 4 2 5-1v11l-5 1.5-4-2-5 1z" />
          <path d="M8 3v13M12 5v12" />
        </svg>
      </button>
      {open && (
        <div
          id={id}
          role="group"
          aria-label="Counties"
          data-testid="counties-panel"
          className="theme-paper absolute bottom-0 right-[38px] w-[260px] rounded-lg bg-surface p-3 text-left text-[13px] leading-snug text-ink shadow-[0_6px_20px_rgba(6,17,30,0.3)]"
        >
          <p className="mb-1.5 font-semibold">Counties</p>
          <label className="flex min-h-[32px] items-center gap-2">
            <input type="checkbox" checked={prefs.lines} onChange={(e) => onChange({ ...prefs, lines: e.target.checked })} data-testid="counties-lines" />
            County lines
          </label>
          <label className="flex min-h-[32px] items-center gap-2">
            <input type="checkbox" checked={prefs.labels} onChange={(e) => onChange({ ...prefs, labels: e.target.checked })} data-testid="counties-labels" />
            County names
          </label>
          <p className="mt-2 text-ink-2" data-testid="counties-note">
            {COUNTIES.note}
          </p>
          <p className="mt-1.5 text-[12px] text-ink-3">{COUNTIES.source}</p>
        </div>
      )}
    </div>,
    ctl.el,
  );
}
