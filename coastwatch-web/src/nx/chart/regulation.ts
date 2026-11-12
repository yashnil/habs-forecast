"use client";

import { useLayoutEffect, useSyncExternalStore } from "react";
export { eastmostShore, shoreLon } from "./shore";

/**
 * Regulation drawn on the chart (README §2 principle 4: magenta means regulation and nothing else; E T43: zones
 * are latitude bands drawn as lines, not polygons). Pure data and a small store; src/nx/chart/RegulationLayer.tsx draws.
 *
 * RAMP fishing zones (CDFW's Risk Assessment Mitigation Program for commercial Dungeness crab, the whale
 * entanglement rules): boundary latitudes from 14 CCR § 132.8(a)(7), as CDFW's BIOS ds3120 metadata states them
 * (https://filelib.wildlife.ca.gov/Public/BDB/GIS/BIOS/metadata/DS3120.html), the same bounds the published
 * `fisheries-status` dataset carries (`ramp.zones[].lat_north/lat_south`, curated in
 * data/curated/fisheries-status/status.yaml; tests/unit/regulation.test.ts holds the two together). Zone 6,
 * south of Point Conception, is not described there and is not drawn.
 */
export type RampZone = { zone: string; north: number; south: number; northName: string; southName: string };

/** degrees and decimal minutes, as the regulation writes them */
const dm = (d: number, m: number) => d + m / 60;

export const RAMP_ZONES: RampZone[] = [
  { zone: "1", north: dm(42, 0), south: dm(40, 10), northName: "California/Oregon border", southName: "Cape Mendocino" },
  { zone: "2", north: dm(40, 10), south: dm(38, 46.125), northName: "Cape Mendocino", southName: "Sonoma/Mendocino county line" },
  { zone: "3", north: dm(38, 46.125), south: dm(37, 11), northName: "Sonoma/Mendocino county line", southName: "Pigeon Point" },
  { zone: "4", north: dm(37, 11), south: dm(36, 0), northName: "Pigeon Point", southName: "Lopez Point" },
  { zone: "5", north: dm(36, 0), south: dm(34, 27), northName: "Lopez Point", southName: "Point Conception" },
];

/** The fisheries RAMP governs (commercial Dungeness; its zones also carry the recreational crab-trap rules). */
export const RAMP_FISHERIES = new Set(["dungeness_commercial", "dungeness_recreational"]);

/** The zone a latitude lies in (a point on a boundary counts in the zone north of it, as src/pilot/facts.ts reads it). */
export const rampZoneAt = (lat: number): string | null => RAMP_ZONES.find((z) => lat <= z.north && lat >= z.south)?.zone ?? null;

/** Every boundary once, north to south, with the place that names it. */
export const RAMP_BOUNDARIES: { lat: number; name: string }[] = [
  ...RAMP_ZONES.map((z) => ({ lat: z.north, name: z.northName })),
  { lat: RAMP_ZONES[RAMP_ZONES.length - 1].south, name: RAMP_ZONES[RAMP_ZONES.length - 1].southName },
];

/** A notice's official latitude band, drawn as a bracket and two limit lines. Never verified (T38). */
export type NoticeExtent = { id: string; label: string; south: number; north: number; count: number };

/** "northern anchovy" → "anchovy", "rock crab (internal organs)" → "rock crab": the species as a label says it. */
const speciesWord = (s: string | undefined) => (s ?? "").replace(/\s*\(.*\)$/, "").replace(/^northern /, "").trim();

/**
 * Notices whose official area is a latitude band, merged where two give the same limits (CDFW's and CDPH's
 * anchovy notices both run Pigeon Point to Point Lobos): one bracket, labelled with the count and the species.
 * The label never names a state the registry does not record; the layer adds "not verified" under each (T38).
 */
export function noticeExtents(items: { id: string; agency: string; action: string; species: string[]; band: { south: number; north: number } | null }[]): NoticeExtent[] {
  const out: (NoticeExtent & { sp: string })[] = [];
  for (const n of items) {
    if (!n.band) continue;
    const sp = speciesWord(n.species[0]);
    const same = out.find((e) => Math.abs(e.south - n.band!.south) < 0.002 && Math.abs(e.north - n.band!.north) < 0.002);
    if (same) {
      same.count += 1;
      same.id += `+${n.id}`;
      if (same.sp !== sp) same.sp = "";
      same.label = same.sp ? `${same.count} ${same.sp} notices` : `${same.count} notices`;
    } else out.push({ id: n.id, sp, label: `${n.agency} ${sp ? `${sp} ` : ""}${n.action.toLowerCase()}`, south: n.band.south, north: n.band.north, count: 1 });
  }
  return out.map((e) => ({ id: e.id, label: e.label, south: e.south, north: e.north, count: e.count }));
}

/** What a page asks the chart to draw in magenta. */
export type ChartRegulationSpec = {
  /** RAMP zone lines; `highlight` is the reader's zone (heavier, named first) */
  zones?: { highlight: string | null } | null;
  /** official notice latitude bands */
  extents?: NoticeExtent[];
};

let current: { owner: number; spec: ChartRegulationSpec } | null = null;
let seq = 0;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());

export const getRegulation = () => current?.spec ?? null;
export function subscribeRegulation(l: () => void) {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}
export function useRegulation(): ChartRegulationSpec | null {
  return useSyncExternalStore(subscribeRegulation, getRegulation, () => null);
}

/**
 * Declares what regulation this page draws on the chart; renders nothing. Like ChartScene it writes in a layout
 * effect, so the lines start drawing with the camera's pan; on leaving, the page's lines leave with it.
 */
export function ChartRegulation({ spec }: { spec: ChartRegulationSpec | null }) {
  const key = JSON.stringify(spec);
  useLayoutEffect(() => {
    const owner = ++seq;
    current = { owner, spec: JSON.parse(key) as ChartRegulationSpec };
    emit();
    return () => {
      if (current?.owner === owner) {
        current = null;
        emit();
      }
    };
  }, [key]);
  return null;
}
