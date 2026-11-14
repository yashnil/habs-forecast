"use client";

import { useLayoutEffect, useSyncExternalStore } from "react";

/**
 * The map's four questions (v14 README §4, SYSTEM.md "Map layers"): each layer answers one question a fisherman asks,
 * and the chip row that switches them is also the legend. A page says which are on when it opens
 * (<MapLayers page="today" defaults={{ regulation: true, depth: true, sea: true }} />); the reader's own changes are
 * kept per page for the visit, so going back to Today finds the chips as they were left.
 *
 * The chart draws Regulation (RegulationLayer) and Depth (contours and fathom labels) itself. Sea now and Toxin tests
 * are marks a page writes (buoys, NWS zone, CDPH tests, shore stations): the page reads `useMapLayer("sea")` and
 * draws them only while the layer is on, or wraps them in <LayerMarks layer="sea">, which fades them in and out.
 *
 * A page may add its own data layers (Explore's fields): `extra` chips are listed after the four with the same
 * behaviour; their state is read with `useMapLayer(id)` like any other.
 */
export type MapLayerId = "regulation" | "depth" | "sea" | "toxin" | (string & {});
export type MapLayerDef = { id: MapLayerId; name: string; question: string; key: string; swatch: "reg" | "depth" | "sea" | "tox" | "data" };

export const MAP_LAYERS: MapLayerDef[] = [
  { id: "regulation", name: "Regulation", question: "What rules apply where I fish?", key: "Coral line: whale-zone limit. Tint: your zone.", swatch: "reg" },
  { id: "depth", name: "Depth (fm)", question: "How deep is it?", key: "Contours at 11, 27, 55 and 109 fm.", swatch: "depth" },
  { id: "sea", name: "Sea now", question: "What is the sea doing right now?", key: "Filled: measured at the buoy. Arrow: wind.", swatch: "sea" },
  { id: "toxin", name: "Toxin tests", question: "Is the shellfish tested near here?", key: "Diamond: CDPH crab test. Square: shore station.", swatch: "tox" },
];

export type LayerState = {
  page: string | null;
  on: Record<string, boolean>;
  /** a page's own data layers, chips after the four */
  extra: MapLayerDef[];
  /** when a layer was last switched (performance.now()), so the chart fades instead of cutting */
  changedAt: number;
};

const SERVER: LayerState = { page: null, on: { regulation: true, depth: true, sea: true, toxin: false }, extra: [], changedAt: -1e9 };
let state: LayerState = SERVER;
const memory = new Map<string, Record<string, boolean>>();
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
};
export const getMapLayers = () => state;
/** A drawing layer (RegulationLayer) arrived on or left the chart: the host writes `data-layers` again. */
export const LAYERS_DRAWN_EVENT = "nx-layers-drawn";
/** For code outside React (the chart host, depth labels). */
export const subscribeMapLayers = subscribe;

/** A page's defaults; the reader's choices for that page this visit win over them. */
export function setMapLayers(page: string, defaults: Partial<Record<MapLayerId, boolean>>, extra: MapLayerDef[] = []) {
  const base: Record<string, boolean> = { regulation: false, depth: false, sea: false, toxin: false };
  for (const e of extra) base[e.id] = false;
  const on = { ...base, ...(defaults as Record<string, boolean>), ...(memory.get(page) ?? {}) };
  if (state.page === page && JSON.stringify(state.on) === JSON.stringify(on) && JSON.stringify(state.extra) === JSON.stringify(extra)) return;
  state = { page, on, extra, changedAt: state.changedAt };
  emit();
}

/** The reader switching a layer (the chips) or a page doing it for them (a fishery picked that has an area). */
export function setMapLayer(id: MapLayerId, value: boolean) {
  if (!!state.on[id] === value) return;
  state = { ...state, on: { ...state.on, [id]: value }, changedAt: typeof performance === "undefined" ? 0 : performance.now() };
  if (state.page) memory.set(state.page, { ...(memory.get(state.page) ?? {}), [id]: value });
  emit();
}
export const toggleMapLayer = (id: MapLayerId) => setMapLayer(id, !state.on[id]);

export const useMapLayerState = (): LayerState => useSyncExternalStore(subscribe, getMapLayers, () => SERVER);
/** Is this layer on now? */
export function useMapLayer(id: MapLayerId): boolean {
  return useSyncExternalStore(
    subscribe,
    () => !!state.on[id],
    () => !!SERVER.on[id],
  );
}
/** The page-facing hook: which layers are on, and the setters. */
export function useMapLayers() {
  const s = useMapLayerState();
  return { on: s.on, page: s.page, layers: [...MAP_LAYERS, ...s.extra], set: setMapLayer, toggle: toggleMapLayer };
}

/** Declares this page's default layers (and its own data layers). Renders nothing; writes before paint. */
export function MapLayers({ page, defaults, extra }: { page: string; defaults: Partial<Record<MapLayerId, boolean>>; extra?: MapLayerDef[] }) {
  const key = JSON.stringify({ page, defaults, extra: extra ?? [] });
  useLayoutEffect(() => {
    const v = JSON.parse(key) as { page: string; defaults: Record<string, boolean>; extra: MapLayerDef[] };
    setMapLayers(v.page, v.defaults, v.extra);
  }, [key]);
  return null;
}

/** For tests and the catalog: forget the visit's choices. */
export function resetMapLayers() {
  memory.clear();
  state = SERVER;
  emit();
}
