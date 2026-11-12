"use client";

import type { Map as MapLibreMap } from "maplibre-gl";
import type { Bounds } from "./mercator";

/**
 * The one live map, for the few modules that must reach it (the opening's timeline, tests). Feature code
 * declares scenes and marks instead of touching the map.
 */
let map: MapLibreMap | null = null;
let ready = false;
const waiters = new Set<(m: MapLibreMap) => void>();

export function setChartMap(m: MapLibreMap | null, isReady = false) {
  map = m;
  ready = !!m && isReady;
  if (m && ready) {
    waiters.forEach((w) => w(m));
    waiters.clear();
  }
}

export const getChartMap = () => (ready ? map : null);

/** Resolves with the map once its style and data have drawn; never if the chart is static. */
export function whenChartReady(): Promise<MapLibreMap> {
  if (map && ready) return Promise.resolve(map);
  return new Promise((res) => waiters.add(res));
}

/* ---------------------------------------------------------------- commands (v13) */


/**
 * What a page or the chart's own buttons may ask of the camera, without touching the map: step the zoom, fit
 * the scene again ("back to port"), fit a box of data, or keep a place in view above whatever covers the chart
 * (the phone sheet at its present height). The chart host carries each out with `pan`, or at once under
 * reduced motion; on the static tier a command does nothing.
 */
export type ChartCommand =
  | { type: "zoom"; by: number }
  | { type: "refit" }
  | { type: "fit"; bounds: Bounds; maxZoom?: number }
  | { type: "keep"; lon: number; lat: number; margin?: number };

const commandListeners = new Set<(c: ChartCommand) => void>();
export function onChartCommand(l: (c: ChartCommand) => void) {
  commandListeners.add(l);
  return () => {
    commandListeners.delete(l);
  };
}
const send = (c: ChartCommand) => commandListeners.forEach((l) => l(c));

/** One zoom step in (1) or out (-1). */
export const zoomChart = (by: number) => send({ type: "zoom", by });
/** Back to the page's own view (its port, its station): the scene fitted again. */
export const refitChart = () => send({ type: "refit" });
/** Fit a box (the extent of the data in view) into the free part of the chart. */
export const fitChartBounds = (bounds: Bounds, maxZoom?: number) => send({ type: "fit", bounds, maxZoom });
/**
 * Keep a place in view: if it is under the phone sheet (at the height the sheet has now), a panel, or within
 * `margin` px of the free area's edge, the chart pans just enough to bring it clear. Already clear: nothing moves.
 */
export const keepChartPointInView = (lon: number, lat: number, margin = 56) => send({ type: "keep", lon, lat, margin });
