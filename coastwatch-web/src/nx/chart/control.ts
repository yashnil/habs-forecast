"use client";

import type { Map as MapLibreMap } from "maplibre-gl";

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
