"use client";

import { useSyncExternalStore } from "react";

/**
 * A place lit on the chart from outside it: the sample under a plot's crosshair, a row hovered in a list
 * (v13 README §4, PlotSync `onPlace`). One at a time; null clears it. The chart host draws a ring there.
 */
export type ChartHighlight = { lon: number; lat: number; label?: string } | null;

let current: ChartHighlight = null;
const listeners = new Set<() => void>();

export function setChartHighlight(h: ChartHighlight | [number, number]) {
  const next = Array.isArray(h) ? { lon: h[0], lat: h[1] } : h;
  if (next?.lon === current?.lon && next?.lat === current?.lat && next?.label === current?.label) return;
  current = next;
  listeners.forEach((l) => l());
}

export const useChartHighlight = (): ChartHighlight =>
  useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => current,
    () => null,
  );
