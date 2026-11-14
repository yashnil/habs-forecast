"use client";

import { useLayoutEffect, useSyncExternalStore, type ReactNode } from "react";

/**
 * The phone's bottom sheet (v14 README §3). On a phone the chart is a strip at the top and the page reads in one
 * sheet under it; the workspace owns the sheet (drag, detents, the handle). A page says how much chart its resting
 * sheet leaves: `<Sheet map={0.42}>` keeps 42 % of the screen for the chart strip (the sheet rests below it).
 * Default 0.27, the Today composition. Taller where the page allows (a trip's zone, a station and its neighbours):
 * the strip frames the page's scene (fitted to its meaningful features) in whatever the sheet leaves.
 *
 * On a wide screen <Sheet> renders its children as they are.
 */
type SheetState = { map: number; n: number };
const DEFAULT: SheetState = { map: 0.27, n: 0 };
let state: SheetState = DEFAULT;
const listeners = new Set<() => void>();
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
};
export const getSheetRest = () => state;
export const useSheetRest = () => useSyncExternalStore(subscribe, getSheetRest, () => DEFAULT);

export function setSheetRest(map: number) {
  const m = Math.max(0.16, Math.min(0.7, map));
  if (Math.abs(m - state.map) < 0.001) return;
  state = { map: m, n: state.n + 1 };
  listeners.forEach((l) => l());
}

export function Sheet({ map = DEFAULT.map, children }: { map?: number; children?: ReactNode }) {
  useLayoutEffect(() => {
    setSheetRest(map);
    return () => setSheetRest(DEFAULT.map);
  }, [map]);
  return <>{children}</>;
}
