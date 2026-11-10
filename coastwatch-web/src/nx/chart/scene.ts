"use client";

import { useLayoutEffect, useSyncExternalStore } from "react";
import type { Bounds, LngLat } from "./mercator";

/**
 * A page's view of the one chart (README §2.1: "Pages are views of that sheet plus a margin"). A page declares
 * its scene with <ChartScene scene={…} />, which renders nothing: it writes the scene in a layout effect, so
 * the camera starts moving in the same commit as the new page (D-motion-feasibility §3). The chart host
 * (src/nx/chart/ChartHost.tsx) owns the only MapLibre instance and pans to each new scene with the `pan`
 * preset; the canvas is never rebuilt.
 */
export type SceneView =
  /** fit this box in the free part of the sheet (the margin, bars and phone sheet excluded) */
  | { bounds: Bounds; maxZoom?: number }
  /** cover the whole sheet with this box, as CSS `cover` would (the opening's still registers to it) */
  | { cover: Bounds; position?: [number, number] };

export type ChartScene = {
  /** stable per view: a new id is a new destination for the camera */
  id: string;
  view: SceneView;
  /** the reader's port, drawn with the heavier port mark and never dropped */
  focus?: { key: string; name: string; lon: number; lat: number } | null;
  /** may the reader pan and zoom (true by default) */
  interactive?: boolean;
  /** index isobath depth labels ("200 m"), on by default */
  depthLabels?: boolean;
};

let current: ChartScene | null = null;
const listeners = new Set<() => void>();

export function setScene(scene: ChartScene) {
  if (current && current.id === scene.id && JSON.stringify(current) === JSON.stringify(scene)) return;
  current = scene;
  listeners.forEach((l) => l());
}

export const getScene = () => current;

export function subscribeScene(l: () => void) {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

export function useScene(): ChartScene | null {
  return useSyncExternalStore(subscribeScene, getScene, () => null);
}

/** Declares this page's view of the chart. Renders nothing. */
export function ChartScene({ scene }: { scene: ChartScene }) {
  const key = JSON.stringify(scene);
  useLayoutEffect(() => {
    setScene(JSON.parse(key) as ChartScene);
  }, [key]);
  return null;
}

/** A box around a point, `km` on each side (a port's view). */
export function boundsAround([lon, lat]: LngLat, kmX: number, kmY = kmX): Bounds {
  const dLat = kmY / 111.32;
  const dLon = kmX / (111.32 * Math.cos((lat * Math.PI) / 180));
  return [
    [lon - dLon, lat - dLat],
    [lon + dLon, lat + dLat],
  ];
}
