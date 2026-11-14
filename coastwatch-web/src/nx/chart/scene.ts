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
  | {
      bounds: Bounds;
      maxZoom?: number;
      /** where the view may slide to so it stops at the data's edge (mercator.ts slideWithin) */
      within?: Bounds | null;
      /**
       * where the box sits in the free area when the area's shape leaves room beside it (0 west/north, 1 east/south;
       * centred by default). The coast runs down the sheet's east side, so [1, 0.5] gives the spare room to the sea.
       */
      align?: [number, number];
      /**
       * extra room at the top of the free area, px: what floats over the chart's top edge without being a measured
       * cover (the phone's bar and port control), so a statewide view's north edge is never under them
       */
      padTop?: number;
      /** fill the free area with the box, cropping its longer side (CSS `cover`), instead of fitting it whole */
      fill?: boolean;
      /**
       * the box to fit instead when the free area is tall and narrow (a desk page's chart pane: at least 1.25 times
       * taller than wide): a run of coast up and down from the place rather than a wide box that would show mostly land
       */
      tall?: Bounds;
    }
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
  /**
   * places the page writes on the chart itself (Research's shore stations): a port whose name one of them repeats
   * close by ("Santa Cruz" beside "Santa Cruz Wharf") is left off, so two names never crowd one place
   */
  named?: { name: string; lon: number; lat: number }[];
  /** how far the reader may zoom out and in on this view (the chart's own limits otherwise), so context is never lost */
  zoomRange?: [number, number];
  /**
   * the chart's own buttons on this view (zoom in, zoom out, back to the page's view), on by default where the
   * reader may pan; `fit` adds one more that fits a box of data ("Whole layer")
   */
  controls?: false | { fit?: { label: string; bounds: Bounds } };
  /**
   * the words on the chart's "back to the page's view" button when the view is not the focus's own (Explore's
   * statewide view keeps the port marked, but its view is the whole coast: "Whole coast", not "Back to Moss Landing")
   */
  refitLabel?: string;
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

/** A run of coast for a tall, narrow pane: `kmAlong` up and down the coast from the place, `kmAcross` each side (default a third). */
export function tallBoundsAround(at: LngLat, kmAlong: number, kmAcross = kmAlong / 3): Bounds {
  return boundsAround(at, kmAcross, kmAlong);
}
