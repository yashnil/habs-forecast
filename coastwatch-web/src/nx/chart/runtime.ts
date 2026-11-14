"use client";

import { useSyncExternalStore } from "react";
import type { Camera, Insets, Point, Viewport } from "./mercator";
import { project as mercProject } from "./mercator";
import type { Anchor, PlaceItem } from "./placement";

/**
 * What the chart host knows each frame, shared with everything drawn over the chart: the camera, the sheet's
 * size, the free area (insets) and a projector. Live, the projector is MapLibre's own (exact through a flight);
 * on the static tier it is the same Web Mercator maths over the baked image.
 */
export type ChartFrame = {
  camera: Camera;
  viewport: Viewport;
  insets: Insets;
  project: (lon: number, lat: number) => Point;
  live: boolean;
};

let frame: ChartFrame | null = null;
const frameListeners = new Set<(f: ChartFrame) => void>();

export function publishFrame(f: ChartFrame) {
  frame = f;
  frameListeners.forEach((l) => l(f));
}
export const getFrame = () => frame;
export function onFrame(l: (f: ChartFrame) => void) {
  frameListeners.add(l);
  return () => {
    frameListeners.delete(l);
  };
}

export function staticProjector(camera: Camera, viewport: Viewport) {
  return (lon: number, lat: number) => mercProject(camera, viewport, lon, lat);
}

/**
 * Everything written on the chart registers here: a lon/lat, a priority, the element that holds it. The host
 * measures each element, places all of them together (src/nx/chart/placement.ts) and writes their positions
 * straight to the DOM each frame, so a moving camera never re-renders React.
 */
export type MarkSpec = Omit<PlaceItem, "x" | "y" | "width" | "height"> & {
  lon: number;
  lat: number;
  /** below this zoom (or above maxZoom) the mark is not drawn */
  minZoom?: number;
  maxZoom?: number;
  /** a leader line and dot from the point to the box (a sounding placed away from its buoy) */
  leader?: boolean;
  /**
   * the map layer the mark belongs to (v14: "sea" for a buoy, "toxin" for a test or a station; mapLayers.ts): it is
   * drawn only while that layer's chip is on, fading in and out
   */
  layer?: string;
};

type Entry = { spec: MarkSpec; el: HTMLElement; w: number; h: number };
const marks = new Map<string, Entry>();
const markListeners = new Set<() => void>();
const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver((rs) => {
  for (const r of rs) {
    const id = (r.target as HTMLElement).dataset.nxMark;
    const e = id ? marks.get(id) : undefined;
    if (e) {
      e.w = (r.target as HTMLElement).offsetWidth;
      e.h = (r.target as HTMLElement).offsetHeight;
    }
  }
  markListeners.forEach((l) => l());
});

export function registerMark(spec: MarkSpec, el: HTMLElement) {
  el.dataset.nxMark = spec.id;
  marks.set(spec.id, { spec, el, w: el.offsetWidth, h: el.offsetHeight });
  ro?.observe(el);
  markListeners.forEach((l) => l());
  return () => {
    ro?.unobserve(el);
    if (marks.get(spec.id)?.el === el) marks.delete(spec.id);
    markListeners.forEach((l) => l());
  };
}

export function updateMark(spec: MarkSpec) {
  const e = marks.get(spec.id);
  if (e) {
    e.spec = spec;
    markListeners.forEach((l) => l());
  }
}

export const allMarks = () => [...marks.values()];
export function onMarksChange(l: () => void) {
  markListeners.add(l);
  return () => {
    markListeners.delete(l);
  };
}

export type { Anchor };

/**
 * The shape of the chart's free area (what the covers leave of the frame), for a page that chooses its scene by
 * it: `tall` is true in a desk page's narrow pane (at least 1.25 times taller than wide). Null until measured.
 * Most pages need only `view.tall` on their scene (scene.ts), which the host applies itself.
 */
export type ChartShape = { width: number; height: number; tall: boolean };
let shape: ChartShape | null = null;
function shapeNow(): ChartShape | null {
  const f = frame;
  if (!f) return shape;
  const w = Math.round(f.viewport.width - f.insets.left - f.insets.right), h = Math.round(f.viewport.height - f.insets.top - f.insets.bottom);
  if (!shape || shape.width !== w || shape.height !== h) shape = { width: w, height: h, tall: h >= w * 1.25 };
  return shape;
}
export function useChartShape(): ChartShape | null {
  return useSyncExternalStore(
    (l) => onFrame(() => l()),
    shapeNow,
    () => null,
  );
}
