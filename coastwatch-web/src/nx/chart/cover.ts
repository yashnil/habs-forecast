"use client";

import { useLayoutEffect, type RefObject } from "react";
import type { Insets } from "./mercator";
import type { Rect } from "./placement";

/**
 * What lies over the sheet: the phone margin at its resting detent, Research's record panel, a readout. A page
 * (or the shell) registers the element that covers part of the chart; the chart host reads every cover each
 * time it places the camera and the names, so a scene's box is fitted into the free part of the sheet and
 * nothing written on the chart (a port name, a sounding, the attribution) lands under a panel.
 *
 * `side` says which edge the cover is attached to: its depth along that edge becomes camera padding and keeps
 * the placement area clear. A cover with side "none" is only an obstacle for the names.
 */
export type CoverSide = "left" | "right" | "top" | "bottom" | "none";
type Cover = { el: HTMLElement; side: CoverSide };

const covers = new Map<string, Cover>();
const listeners = new Set<() => void>();
const ro = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(() => listeners.forEach((l) => l()));

export function registerCover(id: string, el: HTMLElement, side: CoverSide) {
  covers.set(id, { el, side });
  ro?.observe(el);
  listeners.forEach((l) => l());
  return () => {
    ro?.unobserve(el);
    if (covers.get(id)?.el === el) covers.delete(id);
    listeners.forEach((l) => l());
  };
}

export function onCoversChange(l: () => void) {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

/** Registers `ref`'s element as covering the chart along `side` while `active`. */
export function useChartCover(id: string, ref: RefObject<HTMLElement | null>, side: CoverSide, active = true) {
  useLayoutEffect(() => {
    const el = ref.current;
    if (!active || !el) return;
    return registerCover(id, el, side);
  }, [id, ref, side, active]);
}

/**
 * The covers measured against the frame: insets per edge (camera padding, the placement area) and the rects of
 * covers that sit on no edge (obstacles). A cover that does not overlap the frame counts for nothing.
 */
export function measureCovers(frame: DOMRect): { insets: Insets; obstacles: Rect[] } {
  const insets: Insets = { top: 0, right: 0, bottom: 0, left: 0 };
  const obstacles: Rect[] = [];
  for (const { el, side } of covers.values()) {
    if (!el.isConnected) continue;
    const r = el.getBoundingClientRect();
    if (!r.width || !r.height || r.right <= frame.left || r.left >= frame.right || r.bottom <= frame.top || r.top >= frame.bottom) continue;
    if (side === "right") insets.right = Math.max(insets.right, frame.right - r.left);
    else if (side === "left") insets.left = Math.max(insets.left, r.right - frame.left);
    else if (side === "bottom") insets.bottom = Math.max(insets.bottom, frame.bottom - r.top);
    else if (side === "top") insets.top = Math.max(insets.top, r.bottom - frame.top);
    else obstacles.push({ x0: r.left - frame.left, y0: r.top - frame.top, x1: r.right - frame.left, y1: r.bottom - frame.top });
  }
  for (const k of ["top", "right", "bottom", "left"] as const) insets[k] = Math.max(0, Math.round(insets[k]));
  return { insets, obstacles };
}

/* ---------------------------------------------------------------- obstacles (v13) */

/** An element the chart's lettering keeps clear of (a locator inset, a legend set on the water). Not camera padding. */
export function useChartObstacle(id: string, ref: RefObject<HTMLElement | null>, active = true) {
  useChartCover(id, ref, "none", active);
}

/**
 * Something drawn on the chart in its own coordinates that lettering must keep clear of (a transect line, a
 * read cell's outline): a function from the frame's projector to rects in frame px, asked at every placement,
 * so it follows the camera. Return [] when there is nothing to avoid.
 */
export type ObstacleFn = (project: (lon: number, lat: number) => { x: number; y: number }) => Rect[];
const drawn = new Map<string, ObstacleFn>();
export function registerChartObstacle(id: string, fn: ObstacleFn) {
  drawn.set(id, fn);
  listeners.forEach((l) => l());
  return () => {
    if (drawn.get(id) === fn) drawn.delete(id);
    listeners.forEach((l) => l());
  };
}
export function drawnObstacles(project: (lon: number, lat: number) => { x: number; y: number }): Rect[] {
  const out: Rect[] = [];
  for (const fn of drawn.values()) {
    try {
      out.push(...fn(project));
    } catch {
      // a page's obstacle must never stop the chart placing its names
    }
  }
  return out;
}

/** Rects along a line between two places, `pad` px to each side: for keeping names off a transect. */
export function lineObstacle(a: { x: number; y: number }, b: { x: number; y: number }, pad = 8, step = 48): Rect[] {
  const n = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / step));
  const out: Rect[] = [];
  for (let i = 0; i < n; i++) {
    const x0 = a.x + ((b.x - a.x) * i) / n, y0 = a.y + ((b.y - a.y) * i) / n;
    const x1 = a.x + ((b.x - a.x) * (i + 1)) / n, y1 = a.y + ((b.y - a.y) * (i + 1)) / n;
    out.push({ x0: Math.min(x0, x1) - pad, y0: Math.min(y0, y1) - pad, x1: Math.max(x0, x1) + pad, y1: Math.max(y0, y1) + pad });
  }
  return out;
}
