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
