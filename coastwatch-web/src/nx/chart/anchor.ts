"use client";

/**
 * Where the page's ticket stands relative to its place (ds Ticket writes this; the chart host reads it). The camera
 * brings the place to the ticket's near edge, a stub's length away, at the row `anchorY` of a ticket standing at its
 * preferred top, so the composition is the approved one at every size: in the prototype the ticket stands a tenth of
 * the screen in from the right (144 px at 1440, never under the gutter) and level with the nav for Alerts. The ticket
 * itself follows the place when the reader pans. One anchor at a time: the page's lead ticket. Nothing here touches
 * the map.
 */
export type CameraAnchor = {
  id: string;
  lon: number;
  lat: number;
  /** which side of the place the ticket stands */
  side: "east" | "west";
  /** the ticket's size, px */
  width: number;
  height: number;
  /** px from the ticket's top that stands level with the place */
  anchorY: number;
  /** px between the place and the ticket's edge (the stub) */
  stub: number;
  /** objects keep this far from the screen's edges */
  gutter: number;
  /** where the ticket's top stands (under the bar; under the layer chips when it reaches them) */
  top: number;
};

/** The prototype's margin between a ticket and its side of the screen: a tenth of the width, 144 px at most, never under the gutter. */
export const ticketMargin = (vpWidth: number, gutter: number) => Math.max(gutter, Math.min(144, Math.round(vpWidth * 0.1)));

let anchor: CameraAnchor | null = null;
let pendingClear = 0;
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());
export const getCameraAnchor = () => anchor;
export function setCameraAnchor(a: CameraAnchor | null) {
  // a ticket replacing another in the same frame (re-keyed on a new place) is one move of the camera, not two
  if (pendingClear) {
    cancelAnimationFrame(pendingClear);
    pendingClear = 0;
  }
  const same = JSON.stringify(a) === JSON.stringify(anchor);
  if (same) return;
  const moved = !anchor || !a || anchor.id !== a.id || anchor.lon !== a.lon || anchor.lat !== a.lat || anchor.side !== a.side || Math.abs(anchor.width - a.width) > 1 || anchor.anchorY !== a.anchorY || anchor.top !== a.top;
  anchor = a;
  // a ticket that only grew or shrank in height does not move the camera; a new place or a new ticket does
  if (moved) notify();
}
export function clearCameraAnchor(id: string) {
  if (anchor?.id !== id) return;
  anchor = null;
  if (typeof requestAnimationFrame === "undefined") return notify();
  if (pendingClear) cancelAnimationFrame(pendingClear);
  // wait a frame: the next page's (or the re-keyed) ticket usually writes its anchor before then
  pendingClear = requestAnimationFrame(() => {
    pendingClear = 0;
    if (!anchor) notify();
  });
}
export function onCameraAnchor(l: () => void) {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

/** Where the place should stand on screen for this anchor, px. */
export function anchorTarget(a: CameraAnchor, vp: { width: number; height: number }): { x: number; y: number } {
  const m = ticketMargin(vp.width, a.gutter);
  const x = a.side === "east" ? vp.width - m - a.width - a.stub : m + a.width + a.stub;
  // the ticket at its top when it fits; lower only as far as needed for it to fit above the foot
  const maxTop = Math.max(a.top, vp.height - a.gutter - a.height);
  const y = Math.min(a.top, maxTop) + a.anchorY;
  return { x, y };
}

/**
 * The screen offset (px) that brings the anchor's place to where the ticket composition wants it, given where the
 * camera would put it. Pure: the host applies it to its fitted camera.
 */
export function anchorShift(a: CameraAnchor, p: { x: number; y: number }, vp: { width: number; height: number }): { dx: number; dy: number } {
  const t = anchorTarget(a, vp);
  return { dx: p.x - t.x, dy: p.y - t.y };
}
