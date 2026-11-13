"use client";

/**
 * Where the page's ticket stands relative to its place (ds Ticket writes this; the chart host reads it). The camera
 * brings the place to the ticket's near edge, a stub's length away, so the ticket sits beside its port as in the
 * approved compositions; the ticket itself follows the place when the reader pans. One anchor at a time: the page's
 * lead ticket. Nothing here touches the map.
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
  /** the highest the ticket's top may go (under the bar and the layer chips) */
  top: number;
};

let anchor: CameraAnchor | null = null;
const listeners = new Set<() => void>();
export const getCameraAnchor = () => anchor;
export function setCameraAnchor(a: CameraAnchor | null) {
  const same = JSON.stringify(a) === JSON.stringify(anchor);
  if (same) return;
  const moved = !anchor || !a || anchor.id !== a.id || anchor.lon !== a.lon || anchor.lat !== a.lat || anchor.side !== a.side || Math.abs(anchor.width - a.width) > 1;
  anchor = a;
  // a ticket that only grew or shrank in height does not move the camera; a new place or a new ticket does
  if (moved) listeners.forEach((l) => l());
}
export function clearCameraAnchor(id: string) {
  if (anchor?.id === id) {
    anchor = null;
    listeners.forEach((l) => l());
  }
}
export function onCameraAnchor(l: () => void) {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}

/**
 * The screen offset (px) that brings the anchor's place to the ticket's edge, given where the camera would put it.
 * Pure: the host applies it to its fitted camera.
 */
export function anchorShift(a: CameraAnchor, p: { x: number; y: number }, vp: { width: number; height: number }): { dx: number; dy: number } {
  const tx = a.side === "east" ? vp.width - a.gutter - a.width - a.stub : a.gutter + a.width + a.stub;
  const minY = a.top + a.anchorY;
  const maxY = Math.max(minY, vp.height - a.gutter - (a.height - a.anchorY));
  const dy = p.y < minY ? p.y - minY : p.y > maxY ? p.y - maxY : 0;
  return { dx: p.x - tx, dy };
}
