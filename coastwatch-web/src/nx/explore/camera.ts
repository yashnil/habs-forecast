/**
 * Explore's view of the chart, DOM-free. v11 opened with 40 % of the view on empty sea west of the data; here the
 * view is the data's own extent, or the reader's port with its sea before it, clipped to where the data is, so
 * the water with readings fills the sheet. The zoom is clamped so the coast never leaves the view: out to the
 * whole state, in to about a harbour (300 m pixels at a few screen pixels each).
 */
import type { Bounds } from "@/nx/chart/mercator";

/**
 * The data's extent as bounds for a scene's `within` (the view slides to stop at the grid's sea edges), or null.
 * East is left open: California's grids end at the coast, and land beyond a grid's east edge shows no edge.
 */
export const extentBounds = (extent: [number, number, number, number] | null): Bounds | null =>
  extent
    ? [
        [extent[0], extent[1]],
        [-114, extent[3]],
      ]
    : null;

export const EXPLORE_ZOOM: [number, number] = [5.2, 12.6];
/** California's coast, Oregon to Mexico. */
export const CA_COAST: Bounds = [
  [-124.9, 32.4],
  [-117.0, 42.0],
];

type Box = [number, number, number, number];

/**
 * The opening view. With a port: about 90 km of coast around it, the port right of centre so its sea fills the
 * view, clipped to the data (never more than a sliver beyond it). Without one: the data's extent, or the coast.
 */
export function exploreBounds(port: { lon: number; lat: number } | null, extent: Box | null): Bounds {
  if (!port) {
    if (!extent) return CA_COAST;
    return [
      [extent[0], extent[1]],
      [extent[2], extent[3]],
    ];
  }
  const kmLon = 111.32 * Math.cos((port.lat * Math.PI) / 180);
  const hw = 48 / kmLon;
  const hh = 38 / 111.32;
  let box: Box = [port.lon - 1.55 * hw, port.lat - hh, port.lon + 0.45 * hw, port.lat + hh];
  if (extent && extent[0] < box[2] && extent[2] > box[0] && extent[1] < box[3] && extent[3] > box[1]) {
    // keep a little beyond the data so its edge reads as an edge, and never less than 30 km across
    const padX = (extent[2] - extent[0]) * 0.03, padY = (extent[3] - extent[1]) * 0.03;
    const clip: Box = [Math.max(box[0], extent[0] - padX), Math.max(box[1], extent[1] - padY), Math.min(box[2], extent[2] + padX), Math.min(box[3], extent[3] + padY)];
    const minLon = 30 / kmLon, minLat = 30 / 111.32;
    if (clip[2] - clip[0] >= minLon && clip[3] - clip[1] >= minLat) box = [clip[0], clip[1], Math.max(clip[2], port.lon + 0.12 * hw), clip[3]];
  }
  return [
    [box[0], box[1]],
    [box[2], box[3]],
  ];
}
