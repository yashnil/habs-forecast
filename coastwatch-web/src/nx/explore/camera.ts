/**
 * Explore's view of the chart, DOM-free. v11 opened with 40 % of the view on empty sea west of the data; here the
 * view is the data's own extent, or the reader's port with its sea before it, clipped to where the data is, so
 * the water with readings fills the sheet. The zoom is clamped so the coast never leaves the view: out to the
 * whole state, in to about a harbour (300 m pixels at a few screen pixels each).
 */
import type { Bounds } from "@/nx/chart/mercator";
import { eastmostShore } from "@/nx/chart/shore";

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

/**
 * Does a box of published cells reach the water at a port? (Within a quarter of a degree: about 25 km.) A port
 * the layers do not reach is still framed as itself: the view never slides to another bay's data under a bar
 * that names this port.
 */
export function coversPort(box: [number, number, number, number] | null, port: { lon: number; lat: number } | null, pad = 0.25): boolean {
  if (!box || !port) return !!box;
  return port.lon >= box[0] - pad && port.lon <= box[2] + pad && port.lat >= box[1] - pad && port.lat <= box[3] + pad;
}

/** The data's extent as the scene's `within`, only when the reader's port is in it (or there is no port). */
export const withinFor = (extent: [number, number, number, number] | null, port: { lon: number; lat: number } | null): Bounds | null => (coversPort(extent, port) ? extentBounds(extent) : null);

export const EXPLORE_ZOOM: [number, number] = [5.2, 12.6];
/** California's coast, Oregon to Mexico. */
export const CA_COAST: Bounds = [
  [-124.9, 32.4],
  [-117.0, 42.0],
];

/** The Oregon border and the Mexican border, the latitudes the statewide view always holds. */
export const OREGON_BORDER = 42.0;
export const MEXICO_BORDER = 32.53;

/**
 * Explore California Waters' statewide box: the whole coast from the Oregon border to Mexico, with the sea the
 * published grids cover west of Cape Mendocino (their west edge is 125.5 W) and a sliver of land east of San
 * Diego. Beyond each border, so the borders themselves are inside the view, never on its edge.
 */
export const STATE_BOUNDS: Bounds = [
  [-125.2, MEXICO_BORDER - 0.08],
  [-117.0, OREGON_BORDER + 0.08],
];

/**
 * How far the reader may zoom out and in on the statewide view: out to the whole state on a 320 px phone (the
 * fit lands near zoom 4.2 there), in to about a harbour, so a smooth drill-down from the state to a pier never
 * changes view.
 */
export const COAST_ZOOM: [number, number] = [3.6, 12.6];

/** Does a view [w, s, e, n] hold the whole coast, Oregon border to Mexico? (The statewide view's promise.) */
export const holdsCoast = (view: [number, number, number, number]) => view[1] <= MEXICO_BORDER && view[3] >= OREGON_BORDER;

type Box = [number, number, number, number];

/**
 * The opening view. With a port: about 90 km of coast around it, the port right of centre so its sea fills the
 * view, clipped to the data (never more than a sliver beyond it). Without one: the data's extent, or the coast.
 */
export function exploreBounds(port: { lon: number; lat: number } | null, extent: Box | null): Bounds {
  if (!port) {
    if (!extent) return CA_COAST;
    // a grid's box runs on over the hills east of the coast: the view stops a little past the eastmost shore
    const shore = eastmostShore(extent[1], extent[3]);
    return [
      [extent[0], extent[1]],
      [shore == null ? extent[2] : Math.min(extent[2], shore + 0.08), extent[3]],
    ];
  }
  const kmLon = 111.32 * Math.cos((port.lat * Math.PI) / 180);
  const hw = 48 / kmLon;
  const hh = 38 / 111.32;
  // about 14 km of land east of the port, room for its name; the rest of the width is sea
  let box: Box = [port.lon - 1.55 * hw, port.lat - hh, port.lon + 0.3 * hw, port.lat + hh];
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

/**
 * The phone statewide view's extra top padding: the floating bar and its port control stand over the chart's top
 * edge without being a measured cover, so the fit keeps the Oregon border below the lower of the two, with a margin.
 * `pad` is the fit's own top padding (ChartHost sceneInsets: 20 px on a phone).
 */
export function barPadTop(barBottoms: number[], pad = 20, margin = 14): number {
  const bottom = Math.max(0, ...barBottoms.filter((b) => Number.isFinite(b)));
  return Math.max(0, Math.round(bottom + margin - pad));
}
