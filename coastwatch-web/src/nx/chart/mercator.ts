/**
 * The sheet's projection: Web Mercator with MapLibre's 512 px world, pitch 0, north up. Pure, so the live
 * chart, the static tier's baked image, the opening's still, the neatline's ticks and the note placement
 * all agree to the pixel. The camera is always computed here (never MapLibre's own padding), and the live
 * map is moved to exactly this camera.
 */
export type LngLat = [number, number];
export type Bounds = [LngLat, LngLat]; // [[west, south], [east, north]]
export type Camera = { center: LngLat; zoom: number };
export type Viewport = { width: number; height: number };
export type Insets = { top: number; right: number; bottom: number; left: number };
export type Point = { x: number; y: number };

export const TILE = 512;
export const NO_INSETS: Insets = { top: 0, right: 0, bottom: 0, left: 0 };
const EARTH = 40075016.686;
const MAX_LAT = 85.051129;

/** Longitude to the world's 0..1 x. */
export const lngX = (lon: number) => (lon + 180) / 360;
/** Latitude to the world's 0..1 y (down). */
export function latY(lat: number) {
  const s = Math.sin((Math.max(-MAX_LAT, Math.min(MAX_LAT, lat)) * Math.PI) / 180);
  return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI);
}
export const xLng = (x: number) => x * 360 - 180;
export const yLat = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI;

export const worldSize = (zoom: number) => TILE * 2 ** zoom;

/** A lon/lat to screen pixels in a viewport looking at `cam`. */
export function project(cam: Camera, vp: Viewport, lon: number, lat: number): Point {
  const s = worldSize(cam.zoom);
  return { x: vp.width / 2 + (lngX(lon) - lngX(cam.center[0])) * s, y: vp.height / 2 + (latY(lat) - latY(cam.center[1])) * s };
}

export function unproject(cam: Camera, vp: Viewport, x: number, y: number): LngLat {
  const s = worldSize(cam.zoom);
  return [xLng(lngX(cam.center[0]) + (x - vp.width / 2) / s), yLat(latY(cam.center[1]) + (y - vp.height / 2) / s)];
}

/** The lon/lat box a camera shows. */
export function visibleBounds(cam: Camera, vp: Viewport): Bounds {
  const nw = unproject(cam, vp, 0, 0);
  const se = unproject(cam, vp, vp.width, vp.height);
  return [
    [nw[0], se[1]],
    [se[0], nw[1]],
  ];
}

/**
 * The camera that fits `bounds` inside the viewport less `insets` (the margin column, the phone sheet, the
 * bars), centred in that free area. `maxZoom` caps how close a tiny box may come.
 */
export function fitBounds(bounds: Bounds, vp: Viewport, insets: Insets = NO_INSETS, maxZoom = 13): Camera {
  const [[w, s], [e, n]] = bounds;
  const x0 = lngX(w), x1 = lngX(e), y0 = latY(n), y1 = latY(s);
  const fw = Math.max(1, vp.width - insets.left - insets.right);
  const fh = Math.max(1, vp.height - insets.top - insets.bottom);
  const scale = Math.min(fw / ((x1 - x0) * TILE), fh / ((y1 - y0) * TILE));
  const zoom = Math.min(maxZoom, Math.log2(scale));
  const ws = worldSize(zoom);
  // the box's centre sits at the free area's centre; the screen centre is offset from it by half the insets
  const cx = (x0 + x1) / 2 - (insets.left - insets.right) / 2 / ws;
  const cy = (y0 + y1) / 2 - (insets.top - insets.bottom) / 2 / ws;
  return { center: [xLng(cx), yLat(cy)], zoom };
}

/**
 * The camera at which an image covering `bounds` fills the viewport as CSS `object-fit: cover` (or
 * `mask-size: cover`) with `object-position: px% py%` would place it. The opening's still is drawn by CSS
 * from the first frame; the live chart is moved to this camera, so the two register exactly.
 */
export function coverCamera(bounds: Bounds, vp: Viewport, pos: [number, number] = [0.5, 0.5]): Camera {
  const [[w, s], [e, n]] = bounds;
  const x0 = lngX(w), x1 = lngX(e), y0 = latY(n), y1 = latY(s);
  const scale = Math.max(vp.width / ((x1 - x0) * TILE), vp.height / ((y1 - y0) * TILE));
  const zoom = Math.log2(scale);
  const ws = worldSize(zoom);
  const iw = (x1 - x0) * ws, ih = (y1 - y0) * ws;
  // the image's left/top on screen, as CSS positions it
  const left = (vp.width - iw) * pos[0];
  const top = (vp.height - ih) * pos[1];
  const cx = x0 + (vp.width / 2 - left) / ws;
  const cy = y0 + (vp.height / 2 - top) / ws;
  return { center: [xLng(cx), yLat(cy)], zoom };
}

/** Where an image spanning `bounds` lies on screen (for the static tier's baked chart). */
export function screenRect(cam: Camera, vp: Viewport, bounds: Bounds): { left: number; top: number; width: number; height: number } {
  const a = project(cam, vp, bounds[0][0], bounds[1][1]);
  const b = project(cam, vp, bounds[1][0], bounds[0][1]);
  return { left: a.x, top: a.y, width: b.x - a.x, height: b.y - a.y };
}

export const metresPerPixel = (lat: number, zoom: number) => (EARTH * Math.cos((lat * Math.PI) / 180)) / worldSize(zoom);

/** A scale bar no longer than `maxPx`: a round distance in km (or m below 1 km), and its length. */
export function scaleBar(lat: number, zoom: number, maxPx: number): { metres: number; px: number; label: string } {
  const mpp = metresPerPixel(lat, zoom);
  const max = mpp * maxPx;
  const steps = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000];
  const metres = [...steps].reverse().find((m) => m <= max) ?? steps[0];
  return { metres, px: metres / mpp, label: metres >= 1000 ? `${metres / 1000} km` : `${metres} m` };
}

/** Graticule steps in minutes of arc, smallest first. */
const STEPS = [1, 2, 5, 10, 15, 20, 30, 60, 120, 300];

/**
 * The neatline's ticks along one axis: a round step (in minutes) so that labels sit at least `minPx` apart,
 * with minor ticks at a fifth (or a half) of it. Returns the degree values inside [lo, hi].
 */
export function ticks(lo: number, hi: number, spanPx: number, minPx = 110): { major: number[]; minor: number[]; stepMin: number } {
  const perMin = spanPx / ((hi - lo) * 60);
  const stepMin = STEPS.find((s) => s * perMin >= minPx) ?? STEPS[STEPS.length - 1];
  const sub = stepMin % 5 === 0 ? stepMin / 5 : stepMin / 2;
  const major: number[] = [];
  const minor: number[] = [];
  const start = Math.ceil((lo * 60) / sub);
  for (let k = start; k * sub <= hi * 60 + 1e-9; k++) {
    const m = k * sub;
    const deg = m / 60;
    // integer minutes, so 0.1 + 0.2 never makes a "122°10′" into "122°9.99′"
    if (Math.abs(m / stepMin - Math.round(m / stepMin)) < 1e-9) major.push(deg);
    else minor.push(deg);
  }
  return { major, minor, stepMin };
}

/** "122°10′ W", "36°50′ N" (whole minutes; the hemisphere as a letter). */
export function formatDegrees(v: number, axis: "lon" | "lat"): string {
  const hemi = axis === "lon" ? (v < 0 ? "W" : "E") : v < 0 ? "S" : "N";
  const totalMin = Math.round(Math.abs(v) * 60);
  const d = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  return `${d}°${String(m).padStart(2, "0")}′ ${hemi}`;
}
