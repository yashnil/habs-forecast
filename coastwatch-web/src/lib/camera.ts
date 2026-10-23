import type { LngLatBoundsLike, Map as MlMap, PaddingOptions } from "maplibre-gl";

/**
 * Coastal journeys (signature interaction §2), adapted to the device. Place changes keep their
 * geographic context with a distance-scaled flight, but only where it can be smooth:
 * - "full": a near pick (under 40 km) eases in 600 ms; further, a flight scaled by distance
 *   (700 ms + 3 ms/km, at most 1.8 s, so the map arrives while the panel's content is being read);
 * - "short": a 300 ms ease (software WebGL, or a device that measured under 30 fps);
 * - "none": a jump (prefers-reduced-motion).
 * Any pointer, wheel, touch or key input stops a flight where it is (MapLibre's own behaviour); a
 * journey is context, never navigation or travel time.
 *
 * The prototype's journey ran at 60 fps with a GPU and 15 fps on software WebGL (PR #43
 * 05-evaluation), so the tier is chosen from the renderer and then from measured frames.
 */
export type CameraTier = "full" | "short" | "none";

let probed: CameraTier | null = null;
const KEY = "cw-camera-tier-v1";

function software(): boolean {
  try {
    const c = document.createElement("canvas");
    const gl = (c.getContext("webgl") ?? c.getContext("experimental-webgl")) as WebGLRenderingContext | null;
    if (!gl) return true;
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const r = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "";
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return /swiftshader|llvmpipe|software|basic render|softpipe/i.test(r);
  } catch {
    return false;
  }
}

export function cameraTier(): CameraTier {
  if (typeof window === "undefined") return "none";
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return "none";
  if (probed) return probed;
  let saved: string | null = null;
  try {
    saved = sessionStorage.getItem(KEY);
  } catch {
    // not kept
  }
  probed = saved === "short" ? "short" : software() ? "short" : "full";
  return probed;
}

/** After a flight, keep flights only if frames were smooth enough (median frame <= 33 ms). */
function watchFrames(m: MlMap) {
  const times: number[] = [];
  let last = performance.now();
  let id = 0;
  const tick = (t: number) => {
    times.push(t - last);
    last = t;
    id = requestAnimationFrame(tick);
  };
  id = requestAnimationFrame(tick);
  m.once("moveend", () => {
    cancelAnimationFrame(id);
    if (times.length < 8) return;
    const s = [...times].sort((a, b) => a - b);
    if (s[Math.floor(s.length / 2)] > 33) {
      probed = "short";
      try {
        sessionStorage.setItem(KEY, "short");
      } catch {
        // not kept
      }
    }
  });
}

/** The journey's timings (docs/coastwatch/v9/motion.md §2): exported for the map's own small moves. */
export const NEAR_KM = 40;
export const NEAR_MS = 600;
export const SHORT_MS = 300;
export const FLIGHT_MAX_MS = 1800;
export const flightMs = (km: number) => Math.min(FLIGHT_MAX_MS, 700 + 3 * km);

const km = (a: [number, number], b: [number, number]) => {
  const r = Math.PI / 180;
  const x = (b[0] - a[0]) * r * Math.cos(((a[1] + b[1]) / 2) * r);
  const y = (b[1] - a[1]) * r;
  return Math.sqrt(x * x + y * y) * 6371;
};

/** Move the camera to bounds (or a centre and zoom) as a journey suited to this device. */
export function journeyTo(m: MlMap, to: { bounds: LngLatBoundsLike; padding?: number | PaddingOptions; maxZoom?: number } | { center: [number, number]; zoom: number; padding?: PaddingOptions }) {
  // options with only the keys that are set: MapLibre treats a present-but-undefined key as a value
  const fit: { padding?: number | PaddingOptions; maxZoom?: number } = {};
  if ("bounds" in to && to.padding != null) fit.padding = to.padding;
  if ("bounds" in to && to.maxZoom != null) fit.maxZoom = to.maxZoom;
  const cam = "bounds" in to ? m.cameraForBounds(to.bounds, fit) : { center: to.center, zoom: to.zoom };
  if (!cam) return;
  const centre = cam.center ? (Array.isArray(cam.center) ? (cam.center as [number, number]) : [(cam.center as { lng: number }).lng, (cam.center as { lat: number }).lat] as [number, number]) : null;
  if (!centre) return;
  // never pass `padding: undefined`: MapLibre reads the key's presence and the camera becomes NaN
  const opts: { center: [number, number]; zoom: number; padding?: PaddingOptions } = { center: centre, zoom: cam.zoom ?? m.getZoom() };
  if (!("bounds" in to) && to.padding) opts.padding = to.padding;
  if (!Number.isFinite(opts.zoom) || !centre.every(Number.isFinite)) return;
  const tier = cameraTier();
  if (tier === "none") return m.jumpTo(opts);
  if (tier === "short") return m.easeTo({ ...opts, duration: SHORT_MS });
  const here = m.getCenter();
  const d = km([here.lng, here.lat], centre);
  if (d < NEAR_KM) return m.easeTo({ ...opts, duration: NEAR_MS });
  watchFrames(m);
  m.flyTo({ ...opts, duration: flightMs(d), curve: 1.42, essential: false });
}
