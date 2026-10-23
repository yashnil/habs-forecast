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

// ------------------------------------------------------------------------------------- the coastal frame

/**
 * The California coast from Crescent City to San Diego: the shore's own extent, not the state's. The
 * opening camera frames this strip (v10 review: the statewide fit was about 60 % Nevada).
 */
export const COAST_BOUNDS: [[number, number], [number, number]] = [
  [-124.45, 32.62],
  [-117.12, 41.79],
];
export type Insets = { top: number; bottom: number; left: number; right: number };

const mx = (lon: number) => (lon + 180) / 360;
const my = (lat: number) => {
  const s = Math.sin((lat * Math.PI) / 180);
  return 0.5 - (0.25 * Math.log((1 + s) / (1 - s))) / Math.PI;
};
const lonX = (x: number) => x * 360 - 180;
const latY = (y: number) => (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI;

/**
 * The camera that frames the coast with the ocean dominant: the coast's box fills the height left between
 * the insets (the header above, the time rail below), and its east edge (San Diego) sits at the right
 * inset, so the land is a narrow margin on the right and the open ocean fills the rest of the view, under
 * and beside the layer index. A window too narrow for the box fits its width instead.
 */
/** Where a screen too short for the whole coast is centred: the head of Monterey Canyon (the opening's own place). */
export const COAST_ANCHOR: [number, number] = [-121.95, 36.75];
export function coastCamera(w: number, h: number, ins: Insets, bounds: [[number, number], [number, number]] = COAST_BOUNDS, minZoom = 4.5): { center: [number, number]; zoom: number } {
  const [[west, south], [east, north]] = bounds;
  const X0 = mx(west), X1 = mx(east), Y0 = my(north), Y1 = my(south);
  const availH = Math.max(80, h - ins.top - ins.bottom);
  const availW = Math.max(80, w - ins.left - ins.right);
  const z = Math.min(Math.log2(availH / ((Y1 - Y0) * 512)), Math.log2(availW / ((X1 - X0) * 512)), 9);
  if (z < minZoom) {
    // a small phone: the whole coast would be a thread. The central coast instead, at a legible scale, with
    // the canyon left of centre so the ocean takes most of the strip and the land stays a margin
    const wq = 512 * 2 ** minZoom;
    const cx = mx(COAST_ANCHOR[0]) - (ins.left + availW * 0.66 - w / 2) / wq;
    const cy = my(COAST_ANCHOR[1]) - ((ins.top + (h - ins.bottom)) / 2 - h / 2) / wq;
    return { center: [lonX(cx), latY(cy)], zoom: minZoom };
  }
  const wp = 512 * 2 ** z;
  const cx = X1 - (w / 2 - ins.right) / wp;
  const cy = (Y0 + Y1) / 2 - ((ins.top + (h - ins.bottom)) / 2 - h / 2) / wp;
  return { center: [lonX(cx), latY(cy)], zoom: z };
}

/**
 * Which way the water lies from a coastal point, as a unit vector in screen terms (x east, y south), from
 * the bundled water geometry: the centroid of the water within about 10 km. A selected harbour is framed
 * with its water, not centred on the land behind it. Null when the geometry gives no direction.
 */
export function seaward(water: { sample: (lon: number, lat: number) => boolean } | null, lat: number, lon: number): [number, number] | null {
  if (!water) return null;
  let sx = 0, sy = 0, n = 0, all = 0;
  const R = 10; // km
  const dLat = R / 111, dLon = R / (111 * Math.cos((lat * Math.PI) / 180));
  for (let i = -6; i <= 6; i++)
    for (let j = -6; j <= 6; j++) {
      if (i * i + j * j > 36) continue;
      all++;
      if (water.sample(lon + (j / 6) * dLon, lat - (i / 6) * dLat)) {
        sx += j;
        sy += i;
        n++;
      }
    }
  // all water or all land: no side to lean to
  if (!n || n === all) return null;
  const len = Math.hypot(sx, sy);
  return len < 1e-6 ? null : [sx / len, sy / len];
}

/** A point-in-water test on the bundled water rings (even-odd, normalised Mercator), for `seaward`. */
export function waterSampler(g: { rings: Float64Array[]; bbox: Float64Array } | null): { sample: (lon: number, lat: number) => boolean } | null {
  if (!g) return null;
  return {
    sample(lon, lat) {
      const x = mx(lon), y = my(lat);
      let inside = false;
      for (let r = 0; r < g.rings.length; r++) {
        const b = r * 4;
        if (y < g.bbox[b + 1] || y > g.bbox[b + 3] || x > g.bbox[b + 2]) continue;
        const ring = g.rings[r];
        const n = ring.length;
        for (let i = 0, j = n - 2; i < n; j = i, i += 2) {
          const yi = ring[i + 1], yj = ring[j + 1];
          if (yi > y !== yj > y && x < ((ring[j] - ring[i]) * (y - yi)) / (yj - yi) + ring[i]) inside = !inside;
        }
      }
      return inside;
    },
  };
}
