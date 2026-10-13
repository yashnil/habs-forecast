import type { Map as MLMap, PaddingOptions } from "maplibre-gl";
import { km, type LngLat } from "../lib/geo";

/**
 * Coastal journey: moving between places *along the coast* instead of a straight flight over land.
 *
 * The camera follows a smooth curve through offshore waypoints (each port pushed ~10 km out to
 * sea, ordered north→south), so the reader sees the coastline pass beneath them and keeps their
 * bearings. Short hops (< 40 km) are a plain easeTo. Very long ones (> 450 km) use flyTo (zoom out,
 * cross, zoom in) because following every headland would be slow.
 *
 * Rules: any user input (pointer, wheel, key, touch on the map) stops the journey immediately
 * where it is; reduced motion → an instant jump; the duration scales with distance and is capped.
 */
export interface Journey { cancel: () => void; done: Promise<"arrived" | "interrupted"> }

const prefersReduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

function catmull(p: LngLat[], t: number): LngLat {
  const n = p.length - 1;
  const f = Math.min(n - 1e-9, Math.max(0, t * n));
  const i = Math.floor(f), u = f - i;
  const P = (k: number) => p[Math.max(0, Math.min(n, k))];
  const [p0, p1, p2, p3] = [P(i - 1), P(i), P(i + 1), P(i + 2)];
  const c = (a: number, b: number, cc: number, d: number) =>
    0.5 * (2 * b + (-a + cc) * u + (2 * a - 5 * b + 4 * cc - d) * u * u + (-a + 3 * b - 3 * cc + d) * u * u * u);
  return [c(p0[0], p1[0], p2[0], p3[0]), c(p0[1], p1[1], p2[1], p3[1])];
}
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2); // in-out cubic

export function journey(map: MLMap, to: { center: LngLat; zoom: number; padding?: PaddingOptions }, coastPath: LngLat[]): Journey {
  const from = map.getCenter().toArray() as LngLat;
  const z0 = map.getZoom();
  const dist = km(from, to.center);
  let resolve!: (v: "arrived" | "interrupted") => void;
  const done = new Promise<"arrived" | "interrupted">((r) => (resolve = r));

  if (prefersReduced() || dist < 0.5) {
    map.jumpTo({ center: to.center, zoom: to.zoom, padding: to.padding });
    resolve("arrived");
    return { cancel: () => {}, done };
  }
  if (dist < 40 || dist > 450) {
    const opts = { center: to.center, zoom: to.zoom, padding: to.padding, essential: false };
    if (dist < 40) map.easeTo({ ...opts, duration: Math.min(900, 400 + dist * 14) });
    else map.flyTo({ ...opts, curve: 1.5, speed: 1.4, maxDuration: 3200 });
    const onEnd = () => resolve("arrived");
    map.once("moveend", onEnd);
    return { cancel: () => map.stop(), done };
  }

  // waypoints: coast points that lie between the two latitudes, ordered in the travel direction
  const north = Math.max(from[1], to.center[1]), south = Math.min(from[1], to.center[1]);
  const mid = coastPath.filter((p) => p[1] < north - 0.05 && p[1] > south + 0.05);
  mid.sort((a, b) => (from[1] > to.center[1] ? b[1] - a[1] : a[1] - b[1]));
  const pts: LngLat[] = [from, ...mid, to.center];
  // zoom profile: lift out enough to see ~1.5x the leg, then settle at the destination zoom
  const zLift = Math.max(5.6, Math.min(z0, to.zoom) - Math.log2(Math.max(1, dist / 60)) - 0.6);
  const duration = Math.min(2600, 900 + dist * 4.2);
  map.stop();
  const start = performance.now();
  let raf = 0;
  let stopped = false;
  const finish = (how: "arrived" | "interrupted") => {
    if (stopped) return;
    stopped = true;
    cancelAnimationFrame(raf);
    container.removeEventListener("pointerdown", interrupt, true);
    container.removeEventListener("wheel", interrupt, true);
    container.removeEventListener("touchstart", interrupt, true);
    removeEventListener("keydown", interrupt, true);
    resolve(how);
  };
  const interrupt = () => finish("interrupted");
  const container = map.getCanvasContainer();
  container.addEventListener("pointerdown", interrupt, true);
  container.addEventListener("wheel", interrupt, true);
  container.addEventListener("touchstart", interrupt, true);
  addEventListener("keydown", interrupt, true);

  const step = (now: number) => {
    if (stopped) return;
    const t = Math.min(1, (now - start) / duration);
    const e = ease(t);
    const center = catmull(pts, e);
    // zoom: smooth dip — out over the first 35 %, glide, in over the last 40 %
    const up = Math.min(1, e / 0.35), down = Math.max(0, (e - 0.6) / 0.4);
    const zoom = down > 0 ? zLift + (to.zoom - zLift) * (1 - Math.cos(down * Math.PI)) / 2 : z0 + (zLift - z0) * (1 - Math.cos(up * Math.PI)) / 2;
    map.jumpTo({ center, zoom, padding: to.padding });
    if (t < 1) raf = requestAnimationFrame(step);
    else finish("arrived");
  };
  raf = requestAnimationFrame(step);
  return { cancel: () => finish("interrupted"), done };
}

/** Offshore coast waypoints from the port list: each port pushed west by ~10 km. */
export function coastPathFromPorts(ports: { lon: number; lat: number }[]): LngLat[] {
  return ports.map((p) => [p.lon - 10 / (111.32 * Math.cos((p.lat * Math.PI) / 180)), p.lat] as LngLat).sort((a, b) => b[1] - a[1]);
}
