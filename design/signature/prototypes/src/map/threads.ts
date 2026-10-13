import type { GeoJSONSource, Map as MLMap } from "maplibre-gl";
import { arc, type LngLat } from "../lib/geo";

/**
 * Provenance threads: thin lines from the selected water point to where each piece of information
 * actually comes from (the station that was sampled, the model cell, the satellite pixel, an
 * official limit). They draw out once, in order, then settle to a quiet opacity; hovering an item
 * in the panel brightens its thread. A thread is a *connector*, not a value: it is never coloured
 * by data, never animated continuously, and it ends at the source's own location.
 */
export type ThreadKind = "station" | "model" | "satellite" | "notice" | "nearest";
export interface Thread { id: ThreadKind; from: LngLat; to: LngLat; colour: string }

const SRC = "threads";
const prefersReduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export function addThreadLayers(map: MLMap) {
  map.addSource(SRC, { type: "geojson", data: { type: "FeatureCollection", features: [] }, lineMetrics: true, promoteId: "id" });
  map.addLayer({
    id: "threads-halo", type: "line", source: SRC,
    layout: { "line-cap": "round" },
    paint: { "line-color": "rgba(7,19,32,0.55)", "line-width": 4, "line-opacity": ["case", ["boolean", ["feature-state", "dim"], false], 0.25, 0.9] },
  });
  map.addLayer({
    id: "threads", type: "line", source: SRC,
    layout: { "line-cap": "round" },
    paint: {
      "line-width": ["case", ["boolean", ["feature-state", "hl"], false], 2.25, 1.35],
      "line-gradient": ["interpolate", ["linear"], ["line-progress"], 0, "rgba(255,255,255,0.9)", 1, "rgba(255,255,255,0.9)"],
      "line-opacity": ["case", ["boolean", ["feature-state", "dim"], false], 0.3, ["boolean", ["feature-state", "hl"], false], 1, 0.85],
    },
  });
  map.addLayer({
    id: "thread-ends", type: "circle", source: SRC,
    filter: ["==", ["geometry-type"], "Point"],
    paint: {
      "circle-radius": ["case", ["boolean", ["feature-state", "hl"], false], 4.5, 3.25],
      "circle-color": ["get", "colour"],
      "circle-stroke-color": "#071320",
      "circle-stroke-width": 1.5,
      "circle-opacity": ["case", ["boolean", ["feature-state", "dim"], false], 0.35, 1],
      "circle-stroke-opacity": ["case", ["boolean", ["feature-state", "dim"], false], 0.35, 1],
    },
  });
}

let drawing = 0;
/** Draw threads out from the point, one after another (staggered 90 ms), 520 ms each. */
export function setThreads(map: MLMap, threads: Thread[]) {
  const src = map.getSource(SRC) as GeoJSONSource | undefined;
  if (!src) return;
  const lines = threads.map((t) => ({ type: "Feature" as const, id: t.id, properties: { id: t.id, colour: t.colour }, geometry: { type: "LineString" as const, coordinates: arc(t.from, t.to) } }));
  // end points share the thread id as `id` property but need distinct feature ids
  const ends = threads.map((t) => ({ type: "Feature" as const, id: `${t.id}-end`, properties: { id: `${t.id}-end`, colour: t.colour }, geometry: { type: "Point" as const, coordinates: t.to } }));
  src.setData({ type: "FeatureCollection", features: [...lines, ...ends] });
  // per-thread colour with draw-out progress via line-gradient (gradient is per layer, so the
  // colour comes from a match on the line's id through the line-progress ramp)
  const run = ++drawing;
  const n = threads.length;
  const colourOf: unknown[] = ["match", ["get", "id"]];
  threads.forEach((t) => colourOf.push(t.id, t.colour));
  colourOf.push("#ffffff");
  if (!n) return;
  if (prefersReduced()) { setGradient(map, 1, threads); return; }
  const start = performance.now();
  const total = 520 + 90 * (n - 1);
  const frame = (now: number) => {
    if (run !== drawing) return;
    const t = Math.min(1, (now - start) / total);
    setGradient(map, t, threads, start, now);
    if (t < 1) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

// line-gradient cannot read feature properties, so the threads layer is drawn once per kind via
// a filter-free trick: one gradient for all lines (white), and the coloured end dots carry the
// kind. The draw-out progress is shared; the stagger is applied by delaying each dot's opacity.
function setGradient(map: MLMap, t: number, _threads: Thread[], start?: number, now?: number) {
  const p = Math.max(0.0001, Math.min(1, easeOut(t)));
  const c = "rgba(233,239,246,0.92)";
  const expr = p >= 1
    ? ["interpolate", ["linear"], ["line-progress"], 0, c, 1, c]
    : ["interpolate", ["linear"], ["line-progress"], 0, c, p, c, Math.min(1, p + 0.0001), "rgba(233,239,246,0)"];
  map.setPaintProperty("threads", "line-gradient", expr as never);
  map.setPaintProperty("threads-halo", "line-opacity", p >= 1 ? ["case", ["boolean", ["feature-state", "dim"], false], 0.25, 0.9] as never : 0);
  map.setPaintProperty("thread-ends", "circle-opacity", p >= 0.95 ? ["case", ["boolean", ["feature-state", "dim"], false], 0.35, 1] as never : 0);
  map.setPaintProperty("thread-ends", "circle-stroke-opacity", p >= 0.95 ? ["case", ["boolean", ["feature-state", "dim"], false], 0.35, 1] as never : 0);
  void start; void now;
}
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

/** Highlight one thread (others dim), or clear with null. Feature-state only: no re-render. */
export function highlightThread(map: MLMap, id: ThreadKind | null, all: ThreadKind[]) {
  for (const k of all) {
    for (const fid of [k, `${k}-end`]) {
      map.setFeatureState({ source: SRC, id: fid }, { hl: id === k, dim: id != null && id !== k });
    }
  }
}
