"use client";

import type { CustomLayerInterface, Map as MapLibreMap } from "maplibre-gl";

/**
 * Data layers on the one chart (README §9 Sheet; D-motion-feasibility §11). Feature code adds its layers here
 * instead of touching the map: the registry keeps them in their slot in the chart's z-order and puts them back
 * after a lost WebGL context, which MapLibre restores without custom layers (measured in D §11: `onAdd` does
 * not rerun). Every data layer goes under the land, so the coast always masks it, and above the depth tints;
 * the isobaths and their labels stay over the data, as on a printed chart.
 *
 * Slots, bottom to top: `field` (the active layer: chlorophyll, temperature, model), `flow` (measured
 * currents). A layer's own `onRemove` runs on context loss, so it drops its GL objects; `onAdd` runs again in
 * the new context, so it rebuilds them from the data it kept.
 */
export type ChartLayerSlot = "field" | "flow";
const SLOT_ORDER: ChartLayerSlot[] = ["field", "flow"];
/** The chart style's layer that data goes under (src/nx/chart/style.ts). */
export const DATA_BEFORE = "land";

type Entry = { layer: CustomLayerInterface; slot: ChartLayerSlot };
const entries = new Map<string, Entry>();
let bound: MapLibreMap | null = null;
let lost = false;

function ordered(): Entry[] {
  return [...entries.values()].sort((a, b) => SLOT_ORDER.indexOf(a.slot) - SLOT_ORDER.indexOf(b.slot));
}

/**
 * Put every registered layer on the map in slot order (adding the missing ones). `styleReady` is true right
 * after "style.load", when the style accepts layers although its sources are still loading.
 */
function sync(map: MapLibreMap, styleReady = false): boolean {
  if (lost || !map.style || (!styleReady && !map.isStyleLoaded())) return false;
  try {
    const before = map.getLayer(DATA_BEFORE) ? DATA_BEFORE : undefined;
    for (const e of ordered()) {
      if (!map.getLayer(e.layer.id)) map.addLayer(e.layer, before);
      else if (before) map.moveLayer(e.layer.id, before);
    }
    return true;
  } catch {
    return false;
  }
}

function bind(map: MapLibreMap) {
  if (bound === map) return;
  bound = map;
  map.on("webglcontextlost", () => {
    lost = true;
    // the context took the layers' programs and buffers with it: let each drop its references
    for (const e of entries.values()) e.layer.onRemove?.(map, undefined as never);
  });
  map.on("webglcontextrestored", () => {
    lost = false;
    // MapLibre reloads its style after a restore (without custom layers); they come back once it has, and
    // again at the next idle in case the style was still settling
    map.once("style.load", () => {
      if (!sync(map, true)) map.once("idle", () => sync(map, true));
    });
    map.once("idle", () => sync(map, true));
  });
}

/** Add (or replace) a data layer in its slot. Returns the function that removes it. */
export function addChartLayer(map: MapLibreMap, layer: CustomLayerInterface, slot: ChartLayerSlot): () => void {
  bind(map);
  const prev = entries.get(layer.id);
  if (prev && prev.layer !== layer && map.getLayer(layer.id)) map.removeLayer(layer.id);
  entries.set(layer.id, { layer, slot });
  if (map.isStyleLoaded()) sync(map);
  else map.once("idle", () => sync(map));
  return () => {
    if (entries.get(layer.id)?.layer !== layer) return;
    entries.delete(layer.id);
    if (map.style && map.getLayer(layer.id)) map.removeLayer(layer.id);
  };
}

/** The registered layer ids in drawing order (tests and the e2e hook). */
export const chartLayerIds = () => ordered().map((e) => e.layer.id);
