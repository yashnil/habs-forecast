import type { Map as MLMap } from "maplibre-gl";

/**
 * Calls back once the map has loaded and stands still. Not the map's "idle" event alone: an animated layer
 * (the currents) repaints every frame, so "idle" never comes while it runs; each render is checked too.
 * Returns the way to stop waiting.
 */
export function onceSettled(map: MLMap, cb: () => void): () => void {
  let done = false;
  const off = () => {
    map.off("idle", check);
    map.off("render", check);
  };
  function check() {
    if (done || !map.loaded() || map.isMoving()) return;
    done = true;
    off();
    cb();
  }
  map.on("idle", check);
  map.on("render", check);
  return off;
}
