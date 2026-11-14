"use client";

import type { GeoJSONSource, Map as MapLibreMap } from "maplibre-gl";
import { DATA_BEFORE } from "./registry";
import type { Isoline } from "./contours";

/**
 * A model temperature's isotherms on the chart (contours.ts traces them): plain map lines under the land, in the
 * temperature's own warm token (`--sst-6`), heavier every 2 °F, never the depth contours' ink or the currents'
 * ultramarine. Labels are written by the page (°F, as the depth contours carry fathoms).
 */
const SRC = "nx-isotherms";
const LYR = "nx-isotherms-line";

const warm = () => {
  const cs = getComputedStyle(document.documentElement);
  return cs.getPropertyValue("--sst-6").trim() || cs.getPropertyValue("--ink").trim();
};

export function showIsotherms(map: MapLibreMap, lines: Isoline[]) {
  const data: GeoJSON.FeatureCollection<GeoJSON.LineString, { major: boolean; level: number }> = {
    type: "FeatureCollection",
    features: lines.map((l) => ({ type: "Feature", properties: { major: l.major, level: l.level }, geometry: { type: "LineString", coordinates: l.coords } })),
  };
  const src = map.getSource(SRC) as GeoJSONSource | undefined;
  if (src) src.setData(data);
  else map.addSource(SRC, { type: "geojson", data });
  if (!map.getLayer(LYR)) {
    map.addLayer(
      {
        id: LYR,
        type: "line",
        source: SRC,
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": warm(), "line-width": ["case", ["get", "major"], 1.9, 1.0], "line-opacity": ["case", ["get", "major"], 0.95, 0.7] },
      },
      map.getLayer(DATA_BEFORE) ? DATA_BEFORE : undefined,
    );
  }
}

/** The mode changed (day, dusk, night): the line takes the new token. */
export function recolourIsotherms(map: MapLibreMap) {
  if (map.getLayer(LYR)) map.setPaintProperty(LYR, "line-color", warm());
}

export function clearIsotherms(map: MapLibreMap) {
  if (!map.style) return;
  if (map.getLayer(LYR)) map.removeLayer(LYR);
  if (map.getSource(SRC)) map.removeSource(SRC);
}
