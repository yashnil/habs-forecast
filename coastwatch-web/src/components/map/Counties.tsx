"use client";

import { Layer, Source } from "react-map-gl/maplibre";
import type { ExpressionSpecification } from "maplibre-gl";
import { COUNTIES } from "@/content/copy";

/** Bundled county geography (pipeline/scripts/build_counties.py): borders between counties and one label each. */
export const COUNTIES_URL = "/basemap/ca-counties.json";

const LABEL_COAST: ExpressionSpecification = ["!=", ["get", "coast"], "inland"];

/**
 * County borders and names, always drawn, as part of the basemap's own cartography (v5: no switch).
 *
 * - Borders: a fine dash-dot line between counties (the coastline and state line stay the basemap's
 *   own), on a soft dark casing so it reads on the dark land, on the relief's light ridges and
 *   over a light data colour alike. Hairline at state scale, a little firmer at harbour scale.
 * - Names: quiet, letter-spaced capitals, stronger for the counties that face the ocean or San
 *   Francisco Bay. They are placed after towns, harbours and stations (their layers sit below them
 *   in the stack, so those labels claim their space first) and never overlap another label: a county
 *   name that would collide is simply left out at that zoom.
 *
 * Reference only: county lines do not define closures, crab zones, state waters or how far an
 * advisory reaches (the legend says so).
 */
export function CountyLayers({ beforeId }: { beforeId?: string }) {
  return (
    <Source id="counties" type="geojson" data={COUNTIES_URL} attribution={COUNTIES.attribution}>
      <Layer
        id="county-borders-casing"
        type="line"
        beforeId={beforeId}
        minzoom={4.5}
        filter={["==", ["get", "kind"], "border"]}
        layout={{ "line-join": "round", "line-cap": "round" }}
        paint={{
          "line-color": "#06111e",
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 4.5, 0.15, 6, 0.28, 10, 0.4],
          "line-width": ["interpolate", ["linear"], ["zoom"], 5, 1.4, 8, 2, 12, 3],
          "line-blur": 0.6,
        }}
      />
      <Layer
        id="county-borders"
        type="line"
        beforeId={beforeId}
        minzoom={4.5}
        filter={["==", ["get", "kind"], "border"]}
        layout={{ "line-join": "round" }}
        paint={{
          "line-color": "#dfe8f2",
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 4.5, 0.18, 6, 0.34, 8, 0.44, 11, 0.55],
          "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.45, 8, 0.75, 10, 1, 12, 1.3],
          // a dash and a dot: the county line of a printed chart, never confused with a data contour
          "line-dasharray": [5, 2, 1, 2],
        }}
      />
      <Layer
        id="county-labels"
        type="symbol"
        beforeId={beforeId}
        minzoom={6.2}
        maxzoom={11.5}
        filter={["==", ["get", "kind"], "label"]}
        layout={{
          "text-field": ["upcase", ["get", "name"]],
          "text-font": ["Noto Sans Regular"],
          "text-size": ["interpolate", ["linear"], ["zoom"], 6.2, 9.5, 8, 11, 10, 12.5],
          "text-letter-spacing": 0.16,
          "text-max-width": 7,
          // generous padding: a county name keeps clear of towns, harbours and stations
          "text-padding": 10,
          "text-allow-overlap": false,
          "text-ignore-placement": false,
          "symbol-sort-key": ["case", LABEL_COAST, 0, 1],
        }}
        paint={{
          "text-color": ["case", LABEL_COAST, "rgba(232,238,245,0.74)", "rgba(232,238,245,0.46)"],
          "text-halo-color": "rgba(6,17,30,0.82)",
          "text-halo-width": 1.2,
          "text-halo-blur": 0.4,
          "text-opacity": ["interpolate", ["linear"], ["zoom"], 6.2, 0, 6.6, 1, 11, 1, 11.5, 0],
        }}
      />
    </Source>
  );
}
