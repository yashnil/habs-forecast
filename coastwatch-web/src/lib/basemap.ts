import type { StyleSpecification } from "maplibre-gl";

/**
 * Navy marine basemap on OpenFreeMap vector tiles (OpenMapTiles schema; free, no key).
 * Data overlays are inserted before `coastline` so the shoreline and labels stay on top.
 * If the tile service is unreachable the background still renders and overlays still draw.
 */
export const BEFORE_OVERLAY_ID = "coastline";

const FONT = ["Noto Sans Regular"];

export const BASEMAP_STYLE: StyleSpecification = {
  version: 8,
  glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
  sources: {
    omt: {
      type: "vector",
      url: "https://tiles.openfreemap.org/planet",
      attribution:
        '<a href="https://openfreemap.org" target="_blank" rel="noreferrer">OpenFreeMap</a> · © <a href="https://www.openmaptiles.org/" target="_blank" rel="noreferrer">OpenMapTiles</a> · © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">OpenStreetMap contributors</a>',
    },
  },
  layers: [
    { id: "land", type: "background", paint: { "background-color": "#17263b" } },
    {
      id: "landcover-wood",
      type: "fill",
      source: "omt",
      "source-layer": "landcover",
      filter: ["==", ["get", "class"], "wood"],
      paint: { "fill-color": "#1a2b42", "fill-opacity": 0.6 },
    },
    {
      id: "water",
      type: "fill",
      source: "omt",
      "source-layer": "water",
      paint: { "fill-color": "#040e1f" },
    },
    {
      id: "coastline",
      type: "line",
      source: "omt",
      "source-layer": "water",
      paint: {
        "line-color": "#6f93ba",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.5, 8, 1, 12, 1.6],
        "line-opacity": 0.85,
      },
    },
    {
      id: "rivers",
      type: "line",
      source: "omt",
      "source-layer": "waterway",
      minzoom: 7,
      paint: { "line-color": "#1d3858", "line-width": 0.8 },
    },
    {
      id: "roads-major",
      type: "line",
      source: "omt",
      "source-layer": "transportation",
      minzoom: 6,
      filter: ["in", ["get", "class"], ["literal", ["motorway", "trunk"]]],
      paint: { "line-color": "#24364f", "line-width": ["interpolate", ["linear"], ["zoom"], 6, 0.4, 12, 1.4] },
    },
    {
      id: "boundary-state",
      type: "line",
      source: "omt",
      "source-layer": "boundary",
      // land boundaries only; maritime limits would read as coastal features
      filter: ["all", ["<=", ["coalesce", ["get", "admin_level"], 99], 4], ["!=", ["coalesce", ["get", "maritime"], 0], 1]],
      paint: { "line-color": "#34496a", "line-width": 0.8, "line-dasharray": [3, 2] },
    },
    {
      id: "place-city",
      type: "symbol",
      source: "omt",
      "source-layer": "place",
      minzoom: 5,
      filter: ["in", ["get", "class"], ["literal", ["city", "town"]]],
      layout: {
        "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]],
        "text-font": FONT,
        "text-size": ["interpolate", ["linear"], ["zoom"], 5, 10, 10, 13],
        "text-anchor": "left",
        "text-offset": [0.4, 0],
        "symbol-sort-key": ["get", "rank"],
      },
      paint: { "text-color": "#8193ac", "text-halo-color": "#040b17", "text-halo-width": 1.2 },
    },
  ],
};

/** California coast from the Oregon border to Mexico, used for the initial view. */
export const CA_COAST_BOUNDS: [[number, number], [number, number]] = [
  [-125.0, 32.3],
  [-116.9, 42.1],
];
export const CA_BOUNDS: [[number, number], [number, number]] = [
  [-131.5, 29.5],
  [-112.5, 44.8],
];
