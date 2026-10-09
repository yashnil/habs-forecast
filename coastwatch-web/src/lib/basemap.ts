import type { StyleSpecification } from "maplibre-gl";

/**
 * Dark-sea basemap on OpenFreeMap vector tiles (OpenMapTiles schema; free, no key).
 * Design reset rev. 2: land is a lighter slate than the sea so the coast reads first;
 * water carries a faint hatch that only shows where an opaque data layer has no value
 * (so "no value" never looks like a low value); data is inserted before `coastline`, so
 * the shoreline, roads and labels stay on top. If tiles are unreachable the background
 * still renders and overlays still draw.
 */
export const SEA = "#0b1d33";
export const LAND = "#2a3646";
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
    { id: "land", type: "background", paint: { "background-color": LAND } },
    {
      id: "landcover-wood",
      type: "fill",
      source: "omt",
      "source-layer": "landcover",
      filter: ["==", ["get", "class"], "wood"],
      paint: { "fill-color": "#2f3d4f", "fill-opacity": 0.8 },
    },
    {
      id: "water",
      type: "fill",
      source: "omt",
      "source-layer": "water",
      paint: { "fill-color": SEA },
    },
    // pattern image "hatch" is registered by MapCanvas when the style asks for it
    { id: "water-nodata", type: "fill", source: "omt", "source-layer": "water", paint: { "fill-pattern": "hatch" } },
    {
      id: "coastline",
      type: "line",
      source: "omt",
      "source-layer": "water",
      paint: {
        "line-color": "#d3dfeb",
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.6, 9, 1.2, 12, 1.8],
      },
    },
    {
      id: "rivers",
      type: "line",
      source: "omt",
      "source-layer": "waterway",
      minzoom: 7,
      paint: { "line-color": "#3c5574", "line-width": 0.8 },
    },
    {
      id: "roads-major",
      type: "line",
      source: "omt",
      "source-layer": "transportation",
      minzoom: 6,
      filter: ["in", ["get", "class"], ["literal", ["motorway", "trunk"]]],
      paint: { "line-color": "#3a4859", "line-width": ["interpolate", ["linear"], ["zoom"], 6, 0.4, 12, 1.4] },
    },
    {
      id: "boundary-state",
      type: "line",
      source: "omt",
      "source-layer": "boundary",
      // land boundaries only; maritime limits would read as coastal features
      filter: ["all", ["<=", ["coalesce", ["get", "admin_level"], 99], 4], ["!=", ["coalesce", ["get", "maritime"], 0], 1]],
      paint: { "line-color": "#5a6b80", "line-width": 0.8, "line-dasharray": [3, 2] },
    },
    {
      id: "place-city",
      type: "symbol",
      source: "omt",
      "source-layer": "place",
      minzoom: 7,
      filter: ["in", ["get", "class"], ["literal", ["city", "town"]]],
      layout: {
        "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]],
        "text-font": FONT,
        "text-size": ["interpolate", ["linear"], ["zoom"], 5, 10, 10, 13],
        "text-anchor": "left",
        "text-offset": [0.4, 0],
        "symbol-sort-key": ["get", "rank"],
      },
      paint: { "text-color": "#aab8c8", "text-halo-color": LAND, "text-halo-width": 1.2 },
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

/** 8 px diagonal hatch for water without a data value (registered on demand). */
export function hatchImage(): ImageData {
  const n = 8;
  const c = document.createElement("canvas");
  c.width = c.height = n;
  const g = c.getContext("2d")!;
  g.strokeStyle = "rgba(150,178,210,0.26)";
  g.lineWidth = 1;
  g.beginPath();
  g.moveTo(0, n);
  g.lineTo(n, 0);
  g.moveTo(-1, 1);
  g.lineTo(1, -1);
  g.moveTo(n - 1, n + 1);
  g.lineTo(n + 1, n - 1);
  g.stroke();
  return g.getImageData(0, 0, n, n);
}
