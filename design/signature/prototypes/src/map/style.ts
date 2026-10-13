import type { StyleSpecification } from "maplibre-gl";
import { NET } from "../data/net";

/**
 * Signature marine basemap (prototype). Derived from coastwatch-web/src/lib/basemap.ts on
 * feat/coastal-redesign; same sources and drawing order, with these design changes:
 *  - the depth-tinted seafloor is the calm default and cross-fades (not toggles) to the neutral
 *    relief when a data field is shown, so turning a layer on never flashes the sea;
 *  - land is one step darker and less blue, so the sea — the subject — carries the colour;
 *  - labels: ports are the only white labels; towns recede; water names are italic and spaced;
 *  - the coastline is a single crisp hairline that thickens with zoom (the shoreline is the most
 *    important geographic reference in every task);
 *  - data layers are inserted before "coastline" so the shore and labels always stay on top.
 */
export const SEA = "#0b1d33";
export const LAND = "#1b2531";
export const BEFORE_DATA = "isobaths"; // fields sit under isobaths, coastline and labels
const FONT = ["Noto Sans Regular"];
const FONT_ITALIC = ["Noto Sans Italic"];

export const RELIEF: [[number, number], [number, number], [number, number], [number, number]] = [
  [-126.6, 42.6], [-116.4, 42.6], [-116.4, 32.0], [-126.6, 32.0],
];
export const MONTEREY: typeof RELIEF = [
  [-122.72, 37.33], [-121.58, 37.33], [-121.58, 36.22], [-122.72, 36.22],
];
export const CA_COAST: [[number, number], [number, number]] = [[-125.0, 32.3], [-116.9, 42.1]];

const tilesUrl = () => (NET.fail.has("tiles") ? "http://127.0.0.1:9/planet" : "https://tiles.openfreemap.org/planet");

export function signatureStyle(): StyleSpecification {
  return {
    version: 8,
    glyphs: `${location.origin}/fonts/{fontstack}/{range}.pbf`,
    sources: {
      omt: { type: "vector", url: tilesUrl(), attribution: '<a href="https://openfreemap.org" target="_blank" rel="noopener">OpenFreeMap</a> © <a href="https://www.openmaptiles.org/" target="_blank" rel="noopener">OpenMapTiles</a> © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>' },
      "relief-land": { type: "image", url: "/basemap/relief-land.webp", coordinates: RELIEF },
      "relief-sea": { type: "image", url: "/basemap/relief-sea.webp", coordinates: RELIEF },
      "relief-sea-m": { type: "image", url: "/basemap/relief-sea-monterey.webp", coordinates: MONTEREY },
      "relief-tint": { type: "image", url: "/basemap/relief-sea-tint.webp", coordinates: RELIEF },
      "relief-tint-m": { type: "image", url: "/basemap/relief-sea-monterey-tint.webp", coordinates: MONTEREY },
      isobaths: { type: "geojson", data: "/basemap/isobaths.geojson", attribution: 'Relief: <a href="https://doi.org/10.25921/fd45-gt74" target="_blank" rel="noopener">NOAA NCEI ETOPO 2022</a> and Coastal Relief Model' },
      "water-fallback": { type: "geojson", data: "/basemap/water-fallback.json" },
    },
    layers: [
      { id: "land", type: "background", paint: { "background-color": LAND } },
      { id: "relief-land", type: "raster", source: "relief-land", paint: { "raster-fade-duration": 0, "raster-opacity": 0.85, "raster-saturation": -0.35 } },
      // bundled water: drawn always, under the tiled water, so a tile failure leaves a real sea
      { id: "water-fallback", type: "fill", source: "water-fallback", filter: ["==", ["geometry-type"], "Polygon"], paint: { "fill-color": SEA, "fill-antialias": false } },
      { id: "water", type: "fill", source: "omt", "source-layer": "water", paint: { "fill-color": SEA } },
      { id: "relief-sea", type: "raster", source: "relief-sea", paint: { "raster-fade-duration": 0 } },
      { id: "relief-sea-m", type: "raster", source: "relief-sea-m", minzoom: 7, paint: { "raster-fade-duration": 0 } },
      // tint over neutral; its opacity is cross-faded by setSeaMode(), never toggled
      { id: "relief-tint", type: "raster", source: "relief-tint", paint: { "raster-fade-duration": 0, "raster-opacity": 1, "raster-opacity-transition": { duration: 420, delay: 0 } } },
      { id: "relief-tint-m", type: "raster", source: "relief-tint-m", minzoom: 7, paint: { "raster-fade-duration": 0, "raster-opacity": 1, "raster-opacity-transition": { duration: 420, delay: 0 } } },
      {
        id: "isobaths", type: "line", source: "isobaths", minzoom: 5,
        paint: {
          "line-color": ["case", ["==", ["get", "depth_m"], 200], "rgba(170,200,228,0.34)", "rgba(170,200,228,0.15)"],
          "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.5, 10, 1],
        },
      },
      {
        id: "isobath-labels", type: "symbol", source: "isobaths", minzoom: 8.5,
        layout: { "symbol-placement": "line", "symbol-spacing": 480, "text-field": ["get", "label"], "text-font": FONT_ITALIC, "text-size": 10.5, "text-max-angle": 28 },
        paint: { "text-color": "rgba(186,208,230,0.62)", "text-halo-color": "rgba(11,29,51,0.8)", "text-halo-width": 1.2 },
      },
      { id: "coastline-fallback", type: "line", source: "water-fallback", filter: ["==", ["geometry-type"], "LineString"], paint: { "line-color": "#d6e2ee", "line-width": 0.9, "line-opacity": 0.85 } },
      {
        id: "coastline", type: "line", source: "omt", "source-layer": "water",
        paint: {
          "line-color": "#d6e2ee",
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 4, 0.6, 8, 0.9],
          "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.5, 8, 0.9, 11, 1.4, 13, 2],
        },
      },
      { id: "rivers", type: "line", source: "omt", "source-layer": "waterway", minzoom: 8, filter: ["in", ["get", "class"], ["literal", ["river", "canal"]]], paint: { "line-color": "#34506d", "line-width": 0.8 } },
      { id: "roads-major", type: "line", source: "omt", "source-layer": "transportation", minzoom: 7, filter: ["in", ["get", "class"], ["literal", ["motorway", "trunk"]]], paint: { "line-color": "#323f4e", "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.4, 12, 1.3] } },
      {
        id: "boundary-state", type: "line", source: "omt", "source-layer": "boundary",
        filter: ["all", ["<=", ["coalesce", ["get", "admin_level"], 99], 4], ["!=", ["coalesce", ["get", "maritime"], 0], 1]],
        paint: { "line-color": "#4f5f72", "line-width": 0.8, "line-dasharray": [3, 2] },
      },
      {
        id: "water-name", type: "symbol", source: "omt", "source-layer": "water_name", minzoom: 5,
        filter: ["in", ["get", "class"], ["literal", ["ocean", "sea", "bay", "strait", "gulf"]]],
        layout: { "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]], "text-font": FONT_ITALIC, "text-size": ["interpolate", ["linear"], ["zoom"], 5, 11, 10, 15], "text-letter-spacing": 0.14, "text-max-width": 7 },
        paint: { "text-color": "rgba(214,230,244,0.82)", "text-halo-color": "rgba(7,19,32,0.45)", "text-halo-width": 1.2, "text-halo-blur": 0.8 },
      },
      {
        id: "place-city", type: "symbol", source: "omt", "source-layer": "place", minzoom: 6, filter: ["==", ["get", "class"], "city"],
        layout: { "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]], "text-font": FONT, "text-size": ["interpolate", ["linear"], ["zoom"], 6, 11, 10, 13], "text-anchor": "left", "text-offset": [0.5, 0], "symbol-sort-key": ["get", "rank"] },
        paint: { "text-color": "#9fb0c2", "text-halo-color": LAND, "text-halo-width": 1.2 },
      },
      {
        id: "place-town", type: "symbol", source: "omt", "source-layer": "place", minzoom: 9, filter: ["==", ["get", "class"], "town"],
        layout: { "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]], "text-font": FONT, "text-size": 11.5, "text-anchor": "left", "text-offset": [0.5, 0] },
        paint: { "text-color": "#8394a8", "text-halo-color": LAND, "text-halo-width": 1.1 },
      },
    ],
  };
}

/** Hatch for "no value here": thin diagonal lines in a grey no data palette uses. */
export function hatchImage(): { width: number; height: number; data: Uint8Array } {
  const n = 16; // 8 css px at pixelRatio 2
  const data = new Uint8Array(n * n * 4);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const d = (x + y) % n;
    if (d === 0 || d === 1) data.set([176, 196, 220, 105], (y * n + x) * 4);
  }
  return { width: n, height: n, data };
}
