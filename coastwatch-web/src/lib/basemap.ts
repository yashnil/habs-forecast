import type { StyleSpecification } from "maplibre-gl";

/**
 * Marine basemap: OpenFreeMap vector tiles (OpenMapTiles schema; free, no key) over a relief
 * built from NOAA NCEI ETOPO 2022 (pipeline/scripts/build_relief.py, public/basemap/).
 *
 * Drawing order, bottom to top:
 * - land tone, then land hillshade (clipped to the real coastline by the vector water);
 * - the sea, then seafloor shading. The seafloor is a neutral tone darker than every data colour
 *   and lies *under* every data layer, so it is never read as a value;
 * - the no-data hatch, shown only while a raster data layer is on (MapCanvas toggles it),
 *   so a gap in the data never looks like a low value;
 * - data layers, inserted before `coastline`;
 * - coastline, isobaths, rivers, roads, boundaries and labels, which stay above the data: depth
 *   and the shore stay readable over any field (signature handoff #2).
 * Label hierarchy: ports (MapCanvas) above towns above water names above roads. If the
 * tiles are unreachable the background still renders and the data still draws, and MapCanvas
 * shows the bundled water (lib/waterFallback) instead: the sea, its no-data hatch and the
 * coastline (the `*-fallback` layers, hidden otherwise).
 */
export const SEA = "#0b1d33";
// the land tone matches the relief image as drawn (graphite), so its edges never show as a seam
export const LAND = "#212529";
export const BEFORE_OVERLAY_ID = "coastline";
/** Layer ids of the no-data hatch, which MapCanvas shows only under a raster data layer. */
export const NODATA_LAYER_ID = "water-nodata";
/** The bundled water drawn while the basemap's tiles are unavailable (MapCanvas fills the source). */
export const WATER_FALLBACK_SOURCE = "water-fallback";
export const NODATA_FALLBACK_LAYER_ID = "water-nodata-fallback";
export const WATER_FALLBACK_LAYER_IDS = ["water-fallback", "coastline-fallback"];

/**
 * Map label glyphs are self-hosted (public/fonts/): place names, port labels and the port markers
 * (MapLibre holds a source's tiles until their labels' glyphs arrive) never wait on the font
 * server. Ranges not held locally still come from OpenFreeMap. Pass as the map's transformRequest.
 */
const LOCAL_GLYPH_FONTS = new Set(["Noto Sans Regular", "Noto Sans Italic"]);
const LOCAL_GLYPH_RANGES = new Set(["0-255", "256-511", "512-767", "768-1023", "7680-7935", "8192-8447", "8448-8703", "8704-8959"]);
export function transformRequest(url: string, resourceType?: string): { url: string } | undefined {
  if (resourceType !== "Glyphs" || typeof window === "undefined") return undefined;
  const m = /\/fonts\/([^/]+)\/(\d+-\d+)\.pbf$/.exec(url);
  if (!m) return undefined;
  const stack = decodeURIComponent(m[1]);
  if (!LOCAL_GLYPH_FONTS.has(stack) || !LOCAL_GLYPH_RANGES.has(m[2])) return undefined;
  return { url: `${window.location.origin}/fonts/${encodeURIComponent(stack)}/${m[2]}.pbf` };
}

const FONT = ["Noto Sans Regular"];
const FONT_ITALIC = ["Noto Sans Italic"];

/** Corners of the relief images (public/basemap/relief.json). */
export const RELIEF_CORNERS: [[number, number], [number, number], [number, number], [number, number]] = [
  [-126.6, 42.6],
  [-116.4, 42.6],
  [-116.4, 32.0],
  [-126.6, 32.0],
];
export const RELIEF_ATTRIBUTION =
  'Relief: <a href="https://doi.org/10.25921/fd45-gt74" target="_blank" rel="noopener noreferrer">NOAA NCEI ETOPO 2022</a> and <a href="https://www.ncei.noaa.gov/products/coastal-relief-model" target="_blank" rel="noopener noreferrer">Coastal Relief Model</a>';

export const MONTEREY_INSET_CORNERS: [[number, number], [number, number], [number, number], [number, number]] = [
  [-122.72, 37.33],
  [-121.58, 37.33],
  [-121.58, 36.22],
  [-122.72, 36.22],
];

const RELIEF_LAYER_IDS = new Set(["relief-land", "relief-sea", "relief-sea-monterey", "relief-sea-tint", "relief-sea-monterey-tint", "isobaths", "isobath-labels"]);
/** Depth-tinted seafloor, shown only while no data raster is (pipeline/scripts/build_relief.py). */
export const TINT_LAYER_IDS = ["relief-sea-tint", "relief-sea-monterey-tint"];

/** The marine basemap with or without the relief (land and seafloor shading, isobaths). */
export function basemapStyle({ relief }: { relief: boolean }): StyleSpecification {
  if (relief) return BASEMAP_STYLE;
  const sources = { ...BASEMAP_STYLE.sources };
  delete sources["relief-land"];
  delete sources["relief-sea"];
  delete sources["relief-sea-monterey"];
  delete sources["relief-sea-tint"];
  delete sources["relief-sea-monterey-tint"];
  delete sources.isobaths;
  return { ...BASEMAP_STYLE, sources, layers: BASEMAP_STYLE.layers.filter((l) => !RELIEF_LAYER_IDS.has(l.id)) };
}

export const BASEMAP_STYLE: StyleSpecification = {
  version: 8,
  glyphs: "https://tiles.openfreemap.org/fonts/{fontstack}/{range}.pbf",
  sources: {
    omt: {
      type: "vector",
      url: "https://tiles.openfreemap.org/planet",
      attribution:
        '<a href="https://openfreemap.org" target="_blank" rel="noopener noreferrer">OpenFreeMap</a> · © <a href="https://www.openmaptiles.org/" target="_blank" rel="noopener noreferrer">OpenMapTiles</a> · © <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap contributors</a>',
    },
    "relief-land": { type: "image", url: "/basemap/relief-land.webp", coordinates: RELIEF_CORNERS },
    "relief-sea": { type: "image", url: "/basemap/relief-sea.webp", coordinates: RELIEF_CORNERS },
    // Monterey Bay at 3 arc-second from the NOAA Coastal Relief Model, over the ETOPO relief
    "relief-sea-monterey": { type: "image", url: "/basemap/relief-sea-monterey.webp", coordinates: MONTEREY_INSET_CORNERS },
    "relief-sea-tint": { type: "image", url: "/basemap/relief-sea-tint.webp", coordinates: RELIEF_CORNERS },
    "relief-sea-monterey-tint": { type: "image", url: "/basemap/relief-sea-monterey-tint.webp", coordinates: MONTEREY_INSET_CORNERS },
    isobaths: { type: "geojson", data: "/basemap/isobaths.geojson", attribution: RELIEF_ATTRIBUTION },
    [WATER_FALLBACK_SOURCE]: { type: "geojson", data: { type: "FeatureCollection", features: [] } },
  },
  layers: [
    { id: "land", type: "background", paint: { "background-color": LAND } },
    // land recedes (v8, the Felt rule: basemap back, data forward): darker, nearly grey, the relief a
    // texture rather than a picture, so the sea and its data carry the colour
    { id: "relief-land", type: "raster", source: "relief-land", paint: { "raster-fade-duration": 0, "raster-resampling": "linear", "raster-saturation": -0.7, "raster-brightness-max": 0.62, "raster-contrast": -0.12, "raster-opacity": ["interpolate", ["linear"], ["zoom"], 4, 0.7, 9, 0.85] } },
    // under the basemap's water, which covers it wherever the basemap's tiles did arrive
    // (its polygons are cut along a tile grid: no anti-aliased outline, so the cuts never show)
    { id: "water-fallback", type: "fill", source: WATER_FALLBACK_SOURCE, filter: ["==", ["geometry-type"], "Polygon"], layout: { visibility: "none" }, paint: { "fill-color": SEA, "fill-antialias": false } },
    {
      id: "water",
      type: "fill",
      source: "omt",
      "source-layer": "water",
      // inland lakes and reservoirs stay quiet: the sea is the subject
      paint: { "fill-color": ["match", ["get", "class"], ["lake", "river"], "#1b2938", SEA] },
    },
    { id: "relief-sea", type: "raster", source: "relief-sea", paint: { "raster-fade-duration": 0, "raster-resampling": "linear" } },
    { id: "relief-sea-monterey", type: "raster", source: "relief-sea-monterey", minzoom: 7, paint: { "raster-fade-duration": 0, "raster-resampling": "linear" } },
    // the tinted seafloor sits over the neutral one and shows only with no data field on the map;
    // MapCanvas cross-fades it (420 ms) to the neutral relief when a field appears, never a toggle
    { id: "relief-sea-tint", type: "raster", source: "relief-sea-tint", paint: { "raster-fade-duration": 0, "raster-resampling": "linear", "raster-opacity": 1, "raster-opacity-transition": { duration: 420, delay: 0 } } },
    { id: "relief-sea-monterey-tint", type: "raster", source: "relief-sea-monterey-tint", minzoom: 7, paint: { "raster-fade-duration": 0, "raster-resampling": "linear", "raster-opacity": 1, "raster-opacity-transition": { duration: 420, delay: 0 } } },
    {
      // depth labels stay under any data field (a label over the field would hide its values);
      // the contour lines themselves are drawn above it
      id: "isobath-labels",
      type: "symbol",
      source: "isobaths",
      minzoom: 8,
      layout: {
        "symbol-placement": "line",
        "symbol-spacing": 420,
        "text-field": ["get", "label"],
        "text-font": FONT_ITALIC,
        "text-size": 10,
        "text-max-angle": 30,
      },
      paint: { "text-color": "rgba(170,196,222,0.62)", "text-halo-color": SEA, "text-halo-width": 1.2 },
    },
    // no data: fine dots on the water, shown by MapCanvas only under a raster data layer
    // (image "nodata", registered by MapCanvas); the data rasters cover them where a value exists
    { id: NODATA_LAYER_ID, type: "fill", source: "omt", "source-layer": "water", layout: { visibility: "none" }, paint: { "fill-pattern": "nodata" } },
    {
      id: NODATA_FALLBACK_LAYER_ID,
      type: "fill",
      source: WATER_FALLBACK_SOURCE,
      filter: ["==", ["geometry-type"], "Polygon"],
      layout: { visibility: "none" },
      paint: { "fill-pattern": "nodata", "fill-antialias": false },
    },
    {
      id: "coastline",
      type: "line",
      source: "omt",
      "source-layer": "water",
      paint: {
        "line-color": "#d3dfeb",
        "line-opacity": ["interpolate", ["linear"], ["zoom"], 4, 0.7, 8, 0.9],
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.5, 8, 0.9, 10, 1.3, 12, 1.8],
      },
    },
    {
      id: "coastline-fallback",
      type: "line",
      source: WATER_FALLBACK_SOURCE,
      filter: ["==", ["geometry-type"], "LineString"],
      layout: { visibility: "none" },
      paint: {
        "line-color": "#d3dfeb",
        "line-opacity": ["interpolate", ["linear"], ["zoom"], 4, 0.7, 8, 0.9],
        "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.5, 8, 0.9, 10, 1.3, 12, 1.8],
      },
    },
    {
      id: "isobaths",
      type: "line",
      source: "isobaths",
      minzoom: 5,
      paint: {
        // above every data field, so depth stays readable over the forecast (signature §basemap);
        // the shelf break (200 m) slightly stronger than deeper contours
        "line-color": ["case", ["==", ["get", "depth_m"], 200], "rgba(170,198,226,0.36)", "rgba(170,198,226,0.17)"],
        "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.5, 10, 1],
      },
    },
    {
      id: "rivers",
      type: "line",
      source: "omt",
      "source-layer": "waterway",
      minzoom: 8,
      filter: ["in", ["get", "class"], ["literal", ["river", "canal"]]],
      paint: { "line-color": "#3c5574", "line-width": 0.8 },
    },
    {
      id: "roads-major",
      type: "line",
      source: "omt",
      "source-layer": "transportation",
      minzoom: 6.5,
      filter: ["in", ["get", "class"], ["literal", ["motorway", "trunk"]]],
      paint: { "line-color": "#2a3645", "line-width": ["interpolate", ["linear"], ["zoom"], 6.5, 0.4, 12, 1.4] },
    },
    {
      id: "roads-primary",
      type: "line",
      source: "omt",
      "source-layer": "transportation",
      minzoom: 9.5,
      filter: ["==", ["get", "class"], "primary"],
      paint: { "line-color": "#384658", "line-width": ["interpolate", ["linear"], ["zoom"], 9.5, 0.3, 12, 1] },
    },
    {
      id: "boundary-state",
      type: "line",
      source: "omt",
      "source-layer": "boundary",
      // land boundaries only; maritime limits would read as coastal features
      filter: ["all", ["<=", ["coalesce", ["get", "admin_level"], 99], 4], ["!=", ["coalesce", ["get", "maritime"], 0], 1]],
      paint: { "line-color": "rgba(150,170,190,0.24)", "line-width": 0.7, "line-dasharray": [3, 2] },
    },
    {
      // ocean, bay and strait names: italic, letter-spaced, in the colour of the water labels on charts
      id: "water-name",
      type: "symbol",
      source: "omt",
      "source-layer": "water_name",
      minzoom: 5,
      filter: ["in", ["get", "class"], ["literal", ["ocean", "sea", "bay", "strait", "gulf"]]],
      layout: {
        "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]],
        "text-font": FONT_ITALIC,
        "text-size": ["interpolate", ["linear"], ["zoom"], 5, 11, 10, 14],
        "text-letter-spacing": 0.12,
        "text-max-width": 7,
      },
      // light text on a soft, translucent halo: readable over the dark sea and over the
      // forecast and chlorophyll colours without a hard outline
      paint: { "text-color": "rgba(222,234,246,0.9)", "text-halo-color": "rgba(6,17,30,0.5)", "text-halo-width": 1.2, "text-halo-blur": 0.8 },
    },
    {
      id: "place-city",
      type: "symbol",
      source: "omt",
      "source-layer": "place",
      minzoom: 5.5,
      filter: ["==", ["get", "class"], "city"],
      layout: {
        "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]],
        "text-font": FONT,
        "text-size": ["interpolate", ["linear"], ["zoom"], 5.5, 10.5, 10, 13.5],
        "text-anchor": "left",
        "text-offset": [0.4, 0],
        "symbol-sort-key": ["get", "rank"],
      },
      paint: { "text-color": "#93a3b4", "text-halo-color": LAND, "text-halo-width": 1.2 },
    },
    {
      id: "place-town",
      type: "symbol",
      source: "omt",
      "source-layer": "place",
      minzoom: 8,
      filter: ["==", ["get", "class"], "town"],
      layout: {
        "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]],
        "text-font": FONT,
        "text-size": ["interpolate", ["linear"], ["zoom"], 8, 11, 11, 13],
        "text-anchor": "left",
        "text-offset": [0.4, 0],
        "symbol-sort-key": ["get", "rank"],
      },
      paint: { "text-color": "#8494a6", "text-halo-color": LAND, "text-halo-width": 1.2 },
    },
    {
      id: "place-village",
      type: "symbol",
      source: "omt",
      "source-layer": "place",
      minzoom: 10.5,
      filter: ["==", ["get", "class"], "village"],
      layout: {
        "text-field": ["coalesce", ["get", "name:en"], ["get", "name"]],
        "text-font": FONT,
        "text-size": 11,
        "text-anchor": "left",
        "text-offset": [0.4, 0],
      },
      paint: { "text-color": "#8e9db0", "text-halo-color": LAND, "text-halo-width": 1.1 },
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

/** No data: a fine diagonal hatch (the "not known" mark of the design language), faint enough
 *  that a large gap reads as quiet texture rather than noise, never as a low value, and the
 *  seafloor relief shows through. A seamless 11 px tile, drawn at 2x for sharp lines. */
export function noDataImage(): { width: number; height: number; data: Uint8Array } {
  const n = 22; // 11 css px at pixelRatio 2: calmer over a large gap
  const data = new Uint8Array(n * n * 4);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const d = (x + y) % n;
      // a 1.5 css px line with soft edges, so it does not shimmer as the map moves
      const a = d === 0 || d === 1 ? 52 : d === 2 || d === n - 1 ? 18 : 0;
      if (a) data.set([170, 196, 224, a], (y * n + x) * 4);
    }
  return { width: n, height: n, data };
}

/** Current-speed classes (m/s, lower bounds) and the arrow shaft length drawn for each.
 *  Discrete classes: a reader can match an arrow to the legend exactly. */
export const SPEED_CLASSES = [0, 0.1, 0.25, 0.5, 1] as const;
export const ARROW_LENGTHS = [7, 11, 16, 22, 28] as const; // px at icon-size 1

/** Arrow pointing north, centred on the image (and so on its cell when rotated), with a
 *  constant line width and a light fill outlined in dark navy so it reads over the dark
 *  sea and over bright chlorophyll alike. Drawn at 2x; register with pixelRatio 2. */
export function arrowImage(length: number): ImageData {
  const k = 2;
  const W = 16;
  const H = 32;
  const c = document.createElement("canvas");
  c.width = W * k;
  c.height = H * k;
  const g = c.getContext("2d")!;
  g.scale(k, k);
  const cx = W / 2;
  const top = H / 2 - length / 2;
  const bottom = H / 2 + length / 2;
  const head = Math.min(6, length * 0.45);
  const path = () => {
    g.beginPath();
    g.moveTo(cx, bottom);
    g.lineTo(cx, top + head * 0.6);
    g.moveTo(cx - head * 0.62, top + head);
    g.lineTo(cx, top);
    g.lineTo(cx + head * 0.62, top + head);
  };
  g.lineCap = "round";
  g.lineJoin = "round";
  path();
  g.strokeStyle = "rgba(6,17,30,0.92)";
  g.lineWidth = 4;
  g.stroke();
  path();
  g.strokeStyle = "#f2f6fa";
  g.lineWidth = 1.8;
  g.stroke();
  return g.getImageData(0, 0, W * k, H * k);
}
