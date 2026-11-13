import type { StyleSpecification } from "maplibre-gl";
import CHART from "../../../public/chart/chart.json";

/**
 * The chart's MapLibre style, v14 synthesis (docs/coastwatch/v14/design-system/README.md §4), built from the active
 * mode's tokens (src/nx/ds/tokens.css). Everything it draws is bundled under public/chart (pipeline/scripts/
 * build_chart.py, scripts/build-contours.mjs): no tile server, so the chart is the same online, offline and saved.
 *
 * The lit sea (after the approved prototype's canvas basemap, now from the real data): the depth areas painted
 * deepest first, shoal pale and close in value to 200 m, the slope darkening fast, the deep one saturated blue; every
 * area's edge a fine light isobath; the index isobaths brighter. The Depth layer adds the 20, 50, 100 and 200 m edges
 * as ink contours (labelled in fathoms by DepthLabels.tsx). Land in the warm paper tone with a whisper of relief; the
 * shore is a pale surf band and a white casing under the coast line (stroked under the land, so only the water side
 * shows). A graticule, light on the water and faint on the land. Names, ports, marks and depth labels are drawn by
 * the page over the canvas in Figtree (real type, not glyph atlases).
 */
export type ChartPalette = {
  depth: string[]; // 0 (shoal) .. 8 (deep)
  land: string;
  reliefOpacity: number;
  isobath: string;
  contour: string;
  coast: string;
  graticule: string;
  graticuleLand: string;
  ink: string;
  waterName: string;
  halo: string;
  paper: string;
};

export const DEPTH_THRESHOLDS = CHART.depth_areas.thresholds_m as number[];
const TINT_OF = CHART.depth_areas.tint_of as Record<string, number>;
const IMG = CHART.images as unknown as Record<string, { corners_lnglat: [number, number][] }>;
const corners = (name: string) => IMG[name].corners_lnglat as [[number, number], [number, number], [number, number], [number, number]];

export const CHART_FRAME = CHART.frame as { ca: [number, number, number, number]; inset: [number, number, number, number]; bounds: [number, number, number, number] };
export const DEPTH_LAYERS = DEPTH_THRESHOLDS.map((d) => `depth-${d}`);
/** The Depth layer's own map layers (shown and hidden with the chip, faded with `layerIn`/`layerOut`). */
export const DEPTH_LAYER_IDS = ["contours", "contours-shelf"];
/** The depth areas whose edges the Depth layer draws: the nearest levels the data holds to 10/20/30/50/100 fm. */
export const CONTOUR_DEPTHS_M = [20, 50, 100, 200];
export const ISOBATH_LAYERS = ["isobaths-minor", "isobaths-index", "band-edges"];

/** The active mode's colours, read from the CSS custom properties on <html>. */
export function readPalette(el: Element = document.documentElement): ChartPalette {
  const cs = getComputedStyle(el);
  const v = (k: string) => cs.getPropertyValue(`--${k}`).trim();
  return {
    depth: Array.from({ length: 9 }, (_, i) => v(`depth-${i}`)),
    land: v("land"),
    reliefOpacity: Number(v("land-relief-opacity") || 0.2),
    isobath: v("isobath"),
    contour: v("contour"),
    coast: v("coast"),
    graticule: v("graticule"),
    graticuleLand: v("graticule-land"),
    ink: v("ink"),
    waterName: v("water-name"),
    halo: v("halo"),
    paper: v("paper"),
  };
}

/** Paint values that follow the mode; the host re-applies them when the mode changes. */
export function modePaint(p: ChartPalette): Array<[string, string, unknown]> {
  return [
    ["background", "background-color", p.depth[8]],
    ...DEPTH_THRESHOLDS.map((d) => [`depth-${d}`, "fill-color", p.depth[TINT_OF[String(d)]]] as [string, string, unknown]),
    ["land", "fill-color", p.land],
    ["land-beyond", "fill-color", p.land],
    ["relief-land", "raster-opacity", p.reliefOpacity],
    ["relief-land-monterey", "raster-opacity", p.reliefOpacity],
    ["band-edges", "line-color", p.isobath],
    ["isobaths-minor", "line-color", p.isobath],
    ["isobaths-index", "line-color", p.isobath],
    ["contours", "line-color", p.contour],
    ["contours-shelf", "line-color", p.contour],
    ["graticule", "line-color", p.graticule],
    ["graticule-land", "line-color", p.graticuleLand],
    ["shore-surf", "line-color", p.halo],
    ["shore-casing", "line-color", p.halo],
    ["coastline", "line-color", p.coast],
  ];
}

/** Lines of latitude and longitude every quarter degree over the chart's frame, each tagged with its step. */
export function graticule(): GeoJSON.FeatureCollection<GeoJSON.LineString, { step: number }> {
  const [w, s, e, n] = CHART_FRAME.bounds;
  const features: GeoJSON.Feature<GeoJSON.LineString, { step: number }>[] = [];
  const stepOf = (v: number) => (Math.abs(v - Math.round(v)) < 1e-6 ? 1 : Math.abs(v * 2 - Math.round(v * 2)) < 1e-6 ? 0.5 : 0.25);
  for (let lon = Math.ceil(w * 4) / 4; lon <= e; lon += 0.25) features.push({ type: "Feature", properties: { step: stepOf(lon) }, geometry: { type: "LineString", coordinates: [[lon, s], [lon, n]] } });
  for (let lat = Math.ceil(s * 4) / 4; lat <= n; lat += 0.25) {
    const pts: [number, number][] = [];
    for (let lon = w; lon <= e + 1e-9; lon += 0.5) pts.push([lon, lat]);
    features.push({ type: "Feature", properties: { step: stepOf(lat) }, geometry: { type: "LineString", coordinates: pts } });
  }
  return { type: "FeatureCollection", features };
}

/** The graticule's step by zoom, as the prototype's (1° far out, ½° at a coast's scale, ¼° close in). */
const GRATICULE_FILTER = ["any", ["==", ["get", "step"], 1], ["all", [">=", ["zoom"], 7.2], ["==", ["get", "step"], 0.5]], [">=", ["zoom"], 8.6]] as unknown;

export function chartStyle(p: ChartPalette, origin: string, { hidden = false, depth = true }: { hidden?: boolean; depth?: boolean } = {}): StyleSpecification {
  // the opening starts with the water unfilled and the lines undrawn; it raises these
  const water = hidden ? 0 : 1;
  const fade = { duration: 320, delay: 0 };
  return {
    version: 8,
    glyphs: `${origin}/fonts/{fontstack}/{range}.pbf`,
    sources: {
      depth: { type: "geojson", data: "/chart/depth.geojson", attribution: "NOAA NCEI ETOPO 2022 and Coastal Relief Model" },
      contours: { type: "geojson", data: "/chart/contours.geojson" },
      land: { type: "geojson", data: "/chart/land.geojson", attribution: "© OpenMapTiles © OpenStreetMap contributors" },
      isobaths: { type: "geojson", data: "/chart/isobaths.geojson" },
      graticule: { type: "geojson", data: graticule() },
      // inland of the bundled coast the chart is all land: the full-bleed sheet can show past the data's east edge
      "land-beyond": { type: "geojson", data: { type: "Polygon", coordinates: [[[CHART_FRAME.bounds[2] - 0.05, 20], [-60, 20], [-60, 62], [CHART_FRAME.bounds[2] - 0.05, 62], [CHART_FRAME.bounds[2] - 0.05, 20]]] } },
      "relief-land": { type: "image", url: "/chart/relief-land.webp", coordinates: corners("relief-land.webp") },
      "relief-land-monterey": { type: "image", url: "/chart/relief-land-monterey.webp", coordinates: corners("relief-land-monterey.webp") },
    },
    layers: [
      { id: "background", type: "background", paint: { "background-color": p.depth[8] } },
      ...DEPTH_THRESHOLDS.map((d) => ({
        id: `depth-${d}`,
        type: "fill" as const,
        source: "depth",
        filter: ["==", ["get", "d"], d] as unknown as ["==", string, number],
        paint: { "fill-color": p.depth[TINT_OF[String(d)]], "fill-antialias": true, "fill-opacity": water, "fill-opacity-transition": { duration: 0, delay: 0 } },
      })),
      // every area's edge (10, 20, 50 … m) is an isobath: a fine light line on the lit sea
      { id: "band-edges", type: "line", source: "contours", layout: { "line-join": "round" }, paint: { "line-color": p.isobath, "line-width": 0.8, "line-opacity": hidden ? 0 : 0.26 } },
      { id: "isobaths-minor", type: "line", source: "isobaths", minzoom: 8.6, filter: ["==", ["get", "i"], 0], layout: { "line-join": "round", "line-cap": "round" }, paint: { "line-color": p.isobath, "line-width": 0.8, "line-opacity": hidden ? 0 : 0.3 } },
      {
        id: "isobaths-index",
        type: "line",
        source: "isobaths",
        filter: ["==", ["get", "i"], 1],
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": p.isobath, "line-width": 1.1, "line-opacity": hidden ? 0 : ["case", [">=", ["get", "d"], 3000], 0.34, 0.62] },
      },
      // the Depth layer: the 11, 27 and 55 fm contours, and the shelf edge (109 fm, 200 m) a little heavier
      {
        id: "contours",
        type: "line",
        source: "contours",
        filter: ["in", ["get", "d"], ["literal", [20, 50, 100]]] as unknown as ["in", string, number[]],
        layout: { "line-join": "round", visibility: depth ? "visible" : "none" },
        paint: { "line-color": p.contour, "line-width": 1.15, "line-opacity": depth ? 0.42 : 0, "line-opacity-transition": fade },
      },
      {
        id: "contours-shelf",
        type: "line",
        source: "contours",
        filter: ["==", ["get", "d"], 200] as unknown as ["==", string, number],
        layout: { "line-join": "round", visibility: depth ? "visible" : "none" },
        paint: { "line-color": p.contour, "line-width": 1.5, "line-opacity": depth ? 0.62 : 0, "line-opacity-transition": fade },
      },
      { id: "graticule", type: "line", source: "graticule", filter: GRATICULE_FILTER as ["==", string, number], paint: { "line-color": p.graticule, "line-width": 1 } },
      // the shore: a pale surf band and a white casing, stroked before the land is filled so only the water side shows
      { id: "shore-surf", type: "line", source: "land", layout: { "line-join": "round" }, paint: { "line-color": p.halo, "line-width": ["interpolate", ["linear"], ["zoom"], 5, 6, 9, 14], "line-opacity": 0.2 } },
      { id: "shore-casing", type: "line", source: "land", layout: { "line-join": "round" }, paint: { "line-color": p.halo, "line-width": ["interpolate", ["linear"], ["zoom"], 5, 2, 9, 3.4], "line-opacity": 0.95 } },
      { id: "land", type: "fill", source: "land", paint: { "fill-color": p.land, "fill-antialias": true } },
      { id: "land-beyond", type: "fill", source: "land-beyond", paint: { "fill-color": p.land, "fill-antialias": false } },
      { id: "relief-land", type: "raster", source: "relief-land", paint: { "raster-opacity": p.reliefOpacity, "raster-fade-duration": 0, "raster-resampling": "linear", "raster-saturation": -0.6 } },
      // at every zoom: the statewide relief steps back inside the inset (the cross-fade), so hiding this one
      // below some zoom would leave a pale square off Monterey
      { id: "relief-land-monterey", type: "raster", source: "relief-land-monterey", paint: { "raster-opacity": p.reliefOpacity, "raster-fade-duration": 0, "raster-resampling": "linear", "raster-saturation": -0.6 } },
      { id: "graticule-land", type: "line", source: "graticule", filter: GRATICULE_FILTER as ["==", string, number], paint: { "line-color": p.graticuleLand, "line-width": 1 } },
      {
        id: "coastline",
        type: "line",
        source: "land",
        layout: { "line-join": "round" },
        // the coast's outline also traces every slough and lagoon: it thins and softens as the camera nears
        paint: {
          "line-color": p.coast,
          "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.8, 9, 1.15, 13, 1.15],
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 9, 1, 11, 0.85, 13, 0.75],
        },
      },
    ],
  } as StyleSpecification;
}

/** The baked chart images of the static tier, per mode: statewide, then the Monterey inset over it. */
export function posterImages(mode: string): Array<{ url: string; bounds: [[number, number], [number, number]] }> {
  const box = (name: string) => {
    const c = corners(name);
    return [
      [c[0][0], c[2][1]],
      [c[1][0], c[0][1]],
    ] as [[number, number], [number, number]];
  };
  return [`chart-${mode}-ca.webp`, `chart-${mode}-monterey.webp`].map((n) => ({ url: `/chart/${n}`, bounds: box(n) }));
}

/**
 * East of the statewide image the land runs on to the chart's bounds: the static tier lays plain land there,
 * so a wide or zoomed-out still never ends in a pale edge where the bake stops.
 */
export function posterLandEast(): [[number, number], [number, number]] {
  const c = corners(`chart-day-ca.webp`);
  return [
    [c[1][0], CHART_FRAME.bounds[1]],
    [CHART_FRAME.bounds[2], CHART_FRAME.bounds[3]],
  ];
}

export const STILL = (() => {
  const c = corners("opening-canyon.webp");
  return {
    url: "/chart/opening-canyon.webp",
    bounds: [
      [c[0][0], c[2][1]],
      [c[1][0], c[0][1]],
    ] as [[number, number], [number, number]],
    exaggeration: (IMG["opening-canyon.webp"] as unknown as { exaggeration: number }).exaggeration,
  };
})();
