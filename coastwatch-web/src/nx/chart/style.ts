import type { StyleSpecification } from "maplibre-gl";
import CHART from "../../../public/chart/chart.json";

/**
 * The chart's MapLibre style, built from the active mode's tokens (src/nx/ds/tokens.css: day, dusk, night,
 * each specified, never inverted). Everything it draws is bundled under public/chart (pipeline/scripts/
 * build_chart.py): no tile server, so the chart is the same online, offline and in a saved copy.
 *
 * Bottom to top: the open ocean (palest tint), ENC depth areas deepest first (shoal water most saturated),
 * land, land relief at 1x, minor and index isobaths, depth labels, the coastline. Names, ports, soundings and
 * notes are drawn by the page over the canvas (Literata italic water names need real type, not glyph atlases).
 */
export type ChartPalette = {
  depth: string[]; // 0 (shoal) .. 8 (deep)
  land: string;
  reliefOpacity: number;
  isobath: string;
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
export const ISOBATH_LAYERS = ["isobaths-minor", "isobaths-index", "isobath-labels"];

/** The active mode's colours, read from the CSS custom properties on <html>. */
export function readPalette(el: Element = document.documentElement): ChartPalette {
  const cs = getComputedStyle(el);
  const v = (k: string) => cs.getPropertyValue(`--${k}`).trim();
  return {
    depth: Array.from({ length: 9 }, (_, i) => v(`depth-${i}`)),
    land: v("land"),
    reliefOpacity: Number(v("land-relief-opacity") || 0.5),
    isobath: v("isobath"),
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
    ["relief-land", "raster-opacity", p.reliefOpacity],
    ["relief-land-monterey", "raster-opacity", p.reliefOpacity],
    ["isobaths-minor", "line-color", p.isobath],
    ["isobaths-index", "line-color", p.isobath],
    ["isobath-labels", "text-color", p.waterName],
    ["isobath-labels", "text-halo-color", p.halo],
    ["coastline", "line-color", p.ink],
  ];
}

const FONT_ITALIC = ["Noto Sans Italic"];

export function chartStyle(p: ChartPalette, origin: string, { hidden = false }: { hidden?: boolean } = {}): StyleSpecification {
  // the opening starts with the water unfilled and the lines undrawn; it raises these
  const water = hidden ? 0 : 1;
  return {
    version: 8,
    glyphs: `${origin}/fonts/{fontstack}/{range}.pbf`,
    sources: {
      depth: { type: "geojson", data: "/chart/depth.geojson", attribution: "NOAA NCEI ETOPO 2022 and Coastal Relief Model" },
      land: { type: "geojson", data: "/chart/land.geojson", attribution: "© OpenMapTiles © OpenStreetMap contributors" },
      isobaths: { type: "geojson", data: "/chart/isobaths.geojson" },
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
      { id: "land", type: "fill", source: "land", paint: { "fill-color": p.land, "fill-antialias": true } },
      { id: "relief-land", type: "raster", source: "relief-land", paint: { "raster-opacity": p.reliefOpacity, "raster-fade-duration": 0, "raster-resampling": "linear" } },
      { id: "relief-land-monterey", type: "raster", source: "relief-land-monterey", minzoom: 8, paint: { "raster-opacity": p.reliefOpacity, "raster-fade-duration": 0, "raster-resampling": "linear" } },
      {
        id: "isobaths-minor",
        type: "line",
        source: "isobaths",
        minzoom: 8.5,
        filter: ["==", ["get", "i"], 0],
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": p.isobath, "line-width": ["interpolate", ["linear"], ["zoom"], 9, 0.5, 12, 0.8], "line-opacity": hidden ? 0 : 0.42 },
      },
      {
        id: "isobaths-index",
        type: "line",
        source: "isobaths",
        filter: ["==", ["get", "i"], 1],
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": p.isobath, "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.6, 9, 0.9, 12, 1.2], "line-opacity": hidden ? 0 : 0.78 },
      },
      {
        id: "isobath-labels",
        type: "symbol",
        source: "isobaths",
        minzoom: 7.5,
        filter: ["all", ["==", ["get", "i"], 1], ["<=", ["get", "d"], 2000]],
        layout: {
          "symbol-placement": "line",
          "symbol-spacing": ["interpolate", ["linear"], ["zoom"], 8, 300, 12, 520],
          "text-field": ["get", "label"],
          "text-font": FONT_ITALIC,
          "text-size": 14,
          "text-max-angle": 28,
          "text-keep-upright": true,
          "text-padding": 6,
        },
        paint: { "text-color": p.waterName, "text-halo-color": p.halo, "text-halo-width": 3, "text-opacity": hidden ? 0 : 1 },
      },
      {
        id: "coastline",
        type: "line",
        source: "land",
        layout: { "line-join": "round" },
        // the coast's outline also traces every slough and lagoon (Elkhorn Slough at Moss Landing): both banks of a
        // narrow channel drawn heavy read as a black scrawl at harbour zoom, so the line thins and softens as it nears
        paint: {
          "line-color": p.ink,
          "line-width": ["interpolate", ["linear"], ["zoom"], 5, 0.6, 9, 0.95, 11, 0.9, 13, 1.05],
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 9, 1, 11, 0.78, 13, 0.7],
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
