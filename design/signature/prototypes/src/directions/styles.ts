import type { LayerSpecification, StyleSpecification } from "maplibre-gl";
import { signatureStyle } from "../map/style";

/**
 * Basemap variants for the three directions, all derived from the signature style (same sources,
 * same drawing order, same data). Only paints change, so geography is identical across directions.
 */
export type Variant = "a" | "b" | "c";

type Paint = Record<string, unknown>;
function restyle(s: StyleSpecification, edits: Record<string, Paint | "hide">): StyleSpecification {
  return {
    ...s,
    layers: s.layers.map((l) => {
      const e = edits[l.id];
      if (!e) return l;
      if (e === "hide") return { ...l, layout: { ...(l as { layout?: object }).layout, visibility: "none" } } as LayerSpecification;
      return { ...l, paint: { ...(l as { paint?: object }).paint, ...e } } as LayerSpecification;
    }),
  };
}

/** Base colour each variant's forecast cells are mixed toward (the sea under the field). */
export const FIELD_BASE: Record<Variant, [number, number, number]> = { a: [11, 29, 51], b: [203, 219, 228], c: [222, 228, 224] };

export function variantStyle(v: Variant): StyleSpecification {
  const s = signatureStyle();
  if (v === "a") return s;
  if (v === "b") {
    // Living Coast: soft blue-grey sea with the depth tint lifted, warm sand land, dark labels
    const label = { "text-color": "#2c3c4c", "text-halo-color": "rgba(246,243,236,0.9)", "text-halo-width": 1.3, "text-halo-blur": 0 };
    return restyle(s, {
      land: { "background-color": "#ece4d4" },
      "relief-land": { "raster-opacity": 0.32, "raster-brightness-min": 0.62, "raster-saturation": -0.6 },
      "water-fallback": { "fill-color": "#cbdbe4" },
      water: { "fill-color": "#cbdbe4" },
      "relief-sea": "hide",
      "relief-sea-m": "hide",
      "relief-tint": { "raster-opacity": 0.55, "raster-brightness-min": 0.5, "raster-saturation": -0.35, "raster-contrast": -0.25 },
      "relief-tint-m": { "raster-opacity": 0.55, "raster-brightness-min": 0.5, "raster-saturation": -0.35, "raster-contrast": -0.25 },
      isobaths: { "line-color": ["case", ["==", ["get", "depth_m"], 200], "rgba(44,92,122,0.35)", "rgba(44,92,122,0.16)"] },
      "isobath-labels": { "text-color": "rgba(44,92,122,0.7)", "text-halo-color": "rgba(220,232,238,0.9)" },
      coastline: { "line-color": "#6f8798", "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.6, 10, 1.2, 13, 1.8] },
      "coastline-fallback": { "line-color": "#6f8798" },
      rivers: { "line-color": "#9db7c6" },
      "roads-major": { "line-color": "#d8cdb8" },
      "boundary-state": { "line-color": "#b3a78f" },
      "water-name": { "text-color": "#3f6278", "text-halo-color": "rgba(214,228,236,0.7)" },
      "place-city": label,
      "place-town": { ...label, "text-color": "#5b6875" },
    });
  }
  // Editorial Atlas: a printed plate — pale grey-green sea, paper land, ink linework, no tint
  const ink = { "text-color": "#1d2b38", "text-halo-color": "rgba(240,236,226,0.95)", "text-halo-width": 1.4, "text-halo-blur": 0 };
  return restyle(s, {
    land: { "background-color": "#efe9dc" },
    "relief-land": { "raster-opacity": 0.22, "raster-brightness-min": 0.7, "raster-saturation": -1 },
    "water-fallback": { "fill-color": "#dee4e0" },
    water: { "fill-color": "#dee4e0" },
    "relief-sea": { "raster-opacity": 0.18, "raster-brightness-min": 0.7, "raster-saturation": -1 },
    "relief-sea-m": { "raster-opacity": 0.18, "raster-brightness-min": 0.7, "raster-saturation": -1 },
    "relief-tint": "hide",
    "relief-tint-m": "hide",
    isobaths: { "line-color": ["case", ["==", ["get", "depth_m"], 200], "rgba(29,43,56,0.42)", "rgba(29,43,56,0.18)"], "line-width": 0.7 },
    "isobath-labels": { "text-color": "rgba(29,43,56,0.65)", "text-halo-color": "rgba(222,228,224,0.95)" },
    coastline: { "line-color": "#1d2b38", "line-width": ["interpolate", ["linear"], ["zoom"], 4, 0.6, 10, 1.1, 13, 1.6] },
    "coastline-fallback": { "line-color": "#1d2b38" },
    rivers: { "line-color": "#a9b9bf" },
    "roads-major": "hide",
    "boundary-state": { "line-color": "#9a8f78" },
    "water-name": { "text-color": "#41525f", "text-halo-color": "rgba(222,228,224,0.8)" },
    "place-city": ink,
    "place-town": { ...ink, "text-color": "#56626c" },
  });
}
