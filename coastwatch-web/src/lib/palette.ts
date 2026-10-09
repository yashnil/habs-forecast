import type { Palette } from "@/generated/schema";

/**
 * Forecast probability display classes, "cw-probability-classes-v1" (design reset §5.1).
 * Ten 10-percentage-point colour steps for reading the map. They are display intervals,
 * not risk categories: exact values always come from the published grid.
 *
 * Mirrored by the `--cw-p0`…`--cw-p9` tokens in `src/app/globals.css` (a unit test keeps
 * them equal). The map adopts them in phase P1, together with the pipeline palette.
 */
export const FORECAST_CLASSES_ID = "cw-probability-classes-v1";

export const FORECAST_CLASSES = [
  "#3a385b",
  "#4c436a",
  "#5f4e79",
  "#735986",
  "#886492",
  "#9c709c",
  "#af7ea4",
  "#c28cab",
  "#d39cb3",
  "#e5abbc",
] as const;

/** Display class for a probability in [0, 1]; 1.0 falls in the top class. */
export function forecastClass(p: number): number {
  return Math.max(0, Math.min(9, Math.floor(p * 10)));
}

// ---------------------------------------------------------------- palette helpers (pure)

const hex = (h: string, k: number) => parseInt(h.slice(1 + 2 * k, 3 + 2 * k), 16);

function position(p: Palette, v: number): number {
  const x = p.scale === "log10" ? Math.log10(Math.max(v, 1e-12)) : v;
  const [lo, hi] = p.domain;
  return Math.min(hi, Math.max(lo, x));
}

/** Palette colour for a value, exactly as the pipeline renders it (stepped or linear, linear or log scale). */
export function colorAt(p: Palette, v: number): string {
  const t = position(p, v);
  const stops = p.stops;
  if (p.interpolation === "step") {
    let i = 0;
    while (i < stops.length - 1 && stops[i + 1].value <= t) i++;
    return stops[i].color;
  }
  let i = 0;
  while (i < stops.length - 2 && stops[i + 1].value < t) i++;
  const a = stops[i];
  const b = stops[i + 1];
  const f = Math.min(1, Math.max(0, (t - a.value) / Math.max(1e-9, b.value - a.value)));
  const mix = [0, 1, 2].map((k) => Math.round(hex(a.color, k) + (hex(b.color, k) - hex(a.color, k)) * f));
  return `rgb(${mix.join(",")})`;
}

/** CSS background for a legend bar: hard edges for stepped palettes, gradient otherwise. */
export function gradientCss(p: Palette): string {
  const [lo, hi] = p.domain;
  const pos = (x: number) => (((x - lo) / (hi - lo)) * 100).toFixed(2);
  if (p.interpolation === "step") {
    const parts = p.stops.map((s, i) => `${s.color} ${pos(s.value)}% ${pos(p.stops[i + 1]?.value ?? hi)}%`);
    return `linear-gradient(to right, ${parts.join(", ")})`;
  }
  return `linear-gradient(to right, ${p.stops.map((s) => `${s.color} ${pos(s.value)}%`).join(", ")})`;
}

/** Colours for the age of a latest-clear-view pixel, index = days. Mirrors the pipeline's AGE_COLOURS. */
export const AGE_COLOURS = ["#e8f1f8", "#bcd3e6", "#8fb2d0", "#6790b5", "#4a7097", "#365477", "#273d58", "#1c2c40"];
/** Which sensor a multi-sensor pixel comes from: Sentinel-3 OLCI, VIIRS. Mirrors the pipeline's SENSOR_COLOURS. */
export const SENSOR_COLOURS = ["#2f6db5", "#e39a2d"];
