import type { LngLatBoundsLike } from "mapbox-gl";

/** CA shore + nearshore Pacific — slightly padded so zoom/clamp does not blank the canvas. */
export const CA_COAST_MAX_BOUNDS: LngLatBoundsLike = [
  [-128.35, 30.65],
  [-113.65, 43.35],
];

export const CA_COAST_INITIAL_VIEW = {
  longitude: -119.9,
  latitude: 36.05,
  zoom: 5.85,
};
