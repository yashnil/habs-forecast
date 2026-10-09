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
