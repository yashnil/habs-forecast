/**
 * The bottom scrim's stops: [share of the stage's height from the top, alpha of the ground colour]. Text sits
 * in the lower part, from 58% down. Held as data so a test proves the contrast over the brightest possible
 * pixel (white) rather than trusting the picture.
 */
export const SCRIM_STOPS: readonly [number, number][] = [
  [0.26, 0],
  [0.42, 0.46],
  [0.58, 0.72],
  [0.78, 0.82],
  [1, 0.85],
];
/** From here down the scrim guarantees AA for body ink over any map pixel. */
export const SCRIM_TEXT_FROM = 0.58;

export const scrimGradient = (ground = "var(--nx-bg, #04090e)") =>
  `linear-gradient(to bottom, ${SCRIM_STOPS.map(([y, a]) => `color-mix(in srgb, ${ground} ${Math.round(a * 100)}%, transparent) ${Math.round(y * 100)}%`).join(", ")})`;
