/**
 * Neritic motion presets (Motion for React). Every animation in src/nx goes through these; components never
 * invent timings. Spec: docs/coastwatch/v12/design-system/README.md section 8.
 * Truth rule: data frames never tween (T26). Use `cut` (no transition) between time steps.
 */
import type { Transition } from "motion/react";

export const EASE_CHART = [0.65, 0, 0.35, 1] as const;

/** Lines drawing: isobaths, tide curve, series. Pass a duration within the range for the line's length. */
export const ink = (duration = 1.2): Transition => ({ duration: Math.min(2.1, Math.max(0.9, duration)), ease: EASE_CHART });

/** The sheet's camera between pages, and anything that travels with it. */
export const pan: Transition = { duration: 0.62, ease: EASE_CHART };

/** Figures and bars changing value (ledger, readouts, waterfall). */
export const settle: Transition = { type: "spring", stiffness: 260, damping: 30 };

/** A sounding or note appearing at its place on the chart. */
export const register: Transition = { duration: 0.24, ease: [0.2, 0, 0, 1] };
export const registerFrom = { opacity: 0, scale: 0.96 } as const;
export const registerTo = { opacity: 1, scale: 1 } as const;

/** Phone margin sheet and readouts. */
export const sheet: Transition = { type: "spring", stiffness: 380, damping: 36 };

/** Every pressable control. */
export const press = { whileTap: { scale: 0.97 }, transition: { duration: 0.12 } } as const;

/** Between data frames: a cut. Exported so intent is explicit at call sites. */
export const cut: Transition = { duration: 0 };

/** Margin content when the route changes (the chart pans underneath). */
export const marginSwap: Transition = { duration: 0.18, ease: [0.2, 0, 0, 1] };

/** A layer handing over to another (opening caption, drawn lines giving way to live ones). */
export const fadeOut: Transition = { duration: 0.3, ease: [0.2, 0, 0, 1] };

/** The first-visit opening: phase boundaries in seconds. Statement is visible from 0. */
export const OPENING = {
  total: 4.8,
  canyonHold: [0, 1.4],
  settleToChart: [1.2, 3.0],
  isobathsInk: [1.6, 3.6],
  tintsFlood: [2.6, 4.0],
  neatlineClose: [3.6, 4.4],
  soundingsRegister: [4.3, 4.8],
} as const;
