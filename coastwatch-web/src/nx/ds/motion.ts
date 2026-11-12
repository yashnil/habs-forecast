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

/* ---------------------------------------------------------------- v13 additions (v13 README §5) */

/** Layout travel (Motion's `layout` prop): the desk reflowing, disclosure siblings, a saved list reordering. */
export const reflow: Transition = { type: "spring", stiffness: 420, damping: 42 };

/** The chart pane opening, closing or changing size; the chart's refit runs with `pan` beside it. */
export const pane: Transition = { duration: 0.42, ease: EASE_CHART };

/** A shared element (Motion `layoutId`): "this became that" (a port name into a title, the nav underline). */
export const shared: Transition = { duration: 0.36, ease: [0.2, 0, 0, 1] };

/** A selection moving: the segmented indicator, the chosen cell, the pinned point's ring. */
export const select: Transition = { duration: 0.2, ease: [0.2, 0, 0, 1] };

/** Height and opacity of a disclosure's body. */
export const reveal: Transition = { duration: 0.22, ease: [0.2, 0, 0, 1] };

/**
 * A figure moving from its old value to its new one (ds/Figure, `useSettledNumber`): critically damped, so it
 * never passes its target (a figure that is never negative never shows a negative on the way). It never counts
 * up from zero on arrival: the first value is drawn as it is.
 */
export const value: Transition = { type: "spring", stiffness: 260, damping: 34, restDelta: 0.001 };

/** A route's page changing, in sequence: the old page out, then the new one in. Never both on screen at once. */
export const swapOut: Transition = { duration: 0.09, ease: "linear" };
export const swapIn: Transition = { duration: 0.18, ease: [0.2, 0, 0, 1], delay: 0.09 };
export const swap = { out: swapOut, in: swapIn } as const;

/** How far a scroller moves before a sticky header condenses (ds/layout `useScrollCondense`), px. */
export const SCROLL_CONDENSE = 64;
export const scrollCondense = { after: SCROLL_CONDENSE } as const;

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
