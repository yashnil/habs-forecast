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
export const value: Transition = { type: "spring", stiffness: 260, damping: 34, restDelta: 0.005, restSpeed: 0.5 };

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

/* ---------------------------------------------------------------- v14 synthesis (v14 README §5, SYSTEM.md "Motion inventory") */
/* Every value below is the approved prototype's. Reduced motion resolves each to its final state at once (MotionPrefs). */

/** The camera's curve between places: slow out, long glide in. */
export const EASE_FLIGHT = [0.45, 0, 0.15, 1] as const;
/** A camera flight lasts 0.5 to 1.15 s, longer for a longer hop (px of travel at the smaller zoom). */
export const flightSeconds = (distPx: number) => Math.min(1.15, 0.5 + distPx / 3200);
/** The new ticket springs from its pin at this fraction of the flight. */
export const FLIGHT_NEAR = 0.62;
/** The old ticket shrinks toward its anchor before the camera leaves (opaque until its last frames). */
export const ticketOut: Transition = { duration: 0.17, ease: [0.4, 0, 1, 1] };
/** The new ticket grows from its pin: a spring from scale 0.86, opaque from its first frames (a 70 ms opacity snap). */
export const ticketIn: Transition = { type: "spring", stiffness: 340, damping: 26 };
export const TICKET_FROM = 0.86;
export const ticketOpacity: Transition = { duration: 0.07, ease: "linear" };
/** A dock springs up 70 ms after its ticket. */
export const dockIn: Transition = { type: "spring", stiffness: 260, damping: 24, delay: 0.07 };
/** The nav's dark pill slides to the section. */
export const navPill: Transition = { type: "spring", stiffness: 420, damping: 34 };
/** A press sinks; the spring brings it back. */
export const pressSink = { whileTap: { scale: 0.97 }, transition: { type: "spring", stiffness: 700, damping: 18 } } as const;
/** Follow → Following: paper sweeps across, then a tick draws in the stub. */
export const sweep: Transition = { duration: 0.26, ease: [0.4, 0, 0.2, 1] };
export const tickDraw: Transition = { duration: 0.34, ease: [0.3, 0, 0.2, 1] };
/** A map layer fading in or out. */
export const layerIn: Transition = { duration: 0.32, ease: [0.3, 0, 0.2, 1] };
export const layerOut: Transition = { duration: 0.22, ease: [0.3, 0, 0.2, 1] };
/** A panel or record fading and sliding in (a ticket's tab, a record opening). */
export const panelIn: Transition = { duration: 0.22, ease: [0.2, 0, 0, 1] };
/** Today: the dates line draws; the sea bars rise; zone limits draw from the shore out; a buoy pulses once; the port ring expands once. */
export const datesDraw: Transition = { duration: 0.7, ease: [0.3, 0, 0.2, 1] };
export const seaRise: Transition = { duration: 0.62, ease: [0.2, 0.9, 0.3, 1] };
export const zoneDraw: Transition = { duration: 0.8, ease: [0.3, 0, 0.2, 1] };
export const buoyPulse: Transition = { duration: 1.2, ease: [0.2, 0.7, 0.3, 1], delay: 0.5 };
export const portRing: Transition = { duration: 2.6, ease: [0.2, 0.7, 0.3, 1], delay: 0.35 };
/** Trip: the waterfall cascades gross to net, 70 ms apart, and tweens on every input. */
export const cascade = (i: number): Transition => ({ duration: 0.42, ease: [0.2, 0.9, 0.3, 1], delay: i * 0.07 });
export const cascadeTween: Transition = { duration: 0.3, ease: [0.2, 0.9, 0.3, 1] };
/** A figure rolling per digit in clipped columns, then settling to its plain string. */
export const roll: Transition = { duration: 0.26, ease: [0.2, 0.9, 0.25, 1] };
export const rollSlow: Transition = { duration: 0.36, ease: [0.2, 0.9, 0.25, 1] };
/** What-if dots sliding; a changed field popping. */
export const whatIfSlide: Transition = { duration: 0.36, ease: [0.2, 0.9, 0.3, 1] };
export const fieldPop: Transition = { type: "spring", stiffness: 500, damping: 22 };
/** Research: the measured line draws left to right; the crosshair glides between samples; the brush window eases. */
export const lineDraw: Transition = { duration: 1.0, ease: [0.3, 0, 0.2, 1] };
export const crosshairGlide: Transition = { type: "spring", stiffness: 620, damping: 44 };
export const brushEase: Transition = { duration: 0.55, ease: [0.2, 0.9, 0.3, 1] };
/** Alerts: "now" slides with a small overshoot, then "+14 days" appears. */
export const alertSlide: Transition = { duration: 0.8, ease: [0.34, 1.25, 0.64, 1], delay: 0.18 };
/** A zoom button's step. */
export const zoomStep: Transition = { duration: 0.28, ease: [0.2, 0.8, 0.2, 1] };
/** The ambient deep-water texture's speed (Paper Shaders `water`), 0 under reduced motion. */
export const DEEP_SPEED = 0.22;
