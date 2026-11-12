"use client";

import { MotionGlobalConfig } from "motion/react";
import { useLayoutEffect } from "react";

/**
 * Reduced motion means final states (README §8): with the system set to reduce motion, every Motion animation in
 * the app (registers, settles, ink, the phone sheet's spring, layout travel) completes at once. Set before the
 * first paint from this module and kept in step with the setting. The chart's camera (MapLibre) and the margin's
 * view transition honour the same setting on their own (ChartHost, shell.css).
 */
const QUERY = "(prefers-reduced-motion: reduce)";
function apply() {
  try {
    MotionGlobalConfig.skipAnimations = window.matchMedia(QUERY).matches;
  } catch {
    // no matchMedia: motion as designed
  }
}
if (typeof window !== "undefined") apply();

export function MotionPrefs() {
  useLayoutEffect(() => {
    apply();
    const mq = window.matchMedia(QUERY);
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);
  return null;
}
