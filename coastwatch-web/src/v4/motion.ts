"use client";

import { useEffect, useState } from "react";

/**
 * Neritic's motion vocabulary in script (the CSS side is the --nx-t-* and --nx-ease-* tokens in
 * theme.css). Motion is small, short and interruptible; data values never animate (no count-ups, no
 * interpolation between days); reduced motion keeps fades and drops movement (MotionConfig "user"
 * at the root, and --nx-shift: 0 in CSS).
 */

/** The one spring: panels, the layer deck morph, the time pill. A whisper of bounce, never a wobble. */
export const SPRING = { type: "spring", visualDuration: 0.32, bounce: 0.08 } as const;

/** Durations in seconds for motion's `transition` (mirrors theme.css). */
export const T = { press: 0.11, hover: 0.15, pop: 0.18, popExit: 0.12, panel: 0.24, panelExit: 0.16, page: 0.26 } as const;
export const EASE_OUT = [0.23, 1, 0.32, 1] as const;
export const EASE_IN = [0.4, 0, 1, 1] as const;

/**
 * Keeps a popover mounted for its exit. `mounted` stays true for `exitMs` after `open` turns false;
 * `state` is "closed" meanwhile, so the element can play its exit (CSS `[data-state=closed]`) and be
 * made inert and aria-hidden, and drop its test id: an exiting copy never answers for the open one.
 * Reopening during the exit simply turns it back (interruptible).
 */
export function usePresence(open: boolean, exitMs = 120): { mounted: boolean; state: "open" | "closed" } {
  const [mounted, setMounted] = useState(open);
  // adjust state during render when `open` turns true (the React-recommended pattern, no effect needed)
  if (open && !mounted) setMounted(true);
  useEffect(() => {
    if (open) return;
    const t = window.setTimeout(() => setMounted(false), exitMs);
    return () => window.clearTimeout(t);
  }, [open, exitMs]);
  return { mounted: mounted || open, state: open ? "open" : "closed" };
}

/** Props for an element that may be exiting: inert, hidden from assistive tech, no test id. */
export function exitProps(state: "open" | "closed", testid?: string) {
  return state === "open" ? { "data-state": "open", "data-testid": testid } : { "data-state": "closed", inert: true, "aria-hidden": true as const };
}

/* ---- "the app is ready": the launch screen waits for this on Explore (map loaded) ---- */

let ready = false;
const waiting = new Set<() => void>();

/** Explore calls this once its map has loaded (the moment the launch screen can step aside). */
export function markAppReady() {
  if (ready) return;
  ready = true;
  waiting.forEach((f) => f());
  waiting.clear();
}

/** Calls `f` when the app reports ready (now, if it already has). Returns an unsubscribe. */
export function onAppReady(f: () => void): () => void {
  if (ready) {
    f();
    return () => {};
  }
  waiting.add(f);
  return () => waiting.delete(f);
}
