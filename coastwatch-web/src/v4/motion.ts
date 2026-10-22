"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Neritic's motion vocabulary in script (the CSS side is the --nx-t-* and --nx-ease-* tokens in
 * theme.css). Motion is small, short and interruptible; data values never animate (no count-ups, no
 * interpolation between days); reduced motion keeps fades and drops movement (MotionConfig "user"
 * at the root, and --nx-shift: 0 in CSS).
 */

/** The one spring, for what settles under a finger: the time pill (and, in CSS, the sheet's drag release). */
export const SPRING = { type: "spring", visualDuration: 0.32, bounce: 0.08 } as const;

/** Durations in seconds for motion's `transition` (mirrors theme.css). */
export const T = { press: 0.09, hover: 0.15, pop: 0.18, popExit: 0.12, panel: 0.24, panelExit: 0.16, layer: 0.22, page: 0.2 } as const;
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

/**
 * A legend's swap: when `key` changes (a new layer, not a new day), the element's content fades in
 * over 120 ms. Opacity only, and instant under reduced motion; the words and ramp are simply the new
 * ones, nothing is tweened between the two layers.
 */
export function useSwapFade<E extends HTMLElement>(key: string) {
  const ref = useRef<E>(null);
  const last = useRef(key);
  useEffect(() => {
    if (last.current === key) return;
    last.current = key;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    ref.current?.animate?.([{ opacity: 0 }, { opacity: 1 }], { duration: 120, easing: "cubic-bezier(0.23, 1, 0.32, 1)" });
  }, [key]);
  return ref;
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
