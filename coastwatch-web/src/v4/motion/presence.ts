"use client";

import { useEffect, useState } from "react";

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
