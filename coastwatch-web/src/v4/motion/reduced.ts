"use client";

import { useSyncExternalStore } from "react";

/** True when the reader asked for less motion (read at call time; false on the server). */
export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

function media(query: string, server: boolean) {
  const subscribe = (f: () => void) => {
    if (typeof window === "undefined" || !window.matchMedia) return () => {};
    const m = window.matchMedia(query);
    m.addEventListener?.("change", f);
    return () => m.removeEventListener?.("change", f);
  };
  const get = () => (typeof window !== "undefined" && window.matchMedia ? window.matchMedia(query).matches : server);
  return { subscribe, get, server: () => server };
}

const RM = media("(prefers-reduced-motion: reduce)", false);
const HOVER = media("(hover: hover)", false);

/** Reduced motion as live state. The server (and the first client render) say false: decide movement in effects, not markup. */
export function useReducedMotion(): boolean {
  return useSyncExternalStore(RM.subscribe, RM.get, RM.server);
}

/** A device with a real hover (a mouse). False on the server and on touch devices. */
export function canHover(): boolean {
  return typeof window !== "undefined" && !!window.matchMedia?.("(hover: hover)").matches;
}

export function useCanHover(): boolean {
  return useSyncExternalStore(HOVER.subscribe, HOVER.get, HOVER.server);
}

/**
 * Whether the "hide until first seen" primitives may hide anything (Reveal, DrawOn). Automated browsers keep
 * content as served, as the launch screen does: a full-page capture and axe must see every block. A test
 * that wants the real thing sets localStorage `nx-motion-force-v1` to "1". Storage that throws counts as unset.
 */
export const MOTION_FORCE_KEY = "nx-motion-force-v1";
export function maySetAside(): boolean {
  if (typeof window === "undefined" || typeof IntersectionObserver === "undefined") return false;
  if (!navigator.webdriver) return true;
  try {
    return window.localStorage.getItem(MOTION_FORCE_KEY) === "1";
  } catch {
    return false;
  }
}
