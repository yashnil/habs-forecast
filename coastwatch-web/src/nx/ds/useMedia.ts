"use client";
import { useEffect, useState, useSyncExternalStore } from "react";

/** A media query as state; false on the server and before hydration. */
export function useMedia(query: string): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(query);
      m.addEventListener("change", cb);
      return () => m.removeEventListener("change", cb);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

/** The phone layout (spec section 6): the margin becomes a sheet, the destinations a bottom bar. */
export const PHONE_QUERY = "(max-width: 720px)";
export const usePhone = () => useMedia(PHONE_QUERY);

/** An element's content width, kept current with a ResizeObserver. `fallback` until measured. */
export function useElementWidth<T extends HTMLElement>(fallback: number): [(el: T | null) => void, number] {
  const [width, setWidth] = useState(fallback);
  const [el, setEl] = useState<T | null>(null);
  useEffect(() => {
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.round(e.contentRect.width)));
    ro.observe(el);
    setWidth(Math.round(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, [el]);
  return [setEl, width];
}
