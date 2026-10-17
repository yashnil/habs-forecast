"use client";

import { useEffect, useRef, useState } from "react";

/** The element's width in CSS pixels, so a chart draws at real size (its type is real type, never scaled). */
export function useWidth<T extends HTMLElement>(initial = 360, min = 240) {
  const ref = useRef<T | null>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setW(Math.max(min, Math.round(el.getBoundingClientRect().width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, [min]);
  return [ref, w] as const;
}
