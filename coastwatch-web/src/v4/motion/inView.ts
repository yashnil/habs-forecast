"use client";

import { useEffect, useState, type RefObject } from "react";

/**
 * Whether the element has entered the viewport. `once` (default) latches true the first time it does.
 * Use it to start something the first time a block is seen, or (`once: false`) to pause something while
 * it is off screen. False during server render; true where IntersectionObserver is missing.
 */
export function useInView(ref: RefObject<Element | null>, { once = true, rootMargin = "0px", threshold = 0 }: { once?: boolean; rootMargin?: string; threshold?: number } = {}): boolean {
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (typeof IntersectionObserver === "undefined") return setSeen(true);
    const io = new IntersectionObserver(
      (es) => {
        const hit = es.some((e) => e.isIntersecting);
        setSeen(hit);
        if (hit && once) io.disconnect();
      },
      { rootMargin, threshold },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [ref, once, rootMargin, threshold]);
  return seen;
}
