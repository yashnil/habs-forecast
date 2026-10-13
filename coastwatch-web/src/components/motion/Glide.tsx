"use client";

import { useLayoutEffect, useRef, useState } from "react";
import { motion } from "motion/react";

/**
 * An active-item indicator that glides from where it was on the previous page to the current item.
 * Pages remount the shell, so the last position is kept in sessionStorage (per indicator) instead
 * of in React state. Reduced motion: it simply appears (MotionConfig in the shell).
 */
export function Glide({ id, container, activeSelector, className }: { id: string; container: React.RefObject<HTMLElement | null>; activeSelector: string; className: string }) {
  const [box, setBox] = useState<{ x: number; w: number } | null>(null);
  const from = useRef<{ x: number; w: number } | null>(null);
  useLayoutEffect(() => {
    const c = container.current;
    const a = c?.querySelector<HTMLElement>(activeSelector);
    if (!c || !a) return setBox(null);
    const measure = () => {
      const cr = c.getBoundingClientRect();
      const ar = a.getBoundingClientRect();
      return { x: ar.left - cr.left, w: ar.width };
    };
    try {
      const prev = sessionStorage.getItem(`cw-glide-${id}`);
      from.current = prev ? JSON.parse(prev) : null;
    } catch {
      from.current = null;
    }
    const b = measure();
    setBox(b);
    try {
      sessionStorage.setItem(`cw-glide-${id}`, JSON.stringify(b));
    } catch {
      // not kept
    }
    const ro = new ResizeObserver(() => {
      const m = measure();
      setBox(m);
      try {
        sessionStorage.setItem(`cw-glide-${id}`, JSON.stringify(m));
      } catch {
        // not kept
      }
    });
    ro.observe(c);
    return () => ro.disconnect();
  }, [id, container, activeSelector]);
  if (!box) return null;
  const f = from.current ?? box;
  return (
    <motion.span
      aria-hidden
      className={className}
      initial={{ x: f.x, width: f.w }}
      animate={{ x: box.x, width: box.w }}
      transition={{ type: "spring", stiffness: 380, damping: 34 }}
      style={{ left: 0 }}
    />
  );
}
