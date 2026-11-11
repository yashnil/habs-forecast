"use client";

import { animate, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { settle } from "@/nx/ds/motion";

/**
 * The window as drawn. A window set by a preset, a key or a link settles there with the `settle` spring (the
 * time axis moves like a camera; no value is ever tweened, T26); a window being dragged follows the hand
 * exactly. Reduced motion: the final window at once.
 */
export function useSettledWindow(target: [number, number], immediate: boolean): [number, number] {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState<[number, number]>(target);
  const cur = useRef<[number, number]>(target);
  const [a, b] = target;
  useEffect(() => {
    const from = cur.current;
    if (immediate || reduce || (Math.abs(from[0] - a) < 1 && Math.abs(from[1] - b) < 1)) {
      cur.current = [a, b];
      setShown([a, b]);
      return;
    }
    const c = animate(0, 1, {
      ...settle,
      onUpdate: (k) => {
        const w: [number, number] = [from[0] + (a - from[0]) * k, from[1] + (b - from[1]) * k];
        cur.current = w;
        setShown(w);
      },
    });
    return () => c.stop();
  }, [a, b, immediate, reduce]);
  return shown;
}
