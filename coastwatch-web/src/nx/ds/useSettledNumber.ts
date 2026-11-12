"use client";
import { animate, useMotionValue, useReducedMotion, useTransform, type MotionValue } from "motion/react";
import { useEffect, useRef } from "react";
import { value as valuePreset } from "./motion";

/**
 * A number that moves from its old value to its new one with the `value` preset (v13 README §5). The first
 * value is drawn as it is: it never counts up from zero on arrival. A figure that is not negative before and
 * after a change is never negative on the way (and likewise never positive between two negatives). Reduced
 * motion: the new value at once.
 */
export function useSettledNumber(v: number): MotionValue<number> {
  const reduce = useReducedMotion();
  const mv = useMotionValue(v);
  const bounds = useRef<[number, number]>([v, v]);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      mv.set(v);
      return;
    }
    const from = mv.get();
    bounds.current = [Math.min(from, v), Math.max(from, v)];
    if (reduce || !Number.isFinite(v) || !Number.isFinite(from)) {
      mv.set(v);
      return;
    }
    const c = animate(mv, v, valuePreset);
    return () => c.stop();
  }, [v, reduce, mv]);
  // held between where it was and where it is going: no overshoot past either end
  return useTransform(mv, (x) => Math.min(bounds.current[1], Math.max(bounds.current[0], x)));
}
