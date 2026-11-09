"use client";

import { useEffect, useRef, useState, type ElementType, type HTMLAttributes, type ReactNode, type RefObject } from "react";
import { maySetAside } from "./reduced";

export type DrawPhase = "idle" | "wait" | "draw" | "done";

/**
 * A chart draws itself once, the first time it comes into view. A chart already on the screen when the page
 * opens is left as drawn; a chart never redraws on a new window, a hover or new data. Values and labels are
 * never tweened: only the marks are revealed.
 * Phases: "idle" (as served), "wait" (held back until seen), "draw" (playing), "done".
 *
 * SeriesChart keeps its .nx-chart-* classes (500 ms) on this hook; new charts use DrawOn with the
 * .nx-draw-* classes below (--nx-t-draw, 700 ms).
 */
export function useDrawOnce(ref: RefObject<Element | null>, { ms = 700, threshold = 0.25 }: { ms?: number; threshold?: number } = {}): DrawPhase {
  const [phase, setPhase] = useState<DrawPhase>("idle");
  useEffect(() => {
    const el = ref.current;
    if (!el || !maySetAside()) return;
    const r = el.getBoundingClientRect();
    if (r.width > 0 && r.top < window.innerHeight && r.bottom > 0) return;
    let t = 0;
    const io = new IntersectionObserver(
      (es) => {
        if (!es.some((e) => e.isIntersecting)) return setPhase("wait");
        io.disconnect();
        setPhase("draw");
        t = window.setTimeout(() => setPhase("done"), ms);
      },
      { threshold },
    );
    io.observe(el);
    return () => {
      io.disconnect();
      window.clearTimeout(t);
    };
    // once per chart
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return phase;
}

/**
 * The scope of a once-only draw. Mark its parts with classes (`.nx-draw-line` needs `pathLength={1}`):
 * `.nx-draw-line` a solid line, left to right; `.nx-draw-wipe` a dashed or broken line, wiped left to right;
 * `.nx-draw-area` and `.nx-draw-bar` grow from their baseline; `.nx-draw-marks` fade in after the line.
 * Use for one chart's first appearance. Not for sparkline rows, or anything that redraws on input.
 */
export function DrawOn<T extends ElementType = "div">({ as, children, ms, threshold, ...rest }: { as?: T; children: ReactNode; ms?: number; threshold?: number } & HTMLAttributes<HTMLElement>) {
  const Tag = (as ?? "div") as ElementType;
  const ref = useRef<HTMLElement | null>(null);
  const phase = useDrawOnce(ref, { ms, threshold });
  return (
    <Tag ref={ref} data-draw={phase === "idle" ? undefined : phase} {...rest}>
      {children}
    </Tag>
  );
}
