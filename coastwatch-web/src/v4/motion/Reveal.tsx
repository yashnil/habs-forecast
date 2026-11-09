"use client";

import { useLayoutEffect, useRef, type ElementType, type HTMLAttributes, type ReactNode } from "react";
import { maySetAside, prefersReducedMotion } from "./reduced";

const REVEAL_MS = 480;
const STAGGER_MS = 50;
const MAX_STAGGER = 5;

/**
 * True when a block may be set aside for a reveal: only if it lies entirely below the fold at hydration, so
 * what the reader can already see never fades, and the server HTML is what shows without script.
 */
export function belowFold(top: number, viewportHeight: number): boolean {
  return top >= viewportHeight - 1;
}

/**
 * A chapter-level block fades and rises 12 px the first time it enters view.
 *
 * Use for: a chapter or a major section below the fold. `group` staggers its direct children, 50 ms apart,
 * the first five only (a run of three figures, say).
 * Do not use for: anything above the fold, individual list rows, anything that changes often.
 *
 * It renders plain visible markup; only after hydration does it hide a block that is below the fold. Opacity
 * alone never takes it from assistive tech, and the attribute is removed when the animation ends. Reduced
 * motion keeps the fade (and drops the rise).
 */
export function Reveal<T extends ElementType = "div">({ as, group = false, children, ...rest }: { as?: T; group?: boolean; children: ReactNode } & HTMLAttributes<HTMLElement>) {
  const Tag = (as ?? "div") as ElementType;
  const ref = useRef<HTMLElement | null>(null);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || !maySetAside() || !belowFold(el.getBoundingClientRect().top, window.innerHeight)) return;
    el.dataset.reveal = "wait";
    let t = 0;
    const io = new IntersectionObserver(
      (es) => {
        if (!es.some((e) => e.isIntersecting)) return;
        io.disconnect();
        el.dataset.reveal = "in";
        const n = group ? Math.min(el.children.length, MAX_STAGGER) - 1 : 0;
        t = window.setTimeout(() => delete el.dataset.reveal, (prefersReducedMotion() ? 200 : REVEAL_MS) + Math.max(0, n) * STAGGER_MS + 60);
      },
      { rootMargin: "0px 0px -8% 0px" },
    );
    io.observe(el);
    return () => {
      io.disconnect();
      window.clearTimeout(t);
      delete el.dataset.reveal;
    };
  }, [group]);
  return (
    <Tag ref={ref} data-reveal-group={group ? "" : undefined} {...rest}>
      {children}
    </Tag>
  );
}
