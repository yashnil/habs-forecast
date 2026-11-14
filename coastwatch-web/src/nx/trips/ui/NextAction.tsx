"use client";

import { motion, useReducedMotion } from "motion/react";
import { useEffect, useRef } from "react";
import { Button } from "@/nx/ds";
import type { NextAction as Next } from "../lifecycle";
import s from "./trips.module.css";

/**
 * The trip's one filled button, and one short line saying why it is the next thing. When the action changes (a
 * recheck lands, the trip is planned) the button swaps in place; under reduced motion it changes at once.
 */
export function NextAction({ next, onAct, busy, pinned }: { next: Next | null; onAct: (n: Next) => void; busy?: boolean; pinned?: boolean }) {
  const reduce = useReducedMotion();
  // pinned at the sheet's foot, the bar would sit over whatever the sheet scrolls to: its measured height becomes the
  // sheet's scroll padding (and the pinned ticket's bottom padding), so focus and "scroll into view" stop above it
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!pinned || !el || typeof ResizeObserver === "undefined") return;
    const sc = el.closest<HTMLElement>(".n12-margin-scroll") ?? (document.scrollingElement as HTMLElement | null);
    const set = () => {
      const h = Math.ceil(el.getBoundingClientRect().height);
      sc?.style.setProperty("scroll-padding-bottom", `${h + 8}px`);
      document.documentElement.style.setProperty("--trip-pin-h", `${h}px`);
    };
    set();
    const ro = new ResizeObserver(set);
    ro.observe(el);
    return () => {
      ro.disconnect();
      sc?.style.removeProperty("scroll-padding-bottom");
      document.documentElement.style.removeProperty("--trip-pin-h");
    };
  }, [pinned]);
  return (
    <div ref={ref} className={s.next} data-pinned={pinned || undefined} data-testid="trip-next-action" data-action={next?.id ?? "none"}>
        {next ? (
          <motion.div key={next.id} className={s.nextInner} initial={reduce ? false : { y: 3 }} animate={{ y: 0 }} transition={{ duration: 0.16, ease: [0.2, 0, 0, 1] }}>
            <Button variant="primary" full onClick={() => onAct(next)} busy={busy} busyLabel="Rechecking the sources" data-testid="trip-next">
              {next.label}
            </Button>
            <p className={s.why}>{next.why}</p>
          </motion.div>
        ) : null}
    </div>
  );
}
