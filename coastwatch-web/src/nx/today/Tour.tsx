"use client";

import { motion } from "motion/react";
import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Button } from "@/nx/ds/Button";
import { Icon } from "@/nx/ds/Icon";
import { register, registerFrom, registerTo } from "@/nx/ds/motion";
import { dismissTour } from "./store";
import s from "./today.module.css";

/**
 * "Show me around" (journeys.md, Help without a tutorial): optional, four steps, one sentence each, in order
 * the status, a sounding, the dated list and the provenance stamp. A ruled frame marks each in place; no
 * scrim, no arrows. Ending it, or "Don't show again", dismisses it for good on this device.
 */
const STEPS: Array<{ target: string; text: string }> = [
  { target: "status", text: "The season status for your fishery, in the agency's own words, with the day Neritic last checked it." },
  { target: "sounding", text: "Readings sit where they were measured: the latest crab test at its test area, the buoy out on the water." },
  { target: "agenda", text: "The dates, in order: what has happened sits above today's line, what is coming sits below it." },
  { target: "stamp", text: "Every value carries a line like this. Tap it for the source, the date and what to read it with." },
];

export function TourLink({ onStart }: { onStart: () => void }) {
  return (
    <p className={s.tourLink}>
      <Button variant="text" onClick={onStart}>
        Show me around
      </Button>
      <button type="button" className={s.tourClose} onClick={dismissTour} aria-label="Hide Show me around for good">
        <Icon name="close" size={20} />
      </button>
    </p>
  );
}

type Box = { x: number; y: number; w: number; h: number };

export function Tour({ hasSounding, onEnd }: { hasSounding: boolean; onEnd: () => void }) {
  const steps = STEPS.filter((st) => hasSounding || st.target !== "sounding");
  const [i, setI] = useState(0);
  const [box, setBox] = useState<Box | null>(null);
  const note = useRef<HTMLDivElement>(null);
  const id = useId();
  const step = steps[i];
  const end = useCallback(() => {
    dismissTour();
    onEnd();
  }, [onEnd]);

  const measure = useCallback(() => {
    const el = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`);
    if (!el) return setBox(null);
    const r = el.getBoundingClientRect();
    setBox({ x: r.left - 6, y: r.top - 6, w: r.width + 12, h: r.height + 12 });
  }, [step.target]);

  useLayoutEffect(() => {
    document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`)?.scrollIntoView({ block: "nearest" });
    measure();
  }, [measure, step.target]);
  useEffect(() => {
    note.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && end();
    window.addEventListener("resize", measure);
    window.addEventListener("keydown", onKey);
    const t = window.setInterval(measure, 400);
    return () => {
      window.removeEventListener("resize", measure);
      window.removeEventListener("keydown", onKey);
      window.clearInterval(t);
    };
  }, [measure, end, i]);

  if (typeof document === "undefined") return null;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const noteW = Math.min(320, vw - 32);
  // beside the target where there is room, else below it, else above it
  let left = box ? box.x + box.w + 12 : 16;
  let top = box ? box.y : 80;
  if (box && left + noteW > vw - 16) {
    left = Math.max(16, Math.min(box.x, vw - noteW - 16));
    top = box.y + box.h + 12;
    if (top + 180 > vh) top = Math.max(60, box.y - 192);
  }
  return createPortal(
    <div className={s.tour} data-testid="nx-tour">
      {box ? <div className={s.tourFrame} style={{ left: box.x, top: box.y, width: box.w, height: box.h }} aria-hidden="true" /> : null}
      <motion.div
        key={i}
        ref={note}
        className={s.tourNote}
        style={{ left, top, width: noteW }}
        role="dialog"
        aria-modal="false"
        aria-labelledby={`${id}-n`}
        tabIndex={-1}
        initial={registerFrom}
        animate={registerTo}
        transition={register}
      >
        <p className={s.tourCount} id={`${id}-n`}>
          {i + 1} of {steps.length}
        </p>
        <p className={s.tourText}>{step.text}</p>
        <div className={s.tourActions}>
          {i < steps.length - 1 ? (
            <Button variant="secondary" onClick={() => setI(i + 1)}>
              Next
            </Button>
          ) : (
            <Button variant="secondary" onClick={end}>
              Done
            </Button>
          )}
          <button type="button" className={s.tourHide} onClick={end}>
            Don&rsquo;t show again
          </button>
        </div>
      </motion.div>
    </div>,
    document.querySelector("[data-testid=nx-app]") ?? document.body,
  );
}
