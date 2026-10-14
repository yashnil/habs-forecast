"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useWide } from "@/lib/useWidth";

/**
 * A picker surface: a popover under its trigger on wide screens, a bottom sheet on phones (thumb
 * reach, one hand). Escape, the backdrop or a downward drag close it; focus moves in on open and
 * back to the trigger on close.
 */
export function Sheet({
  open,
  onClose,
  title,
  children,
  testid,
  returnFocus,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  testid?: string;
  returnFocus?: React.RefObject<HTMLElement | null>;
}) {
  const wide = useWide();
  const reduce = useReducedMotion();
  const box = useRef<HTMLDivElement | null>(null);
  const [host, setHost] = useState<HTMLElement | null>(null);
  useEffect(() => setHost(document.body), []);
  useEffect(() => {
    if (!open) return;
    // focus the current choice (or the search), not the close button that comes first
    const t = requestAnimationFrame(() => {
      const b = box.current;
      (b?.querySelector<HTMLElement>("[data-autofocus], input, [aria-checked='true']") ?? b?.querySelector<HTMLElement>("[role='radio'], button"))?.focus();
    });
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    const back = returnFocus?.current;
    return () => {
      cancelAnimationFrame(t);
      window.removeEventListener("keydown", key);
      back?.focus();
    };
  }, [open, onClose, returnFocus]);

  const popover = (
    <AnimatePresence>
      {open && wide && <div key="away" className="fixed inset-0 z-30" onClick={onClose} aria-hidden />}
      {open && wide && (
        <motion.div
          key="popover"
          ref={box}
          role="dialog"
          aria-label={title}
          initial={reduce ? { opacity: 0 } : { opacity: 0, y: -6, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -4, transition: { duration: 0.12 } }}
          transition={{ type: "spring", stiffness: 520, damping: 38 }}
          style={{ transformOrigin: "top left" }}
          className="absolute left-0 top-full z-40 mt-2 w-[min(440px,calc(100vw-2rem))] rounded-2xl border border-hairline bg-surface p-4 shadow-[0_18px_48px_rgba(6,17,30,0.22)]"
          data-testid={testid}
        >
          <p className="mb-3 text-[13px] font-medium text-ink-3">{title}</p>
          {children}
        </motion.div>
      )}
    </AnimatePresence>
  );
  // a phone's sheet is fixed to the screen, so it is rendered outside the page's animated (transformed) wrapper
  const sheet =
    host &&
    createPortal(
      <AnimatePresence>
        {open && !wide && (
          <motion.div key="backdrop" className="fixed inset-0 z-[60] bg-[#06111e]/45" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} aria-hidden />
        )}
        {open && !wide && (
          <motion.div
            key="sheet"
            ref={box}
            role="dialog"
            aria-modal="true"
            aria-label={title}
            initial={reduce ? { opacity: 0 } : { y: "100%" }}
            animate={reduce ? { opacity: 1 } : { y: 0 }}
            exit={reduce ? { opacity: 0 } : { y: "100%" }}
            transition={{ type: "spring", stiffness: 420, damping: 40 }}
            drag={reduce ? false : "y"}
            dragConstraints={{ top: 0, bottom: 0 }}
            dragElastic={{ top: 0, bottom: 0.6 }}
            onDragEnd={(_, i) => (i.offset.y > 90 || i.velocity.y > 500) && onClose()}
            className="fixed inset-x-0 bottom-0 z-[61] max-h-[86dvh] overflow-y-auto rounded-t-[22px] bg-surface px-4 pb-[max(20px,env(safe-area-inset-bottom))] pt-2 shadow-[0_-12px_40px_rgba(6,17,30,0.3)]"
            data-testid={testid}
          >
            <div aria-hidden className="mx-auto mb-3 h-1.5 w-10 rounded-full bg-hairline-strong" />
            <div className="mb-3 flex items-center justify-between">
              <p className="font-display text-[20px] font-medium text-ink">{title}</p>
              <button type="button" onClick={onClose} className="grid h-11 w-11 place-items-center rounded-full text-[22px] text-ink-3 hover:bg-surface-3" aria-label="Close">
                ×
              </button>
            </div>
            {children}
          </motion.div>
        )}
      </AnimatePresence>,
      host,
    );
  return (
    <>
      {popover}
      {sheet}
    </>
  );
}
