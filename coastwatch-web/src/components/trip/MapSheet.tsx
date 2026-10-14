"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";

/**
 * The phone's map: a button above the tab bar that opens the map as a sheet over the briefing (the
 * briefing stays one scroll of text; the map is one tap away and follows whatever the reader chose).
 * The map is loaded on the first open only, then kept, so a slow connection does not pay for it
 * before it is wanted.
 */
export function MapSheet({ open, onOpen, onClose, label, children }: { open: boolean; onOpen: () => void; onClose: () => void; label: string; children: React.ReactNode }) {
  const reduce = useReducedMotion();
  const [mounted, setMounted] = useState(false);
  // fixed to the screen, so outside the page's animated (transformed) wrapper
  const [host, setHost] = useState<HTMLElement | null>(null);
  useEffect(() => setHost(document.body), []);
  useEffect(() => {
    if (open) setMounted(true);
  }, [open]);
  useEffect(() => {
    if (!open) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [open, onClose]);
  if (!host) return null;
  return createPortal(
    <>
      <AnimatePresence>
        {!open && (
          <motion.button
            type="button"
            onClick={onOpen}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 12 }}
            className="cw-press fixed bottom-[calc(76px+env(safe-area-inset-bottom))] right-4 z-40 lg:hidden flex min-h-[52px] items-center gap-2 rounded-full bg-navy-900 px-5 text-[15px] font-semibold text-white shadow-[0_10px_30px_rgba(6,17,30,0.35)]"
            data-testid="coast-map-open"
          >
            <svg viewBox="0 0 20 20" className="h-5 w-5" aria-hidden>
              <path d="M2 5l5-2 6 2 5-2v12l-5 2-6-2-5 2z M7 3v12 M13 5v12" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
            </svg>
            Map
          </motion.button>
        )}
      </AnimatePresence>
      <motion.div
        role="dialog"
        aria-modal={open}
        aria-label={label}
        aria-hidden={!open}
        initial={false}
        animate={open ? { y: 0, opacity: 1 } : { y: reduce ? 0 : "100%", opacity: reduce ? 0 : 1 }}
        transition={{ type: "spring", stiffness: 380, damping: 40 }}
        style={{ pointerEvents: open ? "auto" : "none" }}
        className="fixed inset-x-0 bottom-0 top-[56px] z-[55] lg:hidden overflow-hidden rounded-t-[22px] bg-[#0b1d33] shadow-[0_-12px_40px_rgba(6,17,30,0.4)]"
        data-testid="coast-map-mobile"
        data-open={open}
        {...(!open ? { inert: true } : {})}
      >
        {mounted && <div className="absolute inset-0">{children}</div>}
        <button
          type="button"
          onClick={onClose}
          className="cw-press absolute right-3 top-3 z-20 flex min-h-[44px] items-center gap-1.5 rounded-full bg-surface px-4 text-[15px] font-semibold text-ink shadow-[0_6px_18px_rgba(6,17,30,0.3)]"
          data-testid="coast-map-close"
        >
          Briefing ↓
        </button>
      </motion.div>
    </>,
    host,
  );
}
