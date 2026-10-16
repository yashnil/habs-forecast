"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { motion, useReducedMotion } from "motion/react";

/**
 * The phone's map: a bar docked above the tab bar (part of the layout, never floating over the
 * briefing's text) that opens the map as a sheet over the briefing. The map is loaded on the first
 * open only, then kept, so a slow connection does not pay for it before it is wanted. The page
 * leaves room for the bar (pb on the briefing column), so nothing is covered.
 */
export const MAP_BAR_H = 52;

export function MapSheet({ open, onOpen, onClose, label, summary, children }: { open: boolean; onOpen: () => void; onClose: () => void; label: string; /** what the map is showing, one short phrase */ summary?: string | null; children: React.ReactNode }) {
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
      {!open && (
        <div className="fixed inset-x-0 bottom-[var(--cw-tabbar-h)] z-40 border-t border-hairline bg-surface/95 backdrop-blur lg:hidden" data-testid="coast-map-bar">
          <button type="button" onClick={onOpen} className="cw-press flex w-full items-center justify-between gap-3 px-4 text-left" style={{ minHeight: MAP_BAR_H }} data-testid="coast-map-open">
            <span className="flex min-w-0 items-center gap-2.5">
              <svg viewBox="0 0 20 20" className="h-5 w-5 shrink-0 text-accent" aria-hidden>
                <path d="M2 5l5-2 6 2 5-2v12l-5 2-6-2-5 2z M7 3v12 M13 5v12" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
              </svg>
              <span className="min-w-0">
                <span className="block text-[15px] font-semibold text-ink">Map</span>
                {summary && <span className="block truncate text-[13px] text-ink-3">{summary}</span>}
              </span>
            </span>
            <span aria-hidden className="shrink-0 whitespace-nowrap text-[13px] font-medium text-ink-3">Open ▴</span>
          </button>
        </div>
      )}
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
