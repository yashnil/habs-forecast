"use client";

import Link from "next/link";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Icon } from "@/components/ui/Icon";
import { KindChip } from "@/components/ui/Kinds";
import { useOfficialDrawer } from "@/components/shell/OfficialShell";
import { TOUR } from "@/content/copy";

/** Remembered on this device once the introduction or the tour has been seen (same key as M8). */
export const TOUR_SEEN_KEY = "cw-intro-dismissed-v1";

export type TourActions = {
  /** switch the map to a layer group, through the same control the rail and sheet use */
  showForecast: () => void;
  showSatellite: () => void;
  /** open a port in the inspector; returns false when no port is available */
  showPort: () => boolean;
};

type Step = {
  id: string;
  /** CSS selectors of the real control to highlight; the first visible one wins */
  target?: string[];
  /** what the map does when the step is entered (never anything the user cannot undo) */
  enter?: (a: TourActions) => void;
};

const STEPS: Step[] = [
  { id: "what" },
  { id: "place", target: ['[data-testid="place-chip"]', '[data-testid="place-button"]'] },
  { id: "forecast", target: ['[data-testid="layer-dock"]', '[data-testid="mobile-legend"]'], enter: (a) => a.showForecast() },
  { id: "satellite", target: ['[data-testid="layer-dock"]', '[data-testid="mobile-legend"]'], enter: (a) => a.showSatellite() },
  { id: "gaps", target: ['[data-testid="probability-legend"]', '[data-testid="layer-dock"]', '[data-testid="mobile-legend"]'], enter: (a) => a.showForecast() },
  { id: "port", target: ['[data-testid="detail-panel"]', '[data-testid="mobile-sheet"]'], enter: (a) => void a.showPort() },
  { id: "official", target: ['[data-testid="official-pill"]', '[data-testid="tab-notices"]'] },
  { id: "pages", target: ['[data-testid="nav-bloom"]', '[data-testid="tab-bloom"]'] },
  { id: "limits" },
];

function visible(sel: string[] | undefined): HTMLElement | null {
  if (!sel) return null;
  for (const s of sel) {
    for (const el of document.querySelectorAll<HTMLElement>(s)) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== "hidden") return el;
    }
  }
  return null;
}

type Box = { top: number; left: number; width: number; height: number; header: boolean };
const CARD_W = 344;
const GAP = 12;

/** Where the card goes: beside the highlighted control if there is room, otherwise below or above it; on a phone, docked under the place button. */
function place(target: Box | null, card: { w: number; h: number }, phone: boolean, dockTop: number): { top: number; left: number } {
  const vw = window.innerWidth, vh = window.innerHeight;
  if (phone || !target) {
    if (phone) return { top: dockTop, left: 12 };
    return { top: Math.max(dockTop, Math.min(vh - card.h - 16, 120)), left: Math.max(16, Math.min(vw - card.w - 16, 96)) };
  }
  const clampTop = (t: number) => Math.max(8, Math.min(vh - card.h - 8, t));
  const clampLeft = (l: number) => Math.max(8, Math.min(vw - card.w - 8, l));
  // a control in the masthead: the card goes below it, never over the masthead or the status line
  if (target.header) return { top: Math.max(dockTop - 40, target.top + target.height + GAP), left: clampLeft(target.left + target.width / 2 - card.w / 2) };
  if (target.left + target.width + GAP + card.w <= vw - 8) return { top: clampTop(target.top), left: target.left + target.width + GAP };
  if (target.left - GAP - card.w >= 8) return { top: clampTop(target.top), left: target.left - GAP - card.w };
  if (target.top + target.height + GAP + card.h <= vh - 8) return { top: target.top + target.height + GAP, left: clampLeft(target.left) };
  return { top: clampTop(target.top - GAP - card.h), left: clampLeft(target.left) };
}

/**
 * The guided tour of the Ocean Map: nine short steps that highlight the real controls and, where
 * it helps, use them (switch the layer, open a port). It is not modal: the page stays usable, the
 * official notices stay one tap away above it, focus is never trapped, and Escape or the X closes
 * it at any step. Opened from the first-visit card, the "Tour" control or a link with ?tour=1.
 */
export function MapTour({ open, onClose, actions, phone, reducedMotion, dockTop }: { open: boolean; onClose: () => void; actions: TourActions; phone: boolean; reducedMotion: boolean; dockTop: number }) {
  const [i, setI] = useState(0);
  const [box, setBox] = useState<Box | null>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const card = useRef<HTMLDivElement>(null);
  const heading = useRef<HTMLHeadingElement>(null);
  const opener = useRef<Element | null>(null);
  const { openDrawer } = useOfficialDrawer();
  const step = STEPS[i];
  const text = TOUR.steps[step.id as keyof typeof TOUR.steps];

  // start at the first step each time it opens; remember the opener to return focus to it
  useEffect(() => {
    if (!open) return;
    opener.current = document.activeElement;
    setI(0);
  }, [open]);

  // entering a step may use a real control
  useEffect(() => {
    if (open) step.enter?.(actions);
    // actions are stable callbacks from the map; re-running on their identity would loop
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, i]);

  // follow the highlighted control while the layout settles (panels slide in, the map refits)
  const measure = useCallback(() => {
    const el = visible(step.target);
    const r = el?.getBoundingClientRect();
    const b = r && el ? { top: r.top, left: r.left, width: r.width, height: r.height, header: !!el.closest("header") } : null;
    setBox((o) => (o && b && o.top === b.top && o.left === b.left && o.width === b.width && o.height === b.height ? o : b));
    const c = card.current;
    if (c) setPos(place(b, { w: c.offsetWidth || CARD_W, h: c.offsetHeight || 220 }, phone, dockTop));
  }, [step, phone, dockTop]);
  useLayoutEffect(() => {
    if (!open) return;
    measure();
    const t = window.setInterval(measure, 250);
    window.addEventListener("resize", measure);
    return () => {
      window.clearInterval(t);
      window.removeEventListener("resize", measure);
    };
  }, [open, measure]);

  // the heading takes focus on every step, so a screen reader reads the new step from the start
  // (once the card is placed: a hidden element cannot take focus)
  const placed = pos != null;
  useEffect(() => {
    if (open && placed) heading.current?.focus({ preventScroll: true });
  }, [open, i, placed]);

  const close = useCallback(() => {
    try {
      window.localStorage.setItem(TOUR_SEEN_KEY, "1");
    } catch {
      // nothing to remember
    }
    onClose();
    const o = opener.current as HTMLElement | null;
    if (o && document.contains(o) && typeof o.focus === "function") o.focus();
  }, [onClose]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      // the notices drawer handles its own Escape
      if (e.key === "Escape" && !document.querySelector('[data-testid="official-drawer"]')) close();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, close]);

  if (!open || typeof document === "undefined") return null;
  const last = i === STEPS.length - 1;
  const anim = reducedMotion ? "" : "transition-[top,left,width,height] duration-200 ease-out";

  return createPortal(
    <>
      {box && (
        <div
          aria-hidden
          data-testid="tour-highlight"
          className={`pointer-events-none fixed z-[45] rounded-xl border-2 border-accent shadow-[0_0_0_4px_rgba(14,116,144,0.22)] ${anim}`}
          style={{ top: box.top - 4, left: box.left - 4, width: box.width + 8, height: box.height + 8 }}
        />
      )}
      <div
        ref={card}
        role="dialog"
        aria-modal="false"
        aria-labelledby="tour-h"
        aria-describedby="tour-body"
        data-testid="tour"
        data-step={step.id}
        className={`theme-paper fixed z-[45] rounded-xl bg-surface p-4 text-ink shadow-[0_1px_2px_rgba(6,17,30,0.16),0_10px_30px_rgba(6,17,30,0.3)] ${phone ? "right-3" : ""} ${anim}`}
        style={{ top: pos?.top ?? dockTop, left: pos?.left ?? 12, width: phone ? undefined : CARD_W, visibility: pos ? "visible" : "hidden" }}
      >
        <div className="flex items-start justify-between gap-3">
          <p className="text-[13px] font-medium text-ink-3" data-testid="tour-progress">
            Step {i + 1} of {STEPS.length}
          </p>
          <button type="button" onClick={close} aria-label="Close the tour" data-testid="tour-close" className="-mr-2 -mt-2 grid h-10 w-10 shrink-0 place-items-center rounded-md text-ink-3 hover:bg-hairline/50 hover:text-ink">
            <Icon name="close" className="h-5 w-5" />
          </button>
        </div>
        <div className="mt-1 flex gap-1" aria-hidden>
          {STEPS.map((s, k) => (
            <span key={s.id} className={`h-1 flex-1 rounded-full ${k <= i ? "bg-accent" : "bg-hairline-strong"}`} />
          ))}
        </div>
        <h2 id="tour-h" ref={heading} tabIndex={-1} className="mt-3 font-display text-[19px] font-medium leading-snug outline-none">
          {text.title}
        </h2>
        <div id="tour-body" className="mt-1.5 space-y-2 text-[14px] leading-snug text-ink-2">
          {text.body.map((p) => (
            <p key={p}>{p}</p>
          ))}
          {step.id === "what" && (
            <p className="flex flex-wrap gap-1.5" aria-label="Kinds of information">
              <KindChip kind="model" />
              <KindChip kind="measured" />
              <KindChip kind="official" />
            </p>
          )}
          {step.id === "official" && (
            <p>
              <button type="button" onClick={(e) => openDrawer(e.currentTarget)} className="font-medium text-accent underline decoration-accent/35 underline-offset-2 hover:decoration-accent" data-testid="tour-open-notices">
                Open the list of notices
              </button>
            </p>
          )}
          {step.id === "limits" && (
            <p>
              <Link href="/about" className="font-medium text-accent underline decoration-accent/35 underline-offset-2 hover:decoration-accent" data-testid="tour-about">
                About Neritic and its limits
              </Link>
            </p>
          )}
        </div>
        <div className="mt-4 flex items-center justify-between gap-2">
          {last ? (
            <span />
          ) : (
            <button type="button" onClick={close} className="min-h-[40px] px-1 text-[14px] text-ink-3 hover:text-ink" data-testid="tour-skip">
              Skip tour
            </button>
          )}
          <div className="flex gap-2">
            {i > 0 && (
              <button type="button" onClick={() => setI(i - 1)} className="min-h-[40px] rounded-md border border-hairline-strong px-3 text-[14px] font-medium text-ink hover:border-ink-3" data-testid="tour-back">
                Back
              </button>
            )}
            <button
              type="button"
              onClick={() => (last ? close() : setI(i + 1))}
              className="min-h-[40px] rounded-md bg-navy-900 px-4 text-[14px] font-medium text-white hover:bg-navy-800"
              data-testid="tour-next"
            >
              {last ? "Finish" : "Next"}
            </button>
          </div>
        </div>
        <p className="sr-only" aria-live="polite">
          Step {i + 1} of {STEPS.length}: {text.title}
        </p>
      </div>
    </>,
    document.body,
  );
}
