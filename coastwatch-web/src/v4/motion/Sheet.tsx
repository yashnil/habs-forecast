"use client";

import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { exitProps, usePresence } from "./presence";
import { prefersReducedMotion } from "./reduced";

/** Exit lengths: the plain close, and the spring that carries a released drag out. */
const EXIT_MS = 200;
const DRAG_EXIT_MS = 460;
const DRAG_MIN_PX = 6;
export const DISMISS_SHARE = 0.28;
export const DISMISS_VELOCITY = 0.55; // px/ms downward

/** A released drag dismisses when it went more than a share of the sheet's height, or it was a quick flick down. */
export function shouldDismiss(dragPx: number, heightPx: number, velocity: number): boolean {
  return dragPx > heightPx * DISMISS_SHARE || (velocity > DISMISS_VELOCITY && dragPx > DRAG_MIN_PX);
}

let locks = 0;
function lockScroll() {
  if (++locks === 1) document.documentElement.classList.add("nx-sheet-lock");
  return () => {
    if (--locks === 0) document.documentElement.classList.remove("nx-sheet-lock");
  };
}

/**
 * The one detail surface: a right-hand side sheet on desktop (420 px, 24 px travel), a bottom sheet on phones
 * (drag the handle or the head to dismiss, settling on the one spring). Built on the native <dialog>, so the
 * focus trap and Escape come with it: `modal` (default) also locks the page's scroll; `modal={false}` leaves the
 * page or map behind it usable (Escape still closes it).
 * Focus goes to the title on opening, and back to what opened it on closing. Labelled by its title.
 *
 * Use for: a row's or a place's detail.
 * Do not use for: anything short enough for a Fold, or confirmations.
 */
export function Sheet({ open, onClose, title, children, modal = true, closeLabel = "Close", testid, className = "", head }: { open: boolean; onClose: () => void; title: ReactNode; children: ReactNode; modal?: boolean; closeLabel?: string; testid?: string; className?: string; /** beside the title, above the content (a status tag) */ head?: ReactNode }) {
  const [exitMs, setExitMs] = useState(EXIT_MS);
  const { mounted, state } = usePresence(open, prefersReducedMotionSafe() ? 120 : exitMs);
  const dialog = useRef<HTMLDialogElement | null>(null);
  const titleRef = useRef<HTMLHeadingElement | null>(null);
  const opener = useRef<HTMLElement | null>(null);
  const titleId = useId();
  const closeRef = useRef(onClose);
  useEffect(() => {
    closeRef.current = onClose;
  }, [onClose]);

  // from the render that first shows it until it is gone
  useEffect(() => {
    const d = dialog.current;
    if (!mounted || !d) return;
    opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    setExitMs(EXIT_MS);
    if (!d.open) {
      if (modal) d.showModal();
      else d.show();
    }
    titleRef.current?.focus({ preventScroll: true });
    const unlock = modal ? lockScroll() : null;
    return () => {
      unlock?.();
      if (d.open) d.close();
      const o = opener.current;
      if (o?.isConnected) o.focus({ preventScroll: true });
      opener.current = null;
    };
  }, [mounted, modal]);

  // a non-modal sheet has no browser-provided Escape
  useEffect(() => {
    if (!open || modal) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closeRef.current();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, modal]);

  const drag = useRef<{ y: number; t: number; moved: boolean; last: { y: number; t: number }[] } | null>(null);
  const onDown = (e: React.PointerEvent<HTMLElement>) => {
    if (window.matchMedia("(min-width: 768px)").matches || prefersReducedMotion()) return;
    if ((e.target as Element).closest("button, a, input, select, textarea, [role=button]") && !(e.target as Element).closest("[data-sheet-handle]")) return;
    drag.current = { y: e.clientY, t: performance.now(), moved: false, last: [{ y: e.clientY, t: performance.now() }] };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = (e: React.PointerEvent<HTMLElement>) => {
    const d = drag.current;
    const el = dialog.current;
    if (!d || !el) return;
    const dy = e.clientY - d.y;
    if (!d.moved && Math.abs(dy) < DRAG_MIN_PX) return;
    d.moved = true;
    d.last = [...d.last.slice(-4), { y: e.clientY, t: performance.now() }];
    el.style.transition = "none";
    // upward gives a little (a rubber band); downward follows the finger
    el.style.transform = `translateY(${dy >= 0 ? dy : -Math.pow(-dy, 0.6)}px)`;
  };
  const release = (dismiss: boolean) => {
    const el = dialog.current;
    if (!el) return;
    el.style.transition = `transform var(--nx-t-sheet, 460ms) var(--nx-ease-spring, ease-out)`;
    el.style.transform = "";
    window.setTimeout(() => {
      el.style.transition = "";
    }, DRAG_EXIT_MS);
    if (dismiss) {
      setExitMs(DRAG_EXIT_MS);
      closeRef.current();
    }
  };
  const onUp = (e: React.PointerEvent<HTMLElement>) => {
    const d = drag.current;
    drag.current = null;
    if (!d?.moved || !dialog.current) return;
    const dy = e.clientY - d.y;
    const first = d.last[0];
    const v = (e.clientY - first.y) / Math.max(1, performance.now() - first.t);
    release(shouldDismiss(dy, dialog.current.offsetHeight, v));
  };
  const onCancel = () => {
    if (drag.current?.moved) release(false);
    drag.current = null;
  };

  if (!mounted) return null;
  return (
    <dialog
      ref={dialog}
      className={`nx-msheet ${className}`}
      aria-labelledby={titleId}
      aria-modal={modal ? "true" : undefined}
      data-modal={modal ? "true" : "false"}
      {...exitProps(state, testid)}
      onCancel={(e) => {
        e.preventDefault();
        closeRef.current();
      }}
      onClick={(e) => {
        if (modal && e.target === e.currentTarget) closeRef.current();
      }}
    >
      <div className="nx-msheet-inner">
        <header className="nx-msheet-head" onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onCancel}>
          <span className="nx-msheet-grab" aria-hidden />
          <div className="nx-msheet-title-row">
            <div className="min-w-0 flex-1">
              <h2 ref={titleRef} id={titleId} tabIndex={-1} className="nx-msheet-title">
                {title}
              </h2>
              {head ? <div className="nx-msheet-sub">{head}</div> : null}
            </div>
            <button type="button" aria-label={closeLabel} data-testid={testid ? `${testid}-close` : undefined} className="nx-press nx-focus nx-msheet-x" onClick={() => closeRef.current()}>
              <svg aria-hidden viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round">
                <path d="M6 6l12 12M18 6 6 18" />
              </svg>
            </button>
          </div>
        </header>
        <div className="nx-msheet-body">{children}</div>
      </div>
    </dialog>
  );
}

function prefersReducedMotionSafe() {
  return typeof window !== "undefined" && prefersReducedMotion();
}
