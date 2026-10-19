"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

export type SnapId = "peek" | "half" | "full";
export type SnapPoint = { id: SnapId; height: number };

const LABEL: Record<SnapId, string> = { peek: "collapsed", half: "half height", full: "full height" };

/**
 * Bottom sheet for phones and narrow windows (M5). Snap points are heights in pixels, from
 * the parent, which also keeps the map's fit area above the sheet. The handle can be
 * dragged (the sheet follows the finger and settles on the nearest snap, or the next one on a
 * quick flick), tapped (next size; from the largest, back to the smallest), or used from the
 * keyboard (arrow keys, Home and End).
 *
 * Transform-only (signature §9): the sheet keeps the height of its largest snap and slides with
 * translateY, so a drag or a settle never re-lays-out its content (smooth without a fast GPU).
 * The settle uses the spring token; reduced motion turns it into a cut (globals.css). The body's
 * bottom padding follows the snap, so the content scrolls to its end at every size.
 */
export function MobileSheet({
  snaps,
  snap,
  onSnap,
  header,
  children,
  label,
  testid = "mobile-sheet",
  field = false,
  edge = null,
}: {
  snaps: SnapPoint[];
  snap: SnapId;
  onSnap: (s: SnapId) => void;
  /** always visible under the handle (the peek content) */
  header: ReactNode;
  children?: ReactNode;
  label: string;
  testid?: string;
  /** v4 field ergonomics: a 44 px handle */
  field?: boolean;
  /** a fixed line under the handle, outside the scrolling body (v4: the map's credits, which then cover no map label) */
  edge?: ReactNode;
}) {
  const i = Math.max(0, snaps.findIndex((s) => s.id === snap));
  const target = snaps[i]?.height ?? 0;
  const [dragH, setDragH] = useState<number | null>(null);
  const drag = useRef<{ y: number; h: number; t: number; moved: boolean; samples: { y: number; t: number }[] } | null>(null);
  const full = snaps[snaps.length - 1]?.height ?? target;
  const body = useRef<HTMLDivElement>(null);

  // scroll back to the top when the sheet collapses, so the peek always shows its header
  useEffect(() => {
    if (snap === snaps[0]?.id) body.current?.scrollTo({ top: 0 });
  }, [snap, snaps]);

  const go = (k: number) => onSnap(snaps[Math.max(0, Math.min(snaps.length - 1, k))].id);
  const settle = (h: number, velocity: number) => {
    // where the sheet would come to rest 180 ms ahead at this speed; the nearest snap to that
    // wins, so a quick flick carries on to the next snap and a slow drag settles where it is
    const projected = h - velocity * 180;
    let best = 0;
    snaps.forEach((s, k) => {
      if (Math.abs(s.height - projected) < Math.abs(snaps[best].height - projected)) best = k;
    });
    go(best);
  };

  return (
    <section
      data-testid={testid}
      data-snap={snap}
      aria-label={label}
      className="theme-paper absolute inset-x-0 bottom-0 z-20 flex flex-col rounded-t-2xl bg-surface text-ink shadow-[0_-10px_30px_rgba(4,11,23,0.42)] will-change-transform"
      style={{
        height: full,
        transform: `translateY(${full - (dragH ?? target)}px)`,
        transition: dragH == null ? "transform var(--cw-d-medium) var(--cw-ease-settle)" : "none",
      }}
    >
      <button
        type="button"
        aria-label={`${label}: ${LABEL[snap]}. Activate to resize, or use the arrow keys.`}
        aria-expanded={snap !== snaps[0]?.id}
        data-testid="sheet-handle"
        className={`flex w-full shrink-0 touch-none items-center justify-center rounded-t-2xl ${field ? "h-14" : "h-6"}`}
        onClick={() => {
          if (drag.current?.moved) return;
          go(i === snaps.length - 1 ? 0 : i + 1);
        }}
        onKeyDown={(e) => {
          const k = e.key === "ArrowUp" ? i + 1 : e.key === "ArrowDown" ? i - 1 : e.key === "Home" ? 0 : e.key === "End" ? snaps.length - 1 : null;
          if (k == null) return;
          e.preventDefault();
          go(k);
        }}
        onPointerDown={(e) => {
          drag.current = { y: e.clientY, h: target, t: performance.now(), moved: false, samples: [{ y: e.clientY, t: performance.now() }] };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          const dy = e.clientY - d.y;
          if (!d.moved && Math.abs(dy) < 6) return;
          d.moved = true;
          d.samples = [...d.samples.slice(-4), { y: e.clientY, t: performance.now() }];
          const lo = snaps[0].height;
          const hi = snaps[snaps.length - 1].height;
          // a rubber band below the smallest size (a third of the overshoot); the largest is a stop,
          // since the sheet is no taller than it
          const h = d.h - dy;
          setDragH(h > hi ? hi : h < lo ? lo - (lo - h) / 3 : h);
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          if (!d) return;
          if (d.moved) {
            const dy = e.clientY - d.y;
            // velocity from the last few samples (px/ms; positive = downwards)
            const a = d.samples[0];
            const v = (e.clientY - a.y) / Math.max(1, performance.now() - a.t);
            settle(d.h - dy, v);
          }
          setDragH(null);
          // keep `moved` for the click event that follows pointerup, then forget the drag
          setTimeout(() => (drag.current = null), 0);
        }}
        onPointerCancel={() => {
          drag.current = null;
          setDragH(null);
        }}
      >
        <span className="h-1 w-10 rounded-full bg-hairline-strong" aria-hidden />
      </button>
      {/* below the handle's whole hit area: a tap on a credit link never resizes the sheet */}
      {edge && <div className="shrink-0 px-4 pb-1">{edge}</div>}
      <div ref={body} className="cw-fade-in min-h-0 flex-1 overflow-y-auto overscroll-contain px-4" style={{ paddingBottom: 20 + full - target }}>
        {header}
        {children}
      </div>
    </section>
  );
}
