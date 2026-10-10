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
 * keyboard (arrow keys, Home and End). Height changes animate for 200 ms unless the system
 * asks for reduced motion (globals.css).
 */
export function MobileSheet({
  snaps,
  snap,
  onSnap,
  header,
  children,
  label,
  testid = "mobile-sheet",
}: {
  snaps: SnapPoint[];
  snap: SnapId;
  onSnap: (s: SnapId) => void;
  /** always visible under the handle (the peek content) */
  header: ReactNode;
  children?: ReactNode;
  label: string;
  testid?: string;
}) {
  const i = Math.max(0, snaps.findIndex((s) => s.id === snap));
  const target = snaps[i]?.height ?? 0;
  const [dragH, setDragH] = useState<number | null>(null);
  const drag = useRef<{ y: number; h: number; t: number; moved: boolean } | null>(null);
  const body = useRef<HTMLDivElement>(null);

  // scroll back to the top when the sheet collapses, so the peek always shows its header
  useEffect(() => {
    if (snap === snaps[0]?.id) body.current?.scrollTo({ top: 0 });
  }, [snap, snaps]);

  const go = (k: number) => onSnap(snaps[Math.max(0, Math.min(snaps.length - 1, k))].id);
  const settle = (h: number, velocity: number) => {
    // a flick (> 0.5 px/ms) moves one snap in its direction; otherwise the nearest snap wins
    if (Math.abs(velocity) > 0.5) return go(i + (velocity < 0 ? 1 : -1));
    let best = 0;
    snaps.forEach((s, k) => {
      if (Math.abs(s.height - h) < Math.abs(snaps[best].height - h)) best = k;
    });
    go(best);
  };

  return (
    <section
      data-testid={testid}
      data-snap={snap}
      aria-label={label}
      className={`theme-paper absolute inset-x-0 bottom-0 z-20 flex flex-col rounded-t-2xl bg-surface text-ink shadow-[0_-10px_30px_rgba(4,11,23,0.42)] ${dragH == null ? "transition-[height] duration-200 ease-out" : ""}`}
      style={{ height: dragH ?? target }}
    >
      <button
        type="button"
        aria-label={`${label}: ${LABEL[snap]}. Activate to resize, or use the arrow keys.`}
        aria-expanded={snap !== snaps[0]?.id}
        data-testid="sheet-handle"
        className="flex h-6 w-full shrink-0 touch-none items-center justify-center rounded-t-2xl"
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
          drag.current = { y: e.clientY, h: target, t: performance.now(), moved: false };
          e.currentTarget.setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          const d = drag.current;
          if (!d) return;
          const dy = e.clientY - d.y;
          if (!d.moved && Math.abs(dy) < 6) return;
          d.moved = true;
          const lo = snaps[0].height;
          const hi = snaps[snaps.length - 1].height;
          setDragH(Math.max(lo - 24, Math.min(hi + 12, d.h - dy)));
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          if (!d) return;
          if (d.moved) {
            const dy = e.clientY - d.y;
            settle(d.h - dy, dy / Math.max(1, performance.now() - d.t));
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
      <div ref={body} className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-5">
        {header}
        {children}
      </div>
    </section>
  );
}
