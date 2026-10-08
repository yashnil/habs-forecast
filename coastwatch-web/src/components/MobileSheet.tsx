"use client";

import { useRef, useState, type ReactNode } from "react";

type Snap = "peek" | "half" | "full";
const HEIGHT: Record<Snap, string> = { peek: "96px", half: "52dvh", full: "calc(100dvh - 120px)" };
const ORDER: Snap[] = ["peek", "half", "full"];

/** Slide-up information panel for small screens: tap or drag the handle between three heights. */
export function MobileSheet({ peek, children, expandTo }: { peek: ReactNode; children: ReactNode; expandTo?: Snap }) {
  const [snap, setSnap] = useState<Snap>(expandTo ?? "peek");
  const drag = useRef<{ y: number; snap: Snap } | null>(null);
  const dragged = useRef(false);

  const step = (dir: 1 | -1) => setSnap((s) => ORDER[Math.min(2, Math.max(0, ORDER.indexOf(s) + dir))]);

  return (
    <section
      data-testid="mobile-sheet"
      data-snap={snap}
      className="absolute inset-x-0 bottom-0 z-20 flex flex-col rounded-t-xl border-t border-hairline-strong bg-surface shadow-[0_-12px_32px_rgba(0,0,0,0.45)] transition-[height] duration-200 ease-out"
      style={{ height: HEIGHT[snap] }}
    >
      <button
        type="button"
        aria-label={snap === "full" ? "Collapse panel" : "Expand panel"}
        aria-expanded={snap !== "peek"}
        data-testid="sheet-handle"
        onClick={() => {
          if (dragged.current) {
            dragged.current = false;
            return;
          }
          setSnap(snap === "full" ? "peek" : ORDER[ORDER.indexOf(snap) + 1]);
        }}
        onPointerDown={(e) => {
          drag.current = { y: e.clientY, snap };
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (!d) return;
          const dy = e.clientY - d.y;
          if (Math.abs(dy) > 40) {
            dragged.current = true;
            step(dy < 0 ? 1 : -1);
          }
        }}
        className="flex w-full shrink-0 flex-col items-center gap-1.5 px-4 pb-2 pt-2 text-left"
      >
        <span className="h-1 w-10 rounded-full bg-[var(--cw-ink-3)]/60" aria-hidden />
        <span className="w-full truncate text-[13px] font-semibold text-ink">{peek}</span>
      </button>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-6">{children}</div>
    </section>
  );
}
