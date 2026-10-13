import { useEffect, useLayoutEffect, useRef } from "react";

/**
 * Mobile bottom sheet over the map — non-modal: the map above the sheet stays fully interactive.
 * Three snap points: peek (the answer in one line + the primary action), half (Level 2), full
 * (Level 3). Pointer Events with capture; velocity-aware snapping; a CSS `linear()` settle easing
 * (no library). Only `transform` animates (no layout work while dragging). The handle is a real button: tap cycles, ↑/↓ keys move between snaps, and its
 * aria-label states the current snap. Scrolling inside the sheet works at "full"; at "peek"/"half"
 * a vertical drag anywhere on the header moves the sheet.
 *
 * The sheet reports its visible height so the map can pad its camera (selection stays in view).
 */
export type Snap = "peek" | "half" | "full";
const ORDER: Snap[] = ["peek", "half", "full"];

export function Sheet({ snap, onSnap, peek = 132, bottom = 0, top = 56, onHeight, header, children, label }: {
  snap: Snap;
  onSnap: (s: Snap) => void;
  peek?: number;
  /** px reserved below the sheet (tab bar) and above it (masthead) */
  bottom?: number;
  top?: number;
  onHeight?: (px: number) => void;
  header: React.ReactNode;
  children: React.ReactNode;
  label: string;
}) {
  const el = useRef<HTMLDivElement>(null);
  const drag = useRef<{ y0: number; h0: number; t: number; lastY: number; lastT: number; v: number; moved: boolean } | null>(null);
  const heights = () => {
    const vh = (window.visualViewport?.height ?? innerHeight) - bottom;
    return { peek, half: Math.round(vh * 0.55), full: Math.round(vh - top - 8) };
  };
  // transform-only: the sheet is always full height and slides; only `transform` animates
  const apply = (h: number, animate: boolean) => {
    const s = el.current!;
    const full = heights().full;
    s.style.height = `${full}px`;
    s.style.transition = animate ? "transform var(--cw-d-medium) var(--cw-ease-settle)" : "none";
    s.style.transform = `translateY(${Math.round(full - h)}px)`;
  };
  const visible = () => (window.visualViewport?.height ?? innerHeight) - bottom - el.current!.getBoundingClientRect().top;
  useLayoutEffect(() => { apply(heights()[snap], true); onHeight?.(heights()[snap]); /* eslint-disable-next-line */ }, [snap, peek]);
  useEffect(() => {
    const on = () => { apply(heights()[snap], false); onHeight?.(heights()[snap]); };
    window.visualViewport?.addEventListener("resize", on);
    addEventListener("resize", on);
    return () => { window.visualViewport?.removeEventListener("resize", on); removeEventListener("resize", on); };
    // eslint-disable-next-line
  }, [snap]);

  const down = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const body = el.current!.querySelector(".sheet-body") as HTMLElement;
    // inside the scrolled body at "full", let the content scroll unless it is at the top
    if (snap === "full" && body.contains(e.target as Node) && body.scrollTop > 0) return;
    drag.current = { y0: e.clientY, h0: visible(), t: performance.now(), lastY: e.clientY, lastT: performance.now(), v: 0, moved: false };
  };
  const move = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const dy = e.clientY - d.y0;
    if (!d.moved && Math.abs(dy) < 6) return;
    if (!d.moved) { d.moved = true; (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId); }
    const now = performance.now();
    d.v = (e.clientY - d.lastY) / Math.max(1, now - d.lastT); // px/ms, + = downward
    d.lastY = e.clientY; d.lastT = now;
    const H = heights();
    let h = d.h0 - dy;
    if (h > H.full) h = H.full + (h - H.full) * 0.2; // rubber band
    if (h < H.peek) h = H.peek - (H.peek - h) * 0.2;
    apply(h, false);
  };
  const up = () => {
    const d = drag.current;
    drag.current = null;
    if (!d || !d.moved) return;
    const H = heights();
    const h = visible();
    // project 180 ms ahead with the release velocity, then pick the nearest snap
    const projected = h - d.v * 180;
    const next = ORDER.reduce((best, s) => (Math.abs(H[s] - projected) < Math.abs(H[best] - projected) ? s : best), snap);
    apply(H[next], true);
    if (next !== snap) onSnap(next); else onHeight?.(H[next]);
  };
  const cycle = () => onSnap(snap === "peek" ? "half" : snap === "half" ? "full" : "peek");
  const key = (e: React.KeyboardEvent) => {
    const i = ORDER.indexOf(snap);
    if (e.key === "ArrowUp" && i < 2) { e.preventDefault(); onSnap(ORDER[i + 1]); }
    if (e.key === "ArrowDown" && i > 0) { e.preventDefault(); onSnap(ORDER[i - 1]); }
    if (e.key === "Escape" && snap !== "peek") { e.preventDefault(); onSnap("peek"); }
  };

  return (
    <section ref={el} className="sheet" style={{ bottom }} data-snap={snap} aria-label={label} onKeyDown={key}>
      <div className="sheet-grab" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up}>
        <button className="sheet-handle" onClick={cycle} aria-label={`Details: ${snap === "peek" ? "collapsed" : snap === "half" ? "half open" : "fully open"}. Activate to ${snap === "full" ? "collapse" : "expand"}.`}>
          <span aria-hidden="true" />
        </button>
        {header}
      </div>
      <div className="sheet-body" onPointerDown={down} onPointerMove={move} onPointerUp={up} onPointerCancel={up} data-scroll={snap === "full" || undefined}>
        {children}
      </div>
    </section>
  );
}
