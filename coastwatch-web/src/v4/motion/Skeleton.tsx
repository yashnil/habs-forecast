"use client";

import { useEffect, useState, type CSSProperties, type ReactNode } from "react";

export const LATE_DELAY_MS = 300;
export const LATE_MIN_MS = 300;

/**
 * The timing behind useLate, free of React: nothing for `delay` ms; once it shows, it stays at least `min` ms.
 * A fast load never shows a skeleton at all, and one that appeared never flickers away.
 */
export function lateTimer({ delay, min, onChange }: { delay: number; min: number; onChange: (show: boolean) => void }) {
  let shown = false;
  let shownAt = 0;
  let t: ReturnType<typeof setTimeout> | null = null;
  const clear = () => {
    if (t != null) clearTimeout(t);
    t = null;
  };
  return {
    set(loading: boolean, now: () => number) {
      clear();
      if (loading) {
        if (shown) return;
        t = setTimeout(() => {
          shown = true;
          shownAt = now();
          onChange(true);
        }, delay);
      } else if (shown) {
        t = setTimeout(
          () => {
            shown = false;
            onChange(false);
          },
          Math.max(0, min - (now() - shownAt)),
        );
      }
    },
    clear,
  };
}

/** True while a loading state has lasted long enough to deserve a placeholder (nothing for 300 ms; then at least 300 ms). */
export function useLate(loading: boolean, { delay = LATE_DELAY_MS, min = LATE_MIN_MS }: { delay?: number; min?: number } = {}): boolean {
  const [show, setShow] = useState(false);
  const [timer] = useState(() => lateTimer({ delay, min, onChange: setShow }));
  useEffect(() => {
    timer.set(loading, () => performance.now());
  }, [loading, timer]);
  useEffect(() => () => timer.clear(), [timer]);
  return show;
}

/**
 * A flat placeholder the shape of what is coming. No shimmer.
 * Use for: content that takes longer than a moment and has a known size, shown through `useLate` (or `Late`).
 * Do not use for: the map (its poster is real) or anything under 300 ms.
 */
export function Skeleton({ className = "", style, lines }: { className?: string; style?: CSSProperties; lines?: number }) {
  if (lines && lines > 1)
    return (
      <span className={`nx-skel-lines ${className}`} aria-hidden style={style}>
        {Array.from({ length: lines }, (_, i) => (
          <span key={i} className="nx-skel" style={i === lines - 1 ? { width: "62%" } : undefined} />
        ))}
      </span>
    );
  return <span className={`nx-skel ${className}`} aria-hidden style={style} />;
}

/** `loading` true: nothing, then `fallback` after 300 ms; the real content otherwise. */
export function Late({ loading, fallback, children }: { loading: boolean; fallback: ReactNode; children: ReactNode }) {
  const show = useLate(loading);
  if (show) return <div className="nx-fade">{fallback}</div>;
  return loading ? null : <>{children}</>;
}
