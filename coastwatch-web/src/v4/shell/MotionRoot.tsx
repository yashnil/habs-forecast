"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { MotionConfig } from "motion/react";
import { onAppReady, usePresence, exitProps } from "@/v4/motion";
import { onToast } from "@/v4/ui/toast";

/**
 * The root of every page: reduced motion for every motion component (fades stay, movement goes), the
 * launch screen's dismissal, and the one toast.
 */
export function MotionRoot({ children }: { children: React.ReactNode }) {
  return (
    <MotionConfig reducedMotion="user">
      {children}
      <SplashGate />
      <Toaster />
    </MotionConfig>
  );
}

// v7: only until the mark has resolved (the CSS timeline's last frame); a map that is ready earlier
// waits at most this long, and a slower one never holds the screen past CAP
const MIN_HOLD = 1050;
const PAGE_READY = 1050; // a page that never reports ready (not Explore)
const CAP = 2200; // never longer than this (the CSS animation in theme.css enforces it too)

/**
 * Steps the launch screen (Splash) aside: once the app is ready (Explore: its map has loaded), never
 * before MIN_HOLD, and gone by CAP. A tap, click or key skips it at once (the inline script in
 * Splash.tsx, which works before this has hydrated). Times are counted from the navigation's start
 * (performance.now), the same clock as the CSS animation.
 */
function SplashGate() {
  const path = usePathname();
  useEffect(() => {
    const d = document.documentElement;
    if (!d.classList.contains("nx-splash-on")) return;
    const exit = matchMedia("(prefers-reduced-motion: reduce)").matches ? 200 : 380;
    const timers: number[] = [];
    const after = (ms: number, f: () => void) => timers.push(window.setTimeout(f, Math.max(0, ms - performance.now())));
    let left = false;
    const leave = () => {
      if (left) return;
      left = true;
      after(MIN_HOLD, () => {
        // skipped meanwhile: the inline script has already taken it away
        if (!d.classList.contains("nx-splash-on") || d.classList.contains("nx-splash-out")) return;
        d.classList.add("nx-splash-out");
        timers.push(window.setTimeout(() => d.classList.remove("nx-splash-on", "nx-splash-out"), exit));
      });
    };
    after(CAP - exit, leave);
    const off = path === "/" ? onAppReady(leave) : (after(PAGE_READY, leave), () => {});
    return () => {
      off();
      timers.forEach((t) => window.clearTimeout(t));
    };
    // once, for the page the session opened on
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return null;
}

/** A short confirmation at the bottom centre ("Link copied", "Pack saved"); the controls keep their own live text. */
function Toaster() {
  const [shown, setShown] = useState<{ text: string; n: number } | null>(null);
  const [open, setOpen] = useState(false);
  useEffect(
    () =>
      onToast((text) => {
        setShown((s) => ({ text, n: (s?.n ?? 0) + 1 }));
        setOpen(true);
      }),
    [],
  );
  useEffect(() => {
    if (!open) return;
    const t = window.setTimeout(() => setOpen(false), 2500);
    return () => window.clearTimeout(t);
  }, [open, shown?.n]);
  const { mounted, state } = usePresence(open, 150);
  if (!mounted || !shown) return null;
  return (
    <div className="pointer-events-none fixed inset-x-0 z-[60] flex justify-center px-4" style={{ bottom: "calc(132px + env(safe-area-inset-bottom, 0px))" }}>
      {/* the controls that raise a toast announce it themselves, so this one stays out of the accessibility tree */}
      <p key={shown.n} aria-hidden className="nx-toast env-instrument rounded-full border border-n-line bg-n-surface px-4 py-2 text-[14px] font-semibold text-n-ink shadow-[var(--nx-shadow)]" {...exitProps(state, "toast")}>
        {shown.text}
      </p>
    </div>
  );
}
