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

// the launch screen's clock (ms from its first style, times its tempo --nx-o): see theme.css, "launch screen"
const LOCKUP = 2800; // the lockup has formed and held: the earliest handoff
const HANDOFF = 700; // the lockup settles into the header's brand
const LATEST = 3900; // the CSS cap starts here: a map not ready by then gets its plain fade instead
const STILL_MIN = 600; // reduced motion: the still lockup holds at least this long, and at most
const STILL_MAX = 800;
/** --nx-ease-move and --nx-ease-in, for the handoff's script animations. */
const EASE_MOVE = "cubic-bezier(0.77, 0, 0.175, 1)";
const EASE_IN = "cubic-bezier(0.4, 0, 1, 1)";

/**
 * The launch screen's end (Splash, "Sounding"). Once the app is ready (Explore: its map has loaded) and
 * the lockup has formed and held, the handoff: the chart recedes and fades before the map shows (contours
 * are never drawn over land), the ground clears to the live map, and the lockup flies into the header's
 * brand (the screen rises above the header for this) and cross-fades into it. A map not ready by the cap
 * gets the cap's plain fade; reduced motion holds the still lockup 600-800 ms and fades it in 200. A tap,
 * click or key skips it at once (the inline script in Splash.tsx, which works before this has hydrated).
 * Times are read on the screen's own clock, the start of its CSS cap animation, so a slow page cannot cut
 * the play.
 */
function SplashGate() {
  const path = usePathname();
  useEffect(() => {
    const d = document.documentElement;
    const el = document.querySelector<HTMLElement>(".nx-splash");
    if (!d.classList.contains("nx-splash-on") || !el) return;
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const o = parseFloat(getComputedStyle(el).getPropertyValue("--nx-o")) || 1;
    const cap = el.getAnimations().find((a) => (a as CSSAnimation).animationName === "nx-splash-gone");
    const origin = typeof cap?.startTime === "number" ? cap.startTime : 0;
    const clock = () => Number(document.timeline.currentTime ?? performance.now()) - origin;
    const timers: number[] = [];
    const after = (ms: number, f: () => void) => timers.push(window.setTimeout(f, Math.max(0, ms)));
    const on = () => d.classList.contains("nx-splash-on") && !d.classList.contains("nx-splash-out");
    const end = () => d.classList.remove("nx-splash-on", "nx-splash-out", "nx-splash-land");
    const fade = () => {
      if (!on()) return;
      d.classList.add("nx-splash-out");
      after(200, end);
    };
    let started = false;
    const land = () => {
      if (started || !on() || clock() > LATEST * o) return;
      started = true;
      d.classList.add("nx-splash-land");
      // the chart steps back and is gone in 220 ms, before the ground has cleared
      el.querySelector(".nx-splash-chart")?.animate([{ opacity: 0.55, scale: "1" }, { opacity: 0, scale: "0.96" }], { duration: 220 * o, easing: EASE_IN, fill: "forwards" });
      // the lockup: into the header's brand (a phone's header has the mark only: the wordmark fades). It is
      // measured once its settle has finished (on a loaded device the CSS play can trail the screen's clock).
      const settled = document.querySelector(".nx-splash-hero")?.getAnimations().map((a) => a.finished) ?? [];
      Promise.all(settled).then(fly, fly);
    };
    const fly = () => {
      if (!on()) return;
      const ms = HANDOFF * o;
      const hero = document.querySelector<SVGSVGElement>(".nx-splash-hero svg");
      const stage = document.querySelector<HTMLElement>(".nx-splash-stage");
      const target = [...document.querySelectorAll<SVGSVGElement>('[data-testid="brand-home"] svg')].find((s) => s.getBoundingClientRect().width > 0);
      if (hero && stage && target) {
        const h = hero.getBoundingClientRect();
        const t = target.getBoundingClientRect();
        const lockup = t.width > t.height * 1.5;
        const mark = { x: t.left + t.height / 2, y: t.top + t.height / 2, size: t.height };
        const from = { x: h.left + h.width / 2, y: h.top + h.height / 2 };
        stage.style.transformOrigin = `${from.x}px ${from.y}px`;
        // the flight, then a 120 ms cross-fade into the header's own drawing (the brand fades in over the same 120 ms)
        stage.animate(
          [
            { transform: "none", opacity: 1, offset: 0, easing: EASE_MOVE },
            { transform: `translate(${mark.x - from.x}px, ${mark.y - from.y}px) scale(${mark.size / h.height})`, opacity: 1, offset: 0.83 },
            { transform: `translate(${mark.x - from.x}px, ${mark.y - from.y}px) scale(${mark.size / h.height})`, opacity: 0, offset: 1 },
          ],
          { duration: ms, fill: "forwards" },
        );
        if (!lockup) document.querySelector(".nx-splash-word")?.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 150 * o, easing: EASE_IN, fill: "forwards" });
        // it arrives as the header's three-line mark: the 150 and 250 m lines go during the flight
        document.querySelectorAll('.nx-splash-line[data-depth="150"], .nx-splash-line[data-depth="250"]').forEach((l) =>
          l.animate([{ opacity: 1, offset: 0 }, { opacity: 1, offset: 0.4 }, { opacity: 0, offset: 0.8 }], { duration: ms, fill: "forwards" }),
        );
      }
      after(ms * 0.83, () => d.classList.add("nx-splash-landed"));
      after(ms, () => {
        end();
        after(150, () => d.classList.remove("nx-splash-landed"));
      });
    };
    const ready = () => {
      if (!on()) return;
      const t = clock();
      if (still) return after(Math.min(STILL_MAX, Math.max(STILL_MIN, t)) - t, fade);
      after(LOCKUP * o - t, land);
    };
    // reduced motion: never longer than STILL_MAX, ready or not
    if (still) after(STILL_MAX - clock(), fade);
    // the cap's fade (a map not ready in time, or this script late): once it has run, the screen is over
    else after((LATEST + 300) * o - clock(), () => !started && end());
    const off = path === "/" ? onAppReady(ready) : (ready(), () => {});
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
