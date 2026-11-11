"use client";

import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore, ViewTransition } from "react";
import { usePathname } from "next/navigation";
import { animate } from "motion/react";
import { marginSwap } from "@/nx/ds/motion";
import { useChartCover } from "@/nx/chart/cover";
import { MarginSheet } from "@/nx/ds/MarginSheet";
import { PHONE_QUERY, useMedia } from "@/nx/ds/useMedia";
import { OfflineNotice } from "@/nx/offline/OfflineNotice";

/**
 * The phone sheet's detents: resting (the page's statement in view above the tab bar), half (a reading and the
 * chart around it), reading height. A page asks for one with `openMargin`, e.g. Explore when a reading arrives.
 */
export type MarginDetent = "rest" | "half" | "full";
const ORDER: MarginDetent[] = ["rest", "half", "full"];

let asked: { detent: MarginDetent; n: number } = { detent: "rest", n: 0 };
const askListeners = new Set<() => void>();
/** Opens the phone margin sheet to a detent. Does nothing on wider screens, where the margin is a column. */
export function openMargin(detent: MarginDetent) {
  asked = { detent, n: asked.n + 1 };
  askListeners.forEach((l) => l());
}
const useAsked = () =>
  useSyncExternalStore(
    (l) => {
      askListeners.add(l);
      return () => askListeners.delete(l);
    },
    () => asked,
    () => asked,
  );

/** The resting height above the tab bar (shell.css --n12-rest: min(44dvh, 340px)) and the reading heights, px. */
function useDetents(): { px: string[]; rest: number; tabs: number } {
  const [h, setH] = useState(844);
  useEffect(() => {
    const on = () => setH(window.innerHeight);
    on();
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  const tabs = 56;
  const rest = Math.round(Math.min(h * 0.44, 340));
  // the tab bar lies over the sheet's lowest 56 px, so each detent adds it; reading height stops under the top bar
  return { px: [`${rest + tabs}px`, `${Math.round(Math.max(rest + 120, h * 0.62)) + tabs}px`, `${h - 52 - 12}px`], rest, tabs };
}

/**
 * The margin column (README §6): ruled, no container. Desktop and landscape: a column beside the sheet. Phone:
 * the ds MarginSheet, resting at a detent that shows the page's statement, dragged or stepped up to
 * read the rest; the chart stays live under it. Before hydration (and with no script) the same content is a
 * plain sheet at the resting height (shell.css). Contents cross-fade with React's <ViewTransition> when the
 * route changes; the chart is never in the snapshot, so it keeps panning live underneath.
 */
export function MarginFrame({ children }: { children: React.ReactNode }) {
  const phone = useMedia(PHONE_QUERY);
  const pageRef = useBackForwardSwap();
  const page = (
    <ViewTransition name="nx-margin" default="nx-margin-swap">
      <div className="n12-margin-page" ref={pageRef}>
        <OfflineNotice />
        {children}
      </div>
    </ViewTransition>
  );
  if (phone) return <PhoneMargin>{page}</PhoneMargin>;
  return (
    <main id="nx-margin" className="n12-margin" data-nx-margin tabIndex={-1} data-testid="nx-margin">
      <div className="n12-margin-scroll">{page}</div>
    </main>
  );
}

/**
 * Every route change starts the new page at the margin's top. Back and forward: React runs a history traversal without a view transition (it finishes it at once so the
 * browser can restore scroll), so the margin would swap with a cut while a tab or link cross-fades. The new
 * page comes in with the same `marginSwap` fade instead; the chart pans as on every route change. Reduced
 * motion: no fade (ds/MotionPrefs makes every Motion animation its final state).
 */
function useBackForwardSwap() {
  const pathname = usePathname();
  const ref = useRef<HTMLDivElement>(null);
  const traversed = useRef(false);
  useEffect(() => {
    const on = () => (traversed.current = true);
    window.addEventListener("popstate", on, true);
    return () => window.removeEventListener("popstate", on, true);
  }, []);
  const first = useRef(true);
  useLayoutEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    // a new page starts at its top: the margin is its own scroller, so the router's scroll reset does not reach it
    for (let el = ref.current?.parentElement; el; el = el.parentElement) {
      const oy = getComputedStyle(el).overflowY;
      if (oy === "auto" || oy === "scroll") {
        el.scrollTop = 0;
        break;
      }
    }
    if (!traversed.current || !ref.current) return;
    traversed.current = false;
    const a = animate(ref.current, { opacity: [0, 1] }, marginSwap);
    return () => a.stop();
  }, [pathname]);
  return ref;
}

function PhoneMargin({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const ask = useAsked();
  const { px, rest, tabs } = useDetents();
  const [at, setAt] = useState(0);
  // a new page rests again; a page's request moves the sheet
  useEffect(() => setAt(0), [pathname]);
  const seen = useRef(ask.n);
  useEffect(() => {
    if (ask.n === seen.current) return;
    seen.current = ask.n;
    setAt(ORDER.indexOf(ask.detent));
  }, [ask]);
  // the chart frames its scenes in the part the resting sheet leaves free, whatever detent the reader drags to
  const restRef = useRef<HTMLDivElement>(null);
  useChartCover("phone-margin", restRef, "bottom");
  return (
    <>
      <div ref={restRef} className="n12-margin-rest" style={{ height: rest, bottom: tabs }} aria-hidden="true" />
      <MarginSheet
        inline
        label="this page"
        detents={px}
        detent={at}
        onDetentChange={setAt}
        bottomInset={tabs}
        scroll="always"
        className="n12-margin-sheet"
        sheetProps={{ "data-nx-margin": "", "data-testid": "nx-margin" }}
      >
        <main id="nx-margin" className="n12-margin-main" tabIndex={-1}>
          {children}
        </main>
      </MarginSheet>
    </>
  );
}
