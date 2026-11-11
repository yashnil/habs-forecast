"use client";

import { useEffect, useRef, useState, useSyncExternalStore, ViewTransition } from "react";
import { usePathname } from "next/navigation";
import { useChartCover } from "@/nx/chart/cover";
import { MarginSheet } from "@/nx/ds/MarginSheet";
import { PHONE_QUERY, useMedia } from "@/nx/ds/useMedia";

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
  const page = (
    <ViewTransition name="nx-margin" default="nx-margin-swap">
      <div className="n12-margin-page">{children}</div>
    </ViewTransition>
  );
  if (phone) return <PhoneMargin>{page}</PhoneMargin>;
  return (
    <main id="nx-margin" className="n12-margin" data-nx-margin tabIndex={-1} data-testid="nx-margin">
      <div className="n12-margin-scroll">{page}</div>
    </main>
  );
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
