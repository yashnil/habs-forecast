"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState, useSyncExternalStore, ViewTransition, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { usePathname } from "next/navigation";
import { animate, motion, useDragControls, useMotionValue, type PanInfo } from "motion/react";
import { marginSwap, pane as paneMove, sheet as sheetSpring } from "@/nx/ds/motion";
import { useChartCover } from "@/nx/chart/cover";
import { Icon } from "@/nx/ds/Icon";
import { CHART_DOCK_ID } from "@/nx/ds/layout/Desk";
import { PANE_DETENTS, panePlacement, setPaneDetent, setPaneOpen, useChartPane, usePageLayout, type PaneDetent } from "@/nx/ds/layout/pane";
import { PHONE_QUERY, useMedia } from "@/nx/ds/useMedia";
import { OfflineNotice } from "@/nx/offline/OfflineNotice";
import { useCurrentPort } from "./port";

/**
 * The phone sheet's detents: resting (the page's statement in view above the tab bar), half (a reading and the
 * chart around it), reading height. A page asks for one with `openMargin`, e.g. Explore when a reading arrives.
 */
/** `peek` is lower than rest (about 30 % of the screen): a reading's first lines with most of the chart in view. Only a page asks for it. */
export type MarginDetent = "peek" | "rest" | "half" | "full";
const ORDER: MarginDetent[] = ["rest", "half", "full"];
/** `at` for the peek detent (below the first of the sheet's own three). */
const PEEK = -1;

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

const TABS = 56;
const BAR = 52;
/** The sheet's heights above the tab bar, px: resting (shell.css --n12-rest: min(44dvh, 340px)), half, reading. */
function useDetents(): { heights: number[]; rest: number; peek: number } {
  const [h, setH] = useState(844);
  useEffect(() => {
    const on = () => setH(window.innerHeight);
    on();
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  const rest = Math.round(Math.min(h * 0.44, 340));
  return { heights: [rest, Math.round(Math.max(rest + 120, h * 0.62)), h - BAR - 12 - TABS], rest, peek: Math.round(Math.min(rest - 48, h * 0.3)) };
}

/**
 * Every route change starts the new page at its top. Back and forward: React runs a history traversal without a
 * view transition (it finishes it at once so the browser can restore scroll), so the page would swap with a cut
 * while a tab or link cross-fades. The new page comes in with the same `marginSwap` fade instead; the chart pans
 * as on every route change. Reduced motion: no fade (ds/MotionPrefs makes every Motion animation its final state).
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
    // a new page starts at its top: the page's column (or, on a phone desk, the whole work area) is its own
    // scroller, so the router's scroll reset does not reach it
    for (let el = ref.current?.parentElement; el; el = el.parentElement) {
      const oy = getComputedStyle(el).overflowY;
      if (oy === "auto" || oy === "scroll") el.scrollTop = 0;
      if (el.classList.contains("n12-work")) break;
    }
    if (!traversed.current || !ref.current) return;
    traversed.current = false;
    const a = animate(ref.current, { opacity: [0, 1] }, marginSwap);
    return () => a.stop();
  }, [pathname]);
  return ref;
}

/** The pane's width in px for each size beside the desk (shell.css holds the same values). */
function detentWidths(total: number): Record<PaneDetent, number> {
  return { rail: 48, context: Math.min(440, Math.max(320, total * 0.28)), split: total * 0.5, chart: total - 436 };
}

/**
 * The divider between the desk and the chart pane: a separator the reader drags or steps with the arrow keys
 * through the pane's sizes (Enter collapses to the rail and back). While it is dragged the chart is only
 * clipped; on release the pane takes the nearest size and the chart resizes once and refits with `pan`.
 */
function PaneDivider({ work }: { work: React.RefObject<HTMLDivElement | null> }) {
  const pane = useChartPane();
  const start = useRef<{ x: number; w: number; total: number } | null>(null);
  const at = PANE_DETENTS.indexOf(pane.detent);
  const key = (e: KeyboardEvent<HTMLDivElement>) => {
    // the divider moves left as the pane grows
    const d = e.key === "ArrowLeft" ? 1 : e.key === "ArrowRight" ? -1 : 0;
    if (d) {
      e.preventDefault();
      pane.setDetent(PANE_DETENTS[Math.min(PANE_DETENTS.length - 1, Math.max(0, at + d))]);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      pane.toggle();
    } else if (e.key === "Home") {
      e.preventDefault();
      pane.setDetent("rail");
    } else if (e.key === "End") {
      e.preventDefault();
      pane.setDetent("chart");
    }
  };
  const down = (e: ReactPointerEvent<HTMLDivElement>) => {
    const el = work.current;
    const paneEl = el?.querySelector<HTMLElement>(".n12-pane");
    if (!el || !paneEl) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    start.current = { x: e.clientX, w: paneEl.offsetWidth, total: el.clientWidth };
    el.style.setProperty("--pane-hold", `${paneEl.offsetWidth}px`);
    el.setAttribute("data-pane-resizing", "");
  };
  const move = (e: ReactPointerEvent<HTMLDivElement>) => {
    const st = start.current;
    if (!st || !work.current) return;
    const w = Math.min(st.total - 320, Math.max(48, st.w + (st.x - e.clientX)));
    work.current.style.setProperty("--pane-w", `${Math.round(w)}px`);
  };
  const up = (e: ReactPointerEvent<HTMLDivElement>) => {
    const st = start.current;
    const el = work.current;
    start.current = null;
    if (!st || !el) return;
    const w = Math.min(st.total - 320, Math.max(48, st.w + (st.x - e.clientX)));
    const widths = detentWidths(st.total);
    let best: PaneDetent = pane.detent;
    // a click without travel leaves the size alone
    if (Math.abs(st.x - e.clientX) > 4) for (const d of PANE_DETENTS) if (Math.abs(widths[d] - w) < Math.abs(widths[best] - w)) best = d;
    if (best === pane.detent) {
      // back where it was: nothing travels
      el.style.removeProperty("--pane-w");
      el.style.removeProperty("--pane-hold");
      el.removeAttribute("data-pane-resizing");
      return;
    }
    // the new size takes over from where the hand left the divider (Workspace's travel clears the hold)
    setPaneDetent(best);
  };
  return (
    <div
      className="n12-divider"
      role="separator"
      aria-orientation="vertical"
      aria-label="Chart pane size"
      aria-valuenow={at}
      aria-valuemin={0}
      aria-valuemax={PANE_DETENTS.length - 1}
      aria-valuetext={{ rail: "Chart collapsed", context: "Chart beside the page", split: "Half and half", chart: "Chart largest" }[pane.detent]}
      tabIndex={0}
      onKeyDown={key}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      data-testid="nx-pane-divider"
    />
  );
}

/**
 * The workspace (v13 README §1): the page (a margin beside the chart, or a desk with the chart as a pane), the
 * divider and the chart pane, laid out by the stylesheet from the page's declared mode. One tree at every
 * width, before and after hydration: on a phone in chart mode the same <main> becomes the bottom sheet (its
 * handle and "More" control are always in the tree, shown by the stylesheet), dragged with Motion and the
 * `sheet` spring; on a phone desk page it is the page itself, under a chart strip that scrolls away to a bar.
 * Contents cross-fade with React's <ViewTransition> when the route changes; the chart is never in the snapshot.
 */
export function Workspace({ chart, children }: { chart: React.ReactNode; children: React.ReactNode }) {
  const pathname = usePathname();
  const phone = useMedia(PHONE_QUERY);
  const layout = usePageLayout();
  const pane = useChartPane();
  const { port } = useCurrentPort();
  const sheetOn = phone && layout.mode === "chart";
  const pageRef = useBackForwardSwap();
  const workRef = useRef<HTMLDivElement>(null);
  const bodyId = useId();

  // ------------------------------------------------------------ the phone sheet (chart mode)
  const ask = useAsked();
  const { heights, rest, peek } = useDetents();
  const [at, setAt] = useState(0);
  // a new page rests again; a page's request moves the sheet
  useEffect(() => setAt(0), [pathname]);
  const seen = useRef(ask.n);
  useEffect(() => {
    if (ask.n === seen.current) return;
    seen.current = ask.n;
    setAt(ask.detent === "peek" ? PEEK : ORDER.indexOf(ask.detent));
  }, [ask]);
  const full = heights[heights.length - 1];
  const heightAt = (i: number) => (i === PEEK ? peek : heights[i]);
  const offset = (i: number) => full - heightAt(i);
  const y = useMotionValue(0);
  const controls = useDragControls();
  const was = useRef(false);
  useLayoutEffect(() => {
    if (!sheetOn) {
      was.current = false;
      y.set(0);
      return;
    }
    const target = full - heightAt(at);
    // the first frame as a sheet is drawn at its detent (no slide in on arrival)
    if (!was.current) {
      was.current = true;
      y.set(target);
      return;
    }
    const c = animate(y, target, sheetSpring);
    return () => c.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sheetOn, at, full, heightAt(at)]);
  const onDragEnd = (_: unknown, info: PanInfo) => {
    // where the hand was heading: the nearest detent to the position a fling would carry the sheet to
    const aim = y.get() + info.velocity.y * 0.18;
    // the peek detent is offered to the hand only while the page has the sheet there
    let best = at === PEEK ? PEEK : 0;
    heights.forEach((_h, i) => {
      if (Math.abs(offset(i) - aim) < Math.abs(offset(best) - aim)) best = i;
    });
    if (best === at) animate(y, offset(at), sheetSpring);
    else setAt(best);
  };
  const last = at === heights.length - 1;
  // the chart frames its scenes in the part the resting sheet leaves free, whatever detent the reader drags to
  const restRef = useRef<HTMLDivElement>(null);
  useChartCover("phone-margin", restRef, "bottom", sheetOn);

  // ------------------------------------------------------------ the pane changing size beside the desk: the columns travel (`pane`)
  // The chart is never resized per frame: growing, it takes its new size at once and the column uncovers it;
  // shrinking, it keeps its size, is clipped, and resizes once when the column has arrived.
  useLayoutEffect(() => {
    const el = workRef.current;
    const paneEl = el?.querySelector<HTMLElement>(".n12-pane");
    const from = layout.fromW;
    if (!el || !paneEl) return;
    const done = () => {
      el.style.removeProperty("--pane-w");
      el.style.removeProperty("--pane-hold");
      el.removeAttribute("data-pane-resizing");
    };
    // (a drag that ended on a new size leaves its hold in place for this travel to take over)
    if (from === null || layout.mode !== "desk" || panePlacement() !== "beside" || performance.now() - layout.changedAt > 200) return done();
    // the width the new size gives, measured without the drag's or a previous travel's inline width
    el.style.removeProperty("--pane-w");
    const to = paneEl.offsetWidth;
    if (Math.abs(to - from) < 2) return done();
    el.style.setProperty("--pane-hold", `${Math.max(from, to)}px`);
    el.style.setProperty("--pane-w", `${from}px`);
    el.setAttribute("data-pane-resizing", "");
    const a = animate(from, to, { ...paneMove, onUpdate: (v) => el.style.setProperty("--pane-w", `${Math.round(v)}px`), onComplete: done });
    return () => {
      a.stop();
      done();
    };
  }, [layout.detent, layout.fromW, layout.changedAt, layout.mode]);

  // for tests: the page-facing sheet request, as a page would call it
  useEffect(() => {
    (window as unknown as { __nxShell?: { openMargin: typeof openMargin } }).__nxShell = { openMargin };
  }, []);

  // ------------------------------------------------------------ the pane as an overlay: Escape closes it
  useEffect(() => {
    if (!layout.open) return;
    const key = (e: globalThis.KeyboardEvent) => e.key === "Escape" && setPaneOpen(false);
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [layout.open]);
  const toggleStrip = () => {
    if (panePlacement() === "strip" && !layout.open) workRef.current?.scrollTo({ top: 0 });
    pane.toggle();
  };

  return (
    <div className="n12-work" ref={workRef} data-pane={layout.detent} data-pane-open={layout.open ? "" : undefined} data-testid="nx-work">
      <div ref={restRef} className="n12-margin-rest" style={{ height: at === PEEK ? peek : rest }} aria-hidden="true" />
      <motion.main
        id="nx-margin"
        className="n12-margin"
        data-nx-margin=""
        data-testid="nx-margin"
        data-detent={sheetOn ? (at === PEEK ? "peek" : at) : undefined}
        tabIndex={-1}
        style={{ y, height: sheetOn ? full : undefined }}
        drag={sheetOn ? "y" : false}
        dragListener={false}
        dragControls={controls}
        dragConstraints={{ top: 0, bottom: offset(at === PEEK ? PEEK : 0) }}
        dragElastic={0.06}
        dragMomentum={false}
        onDragEnd={onDragEnd}
      >
        {/* the sheet's handle row: in the tree at every width (one tree before and after hydration), shown on a phone in chart mode */}
        <div className="n12-sheet-top" onPointerDown={(e) => sheetOn && controls.start(e)}>
          <span className="n12-sheet-handle" aria-hidden="true" />
          <button
            type="button"
            className="n12-sheet-more"
            onPointerDown={(e) => e.stopPropagation()}
            onClick={() => setAt(last ? 0 : at + 1)}
            aria-expanded={at > 0}
            aria-controls={bodyId}
            aria-label={last ? "Show less of this page" : "Show more of this page"}
          >
            {last ? "Less" : "More"}
            <Icon name="chevron" size={20} direction={last ? "down" : "up"} />
          </button>
        </div>
        <div className="n12-margin-scroll" id={bodyId}>
          <ViewTransition name="nx-margin" default="nx-margin-swap">
            <div className="n12-margin-page" ref={pageRef}>
              <OfflineNotice />
              {children}
            </div>
          </ViewTransition>
        </div>
      </motion.main>
      <PaneDivider work={workRef} />
      <div className="n12-pane" data-testid="nx-pane">
        {chart}
        <div id={CHART_DOCK_ID} className="n12-dock-slot" />
        {/* desk pages only (the stylesheet shows each where it applies): the rail, the phone strip's bar, the overlay's close */}
        <button type="button" className="n12-pane-rail" onClick={pane.toggle} aria-label={port ? `Show the chart of ${port.name}` : "Show the chart"} data-testid="nx-pane-rail">
          <Icon name="chevron" size={20} direction="left" />
          {/* words, not a picture: the chart is out of sight here, so the rail says what opening it shows */}
          <span suppressHydrationWarning>{port ? `Chart · ${port.name}` : "Chart"}</span>
        </button>
        <div className="n12-pane-bar">
          <span className="n12-pane-bar-name" suppressHydrationWarning>
            {port ? `Chart · ${port.name}` : "Chart"}
          </span>
          <button type="button" className="n12-pane-bar-btn" onClick={toggleStrip} aria-expanded={layout.open} data-testid="nx-pane-toggle">
            {layout.open ? "Smaller" : "Larger"}
            <Icon name="chevron" size={20} direction={layout.open ? "up" : "down"} />
          </button>
        </div>
        <button type="button" className="n12-pane-close" onClick={() => setPaneOpen(false)} aria-label="Close the chart">
          <Icon name="close" />
        </button>
      </div>
    </div>
  );
}
