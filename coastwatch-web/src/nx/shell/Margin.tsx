"use client";

import { useEffect, useId, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";
import { usePathname } from "next/navigation";
import { animate, motion, useDragControls, useMotionValue, type PanInfo } from "motion/react";
import { marginSwap, sheet as sheetSpring } from "@/nx/ds/motion";
import { useChartCover } from "@/nx/chart/cover";
import { Icon } from "@/nx/ds/Icon";
import { CHART_DOCK_ID } from "@/nx/ds/layout/Desk";
import { setPaneOpen, useChartPane, usePageLayout } from "@/nx/ds/layout/pane";
import { GHOST_LAYER_ID } from "@/nx/ds/stage/arrive";
import { useSheetRest } from "@/nx/ds/stage/Sheet";
import { PHONE_QUERY, useMedia } from "@/nx/ds/useMedia";
import { OfflineNotice } from "@/nx/offline/OfflineNotice";

/**
 * The phone sheet's detents. `rest`: the page's statement in view under the chart strip; `half`; `full`: reading
 * height. `peek` is lower than rest (about 30 % of the screen): a reading's first lines with most of the chart in
 * view; only a page asks for it (Explore when a reading arrives).
 */
export type MarginDetent = "peek" | "rest" | "half" | "full";
const PEEK = -1;

let asked: { detent: MarginDetent; n: number } = { detent: "rest", n: 0 };
const askListeners = new Set<() => void>();
/** Opens the phone sheet to a detent. Does nothing on wider screens. */
export function openMargin(detent: MarginDetent) {
  asked = { detent, n: asked.n + 1 };
  askListeners.forEach((l) => l());
}
/** v14 name for the same request. */
export const openSheet = openMargin;
const useAsked = () =>
  useSyncExternalStore(
    (l) => {
      askListeners.add(l);
      return () => askListeners.delete(l);
    },
    () => asked,
    () => asked,
  );

/** The bottom tab bar's height (shell.css --tabbar-h) and the room kept at the top for the brand pill. */
const TABS = 62;
const TOP = 64;

/**
 * The sheet's heights above the tab bar, px. A v14 stage page sets how much chart its resting sheet leaves
 * (ds Sheet `map`, 27 % by default); a v13 page rests at min(44 %, 340 px).
 */
function useDetents(stage: boolean, mapShare: number): { heights: number[]; rest: number; peek: number } {
  const [h, setH] = useState(844);
  useEffect(() => {
    const on = () => setH(window.innerHeight);
    on();
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  const full = h - TOP - TABS;
  const rest = stage ? Math.max(160, Math.min(full, Math.round(h - TABS - h * mapShare))) : Math.round(Math.min(h * 0.44, 340));
  const half = Math.round(Math.max(rest + 120, h * 0.62));
  const heights = half < full - 40 ? [rest, half, full] : [rest, full];
  return { heights, rest, peek: Math.round(Math.min(rest - 48, h * 0.3)) };
}

/**
 * Every route change starts the new page at its top. Back and forward bring the new page in with the `marginSwap`
 * fade (a v13 legacy page's whole column; v14 objects arrive with the camera on their own).
 */
function useRouteStart() {
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

/**
 * The workspace (v14 README §2): one persistent full-bleed chart behind every page, and the page over it.
 *
 * - `stage` (v14 pages): the page's objects (Ticket, Dock, Pass) float on the chart and place themselves; the page
 *   element is a transparent stage. On a phone the chart is a strip at the top and the page reads in one bottom
 *   sheet whose resting detent leaves the strip useful (ds Sheet).
 * - `chart` (a v13 page not yet moved): the page's column floats as a card on the right, the chart fitted beside it;
 *   on a phone, the same sheet.
 * - `desk` (a v13 desk page not yet moved): the desk floats as a wide panel on the left; on a phone, v13's chart
 *   strip over a scrolling page.
 *
 * One tree at every width before and after hydration; one MapLibre instance for every page (ChartHost), never
 * resized per frame. The ghost layer on top is where a leaving ticket shrinks after its page has gone.
 */
export function Workspace({ chart, children }: { chart: React.ReactNode; children: React.ReactNode }) {
  const pathname = usePathname();
  const phone = useMedia(PHONE_QUERY);
  const layout = usePageLayout();
  const pane = useChartPane();
  const mode = layout.mode;
  const stage = mode === "stage";
  const sheetOn = phone && mode !== "desk";
  const pageRef = useRouteStart();
  const workRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  const dockRef = useRef<HTMLDivElement>(null);
  const bodyId = useId();
  const sheetRest = useSheetRest();

  // ------------------------------------------------------------ the phone sheet
  const ask = useAsked();
  const { heights, rest, peek } = useDetents(stage, sheetRest.map);
  const [at, setAt] = useState(0);
  useEffect(() => setAt(0), [pathname]);
  const seen = useRef(ask.n);
  useEffect(() => {
    if (ask.n === seen.current) return;
    seen.current = ask.n;
    const order: MarginDetent[] = heights.length === 3 ? ["rest", "half", "full"] : ["rest", "full", "full"];
    setAt(ask.detent === "peek" ? PEEK : Math.min(heights.length - 1, order.indexOf(ask.detent)));
  }, [ask, heights.length]);
  const full = heights[heights.length - 1];
  const heightAt = (i: number) => (i === PEEK ? peek : heights[Math.min(i, heights.length - 1)]);
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
    const aim = y.get() + info.velocity.y * 0.18;
    let best = at === PEEK ? PEEK : 0;
    heights.forEach((_h, i) => {
      if (Math.abs(offset(i) - aim) < Math.abs(offset(best) - aim)) best = i;
    });
    if (best === at) animate(y, offset(at), sheetSpring);
    else setAt(best);
  };
  // back at rest from a raised detent, the sheet shows the page from its first line
  const prevAt = useRef(at);
  useEffect(() => {
    const from = prevAt.current;
    prevAt.current = at;
    if (!sheetOn || at !== 0 || from <= 0) return;
    const body = document.getElementById(bodyId);
    if (body) body.scrollTop = 0;
  }, [at, sheetOn, bodyId]);
  const last = at === heights.length - 1;

  // ------------------------------------------------------------ what covers the chart, for the camera and the names
  const restRef = useRef<HTMLDivElement>(null);
  useChartCover("phone-margin", restRef, "bottom", sheetOn);
  useChartCover("margin-card", mainRef, mode === "desk" ? "left" : "right", !phone && !stage && !(mode === "desk" && layout.open));
  useChartCover("chart-dock", dockRef, "bottom", !phone);

  // for tests: the page-facing sheet request, as a page would call it
  useEffect(() => {
    (window as unknown as { __nxShell?: { openMargin: typeof openMargin } }).__nxShell = { openMargin };
  }, []);

  // a v13 desk page's chart opened over it (narrow screens, the phone strip): Escape closes it
  useEffect(() => {
    if (!layout.open) return;
    const key = (e: globalThis.KeyboardEvent) => e.key === "Escape" && setPaneOpen(false);
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [layout.open]);
  const toggleStrip = () => {
    if (phone && !layout.open) workRef.current?.scrollTo({ top: 0 });
    pane.toggle();
  };

  return (
    <div className="n12-work" ref={workRef} data-mode={mode} data-pane={layout.detent} data-pane-open={layout.open ? "" : undefined} data-testid="nx-work">
      <div ref={restRef} className="n12-margin-rest" style={{ height: at === PEEK ? peek : rest }} aria-hidden="true" />
      <motion.main
        ref={mainRef}
        id="nx-margin"
        className="n12-margin"
        data-nx-margin=""
        data-testid="nx-margin"
        data-detent={sheetOn ? (at === PEEK ? "peek" : at) : undefined}
        tabIndex={-1}
        style={{ y, height: sheetOn ? full : undefined, "--sheet-hidden": `${sheetOn ? offset(at) : 0}px` } as unknown as React.CSSProperties}
        drag={sheetOn ? "y" : false}
        dragListener={false}
        dragControls={controls}
        dragConstraints={{ top: 0, bottom: offset(at === PEEK ? PEEK : 0) }}
        dragElastic={0.06}
        dragMomentum={false}
        onDragEnd={onDragEnd}
      >
        {/* the sheet's handle row: in the tree at every width, shown on a phone */}
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
            <Icon name="chevron" size={20} direction={last ? "down" : "up"} />
          </button>
        </div>
        <div className="n12-margin-scroll" id={bodyId}>
          <div className="n12-margin-page" ref={pageRef}>
            <OfflineNotice />
            {children}
          </div>
        </div>
      </motion.main>
      <div className="n12-pane" data-testid="nx-pane">
        {chart}
        {/* a v13 page's ChartDock (Explore's time) floats along the chart's foot */}
        <div id={CHART_DOCK_ID} ref={dockRef} className="n12-dock-slot" />
        {/* v13 desk pages only (the stylesheet shows each where it applies): the phone strip's bar, the overlay's close */}
        <div className="n12-pane-bar">
          <span className="n12-pane-bar-name" data-testid="nx-pane-subject">
            {layout.subject ? `Chart · ${layout.subject}` : "Chart"}
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
      <div id={GHOST_LAYER_ID} className="nx-ghosts" aria-hidden="true" />
    </div>
  );
}
