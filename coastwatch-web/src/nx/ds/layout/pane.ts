"use client";

import { useLayoutEffect, useSyncExternalStore } from "react";

/**
 * The page's layout (docs/coastwatch/v13/design-system/README.md §1, "Desk and Chart"). A route declares its
 * mode with <PageLayout>: `chart` (the chart is the work: a margin beside the sheet) or `desk` (the page's work
 * takes the space and the chart is a contextual pane). The pane's size is the reader's, kept per route on this
 * device. The mode itself is also written in the DOM (a hidden marker), so the stylesheet lays the page out
 * before any script runs.
 */
export type PageMode = "chart" | "desk";
/** rail: a 48 px rail; context: about 28 % of the width (default); split: half; chart: the desk shrinks to a column. */
export type PaneDetent = "rail" | "context" | "split" | "chart";
export const PANE_DETENTS: PaneDetent[] = ["rail", "context", "split", "chart"];

export type LayoutState = {
  mode: PageMode;
  paneKey: string | null;
  detent: PaneDetent;
  /** the chart shown where it is not beside the desk: the overlay pane on a narrow screen, the phone strip enlarged */
  open: boolean;
  /** when the pane last changed size (performance.now()), so the chart glides to its new fit */
  changedAt: number;
};

const KEY = "nx-pane-v1";
const SERVER: LayoutState = { mode: "chart", paneKey: null, detent: "context", open: false, changedAt: -1e9 };
let state: LayoutState = SERVER;
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((l) => l());
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
};

function stored(): Record<string, PaneDetent> {
  try {
    const v = JSON.parse(window.localStorage.getItem(KEY) ?? "{}") as Record<string, PaneDetent>;
    return v && typeof v === "object" ? v : {};
  } catch {
    return {};
  }
}

export const getLayout = () => state;
/** True for a moment after the pane changed size: the chart host refits with `pan` instead of a jump. */
export const paneJustChanged = () => typeof performance !== "undefined" && performance.now() - state.changedAt < 900;

export function setPageLayout(mode: PageMode, paneKey: string | null, defaultPane: PaneDetent) {
  const saved = paneKey ? stored()[paneKey] : undefined;
  const detent = saved && PANE_DETENTS.includes(saved) ? saved : defaultPane;
  if (state.mode === mode && state.paneKey === paneKey && state.detent === detent) return;
  state = { mode, paneKey, detent, open: false, changedAt: state.changedAt };
  emit();
}

export function setPaneDetent(detent: PaneDetent) {
  if (state.detent === detent) return;
  state = { ...state, detent, changedAt: performance.now() };
  if (state.paneKey) {
    try {
      window.localStorage.setItem(KEY, JSON.stringify({ ...stored(), [state.paneKey]: detent }));
    } catch {
      // this page view only
    }
  }
  emit();
}

export function setPaneOpen(open: boolean) {
  if (state.open === open) return;
  state = { ...state, open, changedAt: performance.now() };
  emit();
}

export const usePageLayout = (): LayoutState => useSyncExternalStore(subscribe, getLayout, () => SERVER);

/** Where the chart pane is laid out: beside the desk, as an overlay (narrow or short screens), or the phone's strip. */
export type PanePlacement = "beside" | "overlay" | "strip";
export const BESIDE_QUERY = "(min-width: 1100px) and (min-height: 520px)";
export const STRIP_QUERY = "(max-width: 720px)";
export function panePlacement(): PanePlacement {
  if (typeof window === "undefined") return "beside";
  if (window.matchMedia(STRIP_QUERY).matches) return "strip";
  return window.matchMedia(BESIDE_QUERY).matches ? "beside" : "overlay";
}

let lastShown: PaneDetent = "context";
/**
 * The chart pane for a desk page: its size, and one toggle that does the right thing where the pane is (beside
 * the desk it collapses to the rail and back; as an overlay or the phone strip it opens and closes).
 */
export function useChartPane() {
  const s = usePageLayout();
  const collapsed = s.detent === "rail";
  return {
    mode: s.mode,
    detent: s.detent,
    setDetent: setPaneDetent,
    collapsed,
    open: s.open,
    setOpen: setPaneOpen,
    toggle: () => {
      if (panePlacement() !== "beside") return setPaneOpen(!getLayout().open);
      const d = getLayout().detent;
      if (d === "rail") setPaneDetent(lastShown);
      else {
        lastShown = d;
        setPaneDetent("rail");
      }
    },
  };
}

/** Writes the page's layout before paint; the hidden marker carries the mode for the stylesheet (no script needed). */
export function usePageLayoutEffect(mode: PageMode, paneKey: string | null, defaultPane: PaneDetent) {
  useLayoutEffect(() => {
    setPageLayout(mode, paneKey, defaultPane);
    return () => setPageLayout("chart", null, "context");
  }, [mode, paneKey, defaultPane]);
}
