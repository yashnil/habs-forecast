"use client";

import { animate, useReducedMotion } from "motion/react";
import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, useSyncExternalStore, type KeyboardEvent, type PointerEvent as ReactPointerEvent, type ReactNode, type RefObject } from "react";
import { createPortal } from "react-dom";
import { registerCover } from "@/nx/chart/cover";
import { Icon } from "../Icon";
import { settle } from "../motion";
import { usePhone } from "../useMedia";
import { clampBox, clampSize, fromPlacement, guidesFor, nudge, toPlacement, type Area, type Box, type Guide, type Limits, type Placement } from "./geometry";
import { getPanelsPage, layoutChanged, livePanels, loadLayout, notePanel, registerPanel, resetCount, resetLayout, savePanel, subscribePanels } from "./store";
import p from "./panels.module.css";

export type PanelKind = "ticket" | "dock" | "pass";

export interface PanelOptions {
  /** The panel's key on its page (persistence, tests): "ticket", "sea-window". */
  key: string;
  kind: PanelKind;
  /** The panel's accessible name ("Sea window"); a function is asked after mount (a ticket named by its heading). */
  name: string | (() => string);
  /** false: this object stays where its page puts it even on a page whose panels are free. */
  enabled?: boolean;
  /** Width and height limits, px (always within the screen). */
  limits: Limits;
  /** The panel's height may be dragged (its content scrolls); else only its width. */
  resizeHeight?: boolean;
  /**
   * The edge the object covers at rest, for the camera (a dock's or pass's "bottom"). Once the reader moves or collapses
   * it, the camera keeps reserving that resting box, so moving a panel never moves the chart (nor the ticket that
   * follows its place); the moved panel is only an obstacle for the chart's names. Omit for a ticket, whose
   * reservation is already a strip on its side.
   */
  restCover?: "bottom";
  /** Called after a reset clears the panel's own placement, so the object lays itself out again at once (a ticket beside its place). */
  relayout?: () => void;
}

export interface PanelState {
  /** The page's panels are free and this one takes part (never on a phone, never inline). */
  on: boolean;
  /** The reader has put it somewhere: it no longer stands at its default place (a ticket stops following its port). */
  moved: boolean;
  movedRef: RefObject<boolean>;
  collapsed: boolean;
  /** Resized by the reader (its default width rules no longer apply). */
  resized: boolean;
  /** The panel covers the chart by itself (moved or collapsed); the object's own default cover stands down. */
  free: boolean;
  /** Attributes and class for the panel's element. */
  attrs: { className?: string; "data-panel"?: string; "data-panel-moved"?: string; "data-panel-collapsed"?: string };
  /** The grip, collapse and reset strip at the panel's top, and its resize corner: render first and last inside it. */
  chrome: ReactNode;
  resizer: ReactNode;
}

const read = (el: Element, v: string, fb: number) => parseFloat(getComputedStyle(el).getPropertyValue(v)) || fb;
const STEP = 8, STEP_FAR = 40;

/** The part of the stage a panel may stand in, in its offset parent's px: inside the gutters, under the bar. */
export function stageArea(el: HTMLElement): Area {
  const parent = (el.offsetParent as HTMLElement | null) ?? document.documentElement;
  const pr = parent.getBoundingClientRect();
  const W = pr.width || window.innerWidth, H = pr.height || window.innerHeight;
  const g = read(el, "--gutter", 24);
  const bar = document.querySelector(".n12-bar")?.getBoundingClientRect();
  const top = bar && bar.height ? Math.round(bar.bottom - pr.top + 12) : g;
  return { left: g, top, right: W - g, bottom: H - g };
}

const boxOf = (el: HTMLElement): Box => ({ x: el.offsetLeft, y: el.offsetTop, w: el.offsetWidth, h: el.offsetHeight });

function usePanelsPage(): string | null {
  return useSyncExternalStore(subscribePanels, getPanelsPage, () => null);
}

function useLayoutChanged(page: string | null): boolean {
  return useSyncExternalStore(
    subscribePanels,
    () => (page ? layoutChanged(loadLayout(page)) : false),
    () => false,
  );
}

function useResetCount(page: string | null): number {
  return useSyncExternalStore(subscribePanels, () => (page ? resetCount(page) : 0), () => 0);
}

/**
 * A floating stage object as a free panel (v15.1, DIRECTION §4): moved by its grip (pointer or arrow keys), resized
 * from its corner within limits, collapsed to its header, kept on screen, its place remembered on this device per page
 * and key. Dragging one panel never moves another and nothing snaps: guides only show where edges line up. "Reset
 * layout" appears in the chrome of a panel the reader has changed and puts the whole page back. Off on a phone (the objects are blocks in the
 * sheet there) and on pages that do not free their panels (PageLayout `panels`).
 */
export function usePanel(ref: RefObject<HTMLElement | null>, o: PanelOptions): PanelState {
  const page = usePanelsPage();
  const phone = usePhone();
  const on = !!page && !phone && o.enabled !== false;
  const changed = useLayoutChanged(on ? page : null);
  const resets = useResetCount(on ? page : null);
  const reduce = useReducedMotion();
  const [moved, setMoved] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [resized, setResized] = useState(false);
  const [guides, setGuides] = useState<Guide[]>([]);
  const [coverRev, setCoverRev] = useState(0);
  const [title, setTitle] = useState(typeof o.name === "string" ? o.name : "");
  const movedRef = useRef(false);
  const placement = useRef<Placement | null>(null);
  /** where the object stood at rest (its page's layout), for the camera's reservation while it is moved */
  const restBox = useRef<Box | null>(null);
  const opts = useRef(o);
  useLayoutEffect(() => {
    opts.current = o;
  });
  const hintId = useId();
  const { key } = o;

  useEffect(() => {
    if (typeof o.name === "function") setTitle(o.name());
    else setTitle(o.name);
  }, [o]);

  const markMoved = useCallback((v: boolean) => {
    movedRef.current = v;
    setMoved(v);
    if (ref.current) notePanel(key, { moved: v });
  }, [key, ref]);

  /** Stands the element at `b` (its own left/top; a dock's or pass's bottom/right let go). */
  const put = useCallback((el: HTMLElement, b: Box) => {
    // a pass or full dock is sized by its left and right: its width is held before the right lets go
    if (!el.style.width) el.style.width = `${el.offsetWidth}px`;
    el.style.left = `${b.x}px`;
    el.style.top = `${b.y}px`;
    el.style.right = "auto";
    el.style.bottom = "auto";
  }, []);

  const putSize = useCallback((el: HTMLElement, w: number | null, h: number | null) => {
    el.style.width = w ? `${w}px` : "";
    el.style.height = h ? `${h}px` : "";
    el.style.maxHeight = h ? "none" : "";
  }, []);

  /** The moved panel at its placement for its present size, kept on screen. */
  const replace = useCallback(() => {
    const el = ref.current, pl = placement.current;
    if (!el || !pl || !movedRef.current) return;
    const area = stageArea(el);
    // a panel taller than the screen now gives up height (its content scrolls), never its header
    const maxH = area.bottom - area.top;
    if (el.offsetHeight > maxH) el.style.maxHeight = `${maxH}px`;
    put(el, fromPlacement(pl, { w: el.offsetWidth, h: el.offsetHeight }, area));
  }, [ref, put]);

  // register on the stage; restore what the reader left on this page
  useLayoutEffect(() => {
    const el = ref.current;
    if (!on || !el || !page) return;
    const off = registerPanel(key, el);
    restBox.current = boxOf(el);
    // the panel last touched stands in front of the others (panels may overlap where the reader puts them)
    const front = () => {
      for (const x of livePanels()) if (x.el !== el) x.el.removeAttribute("data-panel-front");
      el.setAttribute("data-panel-front", "");
    };
    el.addEventListener("pointerdown", front, true);
    el.addEventListener("focusin", front);
    const saved = loadLayout(page).panels[key];
    if (saved?.w || saved?.h) {
      const area = stageArea(el);
      const lim = opts.current.limits;
      const sz = clampSize({ w: saved.w ?? el.offsetWidth, h: saved.h ?? el.offsetHeight }, lim, area);
      putSize(el, saved.w ? sz.w : null, saved.h && opts.current.resizeHeight ? sz.h : null);
      setResized(true);
    }
    if (saved?.c) setCollapsed(true);
    if (saved?.p) {
      placement.current = saved.p;
      markMoved(true);
      replace();
    }
    return () => {
      off();
      el.removeEventListener("pointerdown", front, true);
      el.removeEventListener("focusin", front);
      el.removeAttribute("data-panel-front");
    };
  }, [on, page, key, ref, putSize, markMoved, replace]);

  // leaving the free state (a phone, a page without free panels): the element is the page's again
  useLayoutEffect(() => {
    const el = ref.current;
    if (on || !el) return;
    for (const k of ["left", "top", "right", "bottom", "width", "height", "maxHeight", "translate"] as const) el.style[k] = "";
    placement.current = null;
    movedRef.current = false;
    setMoved(false);
    setCollapsed(false);
    setResized(false);
  }, [on, ref]);

  // the screen changes size, or the panel's content does: a moved panel keeps its corner and stays on screen
  useEffect(() => {
    const el = ref.current;
    if (!on || !el) return;
    let raf = 0;
    const go = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const area = stageArea(el);
        if (!movedRef.current && !el.hasAttribute("data-panel-collapsed")) restBox.current = boxOf(el);
        // a collapsed panel's box is its header: its kept size waits for it to open again
        if (resized && !el.hasAttribute("data-panel-collapsed")) {
          const lim = opts.current.limits;
          const sz = clampSize({ w: el.offsetWidth, h: el.offsetHeight }, lim, area);
          if (el.style.width && sz.w !== el.offsetWidth) el.style.width = `${sz.w}px`;
          if (el.style.height && sz.h !== el.offsetHeight) el.style.height = `${sz.h}px`;
        }
        replace();
      });
    };
    window.addEventListener("resize", go);
    const ro = new ResizeObserver(go);
    ro.observe(el);
    return () => {
      window.removeEventListener("resize", go);
      ro.disconnect();
      cancelAnimationFrame(raf);
    };
  }, [on, ref, replace, resized]);

  // a reset anywhere on the page: back to the default layout, gliding there unless the reader asked for less motion
  const firstReset = useRef(resets);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!on || !el || resets === firstReset.current) return;
    firstReset.current = resets;
    const was = el.getBoundingClientRect();
    for (const k of ["left", "top", "right", "bottom", "width", "height", "maxHeight"] as const) el.style[k] = "";
    placement.current = null;
    movedRef.current = false;
    notePanel(key, { moved: false });
    setMoved(false);
    setCollapsed(false);
    setResized(false);
    setCoverRev((n) => n + 1);
    opts.current.relayout?.();
    if (reduce) return;
    const now = el.getBoundingClientRect();
    const dx = Math.round(was.left - now.left), dy = Math.round(was.top - now.top);
    if (Math.abs(dx) + Math.abs(dy) < 2) return;
    animate(el, { translate: [`${dx}px ${dy}px`, "0px 0px"] }, settle).then(() => {
      el.style.translate = "";
    });
  }, [on, resets, ref, key, reduce]);

  const persist = useCallback(
    (patch: Parameters<typeof savePanel>[2]) => {
      if (page) savePanel(page, key, patch);
    },
    [page, key],
  );

  /** Freezes a panel where it stands now (the first move): from here on it is the reader's. */
  const takeOver = useCallback((el: HTMLElement) => {
    const area = stageArea(el);
    if (!movedRef.current && !el.hasAttribute("data-panel-collapsed")) restBox.current = boxOf(el);
    const b = clampBox(boxOf(el), area);
    put(el, b);
    if (!movedRef.current) markMoved(true);
    placement.current = toPlacement(b, area);
    return { b, area };
  }, [put, markMoved]);

  const others = useCallback((): Box[] => livePanels().filter((x) => x.key !== key && x.el.isConnected).map((x) => boxOf(x.el)), [key]);

  // ---------------------------------------------------------------- drag
  const drag = useRef<{ id: number; x0: number; y0: number; b: Box; area: Area; live: boolean } | null>(null);
  const onStripDown = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const el = ref.current;
      if (!el || e.button !== 0 || (e.target as HTMLElement).closest("[data-panel-tool]")) return;
      e.preventDefault();
      (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
      drag.current = { id: e.pointerId, x0: e.clientX, y0: e.clientY, b: boxOf(el), area: stageArea(el), live: false };
    },
    [ref],
  );
  const onStripMove = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const d = drag.current, el = ref.current;
      if (!d || !el || d.id !== e.pointerId) return;
      const dx = e.clientX - d.x0, dy = e.clientY - d.y0;
      if (!d.live) {
        if (Math.abs(dx) + Math.abs(dy) < 3) return;
        d.live = true;
        const t = takeOver(el);
        d.b = t.b;
        d.area = t.area;
        el.setAttribute("data-panel-dragging", "");
      }
      const b = clampBox({ ...d.b, x: d.b.x + dx, y: d.b.y + dy }, d.area);
      put(el, b);
      const o0 = ((el.offsetParent as HTMLElement | null) ?? document.documentElement).getBoundingClientRect();
      setGuides(guidesFor(b, others(), d.area).map((g) => (g.axis === "x" ? { ...g, at: g.at + o0.left, from: g.from + o0.top, to: g.to + o0.top } : { ...g, at: g.at + o0.top, from: g.from + o0.left, to: g.to + o0.left })));
    },
    [ref, takeOver, put, others],
  );
  const onStripUp = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const d = drag.current, el = ref.current;
      drag.current = null;
      if (!d || !el) return;
      try {
        (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
      } catch {
        // already released
      }
      el.removeAttribute("data-panel-dragging");
      setGuides([]);
      if (!d.live) return;
      const area = stageArea(el);
      placement.current = toPlacement(boxOf(el), area);
      persist({ p: placement.current });
      setCoverRev((n) => n + 1);
    },
    [ref, persist],
  );

  // ---------------------------------------------------------------- keyboard: arrows move the focused grip
  const keyTimer = useRef(0);
  const settleCover = useCallback(() => {
    window.clearTimeout(keyTimer.current);
    keyTimer.current = window.setTimeout(() => setCoverRev((n) => n + 1), 360);
  }, []);
  useEffect(() => () => window.clearTimeout(keyTimer.current), []);
  const onGripKey = useCallback(
    (e: KeyboardEvent<HTMLElement>) => {
      const el = ref.current;
      if (!el || !e.key.startsWith("Arrow")) return;
      e.preventDefault();
      const { b, area } = takeOver(el);
      const n = nudge(b, e.key, e.shiftKey ? STEP_FAR : STEP, area);
      if (!n) return;
      put(el, n);
      placement.current = toPlacement(n, area);
      persist({ p: placement.current });
      settleCover();
    },
    [ref, takeOver, put, persist, settleCover],
  );

  // ---------------------------------------------------------------- resize from the corner
  const sizing = useRef<{ id: number; x0: number; y0: number; b: Box; area: Area } | null>(null);
  const applySize = useCallback(
    (el: HTMLElement, b: Box, w: number, h: number, area: Area) => {
      const lim = opts.current.limits;
      const room: Area = { ...area, right: area.right, bottom: area.bottom };
      const sz = clampSize({ w: Math.min(w, room.right - b.x), h: Math.min(h, room.bottom - b.y) }, { ...lim, maxW: Math.min(lim.maxW, room.right - b.x), maxH: Math.min(lim.maxH, room.bottom - b.y) }, area);
      putSize(el, sz.w, opts.current.resizeHeight ? sz.h : null);
      return sz;
    },
    [putSize],
  );
  const onSizeDown = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const el = ref.current;
      if (!el || e.button !== 0) return;
      e.preventDefault();
      e.stopPropagation();
      e.currentTarget.setPointerCapture(e.pointerId);
      const t = takeOver(el);
      sizing.current = { id: e.pointerId, x0: e.clientX, y0: e.clientY, b: t.b, area: t.area };
      el.setAttribute("data-panel-dragging", "");
    },
    [ref, takeOver],
  );
  const onSizeMove = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const s = sizing.current, el = ref.current;
      if (!s || !el || s.id !== e.pointerId) return;
      applySize(el, s.b, s.b.w + e.clientX - s.x0, s.b.h + e.clientY - s.y0, s.area);
      setResized(true);
    },
    [ref, applySize],
  );
  const onSizeUp = useCallback(
    (e: ReactPointerEvent<HTMLElement>) => {
      const s = sizing.current, el = ref.current;
      sizing.current = null;
      if (!s || !el) return;
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // already released
      }
      el.removeAttribute("data-panel-dragging");
      const area = stageArea(el);
      placement.current = toPlacement(boxOf(el), area);
      persist({ p: placement.current, w: el.offsetWidth, h: opts.current.resizeHeight ? el.offsetHeight : null });
      setCoverRev((n) => n + 1);
    },
    [ref, persist],
  );
  const onSizeKey = useCallback(
    (e: KeyboardEvent<HTMLElement>) => {
      const el = ref.current;
      if (!el || !e.key.startsWith("Arrow")) return;
      e.preventDefault();
      const { b, area } = takeOver(el);
      const st = e.shiftKey ? STEP_FAR : STEP;
      const dw = e.key === "ArrowRight" ? st : e.key === "ArrowLeft" ? -st : 0;
      const dh = e.key === "ArrowDown" ? st : e.key === "ArrowUp" ? -st : 0;
      const sz = applySize(el, b, b.w + dw, b.h + dh, area);
      setResized(true);
      placement.current = toPlacement({ ...b, w: sz.w, h: sz.h }, area);
      persist({ p: placement.current, w: sz.w, h: opts.current.resizeHeight ? sz.h : null });
      settleCover();
    },
    [ref, takeOver, applySize, persist, settleCover],
  );

  // ---------------------------------------------------------------- collapse
  const toggle = useCallback(() => {
    const el = ref.current;
    if (el && !collapsed && !movedRef.current) restBox.current = boxOf(el);
    persist({ c: !collapsed });
    setCollapsed(!collapsed);
    setCoverRev((n) => n + 1);
  }, [persist, collapsed, ref]);

  // ---------------------------------------------------------------- the camera's cover, when the panel stands by itself
  const free = on && (moved || collapsed);
  // the moved panel itself: an obstacle the chart's names keep clear of (never camera padding)
  useLayoutEffect(() => {
    const el = ref.current;
    if (!free || !el) return;
    return registerCover(`panel:${key}`, el, "none");
  }, [free, coverRev, key, ref]);
  // the camera keeps reserving the resting box (a hidden stand-in where the object stood)
  const restSide = o.restCover;
  useLayoutEffect(() => {
    const el = ref.current, rb = restBox.current;
    const parent = el?.offsetParent as HTMLElement | null;
    if (!free || !restSide || !el || !rb || !parent) return;
    const ghost = document.createElement("div");
    ghost.setAttribute("aria-hidden", "true");
    ghost.dataset.panelRest = key;
    Object.assign(ghost.style, { position: "absolute", left: `${rb.x}px`, width: `${rb.w}px`, height: `${rb.h}px`, bottom: `${Math.max(0, parent.clientHeight - rb.y - rb.h)}px`, visibility: "hidden", pointerEvents: "none" });
    parent.appendChild(ghost);
    const off = registerCover(`panel-rest:${key}`, ghost, restSide);
    return () => {
      off();
      ghost.remove();
    };
  }, [free, restSide, key, ref]);

  const chrome = on ? (
    <div className={p.chrome} data-panel-chrome="" onPointerDown={onStripDown} onPointerMove={onStripMove} onPointerUp={onStripUp} onPointerCancel={onStripUp}>
      <button type="button" className={p.grip} data-panel-grip="" aria-label={`Move ${title || "panel"}`} aria-describedby={hintId} onKeyDown={onGripKey}>
        <span className={p.pill} aria-hidden="true" />
      </button>
      {collapsed ? (
        <span className={p.title} aria-hidden="true">
          {title}
        </span>
      ) : null}
      <span className={p.tools}>
        {changed && (moved || collapsed || resized) ? (
          <button type="button" className={p.reset} data-panel-tool="" data-testid="panel-reset" onClick={() => page && resetLayout(page)}>
            Reset layout
          </button>
        ) : null}
        <button type="button" className={p.tool} data-panel-tool="" aria-label={`${collapsed ? "Expand" : "Collapse"} ${title || "panel"}`} title={collapsed ? "Expand" : "Collapse"} aria-expanded={!collapsed} onClick={toggle} data-testid="panel-collapse">
          <Icon name={collapsed ? "plus" : "minus"} />
        </button>
      </span>
      <span id={hintId} hidden>
        Arrow keys move it; with Shift, further.
      </span>
      {guides.length && typeof document !== "undefined" ? createPortal(<Guides guides={guides} />, document.body) : null}
    </div>
  ) : null;

  const resizer =
    on && !collapsed ? (
      <button
        type="button"
        className={p.resize}
        data-panel-resize=""
        data-axis={o.resizeHeight ? "both" : "x"}
        aria-label={`Resize ${title || "panel"}`}
        aria-describedby={hintId}
        onPointerDown={onSizeDown}
        onPointerMove={onSizeMove}
        onPointerUp={onSizeUp}
        onPointerCancel={onSizeUp}
        onKeyDown={onSizeKey}
      >
        <span aria-hidden="true" />
      </button>
    ) : null;

  return {
    on,
    moved: on && moved,
    movedRef,
    collapsed: on && collapsed,
    resized: on && resized,
    free,
    attrs: on ? { className: p.panel, "data-panel": key, "data-panel-moved": moved ? "" : undefined, "data-panel-collapsed": collapsed ? "" : undefined } : {},
    chrome,
    resizer,
  };
}

/** The guides while a panel is dragged: hairlines where its edges meet the screen's or another panel's. */
function Guides({ guides }: { guides: Guide[] }) {
  return (
    <div className={p.guides} aria-hidden="true" data-testid="panel-guides">
      {guides.map((g) => (
        <span
          key={`${g.axis}${g.at}`}
          className={p.guide}
          data-axis={g.axis}
          data-kind={g.kind}
          style={g.axis === "x" ? { left: g.at, top: g.from, height: g.to - g.from } : { top: g.at, left: g.from, width: g.to - g.from }}
        />
      ))}
    </div>
  );
}

/** The width a resting panel may take so it stops short of the other resting panels (ticket) beside it. */
export function restingOthers(key: string): Box[] {
  return livePanels()
    .filter((x) => x.key !== key && x.el.isConnected)
    .map((x) => x.rest ?? (x.moved ? null : boxOf(x.el)))
    .filter((b): b is Box => !!b);
}
