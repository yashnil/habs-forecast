"use client";

import { motion } from "motion/react";
import { useEffect, useId, useLayoutEffect, useRef, useState, type ComponentPropsWithoutRef, type ReactNode } from "react";
import { clearCameraAnchor, setCameraAnchor, ticketMargin } from "@/nx/chart/anchor";
import { useChartCover } from "@/nx/chart/cover";
import { getFrame, onFrame, type ChartFrame } from "@/nx/chart/runtime";
import { TICKET_FROM, ticketIn, ticketOpacity } from "../motion";
import { notePanel } from "../panels/store";
import { usePanel } from "../panels/usePanel";
import { usePhone } from "../useMedia";
import { cx } from "../util";
import { isPhoneNow, useArrival, useLeaveGhost } from "./arrive";
import s from "./stage.module.css";

type Data = { [key: `data-${string}`]: string | number | boolean | undefined };

export type Place = { lon: number; lat: number };

export interface TicketProps extends Data {
  /** The place the ticket belongs to. null: no place (the ticket stands at its side with no stub). */
  at: Place | null;
  /** px from the ticket's top that stands level with the place (default 160: about the statement's stamp). */
  anchorY?: number;
  /** px; the prototype's tickets are 404 (Today) and 472 (Alerts). */
  width?: number;
  /** Which side of the place the ticket stands: east (over the land, the default) or west (over the water). */
  side?: "east" | "west";
  /** The accessible name of the ticket's region (or `labelledBy`). */
  label?: string;
  labelledBy?: string;
  /** The camera brings the place to the ticket's edge (default true). Turn off for a second ticket on the same page. */
  lead?: boolean;
  /** Stand in the flow of its container instead of floating on the chart (the catalog, a page's own column). */
  inline?: boolean;
  /**
   * px from the screen's top where the ticket prefers to stand (the camera brings its place level with `anchorY` from
   * there). Default: just under the bar (94 px at the prototype's size), or under the layer chips when the ticket
   * reaches across them.
   */
  top?: number;
  /**
   * The ticket's distance from its side of the screen: `margin` (default), the prototype's tenth of the width; `gutter`,
   * flush with the gutter (a panel with no place, such as Explore's layers, that stands at the screen's edge).
   */
  edge?: "margin" | "gutter";
  /**
   * On a page whose panels are free (PageLayout `panels`), the ticket is a free panel under this key (default
   * "ticket"): moved by its grip it stops following its place and stays where the reader put it. false: it stays.
   */
  panel?: boolean | string;
  className?: string;
  children: ReactNode;
}

const STUB_GAP = 28; // the place to the ticket's edge, px: the stub is at most 24 px of it, drawn when the gap is ≤ 30
const read = (el: Element, v: string, fb: number) => parseFloat(getComputedStyle(el).getPropertyValue(v)) || fb;

/**
 * A ticket (v14 README §3): the card that sits beside its place on the chart and says one thing. On a wide screen it
 * stands a stub's length east of its port (or west), follows the place while the reader pans, stops at the screen's
 * edges, and a ≤ 24 px leader joins its nearest edge to the place. It grows from its pin as the camera arrives (at 62 %
 * of the flight) and, when its page or place changes, shrinks back toward it before the camera leaves. Pass a new
 * `key` when the place changes so the old ticket leaves and the new one arrives.
 *
 * On a phone it is a block in the bottom sheet, in reading order; nothing floats.
 *
 * Compose it from `TicketMain` and `TicketStub` (the perforated part under the main, with its notches).
 */
/** The highest a ticket whose left edge is at `left` may stand: under the bar, or under the layer chips if it reaches them. */
function topLimit(el: HTMLElement, left: number, width: number): number {
  const stageTop = read(el, "--stage-top", 136);
  const bar = document.querySelector(".n12-bar")?.getBoundingClientRect();
  const chips = document.querySelector(".nx-layers")?.getBoundingClientRect();
  if (!bar || !bar.height) return stageTop;
  const underBar = Math.round(bar.bottom + 26);
  if (chips && chips.width && left < chips.right + 12 && left + width > chips.left - 12) return Math.max(underBar, stageTop);
  return underBar;
}

export function Ticket({ at, anchorY = 160, width, side = "east", label, labelledBy, lead: asked = true, inline = false, top: preferTop, edge = "margin", panel: panelOpt = true, className, children, ...data }: TicketProps) {
  const lead = asked && !inline;
  const ref = useRef<HTMLElement>(null);
  const lineRef = useRef<SVGLineElement>(null);
  const reserveRef = useRef<HTMLDivElement>(null);
  const relayout = useRef<() => void>(() => {});
  const id = useId();
  const phone = usePhone();
  const here = useArrival();
  useLeaveGhost(ref);
  const panelKey = typeof panelOpt === "string" ? panelOpt : "ticket";
  const panel = usePanel(ref, {
    key: panelKey,
    kind: "ticket",
    name: label ?? (() => (labelledBy && document.getElementById(labelledBy)?.textContent?.trim()) || "Ticket"),
    enabled: panelOpt !== false && !inline,
    limits: { minW: 320, maxW: 600, minH: 160, maxH: 2000 },
    resizeHeight: true,
    relayout: () => relayout.current(),
  });
  const { movedRef } = panel;
  // the camera keeps the free part of the chart beside the ticket: a reservation the ticket's width wide, on its side
  // (kept while the reader has moved or collapsed it, so moving a panel never moves the chart: ds/panels)
  useChartCover(`ticket${id}`, reserveRef, side === "east" ? "right" : "left", !phone && lead);
  const lon = at?.lon ?? null, lat = at?.lat ?? null;

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || inline) return;
    let raf = 0;
    const place = (f: ChartFrame | null) => {
      if (isPhoneNow()) {
        el.style.left = el.style.top = "";
        return;
      }
      const g = read(el, "--gutter", 24);
      const W = f?.viewport.width ?? window.innerWidth, H = f?.viewport.height ?? window.innerHeight;
      const w = el.offsetWidth;
      const m = edge === "gutter" ? g : ticketMargin(W, g);
      // the chart's own buttons stand in the bottom-right corner: a ticket over that column stops above them (its stub
      // scrolls when the screen is short), so neither covers the other
      const ctrl = document.querySelector<HTMLElement>(".n12-controls")?.getBoundingClientRect();
      const colX = ctrl && ctrl.width ? [ctrl.left - 8, ctrl.right + 8] : null;
      const sideLeft = side === "east" ? W - m - w : m;
      const top0 = Math.max(topLimit(el, sideLeft, w), preferTop ?? 0);
      const overCtrl = (l: number) => !!colX && l < colX[1] && l + w > colX[0];
      const bottomOf = (l: number) => (overCtrl(l) && ctrl ? ctrl.top - 8 : H - g);
      // a ticket the reader has moved stands where they put it (ds/panels); its resting place is still worked out,
      // so the page's other panels keep their default layout around it
      const free = movedRef.current;
      // never past the stage's foot (or the chart's buttons): the ticket's own scroller takes what does not fit
      // (and no taller than the design's own cap, the screen less the stage's top and a gutter)
      const cap = H - read(el, "--stage-top", 136) - g;
      if (!free) el.style.maxHeight = `${Math.max(240, Math.min(cap, bottomOf(sideLeft) - top0))}px`;
      const h = el.offsetHeight;
      let left: number, top: number;
      const line = lineRef.current;
      if (f && lon != null && lat != null) {
        const p = f.project(lon, lat);
        left = Math.round(Math.max(g, Math.min(W - g - w, side === "east" ? p.x + STUB_GAP : p.x - STUB_GAP - w)));
        top = Math.round(Math.max(Math.max(topLimit(el, left, w), preferTop ?? 0), Math.min(bottomOf(left) - h, p.y - anchorY)));
        if (!free && top + h > bottomOf(left)) el.style.maxHeight = `${Math.max(240, Math.min(cap, bottomOf(left) - top))}px`;
        notePanel(panelKey, { rest: { x: left, y: top, w, h } });
        if (free) {
          if (line) line.style.visibility = "hidden";
          return;
        }
        el.style.transformOrigin = `${Math.round(p.x - left)}px ${Math.round(p.y - top)}px`;
        const py = p.y - top;
        const gap = side === "east" ? left - p.x : p.x - (left + w);
        const show = py > 24 && py < h - 24 && gap > 0 && gap <= 30;
        if (line) {
          const ex = side === "east" ? 0 : w;
          const sx = side === "east" ? -gap + 6 : w + gap - 6; // from just outside the port's ring
          line.setAttribute("x1", sx.toFixed(1));
          line.setAttribute("x2", String(ex));
          line.setAttribute("y1", py.toFixed(1));
          line.setAttribute("y2", py.toFixed(1));
          line.style.visibility = show ? "visible" : "hidden";
        }
      } else {
        left = sideLeft;
        top = top0;
        if (line) line.style.visibility = "hidden";
        notePanel(panelKey, { rest: { x: left, y: top, w, h } });
        if (free) return;
      }
      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
    };
    const go = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => place(getFrame()));
    };
    place(getFrame());
    relayout.current = () => place(getFrame());
    const off = onFrame(place);
    const ro = new ResizeObserver(go);
    ro.observe(el);
    window.addEventListener("resize", go);
    return () => {
      off();
      ro.disconnect();
      window.removeEventListener("resize", go);
      cancelAnimationFrame(raf);
    };
  }, [lon, lat, side, anchorY, phone, inline, preferTop, edge, movedRef, panelKey]);

  // the camera anchor: where the place should stand so the ticket sits beside it
  useEffect(() => {
    const el = ref.current;
    // kept when the reader moves the ticket (ds/panels): moving a panel never moves the chart, so the place stays
    // where the ticket's resting place put it (the moved ticket's own cover keeps names out from under it)
    if (!el || phone || !lead || lon == null || lat == null) return;
    const write = () => {
      const g = read(el, "--gutter", 24), w = el.offsetWidth, W = window.innerWidth;
      const mm = edge === "gutter" ? g : ticketMargin(W, g);
      const left = side === "east" ? W - mm - w : mm;
      setCameraAnchor({ id, lon, lat, side, width: w, height: el.offsetHeight, anchorY, stub: STUB_GAP, gutter: g, top: Math.max(topLimit(el, left, w), preferTop ?? 0), atGutter: edge === "gutter" || undefined });
    };
    write();
    const ro = new ResizeObserver(write);
    ro.observe(el);
    return () => {
      ro.disconnect();
      clearCameraAnchor(id);
    };
  }, [id, lon, lat, side, anchorY, phone, lead, preferTop, edge]);

  const style = width ? ({ "--ticket-w": `${width}px` } as React.CSSProperties) : undefined;
  return (
    <>
      <motion.section
        ref={ref}
        {...panel.attrs}
        className={cx(s.ticket, className, panel.attrs.className)}
        data-side={side}
        data-edge={edge}
        data-inline={inline || undefined}
        style={style}
        aria-label={label}
        aria-labelledby={labelledBy}
        initial={phone || inline ? false : { opacity: 0, scale: TICKET_FROM }}
        animate={phone || inline || here ? { opacity: 1, scale: 1 } : { opacity: 0, scale: TICKET_FROM }}
        transition={{ default: ticketIn, opacity: ticketOpacity }}
        data-nx-ticket=""
        {...data}
      >
        {panel.chrome}
        {/* the resize corner before the content, so the content's last part stays :last-child (it scrolls) */}
        {panel.resizer}
        <svg className={s.leader} aria-hidden="true">
          <line ref={lineRef} />
        </svg>
        {children}
      </motion.section>
      {lead ? <div ref={reserveRef} className={s.reserve} data-side={side} data-edge={edge} style={style} aria-hidden="true" /> : null}
    </>
  );
}

/** The ticket's main part: the statement (port · scope micro line, the statement, its stamp). */
export function TicketMain({ className, children, ...rest }: ComponentPropsWithoutRef<"div">) {
  return (
    <div className={cx(s.main, className)} {...rest}>
      {children}
    </div>
  );
}

/** The perforated stub under the main part: tabs, the panel, the actions. */
/**
 * The stub scrolls when its content outgrows the ticket (a long legend, wider fonts). A region that scrolls must be
 * reachable from the keyboard (WCAG 2.1.1; axe scrollable-region-focusable), so while it overflows it becomes a named,
 * focusable region; otherwise it adds no tab stop.
 */
export function TicketStub({ className, children, ...rest }: ComponentPropsWithoutRef<"div">) {
  const ref = useRef<HTMLDivElement>(null);
  const [scrolls, setScrolls] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const check = () => setScrolls(el.scrollHeight - el.clientHeight > 1 || el.scrollWidth - el.clientWidth > 1);
    check();
    const ro = new ResizeObserver(check);
    ro.observe(el);
    for (const c of Array.from(el.children)) ro.observe(c);
    const mo = new MutationObserver(check);
    mo.observe(el, { childList: true, subtree: true, characterData: true });
    return () => {
      ro.disconnect();
      mo.disconnect();
    };
  }, []);
  return (
    <div ref={ref} className={cx(s.stub, className)} {...(scrolls ? { tabIndex: 0, role: "region", "aria-label": rest["aria-label"] ?? "More of this panel" } : {})} {...rest}>
      {children}
    </div>
  );
}
