"use client";

import { motion } from "motion/react";
import { useEffect, useId, useLayoutEffect, useRef, type ComponentPropsWithoutRef, type ReactNode } from "react";
import { clearCameraAnchor, setCameraAnchor } from "@/nx/chart/anchor";
import { useChartCover } from "@/nx/chart/cover";
import { getFrame, onFrame, type ChartFrame } from "@/nx/chart/runtime";
import { TICKET_FROM, ticketIn, ticketOpacity } from "../motion";
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
export function Ticket({ at, anchorY = 160, width, side = "east", label, labelledBy, lead: asked = true, inline = false, className, children, ...data }: TicketProps) {
  const lead = asked && !inline;
  const ref = useRef<HTMLElement>(null);
  const lineRef = useRef<SVGLineElement>(null);
  const reserveRef = useRef<HTMLDivElement>(null);
  const id = useId();
  const phone = usePhone();
  const here = useArrival();
  useLeaveGhost(ref);
  // the camera keeps the free part of the chart beside the ticket: a reservation the ticket's width wide, on its side
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
      const top0 = read(el, "--stage-top", 136);
      const W = f?.viewport.width ?? window.innerWidth, H = f?.viewport.height ?? window.innerHeight;
      const w = el.offsetWidth;
      // the chart's own buttons stand in the bottom-right corner: a ticket over that column stops above them (its stub
      // scrolls when the screen is short), so neither covers the other
      const ctrl = document.querySelector<HTMLElement>(".n12-controls")?.getBoundingClientRect();
      const colX = ctrl && ctrl.width ? [ctrl.left - 8, ctrl.right + 8] : null;
      const sideLeft = side === "east" ? W - g - w : g;
      const overCtrl = (l: number) => !!colX && l < colX[1] && l + w > colX[0];
      const bottomOf = (l: number) => (overCtrl(l) && ctrl ? ctrl.top - 8 : H - g);
      el.style.maxHeight = overCtrl(sideLeft) ? `${Math.max(240, bottomOf(sideLeft) - top0)}px` : "";
      const h = el.offsetHeight;
      let left: number, top: number;
      const line = lineRef.current;
      if (f && lon != null && lat != null) {
        const p = f.project(lon, lat);
        left = Math.round(Math.max(g, Math.min(W - g - w, side === "east" ? p.x + STUB_GAP : p.x - STUB_GAP - w)));
        top = Math.round(Math.max(top0, Math.min(bottomOf(left) - h, p.y - anchorY)));
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
        left = side === "east" ? W - g - w : g;
        top = top0;
        if (line) line.style.visibility = "hidden";
      }
      el.style.left = `${left}px`;
      el.style.top = `${top}px`;
    };
    const go = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => place(getFrame()));
    };
    place(getFrame());
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
  }, [lon, lat, side, anchorY, phone, inline]);

  // the camera anchor: where the place should stand so the ticket sits beside it
  useEffect(() => {
    const el = ref.current;
    if (!el || phone || !lead || lon == null || lat == null) return;
    const write = () =>
      setCameraAnchor({ id, lon, lat, side, width: el.offsetWidth, height: el.offsetHeight, anchorY, stub: STUB_GAP, gutter: read(el, "--gutter", 24), top: read(el, "--stage-top", 136) });
    write();
    const ro = new ResizeObserver(write);
    ro.observe(el);
    return () => {
      ro.disconnect();
      clearCameraAnchor(id);
    };
  }, [id, lon, lat, side, anchorY, phone, lead]);

  const style = width ? ({ "--ticket-w": `${width}px` } as React.CSSProperties) : undefined;
  return (
    <>
      <motion.section
        ref={ref}
        className={cx(s.ticket, className)}
        data-side={side}
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
        <svg className={s.leader} aria-hidden="true">
          <line ref={lineRef} />
        </svg>
        {children}
      </motion.section>
      {lead ? <div ref={reserveRef} className={s.reserve} data-side={side} style={style} aria-hidden="true" /> : null}
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
export function TicketStub({ className, children, ...rest }: ComponentPropsWithoutRef<"div">) {
  return (
    <div className={cx(s.stub, className)} {...rest}>
      {children}
    </div>
  );
}
