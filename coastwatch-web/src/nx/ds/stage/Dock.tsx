"use client";

import { motion } from "motion/react";
import { useId, useLayoutEffect, useRef, type ReactNode } from "react";
import { useChartCover } from "@/nx/chart/cover";
import { getFrame, onFrame, type ChartFrame } from "@/nx/chart/runtime";
import { dockIn, ticketOpacity } from "../motion";
import { usePhone } from "../useMedia";
import { cx } from "../util";
import { isPhoneNow, useArrival, useLeaveGhost } from "./arrive";
import type { Place } from "./Ticket";
import s from "./stage.module.css";

type Data = { [key: `data-${string}`]: string | number | boolean | undefined };

export interface DockProps extends Data {
  /** The region's accessible name ("Sea, next four days"). */
  label: string;
  /** Where it stands along the bottom: left (default, the sea dock), right, or the full width between the gutters. */
  align?: "left" | "right" | "full";
  /** px (left or right docks); the sea dock is 840. Capped to the screen. */
  width?: number;
  /** The camera keeps the place above the dock (default true). */
  cover?: boolean;
  className?: string;
  children: ReactNode;
}

/**
 * A dock (v14 README §3): the low, wide panel along the chart's foot where time lives (the sea forecast, the alerts
 * delivery, Explore's time). It springs up 70 ms after the page's ticket and the camera keeps the place above it. On a
 * phone it is a block in the sheet after the ticket, ruled off above.
 */
export function Dock({ label, align = "left", width, cover = true, className, children, ...data }: DockProps) {
  const ref = useRef<HTMLElement>(null);
  const id = useId();
  const phone = usePhone();
  const here = useArrival(0);
  useLeaveGhost(ref, 0.97);
  useChartCover(`dock${id}`, ref, "bottom", !phone && cover);
  const style = width ? ({ "--dock-w": `${width}px` } as React.CSSProperties) : undefined;
  return (
    <motion.section
      ref={ref}
      className={cx(s.dock, className)}
      data-align={align}
      style={style}
      aria-label={label}
      initial={phone ? false : { opacity: 0, y: 28 }}
      animate={phone || here ? { opacity: 1, y: 0 } : { opacity: 0, y: 28 }}
      transition={{ default: dockIn, opacity: { ...ticketOpacity, delay: dockIn.delay } }}
      data-nx-dock=""
      {...data}
    >
      {children}
    </motion.section>
  );
}

export interface PassProps extends Data {
  /** The region's accessible name. */
  label: string;
  /** The place under the pass's notch (a caret on its top edge stands under it). */
  at?: Place | null;
  /** The stub on the right: the result (net to boat, the sample's value) and the one filled action. */
  stub: ReactNode;
  /** px; the trip's stub is 320, Research's 340. */
  stubWidth?: number;
  className?: string;
  stubClassName?: string;
  /** The main part: the equation and the figures, or the plot. */
  children: ReactNode;
}

/**
 * A pass (v14 README §3): the wide boarding pass along the bottom of the chart (the trip, a Research record). Its main
 * part holds the work, its perforated stub on the right the one result and the one filled action; a notch on its top
 * edge sits under its place. Springs up with the camera; on a phone the stub comes first and the main follows.
 */
export function Pass({ label, at, stub, stubWidth = 320, className, stubClassName, children, ...data }: PassProps) {
  const ref = useRef<HTMLElement>(null);
  const caretRef = useRef<HTMLSpanElement>(null);
  const id = useId();
  const phone = usePhone();
  const here = useArrival(0);
  useLeaveGhost(ref, 0.97);
  useChartCover(`pass${id}`, ref, "bottom", !phone);
  const lon = at?.lon ?? null, lat = at?.lat ?? null;
  useLayoutEffect(() => {
    const el = ref.current, c = caretRef.current;
    if (!el || !c) return;
    const place = (f: ChartFrame | null) => {
      if (!f || lon == null || lat == null || isPhoneNow()) {
        c.style.display = "none";
        return;
      }
      const p = f.project(lon, lat);
      const r = el.getBoundingClientRect();
      const l = p.x - r.left;
      const ok = l > 30 && l < r.width - 30 && p.y < r.top - 6;
      c.style.display = ok ? "" : "none";
      c.style.left = `${Math.round(l)}px`;
      el.style.transformOrigin = `${Math.round(Math.max(0, Math.min(r.width, l)))}px 0`;
    };
    place(getFrame());
    return onFrame(place);
  }, [lon, lat]);
  return (
    <motion.section
      ref={ref}
      className={cx(s.pass, className)}
      style={{ "--stub-w": `${stubWidth}px` } as React.CSSProperties}
      aria-label={label}
      initial={phone ? false : { opacity: 0, y: 28 }}
      animate={phone || here ? { opacity: 1, y: 0 } : { opacity: 0, y: 28 }}
      transition={{ default: dockIn, opacity: { ...ticketOpacity, delay: dockIn.delay } }}
      data-nx-pass=""
      {...data}
    >
      <span ref={caretRef} className={s.caret} aria-hidden="true" />
      <div className={s.passMain}>{children}</div>
      <div className={cx(s.passStub, stubClassName)}>{stub}</div>
    </motion.section>
  );
}
