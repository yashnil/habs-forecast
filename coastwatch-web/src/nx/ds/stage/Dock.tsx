"use client";

import { motion } from "motion/react";
import { useEffect, useId, useLayoutEffect, useRef, type ReactNode } from "react";
import { useChartCover } from "@/nx/chart/cover";
import { getFrame, onFrame, type ChartFrame } from "@/nx/chart/runtime";
import { dockIn, ticketOpacity } from "../motion";
import { fitBeside } from "../panels/geometry";
import { notePanel } from "../panels/store";
import { restingOthers, stageArea, usePanel } from "../panels/usePanel";
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
  /** Stand in the flow of its container instead of floating on the chart (the catalog). */
  inline?: boolean;
  /** On a page whose panels are free (PageLayout `panels`), its key there (default: from the label). false: it stays put. */
  panel?: boolean | string;
  className?: string;
  children: ReactNode;
}

const slug = (t: string) => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "dock";

/**
 * The default layout of a resting dock on a page of free panels: along the foot it gives up width to stop short of a
 * resting ticket beside it (never a moved one, so moving the ticket never moves the dock), down to 320 px.
 */
function useRestingFit(ref: React.RefObject<HTMLElement | null>, key: string, active: boolean, natural: number) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!active) {
      notePanel(key, { rest: null });
      return;
    }
    let raf = 0;
    const fit = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const area = stageArea(el);
        const g = area.left;
        const want = Math.min(natural, area.right - area.left);
        const w = fitBeside({ x: el.offsetLeft, y: el.offsetTop, w: want, h: el.offsetHeight }, restingOthers(key), g, 320);
        el.style.width = w < want ? `${w}px` : "";
        notePanel(key, { rest: { x: el.offsetLeft, y: el.offsetTop, w: el.offsetWidth, h: el.offsetHeight } });
      });
    };
    fit();
    const off = onFrame(fit);
    window.addEventListener("resize", fit);
    const ro = new ResizeObserver(fit);
    ro.observe(el);
    return () => {
      off();
      window.removeEventListener("resize", fit);
      ro.disconnect();
      cancelAnimationFrame(raf);
      if (el.isConnected && !el.hasAttribute("data-panel-moved")) el.style.width = "";
    };
  }, [ref, key, active, natural]);
}

/**
 * A dock (v14 README §3): the low, wide panel along the chart's foot where time lives (the sea forecast, the alerts
 * delivery, Explore's time). It springs up 70 ms after the page's ticket and the camera keeps the place above it. On a
 * phone it is a block in the sheet after the ticket, ruled off above.
 */
export function Dock({ label, align = "left", width, cover = true, inline = false, panel: panelOpt = true, className, children, ...data }: DockProps) {
  const ref = useRef<HTMLElement>(null);
  const id = useId();
  const phone = usePhone();
  const here = useArrival(0);
  useLeaveGhost(ref, 0.97);
  const key = typeof panelOpt === "string" ? panelOpt : slug(label);
  const panel = usePanel(ref, { key, kind: "dock", name: label, enabled: panelOpt !== false && !inline, limits: { minW: 320, maxW: 1200, minH: 120, maxH: 2000 }, resizeHeight: true, restCover: cover ? "bottom" : undefined });
  useChartCover(`dock${id}`, ref, "bottom", !phone && cover && !inline && !panel.free);
  useRestingFit(ref, key, panel.on && !panel.moved && !panel.resized && !panel.collapsed && align === "left", width ?? 840);
  const style = width ? ({ "--dock-w": `${width}px` } as React.CSSProperties) : undefined;
  return (
    <motion.section
      ref={ref}
      {...panel.attrs}
      className={cx(s.dock, className, panel.attrs.className)}
      data-align={align}
      data-inline={inline || undefined}
      style={style}
      aria-label={label}
      initial={phone || inline ? false : { opacity: 0, y: 28 }}
      animate={phone || inline || here ? { opacity: 1, y: 0 } : { opacity: 0, y: 28 }}
      transition={{ default: dockIn, opacity: { ...ticketOpacity, delay: dockIn.delay } }}
      data-nx-dock=""
      {...data}
    >
      {panel.chrome}
      {children}
      {panel.resizer}
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
  /** Stand in the flow of its container instead of floating on the chart (the catalog). */
  inline?: boolean;
  /** On a page whose panels are free (PageLayout `panels`), its key there (default "pass"). false: it stays put. */
  panel?: boolean | string;
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
export function Pass({ label, at, stub, stubWidth = 320, inline = false, panel: panelOpt = true, className, stubClassName, children, ...data }: PassProps) {
  const ref = useRef<HTMLElement>(null);
  const caretRef = useRef<HTMLSpanElement>(null);
  const id = useId();
  const phone = usePhone();
  const here = useArrival(0);
  useLeaveGhost(ref, 0.97);
  const panel = usePanel(ref, { key: typeof panelOpt === "string" ? panelOpt : "pass", kind: "pass", name: label, enabled: panelOpt !== false && !inline, limits: { minW: 640, maxW: 4000, minH: 160, maxH: 2000 }, resizeHeight: true, restCover: "bottom" });
  useChartCover(`pass${id}`, ref, "bottom", !phone && !inline && !panel.free);
  const lon = at?.lon ?? null, lat = at?.lat ?? null;
  useLayoutEffect(() => {
    const el = ref.current, c = caretRef.current;
    if (!el || !c) return;
    const place = (f: ChartFrame | null) => {
      if (!f || lon == null || lat == null || isPhoneNow() || inline) {
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
  }, [lon, lat, inline]);
  return (
    <motion.section
      ref={ref}
      {...panel.attrs}
      className={cx(s.pass, className, panel.attrs.className)}
      style={{ "--stub-w": `${stubWidth}px` } as React.CSSProperties}
      data-inline={inline || undefined}
      aria-label={label}
      initial={phone || inline ? false : { opacity: 0, y: 28 }}
      animate={phone || inline || here ? { opacity: 1, y: 0 } : { opacity: 0, y: 28 }}
      transition={{ default: dockIn, opacity: { ...ticketOpacity, delay: dockIn.delay } }}
      data-nx-pass=""
      {...data}
    >
      {panel.chrome}
      <span ref={caretRef} className={s.caret} aria-hidden="true" />
      <div className={s.passMain}>{children}</div>
      <div className={cx(s.passStub, stubClassName)}>{stub}</div>
      {panel.resizer}
    </motion.section>
  );
}
