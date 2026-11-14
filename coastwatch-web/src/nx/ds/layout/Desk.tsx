"use client";

import { useEffect, useLayoutEffect, useRef, useState, type ComponentPropsWithoutRef, type ReactNode } from "react";
import { cx } from "../util";
import { SCROLL_CONDENSE } from "../motion";
import { usePageLayoutEffect, type PageMode, type PaneDetent } from "./pane";
import { Sheet } from "../stage/Sheet";
import { setPanelsPage } from "../panels/store";
import s from "./Desk.module.css";

type Data = { [key: `data-${string}`]: string | number | boolean | undefined };

export interface PageLayoutProps {
  mode: PageMode;
  /** Names this route's pane size on the device ("research", "coast"). */
  paneKey?: string;
  /** The pane's size until the reader chooses one. */
  defaultPane?: PaneDetent;
  /**
   * What the chart shows on this page, in a few words ("Monterey Wharf", "Whale zone 4"): named on the phone strip's
   * bar and on the collapsed rail as "Chart · …". Without it they say only "Chart": never a name the chart is not showing.
   */
  chartLabel?: string;
  /** v14 stage pages: the share of a phone's screen the chart strip keeps over the resting sheet (ds Sheet `map`). */
  phoneMap?: number;
  /**
   * v15.1: the page's floating objects (its ticket, docks and pass) become free panels on a wide screen, remembered on
   * this device under this name ("coast", "trip", "today"): moved by their grips, resized, collapsed, reset together
   * (src/nx/ds/panels). Omit it and they stand where the page puts them.
   */
  panels?: string;
}

/**
 * Declares the page's layout (as <ChartScene> declares its view). Renders only a hidden marker, which is what
 * the stylesheet reads, so the server's HTML is already laid out as a desk.
 */
export function PageLayout({ mode, paneKey, defaultPane = "context", chartLabel, phoneMap, panels }: PageLayoutProps) {
  usePageLayoutEffect(mode, paneKey ?? null, defaultPane, chartLabel ?? null);
  useLayoutEffect(() => {
    if (!panels) return;
    setPanelsPage(panels);
    return () => setPanelsPage(null);
  }, [panels]);
  return (
    <>
      <span hidden data-nx-layout={mode} />
      {phoneMap != null ? <Sheet map={phoneMap} /> : null}
    </>
  );
}

/** The desk: the page's work. Header, body (main and aside), foot. */
export function Desk({ className, children, ...rest }: ComponentPropsWithoutRef<"div">) {
  return (
    <div className={cx(s.desk, className)} data-nx-desk {...rest}>
      {children}
    </div>
  );
}

/** The nearest scrolling ancestor (the desk's column on a wide screen, the page on a phone). */
function scrollerOf(el: HTMLElement | null): HTMLElement | null {
  for (let p = el?.parentElement ?? null; p; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if (oy === "auto" || oy === "scroll") return p;
  }
  return null;
}

/** True once the nearest scroller has moved past `after` px (64 by default): a header condenses, never animates. */
export function useScrollCondense<T extends HTMLElement>(after = SCROLL_CONDENSE): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [on, setOn] = useState(false);
  useEffect(() => {
    const sc = scrollerOf(ref.current);
    if (!sc) return;
    // hysteresis: it lets go well before the top, and a caller that changes height keeps its room (DeskHeader),
    // so the scroller's position never moves under the reader because of the change
    const read = () => setOn((was) => (was ? sc.scrollTop > after * 0.4 : sc.scrollTop > after));
    read();
    sc.addEventListener("scroll", read, { passive: true });
    return () => sc.removeEventListener("scroll", read);
  }, [after]);
  return [ref, on];
}

export interface DeskHeaderProps {
  /** What never scrolls away: port · fishery · "Hypothetical", or "Measured at Monterey Wharf". */
  context?: ReactNode;
  title: ReactNode;
  /** Attributes for the <h1> (test ids). */
  titleProps?: Data;
  /** The status sentence under the title; folded away once the header condenses. */
  status?: ReactNode;
  /** Controls at the header's end (one primary at most, the chart toggle). */
  actions?: ReactNode;
  /** Stays at the top of the desk while it scrolls (default true). */
  sticky?: boolean;
  /** Shrinks to one line after 64 px of scroll (default true). */
  condense?: boolean;
  /** Kept in the header when it condenses (the fishery, the official status): what must never scroll away. */
  pinned?: ReactNode;
  className?: string;
  /** Under the status; folded away with it once the header condenses. */
  children?: ReactNode;
}

export function DeskHeader({ context, title, titleProps, status, actions, sticky = true, condense = true, pinned, className, children }: DeskHeaderProps) {
  const [ref, small] = useScrollCondense<HTMLElement>();
  const on = condense && small;
  // Condensing never moves the page. The header keeps the room it had: what it loses in height it carries as a
  // margin under itself, so nothing below shifts, the scroller's position is not adjusted, and the header cannot
  // flip back and forth across its own threshold however much taller it was (a 90 px change on Research).
  // The room is given in the same commit as the change, from the header's two heights. Both are measured while
  // the header is open: its condensed height by putting it in that state for one synchronous read (never
  // painted), so the page is never laid out, even for a moment, with the room missing or wrong.
  const [full, setFull] = useState(0);
  const [slim, setSlim] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el || on) return;
    const read = () => {
      const h = el.offsetHeight;
      // (its own attribute, styled as the condensed state: the real one is never touched, so nothing that
      // watches or styles `data-condensed` sees a state the header is not in)
      el.setAttribute("data-measuring", "");
      const c = el.offsetHeight;
      el.removeAttribute("data-measuring");
      setFull((v) => (Math.abs(v - h) < 0.5 ? v : h));
      setSlim((v) => (Math.abs(v - c) < 0.5 ? v : c));
    };
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [on, ref]);
  const hold = on && full && slim ? Math.max(0, full - slim) : 0;
  return (
    <header ref={ref} className={cx(s.header, sticky && s.sticky, className)} data-condensed={on ? "" : undefined} style={hold ? { marginBottom: hold } : undefined} data-nx-desk-header>
      <div className={s.headMain}>
        {context ? <p className={s.context}>{context}</p> : null}
        <h1 className={s.title} {...titleProps}>
          {title}
        </h1>
        {status ? <div className={s.status}>{status}</div> : null}
        {children ? <div className={s.more}>{children}</div> : null}
        {pinned ? <div className={s.pinned}>{pinned}</div> : null}
      </div>
      {actions ? <div className={s.actions}>{actions}</div> : null}
    </header>
  );
}

export interface DeskBodyProps {
  /** main-aside: the primary visual beside a column of inputs; main-full: the primary visual across, the aside under it. */
  layout?: "main-aside" | "main-full";
  /** Which side the aside takes beside the main (wide screens). Write the aside first in the page for "start": reading order is the page's order. */
  aside?: "start" | "end";
  className?: string;
  children: ReactNode;
}

export function DeskBody({ layout = "main-aside", aside = "end", className, children }: DeskBodyProps) {
  return (
    <div className={cx(s.body, className)} data-layout={layout} data-aside={aside}>
      {children}
    </div>
  );
}

export function DeskMain({ className, children, ...rest }: ComponentPropsWithoutRef<"div">) {
  return (
    <div className={cx(s.main, className)} {...rest}>
      {children}
    </div>
  );
}

export interface DeskAsideProps extends Omit<ComponentPropsWithoutRef<"section">, "aria-label"> {
  /** Accessible name: "Trip inputs". */
  label: string;
  /** Holds its place beside the main while the desk scrolls (wide screens). */
  sticky?: boolean;
}

export function DeskAside({ label, sticky, className, children, ...rest }: DeskAsideProps) {
  return (
    <section className={cx(s.aside, sticky && s.asideSticky, className)} aria-label={label} {...rest}>
      {children}
    </section>
  );
}

export function DeskFoot({ className, children, ...rest }: ComponentPropsWithoutRef<"footer">) {
  return (
    <footer className={cx(s.foot, className)} {...rest}>
      {children}
    </footer>
  );
}

/**
 * A bar that stays at the bottom of a desk page on a phone (above the destinations), for the one result or the
 * one action that must stay in reach. On wider screens it is an ordinary block where it stands.
 */
export function DeskSticky({ className, children, ...rest }: ComponentPropsWithoutRef<"div">) {
  return (
    <div className={cx(s.stickyBottom, className)} data-nx-desk-sticky {...rest}>
      {children}
    </div>
  );
}

/**
 * Stays at the foot of the margin column (or of a desk column) while the page scrolls under it: the legend for
 * a reading that is open, so its marks stay explained however far the reader scrolls. An ordinary block on a
 * phone sheet, where it would crowd the resting detent.
 */
export function MarginSticky({ edge = "bottom", className, children, ...rest }: ComponentPropsWithoutRef<"div"> & { /** bottom (default): at the column's foot; top: at its head, for a legend above the reading it explains */ edge?: "top" | "bottom" }) {
  return (
    <div className={cx(s.marginSticky, className)} data-nx-margin-sticky data-edge={edge} {...rest}>
      {children}
    </div>
  );
}

export interface ChartDockProps {
  /** Accessible name: "C-HARM along 36°48′ N". */
  label: string;
  /** Height in px on a wide screen (default 300). */
  height?: number;
  /** A separator on the top edge: drag, or Up/Down arrows. */
  resizable?: boolean;
  /**
   * frame: the dock's content starts and ends where the chart's frame does (a time rail whose ends line up with
   * the chart's edges); pane (default): the pane's own padding.
   */
  align?: "pane" | "frame";
  /** rail: a thin strip with a hairline instead of the heavy rule and no vertical padding (a time rail). */
  variant?: "panel" | "rail";
  onClose?: () => void;
  className?: string;
  children: ReactNode;
}

export const CHART_DOCK_ID = "nx-chart-dock";
