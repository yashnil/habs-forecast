"use client";

import { useEffect, useRef, useState, type ComponentPropsWithoutRef, type ReactNode } from "react";
import { cx } from "../util";
import { SCROLL_CONDENSE } from "../motion";
import { usePageLayoutEffect, type PageMode, type PaneDetent } from "./pane";
import s from "./Desk.module.css";

type Data = { [key: `data-${string}`]: string | number | boolean | undefined };

export interface PageLayoutProps {
  mode: PageMode;
  /** Names this route's pane size on the device ("research", "coast"). */
  paneKey?: string;
  /** The pane's size until the reader chooses one. */
  defaultPane?: PaneDetent;
}

/**
 * Declares the page's layout (as <ChartScene> declares its view). Renders only a hidden marker, which is what
 * the stylesheet reads, so the server's HTML is already laid out as a desk.
 */
export function PageLayout({ mode, paneKey, defaultPane = "context" }: PageLayoutProps) {
  usePageLayoutEffect(mode, paneKey ?? null, defaultPane);
  return <span hidden data-nx-layout={mode} />;
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
    // hysteresis, so the header's own change of height never flips it back
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
  return (
    <header ref={ref} className={cx(s.header, sticky && s.sticky, className)} data-condensed={condense && small ? "" : undefined} data-nx-desk-header>
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

export interface ChartDockProps {
  /** Accessible name: "C-HARM along 36°48′ N". */
  label: string;
  /** Height in px on a wide screen (default 300). */
  height?: number;
  /** A separator on the top edge: drag, or Up/Down arrows. */
  resizable?: boolean;
  onClose?: () => void;
  className?: string;
  children: ReactNode;
}

export const CHART_DOCK_ID = "nx-chart-dock";
