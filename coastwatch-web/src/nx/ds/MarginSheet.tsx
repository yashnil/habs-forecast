"use client";
import { animate, motion, useDragControls, useMotionValue, useReducedMotion, type PanInfo } from "motion/react";
import { useEffect, useId, useLayoutEffect, useState, type CSSProperties, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Icon } from "./Icon";
import { sheet as sheetSpring } from "./motion";
import { usePortalContainer } from "./Root";
import { cx } from "./util";
import s from "./MarginSheet.module.css";

export interface MarginSheetProps {
  /** Accessible name: "Today at Moss Landing". */
  label: string;
  /** What shows at the resting detent above the body (optional: a page margin may let its own first lines rest). */
  peek?: ReactNode;
  children: ReactNode;
  /**
   * Detents, least visible first: px strings or fractions of the screen. The first is where the sheet
   * rests (the status is visible), the last is reading height.
   */
  detents?: Array<string | number>;
  /** The detent shown, controlled by the page (with `onDetentChange`); uncontrolled when absent. */
  detent?: number;
  onDetentChange?: (index: number) => void;
  /** Space at the bottom kept for the destinations bar, which lies over the sheet's lowest part. */
  bottomInset?: number;
  /** Rendered in place instead of portalled (the shell's margin, which must stack under its bars). */
  inline?: boolean;
  /**
   * Where the body scrolls: only at reading height (default), or at every detent, so a page can bring a new
   * part into view (a reading arriving) without raising the sheet fully.
   */
  scroll?: "reading" | "always";
  /** Shows a close control (for transient sheets such as a readout). The margin itself never closes. */
  onClose?: () => void;
  className?: string;
  /** Extra attributes for the sheet element (test ids). */
  sheetProps?: Record<`data-${string}`, string>;
}

const toPx = (d: string | number, h: number) => (typeof d === "number" ? d * h : parseFloat(d));

/**
 * The phone margin (spec section 6): a bottom sheet resting at a meaningful detent, dragged by its handle
 * row with velocity to the nearest detent (Motion's drag and the `sheet` spring). It is not a dialog: the
 * page and the chart stay live and reachable around it (no focus trap, nothing hidden from assistive tech). A
 * "More"/"Less" control moves between detents for keyboard and switch users. The first frame is drawn at the
 * resting detent (no slide in on arrival); reduced motion jumps between detents.
 */
export function MarginSheet({ label, peek, children, detents = ["176px", 0.55, 0.94], detent, onDetentChange, bottomInset = 0, inline, scroll = "reading", onClose, className, sheetProps }: MarginSheetProps) {
  const [own, setOwn] = useState(0);
  const at = Math.min(detents.length - 1, Math.max(0, detent ?? own));
  const container = usePortalContainer();
  const reduce = useReducedMotion();
  const titleId = useId();
  const controls = useDragControls();
  const [h, setH] = useState(() => (typeof window === "undefined" ? 844 : window.innerHeight));
  useEffect(() => {
    const on = () => setH(window.innerHeight);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  const heights = detents.map((d) => Math.min(h, Math.max(0, toPx(d, h))));
  const full = Math.max(...heights);
  // the sheet is as tall as its reading height and slides down by what each detent leaves hidden
  const offset = (i: number) => full - heights[i];
  const y = useMotionValue(offset(at));
  const last = at === detents.length - 1;
  const go = (i: number) => {
    if (detent === undefined) setOwn(i);
    onDetentChange?.(i);
  };
  useLayoutEffect(() => {
    const target = offset(at);
    if (reduce) {
      y.set(target);
      return;
    }
    const c = animate(y, target, sheetSpring);
    return () => c.stop();
    // heights change with the window; the detent with the page
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [at, full, reduce]);
  const onDragEnd = (_: unknown, info: PanInfo) => {
    // where the hand was heading: the nearest detent to the position a fling would carry the sheet to
    const aim = y.get() + info.velocity.y * 0.18;
    let best = 0;
    heights.forEach((_h, i) => {
      if (Math.abs(offset(i) - aim) < Math.abs(offset(best) - aim)) best = i;
    });
    if (best === at) animate(y, offset(at), reduce ? { duration: 0 } : sheetSpring);
    else go(best);
  };
  const step = () => go(last ? 0 : at + 1);
  const el = (
    <motion.section
      className={cx(s.sheet, className)}
      aria-labelledby={titleId}
      data-detent={at}
      style={{ y, height: full, "--sheet-inset": `${bottomInset}px` } as unknown as CSSProperties}
      drag="y"
      dragListener={false}
      dragControls={controls}
      dragConstraints={{ top: 0, bottom: offset(0) }}
      dragElastic={0.06}
      dragMomentum={false}
      onDragEnd={onDragEnd}
      {...sheetProps}
    >
      <div className={s.top} onPointerDown={(e) => controls.start(e)}>
        <span className={s.handle} aria-hidden />
        {!peek ? <Controls label={label} last={last} step={step} onClose={onClose} floating /> : null}
      </div>
      <h2 id={titleId} className={s.vh}>
        {label}
      </h2>
      {peek ? (
        <div className={s.peekRow} onPointerDown={(e) => controls.start(e)}>
          <div className={s.peek}>{peek}</div>
          <Controls label={label} last={last} step={step} onClose={onClose} />
        </div>
      ) : null}
      <div className={cx(s.body, !peek && s.bodyBare)} data-scroll={last || scroll === "always" || undefined}>
        {children}
      </div>
    </motion.section>
  );
  if (inline) return el;
  if (typeof document === "undefined") return null;
  return createPortal(el, container ?? document.body);
}

function Controls({ label, last, step, onClose, floating }: { label: string; last: boolean; step: () => void; onClose?: () => void; floating?: boolean }) {
  return (
    <div className={cx(s.controls, floating && s.floating)} onPointerDown={(e) => e.stopPropagation()}>
      <button type="button" className={s.more} onClick={step} aria-label={last ? `Show less of ${label}` : `Show more of ${label}`}>
        {last ? "Less" : "More"}
        <Icon name="chevron" size={20} direction={last ? "down" : "up"} />
      </button>
      {onClose ? (
        <button type="button" className={s.close} onClick={onClose} aria-label="Close">
          <Icon name="close" />
        </button>
      ) : null}
    </div>
  );
}
