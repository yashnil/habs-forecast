"use client";
import { useState, type ReactNode } from "react";
import { Drawer } from "vaul";
import { Icon } from "./Icon";
import { usePortalContainer } from "./Root";
import { cx } from "./util";
import s from "./MarginSheet.module.css";

export interface MarginSheetProps {
  /** Accessible name: "Today at Moss Landing". */
  label: string;
  /** What shows at the resting detent: the status, always visible above the bottom bar. */
  peek: ReactNode;
  children: ReactNode;
  /**
   * Detents, least visible first: px strings or fractions of the screen. The first is where the sheet
   * rests (the status is visible), the last is reading height.
   */
  detents?: Array<string | number>;
  /** Space kept free at the bottom for the destinations bar. */
  bottomInset?: number;
  /** Shows a close control (for transient sheets such as a readout). The margin itself never closes. */
  onClose?: () => void;
  className?: string;
}

/**
 * The phone margin (spec section 6): a bottom sheet resting at a meaningful detent, dragged up with
 * velocity (vaul). The chart stays live under it (non-modal). A "More"/"Less" control moves between detents
 * for keyboard and switch users, since a drag handle alone is not operable by keyboard.
 */
export function MarginSheet({ label, peek, children, detents = ["176px", 0.55, 0.94], onClose, className }: MarginSheetProps) {
  const [snap, setSnap] = useState<string | number | null>(detents[0]);
  const container = usePortalContainer();
  const at = Math.max(0, detents.indexOf(snap ?? detents[0]));
  const last = at === detents.length - 1;
  const step = () => setSnap(detents[last ? 0 : at + 1]);
  return (
    <Drawer.Root
      open
      modal={false}
      dismissible={false}
      snapPoints={detents}
      activeSnapPoint={snap}
      setActiveSnapPoint={setSnap}
      noBodyStyles
    >
      {/* the portal lands in the chart-mode scope; Root gets no container, so detents measure the window */}
      <Drawer.Portal container={container}>
        <Drawer.Content className={cx(s.sheet, className)} aria-describedby={undefined} data-detent={at}>
          <div className={s.top}>
            <Drawer.Handle className={s.handle} />
          </div>
          <Drawer.Title className={s.vh}>{label}</Drawer.Title>
          <div className={s.peekRow}>
            <div className={s.peek}>{peek}</div>
            <div className={s.controls}>
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
          </div>
          <div className={s.body} data-scroll={last || undefined}>
            {children}
          </div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
