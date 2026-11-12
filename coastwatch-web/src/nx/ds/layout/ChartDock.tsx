"use client";

import { useEffect, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { createPortal } from "react-dom";
import { Icon } from "../Icon";
import { cx } from "../util";
import { CHART_DOCK_ID, type ChartDockProps } from "./Desk";
import { STRIP_QUERY } from "./pane";
import s from "./Desk.module.css";

const MIN = 160;
const clampH = (h: number) => Math.round(Math.min(Math.max(MIN, h), Math.max(MIN, (typeof window === "undefined" ? 900 : window.innerHeight) * 0.6)));

/**
 * A full-width panel along the bottom of the chart pane (chart mode): a readout's detail chart gets width while
 * the chart above it keeps the context and refits into what is left. On a phone it reads in place, in the
 * margin sheet. `resizable` adds a separator on its top edge (drag, or Up/Down arrows).
 */
export function ChartDock({ label, height = 300, resizable, align = "pane", variant = "panel", onClose, className, children }: ChartDockProps) {
  // undefined until measured (the server and the first client render read in place), null on a phone
  const [slot, setSlot] = useState<HTMLElement | null | undefined>(undefined);
  const [h, setH] = useState(height);
  const [drag, setDrag] = useState<number | null>(null);
  useEffect(() => {
    const mq = window.matchMedia(STRIP_QUERY);
    const on = () => setSlot(mq.matches ? null : document.getElementById(CHART_DOCK_ID));
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  const key = (e: KeyboardEvent) => {
    const d = e.key === "ArrowUp" ? 32 : e.key === "ArrowDown" ? -32 : 0;
    if (!d) return;
    e.preventDefault();
    setH((v) => clampH(v + d));
  };
  // the height follows the hand as a line, and the chart refits once, on release
  const down = (e: ReactPointerEvent<HTMLDivElement>) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    setDrag(h);
  };
  const move = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (drag === null) return;
    setDrag(clampH(window.innerHeight - e.clientY));
  };
  const up = () => {
    if (drag === null) return;
    setH(drag);
    setDrag(null);
  };
  const body = (
    <section className={cx(s.dock, className)} aria-label={label} style={slot ? { height: h } : undefined} data-nx-chart-dock data-placed={slot ? "pane" : "inline"} data-align={align} data-variant={variant}>
      {resizable && slot ? (
        <div
          className={s.dockGrip}
          role="separator"
          aria-orientation="horizontal"
          aria-label={`Resize ${label}`}
          aria-valuenow={h}
          aria-valuemin={MIN}
          tabIndex={0}
          onKeyDown={key}
          onPointerDown={down}
          onPointerMove={move}
          onPointerUp={up}
          onPointerCancel={up}
          style={drag !== null ? { transform: `translateY(${h - drag}px)` } : undefined}
          data-dragging={drag !== null || undefined}
        />
      ) : null}
      <div className={s.dockBody}>{children}</div>
      {onClose ? (
        <button type="button" className={s.dockClose} onClick={onClose} aria-label={`Close ${label}`}>
          <Icon name="close" />
        </button>
      ) : null}
    </section>
  );
  return slot ? createPortal(body, slot) : body;
}
