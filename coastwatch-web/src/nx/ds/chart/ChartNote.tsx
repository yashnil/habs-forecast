"use client";
import { AnimatePresence, motion } from "motion/react";
import type { ReactNode } from "react";
import { Icon } from "../Icon";
import { register, registerFrom, registerTo } from "../motion";
import { cx } from "../util";
import { anchorAlign, anchorTranslate, type Anchor } from "./place";
import s from "./chart.module.css";

export type ChartNoteVariant = "water" | "place" | "port" | "note" | "regulation";

export interface ChartNoteProps {
  x: number;
  y: number;
  children: ReactNode;
  /**
   * Letter style carries class (principle 3): water names italic serif, places upright sans, a port with its
   * circled dot, a note in the interface face, regulation in the one foreign hue.
   */
  variant?: ChartNoteVariant;
  anchor?: Anchor;
  /** Degrees; water names follow the feature, regulation follows its line. */
  rotate?: number;
  /** Water names only: 15 to 22 px by the size of the feature. */
  size?: 15 | 16 | 18 | 20 | 22;
  priority?: number;
  dropped?: boolean;
  halo?: boolean;
  /** Notes wrap at this width (45 to 70 characters is the reading range; on the chart, shorter). */
  width?: number;
}

/** A short note set on the water with a knockout halo; it drops by priority when space is short. */
export function ChartNote({ x, y, children, variant = "note", anchor = "e", rotate = 0, size = 18, priority = 5, dropped = false, halo = true, width }: ChartNoteProps) {
  const align = anchorAlign(anchor);
  // the name clears its symbol on the side it is set
  const clear = variant === "port" ? 14 : variant === "place" ? 8 : 0;
  const dx = align === "right" ? -clear : align === "left" ? clear : 0;
  return (
    <AnimatePresence>
      {dropped ? null : (
        <motion.div
          className={cx(s.placed, halo && s.halo)}
          style={{ left: x, top: y, zIndex: 20 - priority }}
          data-priority={priority}
          initial={registerFrom}
          animate={registerTo}
          exit={registerFrom}
          transition={register}
        >
          {variant === "port" ? (
            <svg className={s.leader} width="1" height="1" aria-hidden focusable="false">
              <circle cx={0} cy={0} r={9.5} className={s.dotHalo} />
              <circle cx={0} cy={0} r={7} className={s.portRing} />
              <circle cx={0} cy={0} r={2.5} className={s.dotSolid} />
            </svg>
          ) : variant === "place" ? (
            <svg className={s.leader} width="1" height="1" aria-hidden focusable="false">
              <circle cx={0} cy={0} r={4.5} className={s.dotHalo} />
              <circle cx={0} cy={0} r={2.75} className={s.dotSolid} />
            </svg>
          ) : null}
          <div
            className={cx(s.block, s.note, s[variant])}
            style={{
              left: dx,
              top: 0,
              transform: `${anchorTranslate(anchor)}${rotate ? ` rotate(${rotate}deg)` : ""}`,
              transformOrigin: "0 50%",
              textAlign: align,
              width,
              fontSize: variant === "water" ? `${size}px` : undefined,
              whiteSpace: width ? "normal" : "nowrap",
            }}
          >
            {children}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/** Re-exported for hosts that print a port name off the chart (lists, the margin). */
export function PortGlyph({ size = 20 }: { size?: number }) {
  return <Icon name="port" size={size} />;
}
