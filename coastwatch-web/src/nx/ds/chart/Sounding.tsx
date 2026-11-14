"use client";
import { AnimatePresence, motion } from "motion/react";
import type { ReactNode } from "react";
import { markFor, type Kind } from "../kinds";
import { Mark } from "../Mark";
import { register, registerFrom, registerTo } from "../motion";
import { cx } from "../util";
import { anchorAlign, anchorGap, anchorTranslate, type Anchor } from "./place";
import s from "./chart.module.css";

export interface SoundingProps {
  /** Screen position of the measured place inside the chart's overlay, in px (from the chart host). */
  x: number;
  y: number;
  /** The figure, already at its honest precision (T15): "6.6", "9.8", "0.69". */
  value: string;
  /** Unit and qualifier, set at 0.46 em: "ft at 14 s", "of 30 ppm". */
  unit?: string;
  /** Line above the figure: what it is. */
  label?: ReactNode;
  /** Line under the figure: kind, source, date (T1). */
  caption?: ReactNode;
  kind: Kind;
  /** Where the type sits relative to the leader's end. */
  anchor?: Anchor;
  /** Offset from the place to the type. Null sets the type at the place without a leader. */
  leader?: { dx: number; dy: number } | null;
  /** 1 is kept longest. The host drops by priority and sets `dropped`. */
  priority?: number;
  dropped?: boolean;
  /** The 3 px knockout halo; off only on plain paper. */
  halo?: boolean;
  size?: "lg" | "md";
  /** Past its usual update: the figure is set lighter and a step smaller with "(old)" beside it, so an old reading never leads the chart (T17). */
  stale?: boolean;
  /** Opens the provenance sheet; the figure becomes a button. */
  onActivate?: () => void;
  /** Full spoken form, when the visible lines are not enough. */
  ariaLabel?: string;
  /** Width the type may take before wrapping, in px. */
  width?: number;
  /**
   * `point` (default): drawn from `x`, `y` with its own leader, for a plain overlay. `box`: the type in
   * normal flow at its own size, no leader or point, for a host that measures each box and places it
   * itself (the chart's marks layer, which draws the leader and the point from its collision pass).
   */
  placement?: "point" | "box";
}

/**
 * A figure placed where it was measured (principle 2), with unit, kind mark and a leader line.
 * Presentational: the chart host owns projection and collisions.
 */
export function Sounding({ x, y, value, unit, label, caption, kind, anchor = "se", leader = null, priority = 5, dropped = false, halo = true, size = "lg", stale, onActivate, ariaLabel, width = 260, placement = "point" }: SoundingProps) {
  const mark = markFor(kind);
  const box = placement === "box";
  if (box) leader = null;
  const lx = leader?.dx ?? 0;
  const ly = leader?.dy ?? 0;
  const gap = leader ? anchorGap(anchor) : { x: 0, y: 0 };
  const align = anchorAlign(anchor);
  const Tag = onActivate ? "button" : "div";
  return (
    <AnimatePresence>
      {dropped ? null : (
        <motion.div
          className={cx(box ? s.boxed : s.placed, halo && s.halo)}
          style={box ? { maxWidth: width } : { left: x, top: y, zIndex: 20 - priority }}
          data-priority={priority}
          initial={registerFrom}
          animate={registerTo}
          exit={registerFrom}
          transition={register}
        >
          {leader ? (
            <svg className={s.leader} width="1" height="1" aria-hidden focusable="false">
              <line x1={0} y1={0} x2={lx} y2={ly} className={s.leaderHalo} />
              <line x1={0} y1={0} x2={lx} y2={ly} className={s.leaderLine} />
              <PlaceMark mark={mark} />
            </svg>
          ) : null}
          <Tag
            className={cx(s.block, s.sounding, s[size], stale && s.staleFig, onActivate && s.activate)}
            data-stale={stale || undefined}
            style={box ? { textAlign: align } : { left: lx + gap.x, top: ly + gap.y, transform: anchorTranslate(anchor), textAlign: align, width }}
            onClick={onActivate}
            type={onActivate ? "button" : undefined}
            aria-label={ariaLabel}
          >
            {label ? <span className={s.lbl}>{label}</span> : null}
            <span className={s.fig}>
              {value}
              {unit ? <em className={s.unit}>{unit}</em> : null}
              {stale ? (
                <span className={s.figOld}>
                  {" "}
                  <Mark kind="stale" />
                </span>
              ) : null}
            </span>
            {caption ? <span className={s.cap}>{caption}</span> : null}
          </Tag>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/** The mark at the measured place: solid for measured, hollow for model, a ring for official. */
function PlaceMark({ mark }: { mark: ReturnType<typeof markFor> }) {
  if (mark === "measured") {
    return (
      <>
        <circle cx={0} cy={0} r={6.5} className={s.dotHalo} />
        <circle cx={0} cy={0} r={4.5} className={s.dotSolid} />
      </>
    );
  }
  return (
    <>
      <circle cx={0} cy={0} r={7} className={s.dotHalo} />
      <circle cx={0} cy={0} r={4.5} className={s.dotHollow} />
      {mark === "official" ? <circle cx={0} cy={0} r={1.5} className={s.dotSolid} /> : null}
    </>
  );
}
