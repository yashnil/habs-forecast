"use client";

import { animate, motion } from "motion/react";
import { useRef, useState } from "react";
import { pressSink, sweep, tickDraw } from "../motion";
import { cx } from "../util";
import b from "../Button.module.css";
import s from "./synth.module.css";

export interface FollowButtonProps {
  /** The port's name ("Fort Bragg"). */
  name: string;
  following: boolean;
  /** Called with the new state; the page writes it (src/nx/alerts/follow). */
  onChange: (next: boolean) => void;
  /**
   * `primary` (default): the filled "Follow Fort Bragg +" ticket, turning into "Following Fort Bragg ✓" on paper.
   * `toggle`: the quiet confirmed toggle that stands beside the screen's real primary once the port is followed
   * (design issue 1: Today at a followed port leads with the next useful action, not with a confirmed state).
   */
  variant?: "primary" | "toggle";
  disabled?: boolean;
  className?: string;
  "data-testid"?: string;
}

const Check = ({ draw }: { draw: boolean }) => (
  <svg className={s.ck} viewBox="0 0 16 12" aria-hidden="true">
    <motion.path d="M1.5 6.5l4 4 9-9" initial={draw ? { pathLength: 0 } : false} animate={{ pathLength: 1 }} transition={tickDraw} />
  </svg>
);

/**
 * Follow → Following (v14 README §3): paper sweeps across the button left to right (260 ms), then a tick draws in its
 * stub (340 ms). Pressing a followed port's button unfollows it at once. Reduced motion shows the final state.
 */
export function FollowButton({ name, following, onChange, variant = "primary", disabled, className, "data-testid": testid }: FollowButtonProps) {
  const sweepRef = useRef<HTMLSpanElement>(null);
  const [drawTick, setDrawTick] = useState(false);
  const [busy, setBusy] = useState(false);
  const press = async () => {
    if (disabled || busy) return;
    if (following) {
      setDrawTick(false);
      onChange(false);
      return;
    }
    const el = sweepRef.current;
    setBusy(true);
    if (el) await animate(el, { scaleX: [0, 1] }, sweep);
    setDrawTick(true);
    onChange(true);
    setBusy(false);
    if (el) el.style.transform = "scaleX(0)";
  };
  const toggle = variant === "toggle";
  const words = following ? (toggle ? "Following" : `Following ${name}`) : `Follow ${name}`;
  return (
    <motion.button
      type="button"
      className={cx(b.btn, b.stubbed, toggle ? s.followToggle : following ? b.secondary : b.primary, className)}
      aria-pressed={following}
      aria-label={toggle ? (following ? `Following ${name}. Stop following` : `Follow ${name}`) : undefined}
      aria-disabled={disabled || undefined}
      onClick={press}
      data-testid={testid}
      {...(disabled ? {} : pressSink)}
    >
      <span ref={sweepRef} className={s.sweep} aria-hidden="true" />
      <span className={b.words}>{words}</span>
      <span className={b.stub} data-g={following ? "✓" : "+"} aria-hidden="true">
        {following ? <Check draw={drawTick} /> : <span className={b.glyph}>+</span>}
      </span>
    </motion.button>
  );
}
