"use client";
import { motion, useReducedMotion } from "motion/react";
import Link from "next/link";
import type { AriaAttributes, MouseEvent, ReactNode } from "react";
import { forwardRef } from "react";
import { ink, pressSink } from "./motion";
import { Icon, type IconName } from "./Icon";
import { cx } from "./util";
import s from "./Button.module.css";

/**
 * `primary`: ultramarine, one per screen. `ink`: the dark variant of the primary (an action set on a coloured ground).
 * `secondary`: paper with an ink edge (also a confirmed state). `quiet`: ultramarine words with a dashed underline.
 * `text` is the v12 name for `quiet`, kept as an alias.
 */
export type ButtonVariant = "primary" | "ink" | "secondary" | "quiet" | "text";
export type ButtonSize = "m" | "l";
/** What the stub's glyph says will happen: → open, ↓ keep, ↑ send, + follow or add, × remove, ✓ done. */
export type ButtonGlyph = "→" | "↓" | "↑" | "+" | "×" | "✓" | "↗";

/** `data-*` attributes pass through to the element (test ids, tour anchors). */
type DataAttributes = { [key: `data-${string}`]: string | number | boolean | undefined };

interface Common extends AriaAttributes, DataAttributes {
  variant?: ButtonVariant;
  children: ReactNode;
  /** m: 46 px (default); l: 50 px. Phone targets are 50 px either way. */
  size?: ButtonSize;
  icon?: IconName;
  /** An icon after the words (a chevron for "opens below"). On a primary it takes the stub's place. */
  iconEnd?: IconName;
  /**
   * The stub's glyph (primary and ink buttons are small tickets: label, a perforation, a stub whose glyph says what
   * happens). Default → on primary and ink; none on secondary unless given. null: no stub.
   */
  glyph?: ButtonGlyph | null;
  /** Disabled stays focusable (aria-disabled) so its reason can be read. */
  disabled?: boolean;
  /** Busy: the label becomes `busyLabel` (worded, never a spinner alone). */
  busy?: boolean;
  busyLabel?: string;
  full?: boolean;
  className?: string;
  title?: string;
}

type AsButton = Common & { href?: undefined; type?: "button" | "submit"; onClick?: (e: MouseEvent<HTMLButtonElement>) => void };
type AsLink = Common & { href: string; external?: boolean; onClick?: (e: MouseEvent<HTMLAnchorElement>) => void };
export type ButtonProps = AsButton | AsLink;

/** A Next link that presses like every control: an address inside Neritic is a client navigation. */
const MotionLink = motion.create(Link);

/**
 * Buttons (v14 README §3). A primary is a small ticket: its words, a dashed perforation, and a stub whose glyph says
 * what happens; hovering slides the glyph (120 ms), pressing sinks the ticket on a spring. One filled button per screen.
 */
export const Button = forwardRef<HTMLButtonElement | HTMLAnchorElement, ButtonProps>(function Button(props, ref) {
  const { variant: asked = "secondary", size = "m", children, icon, iconEnd, glyph: askedGlyph, disabled, busy, busyLabel, full, className, title } = props;
  const variant = asked === "text" ? "quiet" : asked;
  const reduce = useReducedMotion();
  const inert = disabled || busy;
  const filled = variant === "primary" || variant === "ink";
  const external = props.href !== undefined && !!props.external;
  const glyph = askedGlyph === null ? null : (askedGlyph ?? (filled && !iconEnd ? (external ? "↗" : "→") : null));
  const stub = variant !== "quiet" && (glyph || (filled && iconEnd));
  const cls = cx(s.btn, s[variant], size === "l" && s.large, full && s.full, stub && s.stubbed, className);
  const body = (
    <>
      {icon && variant !== "quiet" ? <Icon name={icon} size={20} /> : null}
      <span className={s.words}>{busy && busyLabel ? busyLabel : children}</span>
      {variant === "quiet" && external ? <span aria-hidden="true">{" ↗"}</span> : null}
      {stub ? (
        <span className={s.stub} data-g={glyph ?? ""} aria-hidden="true">
          {glyph ? <span className={s.glyph}>{glyph}</span> : iconEnd ? <Icon name={iconEnd} size={20} /> : null}
        </span>
      ) : iconEnd ? (
        <Icon name={iconEnd} size={20} />
      ) : null}
      {busy ? (
        <motion.span
          aria-hidden
          className={s.busyLine}
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={reduce ? { duration: 0 } : { ...ink(1.6), repeat: Infinity, repeatType: "loop" }}
        />
      ) : null}
    </>
  );
  // every aria-* and data-* prop passes through; disabled and busy set their own states
  const aria: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(props)) if (k.startsWith("aria-") || k.startsWith("data-")) aria[k] = v;
  if (disabled) aria["aria-disabled"] = true;
  if (busy) aria["aria-busy"] = true;
  const tap = inert || variant === "quiet" ? {} : pressSink;
  if (props.href !== undefined) {
    const { href, onClick } = props;
    if (!external && !inert && href.startsWith("/")) {
      return (
        <MotionLink ref={ref as React.Ref<HTMLAnchorElement>} className={cls} href={href} onClick={onClick} title={title} {...tap} {...aria}>
          {body}
        </MotionLink>
      );
    }
    return (
      <motion.a
        ref={ref as React.Ref<HTMLAnchorElement>}
        className={cls}
        href={inert ? undefined : href}
        target={external ? "_blank" : undefined}
        rel={external ? "noreferrer" : undefined}
        onClick={(e) => (inert ? e.preventDefault() : onClick?.(e))}
        role={inert ? "link" : undefined}
        tabIndex={inert ? 0 : undefined}
        title={title}
        {...tap}
        {...aria}
      >
        {body}
      </motion.a>
    );
  }
  const { type = "button", onClick } = props;
  return (
    <motion.button ref={ref as React.Ref<HTMLButtonElement>} type={type} className={cls} title={title} onClick={(e) => (inert ? e.preventDefault() : onClick?.(e))} {...tap} {...aria}>
      {body}
    </motion.button>
  );
});
