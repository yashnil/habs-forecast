"use client";
import { motion, useReducedMotion } from "motion/react";
import Link from "next/link";
import type { AriaAttributes, MouseEvent, ReactNode } from "react";
import { forwardRef } from "react";
import { ink, press } from "./motion";
import { Icon, type IconName } from "./Icon";
import { cx } from "./util";
import s from "./Button.module.css";

/** `text` is the v12 name for `quiet` and is kept as an alias. */
export type ButtonVariant = "primary" | "secondary" | "quiet" | "text";
export type ButtonSize = "m" | "l";

/** `data-*` attributes pass through to the element (test ids, tour anchors). */
type DataAttributes = { [key: `data-${string}`]: string | number | boolean | undefined };

interface Common extends AriaAttributes, DataAttributes {
  variant?: ButtonVariant;
  children: ReactNode;
  /** m: 48 px (default); l: 56 px, for the one action a view leads with. */
  size?: ButtonSize;
  icon?: IconName;
  /** An icon after the words (a chevron for "opens below", share for "leaves the page"). */
  iconEnd?: IconName;
  /** Disabled stays focusable (aria-disabled) so its reason can be read. */
  disabled?: boolean;
  /** Busy: the label becomes `busyLabel` (worded, never a spinner alone). */
  busy?: boolean;
  busyLabel?: string;
  full?: boolean;
  className?: string;
}

type AsButton = Common & { href?: undefined; type?: "button" | "submit"; onClick?: (e: MouseEvent<HTMLButtonElement>) => void };
type AsLink = Common & { href: string; external?: boolean; onClick?: (e: MouseEvent<HTMLAnchorElement>) => void };
export type ButtonProps = AsButton | AsLink;

/** A Next link that presses like every control: an address inside Neritic is a client navigation, so the
 *  chart pans and the margin swaps as on every other route change (never a full page load). */
const MotionLink = motion.create(Link);

/**
 * Three kinds and no more: primary (ink, one per view), secondary (ruled; hover is a wash and a heavier foot
 * rule, never an inversion), quiet (an underlined link). Two sizes, 48 and 56 px. Every control presses with
 * the `press` preset.
 */
export const Button = forwardRef<HTMLButtonElement | HTMLAnchorElement, ButtonProps>(function Button(props, ref) {
  const { variant: asked = "secondary", size = "m", children, icon, iconEnd, disabled, busy, busyLabel, full, className } = props;
  const variant = asked === "text" ? "quiet" : asked;
  const reduce = useReducedMotion();
  const inert = disabled || busy;
  const cls = cx(s.btn, s[variant], size === "l" && s.large, full && s.full, className);
  const body = (
    <>
      {icon && variant !== "quiet" ? <Icon name={icon} size={20} /> : null}
      <span>{busy && busyLabel ? busyLabel : children}</span>
      {iconEnd ? <Icon name={iconEnd} size={20} /> : null}
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
  const tap = inert ? {} : press;
  if (props.href !== undefined) {
    const { href, external, onClick } = props;
    if (!external && !inert && href.startsWith("/")) {
      return (
        <MotionLink ref={ref as React.Ref<HTMLAnchorElement>} className={cls} href={href} onClick={onClick} {...tap} {...aria}>
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
        {...tap}
        {...aria}
      >
        {body}
      </motion.a>
    );
  }
  const { type = "button", onClick } = props;
  return (
    <motion.button ref={ref as React.Ref<HTMLButtonElement>} type={type} className={cls} onClick={(e) => (inert ? e.preventDefault() : onClick?.(e))} {...tap} {...aria}>
      {body}
    </motion.button>
  );
});
