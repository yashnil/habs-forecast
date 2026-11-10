"use client";
import { motion, useReducedMotion } from "motion/react";
import type { MouseEvent, ReactNode } from "react";
import { forwardRef } from "react";
import { ink, press } from "./motion";
import { Icon, type IconName } from "./Icon";
import { cx } from "./util";
import s from "./Button.module.css";

export type ButtonVariant = "primary" | "secondary" | "text";

interface Common {
  variant?: ButtonVariant;
  children: ReactNode;
  icon?: IconName;
  /** Disabled stays focusable (aria-disabled) so its reason can be read. */
  disabled?: boolean;
  /** Busy: the label becomes `busyLabel` (worded, never a spinner alone). */
  busy?: boolean;
  busyLabel?: string;
  full?: boolean;
  className?: string;
  "aria-label"?: string;
  "aria-describedby"?: string;
  "aria-controls"?: string;
  "aria-expanded"?: boolean;
}

type AsButton = Common & { href?: undefined; type?: "button" | "submit"; onClick?: (e: MouseEvent<HTMLButtonElement>) => void };
type AsLink = Common & { href: string; external?: boolean; onClick?: (e: MouseEvent<HTMLAnchorElement>) => void };
export type ButtonProps = AsButton | AsLink;

/**
 * Three kinds and no more: primary (ink, one per view), secondary (ruled), text (an underlined link).
 * Every control presses with the `press` preset.
 */
export const Button = forwardRef<HTMLButtonElement | HTMLAnchorElement, ButtonProps>(function Button(props, ref) {
  const { variant = "secondary", children, icon, disabled, busy, busyLabel, full, className } = props;
  const reduce = useReducedMotion();
  const inert = disabled || busy;
  const cls = cx(s.btn, s[variant], full && s.full, className);
  const body = (
    <>
      {icon && variant !== "text" ? <Icon name={icon} size={20} /> : null}
      <span>{busy && busyLabel ? busyLabel : children}</span>
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
  const aria = {
    "aria-label": props["aria-label"],
    "aria-describedby": props["aria-describedby"],
    "aria-controls": props["aria-controls"],
    "aria-expanded": props["aria-expanded"],
    "aria-disabled": disabled ? true : undefined,
    "aria-busy": busy ? true : undefined,
  };
  const tap = inert ? {} : press;
  if (props.href !== undefined) {
    const { href, external, onClick } = props;
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
