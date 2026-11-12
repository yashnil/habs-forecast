"use client";
import { motion } from "motion/react";
import { forwardRef, type AriaAttributes, type MouseEvent } from "react";
import { Icon, type IconName } from "./Icon";
import { press } from "./motion";
import { cx } from "./util";
import s from "./controls.module.css";

type Data = { [key: `data-${string}`]: string | number | boolean | undefined };

export interface IconButtonProps extends AriaAttributes, Data {
  icon: IconName;
  /** The control's name, always: an icon alone is never the only label (it is read out and shown as a title). */
  label: string;
  /** 48 px (default) or 44 px, the floor for a touch target. */
  size?: 44 | 48;
  direction?: "down" | "up" | "left" | "right";
  /** Ruled like a secondary button instead of bare. */
  ruled?: boolean;
  disabled?: boolean;
  onClick?: (e: MouseEvent<HTMLButtonElement>) => void;
  className?: string;
}

/** A universal control with no word beside it (close, zoom, the chart toggle, a pane's collapse). */
export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton({ icon, label, size = 48, direction, ruled, disabled, onClick, className, ...rest }, ref) {
  return (
    <motion.button
      ref={ref}
      type="button"
      className={cx(s.iconBtn, ruled && s.iconRuled, className)}
      style={{ width: size, height: size }}
      aria-label={label}
      title={label}
      aria-disabled={disabled || undefined}
      onClick={(e) => (disabled ? e.preventDefault() : onClick?.(e))}
      {...(disabled ? {} : press)}
      {...rest}
    >
      <Icon name={icon} direction={direction} />
    </motion.button>
  );
});
