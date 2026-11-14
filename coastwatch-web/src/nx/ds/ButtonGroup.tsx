import type { ComponentPropsWithoutRef } from "react";
import { cx } from "./util";
import s from "./controls.module.css";

export interface ButtonGroupProps extends ComponentPropsWithoutRef<"div"> {
  /** start (default), end, or stretch (each button takes an equal share). */
  align?: "start" | "end" | "stretch";
  /** Buttons wrap to a new line when they do not fit (default true). */
  wrap?: boolean;
}

/**
 * Buttons that belong together (Save, Update, Duplicate): one height for all of them, the primary first, and a
 * full-width stack on a very narrow screen (360 px and under).
 */
export function ButtonGroup({ align = "start", wrap = true, className, children, ...rest }: ButtonGroupProps) {
  return (
    <div role="group" className={cx(s.group, className)} data-align={align} data-wrap={wrap || undefined} {...rest}>
      {children}
    </div>
  );
}
