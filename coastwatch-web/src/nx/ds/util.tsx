import type { ComponentPropsWithoutRef } from "react";
import s from "./Root.module.css";

export const cx = (...c: Array<string | false | null | undefined>) => c.filter(Boolean).join(" ");

/** React ids made safe for SVG `url(#id)` references. */
export function svgId(id: string): string {
  return `nx${id.replace(/[^a-zA-Z0-9_-]/g, "")}`;
}

/** Text for screen readers only. */
export function VisuallyHidden({ children, ...rest }: ComponentPropsWithoutRef<"span">) {
  return (
    <span className={s.vh} {...rest}>
      {children}
    </span>
  );
}
