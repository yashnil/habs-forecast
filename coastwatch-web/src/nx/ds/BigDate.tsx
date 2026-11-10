import type { ReactNode } from "react";
import { range, toDate } from "./format";
import { cx } from "./util";
import s from "./BigDate.module.css";

export interface DateSpan {
  start: string;
  end?: string | null;
}

export interface BigDateProps {
  /** What the date is, in the agency's terms: "Next assessment, subject to change". */
  label: ReactNode;
  start: string;
  end?: string | null;
  /** Later dates of the same thing ("then Jan 13–15"). */
  then?: DateSpan[];
  /** Today, so a year is printed only when it differs. */
  now: string;
  size?: "xl" | "lg";
  /** The stamp: whose date it is and when Neritic read it. */
  children?: ReactNode;
  className?: string;
}

/** The next date that matters, with what it is and its source (spec section 9). */
export function BigDate({ label, start, end, then, now, size = "xl", children, className }: BigDateProps) {
  const iso = toDate(start).toISOString();
  return (
    <div className={cx(s.bigDate, s[size], className)}>
      <p className={s.label}>{label}</p>
      <p className={s.date}>
        <time dateTime={iso}>{range(start, end, { ref: now })}</time>
      </p>
      {then && then.length ? <p className={s.then}>then {then.map((d) => range(d.start, d.end, { ref: now })).join(", then ")}</p> : null}
      {children ? <div className={s.stamp}>{children}</div> : null}
    </div>
  );
}
