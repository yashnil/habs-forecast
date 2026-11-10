"use client";
import { AnimatePresence, motion } from "motion/react";
import { useId, useState, type ReactNode } from "react";
import { Mark } from "./Mark";
import { settle } from "./motion";
import { cx } from "./util";
import s from "./Margin.module.css";

export interface MarginProps {
  /** The page's one sentence saying what this view is, then its statement and notes. */
  children: ReactNode;
  /** The imprint at the foot: data date, build, "Not for navigation". */
  imprint?: ReactNode;
  /** Accessible name of the column (it is a complementary region beside the chart). */
  label: string;
  className?: string;
}

/** The margin column (spec section 6): ruled, no container, beside the sheet. 340 to 372 px on desktop. */
export function Margin({ children, imprint, label, className }: MarginProps) {
  return (
    <aside className={cx(s.margin, className)} aria-label={label}>
      <div className={s.content}>{children}</div>
      {imprint ? <div className={s.imprint}>{imprint}</div> : null}
    </aside>
  );
}

/** A ruled list of margin rows: a heavy rule on top, hairlines between. */
export function MarginRows({ children, label, className }: { children: ReactNode; label?: string; className?: string }) {
  return (
    <ul className={cx(s.rows, className)} aria-label={label}>
      {children}
    </ul>
  );
}

export type MarginRowState = "default" | "stale" | "unavailable" | "loading" | "offline";

export interface MarginRowProps {
  /** The key word: Sea, Tide, Algae, Tests. */
  label: string;
  /** The reading, in words, with its unit. */
  value?: ReactNode;
  /** Expands in place. Without it the row is static. */
  detail?: ReactNode;
  /** The stamp, shown when open (and always when the row has no detail). */
  stamp?: ReactNode;
  state?: MarginRowState;
  /** Unavailable: why, in one line. Loading: what is being read. Offline: when it was saved. */
  note?: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}

/** One line of the margin: key, value and an in-place expansion (spec section 9). */
export function MarginRow({ label, value, detail, stamp, state = "default", note, defaultOpen = false, className }: MarginRowProps) {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  const expandable = Boolean(detail) && state !== "loading" && state !== "unavailable";
  const shown =
    state === "loading" ? (
      <span className={s.quiet} role="status">
        {note ?? "Reading…"}
      </span>
    ) : state === "unavailable" ? (
      <span className={s.unavailable}>
        <Mark kind="noview" />
        <span>{note ?? "Not available"}</span>
      </span>
    ) : (
      <>
        {value}
        {state === "stale" ? (
          <>
            {" "}
            <Mark kind="stale" />
          </>
        ) : null}
        {state === "offline" && note ? <span className={s.quiet}> {note}</span> : null}
      </>
    );
  if (!expandable) {
    return (
      <li className={cx(s.row, className)} data-state={state}>
        <div className={s.line}>
          <span className={s.key}>{label}</span>
          <span className={s.value}>{shown}</span>
        </div>
        {stamp && state !== "loading" && state !== "unavailable" ? <div className={s.stampOnly}>{stamp}</div> : null}
      </li>
    );
  }
  return (
    <li className={cx(s.row, className)} data-state={state} data-open={open || undefined}>
      <button type="button" className={cx(s.line, s.toggle)} aria-expanded={open} aria-controls={id} onClick={() => setOpen((o) => !o)}>
        <span className={s.key}>{label}</span>
        <span className={s.value}>{shown}</span>
        <span className={s.sign} aria-hidden>
          <svg width="14" height="14" viewBox="0 0 14 14" focusable="false">
            <path d="M1 7h12" />
            <motion.path d="M7 1v12" initial={false} animate={{ scaleY: open ? 0 : 1 }} transition={settle} style={{ originY: "50%" }} />
          </svg>
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open ? (
          <motion.div
            id={id}
            key="detail"
            className={s.detailWrap}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={settle}
          >
            <div className={s.detail}>
              {detail}
              {stamp ? <div className={s.stamp}>{stamp}</div> : null}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </li>
  );
}
