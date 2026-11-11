"use client";
import { motion, useReducedMotion } from "motion/react";
import type { ReactNode } from "react";
import { day, daysBetween, dayTime } from "./format";
import { Icon } from "./Icon";
import { hatchSegments, Mark } from "./Mark";
import { ink } from "./motion";
import { NOT_VERIFIED } from "./kinds";
import { cx } from "./util";
import s from "./States.module.css";

/**
 * The designed states (spec section 9; journeys "Help without a tutorial"): each says what is missing, why,
 * and what to do. Worded, never a lone spinner or a blank.
 */

export interface UnavailableProps {
  /** What is missing: "Crab tests aren't available here". */
  what: ReactNode;
  /** Why, in one plain sentence. */
  why: ReactNode;
  /** What the reader can do: a link or button. */
  next?: ReactNode;
  className?: string;
}

const RAIL = hatchSegments(0, 0, 12, 400, 4);

/** A section with no data: the no-view hatch as its mark, never an empty box. */
export function Unavailable({ what, why, next, className }: UnavailableProps) {
  return (
    <section className={cx(s.state, s.unavailable, className)} aria-label="Not available">
      <span className={s.hatchRail} aria-hidden>
        <svg viewBox="0 0 12 400" preserveAspectRatio="xMidYMin slice" focusable="false">
          <path d={RAIL} />
        </svg>
      </span>
      <div>
        <p className={s.what}>{what}</p>
        <p className={s.why}>{why}</p>
        {next ? <div className={s.next}>{next}</div> : null}
      </div>
    </section>
  );
}

export interface StaleProps {
  /** The reading, as last known. */
  children?: ReactNode;
  /** When it was measured or sampled. */
  date: string;
  now: string;
  /** Why it is old, and when to expect a newer one. */
  why: ReactNode;
  className?: string;
}

/** An old value stays readable with "(old)" beside it, its date and age, and why (T17). */
export function Stale({ children, date, now, why, className }: StaleProps) {
  const age = Math.max(0, daysBetween(date, now));
  return (
    <section className={cx(s.state, className)} aria-label="Old reading">
      <div>
        {children ? (
          <p className={s.what}>
            {children} <Mark kind="stale" />
          </p>
        ) : null}
        <p className={s.why}>
          From {day(date, { ref: now })}, {age} days ago. {why}
        </p>
      </div>
    </section>
  );
}

export interface OfflineProps {
  /** When this device saved what is shown; null when the connection dropped on a page that was not saved. */
  savedAt: string | null;
  now?: string;
  /** The first word: "Offline" (default), or "A saved copy" when a slow network gave way to one. */
  heading?: string;
  /** Override for the second sentence. */
  children?: ReactNode;
  className?: string;
}

/** Offline: what is shown and from when. A saved pack only ever shows its own dates. */
export function Offline({ savedAt, now, heading = "Offline", children, className }: OfflineProps) {
  return (
    <section className={cx(s.state, s.offline, className)} role="status" aria-label="Offline">
      <Icon name="offline" />
      <div>
        <p className={s.what}>{savedAt ? `${heading}. Showing what this device saved ${dayTime(savedAt, { ref: now })}.` : `${heading}. You've lost the connection.`}</p>
        <p className={s.why}>{children ?? "Every value is as of then. Nothing updates until you're back online."}</p>
      </div>
    </section>
  );
}

export interface SourceClaim {
  agency: string;
  /** The agency's words, quoted, never paraphrased into a verdict. */
  quote: string;
  /** "SN26-021, Oct 31". */
  reference: string;
  url: string;
}

export interface SourcesDifferProps {
  /** What the disagreement is about: "Sport-harvested bivalves, Monterey County". */
  subject: ReactNode;
  claims: SourceClaim[];
  /** When Neritic last read both. */
  checked: string;
  now?: string;
  /** What to do: whom to check with. */
  next: ReactNode;
  className?: string;
}

/** "Sources differ" is a status, never resolved by Neritic (T40): both quotes, attributed, linked, dated. */
export function SourcesDiffer({ subject, claims, checked, now, next, className }: SourcesDifferProps) {
  return (
    <section className={cx(s.state, s.differ, className)} aria-label={`Sources differ, ${NOT_VERIFIED}`}>
      <div className={s.differHead}>
        <Mark kind="official" withWords={false} />
        <p className={s.what}>
          Sources differ · {NOT_VERIFIED}
        </p>
      </div>
      <p className={s.subject}>{subject}</p>
      <ol className={s.claims}>
        {claims.map((c) => (
          <li key={c.reference} className={s.claim}>
            <p className={s.agency}>{c.agency}</p>
            <blockquote className={s.quote} cite={c.url}>
              “{c.quote}”
            </blockquote>
            <a className={s.ref} href={c.url} target="_blank" rel="noreferrer">
              {c.reference}
            </a>
          </li>
        ))}
      </ol>
      <p className={s.why}>
        Neritic doesn&rsquo;t decide between them. Both read {day(checked, { ref: now })}. {next}
      </p>
    </section>
  );
}

export interface LoadingProps {
  /** What is being read, by name: "Reading buoy 46236". */
  children: ReactNode;
  className?: string;
}

/** Loading says what it is reading. A thin line inks under the words (static under reduced motion). */
export function Loading({ children, className }: LoadingProps) {
  const reduce = useReducedMotion();
  return (
    <div className={cx(s.loading, className)} role="status" aria-live="polite">
      <span className={s.loadingWords}>{children}</span>
      <span className={s.loadingTrack} aria-hidden>
        <motion.span className={s.loadingLine} initial={{ scaleX: 0 }} animate={{ scaleX: reduce ? 0.4 : 1 }} transition={reduce ? { duration: 0 } : { ...ink(1.6), repeat: Infinity, repeatType: "reverse" }} />
      </span>
    </div>
  );
}
