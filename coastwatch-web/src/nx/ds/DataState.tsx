"use client";
import { motion } from "motion/react";
import type { ReactNode } from "react";
import { age, day } from "./format";
import { register, registerFrom, registerTo } from "./motion";
import { cx } from "./util";
import s from "./controls.module.css";

import { DATA_STATE_WORDS, type DataStateKind } from "./dataStates";
export { DATA_STATE_KINDS, DATA_STATE_WORDS, type DataStateKind } from "./dataStates";

/** The state's mark, 14 px, in ink: solid for what was measured, hollow or hatched for everything else. */
export function DataStateMark({ kind, size = 14, className }: { kind: DataStateKind; size?: number; className?: string }) {
  const c = 7;
  const shape = (() => {
    switch (kind) {
      case "measured":
        return <rect x="2" y="2" width="10" height="10" className={s.mFill} />;
      case "reported-zero":
        return <rect x="2.75" y="2.75" width="8.5" height="8.5" className={s.mLine} />;
      case "below-detection":
        return <path d="M2.5 3.5h9L7 11.5z" className={s.mLine} />;
      case "estimated":
        return <path d="M7 1.8 12.2 7 7 12.2 1.8 7z" className={s.mLine} />;
      case "forecast":
        return (
          <>
            <rect x="2.75" y="2.75" width="8.5" height="8.5" className={s.mLine} />
            <path d="M2.75 8 8 2.75M6 11.25 11.25 6" className={s.mHair} />
          </>
        );
      case "historical":
        return (
          <>
            <circle cx={c} cy={c} r="4.6" className={s.mLine} />
            <path d="M7 4.4V7l2 1.4" className={s.mLine} />
          </>
        );
      case "stale":
        return <rect x="2.75" y="2.75" width="8.5" height="8.5" className={s.mDash} />;
      case "rejected":
        return (
          <>
            <circle cx={c} cy={c} r="4.6" className={s.mLine} />
            <path d="M3.4 10.6l7.2-7.2" className={s.mLine} />
          </>
        );
      case "missing":
        return <path d="M2.5 7h3M8.5 7h3" className={s.mLine} />;
      case "not-collected":
        return <path d="M3 3l8 8M11 3l-8 8" className={s.mLine} />;
      case "unavailable":
      case "outside-coverage":
        return (
          <>
            <rect x="1.75" y="1.75" width="10.5" height="10.5" className={s.mHairBox} />
            <path d="M1.75 6 6 1.75M1.75 10.5 10.5 1.75M5.5 12.25 12.25 5.5M10 12.25l2.25-2.25" className={s.mHair} />
          </>
        );
      case "unverified":
        return <circle cx={c} cy={c} r="4.6" className={s.mLine} />;
      case "offline":
        return <path d="M2 11.5h10M7 2.5v6M4.5 6 7 8.5 9.5 6" className={s.mLine} />;
      default:
        return <path d="M2 7h2.5M5.75 7h2.5M9.5 7H12" className={s.mLine} />;
    }
  })();
  return (
    <svg className={cx(s.stateMark, className)} width={size} height={size} viewBox="0 0 14 14" aria-hidden="true" focusable="false" data-state-mark={kind}>
      {shape}
    </svg>
  );
}

export interface DataStateProps {
  kind: DataStateKind;
  /** What the state is about: "pDA", "Crab tests". */
  what?: ReactNode;
  /** The reading's date (measured, sampled, valid, saved): shown as a date and, with `now`, its age. */
  date?: string;
  now?: string;
  source?: ReactNode;
  /** Why, in one plain sentence ("0 of 12 samples in this window measured pDA"). */
  reason?: ReactNode;
  /** What the reader can do next: a Button or a link. */
  action?: ReactNode;
  /** inline: mark and word in a line of text; block: a ruled note with the reason and action; plot: fills a plot's area. */
  variant?: "inline" | "block" | "plot";
  /** The state's word, when the source has its own for it ("At or below an unpublished limit"). The mark does not change. */
  word?: string;
  className?: string;
  children?: ReactNode;
  "data-testid"?: string;
}

/**
 * One reading's state, said the same way everywhere: its mark, its word, what it is about, its date and age,
 * its source, why, and what to do next. Loading and offline are announced politely; nothing here is a spinner
 * alone or an empty box.
 */
export function DataState({ kind, what, date, now, source, reason, action, variant = "inline", word: ownWord, className, children, ...rest }: DataStateProps) {
  const word = ownWord ?? DATA_STATE_WORDS[kind].word;
  const live = kind === "loading" || kind === "offline";
  const when = date ? `${day(date, { ref: now, year: true })}${now ? ` (${age(date, now)})` : ""}` : null;
  if (variant === "inline") {
    return (
      <span className={cx(s.stateInline, className)} data-state={kind} role={live ? "status" : undefined} data-testid={rest["data-testid"]}>
        <DataStateMark kind={kind} />
        <span>
          <span className={s.stateWord}>{word}</span>
          {what ? <> · {what}</> : null}
          {source ? <> · {source}</> : null}
          {when ? <> · {when}</> : null}
          {reason ? <> · {reason}</> : null}
        </span>
        {action}
      </span>
    );
  }
  return (
    <div className={cx(s.stateBlock, variant === "plot" && s.statePlot, className)} data-state={kind} role={live ? "status" : "note"} data-testid={rest["data-testid"]}>
      <p className={s.stateHead}>
        <DataStateMark kind={kind} size={16} />
        <span>
          <span className={s.stateWord}>{word}</span>
          {what ? <>: {what}</> : null}
        </span>
      </p>
      {reason ? <p className={s.stateReason}>{reason}</p> : null}
      {source || when ? (
        <p className={s.stateMeta}>
          {source}
          {source && when ? " · " : null}
          {when}
        </p>
      ) : null}
      {children}
      {action ? <div className={s.stateAction}>{action}</div> : null}
    </div>
  );
}

/**
 * One legend for a view: the states present on it, each with its mark and its sentence. `words` replaces a
 * state's sentence where the page's source says something more exact than the default (a source that publishes
 * no detection limit: "Reported 0: at or below a limit the source does not publish, not absent").
 */
export function DataStateLegend({ kinds, words, className }: { kinds: DataStateKind[]; words?: Partial<Record<DataStateKind, string>>; className?: string }) {
  return (
    <ul className={cx(s.stateLegend, className)} aria-label="Legend">
      {kinds.map((k) => (
        <li key={k}>
          <DataStateMark kind={k} />
          <span>{words?.[k] ?? DATA_STATE_WORDS[k].legend}</span>
        </li>
      ))}
    </ul>
  );
}

/* ---------------------------------------------------------------- Availability */

export interface AvailabilitySpan {
  /** ms, or an ISO date. */
  from: number | string;
  to: number | string;
  kind: DataStateKind;
  /** How many samples the span holds, for the row's words. */
  n?: number;
}
export interface AvailabilityRow {
  id: string;
  label: ReactNode;
  spans: AvailabilitySpan[];
  /** The latest real observation of this variable (ISO), or null when there is none. */
  latest?: string | null;
  /** Counts in words: "12 of 49 samples measured". */
  summary?: ReactNode;
}
export interface AvailabilityProps {
  /** Accessible name: "What was measured, and when". */
  label: string;
  variables: AvailabilityRow[];
  /** The whole record's time range, ms. */
  domain: [number, number];
  /** The window in view, drawn as a frame across every row. */
  window?: [number, number];
  now?: string;
  selected?: string | null;
  /** Choosing a row (the variable becomes the primary one). */
  onPick?: (id: string) => void;
  /** "Latest: Mar 3, 2024" pressed: move the window to that observation. */
  onJumpLatest?: (id: string, iso: string) => void;
  className?: string;
}

const toMs = (t: number | string) => (typeof t === "number" ? t : Date.parse(t));

/**
 * Availability by variable and date: one thin strip per variable across the record's whole span, solid where
 * it was measured, hollow where the lab reported zero or below detection, blank where nothing was collected,
 * with the latest real observation named and one press away. It answers "is there anything here, and when"
 * before the reader guesses a range.
 */
export function Availability({ label, variables, domain, window: win, now, selected, onPick, onJumpLatest, className }: AvailabilityProps) {
  const [d0, d1] = domain;
  const span = Math.max(1, d1 - d0);
  const x = (t: number) => `${(Math.min(1, Math.max(0, (t - d0) / span)) * 100).toFixed(3)}%`;
  const w = (a: number, b: number) => `${Math.max(0.35, ((Math.min(d1, b) - Math.max(d0, a)) / span) * 100).toFixed(3)}%`;
  return (
    <table className={cx(s.avail, className)} aria-label={label} data-testid="availability">
      <tbody>
        {variables.map((v) => {
          const on = selected === v.id;
          const name = onPick ? (
            <button type="button" className={s.availPick} aria-pressed={on} onClick={() => onPick(v.id)} data-testid={`availability-${v.id}`}>
              {v.label}
            </button>
          ) : (
            v.label
          );
          return (
            <tr key={v.id} className={s.availRow} data-selected={on || undefined}>
              <th scope="row" className={s.availName}>
                {name}
              </th>
              <td className={s.availCell}>
                <span className={s.availTrack} aria-hidden="true">
                  {v.spans.map((sp, i) => (
                    <span key={i} className={s.availSpan} data-kind={sp.kind} style={{ left: x(toMs(sp.from)), width: w(toMs(sp.from), toMs(sp.to)) }} />
                  ))}
                  {win ? <span className={s.availWindow} style={{ left: x(win[0]), width: w(win[0], win[1]) }} /> : null}
                </span>
                <span className={s.availWords}>
                  {v.summary ?? (v.spans.length ? null : DATA_STATE_WORDS["not-collected"].word)}
                  {v.latest ? (
                    onJumpLatest ? (
                      <button type="button" className={s.availLatest} onClick={() => onJumpLatest(v.id, v.latest!)} data-testid={`availability-latest-${v.id}`}>
                        Latest: {day(v.latest, { ref: now, year: true })}
                      </button>
                    ) : (
                      <span>Latest: {day(v.latest, { ref: now, year: true })}</span>
                    )
                  ) : v.latest === null ? (
                    <span>No observation in the record</span>
                  ) : null}
                </span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/* ---------------------------------------------------------------- Skeleton and Recovered */

export interface SkeletonProps {
  /** What is being read, in words: "Reading the tests". Announced politely; never a shape alone. */
  label: ReactNode;
  /** Ruled lines standing where the text will be. */
  lines?: number;
  /** A block standing where a result figure will be. */
  figure?: boolean;
  /** A block of this height standing where a plot will be, px. */
  plot?: number;
  className?: string;
}

/**
 * A placeholder at the final geometry, so nothing moves when the data arrives: ruled lines, never a shimmer
 * or a spinner. It says what it is reading.
 */
export function Skeleton({ label, lines = 3, figure, plot, className }: SkeletonProps) {
  return (
    <div className={cx(s.skeleton, className)} role="status" aria-live="polite" data-testid="skeleton">
      <span className={s.skelWords}>{label}</span>
      {figure ? <span className={s.skelFigure} aria-hidden="true" /> : null}
      {plot ? <span className={s.skelPlot} style={{ height: plot }} aria-hidden="true" /> : null}
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} className={s.skelLine} data-last={i === lines - 1 && lines > 1 ? "" : undefined} aria-hidden="true" />
      ))}
    </div>
  );
}

export interface RecoveredProps {
  /** When the page's data was last read (ISO). */
  at: string;
  now?: string;
  children?: ReactNode;
  className?: string;
}

/** Back online, in a line where the offline note stood: "Back online · updated 2 minutes ago". Appears with `register`. */
export function Recovered({ at, now, children, className }: RecoveredProps) {
  return (
    <motion.p className={cx(s.recovered, className)} role="status" initial={registerFrom} animate={registerTo} transition={register} data-testid="recovered">
      <DataStateMark kind="measured" />
      <span>
        {children ?? "Back online"} · updated {now ? age(at, now) : day(at)}
      </span>
    </motion.p>
  );
}
