import type { ReactNode } from "react";
import { day, daysBetween, range, toDate } from "./format";
import { cx, VisuallyHidden } from "./util";
import s from "./Agenda.module.css";

export interface AgendaItem {
  id: string;
  start: string;
  end?: string | null;
  title: ReactNode;
  detail?: ReactNode;
  /** Regulation items (whale measures, delays) take the regulation hue on their title, nothing else does. */
  tone?: "ink" | "regulation" | "closure";
  /** "subject to change": the agency has said the date may move. */
  tentative?: boolean;
  /** The stamp for this item. */
  stamp?: ReactNode;
}

export interface AgendaProps {
  items: AgendaItem[];
  /** Today (Pacific). Items ending before it are past. */
  now: string;
  /** Accessible name for the list. */
  label: string;
  className?: string;
}

export type AgendaPhase = "past" | "now" | "future";

/** Past when it ended before today, now when today falls inside it, future otherwise. */
export function agendaPhase(item: Pick<AgendaItem, "start" | "end">, now: string): AgendaPhase {
  const endDays = daysBetween(item.end ?? item.start, now);
  const startDays = daysBetween(item.start, now);
  if (endDays > 0) return "past";
  if (startDays >= 0) return "now";
  return "future";
}

/**
 * Dated items on a rail (from the Tide concept): past solid, future hollow, a now line between them.
 * Dates are absolute and set in a column, so the eye can run down them.
 */
export function Agenda({ items, now, label, className }: AgendaProps) {
  const sorted = [...items].sort((a, b) => toDate(a.start).getTime() - toDate(b.start).getTime());
  const firstFuture = sorted.findIndex((it) => agendaPhase(it, now) !== "past");
  const nowAt = firstFuture === -1 ? sorted.length : firstFuture;
  const rows: ReactNode[] = [];
  // an item under way today already prints its own date right under the now line: the line then says only "Today"
  const dated = !(nowAt < sorted.length && agendaPhase(sorted[nowAt], now) === "now");
  sorted.forEach((it, i) => {
    if (i === nowAt) rows.push(<NowLine key="now" now={now} dated={dated} />);
    const phase = agendaPhase(it, now);
    rows.push(
      <li key={it.id} className={cx(s.item, PHASE_CLASS[phase])} data-phase={phase}>
        <span className={s.when}>
          <time dateTime={toDate(it.start).toISOString()}>{range(it.start, it.end, { ref: now })}</time>
        </span>
        <span className={s.rail} aria-hidden>
          <svg className={s.dot} width="14" height="14" viewBox="0 0 14 14" focusable="false">
            <circle cx="7" cy="7" r={phase === "future" ? 5.25 : 5.75} />
          </svg>
        </span>
        <span className={s.what}>
          <VisuallyHidden>{phase === "past" ? "Past. " : phase === "now" ? "Under way. " : "Coming. "}</VisuallyHidden>
          <span className={s.title} data-tone={it.tone ?? "ink"}>
            {it.title}
          </span>
          {it.tentative ? <span className={s.tentative}> Subject to change.</span> : null}
          {it.detail ? <span className={s.detail}>{it.detail}</span> : null}
          {it.stamp ? <span className={s.stamp}>{it.stamp}</span> : null}
        </span>
      </li>,
    );
  });
  if (nowAt === sorted.length) rows.push(<NowLine key="now" now={now} dated />);
  return (
    <ol className={cx(s.agenda, className)} aria-label={label}>
      {rows}
    </ol>
  );
}

/** The phases' own classes: an item under way is an item (solid dot, its own date), never the now line. */
const PHASE_CLASS: Record<AgendaPhase, string> = { past: s.past, now: s.underway, future: s.future };

function NowLine({ now, dated }: { now: string; dated: boolean }) {
  return (
    <li className={s.now} aria-label={`Today, ${day(now)}`}>
      <span className={s.nowWhen}>Today</span>
      <span className={s.nowRail} aria-hidden />
      {dated ? <span className={s.nowWhat}>{day(now)}</span> : null}
    </li>
  );
}
