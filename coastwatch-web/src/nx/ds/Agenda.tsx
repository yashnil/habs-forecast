import type { ReactNode } from "react";
import { day, daysBetween, range, toDate } from "./format";
import { hatchSegments } from "./Mark";
import type { MarkKind } from "./kinds";
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
  /**
   * What kind of thing the date is, drawn with the page legend's mark (T2): measured solid, model or prediction
   * hatched, official a ring. Without one the rail shows a plain tick. Time is never the fill: it is the item's
   * place against the now line.
   */
  kind?: Extract<MarkKind, "measured" | "model" | "official">;
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
 * Dated items on a rail (from the Tide concept), in date order with a now line between past and coming. Each
 * item's mark is its kind, the same mark as the page legend (measured solid, model hatched, official a ring),
 * so an NWS forecast for today is never a solid "measured" dot. Past items are told by their place above the
 * now line, greyed dates and a solid rail; coming ones by a dashed rail. Dates are absolute and set in a column.
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
          <RailMark kind={it.kind} />
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

const HATCH = hatchSegments(2.75, 2.75, 11.25, 11.25, 3.2);

/** The legend's mark on the rail, on a paper backing so the rail line does not run through it. */
function RailMark({ kind }: { kind?: AgendaItem["kind"] }) {
  return (
    <svg className={s.dot} width="14" height="14" viewBox="0 0 14 14" focusable="false" data-kind={kind ?? "date"}>
      {kind === "measured" ? <rect x="2" y="2" width="10" height="10" className={s.markSolid} /> : null}
      {kind === "model" ? (
        <>
          <rect x="2.75" y="2.75" width="8.5" height="8.5" className={s.markPaper} />
          <path d={HATCH} className={s.markHatch} />
          <rect x="2.75" y="2.75" width="8.5" height="8.5" className={s.markLine} />
        </>
      ) : null}
      {kind === "official" ? <circle cx="7" cy="7" r="5" className={s.markRing} /> : null}
      {!kind ? <path d="M3 7H11" className={s.markLine} /> : null}
    </svg>
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
