"use client";
import type { ReactNode } from "react";
import { Button } from "./Button";
import { DataStateMark } from "./DataState";
import type { DataStateKind } from "./dataStates";
import { cx } from "./util";
import s from "./gates.module.css";

export type GateItemState = "ok" | "old" | "loading" | "unavailable" | "none";

export interface GateFactItem {
  id: string;
  text: ReactNode;
  /** What kind of line it is; an advisory is set heavier and comes first. */
  role?: "advisory" | "forecast" | "measured" | "predicted" | "record" | "note";
  /** Past its usual update: "(old)" is said beside the words. */
  old?: boolean;
  /** The line's provenance (a Stamp). */
  stamp?: ReactNode;
}
export interface GateItem {
  id: string;
  /** "Whale measures", "Domoic acid", "Sea". */
  title: string;
  /** Who decides or measures: "CDFW RAMP", "CDPH tests", "NWS and NDBC". */
  authority?: string;
  state: GateItemState;
  /** The gate in one line. */
  headline: ReactNode;
  /** regulation: the headline is an agency's delay or measure (the one foreign hue); closure: an official closure. */
  tone?: "plain" | "regulation" | "closure";
  /** The date the headline rests on, with its age and source (a Stamp, or words). */
  date?: ReactNode;
  facts?: GateFactItem[];
  /** How this gate relates to the others: a fixed sentence ("An old test says nothing about the whale delay"). */
  relation?: ReactNode;
  actions?: Array<{ id: string; label: string; href: string; external?: boolean }>;
}

const STATE_MARK: Record<GateItemState, DataStateKind> = { ok: "measured", old: "stale", loading: "loading", unavailable: "unavailable", none: "not-collected" };
const STATE_WORD: Record<GateItemState, string | null> = { ok: null, old: "old", loading: "reading", unavailable: "unavailable", none: "no record" };

export interface GateListProps {
  /** Accessible name: "What decides a trip from Moss Landing". */
  label: string;
  gates: GateItem[];
  /** Heading level for each gate's title (default 2). */
  level?: 2 | 3;
  className?: string;
}

/**
 * The gates of a decision as one ruled list (whale measures, domoic acid, sea): each says who decides it, its
 * state in one line, the date that line rests on, its facts (advisories first, set heavier), how it relates to
 * the other gates, and where to check it. Gates are separate on purpose: nothing here combines them into a
 * verdict, and an old or missing gate says so instead of going quiet.
 */
export function GateList({ label, gates, level = 2, className }: GateListProps) {
  const H = `h${level}` as "h2" | "h3";
  return (
    <ol className={cx(s.gates, className)} aria-label={label} data-testid="gate-list">
      {gates.map((g) => (
        <li key={g.id} className={s.gate} data-gate={g.id} data-state={g.state} data-tone={g.tone ?? "plain"} data-testid={`gate-${g.id}`}>
          <div className={s.gateHead}>
            <H className={s.gateTitle}>{g.title}</H>
            {g.authority ? <span className={s.gateAuthority}>{g.authority}</span> : null}
            {STATE_WORD[g.state] ? (
              <span className={s.gateState}>
                <DataStateMark kind={STATE_MARK[g.state]} /> {STATE_WORD[g.state]}
              </span>
            ) : null}
          </div>
          <p className={s.gateHeadline}>{g.headline}</p>
          {g.date ? <p className={s.gateDate}>{g.date}</p> : null}
          {g.facts?.length ? (
            <ul className={s.facts}>
              {g.facts.map((f) => (
                <li key={f.id} className={s.fact} data-role={f.role ?? "note"}>
                  <span>
                    {f.text}
                    {f.old ? <span className={s.old}> (old)</span> : null}
                  </span>
                  {f.stamp ? <span className={s.factStamp}>{f.stamp}</span> : null}
                </li>
              ))}
            </ul>
          ) : null}
          {g.relation ? <p className={s.relation}>{g.relation}</p> : null}
          {g.actions?.length ? (
            <div className={s.gateActions}>
              {g.actions.map((a) => (
                <Button key={a.id} variant="quiet" href={a.href} external={a.external}>
                  {a.label}
                </Button>
              ))}
            </div>
          ) : null}
        </li>
      ))}
    </ol>
  );
}

export interface AdvisoryItem {
  id: string;
  /** The advisory in the agency's words, shortened: "Small craft advisory until 3 AM Tuesday". */
  text: ReactNode;
  /** Who issued it and when (a Stamp, or words). Always shown: an advisory without its source is not shown as one. */
  source: ReactNode;
  href?: string;
  /** Where it applies, when that is narrower than the page: "Monterey Bay". */
  area?: ReactNode;
}
export interface AdvisoriesProps {
  /** Accessible name and visible heading: "In force now". */
  label: string;
  items: AdvisoryItem[];
  /** Said when there are none and the source was read: "No marine advisory in force (NWS, read 6 AM)". Omit to render nothing. */
  none?: ReactNode;
  className?: string;
}

/**
 * The first-viewport slot for advisories in force: above the dates, at every width. Official words with their
 * source and time, ruled in ink (regulation's hue is for regulation only). When none is in force it says so
 * with the source that was read, or renders nothing when that is not known: an empty slot never stands for "nothing in force".
 */
export function Advisories({ label, items, none, className }: AdvisoriesProps) {
  if (!items.length && !none) return null;
  return (
    <section className={cx(s.advisories, className)} aria-label={label} data-testid="advisories" data-count={items.length}>
      <h2 className={s.advHead}>{label}</h2>
      {items.length ? (
        <ul className={s.advList}>
          {items.map((a) => (
            <li key={a.id} className={s.adv} data-testid={`advisory-${a.id}`}>
              <p className={s.advText}>
                {a.href ? (
                  <a href={a.href} target="_blank" rel="noreferrer">
                    {a.text}
                  </a>
                ) : (
                  a.text
                )}
                {a.area ? <span className={s.advArea}> · {a.area}</span> : null}
              </p>
              <p className={s.advSource}>{a.source}</p>
            </li>
          ))}
        </ul>
      ) : (
        <p className={s.advNone}>{none}</p>
      )}
    </section>
  );
}
