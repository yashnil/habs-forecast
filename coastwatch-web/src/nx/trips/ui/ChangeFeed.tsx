"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useId, useState } from "react";
import { Button, day, dayTime, Micro, reveal, swapIn, VisuallyHidden } from "@/nx/ds";
import { FEED_SHOWN, feedView, groupUnchecked, readableDetail, stopPosition, type TimelineStop } from "../feed";
import type { Change, Recheck, Snapshot } from "../types";
import s from "./evidence.module.css";

/**
 * "Since you planned this trip" (v15 §8): the workspace's one dominant section. A short prioritized feed (five shown,
 * the rest on demand), one descriptive line per change with where and when and a link to the evidence. Its headline
 * is its state: no snapshot yet; checking; complete with no changes (only then); incomplete (never an all-clear,
 * names what could not be checked); offline against the saved copy (its age said). Under it, the change history:
 * the planning snapshot and each recheck along a time axis, each one tap away.
 */
export type ChangeFeedProps = {
  planning: Snapshot | null;
  /** the latest recheck's answer (RecheckResult fits); null before the first */
  result: Recheck | null;
  checking: boolean;
  now: string;
  /** the change history (feed.ts `timelineOf`) */
  timeline?: TimelineStop[];
  selected?: string | null;
  onSelect?: (id: string) => void;
  onRecheck?: () => void;
  /** opens one change's evidence in the workspace (e.g. the snapshot view); without it, the source link is shown */
  onEvidence?: (c: Change) => void;
  /** the recheck action's weight: the workspace keeps one filled button per view */
  actionVariant?: "primary" | "secondary" | "quiet";
  headingId?: string;
};

export function ChangeFeed({ planning, result, checking, now, timeline = [], selected = null, onSelect, onRecheck, onEvidence, actionVariant = "secondary", headingId }: ChangeFeedProps) {
  const v = feedView({ planning, result, checking, now });
  const [all, setAll] = useState(false);
  const reduce = useReducedMotion();
  const own = useId();
  const hid = headingId ?? `${own}-h`;
  const shown = all ? v.changes : v.changes.slice(0, FEED_SHOWN);
  const more = v.changes.length - shown.length;
  return (
    <section className={s.feed} data-state={v.state} aria-labelledby={hid} aria-busy={checking || undefined} data-testid="trip-change-feed">
      <div className={s.feedHead}>
        <Micro>Since you planned this trip</Micro>
        <AnimatePresence mode="wait" initial={false}>
          <motion.h2 key={v.headline} id={hid} className={s.headline} initial={reduce ? false : { opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={reduce ? { duration: 0 } : swapIn}>
            {v.headline}
          </motion.h2>
        </AnimatePresence>
        {v.sub ? <p className={s.sub}>{v.sub}</p> : null}
        <VisuallyHidden role="status" aria-live="polite">
          {checking ? "Checking the sources again." : v.state === "none" || v.state === "unchecked" ? "" : v.headline}
        </VisuallyHidden>
        {checking ? (
          <div className={s.sweep} aria-hidden="true">
            {/* design-lint-allow motion-preset: an indeterminate progress sweep repeats; no preset covers a loop (still under reduced motion) */}
            <motion.span className={s.sweepBar} initial={{ left: "-34%" }} animate={reduce ? { left: "33%" } : { left: ["-34%", "100%"] }} transition={reduce ? { duration: 0 } : { duration: 1.4, ease: "linear", repeat: Infinity }} />
          </div>
        ) : null}
      </div>

      {shown.length ? (
        <ul className={s.list} data-testid="trip-changes">
          <AnimatePresence initial={false}>
            {shown.map((c) => (
              <motion.li key={c.id} className={s.item} data-p={c.priority} data-kind={c.kind} data-category={c.category} layout={!reduce} initial={reduce ? false : { opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} transition={reduce ? { duration: 0 } : reveal}>
                <span className={s.mark} aria-hidden="true" />
                <p className={s.title}>{c.title}</p>
                <ChangeMeta c={c} now={now} onEvidence={onEvidence} />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      ) : null}
      {more > 0 || all ? (
        <div>
          <Button variant="quiet" onClick={() => setAll(!all)} aria-expanded={all}>
            {all ? "Show fewer" : `Show ${more} more`}
          </Button>
        </div>
      ) : null}

      {v.state === "incomplete" || (v.state === "saved" && v.unchecked.length) ? (
        <ul className={s.unchecked} aria-label="Could not be checked" data-testid="trip-unchecked">
          {groupUnchecked(v.unchecked).map((g) => (
            <li key={g.sources.join()}>
              <b>{g.labels}</b> · {g.why}
            </li>
          ))}
        </ul>
      ) : null}

      {onRecheck && v.state !== "none" ? (
        <div className={s.actions}>
          <Button variant={actionVariant} onClick={onRecheck} busy={checking} busyLabel="Checking…" glyph={actionVariant === "quiet" ? null : "→"} data-testid="trip-recheck">
            {result ? "Recheck again" : "Recheck sources"}
          </Button>
        </div>
      ) : null}

      {timeline.length > 1 ? <History stops={timeline} now={now} selected={selected} onSelect={onSelect} /> : null}
    </section>
  );
}

/** Where and when, then the evidence: the record's own page, or the snapshot view. */
export function ChangeMeta({ c, now, onEvidence }: { c: Change; now: string; onEvidence?: (c: Change) => void }) {
  const when = c.when && Number.isFinite(Date.parse(c.when)) ? (/^\d{4}-\d{2}-\d{2}$/.test(c.when) ? day(c.when, { ref: now }) : dayTime(c.when, { ref: now })) : null;
  return (
    <>
      <span className={s.meta}>
        {c.where ? <span>{c.where}</span> : null}
        {when ? <span>{when}</span> : null}
        {onEvidence && c.evidenceKey ? (
          <Button variant="quiet" onClick={() => onEvidence(c)}>
            Evidence
          </Button>
        ) : null}
        {c.url ? (
          <a className={s.link} href={c.url} target="_blank" rel="noreferrer noopener">
            Source ↗
          </a>
        ) : null}
      </span>
      {c.before || c.after ? (
        <dl className={s.was}>
          {c.before ? (
            <>
              <dt>Planned with</dt>
              <dd>{c.before}</dd>
            </>
          ) : null}
          {c.after ? (
            <>
              <dt>Now</dt>
              <dd>{c.after}</dd>
            </>
          ) : null}
        </dl>
      ) : readableDetail(c.detail) ? (
        <p className={s.detail}>{readableDetail(c.detail)}</p>
      ) : null}
    </>
  );
}

const STOP_WORDS: Record<TimelineStop["kind"], string> = { planning: "Planned", superseded: "Earlier plan", recheck: "Recheck" };

/** The change history: planning and rechecks along time, now at the right. */
function History({ stops, now, selected, onSelect }: { stops: TimelineStop[]; now: string; selected: string | null; onSelect?: (id: string) => void }) {
  const first = stops[0].at;
  return (
    <div className={s.history} data-testid="trip-change-history">
      <Micro as="h3">Change history</Micro>
      <div className={s.axis} role="list" aria-label="Planning snapshot and rechecks, oldest first">
        <span className={s.axisLine} aria-hidden="true" />
        <span className={s.axisNow} style={{ left: "100%" }} aria-hidden="true" />
        {stops.map((t) => {
          const label = `${STOP_WORDS[t.kind]}, ${dayTime(t.at, { ref: now })}${t.mode === "saved" ? ", saved copy" : ""}${t.kind === "recheck" ? `, ${t.complete ? "complete" : "incomplete"}, ${t.changes} ${t.changes === 1 ? "change" : "changes"}` : ""}`;
          return (
            <div role="listitem" key={t.id}>
              {/* design-lint-allow raw-button: a point on the history axis is a 44 px target placed by time, not a control row */}
              <button
                type="button"
                className={s.stop}
                style={{ left: `${stopPosition(t.at, first, now) * 100}%` }}
                data-kind={t.kind}
                data-mode={t.mode}
                data-complete={t.complete || undefined}
                data-selected={selected === t.id || undefined}
                aria-pressed={selected === t.id}
                aria-label={label}
                title={label}
                onClick={() => onSelect?.(t.id)}
              >
                <span className={s.dot} aria-hidden="true" />
                <span aria-hidden="true">{day(t.at, { ref: now })}</span>
                {t.kind === "recheck" && t.changes ? (
                  <span className={s.count} aria-hidden="true">
                    {t.changes}
                  </span>
                ) : null}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
