"use client";

import { useEffect, useRef, useState } from "react";
import type { OfficialDataset, OfficialRecord } from "@/generated/official";
import type { SourceStatus } from "@/generated/schema";
import { OFFICIAL_STATUS } from "@/content/copy";
import { ACTION_LABEL, FISHERY_LABEL, criticalFlags, currentRecords, disputedRecords, officialVerification, recordFlags, type Dispute, type Verification } from "@/lib/official";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel } from "@/v4/ui/time";

/**
 * Official notices: one control, always saying they are not verified until a person reviews the
 * registry (cwp review-official). The verification is computed in the browser from dates, so a stale
 * review loses its status even if the pipeline stops. Each notice keeps the agency's own wording,
 * its dates, every warning (an expected end that passed is never hidden), and its sources; the
 * agencies' lists are the authority, Neritic's copy is a transcription.
 */
/** `local`: a place is open (a harbour, a briefing): the notices whose official geography covers its
 *  waters, and those whose reach there is not settled (lib/applicability), led by name and count. */
export type LocalNotices = { name: string; covering: string[]; unsettled: string[] };

export function NoticesButton({ official, status = null, compact = false, local = null }: { official: OfficialDataset | null; status?: SourceStatus | null; compact?: boolean; local?: LocalNotices | null }) {
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => setNow(new Date()), []);
  const box = useRef<HTMLDivElement | null>(null);
  const button = useRef<HTMLButtonElement | null>(null);
  const recs = official ? currentRecords(official) : [];
  // with a place open: its own notices first (current records only: a disputed one is never counted)
  const here = local ? recs.filter((r) => local.covering.includes(r.id)) : [];
  const maybe = local ? recs.filter((r) => local.unsettled.includes(r.id) && !local.covering.includes(r.id)) : [];
  const elsewhere = local ? recs.filter((r) => !here.includes(r) && !maybe.includes(r)) : recs;
  // an agency release appears to lift these: never counted or shown as in effect, always shown with the release
  const disputed = official ? disputedRecords(official) : [];
  const v: Verification | null = now ? officialVerification(official, status, now) : null;
  const state = !official ? "unavailable" : (v?.state ?? "unverified");
  const word = OFFICIAL_STATUS.verification[state];
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setOpen(false);
      button.current?.focus();
    };
    const c = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    window.addEventListener("keydown", k);
    window.addEventListener("mousedown", c);
    return () => {
      window.removeEventListener("keydown", k);
      window.removeEventListener("mousedown", c);
    };
  }, [open]);
  return (
    <div ref={box} className="relative">
      <button
        ref={button}
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label={local ? `${here.length} official notices cover ${local.name}'s waters, ${recs.length} in California, ${word.toLowerCase()}` : `${recs.length} official notices, ${word.toLowerCase()}`}
        data-testid="notices-button"
        data-state={state}
        className="nx-press nx-focus flex h-11 items-center gap-2 rounded-full border border-n-official-line bg-n-surface px-3 text-[14px] font-semibold text-n-ink"
      >
        <Icon name="shield" className="h-[18px] w-[18px] text-n-official" />
        <span className="nx-data" data-testid="notices-count">{official ? (local ? here.length : recs.length) : "?"}</span>
        {local && official && <span className="text-n-ink-2" data-testid="notices-here">here</span>}
        {/* without a place, the count is statewide and says so (it never reads as this place's count) */}
        {!local && official && <span className="text-n-ink-2" data-testid="notices-statewide">in CA</span>}
        {/* below 400 px the word gives way (the count, the shield and the label still say it) so the bar fits */}
        {!compact && <span className="text-n-official max-[399px]:hidden">{word}</span>}
      </button>
      {open && (
        <div
          role="dialog"
          aria-label="Official notices"
          // on a phone the panel spans the screen under the top bar; elsewhere it hangs from the button
          className="nx-rise fixed inset-x-3 top-[64px] z-50 flex max-h-[calc(100dvh-80px)] flex-col rounded-2xl border border-n-line bg-n-surface text-n-ink shadow-[var(--nx-shadow)] md:absolute md:inset-x-auto md:right-0 md:top-[52px] md:max-h-[min(72vh,680px)] md:w-[440px]"
          data-testid="notices-panel"
        >
          <div className="border-b border-n-line p-4">
            <p className="text-[16px] font-semibold" data-testid="notices-heading">
              {recs.length} active notices in California
            </p>
            <p className="mt-1 text-[13.5px] leading-snug text-n-official" data-testid="notices-unverified" data-state={state}>
              {state === "verified" ? `Checked by a person against the agencies' pages${v?.reviewedAt ? ` on ${clockLabel(v.reviewedAt)}` : ""}.` : `${word}. ${OFFICIAL_STATUS.notVerifiedLead}`} Before harvesting or eating seafood, check CDFW and CDPH.
            </p>
            {v && v.state !== "verified" && v.reasons.length > 0 && (
              <details className="mt-1.5 text-[13px] text-n-ink-2" data-testid="notices-reasons">
                <summary className="nx-focus inline-flex min-h-9 cursor-pointer items-center font-semibold text-n-ink">Why not verified</summary>
                <ul className="ml-4 list-disc space-y-0.5">
                  {v.reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
                {(v.reviewedAt || v.lastCheckedAt) && (
                  <p className="mt-1 text-n-ink-3" data-testid="notices-checked">
                    {v.reviewedAt ? `Records last transcribed ${clockLabel(v.reviewedAt)}${v.reviewedBy ? ` by ${v.reviewedBy}` : ""}.` : ""}
                    {v.lastCheckedAt ? ` Agency pages last checked ${clockLabel(v.lastCheckedAt)}.` : ""}
                  </p>
                )}
              </details>
            )}
            <div className="mt-2 flex gap-2">
              <a href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold">
                CDFW <Icon name="external" className="h-3.5 w-3.5" />
              </a>
              <a href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold">
                CDPH <Icon name="external" className="h-3.5 w-3.5" />
              </a>
            </div>
          </div>
          <div className="flex-1 overflow-auto">
            {disputed.length > 0 && (
              <section className="border-b border-n-line px-4 py-3" data-testid="notices-disputed">
                <h3 className="text-[13.5px] font-semibold text-n-stale">Reported lifted by the agency · awaiting review · {disputed.length}</h3>
                <ul className="mt-1 divide-y divide-n-line">
                  {disputed.map(({ record: r, dispute: d }) => (
                    <NoticeRow key={r.id} r={r} now={now} dispute={d} />
                  ))}
                </ul>
              </section>
            )}
            {local && (
              <section className="border-b border-n-line px-4 pt-3" data-testid="notices-local">
                <h3 className="text-[13.5px] font-semibold text-n-ink">
                  Cover {local.name}&rsquo;s waters · {here.length}
                </h3>
                <p className="text-[12.5px] text-n-ink-3">By the agencies&rsquo; official geography, as Neritic transcribed it. Not verified.</p>
                {here.length ? (
                  <ul className="divide-y divide-n-line">
                    {here.map((r) => (
                      <NoticeRow key={r.id} r={r} now={now} />
                    ))}
                  </ul>
                ) : (
                  <p className="py-2 text-[14px] text-n-ink-2">No notice in Neritic&rsquo;s list covers these waters. The list may be incomplete.</p>
                )}
                {maybe.length > 0 && (
                  <div data-testid="notices-unsettled">
                    <h3 className="pt-2 text-[13.5px] font-semibold text-n-stale">May apply here: not settled · {maybe.length}</h3>
                    <ul className="divide-y divide-n-line">
                      {maybe.map((r) => (
                        <NoticeRow key={r.id} r={r} now={now} />
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            )}
            {local && <h3 className="px-4 pt-3 text-[13.5px] font-semibold text-n-ink-2">Elsewhere in California · {elsewhere.length}</h3>}
            <ul className="divide-y divide-n-line px-4" data-testid="notices-current">
              {elsewhere.map((r) => (
                <NoticeRow key={r.id} r={r} now={now} />
              ))}
              {!official && <li className="py-3 text-[14px] text-n-ink-2">Neritic&rsquo;s copy of the notices could not be loaded. Check CDFW and CDPH directly.</li>}
            </ul>
            {official && official.registry.statements.length > 0 && (
              <details className="border-t border-n-line px-4 py-3 text-[13px] text-n-ink-2" data-testid="notices-statements">
                <summary className="nx-focus inline-flex min-h-9 cursor-pointer items-center font-semibold text-n-ink">{OFFICIAL_STATUS.statementsHeading}</summary>
                <p className="text-n-ink-3">{OFFICIAL_STATUS.statementsNote}</p>
                <ul className="mt-1.5 space-y-1.5">
                  {official.registry.statements.map((s) => (
                    <li key={s.id}>
                      <span className="text-n-ink-3">
                        {s.agency} · {s.topic}
                      </span>
                      <br />&ldquo;{s.statement}&rdquo;{" "}
                      <a href={s.source.url} target="_blank" rel="noreferrer" className="nx-focus font-semibold text-n-accent">
                        source
                      </a>
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
          <div className="border-t border-n-line px-4 py-3 text-[13px] leading-snug">
            <p className="font-semibold text-n-ink" data-testid="missing-not-open">
              {OFFICIAL_STATUS.missingNotOpen}
            </p>
            <dl className="mt-1.5 grid gap-0.5" data-testid="notices-hotlines">
              {OFFICIAL_STATUS.hotlines.map((h) => (
                <div key={h.tel} className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <dt className="text-n-ink-3">{h.label}</dt>
                  <dd>
                    <a href={`tel:${h.tel}`} className="nx-focus nx-data inline-flex min-h-8 items-center font-semibold text-n-ink">
                      {h.phone}
                    </a>
                  </dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      )}
    </div>
  );
}

const TOXIN: Record<string, string> = { domoic_acid: "domoic acid", psp: "paralytic shellfish poisoning toxins" };

/** One notice: what, since when, warnings always visible; the agency's wording, area and sources on demand. */
function NoticeRow({ r, now, dispute = null }: { r: OfficialRecord; now: Date | null; dispute?: Dispute | null }) {
  const critical = now || dispute ? criticalFlags(r, now ?? new Date(0), dispute) : [];
  const flags = now ? recordFlags(r, now, dispute).filter((f) => !critical.includes(f)) : [];
  return (
    <li className="py-3" data-testid={`notice-${r.id}`} data-disputed={dispute ? "true" : undefined}>
      <p className="text-[12.5px] font-semibold uppercase tracking-[0.06em] text-n-official">
        {r.agency} · {ACTION_LABEL[r.action]} · {FISHERY_LABEL[r.fishery]}
      </p>
      <p className="mt-0.5 text-[14.5px] font-semibold leading-snug">{r.title}</p>
      <p className="mt-0.5 text-[13px] text-n-ink-2" data-testid="notice-dates">
        {r.effective_date ? `Since ${dayLabel(r.effective_date)}, ${r.effective_date.slice(0, 4)}` : "Start date not published"}
        {r.expected_end_date ? ` · through at least ${dayLabel(r.expected_end_date)}` : ""}
      </p>
      {critical.length > 0 && (
        <ul className="mt-1 space-y-0.5 text-[13px] font-semibold text-n-stale" data-testid="notice-critical">
          {critical.map((f) => (
            <li key={f} className="flex gap-1.5">
              <Icon name="alert" className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{f}</span>
            </li>
          ))}
        </ul>
      )}
      {dispute && (
        <a href={dispute.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-9 items-center gap-1 text-[13px] font-semibold text-n-accent">
          {dispute.agency} {dispute.release} <Icon name="external" className="h-3.5 w-3.5" />
        </a>
      )}
      <details className="mt-1 text-[13px] text-n-ink-2">
        <summary className="nx-focus inline-flex min-h-9 cursor-pointer items-center font-semibold text-n-accent">The agency&rsquo;s wording, area and sources</summary>
        <div className="space-y-2 pt-1">
          <p>{r.summary}</p>
          <blockquote className="border-l-2 border-n-official-line pl-2 text-n-ink" data-testid="official-text">
            &ldquo;{r.official_text}&rdquo;
          </blockquote>
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5">
            <dt className="text-n-ink-3">Species</dt>
            <dd>{r.species.join(", ")}</dd>
            <dt className="text-n-ink-3">Toxin</dt>
            <dd>{r.toxins.map((t) => TOXIN[t] ?? t).join(", ")}</dd>
            <dt className="text-n-ink-3">Area</dt>
            <dd>{r.area.description}</dd>
            {r.area.geometry_note && (
              <>
                <dt className="text-n-ink-3">On the map</dt>
                <dd>
                  {r.area.geometry_basis === "none" ? "Not drawn. " : ""}
                  {r.area.geometry_note}
                </dd>
              </>
            )}
          </dl>
          {r.expected_end_note && <p className="text-n-ink-3">{r.expected_end_note}</p>}
          {flags.length > 0 && (
            <ul className="space-y-0.5 text-n-stale" data-testid="notice-flags">
              {flags.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          )}
          <ul className="flex flex-wrap gap-x-3">
            {r.sources.map((s) => (
              <li key={s.url}>
                <a href={s.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-9 items-center gap-1 font-semibold text-n-accent">
                  {s.label} <Icon name="external" className="h-3.5 w-3.5" />
                </a>
              </li>
            ))}
          </ul>
        </div>
      </details>
    </li>
  );
}
