"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import type { OfficialDataset, OfficialRecord } from "@/generated/official";
import type { SourceStatus } from "@/generated/schema";
import { OFFICIAL_STATUS } from "@/content/copy";
import { ACTION_LABEL, FISHERY_LABEL, criticalFlags, currentRecords, disputedRecords, officialVerification, recordFlags, type Dispute, type Verification } from "@/lib/official";
import { Icon } from "@/v4/ui/Icon";
import { SourceList, useCalmAnnouncement } from "@/v4/readiness/StatusCenter";
import { OVERALL_WORDS, overall, useReadinessSnapshot, useReportSource, type Overall } from "@/v4/readiness/store";
import { exitProps, usePresence } from "@/v4/motion";
import { clockLabel, dayLabel } from "@/v4/ui/time";

/**
 * The trust pill (design.md §7.1): one control on every page saying what Neritic is and that official
 * notices are not verified, in words at every width; a clock when any source is old. It opens one
 * panel: the full disclosure, the agencies, each notice with its own wording and dates, and the data
 * status. Verification is computed in the browser from dates, so a stale review loses its status even
 * if the pipeline stops; notices stay "not verified" until a person reviews the registry
 * (cwp review-official).
 *
 * Test ids live on it: notices-button (the button), registry-disclosure (its words), status-center
 * (the readiness state, with data-state), notices-panel and status-panel (the panel).
 */
/** `local`: a place is open (a harbour, a briefing, Today's port): the notices whose official geography
 *  covers its waters, and those whose reach there is not settled (lib/applicability), led by name and count. */
export type LocalNotices = { name: string; covering: string[]; unsettled: string[] };

/** What the pill says and opens; `loadTrust()` (lib/data) provides the first three on the server. */
export type Trust = { official: OfficialDataset | null; status: SourceStatus | null; fixture?: boolean; local?: LocalNotices | null; generatedAt?: string | null };

/** A page without a map (Today, Ask, account, legal) says its dataset is in hand, so the pill's readiness settles. */
export function useReportTrust(trust: Trust | null) {
  useReportSource(trust?.official ? { id: "data", label: "Neritic dataset", state: "ready", detail: trust.generatedAt ? `Published ${trust.generatedAt.slice(0, 16).replace("T", " ")} UTC` : undefined, asOf: trust.generatedAt ?? null } : trust === null ? null : { id: "data", label: "Neritic dataset", state: "unavailable", detail: "Could not be loaded" });
}

/** Readiness states worth saying on the pill itself; the rest stay behind it. */
const SHOWN: Partial<Record<Overall, string>> = { stale: "clock", outdated: "clock", offline: "offline", unavailable: "alert", incomplete: "alert", downloading: "download" };

function Sep({ className = "" }: { className?: string }) {
  return (
    <span aria-hidden className={`nx-trust-sep ${className}`}>
      ·
    </span>
  );
}

export function TrustPill({ trust, now: nowProp }: { trust: Trust | null; now: Date | null }) {
  const official = trust?.official ?? null;
  const status = trust?.status ?? null;
  const local = trust?.local ?? null;
  const fixture = !!trust?.fixture;
  const [open, setOpen] = useState(false);
  const pop = usePresence(open);
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => setNow(new Date()), []);
  const box = useRef<HTMLDivElement | null>(null);
  const button = useRef<HTMLButtonElement | null>(null);
  const recs = official ? currentRecords(official) : [];
  const meta = official ? noticeMeta(official) : null;
  // with a place open: its own notices first (current records only: a disputed one is never counted)
  const here = local ? recs.filter((r) => local.covering.includes(r.id)) : [];
  const maybe = local ? recs.filter((r) => local.unsettled.includes(r.id) && !local.covering.includes(r.id)) : [];
  const elsewhere = local ? recs.filter((r) => !here.includes(r) && !maybe.includes(r)) : recs;
  // an agency release appears to lift these: never counted or shown as in effect, always shown with the release
  const disputed = official ? disputedRecords(official) : [];
  const v: Verification | null = now ? officialVerification(official, status, now) : null;
  const state = !official ? "unavailable" : (v?.state ?? "unverified");
  const word = OFFICIAL_STATUS.verification[state];
  // the data status
  const snap = useReadinessSnapshot();
  const data = overall(snap);
  const said = useCalmAnnouncement(data, snap.online);
  const dataIcon = SHOWN[data];
  const pack = snap.sources.pack;
  const count = official ? (local ? here.length : recs.length) : null;
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      // used here: the selection behind the panel stays open
      e.preventDefault();
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
  const disputedSection = disputed.length > 0 && (
    <section className="border-b border-n-line px-5 py-3" data-testid="notices-disputed">
      <h3 className="text-[13.5px] font-semibold text-n-stale">Status unclear · awaiting review · {disputed.length}</h3>
      <p className="text-[13px] text-n-ink-3">An agency release appears to lift {disputed.length === 1 ? "this notice" : "these notices"}. Until a person reviews it, Neritic shows {disputed.length === 1 ? "it" : "them"} as neither in effect nor lifted.</p>
      <ul className="mt-1 divide-y divide-n-line">
        {disputed.map(({ record: r, dispute: d }) => (
          <NoticeRow key={r.id} r={r} now={now} dispute={d} meta={meta} />
        ))}
      </ul>
    </section>
  );
  const currentList = (
    <ul className="divide-y divide-n-line px-5" data-testid="notices-current">
      {elsewhere.map((r) => (
        <NoticeRow key={r.id} r={r} now={now} meta={meta} />
      ))}
      {!official && <li className="py-3 text-[14px] text-n-ink-2">Neritic&rsquo;s copy of the notices could not be loaded. Check CDFW and CDPH directly.</li>}
    </ul>
  );
  const dataOpen = data !== "ready" && data !== "initializing" && data !== "partial";
  return (
    <div ref={box} className="relative min-w-0">
      <button ref={button} type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-haspopup="dialog" data-testid="notices-button" data-state={state} className="nx-trust nx-press nx-focus nx-tint">
        <Icon name="shieldOff" className="h-4 w-4 text-n-ink-2 max-md:hidden" />
        <span data-testid="registry-disclosure" className="flex min-w-0 flex-col items-start md:flex-row md:items-center md:gap-1.5">
          {/* a phone: two short lines; wider: one line */}
          <span className="flex max-w-full items-center gap-1 overflow-hidden md:contents">
            {fixture && (
              <>
                <span className="text-n-official" role="status" data-testid="fixture-banner">
                  Test data<span className="sr-only">: recorded fixture data, not a live feed</span>
                </span>
                <Sep />
              </>
            )}
            <span className="nx-trust-soft hidden xl:inline" data-testid="preview-disclosure">
              Research preview
            </span>
            <span className="nx-trust-soft xl:hidden" data-testid="preview-disclosure-short">
              <span className="sr-only">Research </span>Preview
            </span>
            {/* the readiness state: words on the pill when something is old or missing, else for screen readers only */}
            {dataIcon && <Sep className="md:hidden" />}
            <span data-testid="status-center" data-state={data} className={`inline-flex min-w-0 items-center gap-1 md:order-last ${dataIcon ? "md:ml-1.5" : ""}`}>
              {dataIcon ? (
                <>
                  <Icon name={dataIcon} className={`h-[15px] w-[15px] ${data === "offline" ? "text-n-offline" : data === "stale" || data === "outdated" ? "text-n-stale" : "text-n-down"}`} />
                  <span className="truncate">{OVERALL_WORDS[data]}</span>
                </>
              ) : (
                <span className="sr-only">Status: {OVERALL_WORDS[data]}</span>
              )}
            </span>
          </span>
          <span className="flex items-center gap-1 whitespace-nowrap md:contents">
            <Sep className="max-md:hidden" />
            <span className="inline-flex items-baseline gap-1">
              <span className="tabular" data-testid="notices-count">
                {official ? count : "?"}
              </span>
              <span className="max-lg:hidden">{count === 1 ? "notice" : "notices"}</span>
              {local && official && <span className="max-[359px]:sr-only" data-testid="notices-here">here</span>}
              {/* without a place, the count is statewide and says so (it never reads as this place's count) */}
              {!local && official && <span className="max-[359px]:sr-only" data-testid="notices-statewide">in CA</span>}
            </span>
            <Sep />
            <span className="font-semibold">{word}</span>
          </span>
        </span>
        {/* polite and only on settled changes of the overall state */}
        <span className="sr-only" role="status" aria-live="polite">
          {said}
        </span>
      </button>
      {pop.mounted && (
        <div
          role="dialog"
          aria-label="Official notices"
          // on a phone the panel spans the screen under the top bar; elsewhere it hangs from the pill
          className="nx-pop fixed inset-x-3 top-[64px] z-50 flex max-h-[calc(100dvh-80px)] flex-col rounded-2xl bg-n-surface text-n-ink shadow-[var(--nx-shadow)] md:absolute md:inset-x-auto md:right-0 md:top-[52px] md:max-h-[min(78vh,720px)] md:w-[460px]"
          {...exitProps(pop.state, "notices-panel")}
        >
          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
            <div className="px-5 pb-4 pt-5">
              {/* with a place open, the heading leads with that place's count; the statewide count follows */}
              <p className="font-serif text-[22px] leading-[26px] text-n-ink" data-testid="notices-heading">
                {local && official ? `${here.length} ${here.length === 1 ? "notice covers" : "notices cover"} ${local.name}’s waters · ${recs.length} in California` : `${recs.length} active notices in California`}
              </p>
              <p className="mt-2 text-[14px] leading-[20px] text-n-ink" data-testid="notices-unverified" data-state={state}>
                {state === "verified" ? `Checked by a person against the agencies' pages${v?.reviewedAt ? ` on ${clockLabel(v.reviewedAt)}` : ""}.` : `${word}. ${OFFICIAL_STATUS.notVerifiedLead}`} Before harvesting or eating seafood, check CDFW and CDPH.
              </p>
              <p className="mt-2 text-[13.5px] leading-[20px] text-n-ink-2" data-testid="preview-statement">
                <Link href="/about" className="nx-focus font-semibold text-n-ink underline decoration-n-line-strong underline-offset-[3px]">
                  Research preview
                </Link>
                , not an official source and not a seafood-safety service. The list of notices may be incomplete.
              </p>
              {v && v.state !== "verified" && v.reasons.length > 0 && (
                <details className="mt-2 text-[13px] text-n-ink-2" data-testid="notices-reasons">
                  <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center font-medium text-n-ink underline decoration-dotted decoration-n-line-strong underline-offset-[3px]">Why not verified</summary>
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
              <div className="mt-3 flex gap-2">
                <a href="https://wildlife.ca.gov/Fishing/Ocean/Health-Advisories" target="_blank" rel="noreferrer" className="nx-btn nx-btn-secondary nx-press nx-focus">
                  CDFW <Icon name="external" className="h-3.5 w-3.5" />
                </a>
                <a href="https://www.cdph.ca.gov/Programs/OPA/Pages/Shellfish-Advisories.aspx" target="_blank" rel="noreferrer" className="nx-btn nx-btn-secondary nx-press nx-focus">
                  CDPH <Icon name="external" className="h-3.5 w-3.5" />
                </a>
              </div>
            </div>
            {/* the data status: folded when everything is in hand, open when something is old or missing */}
            <details className="group border-y border-n-line" data-testid="status-panel" open={dataOpen || undefined}>
              <summary className="nx-focus flex min-h-11 cursor-pointer items-center justify-between gap-3 px-5 text-[14px] font-medium text-n-ink">
                <span className="flex items-center gap-1.5">
                  {dataIcon && <Icon name={dataIcon} className={`h-4 w-4 ${data === "offline" ? "text-n-offline" : data === "stale" || data === "outdated" ? "text-n-stale" : "text-n-down"}`} />}
                  Data status · {OVERALL_WORDS[data]}
                </span>
                <span className="flex items-center gap-2 text-[13px] font-normal text-n-ink-2">
                  <Icon name={snap.online ? "online" : "offline"} className="h-4 w-4" />
                  {snap.online ? "Online" : "Offline"}
                  <Icon name="chevronDown" className="h-4 w-4 text-n-ink-3 transition-transform group-open:rotate-180" />
                </span>
              </summary>
              <div className="px-5 pb-3">
                <p className="text-[13.5px] text-n-ink-2">Each source keeps its own time. Nothing older is shown as current.</p>
                {!snap.online && (
                  <p className="mt-1 text-[13.5px] font-semibold text-n-ink" data-testid="status-offline-note">
                    {pack && (pack.state === "cached" || pack.state === "ready") && pack.asOf ? `Showing the field pack saved ${clockLabel(pack.asOf)}; nothing here is live.` : "Only pages already open keep working; nothing here is live."} Confirm official notices with CDFW and CDPH when you are back online.
                  </p>
                )}
                <div className="mt-1 max-h-[34dvh] overflow-y-auto overscroll-contain">
                  <SourceList now={nowProp} compact />
                </div>
                <Link href="/sources" className="nx-focus mt-1 inline-flex min-h-11 items-center gap-1 text-[14px] font-medium text-n-ink underline decoration-n-line-strong underline-offset-[3px]">
                  Full data status <Icon name="chevronRight" className="h-4 w-4" />
                </Link>
              </div>
            </details>
            {!local && disputedSection}
            {local && (
              <section className="border-b border-n-line px-5 pt-3" data-testid="notices-local">
                <h3 className="text-[14px] font-semibold text-n-ink">
                  Cover {local.name}&rsquo;s waters · {here.length}
                </h3>
                <p className="text-[13px] text-n-ink-3">By the agencies&rsquo; official geography, as Neritic transcribed it. Not verified.</p>
                {here.length ? (
                  <ul className="divide-y divide-n-line">
                    {here.map((r) => (
                      <NoticeRow key={r.id} r={r} now={now} meta={meta} />
                    ))}
                  </ul>
                ) : (
                  <p className="py-2 text-[14px] text-n-ink-2">No notice in Neritic&rsquo;s list covers these waters. The list may be incomplete.</p>
                )}
                {maybe.length > 0 && (
                  <div data-testid="notices-unsettled">
                    <h3 className="pt-2 text-[14px] font-semibold text-n-stale">May apply here: not settled · {maybe.length}</h3>
                    <ul className="divide-y divide-n-line">
                      {maybe.map((r) => (
                        <NoticeRow key={r.id} r={r} now={now} meta={meta} />
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            )}
            {local && disputedSection}
            {/* with a place open, the rest of California is one tap away (never removed) */}
            {local ? (
              <details className="group border-b border-n-line" data-testid="notices-elsewhere">
                <summary className="nx-focus flex min-h-11 cursor-pointer items-center justify-between px-5 text-[14px] font-medium text-n-ink">
                  Elsewhere in California · {elsewhere.length}
                  <Icon name="chevronDown" className="h-4 w-4 text-n-ink-3 transition-transform group-open:rotate-180" />
                </summary>
                {currentList}
              </details>
            ) : (
              currentList
            )}
            {official && official.registry.statements.length > 0 && (
              <details className="border-t border-n-line px-5 py-3 text-[13px] text-n-ink-2" data-testid="notices-statements">
                <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center font-medium text-n-ink">{OFFICIAL_STATUS.statementsHeading}</summary>
                <p className="text-n-ink-3">{OFFICIAL_STATUS.statementsNote}</p>
                <ul className="mt-1.5 space-y-1.5">
                  {official.registry.statements.map((s) => (
                    <li key={s.id}>
                      <span className="text-n-ink-3">
                        {s.agency} · {s.topic}
                      </span>
                      <br />&ldquo;{s.statement}&rdquo;{" "}
                      <a href={s.source.url} target="_blank" rel="noreferrer" className="nx-focus text-n-accent">
                        source
                      </a>
                    </li>
                  ))}
                </ul>
              </details>
            )}
            <div className="border-t border-n-line px-5 py-3 text-[13px] leading-snug">
              <p className="font-semibold text-n-ink" data-testid="missing-not-open">
                {OFFICIAL_STATUS.missingNotOpen}
              </p>
              <dl className="mt-1.5 grid gap-0.5" data-testid="notices-hotlines">
                {OFFICIAL_STATUS.hotlines.map((h) => (
                  <div key={h.tel} className="flex flex-wrap items-baseline justify-between gap-x-3">
                    <dt className="text-n-ink-3">{h.label}</dt>
                    <dd>
                      <a href={`tel:${h.tel}`} className="nx-focus tabular inline-flex min-h-11 items-center font-semibold text-n-ink">
                        {h.phone}
                      </a>
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const TOXIN: Record<string, string> = { domoic_acid: "domoic acid", psp: "paralytic shellfish poisoning toxins" };

/** The registry's own dates and review state: when Neritic transcribed it, who did, when each agency's pages were last fetched. */
type NoticeMeta = { transcribedAt: string; reviewedByPerson: boolean; reviewedBy: string; checked: Record<string, string> };
function noticeMeta(o: OfficialDataset): NoticeMeta {
  const rv = o.registry.review;
  const checked: Record<string, string> = {};
  // the watcher's latest successful fetch of each agency's pages (source ids are "cdph_…", "cdfw_…")
  for (const w of o.watch ?? []) {
    const agency = w.source_id.split("_")[0].toUpperCase();
    if (w.ok && (!checked[agency] || w.checked_at > checked[agency])) checked[agency] = w.checked_at;
  }
  return { transcribedAt: rv.reviewed_at, reviewedByPerson: rv.status !== "pending_human_review", reviewedBy: rv.reviewed_by, checked };
}

/** One notice: what, since when, warnings always visible; the agency's wording, area and sources on demand. */
function NoticeRow({ r, now, dispute = null, meta = null }: { r: OfficialRecord; now: Date | null; dispute?: Dispute | null; meta?: NoticeMeta | null }) {
  const published = r.sources.filter((x) => x.published_date);
  const critical = now || dispute ? criticalFlags(r, now ?? new Date(0), dispute) : [];
  const flags = now ? recordFlags(r, now, dispute).filter((f) => !critical.includes(f)) : [];
  return (
    <li className="py-3" data-testid={`notice-${r.id}`} data-disputed={dispute ? "true" : undefined}>
      <p className="text-[13px] font-medium text-n-official">
        {r.agency} · {ACTION_LABEL[r.action]} · {FISHERY_LABEL[r.fishery]}
      </p>
      <p className="mt-0.5 text-[14.5px] font-semibold leading-snug">{r.title}</p>
      <p className="mt-0.5 text-[13px] text-n-ink-2" data-testid="notice-dates">
        {r.effective_date ? `Since ${dayLabel(r.effective_date, false, now)}` : "Start date not published"}
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
        <a href={dispute.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 text-[13px] font-medium text-n-accent">
          {dispute.agency} {dispute.release} <Icon name="external" className="h-3.5 w-3.5" />
        </a>
      )}
      <details className="mt-1 text-[13px] text-n-ink-2">
        <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center font-medium text-n-accent">The agency&rsquo;s wording, area and sources</summary>
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
          {/* three different things, never merged: the agency's dates, Neritic's retrieval, and whether a person reviewed it */}
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 border-t border-n-line pt-2" data-testid="notice-provenance">
            <dt className="text-n-ink-3">Took effect</dt>
            <dd>{r.effective_date ? `${dayLabel(r.effective_date, false, now)} (the agency's date)` : "not stated by the agency"}</dd>
            <dt className="text-n-ink-3">Agency published</dt>
            <dd data-testid="notice-published">{published.length ? published.map((x) => `${dayLabel(x.published_date!, false, now)} (${x.label.split(":")[0]})`).join("; ") : "date not stated on the agency's page"}</dd>
            {meta && (
              <>
                <dt className="text-n-ink-3">Neritic transcribed</dt>
                <dd data-testid="notice-retrieved">
                  {dayLabel(meta.transcribedAt, false, now)}
                  {meta.checked[r.agency] ? `; ${r.agency}'s pages last checked ${clockLabel(meta.checked[r.agency])}` : ""}
                </dd>
                <dt className="text-n-ink-3">Reviewed by a person</dt>
                <dd className={meta.reviewedByPerson ? "" : "font-semibold text-n-stale"} data-testid="notice-review">
                  {meta.reviewedByPerson ? `Yes (${meta.reviewedBy}, ${dayLabel(meta.transcribedAt, false, now)})` : "No. Transcribed by an AI assistant; not yet checked by a person against the agency's page"}
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
                <a href={s.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 font-medium text-n-accent">
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
