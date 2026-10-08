"use client";

import { useState } from "react";
import type { OfficialDataset, OfficialRecord } from "@/generated/official";
import type { PortIntel } from "@/generated/port_intel";
import { OFFICIAL_STATUS } from "@/content/copy";
import { ACTION_LABEL, FISHERY_LABEL, activeRecords, criticalFlags, recordFlags, type Verification } from "@/lib/official";
import { formatDate, formatDateTimePT } from "@/lib/time";
import { Notice, SourceLink } from "@/components/ui/Primitives";

const V_STYLE: Record<Verification["state"], { color: string; icon: React.ReactNode }> = {
  verified: { color: "var(--cw-good)", icon: <path d="M3 6.2l2 2 4-4.4" stroke="currentColor" strokeWidth="1.8" fill="none" strokeLinecap="round" strokeLinejoin="round" /> },
  aging: { color: "var(--cw-warning)", icon: <path d="M6 3.2v3l2 1.2" stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round" /> },
  unverified: { color: "var(--cw-serious)", icon: <path d="M6 3v3.6M6 8.6v.1" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /> },
  unavailable: { color: "var(--cw-neutral)", icon: <path d="M3.5 3.5l5 5M8.5 3.5l-5 5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /> },
};

export function VerificationBadge({ v }: { v: Verification | null }) {
  const state = v?.state ?? "unavailable";
  const st = V_STYLE[state];
  return (
    <span
      data-testid="official-verification"
      data-state={v ? state : "checking"}
      className="inline-flex items-center gap-1.5 rounded border border-hairline-strong bg-surface-2 px-1.5 py-0.5 text-[11px] font-semibold text-ink"
    >
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden style={{ color: st.color }}>
        <circle cx="6" cy="6" r="5.3" fill="none" stroke="currentColor" strokeWidth="1.2" />
        {st.icon}
      </svg>
      {v ? OFFICIAL_STATUS.verification[state] : "Checking…"}
    </span>
  );
}

export function VerificationDetail({ v, brief = false }: { v: Verification; brief?: boolean }) {
  if (brief && v.state !== "verified") {
    return (
      <details className="rounded-md border border-serious/40 bg-serious/[0.06] px-2.5 py-1.5 text-[11.5px] text-ink-2" data-testid="official-not-verified-brief">
        <summary className="cursor-pointer text-ink">
          <span className="font-semibold">{OFFICIAL_STATUS.verification[v.state]}:</span> {OFFICIAL_STATUS.notVerifiedLead}
        </summary>
        <ul className="mt-1 list-disc space-y-0.5 pl-4">
          {v.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      </details>
    );
  }
  if (v.state === "verified") {
    return (
      <p className="text-[11.5px] text-ink-3">
        Reviewed by {v.reviewedBy} on {formatDateTimePT(v.reviewedAt!)}.
      </p>
    );
  }
  return (
    <Notice tone={v.state === "aging" ? "warning" : "serious"} title={OFFICIAL_STATUS.notVerifiedLead} testid="official-not-verified">
      <ul className="list-disc space-y-0.5 pl-4">
        {v.reasons.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>
      {v.reviewedAt && (
        <p className="mt-1 text-[11px] text-ink-3">
          Records last {v.state === "aging" ? "reviewed" : "transcribed"} {formatDateTimePT(v.reviewedAt)} by {v.reviewedBy}.
          {v.lastCheckedAt ? ` Official pages last checked ${formatDateTimePT(v.lastCheckedAt)}.` : ""}
        </p>
      )}
    </Notice>
  );
}

const AGENCY_STYLE: Record<string, string> = {
  CDFW: "border-[#ffb547]/60 text-[#ffcf85]",
  CDPH: "border-[#ffb547]/60 text-[#ffcf85]",
  OEHHA: "border-[#ffb547]/60 text-[#ffcf85]",
};

export function NoticeCard({ r, now, relationNote, compact = false }: { r: OfficialRecord; now: Date | null; relationNote?: string; compact?: boolean }) {
  const [open, setOpen] = useState(!compact);
  const critical = now ? criticalFlags(r, now) : [];
  const flags = now ? recordFlags(r, now).filter((f) => !critical.includes(f)) : [];
  return (
    <article data-testid={`notice-${r.id}`} className="rounded-md border border-[#ffb547]/25 bg-[#ffb547]/[0.035]">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full items-start gap-2 px-2.5 py-2 text-left hover:bg-[#ffb547]/[0.05]"
      >
        <span className={`mt-px shrink-0 rounded border px-1 text-[10px] font-bold tracking-wide ${AGENCY_STYLE[r.agency]}`}>{r.agency}</span>
        <span className="min-w-0 flex-1">
          <span className="block text-[10.5px] font-semibold uppercase tracking-wide text-[#ffcf85]">
            {ACTION_LABEL[r.action]} · {FISHERY_LABEL[r.fishery]}
          </span>
          <span className="block text-[12.5px] font-semibold leading-snug text-ink">{r.title}</span>
          <span className="mt-0.5 block text-[11px] text-ink-3">
            {r.effective_date ? `Since ${formatDate(r.effective_date, { year: true })}` : "Start date not published"}
            {r.expected_end_date ? ` · through at least ${formatDate(r.expected_end_date)}` : ""}
            {relationNote ? ` · ${relationNote}` : ""}
          </span>
        </span>
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className={`mt-0.5 shrink-0 text-ink-3 transition-transform duration-150 ${open ? "rotate-90" : ""}`}>
          <path d="M5 3l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {critical.length > 0 && (
        <ul className="space-y-0.5 px-2.5 pb-2 text-[11.5px] font-medium text-warning" data-testid="notice-critical">
          {critical.map((f) => (
            <li key={f}>⚠ {f}</li>
          ))}
        </ul>
      )}
      {open && (
        <div className="space-y-2 border-t border-[#ffb547]/15 px-2.5 py-2 text-[12px] text-ink-2">
          <p>{r.summary}</p>
          <blockquote className="border-l-2 border-[#ffb547]/50 pl-2 text-ink" data-testid="official-text">
            “{r.official_text}”
          </blockquote>
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11.5px]">
            <dt className="text-ink-3">Species</dt>
            <dd>{r.species.join(", ")}</dd>
            <dt className="text-ink-3">Toxin</dt>
            <dd>{r.toxins.map((t) => (t === "domoic_acid" ? "domoic acid" : "paralytic shellfish poisoning toxins")).join(", ")}</dd>
            <dt className="text-ink-3">Area</dt>
            <dd>{r.area.description}</dd>
            {r.area.geometry_note && (
              <>
                <dt className="text-ink-3">On the map</dt>
                <dd>
                  {r.area.geometry_basis === "none" ? "Not drawn. " : ""}
                  {r.area.geometry_note}
                </dd>
              </>
            )}
          </dl>
          {r.expected_end_note && <p className="text-[11.5px] text-ink-3">{r.expected_end_note}</p>}
          {flags.length > 0 && (
            <ul className="space-y-0.5 text-[11.5px] text-warning" data-testid="notice-flags">
              {flags.map((f) => (
                <li key={f}>⚠ {f}</li>
              ))}
            </ul>
          )}
          <ul className="space-y-0.5 text-[11.5px]">
            {r.sources.map((s) => (
              <li key={s.url}>
                <SourceLink href={s.url}>{s.label} ↗</SourceLink>
              </li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}

export function Hotlines() {
  return (
    <dl className="grid gap-1 text-[12px]">
      {OFFICIAL_STATUS.hotlines.map((h) => (
        <div key={h.tel} className="flex flex-wrap justify-between gap-x-3">
          <dt className="text-ink-3">{h.label}</dt>
          <dd>
            <a href={`tel:${h.tel}`} className="font-medium text-ink tabular hover:underline">
              {h.phone}
            </a>
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** Rank-1 summary in the left rail: verification, every active notice, statements, hotlines. */
export function OfficialSummary({ ds, v, now, error }: { ds: OfficialDataset | null; v: Verification | null; now: Date | null; error: string | null }) {
  const [showAll, setShowAll] = useState(false);
  const records = activeRecords(ds);
  const shown = showAll ? records : records.slice(0, 4);
  return (
    <section data-testid="official-status" aria-labelledby="official-h" className="space-y-2.5 rounded-lg border border-[#ffb547]/30 bg-surface p-3.5">
      <div className="flex items-center justify-between gap-2">
        <h2 id="official-h" className="text-[13.5px] font-semibold tracking-tight text-ink">
          {OFFICIAL_STATUS.heading}
        </h2>
        <VerificationBadge v={v} />
      </div>
      {!ds ? (
        <Notice tone="serious" title="Official notices unavailable" testid="official-unavailable">
          {error ? `${error}. ` : ""}Use the official sources below.
        </Notice>
      ) : (
        <>
          {v && <VerificationDetail v={v} />}
          <p className="text-[12px] text-ink-2">
            <span className="font-semibold text-ink tabular">{records.length}</span> active notices statewide from CDFW and CDPH.
          </p>
          <div className="space-y-1.5">
            {shown.map((r) => (
              <NoticeCard key={r.id} r={r} now={now} compact />
            ))}
          </div>
          {records.length > 4 && (
            <button onClick={() => setShowAll(!showAll)} className="w-full rounded-md border border-hairline py-1.5 text-[12px] text-ink-2 hover:text-ink" data-testid="official-show-all">
              {showAll ? "Show fewer" : `Show all ${records.length} notices`}
            </button>
          )}
          {ds.registry.statements.length > 0 && (
            <details className="rounded-md border border-hairline px-3 py-2 text-[12px] text-ink-2">
              <summary className="cursor-pointer font-medium text-ink">{OFFICIAL_STATUS.statementsHeading}</summary>
              <p className="mt-1 text-[11px] text-ink-3">{OFFICIAL_STATUS.statementsNote}</p>
              <ul className="mt-1.5 space-y-1.5">
                {ds.registry.statements.map((s) => (
                  <li key={s.id}>
                    <p className="text-[11px] text-ink-3">
                      {s.agency} · {s.topic}
                    </p>
                    <p>“{s.statement}” <SourceLink href={s.source.url}>source ↗</SourceLink></p>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
      <p className="text-[11.5px] leading-snug text-ink-3" data-testid="missing-not-open">
        {OFFICIAL_STATUS.missingNotOpen}
      </p>
      <details className="text-[12px] text-ink-2">
        <summary className="cursor-pointer font-medium text-ink">Official sources and hotlines</summary>
        <ul className="mt-1.5 space-y-1">
          {OFFICIAL_STATUS.links.map((l) => (
            <li key={l.href}>
              <SourceLink href={l.href}>{l.label} ↗</SourceLink>
            </li>
          ))}
        </ul>
        <div className="mt-2 border-t border-hairline pt-2">
          <Hotlines />
        </div>
      </details>
    </section>
  );
}

/** Notices related to a port (from the pipeline's relation table). */
export function OfficialForPort({ ds, v, port, now }: { ds: OfficialDataset | null; v: Verification | null; port: PortIntel; now: Date | null }) {
  const byId = new Map((ds?.registry.records ?? []).map((r) => [r.id, r]));
  const rel = port.official_relations.filter((x) => byId.get(x.record_id)?.status === "active");
  return (
    <section data-testid="port-official" className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-[13px] font-semibold text-ink">Official notices for nearby waters</h3>
        <VerificationBadge v={v} />
      </div>
      {v && v.state !== "verified" && <VerificationDetail v={v} brief />}
      {!ds ? (
        <Notice tone="serious" title="Official notices unavailable">Check CDFW and CDPH directly.</Notice>
      ) : rel.length ? (
        <div className="space-y-1.5">
          {rel.map((x) => (
            <NoticeCard key={x.record_id} r={byId.get(x.record_id)!} now={now} relationNote={x.note} compact />
          ))}
        </div>
      ) : (
        <Notice tone="neutral" testid="port-no-notices">
          No active notices in CoastWatch&apos;s list mention this port&apos;s county or latitude.
        </Notice>
      )}
      <p className="text-[11.5px] text-ink-3">{OFFICIAL_STATUS.missingNotOpen}</p>
    </section>
  );
}
