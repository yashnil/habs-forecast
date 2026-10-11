"use client";

import type { OfficialDataset } from "@/generated/official";
import { VerificationBadge } from "@/components/official/Official";
import { useOfficialDrawer } from "@/components/shell/OfficialShell";
import { Icon } from "@/components/ui/Icon";
import { ACTION_LABEL } from "@/lib/official";

/**
 * Official notices first, in one strip (spec §3.2 item 1, rule 1): the given notices (on Bloom, those
 * whose stated area may include waters near the station's port; on Fisheries, those the species
 * groups reference), quoted by their registry titles (never truncated), the verification state, and
 * the full list one click away. Official notices take precedence over everything below them.
 */
export function OfficialStrip({
  official,
  error,
  recordIds,
  heading,
  emptyText = "No active notice in CoastWatch’s list names this area.",
  compact = false,
  testid = "bloom-official",
}: {
  official: OfficialDataset | null;
  error: string | null;
  recordIds: string[];
  /** the heading before the count, e.g. "Official notices for waters near Santa Cruz"; "{n}" places the count inside it */
  heading: string;
  emptyText?: string;
  compact?: boolean;
  testid?: string;
}) {
  const { openDrawer, count, verification } = useOfficialDrawer();
  const records = new Map((official?.registry.records ?? []).map((r) => [r.id, r]));
  const rel = [...new Set(recordIds)].filter((id) => records.get(id)?.status === "active");
  const unverified = verification?.state !== "verified";
  return (
    <section className="rounded-lg border border-official-line bg-official-bg px-4 py-2.5" aria-labelledby={`${testid}-h`} data-testid={testid}>
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 id={`${testid}-h`} className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[14px] font-semibold text-official-ink">
          <Icon name="shield" className="h-4 w-4 text-official" />
          {!official ? "Official notices unavailable" : heading.includes("{n}") ? heading.replace("{n}", String(rel.length)) : `${heading}: ${rel.length}`}
          {official && <VerificationBadge v={verification ?? null} />}
        </h2>
        <button type="button" onClick={(e) => openDrawer(e.currentTarget)} className="text-[13.5px] font-medium text-official-ink underline-offset-2 hover:underline" data-testid={`${testid}-all`}>
          All {count} notices ›
        </button>
      </div>
      {!official ? (
        <p className="mt-1 text-[13.5px] text-ink-2">{error ? `${error}. ` : ""}Check CDFW and CDPH directly before harvesting or eating seafood.</p>
      ) : rel.length && compact ? (
        <details className="mt-1.5 text-[13.5px]">
          <summary className="cursor-pointer font-medium text-official-ink">Show the {rel.length} notices</summary>
          <NoticeList rel={rel} records={records} testid={`${testid}-list`} />
        </details>
      ) : rel.length ? (
        <NoticeList rel={rel} records={records} testid={`${testid}-list`} />
      ) : (
        <p className="mt-1 text-[13.5px] text-ink-2">{emptyText}</p>
      )}
      <p className="mt-1 text-[13px] text-ink-3">
        {compact || !unverified ? "" : "Transcribed from CDFW and CDPH, not yet checked by a person. "}A missing notice does not mean an area is open or seafood is safe.
      </p>
    </section>
  );
}

function NoticeList({ rel, records, testid }: { rel: string[]; records: Map<string, NonNullable<OfficialDataset["registry"]["records"]>[number]>; testid: string }) {
  return (
    <ul className="mt-1 grid gap-x-6 gap-y-0.5 text-[13.5px] leading-snug text-ink md:grid-cols-2" data-testid={testid}>
      {rel.map((id) => {
        const r = records.get(id)!;
        return (
          <li key={id} data-testid={`notice-${r.id}`}>
            <span className="mr-1.5 font-mono text-[12px] font-semibold text-official">{r.agency.toUpperCase()}</span>
            <span className="text-official-ink">{ACTION_LABEL[r.action]}:</span> {r.title}
          </li>
        );
      })}
    </ul>
  );
}
