"use client";

import type { Manifest } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import { officialVerification } from "@/lib/official";
import { VerificationBadge } from "@/components/official/Official";
import { classifySource } from "@/lib/freshness";
import { charmRun, leadLabel } from "@/lib/layers";
import { formatDate, formatDateTimePT } from "@/lib/time";
import { useNow } from "@/lib/useNow";
import { FreshnessBadge, ProductClassBadge } from "@/components/ui/Badges";

const OUTCOME: Record<string, string> = {
  updated: "Updated",
  unchanged: "No new data",
  partial: "Partly updated",
  failed: "Update failed",
};

export function SourceTable({ manifest, official }: { manifest: Manifest; official: OfficialDataset | null }) {
  const now = useNow();
  const run = charmRun(manifest);
  return (
    <section className="space-y-3" aria-labelledby="src-h">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 id="src-h" className="text-[15px] font-semibold text-ink">
          Source status
        </h2>
        <p className="text-[12px] text-ink-3">
          Pipeline run <span className="font-mono">{manifest.pipeline_run_id}</span> · version{" "}
          <span className="font-mono">{manifest.pipeline_version}</span> · generated {formatDateTimePT(manifest.generated_at)}
        </p>
      </div>
      <ul className="space-y-2" data-testid="source-table">
        {manifest.sources.map((s) => {
          const f = now ? classifySource(s, now) : null;
          return (
            <li key={s.source_id} className="rounded-xl border border-hairline bg-surface p-4" data-testid={`source-${s.source_id}`}>
              <div className="flex flex-wrap items-center gap-2">
                <ProductClassBadge pc={s.product_class} />
                <h3 className="text-[14px] font-semibold text-ink">{s.title}</h3>
                <span className="ml-auto">
                  {s.source_id === "official" ? (
                    // regulatory records show verification, never a generic "current" badge
                    <VerificationBadge v={now ? officialVerification(official, s, now) : null} />
                  ) : (
                    <FreshnessBadge f={f} basis={s.freshness.basis} compact={s.product_class === "historical_context"} />
                  )}
                </span>
              </div>
              <dl className="mt-2 grid gap-x-6 gap-y-1 text-[12.5px] sm:grid-cols-2">
                <Row k="Last attempt" v={`${formatDateTimePT(s.last_attempt_at)} — ${OUTCOME[s.outcome]}`} />
                <Row k="Last success" v={s.last_success_at ? formatDateTimePT(s.last_success_at) : "never"} />
                {s.latest_issued_date && <Row k="Latest issued" v={formatDate(s.latest_issued_date, { year: true })} />}
                {s.latest_valid_date && (
                  <Row
                    k={s.freshness.basis === "observed_date" ? "Latest observed" : s.freshness.basis === "reviewed_date" ? "Last reviewed or transcribed" : "Latest valid"}
                    v={formatDate(s.latest_valid_date, { year: true })}
                  />
                )}
              </dl>
              <p className="mt-2 text-[12px] text-ink-3">{s.freshness.note}</p>
              {s.error && (
                <p className="mt-2 break-words rounded-md border border-serious/40 bg-surface-2 p-2 text-[12px] text-ink-2">
                  <span className="font-semibold text-serious">Error: </span>
                  <span className="font-mono text-[11px]">{s.error}</span>
                </p>
              )}
              {s.notes && s.notes.length > 0 && (
                <details className="mt-2 text-[12px] text-ink-2">
                  <summary className="cursor-pointer text-ink-3">Processing notes ({s.notes.length})</summary>
                  <ul className="mt-1 list-disc space-y-0.5 pl-4 font-mono text-[11px]">
                    {s.notes.map((n) => (
                      <li key={n}>{n}</li>
                    ))}
                  </ul>
                </details>
              )}
              {s.source_id === "charm" && run && (
                <p className="mt-2 text-[12px] text-ink-2">
                  Run issued {formatDate(run.issued_date, { year: true })}
                  {run.issued_date_derived ? " (inferred)" : ""}: {run.leads_available.map(leadLabel).join(", ")} available
                  {run.leads_missing.length ? `; ${run.leads_missing.map(leadLabel).join(", ")} missing` : ""}.
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex gap-2">
      <dt className="text-ink-3">{k}</dt>
      <dd className="text-ink">{v}</dd>
    </div>
  );
}
