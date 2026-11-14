"use client";

import type { ForecastRun, ProductClass, SourceStatus } from "@/generated/schema";
import { OFFICIAL_STATUS } from "@/content/copy";
import { FRESHNESS_LABEL, classifySource, type Freshness, type FreshnessState } from "@/lib/freshness";
import { leadLabel } from "@/lib/layers";
import type { VerificationState } from "@/lib/official";
import { formatAge, formatDate, formatDateTimePT } from "@/lib/time";
import { useNow } from "@/lib/useNow";
import { Glyph, Micro, type GlyphKind } from "@/nx/ds/synth/Grammar";
import s from "./sources.module.css";

const OUTCOME: Record<string, string> = {
  updated: "Updated",
  unchanged: "No new data",
  partial: "Partly updated",
  failed: "Update failed",
};

const CLASS_LABEL: Record<ProductClass, string> = {
  official_regulatory: "Official · regulatory",
  official_forecast: "Agency forecast",
  observation: "Observation",
  experimental_model: "Experimental",
  historical_context: "Historical",
  reference: "Reference",
  derived_summary: "Derived summary",
};
/** The kind of product in the v14 grammar: a glyph per kind, the same marks as on every stamp. */
const CLASS_GLYPH: Partial<Record<ProductClass, GlyphKind>> = {
  official_regulatory: "unverified-coral",
  official_forecast: "predicted",
  observation: "measured",
  experimental_model: "predicted",
  historical_context: "old",
};
const BASIS_WORD: Record<string, string> = { issued_date: "issued", observed_date: "observed", valid_date: "updated" };

/** Freshness in words with its mark: filled for current, half for stale, a ring for historical, a cross for none. */
function FreshnessMark({ f, basis, compact }: { f: Freshness | null; basis: string; compact: boolean }) {
  const state: FreshnessState = f?.state ?? "unavailable";
  return (
    <span className={s.fresh} data-testid="freshness" data-state={state}>
      <svg viewBox="0 0 12 12" aria-hidden="true" className={s.freshMark} data-state={state}>
        {state === "current" ? <circle cx="6" cy="6" r="4.5" /> : null}
        {state === "stale" ? (
          <>
            <circle cx="6" cy="6" r="4.5" className={s.ring} />
            <path d="M6 1.5 A4.5 4.5 0 0 1 6 10.5 Z" />
          </>
        ) : null}
        {state === "historical" ? <circle cx="6" cy="6" r="4.25" className={s.ring} /> : null}
        {state === "unavailable" ? <path d="M2.5 2.5l7 7M9.5 2.5l-7 7" className={s.ring} /> : null}
      </svg>
      {f ? FRESHNESS_LABEL[state] : "Checking…"}
      {!compact && f?.ageDays != null ? (
        <span className={s.age}>
          · {BASIS_WORD[basis] ?? ""} {formatAge(f.ageDays)}
        </span>
      ) : null}
    </span>
  );
}

export interface SourceRowsProps {
  sources: SourceStatus[];
  run: ForecastRun | null;
  pipeline: { runId: string; version: string; generatedAt: string };
  /** the notice registry's state (registry.server.ts): the official row shows verification, never "current" */
  registry: VerificationState;
}

/**
 * Source status (v15.1; was src/components/SourceTable.tsx): every source the dataset reads, one row each, with its
 * product kind, its freshness (or, for the official notices, their verification), its last attempt and success, its
 * newest dates, the source's own freshness note, any failure in plain words with the technical detail on demand, and
 * C-HARM's run. Freshness is computed in the browser from the dates, so it never hydrates stale.
 */
export function SourceRows({ sources, run, pipeline, registry }: SourceRowsProps) {
  const now = useNow();
  return (
    <section className={s.table} aria-labelledby="src-h">
      <div className={s.tableHead}>
        <h2 id="src-h">Source status</h2>
        <p className={s.pipeline}>
          Pipeline run <span className={s.num}>{pipeline.runId}</span> · version <span className={s.num}>{pipeline.version}</span> · generated {formatDateTimePT(pipeline.generatedAt)}
        </p>
      </div>
      <ul className={s.rows} data-testid="source-table">
        {sources.map((src) => {
          const f = now ? classifySource(src, now) : null;
          const glyph = CLASS_GLYPH[src.product_class];
          return (
            <li key={src.source_id} className={s.row} data-testid={`source-${src.source_id}`}>
              <div className={s.rowHead}>
                <h3>{src.title}</h3>
                <span className={s.rowState}>
                  {src.source_id === "official" ? (
                    // regulatory records show verification, never a generic "current"
                    <span className={s.verify} data-testid="official-verification" data-state={now ? registry : "checking"}>
                      <Glyph kind="unverified-coral" />
                      {now ? OFFICIAL_STATUS.verification[registry] : "Checking…"}
                    </span>
                  ) : (
                    <FreshnessMark f={f} basis={src.freshness.basis} compact={src.product_class === "historical_context"} />
                  )}
                </span>
              </div>
              <Micro glyph={glyph} className={s.kind} data-testid="product-class">
                {CLASS_LABEL[src.product_class]}
              </Micro>
              <dl className={s.facts}>
                <div>
                  <dt>Last attempt</dt>
                  <dd>
                    {formatDateTimePT(src.last_attempt_at)} — {OUTCOME[src.outcome]}
                  </dd>
                </div>
                <div>
                  <dt>Last success</dt>
                  <dd>{src.last_success_at ? formatDateTimePT(src.last_success_at) : "never"}</dd>
                </div>
                {src.latest_issued_date ? (
                  <div>
                    <dt>Latest issued</dt>
                    <dd>{formatDate(src.latest_issued_date, { year: true })}</dd>
                  </div>
                ) : null}
                {src.latest_valid_date ? (
                  <div>
                    <dt>{src.freshness.basis === "observed_date" ? "Latest observed" : src.freshness.basis === "reviewed_date" ? "Last reviewed or transcribed" : "Latest valid"}</dt>
                    <dd>{formatDate(src.latest_valid_date, { year: true })}</dd>
                  </div>
                ) : null}
              </dl>
              <p className={s.note}>{src.freshness.note}</p>
              {src.error ? (
                // a plain statement first; the upstream error text is for whoever maintains the pipeline
                <div className={s.error} data-testid={`source-error-${src.source_id}`}>
                  <p>
                    {src.outcome === "failed" ? (
                      <>
                        <b>The latest update could not reach this source.</b> The last successful data stays on screen with its real dates.
                      </>
                    ) : (
                      <>
                        <b>Part of the latest update did not complete.</b> What did update is shown; the rest keeps its last successful data and dates.
                      </>
                    )}
                  </p>
                  <details>
                    <summary>Technical detail</summary>
                    <p className={s.tech}>{src.error}</p>
                  </details>
                </div>
              ) : null}
              {src.notes && src.notes.length > 0 ? (
                <details className={s.notes}>
                  <summary>Processing notes ({src.notes.length})</summary>
                  <ul>
                    {src.notes.map((n) => (
                      <li key={n}>{n}</li>
                    ))}
                  </ul>
                </details>
              ) : null}
              {src.source_id === "charm" && run ? (
                <p className={s.run}>
                  Run issued {formatDate(run.issued_date, { year: true })}
                  {run.issued_date_derived ? " (date inferred: NOAA does not publish it)" : ""}: {run.leads_available.map(leadLabel).join(", ")} available
                  {run.leads_missing.length ? `; ${run.leads_missing.map(leadLabel).join(", ")} missing` : ""}.
                </p>
              ) : null}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
