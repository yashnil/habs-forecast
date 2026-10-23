"use client";

import { useState } from "react";
import type { StationLite } from "@/lib/data";
import type { Claim, EvidenceGraph } from "@/lib/harborEvidence";
import type { Selection } from "@/v4/state/selection";
import { nodeSelection } from "@/v4/explore/Inspect";
import { Icon } from "@/v4/ui/Icon";

/**
 * "What matters now" (v6): at most three statements, one line each, the rest one tap away; each opens in
 * place to the agency's words, the detail, its source and "Show on map". The rows carry the same data
 * attributes as Explore's ClaimRow (claim id, kind, tone, supports, [data-headline]) so the briefing, the
 * harbour panel and /api/evidence can be compared statement by statement.
 */

const KIND: Record<string, { word: string; tone: string; dot: string; hollow: string }> = {
  official: { word: "Official", tone: "text-n-official", dot: "bg-n-official", hollow: "border-2 border-n-official" },
  agency_page: { word: "Agency page · Neritic's copy", tone: "text-n-official", dot: "border-2 border-n-official", hollow: "border-2 border-n-official" },
  measured: { word: "Measured", tone: "text-n-measured", dot: "bg-n-measured", hollow: "border-2 border-n-measured" },
  model: { word: "Model", tone: "text-n-model", dot: "bg-n-model", hollow: "border-2 border-n-model" },
  satellite: { word: "Satellite", tone: "text-n-measured", dot: "bg-n-measured", hollow: "border-2 border-n-measured" },
  conditions: { word: "Conditions", tone: "text-n-conditions", dot: "bg-n-conditions", hollow: "border-2 border-n-conditions" },
  gap: { word: "Not known", tone: "text-n-ink-3", dot: "bg-n-gap", hollow: "border-2 border-n-gap" },
};

export const MATTERS_SHOWN = 3;

function MatterRow({ c, graph, stations, onSelect, fallback, hidden }: { c: Claim; graph: EvidenceGraph; stations: StationLite[] | null; onSelect: (s: Selection) => void; fallback: Selection | null; hidden: boolean }) {
  const [open, setOpen] = useState(false);
  const k = KIND[c.kind] ?? KIND.gap;
  const nodes = new Map(graph.nodes.map((n) => [n.id, n]));
  const target = (c.supports ?? []).map((id) => nodeSelection(nodes.get(id), stations)).find(Boolean) ?? fallback;
  const body = `claim-body-${c.id}`;
  return (
    <li hidden={hidden} className={c.tone === "alert" ? "-ml-3 border-l-2 border-n-official pl-[10px]" : ""} data-testid={`claim-${c.id}`} data-kind={c.kind} data-tone={c.tone} data-supports={(c.supports ?? []).join(" ")} data-open={open}>
      <button type="button" aria-expanded={open} aria-controls={body} onClick={() => setOpen((o) => !o)} className="nx-tint nx-focus flex min-h-11 w-full items-start gap-3 rounded-lg py-2 text-left" data-testid={`claim-open-${c.id}`}>
        <span className={`mt-[8px] h-2 w-2 shrink-0 rounded-full ${c.aged ? k.hollow : k.dot}`} data-aged={c.aged ? "true" : undefined} aria-hidden />
        <span className="min-w-0 flex-1 text-[16px] font-medium leading-6 text-n-ink" data-headline>
          {c.headline}
        </span>
        <span className="mt-[3px] shrink-0 text-[12.5px] font-semibold text-n-ink-3 max-sm:sr-only">{k.word}</span>
        <Icon name="chevronDown" className={`mt-0.5 h-4 w-4 shrink-0 text-n-ink-3 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      <div id={body} hidden={!open} className="pb-2 pl-[22px]">
        {c.quote && (
          <blockquote className="mt-1 border-l-2 border-n-official-line pl-2.5 font-serif text-[15.5px] leading-snug text-n-ink" data-testid="claim-quote">
            &ldquo;{c.quote}&rdquo;
          </blockquote>
        )}
        {c.also && (
          <p className="mt-1.5 text-[14px] leading-snug text-n-ink-2" data-testid="claim-also">
            <span className="font-semibold text-n-ink">{c.also.label}:</span> &ldquo;{c.also.quote}&rdquo;{" "}
            <a href={c.also.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
              source <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          </p>
        )}
        {c.detail && <p className="mt-0.5 text-[14px] leading-snug text-n-ink-2">{c.detail}</p>}
        {c.history && (
          <p className="mt-0.5 text-[14px] leading-snug text-n-ink-2" data-testid={`claim-${c.id}-history`}>
            {c.history}
          </p>
        )}
        {c.provenance && (
          <p className="mt-0.5 text-[13px] leading-snug text-n-ink-3" data-testid={`claim-${c.id}-provenance`}>
            {c.provenance}
          </p>
        )}
        {c.agencyWords && c.agencyWords.length > 0 && (
          <details className="group mt-0.5 text-[14px]" data-testid={`claim-${c.id}-agency`}>
            <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center font-semibold text-n-accent">
              <span className="group-open:hidden">What the agencies&rsquo; pages say</span>
              <span className="hidden group-open:inline">Hide the agencies&rsquo; words</span>
            </summary>
            <ul className="flex flex-col gap-2">
              {c.agencyWords.map((w) => (
                <li key={w.url + w.quote.slice(0, 24)} data-testid={`claim-${c.id}-word`} data-agency={w.agency}>
                  <blockquote className="border-l-2 border-n-official-line pl-2.5 font-serif text-[15.5px] leading-snug text-n-ink">&ldquo;{w.quote}&rdquo;</blockquote>
                  <a href={w.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 text-[13px] font-semibold text-n-accent">
                    {w.label} <Icon name="external" className="h-3.5 w-3.5" />
                  </a>
                </li>
              ))}
            </ul>
          </details>
        )}
        <div className="flex flex-wrap items-center gap-x-3 text-[13px]">
          <span className="font-semibold text-n-ink-3 sm:hidden">{k.word}</span>
          {c.source?.url && (
            <a href={c.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
              {c.source.label} <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          )}
          {target && (
            <button type="button" onClick={() => onSelect(target)} className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-ink-2 hover:text-n-ink" data-testid={`claim-show-${c.id}`}>
              <Icon name="locate" className="h-3.5 w-3.5" /> Show on map
            </button>
          )}
        </div>
      </div>
    </li>
  );
}

export function Matters({ claims, graph, stations, onSelect, fallback }: { claims: Claim[]; graph: EvidenceGraph; stations: StationLite[] | null; onSelect: (s: Selection) => void; fallback: Selection | null }) {
  const [all, setAll] = useState(false);
  const more = claims.length - MATTERS_SHOWN;
  return (
    <>
      <ul className="flex flex-col" data-testid="matters">
        {claims.map((c, i) => (
          <MatterRow key={c.id} c={c} graph={graph} stations={stations} onSelect={onSelect} fallback={fallback} hidden={!all && i >= MATTERS_SHOWN} />
        ))}
      </ul>
      {more > 0 && (
        <button type="button" aria-expanded={all} onClick={() => setAll((a) => !a)} className="nx-focus inline-flex min-h-11 items-center self-start text-[14.5px] font-semibold text-n-accent" data-testid="matters-more">
          {all ? "Fewer statements" : `${more} more ${more === 1 ? "statement" : "statements"}`}
        </button>
      )}
    </>
  );
}
