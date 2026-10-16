"use client";

import { Mark, type MarkKind } from "@/components/ui/Mark";
import type { EvidenceFocus, EvidenceGraph, EvidenceNode } from "@/lib/harborEvidence";
import { formatDateTimePT } from "@/lib/time";

const ext = { target: "_blank", rel: "noopener noreferrer" } as const;

const KIND: Record<EvidenceNode["class"], MarkKind> = { measurement: "measured", record_statistic: "measured", satellite_retrieval: "satellite", published_forecast: "model", reconstruction: "model", experimental: "model" };
const REP: Record<EvidenceNode["where"]["representative"], string> = { at: "at these waters", near: "nearby", context: "context only", far: "too far to describe these waters" };
const CLASS_WORD: Record<EvidenceNode["class"], string> = { measurement: "measured", record_statistic: "station record", satellite_retrieval: "satellite", published_forecast: "forecast", reconstruction: "reconstructed", experimental: "experimental" };

function when(n: EvidenceNode): string {
  const t = n.time.observed ?? n.time.valid ?? n.time.issued;
  if (!t) return "";
  if (t.length > 10) return formatDateTimePT(t);
  const d = new Date(`${t}T12:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
  const age = n.time.ageDays;
  return age == null || age === 0 ? d : `${d} · ${age === 1 ? "1 day" : `${age} days`} ago`;
}

const isSea = (n: EvidenceNode) => n.id.startsWith("buoy:") || n.id.startsWith("coops:") || n.id.startsWith("nws:") || n.id.startsWith("cencoos:") || n.id.startsWith("tides:");

function focusOf(n: EvidenceNode): EvidenceFocus {
  if (n.id.startsWith("pn_cells:")) return "cells";
  if (n.id.startsWith("pda:") || n.id.startsWith("record:pDA:")) return "toxin";
  if (n.id.startsWith("record:pn:")) return "cells";
  if (n.id.startsWith("charm:")) return "model";
  if (n.id.startsWith("chl:")) return "satellite";
  if (n.id.startsWith("official:") || n.id.startsWith("statement:")) return "official";
  if (n.id.startsWith("buoy:") || n.id.startsWith("coops:") || n.id.startsWith("nws:") || n.id.startsWith("cencoos:") || n.id.startsWith("tides:")) return "conditions";
  return null;
}

/**
 * The evidence, one row per node of the graph: value · where and how far · when, with the class
 * of knowledge (measured, satellite, forecast) and how well the place is described by it. Rows
 * open their source on the map or at the source's page. Gaps are listed last, as gaps.
 */
export function EvidenceList({ graph, focus, onFocus }: { graph: EvidenceGraph; focus: EvidenceFocus; onFocus: (f: EvidenceFocus) => void }) {
  const rows = graph.nodes.filter((n) => !n.id.startsWith("official:") && !n.id.startsWith("statement:") && !isSea(n));
  const sea = graph.nodes.filter(isSea);
  const n = rows.length + (sea.length ? 1 : 0);
  return (
    <section aria-labelledby="evidence-list-h" data-testid="evidence-list">
      <h2 id="evidence-list-h" className="font-display text-[24px] font-medium leading-tight text-ink sm:text-[26px]">
        The evidence
      </h2>
      <p className="mt-1 text-[15px] leading-snug text-ink-2">Each reading with where it was made, how far that is from these waters, and when. Seawater is not seafood; a missing reading is not a clean result.</p>
      {graph.gaps.length > 0 && (
        <ul className="mt-3 space-y-1.5" data-testid="evidence-gaps">
          {graph.gaps.map((g) => (
            <li key={g.id} className="relative pl-7 text-[15px] leading-snug text-ink-2" data-testid={`gap-${g.id}`}>
              <Mark kind="gap" className="absolute left-[9px] top-[6px]" />
              {g.reason}
            </li>
          ))}
        </ul>
      )}
      <details className="group mt-3" data-testid="evidence-readings">
        <summary className="cw-press flex min-h-[44px] cursor-pointer list-none items-center gap-2 text-[15px] font-semibold text-accent [&::-webkit-details-marker]:hidden">
          <span aria-hidden className="text-ink-3 transition-transform group-open:rotate-90">▸</span>
          Every reading ({n})
        </summary>
      <ul className="mt-1 divide-y divide-hairline border-y border-hairline">
        {rows.map((n) => {
          const f = focusOf(n);
          const on = !!f && f === focus;
          const km = n.where.km;
          return (
            <li key={n.id} className={`relative py-3 pl-7 ${on ? "bg-accent/[0.06]" : ""}`} data-testid={`evidence-${n.id}`} data-node={n.id} data-class={n.class} data-representative={n.where.representative}>
              <Mark kind={KIND[n.class]} aged={n.freshness.state === "stale" || n.freshness.state === "historical"} className="absolute left-[9px] top-[19px]" />
              <p className="text-[17px] leading-snug text-ink">
                <span className="font-semibold tabular">{n.text}</span> <span className="text-[15px] text-ink-2">{n.variable}</span>
              </p>
              <p className="mt-0.5 text-[14.5px] leading-snug text-ink-2">
                {n.where.name ?? n.source.name}
                {km > 0 ? `, ${km < 10 ? km.toFixed(1) : Math.round(km)} km${n.where.bearing ? ` ${n.where.bearing}` : ""}` : ""}
                {" · "}
                {when(n)} · {CLASS_WORD[n.class]}
                {n.support.nativeM ? ` ${n.support.nativeM >= 1000 ? `${Math.round(n.support.nativeM / 1000)} km` : `${n.support.nativeM} m`}` : ""} · {REP[n.where.representative]}
              </p>
              <p className="flex flex-wrap items-center gap-x-4 text-[14px] text-ink-3">
                {n.source.url && (
                  <a href={n.source.url} {...ext} className="inline-flex min-h-[44px] min-w-[44px] items-center font-medium text-accent hover:underline">
                    {n.source.name} ↗
                  </a>
                )}
                {f && (
                  <button type="button" onClick={() => onFocus(on ? null : f)} aria-pressed={on} className="inline-flex min-h-[44px] min-w-[44px] items-center font-medium text-ink-3 hover:text-ink" data-testid={`evidence-${n.id}-map`}>
                    {on ? "Showing on map" : "Show on map"}
                  </button>
                )}
              </p>
            </li>
          );
        })}
        {sea.length > 0 && (
          <li className={`relative py-3 pl-7 ${focus === "conditions" ? "bg-accent/[0.06]" : ""}`} data-testid="evidence-sea" data-nodes={sea.map((n) => n.id).join(" ")}>
            <Mark kind="conditions" className="absolute left-[9px] top-[19px]" />
            <p className="text-[17px] leading-snug text-ink">
              <span className="font-semibold">Sea conditions</span> <span className="text-[15px] text-ink-2">measured and forecast, not a HAB or fishing forecast</span>
            </p>
            <ul className="mt-0.5 space-y-0.5 text-[14.5px] leading-snug text-ink-2">
              {sea.map((n) => (
                <li key={n.id} data-node={n.id}>
                  <span className="font-medium text-ink">{n.text}</span> {n.variable} · {n.where.name}
                  {n.where.km > 0 ? `, ${n.where.km < 10 ? n.where.km.toFixed(1) : Math.round(n.where.km)} km` : ""} · {when(n)}
                </li>
              ))}
            </ul>
            <p className="flex flex-wrap items-center gap-x-4 text-[14px] text-ink-3">
              <button type="button" onClick={() => onFocus(focus === "conditions" ? null : "conditions")} aria-pressed={focus === "conditions"} className="inline-flex min-h-[44px] min-w-[44px] items-center font-medium text-ink-3 hover:text-ink" data-testid="evidence-sea-map">
                {focus === "conditions" ? "Showing on map" : "Show on map"}
              </button>
            </p>
          </li>
        )}
      </ul>
      </details>
    </section>
  );
}
