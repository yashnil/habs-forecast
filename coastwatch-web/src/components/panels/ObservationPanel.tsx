"use client";

import type { LayerArtifact, SourceStatus } from "@/generated/schema";
import { CHLOROPHYLL_COPY } from "@/content/copy";
import { classifyTime } from "@/lib/freshness";
import { formatDate, formatDateTimePT } from "@/lib/time";
import { FreshnessBadge, ProductClassBadge } from "@/components/ui/Badges";

type Props = {
  layers: LayerArtifact[];
  status: SourceStatus | null;
  selected: string | null; // layer_id or null
  onSelect: (layerId: string | null) => void;
  now: Date | null;
};

export function ObservationPanel({ layers, status, selected, onSelect, now }: Props) {
  const active = layers.find((l) => l.layer_id === selected) ?? null;
  return (
    <section data-testid="observation-panel" aria-labelledby="obs-h" className="space-y-3 rounded-xl border border-hairline bg-surface p-4">
      <header className="space-y-1.5">
        <ProductClassBadge pc="observation" />
        <h2 id="obs-h" className="text-[15px] font-semibold tracking-tight text-ink">
          {CHLOROPHYLL_COPY.heading}
        </h2>
        <p className="text-[12px] leading-snug text-ink-2">{CHLOROPHYLL_COPY.biomass}</p>
      </header>

      {layers.length === 0 ? (
        <p data-testid="chl-unavailable" className="rounded-lg border border-dashed border-hairline-strong p-3 text-[12.5px] text-ink-2">
          Satellite chlorophyll unavailable{status?.error ? `: ${status.error}` : "."}
        </p>
      ) : (
        <div role="radiogroup" aria-label="Satellite chlorophyll layer" className="space-y-1">
          {[null, ...layers].map((l) => {
            const id = l?.layer_id ?? null;
            const checked = selected === id;
            const f = l && now ? classifyTime(l.freshness, l.time, now) : null;
            return (
              <button
                key={id ?? "none"}
                role="radio"
                aria-checked={checked}
                onClick={() => onSelect(id)}
                className={`flex w-full items-center justify-between gap-2 rounded-lg border px-3 py-2 text-left text-[12.5px] transition-colors ${
                  checked ? "border-accent/60 bg-surface-3 text-ink" : "border-hairline bg-surface-2 text-ink-2 hover:border-hairline-strong"
                }`}
              >
                <span>
                  {l ? l.short_title : "Off"}
                  {l?.time.observed_date && <span className="ml-1.5 text-ink-3 tabular">{formatDate(l.time.observed_date)}</span>}
                </span>
                {l && <FreshnessBadge f={f} compact />}
              </button>
            );
          })}
        </div>
      )}

      {active && active.tiles && (
        <div className="space-y-2" data-testid="chl-legend">
          {active.tiles.legend_verified && active.tiles.legend_url ? (
            // NASA's legend SVG is drawn for a light background
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={active.tiles.legend_url}
              alt={`NASA colour legend for ${active.title} (mg per cubic metre)`}
              className="w-full rounded-md bg-white p-1"
              loading="lazy"
            />
          ) : (
            <p className="text-[12px] text-serious">Legend unavailable — colours cannot be read quantitatively.</p>
          )}
          <ul data-testid="chl-caveats" className="list-disc space-y-1 pl-4 text-[12px] leading-snug text-ink-2">
            {active.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
          <details className="rounded-lg border border-hairline px-3 py-2 text-[11.5px] text-ink-2">
            <summary className="cursor-pointer font-medium text-ink">How the date was chosen</summary>
            <p className="mt-1.5">{active.tiles.date_selection}</p>
            <p className="mt-1.5 text-ink-3">
              {active.provenance.source_name} · layer <span className="font-mono">{active.provenance.dataset_id}</span> · checked{" "}
              {formatDateTimePT(active.provenance.retrieved_at)}
            </p>
          </details>
        </div>
      )}
    </section>
  );
}
