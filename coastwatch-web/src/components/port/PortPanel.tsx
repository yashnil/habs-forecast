"use client";

import type { Manifest } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntel, PortIntelCollection } from "@/generated/port_intel";
import { PORT_COPY } from "@/content/copy";
import { OfficialForPort } from "@/components/official/Official";
import { TimeSeriesChart } from "@/components/charts/TimeSeriesChart";
import { Notice, Segmented, SourceLink } from "@/components/ui/Primitives";
import { colorAt } from "@/components/ui/ProbabilityLegend";
import { charmLayer, leadLabel, CHARM_VARIABLES, type CharmVariable } from "@/lib/layers";
import type { Verification } from "@/lib/official";
import { formatDate, formatDateTimePT } from "@/lib/time";

const VAR_LABEL: Record<CharmVariable, string> = {
  pseudo_nitzschia: "Pseudo-nitzschia bloom",
  particulate_domoic: "Particulate domoic acid",
  cellular_domoic: "Cellular domoic acid",
};
const VAR_SHORT: Record<CharmVariable, string> = { pseudo_nitzschia: "Bloom", particulate_domoic: "Particulate DA", cellular_domoic: "Cellular DA" };
const pct = (v: number) => `${Math.round(v * 100)}%`;

type Props = {
  port: PortIntel;
  coll: PortIntelCollection;
  manifest: Manifest;
  official: OfficialDataset | null;
  verification: Verification | null;
  lead: number;
  onLead: (l: number) => void;
  variable: CharmVariable;
  onVariable: (v: CharmVariable) => void;
  now: Date | null;
  onClose: () => void;
};

export function PortPanel({ port, coll, manifest, official, verification, lead, onLead, variable, onVariable, now, onClose }: Props) {
  const ch = port.charm;
  const leadSummary = ch?.leads.find((l) => l.lead_days === lead) ?? null;
  const palette = charmLayer(manifest, variable, lead)?.palette ?? charmLayer(manifest, "pseudo_nitzschia", 0)?.palette;
  const hist = (ch?.history?.[variable] ?? []) as { date: string; value: number | null; n: number }[];
  const chl = port.chlorophyll;
  const threshold = charmLayer(manifest, variable, lead)?.threshold_text;

  return (
    <article data-testid="port-panel" aria-labelledby="port-h" className="space-y-4">
      <header className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[10.5px] font-semibold uppercase tracking-[0.08em] text-accent">Port</p>
          <h2 id="port-h" className="font-display text-[28px] font-medium leading-tight text-ink">
            {port.display_name}
          </h2>
          <p className="text-[12px] text-ink-3">
            {port.county} County · {port.lat.toFixed(3)}°N, {Math.abs(port.lon).toFixed(3)}°W · CDFW port code {port.port_code}
          </p>
        </div>
        <button onClick={onClose} aria-label="Close port details" className="rounded px-2 py-1 text-[13px] text-ink-3 hover:bg-surface-3 hover:text-ink">
          ✕
        </button>
      </header>

      <OfficialForPort ds={official} v={verification} port={port} now={now} />

      <section className="space-y-2.5" data-testid="port-forecast" aria-labelledby="pf-h">
        <div className="flex items-baseline justify-between gap-2">
          <h3 id="pf-h" className="text-[13px] font-semibold text-ink">
            Agency forecast near this port
          </h3>
          <span className="text-[11px] text-ink-3">C-HARM v3.1{ch?.issued_date ? ` · issued ${formatDate(ch.issued_date)} (inferred)` : ""}</span>
        </div>
        {!ch || !ch.leads.length ? (
          <Notice tone="neutral">No forecast available.</Notice>
        ) : (
          <>
            <Segmented
              label="Forecast valid day"
              value={lead}
              onChange={onLead}
              testidPrefix="port-lead"
              options={ch.leads.map((l) => ({ value: l.lead_days, label: leadLabel(l.lead_days), sub: formatDate(l.valid_date).replace(/^\w+, /, "") }))}
            />
            {ch.cells_in_radius === 0 ? (
              <Notice tone="neutral" testid="port-no-cells">
                No C-HARM forecast cells within {ch.radius_km} km of this port.
              </Notice>
            ) : (
              <ul className="space-y-2">
                {CHARM_VARIABLES.map((v) => {
                  const s = leadSummary?.variables[v];
                  return (
                    <li key={v} data-testid={`port-stat-${v}`}>
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="text-[12.5px] text-ink-2">{VAR_LABEL[v]}</span>
                        <span className="text-[14px] font-semibold text-ink tabular" data-testid={`port-median-${v}`}>
                          {s?.median != null ? pct(s.median) : "—"}
                        </span>
                      </div>
                      {s?.median != null && palette && (
                        <div className="relative mt-1 h-1.5 rounded-full bg-surface-3">
                          <div
                            className="absolute inset-y-0 rounded-full bg-[var(--cw-ink-3)] opacity-40"
                            style={{ left: `${s.min! * 100}%`, width: `${Math.max(1, (s.max! - s.min!) * 100)}%` }}
                            aria-hidden
                          />
                          <div className="absolute top-1/2 h-3 w-1 -translate-x-1/2 -translate-y-1/2 rounded-sm" style={{ left: `${s.median * 100}%`, background: palette ? colorAt(palette, s.median) : "var(--cw-ink)" }} aria-hidden />
                        </div>
                      )}
                      <p className="mt-0.5 text-[11px] text-ink-3">
                        {s && s.n > 0 ? `Median of ${s.n} cells; range ${pct(s.min!)}–${pct(s.max!)}` : "No cells with a value within 15 km"}
                      </p>
                    </li>
                  );
                })}
              </ul>
            )}
            <p className="text-[11.5px] text-ink-3">
              {PORT_COPY.spatial}
              {ch.nearest_cell_km != null ? ` Nearest forecast cell: ${ch.nearest_cell_km.toFixed(1)} km.` : ""} {PORT_COPY.nearshore}
            </p>
          </>
        )}
      </section>

      <section className="space-y-2" data-testid="port-history" aria-labelledby="ph-h">
        <div className="flex items-baseline justify-between gap-2">
          <h3 id="ph-h" className="text-[13px] font-semibold text-ink">
            Last 30 days of nowcasts
          </h3>
        </div>
        <Segmented
          label="History variable"
          value={variable}
          onChange={onVariable}
          options={CHARM_VARIABLES.map((v) => ({ value: v, label: VAR_SHORT[v] }))}
        />
        {ch?.history_error ? (
          <Notice tone="warning" title="History unavailable" testid="history-unavailable">
            {ch.history_error}
          </Notice>
        ) : (
          <TimeSeriesChart
            label={`${VAR_LABEL[variable]} near ${port.display_name}, last 30 days`}
            points={hist}
            domain={[0, 1]}
            ticks={[0, 0.5, 1]}
            format={pct}
            unit={threshold ? `Median probability · ${threshold.replace(/^Probability that /, "")}` : "Median probability"}
            color="var(--cw-forecast)"
            maxGapDays={1}
          />
        )}
        <p className="text-[11px] text-ink-3">{PORT_COPY.history}</p>
      </section>

      <section className="space-y-2" data-testid="port-chlorophyll" aria-labelledby="pc-h">
        <div className="flex items-baseline justify-between gap-2">
          <h3 id="pc-h" className="text-[13px] font-semibold text-ink">
            Satellite chlorophyll near this port
          </h3>
          <span className="rounded border border-hairline-strong px-1 text-[10px] font-semibold uppercase tracking-wider text-ink-2">Observation</span>
        </div>
        {!chl ? (
          <Notice tone="neutral">Not available.</Notice>
        ) : chl.error && !chl.latest ? (
          <Notice tone="warning" title="Satellite values unavailable" testid="chl-port-unavailable">
            {chl.error}
          </Notice>
        ) : !chl.latest ? (
          <Notice tone="neutral" testid="chl-port-cloudy">
            No clear-sky pixels within {chl.radius_km} km in the last 60 days of composites.
          </Notice>
        ) : (
          <>
            <div className="flex items-baseline gap-2">
              <span className="text-[20px] font-semibold text-ink tabular" data-testid="chl-port-latest">
                {chl.latest.median!.toFixed(2)}
              </span>
              <span className="text-[12px] text-ink-2">mg m⁻³ median</span>
            </div>
            <p className="text-[11.5px] text-ink-3">
              8-day composite centred on {formatDate(chl.latest_center_date!, { year: true })} · {chl.latest.n} clear pixels (
              {Math.round((chl.latest_valid_fraction ?? 0) * 100)}% of pixels within {chl.radius_km} km; land and cloud excluded) · range{" "}
              {chl.latest.min!.toFixed(2)}–{chl.latest.max!.toFixed(2)}
            </p>
            <TimeSeriesChart
              label={`Satellite chlorophyll near ${port.display_name}, last 60 days`}
              points={chl.history ?? []}
              domain={[0.1, 30]}
              scale="log"
              ticks={[0.1, 1, 10]}
              format={(v) => (v < 1 ? v.toFixed(1) : v.toFixed(v < 10 ? 1 : 0))}
              unit="mg m⁻³ (log scale), median of clear pixels"
              color="var(--cw-chl)"
              maxGapDays={1}
            />
          </>
        )}
        <p className="text-[11px] text-ink-3">{PORT_COPY.chlorophyll}</p>
      </section>

      <details className="rounded-md border border-hairline px-3 py-2 text-[11.5px] text-ink-2">
        <summary className="cursor-pointer font-medium text-ink">How these numbers are made</summary>
        <ul className="mt-1.5 list-disc space-y-1 pl-4">
          {Object.entries(coll.method).map(([k, v]) => (
            <li key={k}>{v}</li>
          ))}
          {port.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
        <p className="mt-1.5 text-ink-3">
          Built {formatDateTimePT(coll.generated_at)} ·{" "}
          {coll.provenance.map((p, i) => (
            <span key={p.source_id}>
              {i ? " · " : ""}
              <SourceLink href={p.source_url}>{p.source_name}</SourceLink>
            </span>
          ))}
        </p>
      </details>
    </article>
  );
}
