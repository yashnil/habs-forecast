"use client";

import type { ForecastRun, LayerArtifact, Manifest, SourceStatus } from "@/generated/schema";
import { FORECAST_COPY } from "@/content/copy";
import { classifyTime, type Freshness } from "@/lib/freshness";
import { CHARM_LEADS, CHARM_VARIABLES, charmLayer, leadLabel, type CharmVariable } from "@/lib/layers";
import { formatDate, formatDateTimePT, relativeDay } from "@/lib/time";
import { FreshnessBadge, ProductClassBadge } from "@/components/ui/Badges";

type Props = {
  manifest: Manifest;
  run: ForecastRun | null;
  status: SourceStatus | null;
  variable: CharmVariable;
  lead: number;
  onVariable: (v: CharmVariable) => void;
  onLead: (l: number) => void;
  shown: boolean;
  onShow: (on: boolean) => void;
  opacity: number;
  onOpacity: (o: number) => void;
  now: Date | null;
};

const VAR_SHORT: Record<CharmVariable, string> = {
  pseudo_nitzschia: "Bloom",
  particulate_domoic: "Particulate DA",
  cellular_domoic: "Cellular DA",
};

export function ForecastPanel(p: Props) {
  const layer = charmLayer(p.manifest, p.variable, p.lead);
  const anyLayer = layer ?? CHARM_LEADS.map((l) => charmLayer(p.manifest, p.variable, l)).find(Boolean) ?? null;
  const fresh: Freshness | null = p.now && anyLayer ? classifyTime(anyLayer.freshness, anyLayer.time, p.now) : p.now ? { state: "unavailable", basisDate: null, ageDays: null } : null;

  return (
    <section data-testid="forecast-panel" aria-labelledby="forecast-h" className="space-y-3 rounded-xl border border-hairline bg-surface p-4">
      <header className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <ProductClassBadge pc="official_forecast" />
          <FreshnessBadge f={fresh} basis={anyLayer?.freshness.basis} />
        </div>
        <h2 id="forecast-h" className="text-[15px] font-semibold tracking-tight text-ink">
          {FORECAST_COPY.heading}
        </h2>
        <p className="text-[12px] text-ink-3">{FORECAST_COPY.productLine}</p>
      </header>

      {!p.run || !anyLayer ? (
        <ForecastUnavailable status={p.status} />
      ) : (
        <>
          <RunLine run={p.run} status={p.status} fresh={fresh} />

          <fieldset>
            <legend className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-ink-3">Forecast</legend>
            <div role="radiogroup" className="grid grid-cols-3 gap-1 rounded-lg bg-surface-2 p-1">
              {CHARM_VARIABLES.map((v) => (
                <button
                  key={v}
                  role="radio"
                  aria-checked={p.variable === v}
                  onClick={() => p.onVariable(v)}
                  className={`rounded-md px-2 py-1.5 text-[12px] font-medium transition-colors ${
                    p.variable === v ? "bg-surface-3 text-ink shadow-sm ring-1 ring-hairline-strong" : "text-ink-2 hover:text-ink"
                  }`}
                >
                  {VAR_SHORT[v]}
                </button>
              ))}
            </div>
          </fieldset>

          <fieldset>
            <legend className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-ink-3">Valid day</legend>
            <div role="radiogroup" className="grid grid-cols-4 gap-1">
              {CHARM_LEADS.map((l) => {
                const lyr = charmLayer(p.manifest, p.variable, l);
                const missing = !lyr;
                const active = p.lead === l;
                return (
                  <button
                    key={l}
                    role="radio"
                    aria-checked={active}
                    disabled={missing}
                    data-testid={`lead-${l}`}
                    onClick={() => p.onLead(l)}
                    title={missing ? "Not issued in this run" : undefined}
                    className={`rounded-lg border px-1.5 py-1.5 text-left transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${
                      active ? "border-forecast/70 bg-surface-3" : "border-hairline bg-surface-2 hover:border-hairline-strong"
                    }`}
                  >
                    <span className="block text-[10px] font-medium uppercase tracking-wide text-ink-3">{leadLabel(l)}</span>
                    <span className="block text-[12px] font-semibold text-ink tabular">
                      {lyr?.time.valid_date ? formatDate(lyr.time.valid_date).replace(/^\w+, /, "") : "—"}
                    </span>
                    <span className="block text-[10px] text-ink-3">
                      {missing ? "not issued" : lyr?.time.valid_date && p.now ? relativeDay(lyr.time.valid_date, p.now) : lyr?.time.valid_date ? formatDate(lyr.time.valid_date).split(",")[0] : ""}
                    </span>
                  </button>
                );
              })}
            </div>
          </fieldset>

          {layer && (
            <div className="space-y-3">
              <p className="text-[12.5px] leading-snug text-ink-2" data-testid="valid-line">
                <span className="font-medium text-ink">{layer.title}</span> · valid{" "}
                <span className="tabular">{formatDate(layer.time.valid_date!, { year: true })}</span>
              </p>
              <p className="text-[12px] leading-snug text-ink-2">{layer.threshold_text} The colour key is on the map.</p>
              <Controls shown={p.shown} onShow={p.onShow} opacity={p.opacity} onOpacity={p.onOpacity} />
              <Caveats layer={layer} />
            </div>
          )}
        </>
      )}
    </section>
  );
}

function RunLine({ run, status, fresh }: { run: ForecastRun; status: SourceStatus | null; fresh: Freshness | null }) {
  return (
    <div className="space-y-1 rounded-lg bg-surface-2 px-3 py-2 text-[12px] text-ink-2" data-testid="run-line">
      <p>
        Issued <span className="font-medium text-ink tabular">{formatDate(run.issued_date, { year: true })}</span>
        {run.issued_date_derived && <span className="text-ink-3"> (inferred)</span>}
        {run.leads_missing.length > 0 && (
          <span className="text-ink-3"> · {run.leads_missing.map(leadLabel).join(", ")} not issued in this run</span>
        )}
      </p>
      {fresh?.state === "stale" && (
        <p className="text-warning" data-testid="stale-note">
          No newer C-HARM run has been published. This is the newest available forecast; its dates are shown as issued.
        </p>
      )}
      {fresh?.state === "historical" && (
        <p className="font-medium text-serious" data-testid="historical-note">
          This forecast is out of date. It is shown for reference only and does not describe current conditions.
        </p>
      )}
      {status?.outcome === "failed" && (
        <p className="text-serious" data-testid="update-failed">
          Latest update attempt failed ({formatDateTimePT(status.last_attempt_at)}). Showing the last successful run.
        </p>
      )}
    </div>
  );
}

function ForecastUnavailable({ status }: { status: SourceStatus | null }) {
  return (
    <div data-testid="forecast-unavailable" className="space-y-1 rounded-lg border border-dashed border-hairline-strong p-3 text-[12.5px] text-ink-2">
      <p className="font-medium text-ink">Forecast unavailable</p>
      <p>No C-HARM forecast could be loaded, so none is shown. This does not mean conditions are normal.</p>
      {status?.error && <p className="break-words text-[11px] text-ink-3">Last error: {status.error}</p>}
    </div>
  );
}

function Controls({ shown, onShow, opacity, onOpacity }: { shown: boolean; onShow: (b: boolean) => void; opacity: number; onOpacity: (n: number) => void }) {
  return (
    <div className="flex items-center gap-3">
      <label className="flex cursor-pointer items-center gap-2 text-[12px] text-ink-2">
        <input type="checkbox" checked={shown} onChange={(e) => onShow(e.target.checked)} className="accent-[var(--cw-forecast)]" />
        Show on map
      </label>
      <label className="flex flex-1 items-center gap-2 text-[11px] text-ink-3">
        Opacity
        <input
          type="range"
          min={0.2}
          max={1}
          step={0.05}
          value={opacity}
          disabled={!shown}
          onChange={(e) => onOpacity(Number(e.target.value))}
          className="flex-1 accent-[var(--cw-forecast)]"
          aria-label="Forecast layer opacity"
        />
      </label>
    </div>
  );
}

export function Caveats({ layer }: { layer: LayerArtifact }) {
  return (
    <div className="space-y-2">
      <div data-testid="forecast-caveats" className="rounded-lg border border-hairline bg-surface-2/60 px-3 py-2 text-[12px] leading-snug text-ink-2">
        <p className="mb-1 text-[11px] font-medium uppercase tracking-wider text-ink-3">Read before using</p>
        <ul className="list-disc space-y-1 pl-4">
          {layer.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      </div>
    <details className="group rounded-lg border border-hairline px-3 py-2 text-[12px] text-ink-2">
      <summary className="cursor-pointer list-none font-medium text-ink marker:hidden [&::-webkit-details-marker]:hidden">
        <span className="group-open:hidden">▸</span>
        <span className="hidden group-open:inline">▾</span> Source and provenance
      </summary>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
        <dt className="text-ink-3">Source</dt>
        <dd>
          <a className="text-accent hover:underline" href={layer.provenance.source_url} target="_blank" rel="noreferrer">
            {layer.provenance.source_name}
          </a>
        </dd>
        {layer.provenance.dataset_id && (
          <>
            <dt className="text-ink-3">Dataset</dt>
            <dd className="font-mono">{layer.provenance.dataset_id}</dd>
          </>
        )}
        {layer.time.valid_time && (
          <>
            <dt className="text-ink-3">Valid time</dt>
            <dd className="font-mono">{layer.time.valid_time}</dd>
          </>
        )}
        <dt className="text-ink-3">Retrieved</dt>
        <dd>{formatDateTimePT(layer.provenance.retrieved_at)}</dd>
        {layer.resolution_deg && (
          <>
            <dt className="text-ink-3">Resolution</dt>
            <dd>{layer.resolution_deg}° (~3 km)</dd>
          </>
        )}
        <dt className="text-ink-3">License</dt>
        <dd>{layer.provenance.license}</dd>
        {layer.provenance.citation && (
          <>
            <dt className="text-ink-3">Cite</dt>
            <dd>{layer.provenance.citation}</dd>
          </>
        )}
      </dl>
    </details>
    </div>
  );
}
