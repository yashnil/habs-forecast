"use client";

import type { ForecastRun, LayerArtifact, Manifest, SourceStatus } from "@/generated/schema";
import { CHLOROPHYLL_COPY, FORECAST_COPY } from "@/content/copy";
import { classifyTime, type Freshness } from "@/lib/freshness";
import {
  CHARM_LEADS,
  CHARM_VARIABLES,
  charmLayer,
  chlorophyllLayers,
  leadLabel,
  nativeLabel,
  regionCoverage,
  satelliteDays,
  satelliteLatest,
  satelliteUpdateFailed,
  type CharmVariable,
  type SatProduct,
} from "@/lib/layers";
import { formatDate, formatDateTimePT, relativeDay } from "@/lib/time";
import { FreshnessBadge, ProductClassBadge } from "@/components/ui/Badges";
import { AgeLegend, ChlorophyllLegend, ProbabilityLegend, SensorLegend } from "@/components/ui/ProbabilityLegend";
import { multiSensorMembers, ratioPhrase } from "@/lib/multisensor";
import { currentsHourly, currentsMean, hourStamp, hoursAgo, KNOTS_PER_MS } from "@/lib/currents";
import { combinedGapDays } from "@/lib/stamp";
import { ARROW_LENGTHS, SPEED_CLASSES } from "@/lib/basemap";
import { safeHref } from "@/lib/links";
import { MapStamp, type StampLine } from "./MapStamp";

export type LayerGroup = "forecast" | "satellite" | "currents";
/** Currents: one observed hour (null = newest) or the 24-hour mean, drawn as arrows or particles. */
export type CurChoice = { hour: string | null; mean: boolean };
export type FlowMode = "arrows" | "particles";
export type SatChoice = { product: SatProduct; day: string | null } | { imagery: string };

type Props = {
  manifest: Manifest;
  group: LayerGroup;
  onGroup: (g: LayerGroup) => void;
  run: ForecastRun | null;
  charmStatus: SourceStatus | null;
  satStatus: SourceStatus | null;
  gibsStatus: SourceStatus | null;
  variable: CharmVariable;
  onVariable: (v: CharmVariable) => void;
  lead: number;
  onLead: (l: number) => void;
  sat: SatChoice;
  onSat: (c: SatChoice) => void;
  showAge: boolean;
  onShowAge: (b: boolean) => void;
  showSensor: boolean;
  onShowSensor: (b: boolean) => void;
  regionId: string;
  regionLabel: string;
  now: Date | null;
  cur: CurChoice;
  onCur: (c: CurChoice) => void;
  flow: FlowMode;
  onFlow: (m: FlowMode) => void;
  curStatus: SourceStatus | null;
  reducedMotion: boolean;
  /** combined view (opt-in): satellite chlorophyll under the currents */
  underlay?: boolean;
  onUnderlay?: (b: boolean) => void;
  /** what the map shows and when, one line per layer drawn */
  stamp: StampLine[];
  /** "desktop": floating dock over the map with its own details toggle; "sheet": phone bottom sheet with layer tabs */
  variant: "desktop" | "sheet";
  /** details tier: options, notes, caveats and provenance */
  expanded: boolean;
  onExpanded?: (b: boolean) => void;
};

export const VAR_NAME: Record<CharmVariable, string> = {
  particulate_domoic: "Particulate domoic acid",
  pseudo_nitzschia: "Pseudo-nitzschia bloom",
  cellular_domoic: "Cellular domoic acid",
};

export const GROUPS: { id: LayerGroup; label: string; short: string; kind: "Model" | "Observation" }[] = [
  { id: "forecast", label: "HAB forecast", short: "Forecast", kind: "Model" },
  { id: "satellite", label: "Satellite chlorophyll", short: "Satellite", kind: "Observation" },
  { id: "currents", label: "Ocean currents", short: "Currents", kind: "Observation" },
];

function Res({ l }: { l: LayerArtifact | null | undefined }) {
  const n = l?.multisensor
    ? l.multisensor.members.map((m) => nativeLabel({ native_resolution_m: m.native_resolution_m } as LayerArtifact)).join(" + ")
    : nativeLabel(l);
  return n ? (
    <span data-testid="native-resolution" className="rounded bg-surface-3 px-1.5 py-px font-mono text-[11px] text-ink-2">
      native {n}
    </span>
  ) : null;
}

export function Seg<T extends string | number>({
  value,
  options,
  onChange,
  label,
  testid,
  size = "md",
}: {
  value: T;
  options: { value: T; label: React.ReactNode; title?: string; disabled?: boolean; testid?: string }[];
  onChange: (v: T) => void;
  label: string;
  testid?: string;
  size?: "sm" | "md";
}) {
  return (
    <div role="radiogroup" aria-label={label} data-testid={testid} className="flex rounded-[9px] bg-surface-3 p-[3px]">
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          disabled={o.disabled}
          title={o.title}
          data-testid={o.testid}
          onClick={() => onChange(o.value)}
          className={`flex-1 whitespace-nowrap rounded-[7px] font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40 ${size === "sm" ? "px-2 py-0.5 text-[12px]" : "px-2.5 py-1 text-[13px] max-lg:py-1.5"} ${o.value === value ? "bg-surface text-ink shadow-[0_1px_2px_rgba(13,27,42,0.12)] ring-1 ring-hairline-strong" : "text-ink-2 hover:text-ink"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Layer tabs for the phone sheet (the desktop has the control rail). */
function GroupTabs({ manifest, group, onGroup }: Pick<Props, "manifest" | "group" | "onGroup">) {
  const hasCurrents = currentsHourly(manifest).length > 0;
  return (
    <div role="tablist" aria-label="Map layer" className="grid grid-cols-3 gap-1 rounded-[11px] bg-surface-3 p-[3px]">
      {GROUPS.map((g) => {
        const off = g.id === "currents" && !hasCurrents;
        return (
          <button
            key={g.id}
            type="button"
            role="tab"
            aria-selected={group === g.id}
            aria-disabled={off || undefined}
            disabled={off}
            data-testid={`group-${g.id}`}
            title={off ? "Observed currents are not in this dataset." : undefined}
            onClick={() => onGroup(g.id)}
            className={`flex h-10 flex-col items-center justify-center rounded-[9px] leading-tight transition-colors duration-150 disabled:opacity-40 ${group === g.id ? "bg-surface text-ink shadow-[0_1px_2px_rgba(13,27,42,0.12)] ring-1 ring-hairline-strong" : "text-ink-2"}`}
          >
            <span className="text-[13.5px] font-semibold">{g.short}</span>
            <span className={`text-[10px] font-semibold uppercase tracking-wider ${g.kind === "Model" ? "text-model-ink" : "text-measured"}`}>{g.kind}</span>
          </button>
        );
      })}
    </div>
  );
}

/**
 * The map's layer dock (design reset §3.1, M5): what is drawn and when, the time steps,
 * a compact legend and the essential qualifiers stay visible; options, notes, caveats and
 * provenance sit one step deeper. On the desktop it floats bottom-left of the map; on a
 * phone it is the bottom sheet's content.
 */
export function LayerDock(p: Props) {
  const desktop = p.variant === "desktop";
  return (
    <section
      data-testid="layer-dock"
      data-expanded={p.expanded}
      aria-label="Map layers"
      className={desktop ? "theme-paper flex max-h-full flex-col rounded-xl bg-surface text-ink shadow-[0_1px_2px_rgba(6,17,30,0.14),0_10px_30px_rgba(6,17,30,0.22)]" : "text-ink"}
    >
      {!desktop && (
        <div className="pb-2.5">
          <GroupTabs manifest={p.manifest} group={p.group} onGroup={p.onGroup} />
        </div>
      )}
      <header className={`flex items-start gap-2 ${desktop ? "px-3.5 pt-2.5" : ""}`}>
        <MapStamp lines={p.stamp} className="min-w-0 flex-1" />
        {desktop && p.onExpanded && (
          <button
            type="button"
            onClick={() => p.onExpanded!(!p.expanded)}
            aria-expanded={p.expanded}
            aria-controls="layer-details"
            data-testid="dock-details"
            className="-mr-1 -mt-0.5 flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-[12px] font-medium text-accent hover:bg-surface-3"
          >
            {p.expanded ? "Less" : "Details"}
            <svg viewBox="0 0 12 12" className={`h-3 w-3 transition-transform duration-200 ${p.expanded ? "rotate-180" : ""}`} aria-hidden>
              <path d="M3 7.5 6 4.5l3 3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        )}
      </header>
      <div className={`min-h-0 space-y-2.5 overflow-y-auto overscroll-contain [scrollbar-width:thin] ${desktop ? "px-3.5 pb-3 pt-2" : "pt-2"}`}>
        {p.group === "forecast" ? <ForecastSection {...p} /> : p.group === "currents" ? <CurrentsSection {...p} /> : <SatelliteSection {...p} />}
      </div>
    </section>
  );
}

/** Second tier: everything that explains rather than shows. */
function Details({ open, children }: { open: boolean; children: React.ReactNode }) {
  if (!open) return null;
  return (
    <div id="layer-details" className="space-y-2.5 border-t border-hairline pt-2.5">
      {children}
    </div>
  );
}

function MetaRow({ children, fresh, basis, compactAge }: { children: React.ReactNode; fresh: Freshness | null; basis?: string; compactAge?: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5 text-[12px] text-ink-3">
      {children}
      <span className="ml-auto">
        <FreshnessBadge f={fresh} basis={basis} compact={compactAge} />
      </span>
    </div>
  );
}

// ---------------------------------------------------------------- HAB forecast (C-HARM)
function ForecastSection(p: Props) {
  const layer = charmLayer(p.manifest, p.variable, p.lead);
  const anyLayer = layer ?? CHARM_LEADS.map((l) => charmLayer(p.manifest, p.variable, l)).find(Boolean) ?? null;
  const fresh: Freshness | null =
    p.now && anyLayer ? classifyTime(anyLayer.freshness, anyLayer.time, p.now) : p.now ? { state: "unavailable", basisDate: null, ageDays: null } : null;
  if (!p.run || !anyLayer) {
    return (
      <div data-testid="forecast-panel">
        <div data-testid="forecast-unavailable" className="space-y-1 rounded-lg border border-dashed border-hairline-strong p-3 text-[13px] text-ink-2">
          <p className="font-medium text-ink">Forecast unavailable</p>
          <p>No C-HARM forecast could be loaded, so none is shown. This does not mean conditions are normal.</p>
          {p.charmStatus?.error && <p className="break-words text-[12px] text-ink-3">Last error: {p.charmStatus.error}</p>}
        </div>
      </div>
    );
  }
  return (
    <div data-testid="forecast-panel" className="space-y-2.5">
      <div className="flex flex-wrap items-stretch gap-2">
        <label className="flex h-[42px] min-w-[176px] flex-1 basis-[176px] items-center rounded-[9px] bg-surface-3 pl-2.5 pr-1.5 text-[13px] font-medium sm:flex-none">
          <span className="sr-only">Forecast quantity</span>
          <select
            value={p.variable}
            onChange={(e) => p.onVariable(e.target.value as CharmVariable)}
            className="w-full cursor-pointer bg-transparent outline-none"
            data-testid="variable-select"
          >
            {CHARM_VARIABLES.map((v) => (
              <option key={v} value={v}>
                {VAR_NAME[v]}
              </option>
            ))}
          </select>
        </label>
        <div role="radiogroup" aria-label="Forecast day" className="grid min-w-[280px] flex-[2] grid-cols-4 gap-1">
          {CHARM_LEADS.map((l) => {
            const lyr = charmLayer(p.manifest, p.variable, l);
            const active = p.lead === l;
            return (
              <button
                key={l}
                type="button"
                role="radio"
                aria-checked={active}
                disabled={!lyr}
                data-testid={`lead-${l}`}
                onClick={() => p.onLead(l)}
                title={!lyr ? "Not issued in this run" : l === 0 ? "Nowcast" : `Forecast ${leadLabel(l)}`}
                className={`rounded-lg border px-1.5 py-1 text-left transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-40 ${active ? "border-navy-900 bg-navy-900 text-white" : "border-hairline hover:border-hairline-strong"}`}
              >
                <span className="block text-[13px] font-semibold leading-tight tabular">{lyr?.time.valid_date ? formatDate(lyr.time.valid_date).replace(/^\w+, /, "") : "—"}</span>
                <span className={`block truncate text-[11px] leading-tight ${active ? "text-on-navy-2" : "text-ink-3"}`}>
                  {!lyr ? "not issued" : l === 0 ? "nowcast" : p.now && lyr.time.valid_date ? relativeDay(lyr.time.valid_date, p.now) : leadLabel(l)}
                </span>
              </button>
            );
          })}
        </div>
      </div>
      {layer?.palette && (
        <>
          <ProbabilityLegend palette={layer.palette} threshold={layer.threshold_text} compact />
          <p data-testid="valid-line" className="sr-only">
            {VAR_NAME[p.variable]} · valid {formatDate(layer.time.valid_date!, { year: true })}
          </p>
        </>
      )}
      {/* freshness problems are never hidden behind "Details" */}
      {fresh?.state === "stale" && (
        <p className="text-[12px] leading-snug text-warning" data-testid="stale-note">
          No newer C-HARM run has been published. This is the newest forecast; its dates are as issued.
        </p>
      )}
      {fresh?.state === "historical" && (
        <p className="text-[12px] font-medium leading-snug text-serious" data-testid="historical-note">
          This forecast is out of date. It is shown for reference only and does not describe current conditions.
        </p>
      )}
      {p.charmStatus?.outcome === "failed" && (
        <p className="text-[12px] leading-snug text-serious" data-testid="update-failed">
          The last update attempt failed ({formatDateTimePT(p.charmStatus.last_attempt_at)}). Showing the last successful run.
        </p>
      )}
      <MetaRow fresh={fresh} basis={anyLayer.freshness.basis}>
        <ProductClassBadge pc="official_forecast" />
        <span>C-HARM v3.1 · NOAA</span>
        <Res l={layer ?? anyLayer} />
      </MetaRow>
      <p className="text-[12px] leading-snug text-ink-2">
        {p.expanded ? `${FORECAST_COPY.notA} ` : "Not a closure decision. "}
        {FORECAST_COPY.lowNotSafe}
      </p>
      <Details open={p.expanded}>
        <p className="text-[12px] leading-snug text-ink-2" data-testid="run-line">
          Issued <span className="font-medium text-ink tabular">{formatDate(p.run.issued_date, { year: true })}</span>
          {p.run.issued_date_derived && <span className="text-ink-3"> (inferred)</span>}
          {layer?.time.valid_date && (
            <span className="text-ink-3">
              {" "}
              · showing {formatDate(layer.time.valid_date)}, {p.lead === 0 ? "nowcast" : `forecast ${leadLabel(p.lead)}`}
            </span>
          )}
          {p.run.leads_missing.length > 0 && <span className="text-ink-3"> · {p.run.leads_missing.map(leadLabel).join(", ")} not issued in this run</span>}
          <span className="text-ink-3"> · no forecast exists beyond day 3</span>
        </p>
        {layer && <AboutLayer layer={layer} title="About this forecast" testid="forecast-caveats" />}
      </Details>
    </div>
  );
}

// ---------------------------------------------------------------- satellite
function SatelliteSection(p: Props) {
  const imagery = chlorophyllLayers(p.manifest);
  const olci = satelliteLatest(p.manifest, "olci300");
  const viirs = satelliteLatest(p.manifest, "viirs750");
  const multi = satelliteLatest(p.manifest, "multi");
  const days = satelliteDays(p.manifest, "olci300");
  const isImagery = "imagery" in p.sat;
  const product = isImagery ? null : (p.sat as { product: SatProduct }).product;
  const day = isImagery ? null : (p.sat as { day: string | null }).day;
  const latest = product === "viirs750" ? viirs : product === "multi" ? multi : olci;
  const layer = day ? (days.find((d) => d.time.observed_date === day) ?? null) : latest;
  const fresh: Freshness | null = p.now && layer ? classifyTime(layer.freshness, layer.time, p.now) : null;
  const cov = regionCoverage(layer, p.regionId);
  const comp = layer?.composite;
  const ms = layer?.multisensor;
  const [mPrimary, mSecondary] = multiSensorMembers(p.manifest, layer);

  const options = [
    { value: "multi", label: "Multi-sensor", disabled: !multi, testid: "sat-multi", title: "Sentinel-3 300 m where it has a recent observation, VIIRS 750 m elsewhere; nothing averaged" },
    { value: "olci300", label: "Sentinel-3", disabled: !olci, testid: "sat-olci300", title: "Sentinel-3 OLCI, 300 m" },
    { value: "viirs750", label: "VIIRS", disabled: !viirs, testid: "sat-viirs750", title: "VIIRS, 750 m" },
    { value: "imagery", label: "Imagery", disabled: imagery.length === 0, testid: "sat-imagery", title: "Same-day pictures from NASA GIBS; values cannot be read from them" },
  ];
  const value = isImagery ? "imagery" : (product as string);
  const colour = p.showAge && (comp || ms) ? "age" : p.showSensor && ms ? "sensor" : "value";

  return (
    <div data-testid="satellite-panel" className="space-y-2.5">
      <Seg
        label="Satellite product"
        value={value}
        options={options}
        onChange={(v) => p.onSat(v === "imagery" ? { imagery: imagery[0]?.layer_id ?? "" } : { product: v as SatProduct, day: null })}
      />
      {isImagery ? (
        <ImagerySection {...p} imagery={imagery} />
      ) : !latest ? (
        <div data-testid="satellite-unavailable" className="rounded-lg border border-dashed border-hairline-strong p-3 text-[13px] text-ink-2">
          <p className="font-medium text-ink">{product === "olci300" ? "Sentinel-3 OLCI 300 m is unavailable" : product === "multi" ? "The multi-sensor view is unavailable" : "VIIRS 750 m is unavailable"}</p>
          <p>
            {p.satStatus?.error ? `Last error: ${p.satStatus.error.slice(0, 200)}` : "No recent observation could be loaded."} {CHLOROPHYLL_COPY.gaps}
          </p>
        </div>
      ) : (
        <>
          {product === "olci300" && days.length > 0 && (
            <div role="radiogroup" aria-label="Observation day" className="flex gap-1">
              <button
                type="button"
                role="radio"
                aria-checked={day == null}
                data-testid="sat-day-latest"
                onClick={() => p.onSat({ product: "olci300", day: null })}
                className={`min-w-[84px] rounded-lg border px-2 py-1 text-left transition-colors duration-150 ${day == null ? "border-navy-900 bg-navy-900 text-white" : "border-hairline hover:border-hairline-strong"}`}
              >
                <span className="block text-[13px] font-semibold leading-tight">Latest</span>
                <span className={`block text-[11px] leading-tight ${day == null ? "text-on-navy-2" : "text-ink-3"}`}>clear view, {comp?.window_days ?? 7} d</span>
              </button>
              {days.map((d) => {
                const c = regionCoverage(d, p.regionId)?.observed_fraction ?? d.coverage?.domain_observed_fraction ?? 0;
                const on = day === d.time.observed_date;
                return (
                  <button
                    key={d.layer_id}
                    type="button"
                    role="radio"
                    aria-checked={on}
                    aria-label={`${formatDate(d.time.observed_date!)}: ${Math.round(c * 100)}% of ${p.regionLabel} ocean observed`}
                    data-testid={`sat-day-${d.time.observed_date}`}
                    onClick={() => p.onSat({ product: "olci300", day: d.time.observed_date! })}
                    title={`${formatDate(d.time.observed_date!)}: ${Math.round(c * 100)}% of ${p.regionLabel} ocean observed`}
                    className={`flex flex-1 flex-col items-center gap-0.5 rounded-lg border px-0.5 pb-0.5 pt-1 transition-colors duration-150 ${on ? "border-navy-900 ring-1 ring-navy-900" : "border-hairline hover:border-hairline-strong"}`}
                  >
                    {/* the bar is the share of the region's ocean observed that day: clouds leave it low */}
                    <span className="relative h-5 w-3/4 overflow-hidden rounded-[3px] bg-surface-3" aria-hidden>
                      <span className="absolute inset-x-0 bottom-0 bg-measured" style={{ height: `${Math.max(c > 0 ? 8 : 0, c * 100)}%` }} />
                    </span>
                    <span className="text-[11px] font-medium text-ink-2 tabular">{d.time.observed_date!.slice(8)}</span>
                  </button>
                );
              })}
            </div>
          )}
          {layer?.palette && (
            <div className="space-y-1.5">
              {colour === "age" ? <AgeLegend maxDays={comp?.window_days ?? 7} /> : colour === "sensor" && ms ? <SensorLegend labels={ms.members.map((m) => m.label)} /> : <ChlorophyllLegend palette={layer.palette} compact />}
              {(comp || ms) && (
                <div className="flex items-center gap-2 text-[12px] text-ink-3">
                  <span id="colour-shows">Colour shows</span>
                  <div role="radiogroup" aria-labelledby="colour-shows" className="flex gap-1">
                    {(
                      [
                        ["value", "Chlorophyll", "toggle-value", true],
                        ["age", "Observation date", "toggle-age", true],
                        ["sensor", "Sensor", "toggle-sensor", !!ms],
                      ] as const
                    )
                      .filter((o) => o[3])
                      .map(([k, label, tid]) => (
                        <button
                          key={k}
                          type="button"
                          role="radio"
                          aria-checked={colour === k}
                          data-testid={tid}
                          onClick={() => {
                            if (k === "age") p.onShowAge(true);
                            else if (k === "sensor") p.onShowSensor(true);
                            else {
                              p.onShowAge(false);
                              p.onShowSensor(false);
                            }
                          }}
                          className={`rounded-full border px-2 py-0.5 text-[12px] transition-colors duration-150 max-lg:py-1 ${colour === k ? "border-navy-900 bg-navy-900 text-white" : "border-hairline text-ink-2 hover:border-hairline-strong"}`}
                        >
                          {label}
                        </button>
                      ))}
                  </div>
                </div>
              )}
            </div>
          )}
          {product && satelliteUpdateFailed(p.satStatus, product) && p.satStatus && (
            <p className="text-[12px] leading-snug text-serious" data-testid="sat-update-failed">
              The latest update ({formatDateTimePT(p.satStatus.last_attempt_at)}) did not refresh this product. Showing the last published observations, with their own dates.
            </p>
          )}
          <MetaRow fresh={fresh} basis="observed_date">
            <ProductClassBadge pc="observation" />
            <span className={p.expanded ? "" : "hidden"}>{product === "viirs750" ? "VIIRS · NOAA" : product === "multi" ? "Sentinel-3 OLCI + VIIRS · NOAA CoastWatch" : "Sentinel-3 OLCI · NOAA CoastWatch"}</span>
            <Res l={layer} />
          </MetaRow>
          <p className={`text-[12px] leading-snug text-ink-2 ${p.expanded ? "" : "line-clamp-2"}`} data-testid="sat-dates">
            {ms ? (
              <MultiDates ms={ms} primary={mPrimary} secondary={mSecondary} regionId={p.regionId} regionLabel={p.regionLabel} />
            ) : layer && !layer.grid ? (
              <>No clear observation on {formatDate(layer.time.observed_date!)}: clouds, fog or no overpass. Nothing is shown for that day.</>
            ) : comp ? (
              <>
                Pixels observed {formatDate(comp.oldest_observed_date)}–{formatDate(comp.newest_observed_date)}.{" "}
                {cov ? `${Math.round(cov.observed_fraction * 100)}% of ${p.regionLabel} ocean observed in the last ${comp.window_days} days.` : ""}
              </>
            ) : layer ? (
              <>
                Overpass {(layer.time.observed_times ?? []).map((t) => formatDateTimePT(t)).join(", ")}.{" "}
                {cov ? `${Math.round(cov.observed_fraction * 100)}% of ${p.regionLabel} ocean observed.` : ""}
              </>
            ) : null}{" "}
            {CHLOROPHYLL_COPY.biomass.split(".")[0]}. {CHLOROPHYLL_COPY.gaps}
          </p>
          <Details open={p.expanded}>
            {ms && (
              <p data-testid="multi-label" className="text-[12px] leading-snug text-ink-2">
                <span className="font-medium text-ink">Multi-sensor display.</span> Sentinel-3 300 m where it has an observation in the last {comp?.window_days ?? 7} days; VIIRS 750 m elsewhere, or where VIIRS is more than {ms.prefer_primary_within_days} days newer. Each pixel is one sensor&apos;s own value and date; nothing is averaged.
              </p>
            )}
            {layer && <AboutLayer layer={layer} title="About this layer" testid="satellite-caveats" />}
          </Details>
        </>
      )}
    </div>
  );
}

function ImagerySection(p: Props & { imagery: LayerArtifact[] }) {
  const id = "imagery" in p.sat ? p.sat.imagery : null;
  const active = p.imagery.find((l) => l.layer_id === id) ?? p.imagery[0];
  const fresh = p.now && active ? classifyTime(active.freshness, active.time, p.now) : null;
  return (
    <div data-testid="observation-panel" className="space-y-2">
      <div role="radiogroup" aria-label="Imagery" className="flex flex-wrap gap-1.5">
        {p.imagery.map((l) => (
          <button
            key={l.layer_id}
            type="button"
            role="radio"
            aria-checked={active?.layer_id === l.layer_id}
            aria-label={l.short_title}
            onClick={() => p.onSat({ imagery: l.layer_id })}
            className={`rounded-lg border px-2.5 py-1 text-[13px] ${active?.layer_id === l.layer_id ? "border-navy-900 bg-navy-900 text-white" : "border-hairline text-ink-2 hover:border-hairline-strong"}`}
          >
            {l.short_title} · {l.time.observed_date ? formatDate(l.time.observed_date).replace(/^\w+, /, "") : "—"}
          </button>
        ))}
      </div>
      {active && (
        <div data-testid="chl-legend" className="space-y-1">
          {active.tiles?.legend_verified && active.tiles.legend_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={active.tiles.legend_url} alt={`NASA colour legend for ${active.title} (mg per cubic metre)`} className="w-full rounded bg-white p-0.5" loading="lazy" />
          ) : (
            <p className="text-[12px] text-serious">Legend unavailable: colours cannot be read quantitatively.</p>
          )}
        </div>
      )}
      <MetaRow fresh={fresh} basis="observed_date">
        <ProductClassBadge pc="observation" />
        <span>NASA GIBS pictures · values cannot be read from them</span>
      </MetaRow>
      <p className="text-[12px] text-ink-2">{CHLOROPHYLL_COPY.biomass}</p>
    </div>
  );
}

// ---------------------------------------------------------------- disclosure
function AboutLayer({ layer, title, testid }: { layer: LayerArtifact; title: string; testid: string }) {
  return (
    <details className="group text-[12px] text-ink-2">
      <summary className="cursor-pointer list-none font-medium text-accent [&::-webkit-details-marker]:hidden">
        <span className="inline-block transition-transform duration-150 group-open:rotate-90" aria-hidden>
          ▸
        </span>{" "}
        {title}: caveats, source and method
      </summary>
      <div className="mt-1.5 space-y-2">
        <ul data-testid={testid} className="list-disc space-y-1 pl-4 leading-snug">
          {layer.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11.5px]">
          <dt className="text-ink-3">Source</dt>
          <dd>
            <a className="text-accent hover:underline" href={safeHref(layer.provenance.source_url)} target="_blank" rel="noopener noreferrer">
              {layer.provenance.source_name}
            </a>
          </dd>
          {layer.provenance.dataset_id && (
            <>
              <dt className="text-ink-3">Dataset</dt>
              <dd className="break-all font-mono">{layer.provenance.dataset_id}</dd>
            </>
          )}
          {layer.time.valid_time && (
            <>
              <dt className="text-ink-3">Valid time</dt>
              <dd className="font-mono">{layer.time.valid_time}</dd>
            </>
          )}
          {(layer.time.observed_times ?? []).length > 0 && (
            <>
              <dt className="text-ink-3">Overpasses</dt>
              <dd className="font-mono">{(layer.time.observed_times ?? []).join(", ")}</dd>
            </>
          )}
          <dt className="text-ink-3">Retrieved</dt>
          <dd>{formatDateTimePT(layer.provenance.retrieved_at)}</dd>
          <dt className="text-ink-3">Native grid</dt>
          <dd>
            {layer.resolution_deg}° ({nativeLabel(layer)})
          </dd>
          <dt className="text-ink-3">License</dt>
          <dd>{layer.provenance.license}</dd>
          {layer.provenance.citation && (
            <>
              <dt className="text-ink-3">Cite</dt>
              <dd>{layer.provenance.citation}</dd>
            </>
          )}
        </dl>
      </div>
    </details>
  );
}

function MultiDates({
  ms,
  primary,
  secondary,
  regionId,
  regionLabel,
}: {
  ms: NonNullable<LayerArtifact["multisensor"]>;
  primary: LayerArtifact | null;
  secondary: LayerArtifact | null;
  regionId: string;
  regionLabel: string;
}) {
  const range = (l: LayerArtifact | null) => (l?.composite ? `${formatDate(l.composite.oldest_observed_date)}–${formatDate(l.composite.newest_observed_date)}` : "—");
  const cov = ms.coverage_comparison.find((c) => c.region_id === regionId) ?? ms.coverage_comparison.find((c) => c.region_id === "domain");
  const agree = [ms.agreement.find((a) => a.region_id === regionId), ms.agreement.find((a) => a.region_id === "domain")].find((a) => a && a.median_log10_ratio != null);
  const where = cov?.region_id === "domain" ? "the domain" : regionLabel;
  const sec = ms.shown?.find((x) => x.order === 1);
  return (
    <>
      Sentinel-3 pixels observed {range(primary)}; VIIRS {range(secondary)}.{" "}
      {sec?.median_age_days != null && sec.pixels > 0 && (
        <span data-testid="multi-viirs-age">VIIRS pixels shown are a median {Math.round(sec.median_age_days)} days old (VIIRS is published about 5 days after observation). </span>
      )}
      {cov && (
        <span data-testid="multi-coverage">
          {Math.round(cov.combined_fraction * 100)}% of {where} ocean observed by either sensor (Sentinel-3 alone {Math.round(cov.primary_fraction * 100)}%).{" "}
        </span>
      )}
      <span data-testid="sat-agreement">
        {agree ? (
          <>
            Where both saw the same water on the same day{agree.region_id === "domain" ? "" : ` in ${agree.label}`}, {ratioPhrase(agree.median_log10_ratio!)} ({agree.n_cells.toLocaleString("en-US")} cells): a colour step at a sensor edge may be the sensors, not the water.{" "}
          </>
        ) : (
          <>No same-day overlap this week to compare the two sensors. </>
        )}
      </span>
    </>
  );
}

// ---------------------------------------------------------------- observed currents (HF radar)
export function selectedCurrents(m: Manifest, c: CurChoice): LayerArtifact | null {
  if (c.mean) return currentsMean(m);
  const hours = currentsHourly(m);
  return (c.hour ? hours.find((h) => h.time.valid_time && hourStamp(h.time.valid_time) === c.hour) : null) ?? hours[hours.length - 1] ?? null;
}

function CurrentsSection(p: Props) {
  const hours = currentsHourly(p.manifest);
  const mean = currentsMean(p.manifest);
  const layer = selectedCurrents(p.manifest, p.cur);
  const idx = layer && !p.cur.mean ? hours.findIndex((h) => h.layer_id === layer.layer_id) : hours.length - 1;
  const fresh = p.now && layer ? classifyTime(layer.freshness, layer.time, p.now) : null;
  const cov = regionCoverage(layer, p.regionId);
  const go = (i: number) => {
    const h = hours[Math.max(0, Math.min(hours.length - 1, i))];
    if (h?.time.valid_time) p.onCur({ hour: hourStamp(h.time.valid_time), mean: false });
  };
  const t = layer?.time.valid_time;
  const failed = p.curStatus?.outcome === "failed";
  const particles = p.flow === "particles" && !p.reducedMotion;
  return (
    <div data-testid="currents-panel" className="space-y-2.5">
      <div className="flex flex-wrap gap-2">
        <div className="min-w-[200px] flex-1">
          <Seg
            label="Currents view"
            value={p.cur.mean ? "mean" : "hourly"}
            options={[
              { value: "hourly", label: "Hourly", testid: "cur-hourly" },
              { value: "mean", label: "24-hour mean", disabled: !mean, testid: "cur-mean", title: "Mean of the last 24 hours at each cell with at least 18 valid hours" },
            ]}
            onChange={(v) => p.onCur({ hour: p.cur.hour, mean: v === "mean" })}
          />
        </div>
        <div className="min-w-[180px] flex-1">
          <Seg
            label="Currents drawing"
            value={p.reducedMotion ? "arrows" : p.flow}
            options={[
              { value: "arrows", label: "Arrows", testid: "cur-mode-arrows" },
              {
                value: "particles",
                label: "Flow",
                disabled: p.reducedMotion,
                testid: "cur-mode-particles",
                title: p.reducedMotion ? "Off because your system asks for reduced motion" : "Particles moving through this one observed field",
              },
            ]}
            onChange={(v) => p.onFlow(v as FlowMode)}
          />
        </div>
      </div>
      {!p.cur.mean && hours.length > 0 && (
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => go(idx - 1)} disabled={idx <= 0} aria-label="Previous hour" data-testid="cur-hour-prev" className="grid h-8 w-8 place-items-center rounded-md border border-hairline text-[15px] disabled:opacity-40 max-lg:h-10 max-lg:w-10">
            ‹
          </button>
          <input
            type="range"
            min={0}
            max={hours.length - 1}
            step={1}
            value={Math.max(0, idx)}
            onChange={(e) => go(Number(e.target.value))}
            aria-label="Observation hour"
            aria-valuetext={t ? formatDateTimePT(t) : ""}
            data-testid="cur-hour-slider"
            className="min-w-0 flex-1 accent-[var(--color-accent)]"
          />
          <button type="button" onClick={() => go(idx + 1)} disabled={idx >= hours.length - 1} aria-label="Next hour" data-testid="cur-hour-next" className="grid h-8 w-8 place-items-center rounded-md border border-hairline text-[15px] disabled:opacity-40 max-lg:h-10 max-lg:w-10">
            ›
          </button>
        </div>
      )}
      <p className="text-[12.5px] text-ink-2" data-testid="cur-time">
        {layer ? (
          p.cur.mean ? (
            <>
              <span className="font-medium text-ink">24-hour mean</span>, {formatDateTimePT((layer.time.observed_times ?? [])[0] ?? "")} to {formatDateTimePT((layer.time.observed_times ?? []).at(-1) ?? "")}
            </>
          ) : (
            <>
              Observed <span className="font-medium text-ink">{t ? formatDateTimePT(t) : "—"}</span>
              {t && <span className="text-ink-3"> ({t.slice(11, 16)} UTC)</span>}
              {t && p.now && (
                <span className="text-ink-3" data-testid="cur-age">
                  {" "}
                  · {hoursAgo(t, p.now)} h ago
                </span>
              )}
              {idx === hours.length - 1 && <span className="text-ink-3"> · newest hour</span>}
            </>
          )
        ) : (
          "No observed currents in this dataset."
        )}
      </p>
      <CurrentsLegend particles={particles} detail={p.expanded} />
      {failed && p.curStatus && (
        <p className="text-[12px] leading-snug text-serious" data-testid="cur-update-failed">
          The latest update ({formatDateTimePT(p.curStatus.last_attempt_at)}) did not refresh currents. Showing the last published hours, with their own times.
        </p>
      )}
      <CombinedSection {...p} currents={layer} />
      <MetaRow fresh={fresh} compactAge>
        <ProductClassBadge pc="observation" />
        <span className={p.expanded ? "" : "hidden"}>HF radar · HFRNet / NOAA CoastWatch</span>
        <Res l={layer} />
      </MetaRow>
      <p className={`text-[12px] leading-snug text-ink-2 ${p.expanded ? "" : "line-clamp-2"}`} data-testid="cur-notes">
        {cov && (
          <span data-testid="cur-coverage">
            {Math.round(cov.observed_fraction * 100)}% of {p.regionLabel} ocean observed {p.cur.mean ? "in the mean" : "this hour"}.{" "}
          </span>
        )}
        Observed currents, not a forecast. Gaps are no data, not calm water.{" "}
        {p.cur.mean ? "The mean smooths out most daily and tidal back-and-forth." : "Each hour is a separate snapshot; nothing is shown between hours."}
      </p>
      <Details open={p.expanded}>{layer && <AboutLayer layer={layer} title="About this layer" testid="currents-caveats" />}</Details>
    </div>
  );
}

const CLASS_LABELS = ["< 0.1", "0.1–0.25", "0.25–0.5", "0.5–1", "≥ 1"];

/** The map's arrow glyph for a speed class (same geometry as lib/basemap arrowImage). */
function ArrowGlyph({ length }: { length: number }) {
  const W = 16;
  const H = 32;
  const top = H / 2 - length / 2;
  const bottom = H / 2 + length / 2;
  const head = Math.min(6, length * 0.45);
  const d = `M${W / 2} ${bottom} L${W / 2} ${top + head * 0.6} M${W / 2 - head * 0.62} ${top + head} L${W / 2} ${top} L${W / 2 + head * 0.62} ${top + head}`;
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden className="rotate-90">
      <path d={d} stroke="rgba(6,17,30,0.92)" strokeWidth={4} fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <path d={d} stroke="#f2f6fa" strokeWidth={1.8} fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CurrentsLegend({ particles, detail }: { particles: boolean; detail: boolean }) {
  return (
    <figure data-testid="currents-legend" className="space-y-1" aria-label="Legend: current speed classes">
      <figcaption className="text-[12px] font-medium text-ink">
        {particles ? "Particles move with the observed current" : "Arrows point where the surface water is moving"}
        <span className="font-normal text-ink-3"> · speed, m/s</span>
      </figcaption>
      <div className="flex items-center gap-1 rounded-lg bg-navy-900 px-2 py-0.5">
        {ARROW_LENGTHS.map((len, i) => (
          <span key={len} className="flex flex-1 items-center justify-center gap-1 text-[11px] leading-tight text-on-navy-2 tabular" data-testid="currents-legend-item">
            {particles ? (
              <svg width={30} height={24} viewBox="0 0 30 24" aria-hidden>
                <line x1={15 - len / 2} y1={12} x2={15 + len / 2} y2={12} stroke="#eef4fa" strokeWidth={1.6} strokeLinecap="round" strokeOpacity={Math.min(0.95, 0.35 + SPEED_CLASSES[i] * 1.6 + 0.1)} />
              </svg>
            ) : (
              <span className="-my-1 flex h-6 w-8 items-center justify-center">
                <ArrowGlyph length={len} />
              </span>
            )}
            {CLASS_LABELS[i]}
          </span>
        ))}
      </div>
      {detail && (
        <p className="text-[11.5px] text-ink-3">
          1 m/s ≈ {KNOTS_PER_MS.toFixed(1)} knots.{" "}
          {particles ? "Screen speed shows relative speed at every zoom; particles stop where there is no observation. Not a trajectory." : "One arrow per 2 km radar cell when zoomed in; fewer when zoomed out."}
        </p>
      )}
    </figure>
  );
}

// ---------------------------------------------------------------- combined currents + chlorophyll (opt-in)
function CombinedSection(p: Props & { currents: LayerArtifact | null }) {
  const chl = satelliteLatest(p.manifest, "olci300");
  if (!chl || !p.onUnderlay) return null;
  const cTime = p.currents?.layer_id.endsWith("mean24h") ? (p.currents.time.observed_times ?? []).at(-1) : p.currents?.time.valid_time;
  const gap = combinedGapDays(chl, p.currents);
  const range = chl.composite ? `${formatDate(chl.composite.oldest_observed_date)}–${formatDate(chl.composite.newest_observed_date)}` : "—";
  return (
    <div className="space-y-1.5">
      <label className="flex min-h-8 cursor-pointer items-center gap-2 text-[13px] font-medium text-ink">
        <input type="checkbox" className="h-4 w-4 accent-[var(--color-accent)]" checked={!!p.underlay} onChange={(e) => p.onUnderlay!(e.target.checked)} data-testid="toggle-combined" />
        Show satellite chlorophyll underneath
      </label>
      {p.underlay && (
        <>
          {chl.palette && (
            <div data-testid="combined-chl-legend">
              <ChlorophyllLegend palette={chl.palette} compact />
            </div>
          )}
          {/* the four statements that make the combined view honest; always shown with it */}
          <ul data-testid="combined-notes" className="list-disc space-y-0.5 pl-4 text-[12px] leading-snug text-ink-2">
            <li>Chlorophyll is ocean colour (algae biomass), not toxin.</li>
            <li>Currents are observed surface motion, not a forecast.</li>
            <li data-testid="combined-times">
              Different times: chlorophyll pixels observed {range}
              {gap != null && gap > 0 ? `, a median ${gap} day${gap === 1 ? "" : "s"} before the currents` : ""}; currents {cTime ? formatDateTimePT(cTime) : "—"}.
            </li>
            <li className="font-medium text-ink">Arrows do not show where a bloom will travel.</li>
          </ul>
        </>
      )}
    </div>
  );
}
