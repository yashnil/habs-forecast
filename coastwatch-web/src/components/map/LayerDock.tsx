"use client";

import { useEffect, useState } from "react";
import type {
  ForecastRun,
  LayerArtifact,
  Manifest,
  SourceStatus,
} from "@/generated/schema";
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
import {
  AgeLegend,
  ChlorophyllLegend,
  ProbabilityLegend,
  SensorLegend,
} from "@/components/ui/ProbabilityLegend";
import { multiSensorMembers, ratioPhrase } from "@/lib/multisensor";
import {
  currentsHourly,
  currentsMean,
  hourStamp,
  hoursAgo,
  KNOTS_PER_MS,
} from "@/lib/currents";
import { DEMO } from "@/lib/demo";

export type LayerGroup = "forecast" | "satellite" | "currents";
/** Currents: one observed hour (null = newest) or the 24-hour mean, drawn as arrows or particles. */
export type CurChoice = { hour: string | null; mean: boolean };
export type FlowMode = "arrows" | "particles";
export type SatChoice =
  { product: SatProduct; day: string | null } | { imagery: string };

type Props = {
  expanded?: boolean;
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
  compact?: boolean;
  cur: CurChoice;
  onCur: (c: CurChoice) => void;
  flow: FlowMode;
  onFlow: (m: FlowMode) => void;
  curStatus: SourceStatus | null;
  reducedMotion: boolean;
};

const VAR_SHORT: Record<CharmVariable, string> = {
  particulate_domoic: "Particulate DA",
  pseudo_nitzschia: "Pseudo-nitzschia",
  cellular_domoic: "Cellular DA",
};
const VAR_NAME: Record<CharmVariable, string> = {
  particulate_domoic: "Particulate domoic acid",
  pseudo_nitzschia: "Pseudo-nitzschia bloom",
  cellular_domoic: "Cellular domoic acid",
};

function Res({ l }: { l: LayerArtifact | null | undefined }) {
  const n = l?.multisensor
    ? l.multisensor.members
        .map((m) =>
          nativeLabel({
            native_resolution_m: m.native_resolution_m,
          } as LayerArtifact),
        )
        .join(" + ")
    : nativeLabel(l);
  return n ? (
    <span
      data-testid="native-resolution"
      className="rounded bg-surface-3 px-1.5 py-px font-mono text-[11.5px] text-ink-2"
    >
      native {n}
    </span>
  ) : null;
}

function Seg<T extends string | number>({
  value,
  options,
  onChange,
  label,
  testid,
}: {
  value: T;
  options: {
    value: T;
    label: React.ReactNode;
    title?: string;
    disabled?: boolean;
    testid?: string;
  }[];
  onChange: (v: T) => void;
  label: string;
  testid?: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      data-testid={testid}
      className="flex rounded-[9px] bg-surface-3 p-[3px]"
    >
      {options.map((o) => (
        <button
          key={String(o.value)}
          role="radio"
          aria-checked={o.value === value}
          disabled={o.disabled}
          title={o.title}
          data-testid={o.testid}
          onClick={() => onChange(o.value)}
          className={`flex-1 whitespace-nowrap rounded-[7px] px-2.5 py-1 text-[13px] font-medium disabled:cursor-not-allowed disabled:opacity-40 ${o.value === value ? "bg-surface text-ink shadow-[0_1px_2px_rgba(13,27,42,0.12)] ring-1 ring-hairline-strong" : "text-ink-2 hover:text-ink"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function LayerDock(p: Props) {
  // On phones and short screens the dock starts collapsed (controls, colour scale and the
  // essential qualifiers) and expands on demand.
  const [expanded, setExpanded] = useState(!p.compact);
  const [short, setShort] = useState(false);
  useEffect(() => {
    if (!p.compact && window.innerHeight < 800) {
      setShort(true);
      setExpanded(false);
    }
  }, [p.compact]);
  const q = { ...p, expanded };
  return (
    <section
      data-testid="layer-dock"
      data-expanded={expanded}
      aria-label="Map layers"
      className="theme-paper rounded-2xl bg-surface text-ink shadow-[0_1px_2px_rgba(6,17,30,0.12),0_8px_24px_rgba(6,17,30,0.18)]"
    >
      {(p.compact || short) && (
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
          aria-label={expanded ? "Show less" : "Show layer details"}
          data-testid="dock-handle"
          className="flex w-full justify-center pb-0.5 pt-2"
        >
          <span
            className="h-1 w-9 rounded-full bg-hairline-strong"
            aria-hidden
          />
        </button>
      )}
      <div
        role="tablist"
        aria-label="Layer group"
        className={`flex items-end gap-1 border-b border-hairline px-3 ${p.compact ? "" : "pt-2"}`}
      >
        {(
          [
            ["forecast", "HAB forecast", "Model"],
            ["satellite", "Satellite", "Observation"],
          ] as const
        ).map(([g, label, kind]) => (
          <button
            key={g}
            role="tab"
            aria-selected={p.group === g}
            data-testid={`group-${g}`}
            onClick={() => p.onGroup(g)}
            className={`-mb-px flex items-center gap-1.5 whitespace-nowrap border-b-2 px-2.5 py-2 text-[14px] font-medium ${p.group === g ? "border-accent text-ink" : "border-transparent text-ink-3 hover:text-ink"}`}
          >
            {p.compact && g === "forecast" ? "Forecast" : label}
            <span
              className={`hidden rounded-full px-1.5 text-[10.5px] font-semibold uppercase tracking-wider sm:inline ${g === "forecast" ? "bg-model-bg text-model-ink" : "bg-measured-bg text-measured"}`}
            >
              {kind}
            </span>
          </button>
        ))}
        {DEMO ? null : currentsHourly(p.manifest).length > 0 ? (
          <button
            role="tab"
            aria-selected={p.group === "currents"}
            data-testid="group-currents"
            onClick={() => p.onGroup("currents")}
            className={`-mb-px flex items-center gap-1.5 whitespace-nowrap border-b-2 px-2.5 py-2 text-[14px] font-medium ${p.group === "currents" ? "border-accent text-ink" : "border-transparent text-ink-3 hover:text-ink"}`}
          >
            {p.compact ? "Currents" : "Ocean currents"}
            <span className="hidden rounded-full bg-measured-bg px-1.5 text-[10.5px] font-semibold uppercase tracking-wider text-measured sm:inline">
              Observation
            </span>
          </button>
        ) : (
          <button
            role="tab"
            aria-selected={false}
            aria-disabled="true"
            disabled
            data-testid="group-currents"
            title="Observed currents are not in this dataset."
            className="-mb-px flex cursor-not-allowed items-center gap-1.5 whitespace-nowrap border-b-2 border-transparent px-2.5 py-2 text-[14px] font-medium text-ink-3/70"
          >
            {p.compact ? "Currents" : "Ocean currents"}{" "}
            <span className="rounded-full border border-hairline-strong px-1.5 text-[10.5px] font-semibold uppercase tracking-wider">
              {p.compact ? "Soon" : "Next phase"}
            </span>
          </button>
        )}
      </div>
      <div
        className={`space-y-2.5 px-3.5 pt-2.5 ${p.compact ? "pb-2" : "pb-3"}`}
      >
        {p.group === "forecast" ? (
          <ForecastSection {...q} />
        ) : p.group === "currents" ? (
          <CurrentsSection {...q} />
        ) : (
          <SatelliteSection {...q} />
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- HAB forecast (C-HARM)
function ForecastSection(p: Props) {
  const layer = charmLayer(p.manifest, p.variable, p.lead);
  const anyLayer =
    layer ??
    CHARM_LEADS.map((l) => charmLayer(p.manifest, p.variable, l)).find(
      Boolean,
    ) ??
    null;
  const fresh: Freshness | null =
    p.now && anyLayer
      ? classifyTime(anyLayer.freshness, anyLayer.time, p.now)
      : p.now
        ? { state: "unavailable", basisDate: null, ageDays: null }
        : null;
  if (!p.run || !anyLayer) {
    return (
      <div data-testid="forecast-panel">
        <div
          data-testid="forecast-unavailable"
          className="space-y-1 rounded-lg border border-dashed border-hairline-strong p-3 text-[13px] text-ink-2"
        >
          <p className="font-medium text-ink">Forecast unavailable</p>
          <p>
            No C-HARM forecast could be loaded, so none is shown. This does not
            mean conditions are normal.
          </p>
          {p.charmStatus?.error && (
            <p className="break-words text-[12px] text-ink-3">
              Last error: {p.charmStatus.error}
            </p>
          )}
        </div>
      </div>
    );
  }
  return (
    <div data-testid="forecast-panel" className="space-y-2.5">
      <div className={`flex gap-2 ${p.compact ? "flex-col" : "items-center"}`}>
        {p.compact ? (
          <label className="flex h-9 items-center rounded-[9px] bg-surface-3 px-2 text-[13px] font-medium">
            <span className="sr-only">Forecast quantity</span>
            <select
              value={p.variable}
              onChange={(e) => p.onVariable(e.target.value as CharmVariable)}
              className="w-full bg-transparent outline-none"
              data-testid="variable-select"
            >
              {CHARM_VARIABLES.map((v) => (
                <option key={v} value={v}>
                  {VAR_SHORT[v]}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <Seg
            label="Forecast quantity"
            value={p.variable}
            onChange={p.onVariable}
            options={CHARM_VARIABLES.map((v) => ({
              value: v,
              label: VAR_SHORT[v],
              testid: `variable-${v}`,
            }))}
          />
        )}
        <div
          role="radiogroup"
          aria-label="Forecast day"
          className="grid flex-1 grid-cols-4 gap-1"
        >
          {CHARM_LEADS.map((l) => {
            const lyr = charmLayer(p.manifest, p.variable, l);
            const active = p.lead === l;
            return (
              <button
                key={l}
                role="radio"
                aria-checked={active}
                disabled={!lyr}
                data-testid={`lead-${l}`}
                onClick={() => p.onLead(l)}
                title={
                  !lyr
                    ? "Not issued in this run"
                    : l === 0
                      ? "Nowcast"
                      : `Forecast ${leadLabel(l)}`
                }
                className={`rounded-lg border px-1.5 py-1 text-left disabled:cursor-not-allowed disabled:opacity-40 ${active ? "border-navy-900 bg-navy-900 text-white" : "border-hairline hover:border-hairline-strong"}`}
              >
                <span className="block text-[13px] font-semibold leading-tight tabular">
                  {lyr?.time.valid_date
                    ? formatDate(lyr.time.valid_date).replace(/^\w+, /, "")
                    : "—"}
                </span>
                <span
                  className={`block text-[11px] leading-tight ${active ? "text-on-navy-2" : "text-ink-3"}`}
                >
                  {!lyr
                    ? "not issued"
                    : l === 0
                      ? "nowcast"
                      : p.now && lyr.time.valid_date
                        ? relativeDay(lyr.time.valid_date, p.now)
                        : leadLabel(l)}
                </span>
              </button>
            );
          })}
        </div>
      </div>
      {layer?.palette && (
        <div className="space-y-1">
          <ProbabilityLegend
            palette={layer.palette}
            threshold={layer.threshold_text}
          />
          <p data-testid="valid-line" className="sr-only">
            {VAR_NAME[p.variable]} · valid{" "}
            {formatDate(layer.time.valid_date!, { year: true })}
          </p>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 border-t border-hairline pt-2 text-[12px] text-ink-3">
        <ProductClassBadge pc="official_forecast" />
        <span className={p.compact ? "hidden" : ""}>C-HARM v3.1 · NOAA</span>
        <Res l={layer ?? anyLayer} />
        <span className="ml-auto">
          <FreshnessBadge f={fresh} basis={anyLayer.freshness.basis} />
        </span>
      </div>
      <div
        className={`space-y-0.5 text-[12px] text-ink-2 ${p.expanded ? "" : "[&>p:first-child]:hidden"}`}
        data-testid="run-line"
      >
        <p>
          Issued{" "}
          <span className="font-medium text-ink tabular">
            {formatDate(p.run.issued_date, { year: true })}
          </span>
          {p.run.issued_date_derived && (
            <span className="text-ink-3"> (inferred)</span>
          )}
          {layer?.time.valid_date && (
            <span className="text-ink-3">
              {" "}
              · showing {formatDate(layer.time.valid_date)},{" "}
              {p.lead === 0 ? "nowcast" : `forecast ${leadLabel(p.lead)}`}
            </span>
          )}
          {p.run.leads_missing.length > 0 && (
            <span className="text-ink-3">
              {" "}
              · {p.run.leads_missing.map(leadLabel).join(", ")} not issued in
              this run
            </span>
          )}
          <span className="text-ink-3"> · no forecast exists beyond day 3</span>
        </p>
        {fresh?.state === "stale" && (
          <p className="text-warning" data-testid="stale-note">
            No newer C-HARM run has been published. This is the newest available
            forecast; its dates are shown as issued.
          </p>
        )}
        {fresh?.state === "historical" && (
          <p className="font-medium text-serious" data-testid="historical-note">
            This forecast is out of date. It is shown for reference only and
            does not describe current conditions.
          </p>
        )}
        {p.charmStatus?.outcome === "failed" && (
          <p className="text-serious" data-testid="update-failed">
            Latest update attempt failed (
            {formatDateTimePT(p.charmStatus.last_attempt_at)}). Showing the last
            successful run.
          </p>
        )}
      </div>
      <p className="text-[12px] leading-snug text-ink-2">
        {p.expanded ? `${FORECAST_COPY.notA} ` : "Not a closure decision. "}
        {FORECAST_COPY.lowNotSafe}
      </p>
      {layer && p.expanded && (
        <AboutLayer
          layer={layer}
          title="About this forecast"
          testid="forecast-caveats"
        />
      )}
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
  const latest =
    product === "viirs750" ? viirs : product === "multi" ? multi : olci;
  const layer = day
    ? (days.find((d) => d.time.observed_date === day) ?? null)
    : latest;
  const fresh: Freshness | null =
    p.now && layer ? classifyTime(layer.freshness, layer.time, p.now) : null;
  const cov = regionCoverage(layer, p.regionId);
  const comp = layer?.composite;
  const ms = layer?.multisensor;
  const [mPrimary, mSecondary] = multiSensorMembers(p.manifest, layer);

  const options = [
    {
      value: "multi",
      label: "Multi-sensor",
      disabled: !multi,
      testid: "sat-multi",
      title:
        "Sentinel-3 300 m where it has a recent observation, VIIRS 750 m elsewhere; nothing averaged",
    },
    {
      value: "olci300",
      label: "OLCI 300 m",
      disabled: !olci,
      testid: "sat-olci300",
    },
    {
      value: "viirs750",
      label: "VIIRS 750 m",
      disabled: !viirs,
      testid: "sat-viirs750",
    },
    {
      value: "imagery",
      label: "Imagery",
      disabled: imagery.length === 0,
      testid: "sat-imagery",
      title:
        "Same-day pictures from NASA GIBS; values cannot be read from them",
    },
  ];
  const value = isImagery ? "imagery" : (product as string);

  return (
    <div data-testid="satellite-panel" className="space-y-2.5">
      <Seg
        label="Satellite product"
        value={value}
        options={options}
        onChange={(v) =>
          p.onSat(
            v === "imagery"
              ? { imagery: imagery[0]?.layer_id ?? "" }
              : { product: v as SatProduct, day: null },
          )
        }
      />
      {isImagery ? (
        <ImagerySection {...p} imagery={imagery} />
      ) : !latest ? (
        <div
          data-testid="satellite-unavailable"
          className="rounded-lg border border-dashed border-hairline-strong p-3 text-[13px] text-ink-2"
        >
          <p className="font-medium text-ink">
            {product === "olci300"
              ? "Sentinel-3 OLCI 300 m is unavailable"
              : "VIIRS 750 m is unavailable"}
          </p>
          <p>
            {p.satStatus?.error
              ? `Last error: ${p.satStatus.error.slice(0, 200)}`
              : "No recent observation could be loaded."}{" "}
            {CHLOROPHYLL_COPY.gaps}
          </p>
        </div>
      ) : (
        <>
          {product === "olci300" && days.length > 0 && (
            <div
              role="radiogroup"
              aria-label="Observation day"
              className="flex gap-1"
            >
              <button
                role="radio"
                aria-checked={day == null}
                data-testid="sat-day-latest"
                onClick={() => p.onSat({ product: "olci300", day: null })}
                className={`min-w-[86px] rounded-lg border px-2 py-1 text-left ${day == null ? "border-navy-900 bg-navy-900 text-white" : "border-hairline hover:border-hairline-strong"}`}
              >
                <span className="block text-[13px] font-semibold leading-tight">
                  Latest
                </span>
                <span
                  className={`block text-[11px] leading-tight ${day == null ? "text-on-navy-2" : "text-ink-3"}`}
                >
                  clear view, {comp?.window_days ?? 7} d
                </span>
              </button>
              {days.map((d) => {
                const c =
                  regionCoverage(d, p.regionId)?.observed_fraction ??
                  d.coverage?.domain_observed_fraction ??
                  0;
                const on = day === d.time.observed_date;
                return (
                  <button
                    key={d.layer_id}
                    role="radio"
                    aria-checked={on}
                    data-testid={`sat-day-${d.time.observed_date}`}
                    onClick={() =>
                      p.onSat({
                        product: "olci300",
                        day: d.time.observed_date!,
                      })
                    }
                    title={`${formatDate(d.time.observed_date!)}: ${Math.round(c * 100)}% of ${p.regionLabel} ocean observed`}
                    className={`flex flex-1 flex-col items-center gap-0.5 rounded-lg border px-0.5 pb-0.5 pt-1 ${on ? "border-navy-900 ring-1 ring-navy-900" : "border-hairline hover:border-hairline-strong"}`}
                  >
                    <span
                      className="relative h-5 w-3/4 overflow-hidden rounded-[3px] bg-surface-3"
                      aria-hidden
                    >
                      <span
                        className="absolute inset-x-0 bottom-0 bg-measured"
                        style={{
                          height: `${Math.max(c > 0 ? 8 : 0, c * 100)}%`,
                        }}
                      />
                    </span>
                    <span className="text-[11px] font-medium text-ink-2 tabular">
                      {d.time.observed_date!.slice(8)}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
          {ms && (
            <p
              data-testid="multi-label"
              className="text-[12px] leading-snug text-ink-2"
            >
              <span className="font-medium text-ink">
                Multi-sensor display.
              </span>{" "}
              Sentinel-3 300 m where it has an observation in the last{" "}
              {comp?.window_days ?? 7} days; VIIRS 750 m elsewhere, or where
              VIIRS is more than {ms.prefer_primary_within_days} days newer.
              Each pixel is one sensor&apos;s own value and date; nothing is
              averaged.
            </p>
          )}
          {layer?.palette &&
            (p.showAge && (comp || ms) ? (
              <AgeLegend maxDays={comp?.window_days ?? 7} />
            ) : p.showSensor && ms ? (
              <SensorLegend labels={ms.members.map((m) => m.label)} />
            ) : (
              <ChlorophyllLegend palette={layer.palette} />
            ))}
          {(comp || ms) && p.expanded && (
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              <label className="flex items-center gap-2 text-[13px] text-ink-2">
                <input
                  type="checkbox"
                  checked={p.showAge}
                  onChange={(e) => p.onShowAge(e.target.checked)}
                  data-testid="toggle-age"
                />
                Show each pixel&apos;s observation date
              </label>
              {ms && (
                <label className="flex items-center gap-2 text-[13px] text-ink-2">
                  <input
                    type="checkbox"
                    checked={p.showSensor}
                    onChange={(e) => p.onShowSensor(e.target.checked)}
                    data-testid="toggle-sensor"
                  />
                  Show which sensor
                </label>
              )}
            </div>
          )}
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 border-t border-hairline pt-2 text-[12px] text-ink-3">
            <ProductClassBadge pc="observation" />
            <span>
              {product === "viirs750"
                ? "VIIRS · NOAA"
                : product === "multi"
                  ? "Sentinel-3 OLCI + VIIRS · NOAA CoastWatch"
                  : "Sentinel-3 OLCI · NOAA CoastWatch"}
            </span>
            <Res l={layer} />
            <span className="ml-auto">
              <FreshnessBadge f={fresh} basis="observed_date" />
            </span>
          </div>
          {product &&
            satelliteUpdateFailed(p.satStatus, product) &&
            p.satStatus && (
              <p
                className="text-[12px] text-serious"
                data-testid="sat-update-failed"
              >
                The latest update (
                {formatDateTimePT(p.satStatus.last_attempt_at)}) did not refresh
                this product. Showing the last published observations, with
                their own dates.
              </p>
            )}
          <p
            className={`text-[12px] leading-snug text-ink-2 ${p.expanded ? "" : "line-clamp-2"}`}
            data-testid="sat-dates"
          >
            {ms ? (
              <MultiDates
                ms={ms}
                primary={mPrimary}
                secondary={mSecondary}
                regionId={p.regionId}
                regionLabel={p.regionLabel}
              />
            ) : layer && !layer.grid ? (
              <>
                No clear observation on {formatDate(layer.time.observed_date!)}:
                clouds, fog or no overpass. Nothing is shown for that day.
              </>
            ) : comp ? (
              <>
                Pixels observed {formatDate(comp.oldest_observed_date)}–
                {formatDate(comp.newest_observed_date)}.{" "}
                {cov
                  ? `${Math.round(cov.observed_fraction * 100)}% of ${p.regionLabel} ocean observed in the last ${comp.window_days} days.`
                  : ""}
              </>
            ) : layer ? (
              <>
                Overpass{" "}
                {(layer.time.observed_times ?? [])
                  .map((t) => formatDateTimePT(t))
                  .join(", ")}
                .{" "}
                {cov
                  ? `${Math.round(cov.observed_fraction * 100)}% of ${p.regionLabel} ocean observed.`
                  : ""}
              </>
            ) : null}{" "}
            {CHLOROPHYLL_COPY.biomass.split(".")[0]}. {CHLOROPHYLL_COPY.gaps}
          </p>
          {layer && p.expanded && (
            <AboutLayer
              layer={layer}
              title="About this layer"
              testid="satellite-caveats"
            />
          )}
        </>
      )}
    </div>
  );
}

function ImagerySection(p: Props & { imagery: LayerArtifact[] }) {
  const id = "imagery" in p.sat ? p.sat.imagery : null;
  const active = p.imagery.find((l) => l.layer_id === id) ?? p.imagery[0];
  const fresh =
    p.now && active ? classifyTime(active.freshness, active.time, p.now) : null;
  return (
    <div data-testid="observation-panel" className="space-y-2">
      <div
        role="radiogroup"
        aria-label="Imagery"
        className="flex flex-wrap gap-1.5"
      >
        {p.imagery.map((l) => (
          <button
            key={l.layer_id}
            role="radio"
            aria-checked={active?.layer_id === l.layer_id}
            aria-label={l.short_title}
            onClick={() => p.onSat({ imagery: l.layer_id })}
            className={`rounded-lg border px-2.5 py-1 text-[13px] ${active?.layer_id === l.layer_id ? "border-navy-900 bg-navy-900 text-white" : "border-hairline text-ink-2 hover:border-hairline-strong"}`}
          >
            {l.short_title} ·{" "}
            {l.time.observed_date
              ? formatDate(l.time.observed_date).replace(/^\w+, /, "")
              : "—"}
          </button>
        ))}
      </div>
      {active && (
        <div data-testid="chl-legend" className="space-y-1">
          {active.tiles?.legend_verified && active.tiles.legend_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={active.tiles.legend_url}
              alt={`NASA colour legend for ${active.title} (mg per cubic metre)`}
              className="w-full rounded bg-white p-0.5"
              loading="lazy"
            />
          ) : (
            <p className="text-[12px] text-serious">
              Legend unavailable: colours cannot be read quantitatively.
            </p>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-center gap-2 border-t border-hairline pt-2 text-[12px] text-ink-3">
        <ProductClassBadge pc="observation" />
        <span>
          NASA GIBS pictures · values cannot be read from them · tiles to zoom 7
        </span>
        <span className="ml-auto">
          <FreshnessBadge f={fresh} basis="observed_date" />
        </span>
      </div>
      <p className="text-[12px] text-ink-2">{CHLOROPHYLL_COPY.biomass}</p>
    </div>
  );
}

// ---------------------------------------------------------------- disclosure
function AboutLayer({
  layer,
  title,
  testid,
}: {
  layer: LayerArtifact;
  title: string;
  testid: string;
}) {
  return (
    <details className="group text-[12px] text-ink-2">
      <summary className="cursor-pointer list-none font-medium text-accent [&::-webkit-details-marker]:hidden">
        <span className="group-open:hidden">▸</span>
        <span className="hidden group-open:inline">▾</span> {title}: caveats,
        source and method
      </summary>
      <div className="mt-1.5 space-y-2">
        <ul
          data-testid={testid}
          className="list-disc space-y-1 pl-4 leading-snug"
        >
          {layer.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11.5px]">
          <dt className="text-ink-3">Source</dt>
          <dd>
            <a
              className="text-accent hover:underline"
              href={layer.provenance.source_url}
              target="_blank"
              rel="noreferrer"
            >
              {layer.provenance.source_name}
            </a>
          </dd>
          {layer.provenance.dataset_id && (
            <>
              <dt className="text-ink-3">Dataset</dt>
              <dd className="break-all font-mono">
                {layer.provenance.dataset_id}
              </dd>
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
              <dd className="font-mono">
                {(layer.time.observed_times ?? []).join(", ")}
              </dd>
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
  const range = (l: LayerArtifact | null) =>
    l?.composite
      ? `${formatDate(l.composite.oldest_observed_date)}–${formatDate(l.composite.newest_observed_date)}`
      : "—";
  const cov =
    ms.coverage_comparison.find((c) => c.region_id === regionId) ??
    ms.coverage_comparison.find((c) => c.region_id === "domain");
  const agree = [
    ms.agreement.find((a) => a.region_id === regionId),
    ms.agreement.find((a) => a.region_id === "domain"),
  ].find((a) => a && a.median_log10_ratio != null);
  const where = cov?.region_id === "domain" ? "the domain" : regionLabel;
  const sec = ms.shown?.find((x) => x.order === 1);
  return (
    <>
      Sentinel-3 pixels observed {range(primary)}; VIIRS {range(secondary)}.{" "}
      {sec?.median_age_days != null && sec.pixels > 0 && (
        <span data-testid="multi-viirs-age">
          VIIRS pixels shown are a median {Math.round(sec.median_age_days)} days
          old (VIIRS is published about 5 days after observation).{" "}
        </span>
      )}
      {cov && (
        <span data-testid="multi-coverage">
          {Math.round(cov.combined_fraction * 100)}% of {where} ocean observed
          by either sensor (Sentinel-3 alone{" "}
          {Math.round(cov.primary_fraction * 100)}%).{" "}
        </span>
      )}
      <span data-testid="sat-agreement">
        {agree ? (
          <>
            Where both saw the same water on the same day
            {agree.region_id === "domain" ? "" : ` in ${agree.label}`},{" "}
            {ratioPhrase(agree.median_log10_ratio!)} (
            {agree.n_cells.toLocaleString("en-US")} cells): a colour step at a
            sensor edge may be the sensors, not the water.{" "}
          </>
        ) : (
          <>No same-day overlap this week to compare the two sensors. </>
        )}
      </span>
    </>
  );
}

// ---------------------------------------------------------------- observed currents (HF radar)
export function selectedCurrents(
  m: Manifest,
  c: CurChoice,
): LayerArtifact | null {
  if (c.mean) return currentsMean(m);
  const hours = currentsHourly(m);
  return (
    (c.hour
      ? hours.find(
          (h) => h.time.valid_time && hourStamp(h.time.valid_time) === c.hour,
        )
      : null) ??
    hours[hours.length - 1] ??
    null
  );
}

function CurrentsSection(p: Props & { expanded: boolean }) {
  const hours = currentsHourly(p.manifest);
  const mean = currentsMean(p.manifest);
  const layer = selectedCurrents(p.manifest, p.cur);
  const idx =
    layer && !p.cur.mean
      ? hours.findIndex((h) => h.layer_id === layer.layer_id)
      : hours.length - 1;
  const fresh =
    p.now && layer ? classifyTime(layer.freshness, layer.time, p.now) : null;
  const cov = regionCoverage(layer, p.regionId);
  const go = (i: number) => {
    const h = hours[Math.max(0, Math.min(hours.length - 1, i))];
    if (h?.time.valid_time)
      p.onCur({ hour: hourStamp(h.time.valid_time), mean: false });
  };
  const t = layer?.time.valid_time;
  const failed = p.curStatus?.outcome === "failed";
  return (
    <div data-testid="currents-panel" className="space-y-2.5">
      <Seg
        label="Currents view"
        value={p.cur.mean ? "mean" : "hourly"}
        options={[
          { value: "hourly", label: "Hourly", testid: "cur-hourly" },
          {
            value: "mean",
            label: "24-hour mean",
            disabled: !mean,
            testid: "cur-mean",
            title:
              "Mean of the last 24 hours at each cell with at least 18 valid hours",
          },
        ]}
        onChange={(v) =>
          v === "mean"
            ? p.onCur({ hour: p.cur.hour, mean: true })
            : p.onCur({ hour: p.cur.hour, mean: false })
        }
      />
      {!p.cur.mean && hours.length > 0 && (
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => go(idx - 1)}
            disabled={idx <= 0}
            aria-label="Previous hour"
            data-testid="cur-hour-prev"
            className="rounded-md border border-hairline px-2 py-0.5 text-[13px] disabled:opacity-40"
          >
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
          <button
            type="button"
            onClick={() => go(idx + 1)}
            disabled={idx >= hours.length - 1}
            aria-label="Next hour"
            data-testid="cur-hour-next"
            className="rounded-md border border-hairline px-2 py-0.5 text-[13px] disabled:opacity-40"
          >
            ›
          </button>
        </div>
      )}
      <p className="text-[12.5px] text-ink-2" data-testid="cur-time">
        {layer ? (
          p.cur.mean ? (
            <>
              <span className="font-medium text-ink">24-hour mean</span>,{" "}
              {formatDateTimePT((layer.time.observed_times ?? [])[0] ?? "")} to{" "}
              {formatDateTimePT((layer.time.observed_times ?? []).at(-1) ?? "")}
            </>
          ) : (
            <>
              Observed{" "}
              <span className="font-medium text-ink">
                {t ? formatDateTimePT(t) : "—"}
              </span>
              {t && (
                <span className="text-ink-3"> ({t.slice(11, 16)} UTC)</span>
              )}
              {t && p.now && (
                <span className="text-ink-3" data-testid="cur-age">
                  {" "}
                  · {hoursAgo(t, p.now)} h ago
                </span>
              )}
              {idx === hours.length - 1 && (
                <span className="text-ink-3"> · newest hour</span>
              )}
            </>
          )
        ) : (
          "No observed currents in this dataset."
        )}
      </p>
      {p.expanded && (
        <Seg
          label="Currents drawing"
          value={p.reducedMotion ? "arrows" : p.flow}
          options={[
            { value: "arrows", label: "Arrows", testid: "cur-mode-arrows" },
            {
              value: "particles",
              label: "Animated flow",
              disabled: p.reducedMotion,
              testid: "cur-mode-particles",
              title: p.reducedMotion
                ? "Off because your system asks for reduced motion"
                : "Particles moving through this one observed field",
            },
          ]}
          onChange={(v) => p.onFlow(v as FlowMode)}
        />
      )}
      <CurrentsLegend
        particles={p.flow === "particles" && !p.reducedMotion}
        detail={p.expanded}
      />
      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5 border-t border-hairline pt-2 text-[12px] text-ink-3">
        <ProductClassBadge pc="observation" />
        <span>HF radar · HFRNet / NOAA CoastWatch</span>
        <Res l={layer} />
        <span className="ml-auto">
          {/* hourly data: the age is given in hours above; calendar days here would read "1 day ago" for a 4-hour-old hour */}
          <FreshnessBadge f={fresh} compact />
        </span>
      </div>
      <p
        className={`text-[12px] leading-snug text-ink-2 ${p.expanded ? "" : "line-clamp-2"}`}
        data-testid="cur-notes"
      >
        {cov && (
          <span data-testid="cur-coverage">
            {Math.round(cov.observed_fraction * 100)}% of {p.regionLabel} ocean
            observed {p.cur.mean ? "in the mean" : "this hour"}.{" "}
          </span>
        )}
        {p.cur.mean
          ? "The mean smooths out most daily and tidal back-and-forth. "
          : "Each hour is a separate snapshot; nothing is shown between hours. "}
        Gaps are no data, not calm water. Observed currents, not a forecast.
      </p>
      {failed && p.curStatus && (
        <p className="text-[12px] text-serious" data-testid="cur-update-failed">
          The latest update ({formatDateTimePT(p.curStatus.last_attempt_at)})
          did not refresh currents. Showing the last published hours, with their
          own times.
        </p>
      )}
      {layer && p.expanded && (
        <AboutLayer
          layer={layer}
          title="About this layer"
          testid="currents-caveats"
        />
      )}
    </div>
  );
}

const LEGEND_SPEEDS = [0.1, 0.25, 0.5, 1];
// mirrors the map's icon-size interpolation (MapCanvas): 0 -> 0.32, 0.25 -> 0.6, 0.5 -> 0.85, 1 -> 1.05
const iconScale = (sp: number) =>
  sp <= 0.25
    ? 0.32 + (sp / 0.25) * 0.28
    : sp <= 0.5
      ? 0.6 + ((sp - 0.25) / 0.25) * 0.25
      : 0.85 + Math.min(1, (sp - 0.5) / 0.5) * 0.2;

function CurrentsLegend({
  particles,
  detail,
}: {
  particles: boolean;
  detail: boolean;
}) {
  return (
    <figure
      data-testid="currents-legend"
      className="space-y-1"
      aria-label="Legend: current speed"
    >
      <figcaption className="text-[13px] font-medium text-ink">
        {particles
          ? "Particles move with the observed current"
          : "Arrows point where the surface water is moving"}
        <span className="font-normal text-ink-3"> · speed in m/s</span>
      </figcaption>
      <div className="flex items-end gap-4 rounded-lg bg-navy-900 px-3 py-2">
        {LEGEND_SPEEDS.map((sp) => (
          <span
            key={sp}
            className="flex flex-col items-center gap-1 text-[11px] text-on-navy-2 tabular"
            data-testid="currents-legend-item"
          >
            {particles ? (
              // a particle streak: length and brightness grow with speed (FlowParticles)
              <svg width={34} height={30} viewBox="0 0 34 30" aria-hidden>
                <line
                  x1={17 - sp * 15}
                  y1={15}
                  x2={17 + sp * 15}
                  y2={15}
                  stroke="#eef4fa"
                  strokeWidth={1.6}
                  strokeLinecap="round"
                  strokeOpacity={Math.min(0.95, 0.35 + sp * 1.6)}
                />
              </svg>
            ) : (
              <svg
                width={20}
                height={30}
                viewBox="0 0 20 30"
                style={{
                  transform: `scale(${iconScale(sp)})`,
                  transformOrigin: "bottom center",
                }}
                aria-hidden
              >
                <path
                  d="M10 2 L16.5 12 L11.6 12 L11.6 28 L8.4 28 L8.4 12 L3.5 12 Z"
                  fill="#eef4fa"
                  stroke="rgba(6,17,30,0.9)"
                  strokeWidth={1.1}
                  strokeLinejoin="round"
                />
              </svg>
            )}
            {sp} m/s
          </span>
        ))}
        <span className="ml-auto self-center text-[11px] text-on-navy-2">
          1 m/s ≈ {KNOTS_PER_MS.toFixed(1)} knots
        </span>
      </div>
      {detail && (
        <p className="text-[11.5px] text-ink-3">
          {particles
            ? "Screen speed shows relative speed at every zoom; particles stop where there is no observation. Not a trajectory."
            : "Longer, brighter arrows are faster. Zoom in for more arrows (one per 2 km cell)."}
        </p>
      )}
    </figure>
  );
}
