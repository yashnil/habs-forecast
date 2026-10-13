"use client";

import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState, useTransition } from "react";
import type { ObsStation, ObsVariable, ObservationDataset } from "@/generated/observations";
import type { OfficialDataset } from "@/generated/official";
import type { PortOfficialRelation } from "@/generated/port_intel";
import type { SourceStatus } from "@/generated/schema";
import { StationChart } from "@/components/charts/StationChart";
import { PlainReading } from "./PlainReading";
import { useMode } from "@/lib/mode";
import { AXES, SELECTOR_VARS, type SelectorVar } from "@/lib/bloom";
import { DAY, monthYear, samples, summaryOf } from "@/lib/observations";
import { useNow } from "@/lib/useNow";
import { AboutData } from "./AboutData";
import { OfficialStrip } from "./OfficialStrip";
import { QUANTITY_SHORT, QuantityIndex } from "./QuantityIndex";
import { SeasonHeatmap } from "./SeasonHeatmap";
import { StationHeader } from "./StationHeader";
import { StationRail } from "./StationRail";

const RANGES = [
  { id: "3m", label: "3 mo", days: 91 },
  { id: "1y", label: "1 yr", days: 365 },
  { id: "3y", label: "3 yr", days: 3 * 365 },
  { id: "all", label: "Since 2014", days: null },
] as const;
type RangeId = (typeof RANGES)[number]["id"];

const MODEL_VARS = [
  { id: "particulate_domoic", label: "Particulate DA", title: "Probability that particulate domoic acid exceeds the model threshold" },
  { id: "pseudo_nitzschia", label: "Pseudo-nitzschia", title: "Probability that Pseudo-nitzschia exceeds the model threshold" },
  { id: "cellular_domoic", label: "Cellular DA", title: "Probability that cellular domoic acid exceeds the model threshold" },
] as const;
type ModelVarId = (typeof MODEL_VARS)[number]["id"];

export type BloomInitial = { variable?: string; range?: string; year?: string };

export type BloomProps = {
  /** every station without its series (list and map) */
  stations: ObsStation[];
  /** the selected station with full series */
  station: ObsStation;
  dataset: Pick<ObservationDataset, "variables" | "method" | "caveats" | "provenance" | "generated_at" | "window_start" | "program">;
  status: SourceStatus | null;
  official: OfficialDataset | null;
  officialStatus: SourceStatus | null;
  officialError: string | null;
  relations: PortOfficialRelation[];
  portCode?: number | null;
  initial?: BloomInitial;
  /** the producer's threshold sentence per C-HARM variable, verbatim from the manifest (as on the Ocean Map) */
  modelTitles?: Record<string, string>;
};

/**
 * Bloom Intelligence (M6): one station, one timeline. Official notices first; then the station and
 * how fresh its record is; the latest value of each quantity; one primary chart of the chosen
 * quantity with the C-HARM model directly beneath on the same time axis; the season in context
 * since 2014; and what the numbers mean. Every value is from the published dataset.
 */
/** The opening quantity and range for a station when the link names none. */
export function defaultView(station: ObsStation, now: number): { variable: SelectorVar; range: RangeId } {
  const measured = (id: SelectorVar) => (station.summaries.find((x) => x.variable === id)?.n_measured ?? 0) > 0;
  const recent = (id: SelectorVar) => samples(station, id).some((x) => x.value != null && x.t >= now - 365 * DAY);
  const variable = measured("pDA") ? "pDA" : (SELECTOR_VARS.find(recent) ?? SELECTOR_VARS.find(measured) ?? "pDA");
  return { variable, range: recent(variable) ? "1y" : "all" };
}

export function BloomIntelligence({ stations, station, dataset, status, official, officialError, relations, portCode = null, initial = {}, modelTitles = {} }: BloomProps) {
  const now = useNow();
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  // defaults follow the data (the link always wins): particulate domoic acid unless it was never
  // measured here; the last 12 months unless the quantity has nothing in them (then its whole record)
  const defaults = useMemo(() => defaultView(station, Date.parse(dataset.generated_at)), [station, dataset.generated_at]);
  const [variable, setVariable] = useState<SelectorVar>((SELECTOR_VARS as readonly string[]).includes(initial.variable ?? "") ? (initial.variable as SelectorVar) : defaults.variable);
  const [range, setRange] = useState<RangeId>(
    RANGES.some((r) => r.id === initial.range) ? (initial.range as RangeId) : initial.variable && initial.variable !== defaults.variable ? "1y" : defaults.range,
  );
  const [year, setYear] = useState<number | null>(initial.year && /^\d{4}$/.test(initial.year) ? Number(initial.year) : null);
  const [modelVar, setModelVar] = useState<ModelVarId>("particulate_domoic");
  const [pickerOpen, setPickerOpen] = useState(false);
  // phone structure (spec §7): heatmap and notes collapsed, titles still visible
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 720px)");
    const on = () => setNarrow(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  const policy = status?.freshness ?? null;
  const vars = useMemo(() => Object.fromEntries(dataset.variables.map((v) => [v.id, v])) as Record<string, ObsVariable>, [dataset.variables]);
  const [mode, , modeReady] = useMode();

  // the view (quantity, range, year) lives in the link, without a server round trip
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const set = (k: string, v: string | null, dflt: string | null) => (v == null || v === dflt ? q.delete(k) : q.set(k, v));
    set("var", variable, defaults.variable);
    set("range", year ? null : range, defaults.range);
    set("year", year ? String(year) : null, null);
    const next = `${window.location.pathname}${q.toString() ? `?${q}` : ""}`;
    if (next !== `${window.location.pathname}${window.location.search}`) window.history.replaceState(null, "", next);
  }, [variable, range, year, defaults]);

  const selectStation = (id: string) => {
    setPickerOpen(false);
    if (id === station.station_id) return;
    const q = new URLSearchParams(window.location.search);
    q.set("station", id);
    q.delete("year");
    setYear(null);
    startTransition(() => router.push(`/bloom?${q}`, { scroll: false }));
  };

  const end = now ? now.getTime() : Date.parse(dataset.generated_at);
  const first = station.sample_times.length ? Date.parse(station.sample_times[0]) : end - 365 * DAY;
  const [x0, x1] = useMemo((): [number, number] => {
    if (year) return [Date.UTC(year, 0, 1, 8), Math.min(end, Date.UTC(year + 1, 0, 1, 8))];
    const r = RANGES.find((x) => x.id === range)!;
    return [r.days == null ? first : end - r.days * DAY, end];
  }, [year, range, end, first]);

  const visits = useMemo(() => station.sample_times.map((t) => Date.parse(t)), [station.sample_times]);
  const series = useMemo(() => samples(station, variable), [station, variable]);
  const v = vars[variable];
  const axis = AXES[variable];
  const mv = MODEL_VARS.find((m) => m.id === modelVar)!;

  const rangeControls = (
    <div className="flex flex-wrap items-center gap-2">
      <div role="radiogroup" aria-label="Time range" className="flex rounded-[10px] bg-surface-3 p-[3px]">
        {RANGES.map((r) => {
          const active = !year && r.id === range;
          return (
            <button
              key={r.id}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => {
                setYear(null);
                setRange(r.id);
              }}
              data-testid={`range-${r.id}`}
              className={`rounded-[8px] px-3 py-1.5 text-[13.5px] font-medium transition-colors duration-150 ${active ? "bg-surface text-ink shadow-[0_1px_2px_rgba(13,27,42,0.12)] ring-1 ring-hairline-strong" : "text-ink-2 hover:text-ink"}`}
            >
              {r.label}
            </button>
          );
        })}
      </div>
      {year && (
        <button type="button" onClick={() => setYear(null)} className="rounded-full border border-accent/40 bg-surface px-3 py-1 text-[13.5px] font-medium text-accent" data-testid="range-year" aria-label={`Showing ${year}. Clear the year`}>
          {year} ×
        </button>
      )}
    </div>
  );

  const modelControls = (
    <div role="radiogroup" aria-label="Model quantity" className="flex flex-wrap rounded-[10px] bg-surface/70 p-[3px]">
      {MODEL_VARS.map((m) => (
        <button
          key={m.id}
          type="button"
          role="radio"
          aria-checked={m.id === modelVar}
          onClick={() => setModelVar(m.id)}
          data-testid={`model-var-${m.id}`}
          className={`rounded-[8px] px-2.5 py-1 text-[13px] font-medium ${m.id === modelVar ? "bg-surface text-model-ink ring-1 ring-model-line" : "text-ink-2 hover:text-ink"}`}
        >
          {m.label}
        </button>
      ))}
    </div>
  );

  return (
    <div className="mx-auto w-full max-w-[1440px] space-y-5 px-4 pb-16 pt-4 sm:px-8" data-testid="bloom-page" aria-busy={pending}>
      <OfficialStrip
        official={official}
        error={officialError}
        recordIds={relations.map((r) => r.record_id)}
        heading={narrow ? `{n} notices near ${station.nearest_port_name ?? "this station"}` : `Official notices for waters near ${station.nearest_port_name ?? "this station"}`}
        compact={narrow}
      />

      {pickerOpen && (
        <div role="dialog" aria-modal="true" aria-label="Choose a shore station" className="fixed inset-0 z-50 overflow-y-auto bg-page px-4 pb-10 pt-4 xl:hidden" data-testid="station-picker">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="font-display text-[24px] font-medium text-ink">Shore stations</h2>
            <button type="button" onClick={() => setPickerOpen(false)} className="rounded-md px-3 py-2 text-[14px] font-medium text-accent" autoFocus>
              Close
            </button>
          </div>
          <StationRail stations={stations} selected={station} onSelect={selectStation} policy={policy} now={now} showMap={false} testid="station-list-mobile" scrollBox="dialog" />
        </div>
      )}

      <div className="grid gap-x-10 gap-y-6 xl:grid-cols-[296px_minmax(0,1fr)]">
        <aside className="hidden xl:block" aria-label="Stations">
          <div className="sticky top-[76px] flex max-h-[calc(100dvh-96px)] flex-col">
            <StationRail stations={stations} selected={station} onSelect={selectStation} policy={policy} now={now} />
          </div>
        </aside>

        <article className="min-w-0 space-y-5 sm:space-y-7" aria-labelledby="station-h" data-testid="station-detail" data-station={station.station_id}>
          <StationHeader station={station} policy={policy} now={now} portCode={portCode} onChange={() => setPickerOpen(true)} stationCount={stations.length} />

          {/* the server renders the Coastal layout; Research swaps in the full analysis once read */}
          {(!modeReady || mode === "coastal") && <PlainReading station={station} vars={vars} now={now} />}

          {/* Coastal mode: the full analysis is one tap away; Research mode shows it as before */}
          <AnalysisFold coastal={!modeReady || mode === "coastal"}>
          <QuantityIndex station={station} vars={vars} selected={variable} onSelect={setVariable} policy={policy} now={now} x0={x0} x1={x1} />

          <StationChart
            variable={v}
            samples={series}
            visits={visits}
            axis={axis}
            x0={x0}
            x1={x1}
            title={QUANTITY_SHORT[variable]}
            empty={(() => {
              const sm = summaryOf(station, variable);
              const lastDate = sm && sm.n_measured > 0 ? sm.last_date : null;
              return (
                <>
                  <p className="font-medium text-ink">{lastDate ? `Not measured in this period. Last measured ${monthYear(lastDate)}.` : "Never measured at this station."}</p>
                  {lastDate && !(range === "all" && !year) && (
                    <button
                      type="button"
                      onClick={() => {
                        setYear(null);
                        setRange("all");
                      }}
                      className="mt-1.5 font-medium text-accent hover:underline"
                      data-testid="chart-empty-widen"
                    >
                      Show since {station.sample_times[0]?.slice(0, 4) ?? "2014"}
                    </button>
                  )}
                </>
              );
            })()}
            controls={rangeControls}
            note={
              variable === "pDA" ? (
                <span>Toxin in seawater is not toxin in seafood.</span>
              ) : variable === "chl_extracted" ? (
                <span>Chlorophyll measures algae biomass, not toxin.</span>
              ) : variable === "pn_seriata" || variable === "pn_delicatissima" ? (
                <span>A Pseudo‑nitzschia count is a size group of cells; not every Pseudo‑nitzschia produces toxin.</span>
              ) : null
            }
            model={{
              variable: modelVar,
              label: mv.label,
              title: modelTitles[modelVar] ?? mv.title,
              history: station.charm?.history?.[modelVar],
              error: station.charm?.error,
              radiusKm: station.charm?.radius_km ?? 15,
              nearestCellKm: station.charm?.nearest_cell_km,
              controls: modelControls,
            }}
          />

          <section aria-labelledby="season-h" className="border-t border-hairline pt-6">
            <details open={!narrow} className="group space-y-3" data-testid="season-section" key={narrow ? "n" : "w"}>
              <summary className="flex cursor-pointer list-none flex-wrap items-baseline justify-between gap-x-6 gap-y-1 [&::-webkit-details-marker]:hidden">
                <div>
                  <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-measured">Every {narrow ? "month" : "week"} since {station.sample_times[0]?.slice(0, 4) ?? "2014"}</p>
                  <h2 id="season-h" className="font-display text-[24px] font-medium leading-tight text-ink">
                    How this season compares: {QUANTITY_SHORT[variable].toLowerCase().replace("pseudo‑nitzschia", "Pseudo‑nitzschia")}
                    <span className="ml-2 align-middle text-[14px] text-accent group-open:hidden lg:hidden">Show</span>
                  </h2>
                </div>
                <p className="text-[13px] text-ink-3">Highest value each {narrow ? "month" : "week"}. Display bins, not risk levels.</p>
              </summary>
              <div className="pt-3">
                <SeasonHeatmap variable={v} samples={series} axis={axis} selectedYear={year} onYear={setYear} now={now} />
              </div>
            </details>
          </section>

          <div className="border-t border-hairline pt-6">
            <AboutData station={station} dataset={dataset} compact={narrow} />
          </div>
          </AnalysisFold>
        </article>
      </div>
    </div>
  );
}

/**
 * The full analysis inside one element whatever the mode: Coastal shows it as a closed fold;
 * Research holds it open with no summary. The same tree in both modes, so the chart is never
 * re-mounted when the device's choice is read after the first render.
 */
function AnalysisFold({ coastal, children }: { coastal: boolean; children: React.ReactNode }) {
  return (
    <details
      open={coastal ? undefined : true}
      className={coastal ? "group rounded-[18px] border border-hairline bg-surface" : "contents"}
      data-testid="bloom-analysis"
      data-mode={coastal ? "coastal" : "research"}
    >
      <summary className={coastal ? "flex min-h-[52px] cursor-pointer items-center justify-between gap-3 px-4 text-[15px] font-medium text-ink sm:px-5" : "hidden"}>
        <span>
          Full station analysis
          <span className="block text-[13px] font-normal text-ink-3">Every quantity, the chart over time, how this season compares, methods and downloads</span>
        </span>
        <span aria-hidden className="text-ink-3 transition-transform group-open:rotate-90">›</span>
      </summary>
      <div className={coastal ? "space-y-5 border-t border-hairline px-4 pb-6 pt-5 sm:space-y-7 sm:px-5" : "space-y-5 sm:space-y-7"}>{children}</div>
    </details>
  );
}
