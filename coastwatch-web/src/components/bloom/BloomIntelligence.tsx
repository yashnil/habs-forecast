"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState, useTransition } from "react";
import type { ObsStation, ObsVariable, ObservationDataset } from "@/generated/observations";
import type { OfficialDataset } from "@/generated/official";
import type { PortOfficialRelation } from "@/generated/port_intel";
import type { SourceStatus } from "@/generated/schema";
import { BLOOM_COPY, OFFICIAL_STATUS, REGION_LABEL } from "@/content/copy";
import { ModelTrack, ObservationChart, fmtDay } from "@/components/charts/ObservationChart";
import { NoticeCard, VerificationBadge, VerificationDetail } from "@/components/official/Official";
import { FreshnessBadge, ProductClassBadge } from "@/components/ui/Badges";
import { Notice, Segmented, SourceLink } from "@/components/ui/Primitives";
import { classifySource } from "@/lib/freshness";
import { officialVerification } from "@/lib/official";
import {
  CALIFORNIA_BOUNDS,
  RANGES,
  REGION_BOUNDS,
  cadenceLabel,
  formatObs,
  logDomain,
  samples,
  stationFreshness,
  stationsByRegion,
  summaryOf,
  toxinStatus,
  unitLabel,
  variableFreshness,
  windowStart,
  type ObsVariableId,
  type RangeId,
} from "@/lib/observations";
import { formatDateTimePT } from "@/lib/time";
import { useNow } from "@/lib/useNow";
import type { StationPoint } from "./StationMap";

const StationMap = dynamic(() => import("./StationMap"), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-[var(--cw-water)]" aria-label="Loading map" />,
});

const COLOR: Record<ObsVariableId, string> = {
  pDA: "var(--cw-series-toxin)",
  tDA: "var(--cw-series-toxin)",
  dDA: "var(--cw-series-toxin)",
  pn_seriata: "var(--cw-series-cells)",
  pn_delicatissima: "var(--cw-series-cells)",
  chl_extracted: "var(--cw-chl)",
  temp: "var(--cw-series-temp)",
};
const MODEL_VARS = [
  { value: "particulate_domoic", label: "Particulate DA" },
  { value: "pseudo_nitzschia", label: "Bloom" },
  { value: "cellular_domoic", label: "Cellular DA" },
] as const;
const MODEL_TITLE: Record<string, string> = {
  particulate_domoic: "Probability of particulate domoic acid above the model threshold",
  pseudo_nitzschia: "Probability of a Pseudo-nitzschia bloom",
  cellular_domoic: "Probability of cellular domoic acid above the model threshold",
};

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
};

export function BloomIntelligence({ stations, station, dataset, status, official, officialStatus, officialError, relations }: BloomProps) {
  const now = useNow();
  const router = useRouter();
  const [pending, startTransition] = useTransition();
  const [range, setRange] = useState<RangeId>("1y");
  const [modelVar, setModelVar] = useState<string>("particulate_domoic");
  const [hoverT, setHoverT] = useState<number | null>(null);

  const select = (id: string) => {
    if (id === station.station_id) return;
    setHoverT(null);
    startTransition(() => router.push(`/bloom?station=${encodeURIComponent(id)}`, { scroll: false }));
  };

  const policy = status?.freshness ?? null;
  const vars = Object.fromEntries(dataset.variables.map((v) => [v.id, v])) as Record<ObsVariableId, ObsVariable>;
  const verification = useMemo(() => (now ? officialVerification(official, officialStatus, now) : null), [official, officialStatus, now]);
  const points: StationPoint[] = stations
    .filter((s) => s.lat != null && s.lon != null)
    .map((s) => ({ id: s.station_id, name: s.name, lat: s.lat!, lon: s.lon!, recency: stationFreshness(s, policy, now)?.state ?? "unavailable" }));
  const groups = stationsByRegion({ ...dataset, stations } as ObservationDataset, Object.keys(REGION_LABEL));

  // shared time window for every chart on the page
  const end = now ? now.getTime() : Date.parse(dataset.generated_at);
  const first = station.sample_times.length ? Date.parse(station.sample_times[0]) : null;
  const x0 = windowStart(range, end, first);
  const x1 = end;

  const series = (id: ObsVariableId) => samples(station, id);
  const pnDomain = logDomain([...series("pn_seriata"), ...series("pn_delicatissima")].filter((s) => s.t >= x0 && s.value != null).map((s) => s.value!)) ?? [100, 1e6];
  const daDomain = logDomain(
    (["pDA", "tDA", "dDA"] as ObsVariableId[]).flatMap((id) => series(id)).filter((s) => s.t >= x0 && s.value != null).map((s) => s.value!),
  ) ?? [0.01, 10];
  const chlDomain = logDomain(series("chl_extracted").filter((s) => s.t >= x0 && s.value != null).map((s) => s.value!)) ?? [0.1, 100];
  const temps = series("temp").filter((s) => s.t >= x0 && s.value != null).map((s) => s.value!);
  const tempDomain: [number, number] = temps.length ? [Math.floor(Math.min(...temps) - 1), Math.ceil(Math.max(...temps) + 1)] : [8, 22];
  const measuredEver = (id: ObsVariableId) => (summaryOf(station, id)?.n_measured ?? 0) > 0;
  const modelHistory = station.charm?.history?.[modelVar] ?? [];
  const records = new Map((official?.registry.records ?? []).map((r) => [r.id, r]));
  const rel = relations.filter((r) => records.get(r.record_id)?.status === "active");
  const sf = stationFreshness(station, policy, now);
  const dsFresh = status && now ? classifySource(status, now) : null;

  return (
    <div className="mx-auto w-full max-w-[1440px] space-y-4 px-4 py-5" data-testid="bloom-page" aria-busy={pending}>
      <header className="flex flex-wrap items-end justify-between gap-3">
        <div className="max-w-3xl space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <ProductClassBadge pc="observation" />
            <FreshnessBadge f={dsFresh} basis="observed_date" />
            <span className="text-[11px] text-ink-3">newest sample in the network; each station shows its own</span>
            <span className="rounded-md border border-dashed border-hairline-strong px-1.5 py-px text-[10px] font-semibold uppercase tracking-wider text-ink-3" data-testid="science-review-pending">
              Scientific review pending
            </span>
          </div>
          <h1 className="text-[22px] font-semibold tracking-tight text-ink">{BLOOM_COPY.heading}</h1>
          <p className="text-[13.5px] leading-relaxed text-ink-2">{BLOOM_COPY.lede}</p>
        </div>
        <p className="text-[11.5px] text-ink-3">
          {dataset.program} · retrieved {formatDateTimePT(dataset.generated_at)}
        </p>
      </header>

      {/* official notices first */}
      <section className="space-y-2 rounded-lg border border-official-line bg-surface p-3.5" data-testid="bloom-official" aria-labelledby="bloom-official-h">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id="bloom-official-h" className="text-[13.5px] font-semibold text-ink">
            {OFFICIAL_STATUS.heading} — these take precedence over anything on this page
          </h2>
          <VerificationBadge v={verification} />
        </div>
        {!official ? (
          <Notice tone="serious" title="Official notices unavailable">
            {officialError ? `${officialError}. ` : ""}Check CDFW and CDPH directly.
          </Notice>
        ) : (
          <>
            {verification && verification.state !== "verified" && <VerificationDetail v={verification} brief />}
            {rel.length > 0 ? (
              <div className="grid gap-1.5 lg:grid-cols-2">
                {rel.map((x) => (
                  <NoticeCard key={x.record_id} r={records.get(x.record_id)!} now={now} relationNote={x.note} compact />
                ))}
              </div>
            ) : (
              <p className="text-[12px] text-ink-2">
                No active notice in CoastWatch&apos;s list names {station.nearest_port_name ? `${station.nearest_port_name}'s county or latitude` : "this area"}.
              </p>
            )}
            <p className="text-[11.5px] text-ink-3">
              Notices shown are those whose stated area may include waters near {station.nearest_port_name ?? "this station"}. {OFFICIAL_STATUS.missingNotOpen}{" "}
              <Link href="/" className="text-accent hover:underline">
                All notices on the Live Ocean Map →
              </Link>
            </p>
          </>
        )}
      </section>

      <div className="grid gap-4 lg:grid-cols-[360px_minmax(0,1fr)]">
        <aside className="space-y-3 lg:sticky lg:top-16 lg:self-start" aria-label="Stations">
          {/* the station map keeps the dark sea, with its key, on the paper page */}
          <div className="theme-dark space-y-2 rounded-lg p-2">
            <div className="h-[230px] overflow-hidden rounded-md border border-hairline lg:h-[280px]" data-testid="station-map">
              <StationMap stations={points} selected={station.station_id} bounds={REGION_BOUNDS[station.region ?? ""] ?? CALIFORNIA_BOUNDS} onSelect={select} />
            </div>
            <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10.5px] text-ink-3" aria-label="Map key">
              <span className="flex items-center gap-1">
                <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#eef3fa]" /> sampled in the last 2 weeks
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#b4c2d6]" /> 2–6 weeks
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2.5 w-2.5 rounded-full border-2 border-[#5d6f88]" /> older
              </span>
              <span>Dots mark sampling piers, not areas.</span>
            </div>
          </div>
          <label className="block lg:hidden">
            <span className="mb-1 block text-[11px] font-medium uppercase tracking-wider text-ink-3">Station</span>
            <select
              value={station.station_id}
              onChange={(e) => select(e.target.value)}
              className="w-full rounded-md border border-hairline-strong bg-surface-2 px-2 py-2 text-[13px] text-ink"
              data-testid="station-select"
            >
              {groups.map((g) => (
                <optgroup key={g.region} label={REGION_LABEL[g.region] ?? g.region}>
                  {g.stations.map((s) => (
                    <option key={s.station_id} value={s.station_id}>
                      {s.name}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </label>
          <nav className="hidden max-h-[calc(100dvh-440px)] min-h-[240px] space-y-3 overflow-y-auto rounded-lg border border-hairline bg-surface p-2.5 lg:block [scrollbar-width:thin]" aria-label="Station list" data-testid="station-list">
            {groups.map((g) => (
              <div key={g.region}>
                <p className="px-1 pb-1 text-[10.5px] font-semibold uppercase tracking-[0.08em] text-ink-3">
                  {REGION_LABEL[g.region] ?? g.region}
                  {g.region === "monterey_bay" && <span className="ml-1.5 normal-case tracking-normal text-accent">· flagship</span>}
                </p>
                <ul className="space-y-0.5">
                  {g.stations.map((s) => (
                    <StationRow key={s.station_id} s={s} active={s.station_id === station.station_id} pda={vars.pDA} onSelect={select} policy={policy} now={now} />
                  ))}
                </ul>
              </div>
            ))}
          </nav>
        </aside>

        <article className="min-w-0 space-y-4 rounded-lg border border-hairline bg-surface p-4" aria-labelledby="station-h" data-testid="station-detail" data-station={station.station_id}>
          <header className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-[10.5px] font-semibold uppercase tracking-[0.08em] text-accent">Shore station · {REGION_LABEL[station.region ?? "other"]}</p>
              <h2 id="station-h" className="text-[20px] font-semibold tracking-tight text-ink">
                {station.name}
              </h2>
              <p className="text-[12px] text-ink-3">
                {station.lat != null && `${station.lat.toFixed(3)}°N, ${Math.abs(station.lon!).toFixed(3)}°W · `}
                location code {station.location_code ?? "—"}
                {station.nearest_port_name && ` · ${station.nearest_port_km} km from ${station.nearest_port_name} (CDFW port)`}
              </p>
            </div>
            <div className="text-right">
              <FreshnessBadge f={sf} basis="observed_date" />
              <p className="mt-1 text-[11px] text-ink-3" data-testid="last-sample">
                Last sample {station.sample_times.length ? fmtDay(Date.parse(station.sample_times.at(-1)!)) : "—"}
              </p>
            </div>
          </header>

          {station.status === "carried_forward" && (
            <Notice tone="warning" title="Latest update failed for this station" testid="station-carried">
              Showing the data retrieved {station.retrieved_at ? formatDateTimePT(station.retrieved_at) : "earlier"}, with their original sample dates. Newer samples may exist.
            </Notice>
          )}
          {sf?.state === "historical" && (
            <Notice tone="serious" title="No recent samples" testid="station-historical">
              The newest sample is {sf.ageDays} days old. These values do not describe current conditions.
            </Notice>
          )}

          <p className="rounded-md border border-hairline bg-surface-2/60 px-3 py-2 text-[12px] leading-snug text-ink-2" data-testid="measured-vs-model">
            {BLOOM_COPY.measuredVsModel} {BLOOM_COPY.notSeafood}
          </p>

          <section aria-labelledby="latest-h" className="space-y-2">
            <h3 id="latest-h" className="text-[13px] font-semibold text-ink">
              Latest measurement of each quantity
            </h3>
            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-5" data-testid="latest-tiles">
              {(["pDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"] as ObsVariableId[]).map((id) => (
                <LatestTile key={id} v={vars[id]} st={station} policy={policy} now={now} />
              ))}
            </div>
            <p className="text-[11px] text-ink-3">{BLOOM_COPY.absence}</p>
          </section>

          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-hairline pt-3">
            <h3 className="text-[13px] font-semibold text-ink">Measurements over time</h3>
            <div className="w-[300px] max-w-full">
              <Segmented label="Time range" value={range} onChange={setRange} options={RANGES.map((r) => ({ value: r.id, label: r.label }))} testidPrefix="range" />
            </div>
          </div>

          <div className="space-y-5">
            <ObservationChart
              testid="chart-pDA"
              title={vars.pDA.label}
              variable={vars.pDA}
              samples={series("pDA")}
              x0={x0}
              x1={x1}
              scale="log"
              yDomain={daDomain}
              color={COLOR.pDA}
              hoverT={hoverT}
              onHoverT={setHoverT}
            />
            {(["dDA", "tDA"] as ObsVariableId[]).filter(measuredEver).map((id) => (
              <ObservationChart key={id} testid={`chart-${id}`} title={vars[id].label} variable={vars[id]} samples={series(id)} x0={x0} x1={x1} scale="log" yDomain={daDomain} color={COLOR[id]} hoverT={hoverT} onHoverT={setHoverT} height={130} />
            ))}
            <div className="grid gap-5 xl:grid-cols-2">
              {(["pn_seriata", "pn_delicatissima"] as ObsVariableId[]).map((id) =>
                !measuredEver(id) ? (
                  <div key={id} className="space-y-1" data-testid={`chart-${id}-never`}>
                    <p className="text-[12.5px] font-semibold text-ink">{vars[id].label}</p>
                    <p className="rounded-md border border-dashed border-hairline-strong px-3 py-2 text-[12px] text-ink-2">
                      This station has not reported this quantity since {fmtYear(station)}. No measurement is not the same as no cells.
                    </p>
                  </div>
                ) : (
                <ObservationChart key={id} testid={`chart-${id}`} title={vars[id].label} variable={vars[id]} samples={series(id)} x0={x0} x1={x1} scale="log" yDomain={pnDomain} color={COLOR[id]} hoverT={hoverT} onHoverT={setHoverT} height={140} />
                ),
              )}
            </div>
            <p className="-mt-2 text-[11px] text-ink-3">Both Pseudo-nitzschia panels share one axis. A cell count is not a toxin measurement; not every Pseudo-nitzschia produces toxin.</p>
            <div className="grid gap-5 xl:grid-cols-2">
              <ObservationChart testid="chart-chl" title={vars.chl_extracted.label} variable={vars.chl_extracted} samples={series("chl_extracted")} x0={x0} x1={x1} scale="log" yDomain={chlDomain} color={COLOR.chl_extracted} hoverT={hoverT} onHoverT={setHoverT} height={130} />
              <ObservationChart testid="chart-temp" title={vars.temp.label} variable={vars.temp} samples={series("temp")} x0={x0} x1={x1} scale="linear" yDomain={tempDomain} color={COLOR.temp} hoverT={hoverT} onHoverT={setHoverT} height={130} />
            </div>
            <p className="-mt-2 text-[11px] text-ink-3">{BLOOM_COPY.chlNotToxin}</p>
          </div>

          <section className="space-y-2 border-t-2 border-dashed border-hairline-strong pt-4" aria-labelledby="model-h" data-testid="model-section">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <div className="flex items-center gap-2">
                  <ProductClassBadge pc="official_forecast" />
                  <h3 id="model-h" className="text-[13px] font-semibold text-ink">
                    C-HARM v3.1 model near this station
                  </h3>
                </div>
                <p className="mt-1 max-w-2xl text-[11.5px] text-ink-3">
                  Median of model cells within {station.charm?.radius_km ?? 15} km
                  {station.charm?.nearest_cell_km != null ? ` (nearest model cell ${station.charm.nearest_cell_km} km away)` : ""}. Different quantity, different axis: not
                  comparable with the measurements above.
                </p>
              </div>
              <div className="w-[330px] max-w-full">
                <Segmented label="Model variable" value={modelVar} onChange={setModelVar} options={MODEL_VARS.map((m) => ({ value: m.value, label: m.label }))} testidPrefix="model-var" />
              </div>
            </div>
            {station.charm && !station.charm.error && modelHistory.length > 0 ? (
              <ModelTrack testid="model-track" title={MODEL_TITLE[modelVar]} points={modelHistory} x0={x0} x1={x1} hoverT={hoverT} onHoverT={setHoverT} color="var(--cw-forecast)" />
            ) : (
              <Notice tone="neutral" title="Model history unavailable for this station" testid="model-unavailable">
                <p>C-HARM history could not be retrieved for this location in this run. This says nothing about bloom conditions.</p>
                {station.charm?.error && (
                  <details className="mt-1 text-[11px] text-ink-3">
                    <summary className="cursor-pointer">Technical detail</summary>
                    <p className="break-all">{station.charm.error}</p>
                  </details>
                )}
              </Notice>
            )}
            <p className="text-[11px] text-ink-3">
              Model probability is not a closure decision and a low value does not mean an area is safe.{" "}
              <Link href={`/?region=${station.region ?? "california"}`} className="text-accent hover:underline">
                Map forecast →
              </Link>
            </p>
          </section>

          <Methods station={station} dataset={dataset} />
        </article>
      </div>
    </div>
  );
}

function StationRow({
  s,
  active,
  pda,
  onSelect,
  policy,
  now,
}: {
  s: ObsStation;
  active: boolean;
  pda: ObsVariable;
  onSelect: (id: string) => void;
  policy: SourceStatus["freshness"] | null;
  now: Date | null;
}) {
  const f = stationFreshness(s, policy, now);
  const sm = summaryOf(s, "pDA");
  return (
    <li>
      <button
        onClick={() => onSelect(s.station_id)}
        aria-current={active ? "true" : undefined}
        data-testid={`station-row-${s.station_id}`}
        className={`w-full rounded-md px-2 py-1.5 text-left transition-colors hover:bg-surface-2 ${active ? "bg-surface-3 ring-1 ring-hairline-strong" : ""}`}
      >
        <span className="flex items-center justify-between gap-2">
          <span className="truncate text-[12.5px] font-medium text-ink">{s.name}</span>
          <FreshnessBadge f={f} compact />
        </span>
        <span className="block truncate text-[11px] text-ink-3">
          {s.status === "failed"
            ? "Unavailable this run"
            : `${s.sample_times.length ? `Last sample ${fmtDay(Date.parse(s.sample_times.at(-1)!), false)}` : "No samples"} · ${toxinStatus(pda, sm)}`}
        </span>
      </button>
    </li>
  );
}

function LatestTile({ v, st, policy, now }: { v: ObsVariable; st: ObsStation; policy: SourceStatus["freshness"] | null; now: Date | null }) {
  const sm = summaryOf(st, v.id);
  const f = variableFreshness(sm, policy, now);
  return (
    <div className="space-y-1 rounded-md border border-hairline bg-surface-2/60 p-2.5" data-testid={`latest-${v.id}`}>
      <p className="text-[11px] leading-tight text-ink-3">{v.label}</p>
      {sm && sm.n_measured > 0 ? (
        <>
          <p className="text-[15px] font-semibold leading-tight text-ink tabular">{formatObs(v, sm.last_value ?? null, sm.last_qualifier ?? null)}</p>
          <p className="text-[11px] text-ink-3">
            {fmtDay(Date.parse(`${sm.last_date}T12:00:00Z`))} · <FreshnessBadge f={f} compact />
          </p>
          <p className="text-[10.5px] text-ink-3">{sm.n_last_365d ? `${sm.n_last_365d} in last 12 months, ${cadenceLabel(sm.median_interval_days_365d)}` : "none in the last 12 months"}</p>
        </>
      ) : (
        <p className="text-[12.5px] font-medium text-ink-2">Not measured at this station since {fmtYear(st)}</p>
      )}
    </div>
  );
}

const fmtYear = (st: ObsStation) => (st.sample_times[0] ? st.sample_times[0].slice(0, 4) : "2014");

function Methods({ station, dataset }: { station: ObsStation; dataset: BloomProps["dataset"] }) {
  return (
    <section className="space-y-2 border-t border-hairline pt-3 text-[12px] text-ink-2" aria-labelledby="methods-h">
      <h3 id="methods-h" className="text-[13px] font-semibold text-ink">
        Methods, limits and sources
      </h3>
      <ul className="list-disc space-y-1 pl-4" data-testid="bloom-caveats">
        {dataset.caveats.map((c) => (
          <li key={c}>{c}</li>
        ))}
      </ul>
      <p className="text-[11.5px] text-ink-3">{BLOOM_COPY.reviewPending}</p>
      <details className="rounded-md border border-hairline px-3 py-2">
        <summary className="cursor-pointer font-medium text-ink">What each quantity is</summary>
        <div className="mt-2 overflow-x-auto">
          <table className="w-full min-w-[640px] text-left text-[11.5px]" data-testid="variable-table">
            <thead className="text-ink-3">
              <tr>
                <th className="py-1 pr-3 font-medium">Quantity</th>
                <th className="py-1 pr-3 font-medium">Units</th>
                <th className="py-1 pr-3 font-medium">Matrix / fraction</th>
                <th className="py-1 pr-3 font-medium">Method</th>
                <th className="py-1 font-medium">Detection limit</th>
              </tr>
            </thead>
            <tbody>
              {dataset.variables.map((v) => (
                <tr key={v.id} className="border-t border-hairline align-top">
                  <td className="py-1 pr-3 text-ink">
                    {v.label}
                    <span className="block font-mono text-[10.5px] text-ink-3">{v.source_variable}</span>
                  </td>
                  <td className="py-1 pr-3 tabular">{unitLabel(v.units)}</td>
                  <td className="py-1 pr-3">
                    {v.matrix}
                    {v.fraction ? `, ${v.fraction}` : ""}
                  </td>
                  <td className="py-1 pr-3">{v.method}</td>
                  <td className="py-1">
                    {v.detection_limit ?? "not published"}
                    <span className="block text-ink-3">{v.zero_policy}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <details className="rounded-md border border-hairline px-3 py-2">
        <summary className="cursor-pointer font-medium text-ink">Quality checks for this station</summary>
        <ul className="mt-2 space-y-1 text-[11.5px]" data-testid="station-qc">
          {station.qc.map((q) => (
            <li key={q.name}>
              <span className={q.passed ? "text-ink-3" : "text-warning"}>{q.passed ? "✓" : "⚠"}</span> <span className="font-medium text-ink">{q.name}</span> — {q.detail}
            </li>
          ))}
        </ul>
      </details>
      <details className="rounded-md border border-hairline px-3 py-2">
        <summary className="cursor-pointer font-medium text-ink">How CoastWatch processes these data</summary>
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[11.5px]">
          {Object.entries(dataset.method).map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="capitalize text-ink-3">{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
      </details>
      <p className="text-[11.5px] text-ink-3">
        Source: <SourceLink href={station.source_url}>{station.station_id} on SCCOOS ERDDAP ↗</SourceLink> ·{" "}
        <SourceLink href={dataset.provenance.source_url}>CalHABMAP ↗</SourceLink> · {dataset.provenance.citation} License: {dataset.provenance.license.slice(0, 120)}…
      </p>
    </section>
  );
}

