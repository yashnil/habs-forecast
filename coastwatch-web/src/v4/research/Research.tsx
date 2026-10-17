"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { ObservationDataset, ObsStation } from "@/generated/observations";
import type { OfficialDataset } from "@/generated/official";
import { DAY, samples, sourceCsvUrl, unitLabel, type ObsVariableId, type Sample } from "@/lib/observations";
import { modelOnAxis, modelSegments, type ModelPoint } from "@/lib/bloom";
import { CHARM_LEADS, CHARM_VARIABLES, artifactUrl, charmLayer, charmRun, type CharmVariable } from "@/lib/layers";
import { cellEdges, distanceKm, loadGrid } from "@/lib/grid";
import { useNow } from "@/lib/useNow";
import { labelledGeometry } from "@/lib/officialGeometry";
import { TopBar, NavTabs } from "@/v4/shell/TopBar";
import { NoticesButton } from "@/v4/shell/Notices";
import { RegistryDisclosure } from "@/v4/shell/Disclosure";
import { Icon } from "@/v4/ui/Icon";
import { dayLabel, kmLabel, relTime } from "@/v4/ui/time";
import { SampleValue, SeriesChart, monthTicks, sampleOnDay } from "@/v4/charts/SeriesChart";
import { useReportSource } from "@/v4/readiness/store";
import { AreaModel, CellBody, preciseProbability } from "@/v4/explore/Inspect";
import { useCellRead } from "@/v4/explore/reads";
import { VARIABLE_WORDS } from "@/v4/explore/layers";
import { boxFeature, useAreaDrag, type BBox } from "@/v4/explore/AreaDrag";
import { transect, type TransectCell } from "./transect";
import { areaCsv, cellLeadsCsv, download, modelHistoryCsv, stationCsv, transectCsv } from "./exports";
import { Lineage } from "./Lineage";
import { useWidth } from "@/v4/charts/useWidth";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), { ssr: false, loading: () => null });

type Props = { manifest: Manifest; ports: PortsCollection | null; observations: ObservationDataset; official: OfficialDataset | null; baseUrl: string };

const RANGES = [
  { id: "90d", label: "90 days", days: 90 },
  { id: "1y", label: "1 year", days: 365 },
  { id: "3y", label: "3 years", days: 3 * 365 },
  { id: "all", label: "All", days: Infinity },
] as const;

/** C-HARM's own thresholds (its `threshold_text`): references, not safety limits. 500 ng/L = 0.5 ng/mL. */
const PDA_REF = { value: 0.5, label: "C-HARM threshold, 500 ng/L" };
const PN_REF = { value: 10_000, label: "C-HARM threshold, 10,000 cells/L (all Pseudo-nitzschia)" };

/** Short names for the y-axis title. */
const AXIS_NAME: Record<ObsVariableId, string> = {
  pDA: "pDA",
  dDA: "dDA",
  tDA: "tDA",
  pn_seriata: "P.-n. seriata group",
  pn_delicatissima: "P.-n. delicatissima group",
  chl_extracted: "Chlorophyll-a",
  temp: "Temperature",
};

const dayKey = (t: number) => new Date(t).toISOString().slice(0, 10);
const pct = (v: number) => `${Math.round(v * 100)}%`;

const BTN = "nx-press nx-focus flex min-h-11 items-center gap-2 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold";

function ExportButton({ onClick, testid, children }: { onClick: () => void; testid: string; children: React.ReactNode }) {
  return (
    <button type="button" onClick={onClick} className={`${BTN} self-start`} data-testid={testid}>
      <Icon name="download" className="h-4 w-4" /> {children}
    </button>
  );
}

function ModelPanel({ st, variable, from, to, cursorT, onCursor, onExport }: { st: ObsStation; variable: CharmVariable; from: number; to: number; cursorT: number | null; onCursor: (t: number) => void; onExport: () => void }) {
  const hist = st.charm?.history?.[variable];
  const { points, historyStart } = modelOnAxis(hist, from, to);
  const segs = modelSegments(points);
  const [box, W] = useWidth<HTMLElement>(720);
  const svg = useRef<SVGSVGElement | null>(null);
  const H = 132;
  const pad = { l: 50, r: 10, t: 8, b: 24 };
  const x = (t: number) => pad.l + ((t - from) / Math.max(1, to - from)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const months = monthTicks(from, to, W);
  if (!st.charm?.history?.[variable]?.length)
    return (
      <div className="rounded-lg border border-dashed border-n-line-strong px-3 py-3 text-[14px] text-n-ink-2" data-testid="model-panel-none">
        <p>No C-HARM nowcast history is published for this station.</p>
        {st.charm?.error && (
          <details className="mt-1 text-[12.5px] text-n-ink-3">
            <summary className="nx-focus min-h-11 cursor-pointer py-2 font-semibold text-n-accent">What the pipeline reported</summary>
            <p className="break-all">{st.charm.error}</p>
          </details>
        )}
      </div>
    );
  const near = (clientX: number) => {
    const r = svg.current?.getBoundingClientRect();
    if (!r || !points.length) return;
    const t = from + ((clientX - r.left) / r.width * W - pad.l) / (W - pad.l - pad.r) * (to - from);
    let best = points[0];
    for (const p of points) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
    onCursor(best.t);
  };
  const curX = cursorT != null && cursorT >= from && cursorT <= to ? x(cursorT) : null;
  const onDay = cursorT != null ? points.find((p) => p.date === dayKey(cursorT)) : undefined;
  return (
    <figure ref={box} className="flex flex-col gap-1.5" data-testid="model-panel">
      <figcaption className="text-[13.5px] leading-snug text-n-ink-2">
        NOAA C-HARM nowcast, {VARIABLE_WORDS[variable].what.toLowerCase()}: median of the valued cells within {st.charm!.radius_km} km of the station (nearest cell {st.charm!.nearest_cell_km != null ? kmLabel(st.charm!.nearest_cell_km) : "—"}). Shown beside the measurements on the same dates, never compared with them as numbers.
      </figcaption>
      <svg
        ref={svg}
        width={W}
        height={H}
        className="block touch-pan-y select-none"
        role="img"
        aria-label={`C-HARM nowcast probability near ${st.name}, ${points.length} days in view`}
        onPointerMove={(e) => near(e.clientX)}
        onPointerDown={(e) => near(e.clientX)}
        data-testid="model-svg"
      >
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--nx-line)" />
            <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--nx-ink-3)">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {months.filter((t) => x(t) <= W - pad.r - 18).map((t) => (
          <text key={t} x={x(t)} y={H - 6} fontSize="11.5" fill="var(--nx-ink-3)" textAnchor="middle">
            {new Date(t).toLocaleDateString("en-US", { month: "short", year: "2-digit", timeZone: "UTC" })}
          </text>
        ))}
        {historyStart != null && historyStart > from && (
          <g>
            <rect x={pad.l} y={pad.t} width={Math.max(0, x(historyStart) - pad.l)} height={H - pad.t - pad.b} fill="url(#nx-hatch)" />
            {x(historyStart) - pad.l > 230 && (
              <text x={pad.l + 6} y={pad.t + 14} fontSize="11" fill="var(--nx-ink-3)">
                no model history kept before {dayLabel(new Date(historyStart).toISOString().slice(0, 10))}
              </text>
            )}
          </g>
        )}
        <defs>
          <pattern id="nx-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--nx-line-strong)" strokeWidth="1.5" />
          </pattern>
        </defs>
        {segs.map((sg, i) => (
          <polyline key={i} points={sg.map((p) => `${x(p.t)},${y(p.v)}`).join(" ")} fill="none" stroke="var(--nx-model)" strokeWidth={1.8} />
        ))}
        {points.map((p) => (
          <circle key={p.t} cx={x(p.t)} cy={y(p.v)} r={p.partial ? 2.8 : 1.8} fill={p.partial ? "var(--nx-surface)" : "var(--nx-model)"} stroke="var(--nx-model)" strokeWidth={p.partial ? 1.2 : 0} />
        ))}
        {onDay && <circle cx={x(onDay.t)} cy={y(onDay.v)} r={4.5} fill="var(--nx-surface)" stroke="var(--nx-model)" strokeWidth={2} />}
        {curX != null && <line x1={curX} x2={curX} y1={pad.t} y2={H - pad.b} stroke="var(--nx-ink-3)" strokeDasharray="2 3" data-testid="model-cursor" />}
      </svg>
      <p className="text-[12.5px] text-n-ink-3">
        Open circles: days when fewer than 60% of the usual cells had a value (not joined). Days without a value are left out.
        {historyStart != null && historyStart > from ? ` Hatched: before ${dayLabel(new Date(historyStart).toISOString().slice(0, 10))} ${new Date(historyStart).getUTCFullYear()}, no model history is kept.` : ""}
      </p>
      <ExportButton onClick={onExport} testid="export-model">
        Model history (CSV)
      </ExportButton>
    </figure>
  );
}

/**
 * The shared cursor's readout: what was measured at the pier that day and what C-HARM published for
 * the same day, as two separate labelled lines. Two different quantities; nothing is computed
 * between them.
 */
function CursorReadout({ t, sample, unit, fmt, model, variable, radiusKm, historyStart }: { t: number | null; sample: Sample | null; unit: string; fmt: (v: number) => string; model: ModelPoint | null; variable: CharmVariable; radiusKm: number; historyStart: number | null }) {
  if (t == null) return <p className="min-h-[52px] text-[14px] text-n-ink-2">Point at either chart to read a day.</p>;
  const day = dayKey(t);
  return (
    <div className="flex min-h-[52px] flex-col gap-0.5 text-[14px]" aria-live="polite" data-testid="cursor-readout">
      <p data-testid="cursor-measured">
        <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full bg-n-measured align-middle" />
        <span className="font-semibold">Measured at the pier, {dayLabel(day, true)}:</span>{" "}
        {sample ? <SampleValue s={sample} unit={unit} fmt={fmt} /> : <span className="text-n-ink-2">no sample that day</span>}
      </p>
      <p data-testid="cursor-model">
        <span className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full bg-n-model align-middle" />
        <span className="font-semibold">C-HARM nowcast, same day ({VARIABLE_WORDS[variable].short}):</span>{" "}
        {model ? (
          <>
            <span className="nx-data text-[17px] font-semibold">{pct(model.v)}</span> <span className="text-n-ink-2">probability ({preciseProbability(model.v, null)}), median of {model.n} cells within {radiusKm} km{model.partial ? "; fewer cells than usual" : ""}</span>
          </>
        ) : (
          <span className="text-n-ink-2">{historyStart != null && t < historyStart ? "before the model history Neritic keeps" : "no model value kept for that day"}</span>
        )}
      </p>
    </div>
  );
}

export function Research({ manifest, ports, observations, official, baseUrl }: Props) {
  const router = useRouter();
  const sp = useSearchParams();
  const now = useNow();
  const stations = useMemo(() => observations.stations.filter((s) => s.lat != null && s.lon != null), [observations]);
  // a My Coast place opens on its nearest station
  const fromPlace = sp.get("place");
  const nearestTo = useMemo(() => {
    const pc = fromPlace?.startsWith("port-") ? Number(fromPlace.slice(5)) : null;
    if (pc == null) return null;
    return [...stations].filter((s) => s.nearest_port_code === pc).sort((a, b) => (a.nearest_port_km ?? 99) - (b.nearest_port_km ?? 99))[0] ?? null;
  }, [fromPlace, stations]);
  const sid = sp.get("station") ?? nearestTo?.station_id ?? "HABs-SantaCruzWharf";
  const st = stations.find((s) => s.station_id === sid) ?? stations[0];
  const vars = observations.variables.filter((v) => st.series.some((x) => x.variable === v.id && x.values.some((y) => y != null)));
  const [variable, setVariable] = useState<ObsVariableId>(vars.find((v) => v.id === "pDA")?.id ?? vars[0]?.id ?? "pDA");
  useEffect(() => {
    if (!vars.some((v) => v.id === variable)) setVariable(vars.find((v) => v.id === "pDA")?.id ?? vars[0]?.id ?? "pDA");
  }, [st.station_id]); // eslint-disable-line react-hooks/exhaustive-deps
  const vmeta = observations.variables.find((v) => v.id === variable)!;
  const unit = unitLabel(vmeta.units);
  const [range, setRange] = useState<(typeof RANGES)[number]["id"]>("1y");
  const first = Date.parse(st.sample_times[0] ?? new Date().toISOString());
  const last = Date.parse(st.sample_times[st.sample_times.length - 1] ?? new Date().toISOString());
  const days = RANGES.find((r) => r.id === range)!.days;
  const to = last + 4 * DAY;
  const from = Number.isFinite(days) ? Math.max(first - 4 * DAY, to - days * DAY) : first - 4 * DAY;
  const ss = useMemo(() => samples(st, variable), [st, variable]);
  const win = ss.filter((s) => s.t >= from && s.t <= to);
  const measured = win.filter((s) => s.value != null && s.q !== "rejected_negative");
  const zeros = measured.filter((s) => s.q === "reported_zero" || s.value === 0).length;
  const rejectedN = win.filter((s) => s.q === "rejected_negative").length;
  const flaggedN = measured.filter((s) => s.q === "flag_high").length;
  const max = measured.reduce<(typeof measured)[number] | null>((m, s) => (m == null || (s.value ?? -1) > (m.value ?? -1) ? s : m), null);
  const sorted = measured.map((s) => s.value!).sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] : null;
  const csv = sourceCsvUrl(st);
  const modelVar: CharmVariable = variable === "pn_seriata" || variable === "pn_delicatissima" ? "pseudo_nitzschia" : "particulate_domoic";
  const showModel = variable === "pDA" || variable === "pn_seriata" || variable === "pn_delicatissima";
  const hasHistory = showModel && !!st.charm?.history?.[modelVar]?.length;
  const setStation = (id: string) => router.replace(`/research?station=${encodeURIComponent(id)}`, { scroll: false });
  const fmt = (v: number) => (v >= 1000 ? Math.round(v).toLocaleString("en-US") : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toPrecision(2));

  // the shared time cursor (station chart + model panel): by default the window's newest sample
  const cursorKey = `${st.station_id}|${variable}|${range}`;
  const [cursor, setCursor] = useState<{ k: string; t: number } | null>(null);
  const defaultT = win.length ? win[win.length - 1].t : null;
  const cursorT = cursor?.k === cursorKey ? cursor.t : defaultT;
  const onCursor = (t: number) => setCursor({ k: cursorKey, t });
  const model = useMemo(() => modelOnAxis(st.charm?.history?.[modelVar], from, to), [st, modelVar, from, to]);
  const cursorSampleI = cursorT != null ? sampleOnDay(win, cursorT) : null;
  const cursorModel = cursorT != null ? (model.points.find((p) => p.date === dayKey(cursorT)) ?? null) : null;

  useReportSource({ id: "data", label: "Neritic dataset", state: "ready", detail: `Observations of ${stations.length} stations`, asOf: observations.generated_at });

  // ------------------------------------------------------------------ map: native grid inspection, transects, areas
  const run = charmRun(manifest);
  const leads = useMemo(() => (run?.leads_available ?? []).filter((l) => (CHARM_LEADS as readonly number[]).includes(l)).sort((a, b) => a - b), [run]);
  const [gridVarSel, setGridVar] = useState<CharmVariable | null>(null);
  const gridVar = gridVarSel ?? modelVar;
  const [leadSel, setLead] = useState<number | null>(null);
  const lead = leadSel != null && leads.includes(leadSel) ? leadSel : leads.includes(0) ? 0 : (leads[0] ?? 0);
  const [showEdges, setShowEdges] = useState(true);
  const [tool, setTool] = useState<"cell" | "transect" | "area">("cell");
  const [cellAtPt, setCellAtPt] = useState<{ lat: number; lon: number } | null>(null);
  const [line, setLine] = useState<{ lat: number; lon: number }[]>([]);
  const [area, setArea] = useState<BBox | null>(null);
  const [areaArmed, setAreaArmed] = useState(true);
  const mapRef = useRef<MlMap | null>(null);
  const { dragBox, handlers: areaHandlers } = useAreaDrag(() => mapRef.current, setArea, () => setAreaArmed(false));
  const cellRead = useCellRead(manifest, baseUrl, ports, cellAtPt, gridVar, lead);
  const forecastLayer = charmLayer(manifest, gridVar, lead);
  // the active layer's codes: cell edges and transects read the same published grid
  const [codes, setCodes] = useState<{ id: string; codes: Uint16Array } | null>(null);
  useEffect(() => {
    const l = forecastLayer;
    if (!l?.grid || codes?.id === l.layer_id) return;
    let live = true;
    loadGrid(artifactUrl(baseUrl, l.grid.url))
      .then((c) => live && setCodes({ id: l.layer_id, codes: c }))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [forecastLayer, baseUrl, codes?.id]);
  const active = forecastLayer?.grid && codes?.id === forecastLayer.layer_id ? codes.codes : null;
  const edges = useMemo(() => (forecastLayer?.grid && active ? cellEdges(forecastLayer.grid, active) : null), [forecastLayer, active]);
  const profile = useMemo<TransectCell[] | null>(() => (line.length === 2 && forecastLayer?.grid && active ? transect(forecastLayer.grid, active, line[0], line[1]) : null), [line, forecastLayer, active]);
  const highlight = useMemo<GeoJSON.FeatureCollection>(() => {
    const f: GeoJSON.Feature[] = [{ type: "Feature", geometry: { type: "Point", coordinates: [st.lon!, st.lat!] }, properties: { kind: "station", label: st.name } }];
    if (tool === "cell" && cellRead?.status === "ready" && cellRead.polygon) f.push({ type: "Feature", geometry: cellRead.polygon, properties: { kind: "cell" } });
    if (tool === "transect" && line.length === 2) f.push({ type: "Feature", geometry: { type: "LineString", coordinates: line.map((p) => [p.lon, p.lat]) }, properties: { kind: "transect" } });
    else if (tool === "transect" && line.length === 1) f.push({ type: "Feature", geometry: { type: "Point", coordinates: [line[0].lon, line[0].lat] }, properties: { kind: "point" } });
    const box = tool === "area" ? (dragBox ?? area) : null;
    if (box) f.push(boxFeature(box));
    return { type: "FeatureCollection", features: f };
  }, [st, cellRead, line, tool, dragBox, area]);
  const runLayers = CHARM_LEADS.map((l) => charmLayer(manifest, gridVar, l)).filter((l): l is NonNullable<typeof l> => !!l);
  const fileTag = `${gridVar}-${forecastLayer?.time.valid_date ?? "na"}`;
  const stationsIn = area ? stations.filter((s) => s.lat! >= area[1] && s.lat! <= area[3] && s.lon! >= area[0] && s.lon! <= area[2]) : [];

  return (
    <div className="env-chart min-h-dvh bg-n-bg text-n-ink" data-testid="research">
      <TopBar active="research" now={now} notices={<NoticesButton official={official} />} />
      <RegistryDisclosure className="mx-auto w-full max-w-[1440px] px-4 pb-2 sm:px-6" />
      <div className="mx-auto grid w-full max-w-[1440px] gap-6 px-4 pb-32 pt-2 sm:px-6 lg:grid-cols-[280px_minmax(0,1fr)]">
        <nav aria-label="Stations" className="lg:sticky lg:top-4 lg:max-h-[calc(100dvh-110px)] lg:self-start lg:overflow-y-auto" data-testid="research-stations">
          <p className="nx-eyebrow mb-2">Shore stations · CalHABMAP</p>
          <label htmlFor="r-station" className="sr-only">
            Station
          </label>
          <select id="r-station" value={st.station_id} onChange={(e) => setStation(e.target.value)} className="nx-focus min-h-12 w-full rounded-xl border border-n-line-strong bg-n-surface px-3 text-[15px] font-semibold lg:hidden">
            {stations.map((s) => (
              <option key={s.station_id} value={s.station_id}>
                {s.name}
              </option>
            ))}
          </select>
          <ul className="hidden flex-col gap-0.5 lg:flex">
            {stations.map((s) => {
              const lastD = s.sample_times[s.sample_times.length - 1]?.slice(0, 10);
              const on = s.station_id === st.station_id;
              return (
                <li key={s.station_id}>
                  <button type="button" onClick={() => setStation(s.station_id)} aria-current={on ? "true" : undefined} className={`nx-focus flex min-h-12 w-full flex-col justify-center rounded-lg px-3 py-1.5 text-left ${on ? "bg-n-ink text-n-bg" : "hover:bg-n-surface-2"}`}>
                    <span className="text-[14.5px] font-semibold leading-tight">{s.name}</span>
                    <span className={`text-[12.5px] ${on ? "opacity-80" : "text-n-ink-3"}`}>
                      {lastD ? `last ${dayLabel(lastD)}${lastD.slice(0, 4) !== String(new Date().getUTCFullYear()) ? ` ${lastD.slice(0, 4)}` : ""}` : "no samples"} · {s.sample_times.length} visits
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        <main id="main" className="flex min-w-0 flex-col gap-6">
          <header className="flex flex-wrap items-end justify-between gap-4">
            <div className="min-w-0">
              <p className="nx-eyebrow">Research · station record</p>
              <h1 className="mt-1 font-serif text-[34px] font-semibold leading-tight" data-testid="research-title">
                {st.name}
              </h1>
              <p className="mt-1 text-[14.5px] text-n-ink-2">
                {st.lat!.toFixed(3)}°N {Math.abs(st.lon!).toFixed(3)}°W
                {st.nearest_port_name ? ` · ${kmLabel(st.nearest_port_km ?? 0)} from ${st.nearest_port_name}` : ""} · sampled {dayLabel(st.sample_times[0] ?? "")} {st.sample_times[0]?.slice(0, 4)} to {dayLabel(st.sample_times[st.sample_times.length - 1] ?? "")} {st.sample_times[st.sample_times.length - 1]?.slice(0, 4)}
                {st.status !== "updated" ? ` · last update ${st.status.replace("_", " ")}` : ""}
              </p>
            </div>
            <Link href={`/?sel=station:${st.station_id}`} className="nx-press nx-focus flex min-h-11 items-center gap-2 rounded-xl border border-n-line-strong px-3 text-[14.5px] font-semibold">
              <Icon name="map" className="h-[18px] w-[18px]" /> Open in Explore
            </Link>
          </header>

          <section className="nx-deck flex flex-col gap-4 p-4 sm:p-5" data-testid="research-chart">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap gap-1.5" role="radiogroup" aria-label="Quantity">
                {vars.map((v) => (
                  <button key={v.id} type="button" role="radio" aria-checked={variable === v.id} onClick={() => setVariable(v.id)} className={`nx-press nx-focus min-h-11 rounded-lg px-3 text-[14px] font-semibold ${variable === v.id ? "bg-n-ink text-n-bg" : "border border-n-line-strong text-n-ink-2"}`} data-testid={`rvar-${v.id}`}>
                    {v.label}
                  </button>
                ))}
              </div>
              <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Time range">
                {RANGES.map((r) => (
                  <button key={r.id} type="button" role="radio" aria-checked={range === r.id} onClick={() => setRange(r.id)} className={`nx-press nx-focus min-h-11 rounded-lg px-3 text-[14px] font-semibold ${range === r.id ? "bg-n-accent-bg text-n-accent ring-1 ring-n-accent" : "text-n-ink-2"}`} data-testid={`range-${r.id}`}>
                    {r.label}
                  </button>
                ))}
              </div>
            </div>
            {hasHistory && <CursorReadout t={cursorT} sample={cursorSampleI != null ? win[cursorSampleI] : null} unit={unit} fmt={fmt} model={cursorModel} variable={modelVar} radiusKm={st.charm!.radius_km} historyStart={model.historyStart} />}
            <SeriesChart
              samples={ss}
              variable={variable}
              units={vmeta.units}
              label={vmeta.label}
              from={from}
              to={to}
              height={250}
              axisTitle={`${AXIS_NAME[variable]} (${unit}${variable === "temp" ? "" : ", log scale"})`}
              refLine={variable === "pDA" ? PDA_REF : variable === "pn_seriata" || variable === "pn_delicatissima" ? PN_REF : undefined}
              {...(hasHistory ? { cursorT, onCursor, hideReadout: true } : {})}
            />
            {showModel && (
              <ModelPanel
                st={st}
                variable={modelVar}
                from={from}
                to={to}
                cursorT={cursorT}
                onCursor={onCursor}
                onExport={() => download(`neritic-${st.station_id}-charm-${modelVar}.csv`, modelHistoryCsv(st, modelVar, charmLayer(manifest, modelVar, 0), manifest, observations.method.charm))}
              />
            )}
          </section>

          <section className="grid gap-4 md:grid-cols-2">
            <div className="nx-deck p-4" data-testid="research-stats">
              <h2 className="nx-eyebrow">In this window</h2>
              <dl className="mt-3 grid grid-cols-2 gap-3">
                <div>
                  <dt className="text-[12.5px] text-n-ink-3">Samples with a value</dt>
                  <dd className="nx-data text-[22px] font-semibold">{measured.length}</dd>
                </div>
                <div>
                  <dt className="text-[12.5px] text-n-ink-3">Reported as zero</dt>
                  <dd className="nx-data text-[22px] font-semibold">{zeros}</dd>
                </div>
                <div>
                  <dt className="text-[12.5px] text-n-ink-3">Median</dt>
                  <dd className="nx-data text-[22px] font-semibold">
                    {median == null ? "—" : median.toLocaleString("en-US", { maximumSignificantDigits: 3 })} <span className="text-[13px] font-normal text-n-ink-3">{unit}</span>
                  </dd>
                </div>
                <div>
                  <dt className="text-[12.5px] text-n-ink-3">Highest</dt>
                  <dd className="nx-data text-[22px] font-semibold">
                    {max?.value == null ? "—" : max.value.toLocaleString("en-US", { maximumSignificantDigits: 3 })} <span className="text-[13px] font-normal text-n-ink-3">{max ? dayLabel(max.time) : ""}</span>
                  </dd>
                </div>
              </dl>
              <p className="mt-3 text-[12.5px] leading-snug text-n-ink-3">
                Of {win.length} visits in the window; a visit without this measurement is not counted and is never a zero.
                {rejectedN ? ` ${rejectedN} rejected by QC (no value, not counted).` : ""}
                {flaggedN ? ` ${flaggedN} flagged high by QC (kept as published, counted).` : ""}
              </p>
            </div>
            <div className="nx-deck p-4" data-testid="research-provenance">
              <h2 className="nx-eyebrow">Provenance and quality</h2>
              <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[13.5px]">
                <dt className="text-n-ink-3">Method</dt>
                <dd>{vmeta.method}</dd>
                <dt className="text-n-ink-3">Matrix</dt>
                <dd>{vmeta.matrix}</dd>
                <dt className="text-n-ink-3">Units</dt>
                <dd>{unit}</dd>
                <dt className="text-n-ink-3">Zeros</dt>
                <dd>{vmeta.zero_policy}</dd>
                {vmeta.detection_limit_note && (
                  <>
                    <dt className="text-n-ink-3">Detection</dt>
                    <dd>{vmeta.detection_limit_note}</dd>
                  </>
                )}
                <dt className="text-n-ink-3">QC rules</dt>
                <dd>{observations.method.qc ?? "—"}</dd>
                <dt className="text-n-ink-3">Retrieved</dt>
                <dd>{st.retrieved_at ? `${dayLabel(st.retrieved_at)}${now ? ` (${relTime(st.retrieved_at, now)})` : ""}` : "—"}</dd>
                <dt className="text-n-ink-3">Pipeline run</dt>
                <dd className="nx-data" data-testid="research-obs-run">
                  {observations.provenance.pipeline_run_id} · version {observations.provenance.pipeline_version}
                </dd>
              </dl>
              <ul className="mt-3 flex flex-col gap-1 text-[13px]" data-testid="research-qc">
                {st.qc.map((q) => (
                  <li key={q.name} className="flex gap-2">
                    <span className={q.passed ? "text-n-ok" : "text-n-down"}>
                      <Icon name={q.passed ? "check" : "alert"} className="h-4 w-4" />
                    </span>
                    <span>
                      <b className="font-semibold">{q.name}</b> · {q.detail}
                    </span>
                  </li>
                ))}
              </ul>
              <details className="mt-2 text-[13px] text-n-ink-2">
                <summary className="nx-focus min-h-11 cursor-pointer py-2 font-semibold text-n-accent">Citation and licence</summary>
                {observations.provenance.citation && <p className="mb-1">{observations.provenance.citation}</p>}
                <p>{observations.provenance.license}</p>
              </details>
              <div className="mt-2 flex flex-wrap gap-2">
                <button type="button" onClick={() => download(`neritic-${st.station_id}-${variable}.csv`, stationCsv(observations, st, vmeta, from, to))} className={BTN} data-testid="research-export">
                  <Icon name="download" className="h-4 w-4" /> Export window (CSV)
                </button>
                {csv && (
                  <a href={csv} target="_blank" rel="noreferrer" className={BTN.replace("nx-press ", "")}>
                    Original data (SCCOOS) <Icon name="external" className="h-4 w-4" />
                  </a>
                )}
              </div>
            </div>
          </section>

          <section className="nx-deck flex flex-col gap-3 p-4 sm:p-5" data-testid="research-grid">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="nx-eyebrow" data-testid="grid-heading">
                Native C-HARM grid · {VARIABLE_WORDS[gridVar].short} · {forecastLayer?.time.valid_date ? `${lead === 0 ? "nowcast" : "forecast"} for ${dayLabel(forecastLayer.time.valid_date, true)}` : "—"}
              </h2>
              <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Map tool">
                {(["cell", "transect", "area"] as const).map((t) => (
                  <button
                    key={t}
                    type="button"
                    role="radio"
                    aria-checked={tool === t}
                    onClick={() => {
                      setTool(t);
                      setLine([]);
                      if (t === "area") setAreaArmed(true);
                    }}
                    className={`nx-press nx-focus min-h-11 rounded-lg px-3 text-[14px] font-semibold ${tool === t ? "bg-n-ink text-n-bg" : "text-n-ink-2"}`}
                    data-testid={`tool-${t}`}
                  >
                    {t === "cell" ? "Inspect a cell" : t === "transect" ? "Draw a transect" : "Select an area"}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex flex-wrap items-end gap-x-5 gap-y-3" data-testid="grid-controls">
              <fieldset className="min-w-0">
                <legend className="mb-1 text-[12.5px] text-n-ink-3">C-HARM variable</legend>
                <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="C-HARM variable">
                  {CHARM_VARIABLES.map((v) => (
                    <button key={v} type="button" role="radio" aria-checked={gridVar === v} onClick={() => setGridVar(v)} className={`nx-press nx-focus min-h-11 rounded-lg px-3 text-[14px] font-semibold ${gridVar === v ? "bg-n-accent-bg text-n-accent ring-1 ring-n-accent" : "border border-n-line-strong text-n-ink-2"}`} data-testid={`gvar-${v}`}>
                      {VARIABLE_WORDS[v].short}
                    </button>
                  ))}
                </div>
              </fieldset>
              <fieldset className="min-w-0">
                <legend className="mb-1 text-[12.5px] text-n-ink-3">Day of this run (issued {run ? `${dayLabel(run.issued_date)}${run.issued_date_derived ? ", inferred" : ""}` : "—"})</legend>
                <div className="flex flex-wrap gap-1" role="radiogroup" aria-label="Day of the run">
                  {leads.map((l) => {
                    const d = charmLayer(manifest, gridVar, l)?.time.valid_date;
                    return (
                      <button key={l} type="button" role="radio" aria-checked={lead === l} onClick={() => setLead(l)} className={`nx-press nx-focus flex min-h-11 flex-col items-center justify-center rounded-lg px-2.5 leading-tight ${lead === l ? "bg-n-accent-bg text-n-accent ring-1 ring-n-accent" : "border border-n-line-strong text-n-ink-2"}`} data-testid={`lead-${l}`}>
                        <span className="text-[13.5px] font-semibold">{d ? dayLabel(d, true) : "—"}</span>
                        <span className="text-[11.5px]">{l === 0 ? "nowcast" : `forecast +${l}`}</span>
                      </button>
                    );
                  })}
                </div>
              </fieldset>
              <label className="flex min-h-11 cursor-pointer items-center gap-2 text-[14px] font-semibold text-n-ink-2">
                <input type="checkbox" checked={showEdges} onChange={(e) => setShowEdges(e.target.checked)} className="nx-focus h-4 w-4 accent-[var(--nx-accent)]" data-testid="grid-edges" />
                Cell edges
              </label>
            </div>
            <p className="text-[12.5px] leading-snug text-n-ink-3">Only the days this run published; nothing beyond its horizon. Each value is model output for one 3 km cell, not a toxin measurement.</p>
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
              <div className="flex min-w-0 flex-col gap-2">
                <div className="env-instrument relative h-[380px] overflow-hidden rounded-xl">
                  <MapCanvas
                    forecast={forecastLayer?.image ? { id: `r-${forecastLayer.layer_id}`, url: artifactUrl(baseUrl, forecastLayer.image.url), corners: forecastLayer.image.corners_lnglat, grid: forecastLayer.grid ? { layer: forecastLayer, url: artifactUrl(baseUrl, forecastLayer.grid.url) } : null } : null}
                    satellite={null}
                    satelliteAge={null}
                    imagery={null}
                    ports={ports}
                    showPorts
                    officialGeometry={labelledGeometry(official)}
                    showOfficial
                    selectedPort={null}
                    inspect={null}
                    chrome="none"
                    highlight={highlight}
                    cellEdges={showEdges ? edges : null}
                    initialBounds={[[st.lon! - 0.35, st.lat! - 0.25], [st.lon! + 0.35, st.lat! + 0.25]]}
                    initialPadding={16}
                    onPort={() => undefined}
                    onReady={(m) => {
                      mapRef.current = m;
                    }}
                    onPoint={(p) => {
                      if (tool === "cell") setCellAtPt(p);
                      else if (tool === "transect") setLine((l) => (l.length >= 2 ? [p] : [...l, p]));
                    }}
                  />
                  {tool === "area" && areaArmed && (
                    <div {...areaHandlers} className="absolute inset-0 z-[5] cursor-crosshair touch-none" data-testid="area-overlay">
                      <p className="pointer-events-none absolute left-1/2 top-3 -translate-x-1/2 whitespace-nowrap rounded-full bg-n-surface px-4 py-2 text-[13.5px] font-semibold text-n-ink shadow-[var(--nx-shadow)]">Drag across the water to select an area</p>
                    </div>
                  )}
                  <p className="pointer-events-none absolute bottom-2 left-2 rounded bg-[rgba(4,18,26,0.75)] px-2 py-1 text-[11px] text-[#c8d6de]">© OpenStreetMap · OpenMapTiles · NOAA relief · not for navigation</p>
                </div>
                {tool === "transect" && (
                  <div data-testid="transect" className="flex flex-col gap-2">
                    {line.length < 2 ? (
                      <p className="text-[14px] text-n-ink-2">Click two points on the map. The profile lists every published cell the line crosses; cells without a value stay gaps.</p>
                    ) : profile && forecastLayer ? (
                      <>
                        <TransectChart cells={profile} a={line[0]} b={line[1]} />
                        <ExportButton testid="export-transect" onClick={() => download(`neritic-transect-${fileTag}.csv`, transectCsv(line[0], line[1], profile, forecastLayer, manifest))}>
                          Transect cells (CSV)
                        </ExportButton>
                      </>
                    ) : (
                      <p className="text-[14px] text-n-ink-2">Reading the grid…</p>
                    )}
                  </div>
                )}
              </div>
              <div className="min-w-0">
                {tool === "cell" &&
                  (cellAtPt ? (
                    <div className="flex flex-col gap-3">
                      <CellBody read={cellRead} ctx={{ manifest, baseUrl, ports, portIntel: null, stations: null, now, variable: gridVar, lead, onSelect: (s) => s.kind === "cell" && setCellAtPt({ lat: s.lat, lon: s.lon }) }} variable={gridVar} precise={{ quantErr: forecastLayer?.grid?.max_quantization_error ?? null }} />
                      {cellRead?.status === "ready" && cellRead.cell && (
                        <ExportButton testid="export-cell" onClick={() => download(`neritic-cell-${cellRead.cell!.row}-${cellRead.cell!.col}-${gridVar}.csv`, cellLeadsCsv(cellRead.cell!, cellRead.leads, runLayers, manifest, gridVar))}>
                          This cell, every day of the run (CSV)
                        </ExportButton>
                      )}
                    </div>
                  ) : (
                    <p className="text-[14px] text-n-ink-2">Click the water to read a native 3 km cell: its published value, footprint, the run&rsquo;s days and whether it represents a harbour.</p>
                  ))}
                {tool === "area" &&
                  (area ? (
                    <div className="flex flex-col gap-3" data-testid="inspect-area">
                      <p className="nx-data text-[12.5px] text-n-ink-3">
                        {area[1].toFixed(3)}–{area[3].toFixed(3)}°N, {Math.abs(area[2]).toFixed(3)}–{Math.abs(area[0]).toFixed(3)}°W
                      </p>
                      <AreaModel
                        layer={forecastLayer}
                        baseUrl={baseUrl}
                        bbox={area}
                        variable={gridVar}
                        footer={(cells) =>
                          forecastLayer && (
                            <ExportButton testid="export-area" onClick={() => download(`neritic-area-${fileTag}.csv`, areaCsv(area, cells, forecastLayer, manifest))}>
                              Cells in the box (CSV)
                            </ExportButton>
                          )
                        }
                      />
                      <section className="flex flex-col gap-1 border-t border-n-line pt-3">
                        <h3 className="nx-eyebrow">Shore stations in the box</h3>
                        {stationsIn.length ? (
                          <ul className="text-[14px]">
                            {stationsIn.map((s) => (
                              <li key={s.station_id}>
                                <button type="button" onClick={() => setStation(s.station_id)} className="nx-focus min-h-11 text-left font-semibold text-n-accent underline-offset-2 hover:underline">
                                  {s.name}
                                </button>
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-[14px] text-n-ink-2">No CalHABMAP station in this box: no sample is not a clean result.</p>
                        )}
                      </section>
                      <button type="button" onClick={() => setAreaArmed(true)} className={`${BTN} self-start`} data-testid="area-redraw">
                        Draw another box
                      </button>
                    </div>
                  ) : (
                    <p className="text-[14px] text-n-ink-2">Drag a box across the water. It counts the cells with and without a published value and gives the spread of the published probabilities; gaps are counted, never filled.</p>
                  ))}
              </div>
            </div>
          </section>

          {forecastLayer && <Lineage layer={forecastLayer} manifest={manifest} />}
        </main>
      </div>
      <div className="env-instrument fixed inset-x-0 bottom-0 z-40 border-t border-n-line bg-n-surface lg:hidden" style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
        <NavTabs active="research" size="dock" />
      </div>
    </div>
  );
}

function TransectChart({ cells, a, b }: { cells: TransectCell[]; a: { lat: number; lon: number }; b: { lat: number; lon: number } }) {
  const [box, W] = useWidth<HTMLElement>(640);
  // the whole line, so the last cell keeps its own width
  const end = Math.max(0.1, distanceKm(a.lat, a.lon, b.lat, b.lon));
  const H = 150;
  const pad = { l: 40, r: 8, t: 8, b: 26 };
  const x = (km: number) => pad.l + (km / end) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const valued = cells.filter((c) => c.value != null).length;
  const ll = (p: { lat: number; lon: number }) => `${p.lat.toFixed(3)}°N ${Math.abs(p.lon).toFixed(3)}°W`;
  return (
    <figure ref={box} className="flex flex-col gap-1">
      <svg width={W} height={H} className="block" role="img" aria-label={`Transect of ${cells.length} cells; ${valued} with a published value`}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--nx-line)" />
            <text x={pad.l - 5} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--nx-ink-3)">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {cells.map((c, i) => {
          const x0 = x(c.km);
          const x1 = x(cells[i + 1]?.km ?? end);
          return c.value == null ? (
            <rect key={i} x={x0} y={pad.t} width={Math.max(1, x1 - x0)} height={H - pad.t - pad.b} fill="url(#nx-hatch-t)" />
          ) : (
            <line key={i} x1={x0} x2={x1} y1={y(c.value)} y2={y(c.value)} stroke="var(--nx-model)" strokeWidth={2.2} />
          );
        })}
        <defs>
          <pattern id="nx-hatch-t" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--nx-line-strong)" strokeWidth="1.5" />
          </pattern>
        </defs>
        <text x={pad.l} y={H - 6} fontSize="11" fill="var(--nx-ink-3)">
          0 km
        </text>
        <text x={W - pad.r} y={H - 6} fontSize="11" fill="var(--nx-ink-3)" textAnchor="end">
          {end.toFixed(1)} km
        </text>
      </svg>
      <figcaption className="text-[12.5px] text-n-ink-3">
        {cells.length} cells crossed, {valued} with a published value; each value is held flat across its own cell (no interpolation). Hatched: no value (land, nearshore or masked). From {ll(a)} to {ll(b)}.
      </figcaption>
    </figure>
  );
}
