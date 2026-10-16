"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { ObservationDataset, ObsStation } from "@/generated/observations";
import type { OfficialDataset } from "@/generated/official";
import { DAY, samples, sourceCsvUrl, type ObsVariableId } from "@/lib/observations";
import { modelOnAxis, modelSegments } from "@/lib/bloom";
import { artifactUrl, charmLayer, charmRun, type CharmVariable } from "@/lib/layers";
import { loadGrid } from "@/lib/grid";
import { useNow } from "@/lib/useNow";
import { labelledGeometry } from "@/lib/officialGeometry";
import { TopBar, NavTabs } from "@/v4/shell/TopBar";
import { NoticesButton } from "@/v4/shell/Notices";
import { RegistryDisclosure } from "@/v4/shell/Disclosure";
import { Icon } from "@/v4/ui/Icon";
import { dayLabel, kmLabel, relTime } from "@/v4/ui/time";
import { SeriesChart } from "@/v4/charts/SeriesChart";
import { useReportSource } from "@/v4/readiness/store";
import { CellBody } from "@/v4/explore/Inspect";
import { useCellRead } from "@/v4/explore/reads";
import { VARIABLE_WORDS } from "@/v4/explore/layers";
import { transect, type TransectCell } from "./transect";
import { useWidth } from "@/v4/charts/useWidth";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), { ssr: false, loading: () => null });

type Props = { manifest: Manifest; ports: PortsCollection | null; observations: ObservationDataset; official: OfficialDataset | null; baseUrl: string };

const RANGES = [
  { id: "90d", label: "90 days", days: 90 },
  { id: "1y", label: "1 year", days: 365 },
  { id: "3y", label: "3 years", days: 3 * 365 },
  { id: "all", label: "All", days: Infinity },
] as const;

/** C-HARM's own threshold for particulate domoic acid (500 ng/L = 0.5 ng/mL). A reference, not a safety limit. */
const PDA_REF = { value: 0.5, label: "C-HARM threshold, 500 ng/L" };

function ModelPanel({ st, variable, from, to }: { st: ObsStation; variable: CharmVariable; from: number; to: number }) {
  const hist = st.charm?.history?.[variable];
  const { points, historyStart } = modelOnAxis(hist, from, to);
  const segs = modelSegments(points);
  const [box, W] = useWidth<HTMLElement>(720);
  const H = 120;
  const pad = { l: 44, r: 10, t: 8, b: 22 };
  const x = (t: number) => pad.l + ((t - from) / Math.max(1, to - from)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  if (!st.charm) return <p className="text-[14px] text-n-ink-2">No C-HARM history is kept for this station.</p>;
  return (
    <figure ref={box} className="flex flex-col gap-1.5" data-testid="model-panel">
      <figcaption className="text-[13.5px] leading-snug text-n-ink-2">
        NOAA C-HARM nowcast, {VARIABLE_WORDS[variable].what.toLowerCase()}: median of the valued cells within {st.charm.radius_km} km of the station (nearest cell {st.charm.nearest_cell_km != null ? kmLabel(st.charm.nearest_cell_km) : "—"}). Shown beside the measurements on the same dates, never compared with them as numbers.
      </figcaption>
      <svg width={W} height={H} className="block" role="img" aria-label={`C-HARM nowcast probability near ${st.name}, ${points.length} days in view`}>
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={pad.l} x2={W - pad.r} y1={y(v)} y2={y(v)} stroke="var(--nx-line)" />
            <text x={pad.l - 6} y={y(v) + 4} textAnchor="end" fontSize="11" fill="var(--nx-ink-3)">
              {Math.round(v * 100)}%
            </text>
          </g>
        ))}
        {historyStart != null && historyStart > from && (
          <g>
            <rect x={pad.l} y={pad.t} width={Math.max(0, x(historyStart) - pad.l)} height={H - pad.t - pad.b} fill="url(#nx-hatch)" />
            <text x={pad.l + 6} y={pad.t + 14} fontSize="11" fill="var(--nx-ink-3)">
              no model history kept before {dayLabel(new Date(historyStart).toISOString().slice(0, 10))}
            </text>
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
      </svg>
      <p className="text-[12.5px] text-n-ink-3">Open circles: days when fewer than 60% of the usual cells had a value (not joined). Days without a value are left out.</p>
    </figure>
  );
}

function csvFor(st: ObsStation, variable: ObsVariableId, units: string, from: number, to: number, generated: string): string {
  const rows = samples(st, variable).filter((s) => s.t >= from && s.t <= to);
  const head = [
    `# Neritic research export: ${st.name} (${st.station_id}), ${variable} in ${units}`,
    `# Source: CalHABMAP via SCCOOS ERDDAP, ${st.source_url}`,
    `# Values as published in Neritic's dataset generated ${generated}; empty value = not measured (never zero); qualifier as published`,
    `# Not a seafood test. Independent research preview; not affiliated with NOAA, CDFW or CDPH.`,
    "sample_time_utc,value,qualifier",
  ];
  return [...head, ...rows.map((r) => `${r.time},${r.value ?? ""},${r.q ?? ""}`)].join("\n") + "\n";
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
  const [range, setRange] = useState<(typeof RANGES)[number]["id"]>("1y");
  const first = Date.parse(st.sample_times[0] ?? new Date().toISOString());
  const last = Date.parse(st.sample_times[st.sample_times.length - 1] ?? new Date().toISOString());
  const days = RANGES.find((r) => r.id === range)!.days;
  const to = last + 4 * DAY;
  const from = Number.isFinite(days) ? Math.max(first - 4 * DAY, to - days * DAY) : first - 4 * DAY;
  const ss = useMemo(() => samples(st, variable), [st, variable]);
  const win = ss.filter((s) => s.t >= from && s.t <= to);
  const measured = win.filter((s) => s.value != null);
  const zeros = measured.filter((s) => s.q === "reported_zero" || s.value === 0).length;
  const max = measured.reduce<(typeof measured)[number] | null>((m, s) => (m == null || (s.value ?? -1) > (m.value ?? -1) ? s : m), null);
  const sorted = measured.map((s) => s.value!).sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] : null;
  const csv = sourceCsvUrl(st);
  const modelVar: CharmVariable = variable === "pn_seriata" || variable === "pn_delicatissima" ? "pseudo_nitzschia" : "particulate_domoic";
  const setStation = (id: string) => router.replace(`/research?station=${encodeURIComponent(id)}`, { scroll: false });

  useReportSource({ id: "data", label: "Neritic dataset", state: "ready", detail: `Observations of ${stations.length} stations`, asOf: observations.generated_at });

  // ------------------------------------------------------------------ map: native grid inspection and transects
  const [tool, setTool] = useState<"cell" | "transect">("cell");
  const [cellAtPt, setCellAtPt] = useState<{ lat: number; lon: number } | null>(null);
  const [line, setLine] = useState<{ lat: number; lon: number }[]>([]);
  const run = charmRun(manifest);
  const lead = run?.leads_available.includes(0) ? 0 : (run?.leads_available[0] ?? 0);
  const cellRead = useCellRead(manifest, baseUrl, ports, cellAtPt, modelVar, lead);
  const [profile, setProfile] = useState<TransectCell[] | null>(null);
  useEffect(() => {
    if (line.length !== 2) return setProfile(null);
    const l = charmLayer(manifest, modelVar, lead);
    if (!l?.grid) return setProfile(null);
    let live = true;
    loadGrid(artifactUrl(baseUrl, l.grid.url)).then((codes) => live && setProfile(transect(l.grid!, codes, line[0], line[1])));
    return () => {
      live = false;
    };
  }, [line, manifest, baseUrl, modelVar, lead]);
  const highlight = useMemo<GeoJSON.FeatureCollection>(() => {
    const f: GeoJSON.Feature[] = [{ type: "Feature", geometry: { type: "Point", coordinates: [st.lon!, st.lat!] }, properties: { kind: "station", label: st.name } }];
    if (cellRead?.status === "ready" && cellRead.polygon) f.push({ type: "Feature", geometry: cellRead.polygon, properties: { kind: "cell" } });
    if (line.length === 2)
      f.push({ type: "Feature", geometry: { type: "LineString", coordinates: line.map((p) => [p.lon, p.lat]) }, properties: { kind: "transect" } });
    else if (line.length === 1) f.push({ type: "Feature", geometry: { type: "Point", coordinates: [line[0].lon, line[0].lat] }, properties: { kind: "point" } });
    return { type: "FeatureCollection", features: f };
  }, [st, cellRead, line]);
  const forecastLayer = charmLayer(manifest, modelVar, lead);

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
              <div className="flex gap-1" role="radiogroup" aria-label="Time range">
                {RANGES.map((r) => (
                  <button key={r.id} type="button" role="radio" aria-checked={range === r.id} onClick={() => setRange(r.id)} className={`nx-press nx-focus min-h-11 rounded-lg px-3 text-[14px] font-semibold ${range === r.id ? "bg-n-accent-bg text-n-accent ring-1 ring-n-accent" : "text-n-ink-2"}`} data-testid={`range-${r.id}`}>
                    {r.label}
                  </button>
                ))}
              </div>
            </div>
            <SeriesChart samples={ss} variable={variable} units={vmeta.units} label={vmeta.label} from={from} to={to} height={240} refLine={variable === "pDA" ? PDA_REF : undefined} />
            {(variable === "pDA" || variable === "pn_seriata" || variable === "pn_delicatissima") && <ModelPanel st={st} variable={modelVar} from={from} to={to} />}
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
                    {median == null ? "—" : median.toLocaleString("en-US", { maximumSignificantDigits: 3 })} <span className="text-[13px] font-normal text-n-ink-3">{vmeta.units}</span>
                  </dd>
                </div>
                <div>
                  <dt className="text-[12.5px] text-n-ink-3">Highest</dt>
                  <dd className="nx-data text-[22px] font-semibold">
                    {max?.value == null ? "—" : max.value.toLocaleString("en-US", { maximumSignificantDigits: 3 })} <span className="text-[13px] font-normal text-n-ink-3">{max ? dayLabel(max.time) : ""}</span>
                  </dd>
                </div>
              </dl>
              <p className="mt-3 text-[12.5px] leading-snug text-n-ink-3">Of {win.length} visits in the window; a visit without this measurement is not counted and is never a zero.</p>
            </div>
            <div className="nx-deck p-4" data-testid="research-provenance">
              <h2 className="nx-eyebrow">Provenance and quality</h2>
              <dl className="mt-3 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5 text-[13.5px]">
                <dt className="text-n-ink-3">Method</dt>
                <dd>{vmeta.method}</dd>
                <dt className="text-n-ink-3">Matrix</dt>
                <dd>{vmeta.matrix}</dd>
                <dt className="text-n-ink-3">Units</dt>
                <dd>{vmeta.units}</dd>
                <dt className="text-n-ink-3">Zeros</dt>
                <dd>{vmeta.zero_policy}</dd>
                {vmeta.detection_limit_note && (
                  <>
                    <dt className="text-n-ink-3">Detection</dt>
                    <dd>{vmeta.detection_limit_note}</dd>
                  </>
                )}
                <dt className="text-n-ink-3">Retrieved</dt>
                <dd>{st.retrieved_at ? `${dayLabel(st.retrieved_at)}${now ? ` (${relTime(st.retrieved_at, now)})` : ""}` : "—"}</dd>
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
              <div className="mt-3 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => {
                    const blob = new Blob([csvFor(st, variable, vmeta.units, from, to, observations.generated_at)], { type: "text/csv" });
                    const a = document.createElement("a");
                    a.href = URL.createObjectURL(blob);
                    a.download = `neritic-${st.station_id}-${variable}.csv`;
                    a.click();
                    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
                  }}
                  className="nx-press nx-focus flex min-h-11 items-center gap-2 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold"
                  data-testid="research-export"
                >
                  <Icon name="download" className="h-4 w-4" /> Export window (CSV)
                </button>
                {csv && (
                  <a href={csv} target="_blank" rel="noreferrer" className="nx-focus flex min-h-11 items-center gap-2 rounded-lg border border-n-line-strong px-3 text-[14px] font-semibold">
                    Original data (SCCOOS) <Icon name="external" className="h-4 w-4" />
                  </a>
                )}
              </div>
            </div>
          </section>

          <section className="nx-deck grid gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_360px]" data-testid="research-grid">
            <div className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="nx-eyebrow">Native C-HARM grid · {VARIABLE_WORDS[modelVar].short} · {forecastLayer?.time.valid_date ? dayLabel(forecastLayer.time.valid_date, true) : "—"}</h2>
                <div className="flex gap-1" role="radiogroup" aria-label="Map tool">
                  {(["cell", "transect"] as const).map((t) => (
                    <button key={t} type="button" role="radio" aria-checked={tool === t} onClick={() => { setTool(t); setLine([]); }} className={`nx-press nx-focus min-h-11 rounded-lg px-3 text-[14px] font-semibold ${tool === t ? "bg-n-ink text-n-bg" : "text-n-ink-2"}`} data-testid={`tool-${t}`}>
                      {t === "cell" ? "Inspect a cell" : "Draw a transect"}
                    </button>
                  ))}
                </div>
              </div>
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
                  initialBounds={[[st.lon! - 0.35, st.lat! - 0.25], [st.lon! + 0.35, st.lat! + 0.25]]}
                  initialPadding={16}
                  onPort={() => undefined}
                  onReady={() => undefined}
                  onPoint={(p) => {
                    if (tool === "cell") setCellAtPt(p);
                    else setLine((l) => (l.length >= 2 ? [p] : [...l, p]));
                  }}
                />
                <p className="pointer-events-none absolute bottom-2 left-2 rounded bg-[rgba(4,18,26,0.75)] px-2 py-1 text-[11px] text-[#c8d6de]">© OpenStreetMap · OpenMapTiles · NOAA relief · not for navigation</p>
              </div>
              {tool === "transect" && (
                <div data-testid="transect">
                  {line.length < 2 ? (
                    <p className="text-[14px] text-n-ink-2">Click two points on the map. The profile lists every published cell the line crosses; cells without a value stay gaps.</p>
                  ) : profile ? (
                    <TransectChart cells={profile} />
                  ) : (
                    <p className="text-[14px] text-n-ink-2">Reading the grid…</p>
                  )}
                </div>
              )}
            </div>
            <div>{tool === "cell" ? cellAtPt ? <CellBody read={cellRead} ctx={{ manifest, baseUrl, ports, portIntel: null, stations: null, now, variable: modelVar, lead, onSelect: (s) => s.kind === "cell" && setCellAtPt({ lat: s.lat, lon: s.lon }) }} variable={modelVar} /> : <p className="text-[14px] text-n-ink-2">Click the water to read a native 3 km cell: its published value, footprint, the run&rsquo;s days and whether it represents a harbour.</p> : null}</div>
          </section>
        </main>
      </div>
      <div className="env-instrument fixed inset-x-0 bottom-0 z-40 border-t border-n-line bg-n-surface lg:hidden" style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
        <NavTabs active="research" size="dock" />
      </div>
    </div>
  );
}

function TransectChart({ cells }: { cells: TransectCell[] }) {
  const total = cells[cells.length - 1]?.km ?? 1;
  const [box, W] = useWidth<HTMLElement>(640);
  const H = 150;
  const pad = { l: 40, r: 8, t: 8, b: 26 };
  const x = (km: number) => pad.l + (km / Math.max(0.1, total)) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v) * (H - pad.t - pad.b);
  const valued = cells.filter((c) => c.value != null).length;
  return (
    <figure ref={box} className="flex flex-col gap-1">
      <svg width={W} height={H} className="block" role="img" aria-label={`Transect of ${cells.length} cells over ${total.toFixed(1)} km; ${valued} with a published value`}>
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
          const x1 = x(cells[i + 1]?.km ?? total);
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
        <text x={pad.l} y={H - 6} fontSize="11" fill="var(--nx-ink-3)">0 km</text>
        <text x={W - pad.r} y={H - 6} fontSize="11" fill="var(--nx-ink-3)" textAnchor="end">
          {total.toFixed(1)} km
        </text>
      </svg>
      <figcaption className="text-[12.5px] text-n-ink-3">
        {cells.length} cells crossed, {valued} with a published value; each value is held flat across its own cell (no interpolation). Hatched: no value (land, nearshore or masked).
      </figcaption>
    </figure>
  );
}
