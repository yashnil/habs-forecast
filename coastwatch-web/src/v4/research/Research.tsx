"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { ObservationDataset } from "@/generated/observations";
import type { OfficialDataset } from "@/generated/official";
import { DAY, samples, sourceCsvUrl, unitLabel, type ObsVariableId } from "@/lib/observations";
import { modelOnAxis } from "@/lib/bloom";
import { CHARM_LEADS, CHARM_VARIABLES, artifactUrl, charmLayer, charmRun, isFixture, satelliteLatest, sourceStatus, tileIndexUrl, type CharmVariable } from "@/lib/layers";
import { currentsHourly, fieldFeatures } from "@/lib/currents";
import { cellEdges, distanceKm, loadGrid } from "@/lib/grid";
import { useNow } from "@/lib/useNow";
import { coastHref, pointKey, resolvePlace } from "@/lib/coast";
import { labelledGeometry } from "@/lib/officialGeometry";
import { TopBar, NavTabs } from "@/v4/shell/TopBar";
import { NoticesButton } from "@/v4/shell/Notices";
import { RegistryDisclosure } from "@/v4/shell/Disclosure";
import { SkipLink } from "@/v4/shell/SkipLink";
import { Icon } from "@/v4/ui/Icon";
import { dayLabel, kmLabel, lastHabWords, relTime } from "@/v4/ui/time";
import { HAB_SAMPLE_VARIABLES } from "@/lib/provenance";
import { pacificDate } from "@/lib/time";
import { LinkedSeries, readingAt, realSamples } from "@/v4/charts/LinkedSeries";
import { BrushStrip } from "@/v4/charts/BrushStrip";
import { useReportSource } from "@/v4/readiness/store";
import { currentsEntry, modelEntry, satelliteEntry } from "@/v4/readiness/shown";
import { AreaModel, CellBody } from "@/v4/explore/Inspect";
import { useCellRead } from "@/v4/explore/reads";
import { VARIABLE_WORDS } from "@/v4/explore/layers";
import { boxFeature, useAreaDrag, type BBox } from "@/v4/explore/AreaDrag";
import { transect, type TransectCell } from "./transect";
import { CoastLadder, type LadderStation } from "./CoastLadder";
import { LocatorMap, StationSheet } from "./Stations";
import { ExportButton, ExportMenu, MENU_ITEM, QUIET, Readings, SamplesTable, Seg, TabPanel, Tabs, TransectChart, VariableTabs } from "./parts";
import { areaCsv, cellLeadsCsv, download, modelHistoryCsv, stationCsv, transectCsv } from "./exports";
import { Lineage } from "./Lineage";
import { footprintOf, footprintOutline, obsArea, obsPoint, obsTransect, pixelPolygon, radarLayerFor, satLayerFor, type BBox as ObsBox, type RadarChoice, type SatChoice } from "./obsGrid";
import { ObsArea, ObsControls, ObsNote, ObsPointPanel, ObsTransect, obsSensor, obsWhen, useObsSampler, type ObsSource } from "./ObsTools";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), { ssr: false, loading: () => null });

type Props = { manifest: Manifest; ports: PortsCollection | null; observations: ObservationDataset; official: OfficialDataset | null; baseUrl: string };

export const RANGES = [
  { id: "90d", label: "90 days", short: "90 d", days: 90 },
  { id: "1y", label: "1 year", short: "1 y", days: 365 },
  { id: "3y", label: "3 years", short: "3 y", days: 3 * 365 },
  { id: "all", label: "All", short: "All", days: Infinity },
] as const;
export type RangeId = (typeof RANGES)[number]["id"];

/** C-HARM's own thresholds (its `threshold_text`): references, not safety limits. 500 ng/L = 0.5 ng/mL. */
const PDA_REF = { value: 0.5, label: "C-HARM's own threshold · 500 ng/L" };
const PN_REF = { value: 10_000, label: "C-HARM's own threshold · 10,000 cells/L (all Pseudo-nitzschia)" };

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
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/;
/** A link's day (YYYY-MM-DD) as the chart's instant (noon UTC, as samples' days are read), or null. */
export const parseDay = (v: string | null) => (v && DAY_RE.test(v) && Number.isFinite(Date.parse(`${v}T12:00:00Z`)) ? Date.parse(`${v}T12:00:00Z`) : null);

/**
 * The window the chart shows: a brushed window from the link (`start`, `end`), clamped to the record,
 * else a preset (`range`), else the default: a year, or the whole record when the last year never
 * measured this quantity or the day asked for is older (never an empty frame on arrival).
 */
export function researchWindow(o: { first: number; last: number; start: number | null; end: number | null; range: string | null; recent: boolean; askedDay: number | null }): { from: number; to: number; range: RangeId | null } {
  const x0 = o.first - 4 * DAY;
  const x1 = o.last + 4 * DAY;
  if (o.start != null && o.end != null) {
    const from = Math.max(x0, Math.min(o.start, o.end));
    const to = Math.min(x1, Math.max(o.start, o.end));
    if (to - from >= 7 * DAY) return { from, to, range: null };
  }
  const dayFits = o.askedDay != null && o.askedDay >= x0 && o.askedDay <= x1;
  const asked = RANGES.find((r) => r.id === o.range)?.id ?? null;
  const range: RangeId = asked ?? (dayFits && o.askedDay! < x1 - 365 * DAY ? "all" : o.recent ? "1y" : "all");
  const days = RANGES.find((r) => r.id === range)!.days;
  return { from: Number.isFinite(days) ? Math.max(x0, x1 - days * DAY) : x0, to: x1, range };
}

/** The link with these parameters changed (null removes one); the page is not reloaded. */
function setParams(patch: Record<string, string | null>) {
  const q = new URLSearchParams(window.location.search);
  for (const [k, v] of Object.entries(patch)) {
    if (v == null) q.delete(k);
    else q.set(k, v);
  }
  const s = q.toString().replace(/%2C/g, ",");
  window.history.replaceState(null, "", s ? `${window.location.pathname}?${s}` : window.location.pathname);
}

/** The newest HAB sample (the variables lastHabSample uses elsewhere), from a station's full series: the same date the map, Explore and the briefing give. */
function lastHabObs(s: { sample_times: string[]; series: { variable: string; values: (number | null)[] }[] }): string | null {
  let best: string | null = null;
  for (const se of s.series) {
    if (!(HAB_SAMPLE_VARIABLES as readonly string[]).includes(se.variable)) continue;
    for (let i = se.values.length - 1; i >= 0; i--) {
      if (se.values[i] == null) continue;
      const d = s.sample_times[i] ? pacificDate(s.sample_times[i]) : null;
      if (d && (!best || d > best)) best = d;
      break;
    }
  }
  return best;
}

/** The station nearest a place: one whose nearest harbour is that port first (as published), else the nearest by distance. */
export function nearestStation<S extends { station_id: string; lat: number | null; lon: number | null; nearest_port_code?: number | null; nearest_port_km?: number | null }>(stations: S[], lat: number, lon: number, key: string): { station: S; km: number } | null {
  const pc = key.startsWith("port-") ? Number(key.slice(5)) : null;
  const own = pc == null ? [] : stations.filter((s) => s.nearest_port_code === pc).sort((a, b) => (a.nearest_port_km ?? 99) - (b.nearest_port_km ?? 99));
  let best: { station: S; km: number } | null = null;
  for (const s of own.length ? own.slice(0, 1) : stations) {
    if (s.lat == null || s.lon == null) continue;
    const km = distanceKm(lat, lon, s.lat, s.lon);
    if (!best || km < best.km) best = { station: s, km };
  }
  return best;
}

/** A `variable` link parameter as a pier quantity: its own id, or a C-HARM variable's counterpart. */
export function researchVar(v: string | null): ObsVariableId | null {
  if (!v) return null;
  if (v === "particulate_domoic" || v === "cellular_domoic") return "pDA";
  if (v === "pseudo_nitzschia") return "pn_seriata";
  return ["pDA", "dDA", "tDA", "pn_seriata", "pn_delicatissima", "chl_extracted", "temp"].includes(v) ? (v as ObsVariableId) : null;
}

export function Research({ manifest, ports, observations, official, baseUrl }: Props) {
  const sp = useSearchParams();
  const now = useNow();
  const wide = useWide();
  const stations = useMemo(() => observations.stations.filter((s) => s.lat != null && s.lon != null), [observations]);
  // a place (My Coast, or Explore's "Research this") opens on its nearest station, and says so (v6);
  // the place is kept as context when another station is chosen (the links back keep it)
  const fromPlace = sp.get("place");
  const place = useMemo(() => resolvePlace(fromPlace, ports), [fromPlace, ports]);
  const nearestTo = useMemo(() => (place ? nearestStation(stations, place.lat, place.lon, place.key) : null), [place, stations]);
  const sid = sp.get("station") ?? nearestTo?.station.station_id ?? "HABs-SantaCruzWharf";
  const st = stations.find((s) => s.station_id === sid) ?? stations[0];
  const arrived = place && st.lat != null && st.lon != null && (!nearestTo || nearestTo.station.station_id === st.station_id) ? { name: place.name, km: distanceKm(place.lat, place.lon, st.lat, st.lon) } : null;
  const backHref = (() => {
    const f = sp.get("from");
    return f && f.startsWith("/") && !f.startsWith("//") ? f : place ? coastHref({ place: place.key }) : null;
  })();
  const vars = useMemo(() => observations.variables.filter((v) => st.series.some((x) => x.variable === v.id && x.values.some((y) => y != null))), [observations, st]);
  // the quantity (`variable`: a pier quantity, or a C-HARM variable read as its pier counterpart)
  const askedVar = researchVar(sp.get("variable"));
  const variable = useMemo<ObsVariableId>(() => (askedVar && vars.find((v) => v.id === askedVar)?.id) ?? vars.find((v) => v.id === "pDA")?.id ?? vars[0]?.id ?? "pDA", [askedVar, vars]);
  const vmeta = observations.variables.find((v) => v.id === variable)!;
  const unit = unitLabel(vmeta.units);
  const first = Date.parse(st.sample_times[0] ?? new Date().toISOString());
  const last = Date.parse(st.sample_times[st.sample_times.length - 1] ?? new Date().toISOString());
  const ss = useMemo(() => samples(st, variable), [st, variable]);
  const recent = ss.some((x) => x.value != null && x.t >= last + 4 * DAY - 365 * DAY);
  // the day asked for (`day`): the window reaches back to it and the reading is pinned on it
  const askedDay = parseDay(sp.get("day"));
  const spStart = sp.get("start");
  const spEnd = sp.get("end");
  const linked = researchWindow({ first, last, start: parseDay(spStart), end: parseDay(spEnd), range: sp.get("range"), recent, askedDay });
  // while the brush is dragged the chart follows; the link is written when the drag ends
  const [preview, setPreview] = useState<[number, number] | null>(null);
  useEffect(() => setPreview(null), [spStart, spEnd, sid, variable]);
  const [from, to] = preview ?? [linked.from, linked.to];
  const range = preview ? null : linked.range;
  const x0 = first - 4 * DAY;
  const x1 = last + 4 * DAY;
  const win = ss.filter((s) => s.t >= from && s.t <= to);
  const real = realSamples(win);
  const measured = win.filter((s) => s.value != null && s.q !== "rejected_negative");
  const zeros = measured.filter((s) => s.q === "reported_zero" || s.value === 0).length;
  const rejectedN = win.filter((s) => s.q === "rejected_negative").length;
  const flaggedN = measured.filter((s) => s.q === "flag_high").length;
  const max = measured.reduce<(typeof measured)[number] | null>((m, s) => (m == null || (s.value ?? -1) > (m.value ?? -1) ? s : m), null);
  const sorted = measured.map((s) => s.value!).sort((a, b) => a - b);
  const median = sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] : null;
  const allMeasured = ss.filter((s) => s.value != null && s.q !== "rejected_negative");
  const allZeros = allMeasured.filter((s) => s.q === "reported_zero" || s.value === 0).length;
  const csv = sourceCsvUrl(st);
  const modelVar: CharmVariable = variable === "pn_seriata" || variable === "pn_delicatissima" ? "pseudo_nitzschia" : "particulate_domoic";
  const showModel = variable === "pDA" || variable === "pn_seriata" || variable === "pn_delicatissima";
  const history = st.charm?.history?.[modelVar];
  const hasHistory = showModel && !!history?.length;
  const fmt = (v: number) => (v >= 1000 ? Math.round(v).toLocaleString("en-US") : v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toPrecision(2));

  // the crosshair (chart, model panel, readings): the pointer's reading, else the pinned day, else the window's newest sample
  const hoverKey = `${st.station_id}|${variable}`;
  const [hover, setHover] = useState<{ k: string; t: number } | null>(null);
  const hoverT = hover?.k === hoverKey ? hover.t : null;
  const pinnedT = askedDay;
  const defaultT = real.length ? real[real.length - 1].t : null;
  const cursorT = hoverT ?? (pinnedT != null && pinnedT >= from && pinnedT <= to ? pinnedT : defaultT);
  const onHover = useCallback((t: number | null) => setHover(t == null ? null : { k: hoverKey, t }), [hoverKey]);
  const onPin = useCallback((t: number | null) => setParams({ day: t == null ? null : dayKey(t) }), []);
  const model = useMemo(() => (hasHistory ? modelOnAxis(history, from, to) : null), [hasHistory, history, from, to]);
  const linkedModel = model ? { points: model.points, historyStart: model.historyStart, what: VARIABLE_WORDS[modelVar].short, radiusKm: st.charm?.radius_km ?? 15 } : null;
  const reading = cursorT != null ? readingAt(cursorT, win, linkedModel) : null;
  const modelAll = useMemo(() => (hasHistory ? modelOnAxis(history, -Infinity, Infinity) : null), [hasHistory, history]);
  const modelEnd = modelAll?.points.length ? modelAll.points[modelAll.points.length - 1].t : null;

  // the link: the station, the quantity and the window are always in it, so a reload or a shared link opens the same view
  const chooseVariable = (v: ObsVariableId) => setParams({ variable: v, day: null });
  const chooseRange = (r: RangeId) => setParams({ range: r, start: null, end: null });
  const commitWindow = (a: number, b: number) => {
    setPreview([a, b]);
    setParams({ start: dayKey(a), end: dayKey(b), range: null });
  };
  const setStation = (id: string) => setParams({ station: id, day: null });
  const [sheet, setSheet] = useState(false);
  const closeSheet = useCallback(() => setSheet(false), []);

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
  // the grid map (WebGL, its style, tiles and worker) is made only when it comes within a screen of view,
  // or a map tool is chosen: on a phone it sits far below the record, and many readers never reach it
  const mapBox = useRef<HTMLDivElement | null>(null);
  const [mapNear, setMapNear] = useState(false);
  useEffect(() => {
    const el = mapBox.current;
    if (mapNear || !el) return;
    if (typeof IntersectionObserver === "undefined") return setMapNear(true);
    const io = new IntersectionObserver((es) => es.some((e) => e.isIntersecting) && setMapNear(true), { rootMargin: "100% 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [mapNear]);
  const { dragBox, handlers: areaHandlers } = useAreaDrag(() => mapRef.current, setArea, () => setAreaArmed(false));
  // a new station moves the grid map to it; a point, line or area chosen near the old one no longer applies
  const chooseStation = (id: string) => {
    const next = stations.find((x) => x.station_id === id);
    setCellAtPt(null);
    setLine([]);
    setArea(null);
    if (next?.lat != null && next.lon != null) mapRef.current?.fitBounds([[next.lon - 0.35, next.lat - 0.25], [next.lon + 0.35, next.lat + 0.25]], { padding: 16, duration: 500 });
    setStation(id);
  };
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
  // ------------------------------------------------------------------ observation grids: satellite chlorophyll, HF radar
  const [source, setSource] = useState<ObsSource>("charm");
  const [sat, setSat] = useState<SatChoice>(() => ({ product: satelliteLatest(manifest, "multi") ? "multi" : "olci300", day: null }));
  const [radar, setRadar] = useState<RadarChoice>({ hour: null, mean: false });
  const hasSat = !!(satelliteLatest(manifest, "multi") ?? satelliteLatest(manifest, "olci300") ?? satelliteLatest(manifest, "viirs750"));
  const hasRadar = currentsHourly(manifest).length > 0;
  const obsLayer = source === "satellite" ? satLayerFor(manifest, sat) : source === "radar" ? radarLayerFor(manifest, radar) : null;
  // satellite grids are chunked: read only around the selection (with a margin for the coverage reference)
  const obsBox = useMemo<ObsBox | null>(() => {
    if (tool === "cell" && cellAtPt) return [cellAtPt.lon - 0.01, cellAtPt.lat - 0.01, cellAtPt.lon + 0.01, cellAtPt.lat + 0.01];
    if (tool === "transect" && line.length === 2) return [Math.min(line[0].lon, line[1].lon), Math.min(line[0].lat, line[1].lat), Math.max(line[0].lon, line[1].lon), Math.max(line[0].lat, line[1].lat)];
    if (tool === "area" && area) return [area[0] - 0.02, area[1] - 0.02, area[2] + 0.02, area[3] + 0.02];
    return null;
  }, [tool, cellAtPt, line, area]);
  // the published sources this page shows, with their freshness (the same rule as Explore): the model's
  // record and grid always, the satellite or radar when chosen
  useReportSource(modelEntry(manifest, now, "In the model chart and grid"));
  useReportSource(source === "satellite" && obsLayer ? satelliteEntry(manifest, now, "In the observation tools", obsLayer) : null);
  useReportSource(source === "radar" && obsLayer ? currentsEntry(manifest, now, "In the observation tools", obsLayer) : null);
  const obsRead = useObsSampler(manifest, baseUrl, obsLayer, obsBox);
  const obsS = obsRead.sampler;
  const obsPx = useMemo(() => (obsS && tool === "cell" && cellAtPt ? obsPoint(obsS, cellAtPt.lat, cellAtPt.lon) : null), [obsS, tool, cellAtPt]);
  const obsProfile = useMemo(() => (obsS && tool === "transect" && line.length === 2 ? obsTransect(obsS, line[0], line[1]) : null), [obsS, tool, line]);
  const obsCells = useMemo(() => (obsS && tool === "area" && area ? obsArea(obsS, area) : null), [obsS, tool, area]);
  const obsFootprint = useMemo<GeoJSON.Feature[]>(() => {
    if (!obsS) return [];
    if (obsPx?.value != null) {
      const fp = footprintOf(obsS, obsPx);
      return [{ type: "Feature", geometry: pixelPolygon(fp.g, fp.row, fp.col), properties: { kind: "pixel" } }];
    }
    const on = (obsProfile ?? obsCells ?? []).filter((c) => c.value != null && ("inDomain" in c ? c.inDomain : true));
    return on.length ? [{ type: "Feature", geometry: footprintOutline(obsS.g, on), properties: { kind: "footprint" } }] : [];
  }, [obsS, obsPx, obsProfile, obsCells]);
  const satRaster = useMemo(() => {
    const t = source === "satellite" ? obsLayer?.tiles : null;
    return obsLayer && t ? { id: `r-${obsLayer.layer_id}`, template: t.relative ? artifactUrl(baseUrl, t.url_template) : t.url_template, minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat, indexUrl: tileIndexUrl(baseUrl, t) } : null;
  }, [source, obsLayer, baseUrl]);
  const radarField = source === "radar" && obsS?.kind === "radar" ? obsS.field : null;
  const currentsOverlay = useMemo(() => (radarField && obsS ? { id: `r-${obsS.layer.layer_id}`, field: radarField, features: fieldFeatures(radarField), mode: "arrows" as const } : null), [radarField, obsS]);
  const pickSource = (x: ObsSource) => {
    setSource(x);
    setLine([]);
  };

  const highlight = useMemo<GeoJSON.FeatureCollection>(() => {
    const f: GeoJSON.Feature[] = [{ type: "Feature", geometry: { type: "Point", coordinates: [st.lon!, st.lat!] }, properties: { kind: "station", label: st.name } }];
    if (source === "charm" && tool === "cell" && cellRead?.status === "ready" && cellRead.polygon) f.push({ type: "Feature", geometry: cellRead.polygon, properties: { kind: "cell" } });
    if (source !== "charm" && tool === "cell" && cellAtPt) f.push({ type: "Feature", geometry: { type: "Point", coordinates: [cellAtPt.lon, cellAtPt.lat] }, properties: { kind: "point" } });
    if (source !== "charm") f.push(...obsFootprint);
    if (tool === "transect" && line.length === 2) f.push({ type: "Feature", geometry: { type: "LineString", coordinates: line.map((p) => [p.lon, p.lat]) }, properties: { kind: "transect" } });
    else if (tool === "transect" && line.length === 1) f.push({ type: "Feature", geometry: { type: "Point", coordinates: [line[0].lon, line[0].lat] }, properties: { kind: "point" } });
    const box = tool === "area" ? (dragBox ?? area) : null;
    if (box) f.push(boxFeature(box));
    return { type: "FeatureCollection", features: f };
  }, [st, cellRead, line, tool, dragBox, area, source, cellAtPt, obsFootprint]);
  const runLayers = CHARM_LEADS.map((l) => charmLayer(manifest, gridVar, l)).filter((l): l is NonNullable<typeof l> => !!l);
  const fileTag = `${gridVar}-${forecastLayer?.time.valid_date ?? "na"}`;
  const stationsIn = area ? stations.filter((s) => s.lat! >= area[1] && s.lat! <= area[3] && s.lon! >= area[0] && s.lon! <= area[2]) : [];


  // ------------------------------------------------------------------ v10 composition: the instrument first; stations beside it; the record's sections one step away
  const ladder = useMemo<LadderStation[]>(() => stations.map((s) => ({ id: s.station_id, name: s.name, lat: s.lat!, lon: s.lon!, last: lastHabWords(lastHabObs(s), now) })), [stations, now]);
  const ladderSel = ladder.find((l) => l.id === st.station_id) ?? ladder[0];
  const [tab, setTabState] = useState<TabId>("window");
  // an old "#r-tools" / "#r-stats" link (and a section's own hash) opens that section
  useEffect(() => {
    const h = window.location.hash.slice(1);
    const t = HASH_TAB[h];
    if (t) setTabState(t);
  }, []);
  const setTab = (t: TabId) => {
    setTabState(t);
    if (t === "grid") setMapNear(true);
  };
  const openSection = (t: TabId) => {
    setTab(t);
    requestAnimationFrame(() => document.getElementById("r-sections")?.scrollIntoView({ block: "start", behavior: "auto" }));
  };
  const [table, setTable] = useState(false);
  const exportWindow = () => download(`neritic-${st.station_id}-${variable}.csv`, stationCsv(observations, st, vmeta, from, to));
  const exportModel = () => download(`neritic-${st.station_id}-charm-${modelVar}.csv`, modelHistoryCsv(st, modelVar, charmLayer(manifest, modelVar, 0), manifest, observations.method.charm));
  const varShort = (id: ObsVariableId) => (id === "pn_seriata" || id === "pn_delicatissima" ? (vars.some((v) => v.id === "pn_seriata") && vars.some((v) => v.id === "pn_delicatissima") ? SHORT[id] : "Pseudo-nitzschia") : SHORT[id]);
  const sampledFrom = dayLabel(st.sample_times[0] ?? "", false, now);
  const sampledTo = dayLabel(st.sample_times[st.sample_times.length - 1] ?? "", false, now);
  const windowWords = range ? RANGES.find((r) => r.id === range)!.label.toLowerCase() : `${dayLabel(dayKey(from), false, now)} to ${dayLabel(dayKey(to), false, now)}`;
  const coastPlace = place?.key ?? (st.nearest_port_code != null ? `port-${st.nearest_port_code}` : pointKey(st.lat!, st.lon!));
  const coastName = place?.name ?? st.nearest_port_name ?? null;
  const nowcast = charmLayer(manifest, modelVar, 0);
  const locatorRaster = nowcast?.image ? { id: `loc-${nowcast.layer_id}`, url: artifactUrl(baseUrl, nowcast.image.url), corners: nowcast.image.corners_lnglat } : null;
  const radiusKm = st.charm?.radius_km ?? 15;
  const locatorLabel = `Ring: the ${radiusKm} km of model cells charted below${nowcast?.time.valid_date ? `, on the nowcast for ${dayLabel(nowcast.time.valid_date, false, now)}` : ""}.`;
  const pickStation = (id: string) => {
    chooseStation(id);
    setSheet(false);
  };
  const ladderH = Math.max(470, Math.min(580, stations.length * 28 + 20));

  return (
    <div className="env-chart min-h-dvh bg-n-bg text-n-ink" data-testid="research">
      <SkipLink />
      <TopBar active="research" now={now} notices={<NoticesButton official={official} status={sourceStatus(manifest, "official")} />} />
      <RegistryDisclosure className="mx-auto w-full max-w-[1520px] px-4 pb-2 sm:px-6 lg:px-8" fixture={isFixture(manifest)} />
      <div className="mx-auto grid w-full max-w-[1520px] gap-x-10 px-4 pb-32 pt-2 sm:px-6 lg:grid-cols-[248px_minmax(0,1fr)] lg:px-8 lg:pb-16">
        {/* the stations: the linked map (the station and its model neighbourhood) and the coast ladder (north to south, true latitude) */}
        <nav aria-label="Shore stations" className="hidden lg:sticky lg:top-3 lg:flex lg:max-h-[calc(100dvh-24px)] lg:flex-col lg:gap-5 lg:self-start lg:overflow-y-auto lg:[scrollbar-width:none]" data-testid="research-stations">
          {wide && <LocatorMap stations={ladder} selected={ladderSel} radiusKm={radiusKm} forecast={locatorRaster} onPick={pickStation} label={locatorLabel} height={180} />}
          <div>
            <p className="nx-caps mb-1.5 flex items-baseline justify-between pr-1">
              <span>Shore stations · {stations.length}</span>
              <span>last HAB sample</span>
            </p>
            <CoastLadder stations={ladder} selected={st.station_id} onSelect={pickStation} height={ladderH} />
                      </div>
        </nav>

        <main id="main" className="cw-fade-in flex min-w-0 flex-col">
          {/* the station, where and how long it was sampled, and the crosshair's two readings */}
          <header className="grid gap-x-10 gap-y-2 pb-2 sm:gap-y-4 sm:pb-6 xl:grid-cols-[minmax(0,1fr)_minmax(0,500px)] xl:items-end">
            <div className="min-w-0">
              {(arrived || backHref) && (
                <p className="mb-2 flex flex-wrap items-center gap-x-3 text-[13.5px] leading-snug text-n-ink-2" data-testid="research-arrived">
                  {arrived && (
                    <span>
                      Arrived from <span className="font-semibold text-n-ink">{arrived.name}</span> · nearest station {st.name}, {kmLabel(arrived.km)}
                    </span>
                  )}
                  {backHref && (
                    <Link href={backHref} className="nx-focus -my-2 inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent" data-testid="research-back">
                      <Icon name="back" className="h-4 w-4" /> Back{place ? ` to ${place.name}` : ""}
                    </Link>
                  )}
                </p>
              )}
              <p className="nx-caps !text-n-measured max-sm:hidden">Shore station · measured · CalHABMAP</p>
              <h1 className="nx-display text-[24px] leading-[28px] sm:mt-1.5 sm:text-[46px] sm:leading-[50px]" data-testid="research-title">
                {/* the name is the station picker. Below 1024 px (no ladder beside it) it is drawn as a control of its
                    own, a bordered field with its chevron that opens the sheet; on a wide screen it focuses the ladder */}
                <button
                  type="button"
                  onClick={() => (wide ? document.querySelector<HTMLElement>(`[data-testid="ladder-${st.station_id}"]`)?.focus() : setSheet(true))}
                  aria-haspopup={wide ? undefined : "dialog"}
                  aria-describedby="r-station-hint"
                  className="nx-focus inline-flex max-w-full items-center gap-2 text-left max-lg:min-h-12 max-lg:w-full max-lg:justify-between max-lg:rounded-[6px] max-lg:border max-lg:border-n-line-strong max-lg:bg-n-surface max-lg:py-1.5 max-lg:pl-3 max-lg:pr-2 lg:-mx-1 lg:cursor-default lg:rounded-[4px] lg:px-1"
                  data-testid="research-station-button"
                >
                  <span className="min-w-0">{st.name}</span>
                  <span aria-hidden className="flex shrink-0 items-center gap-1 lg:hidden" data-testid="research-station-picker">
                    <Icon name="chevronDown" className="h-5 w-5 text-n-ink-2" />
                  </span>
                </button>
              </h1>
              <span id="r-station-hint" hidden>
                Choose another shore station ({stations.length} stations)
              </span>
              <p className="mt-0.5 truncate font-data text-[11.5px] leading-[18px] text-n-ink-3 sm:mt-2 sm:whitespace-normal sm:text-[12px]">
                <span className="max-sm:hidden">
                  {st.lat!.toFixed(3)}°N {Math.abs(st.lon!).toFixed(3)}°W
                </span>
                {st.nearest_port_name && <span className="max-sm:hidden"> · {kmLabel(st.nearest_port_km ?? 0)} from {st.nearest_port_name}</span>}
                <span className="max-sm:hidden">
                  {" "}
                  · sampled {sampledFrom} – {sampledTo}
                </span>
                <span className="sm:hidden">{(st.sample_times[0] ?? "").slice(0, 4)}–{sampledTo}</span> · {allMeasured.length.toLocaleString("en-US")} {AXIS_NAME[variable]} values · {allZeros.toLocaleString("en-US")} reported 0
                {st.status !== "updated" ? ` · last update ${st.status.replace("_", " ")}` : ""}
              </p>
            </div>
            <Readings reading={reading} unit={unit} fmt={fmt} quantity={vmeta.label} variable={hasHistory ? modelVar : null} radiusKm={radiusKm} pinned={pinnedT != null && reading != null && reading.day === dayKey(pinnedT)} />
          </header>

          {/* the instrument: quantity and window, the two linked panels, the record as a brush */}
          <section id="r-record" className="flex scroll-mt-4 flex-col" data-testid="research-chart">
            <div className="flex flex-wrap items-end justify-between gap-x-6 border-b border-n-line">
              <VariableTabs value={variable} onChange={chooseVariable} options={vars.map((v) => ({ id: v.id, label: varShort(v.id), unit: unitLabel(v.units), title: v.label }))} />
              <div className="flex min-w-0 flex-wrap items-center gap-x-1 gap-y-1 py-0.5 sm:py-1.5 max-sm:w-full max-sm:flex-nowrap max-sm:justify-between">
                <RangePresets value={range} onChange={chooseRange} />
                <span aria-hidden className="mx-2 h-5 w-px bg-n-line max-sm:hidden" />
                <div className="flex items-center">
                  <button type="button" onClick={() => setTable((o) => !o)} aria-expanded={table} aria-controls="r-table" className={TOOL} data-testid="research-table-toggle">
                    Table
                  </button>
                  <ExportMenu>
                    {(close) => (
                      <>
                        <button
                          type="button"
                          className={MENU_ITEM}
                          onClick={() => {
                            exportWindow();
                            close();
                          }}
                          data-testid="research-export"
                        >
                          <span className="text-[14px] font-semibold">This window (CSV)</span>
                          <span className="nx-note-t">
                            {vmeta.label}, {win.length} visits, with its provenance header
                          </span>
                        </button>
                        {hasHistory && (
                          <button
                            type="button"
                            className={MENU_ITEM}
                            onClick={() => {
                              exportModel();
                              close();
                            }}
                            data-testid="export-model"
                          >
                            <span className="text-[14px] font-semibold">Model history (CSV)</span>
                            <span className="nx-note-t">C-HARM nowcast near the station, every day kept</span>
                          </button>
                        )}
                        {csv && (
                          <a href={csv} target="_blank" rel="noreferrer" className={MENU_ITEM} onClick={close} data-testid="research-original">
                            <span className="inline-flex items-center gap-1 text-[14px] font-semibold">
                              Original data (SCCOOS) <Icon name="external" className="h-3.5 w-3.5" />
                            </span>
                            <span className="nx-note-t">The station&rsquo;s own file, as published</span>
                          </a>
                        )}
                        <button
                          type="button"
                          className={`${MENU_ITEM} mt-1 !rounded-t-none border-t border-n-line`}
                          onClick={() => {
                            close();
                            openSection("exports");
                          }}
                        >
                          <span className="text-[13.5px] font-semibold text-n-accent">What each file holds · grid exports</span>
                        </button>
                      </>
                    )}
                  </ExportMenu>
                  <button type="button" onClick={() => openSection("lineage")} className={`${TOOL} max-sm:hidden`} data-testid="research-lineage-open">
                    Lineage
                  </button>
                  <button type="button" onClick={() => openSection("grid")} className={`${TOOL} lg:hidden`} data-testid="research-jump-r-tools">
                    <Icon name="cell" className="h-4 w-4" />
                    <span className="max-sm:sr-only">Grid tools</span>
                  </button>
                </div>
              </div>
            </div>

            <div className="pt-2 sm:pt-5">
              <LinkedSeries
                key={`${st.station_id}|${variable}`}
                samples={ss}
                variable={variable}
                units={vmeta.units}
                label={vmeta.label}
                axisTitle={`${wide ? "Measured at the pier · " : ""}${AXIS_NAME[variable]}, ${unit}${variable === "temp" ? "" : " · log scale"}`}
                from={from}
                to={to}
                refLine={variable === "pDA" ? PDA_REF : variable === "pn_seriata" || variable === "pn_delicatissima" ? PN_REF : undefined}
                model={linkedModel}
                modelNote={
                  showModel ? (
                    <div className="nx-rule pt-3 text-[14px] text-n-ink-2" data-testid="model-panel-none">
                      <p>No C-HARM nowcast history is published for this station.</p>
                      {st.charm?.error && (
                        <details className="mt-1 text-[12.5px] text-n-ink-3">
                          <summary className="nx-focus min-h-11 cursor-pointer py-2 font-semibold text-n-accent">What the pipeline reported</summary>
                          <p className="break-all">{st.charm.error}</p>
                        </details>
                      )}
                    </div>
                  ) : (
                    <p className="nx-margin nx-rule pt-3" data-testid="model-panel-absent">
                      C-HARM models domoic acid and <i>Pseudo-nitzschia</i> only: there is no model panel for {vmeta.label.toLowerCase()}.
                    </p>
                  )
                }
                cursorT={cursorT}
                pinnedT={pinnedT}
                onHover={onHover}
                onPin={onPin}
                source={`CalHABMAP · ${st.name} · via SCCOOS`}
                obsHeight={wide ? 264 : 180}
                modelHeight={wide ? 100 : 84}
                compact={!wide}
              />
              <Legend samples={win} hasModel={!!linkedModel} partial={!!model?.points.some((p) => p.partial)} />
            </div>

            <div className="mt-4 flex flex-col gap-1" data-testid="research-window">
              <p className="flex flex-wrap items-baseline justify-between gap-x-4">
                <span className="nx-caps">
                  Window · <span className="text-n-ink-2">{dayLabel(dayKey(from), false, now)} – {dayLabel(dayKey(to), false, now)}</span> · {real.length} samples
                </span>
                <span className="nx-margin text-[12.5px] max-sm:hidden">Drag across the record to set the window; drag the window to move it.</span>
              </p>
              <BrushStrip samples={ss} variable={variable} x0={x0} x1={x1} from={from} to={to} modelStart={modelAll?.historyStart ?? null} modelEnd={modelEnd} onBrush={(a, b) => setPreview([a, b])} onCommit={commitWindow} />
            </div>

            <div id="r-table" hidden={!table} className="mt-5" data-testid="research-table">
              <SamplesTable samples={win} unit={unit} model={linkedModel} />
            </div>

            <div className="nx-rule mt-6 grid gap-x-8 gap-y-3 pt-4 md:grid-cols-3" data-testid="research-notes">
              <p className="nx-margin">
                <b>Two panels, never one axis.</b> A pier sample and a model probability are different quantities: they are drawn on the same dates and never compared as numbers.
              </p>
              <p className="nx-margin">
                <b>Toxin in seawater is not toxin in seafood.</b> One sample, one pier, one day. Lines join samples less than 21 days apart; a reported 0 is not quantified, not absent.
              </p>
              <p className="nx-margin">
                <b>Sources.</b> CalHABMAP via SCCOOS ERDDAP{st.retrieved_at ? `, retrieved ${dayLabel(st.retrieved_at, false, now)}` : ""}
                {modelAll?.historyStart != null ? ` · NOAA C-HARM v3.1 nowcast, kept daily since ${dayLabel(dayKey(modelAll.historyStart), false, now)}` : ""} · fixed axes, never rescaled to the data.{" "}
                <Link href={`/?sel=station:${encodeURIComponent(st.station_id)}`} className="nx-focus whitespace-nowrap font-sans text-[13px] not-italic text-n-accent" data-testid="research-explore">
                  Open in Explore
                </Link>
                {" · "}
                <Link href={coastHref({ place: coastPlace })} className="nx-focus whitespace-nowrap font-sans text-[13px] not-italic text-n-accent" data-testid="research-coast">
                  My Coast{coastName ? `: ${coastName}` : ""}
                </Link>
              </p>
            </div>
          </section>

          {/* the record's sections: one tap away, every panel mounted (its wording stays in the page) */}
          <div id="r-sections" className="mt-8 scroll-mt-4">
            <Tabs idBase="r" tabs={TABS} value={tab} onChange={setTab} />

            <TabPanel idBase="r" id="window" active={tab === "window"}>
              <div id="r-stats" data-testid="research-stats">
                <h2 className="sr-only">In this window</h2>
                <dl className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-4">
                  <Stat label="Samples with a value" value={measured.length} />
                  <Stat label="Reported as zero" value={zeros} />
                  <Stat label="Median" value={median == null ? "—" : median.toLocaleString("en-US", { maximumSignificantDigits: 3 })} unit={unit} />
                  <Stat label="Highest" value={max?.value == null ? "—" : max.value.toLocaleString("en-US", { maximumSignificantDigits: 3 })} unit={unit} note={max ? dayLabel(max.time) : undefined} />
                </dl>
                <p className="nx-note-t mt-4 max-w-[72ch]">
                  {vmeta.label}, {windowWords} window. Of {win.length} visits in the window; a visit without this measurement is not counted and is never a zero.
                  {rejectedN ? ` ${rejectedN} rejected by QC (no value, not counted).` : ""}
                  {flaggedN ? ` ${flaggedN} flagged high by QC (kept as published, counted).` : ""}
                </p>
              </div>
            </TabPanel>

            <TabPanel idBase="r" id="provenance" active={tab === "provenance"}>
              <div className="grid gap-8 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]" data-testid="research-provenance">
                <div className="min-w-0">
                  <h2 className="nx-caps mb-3">How {vmeta.label.toLowerCase()} is measured</h2>
                  <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2 text-[14px] leading-snug">
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
                    <dd className="nx-data text-[13px]" data-testid="research-obs-run">
                      {observations.provenance.pipeline_run_id} · version {observations.provenance.pipeline_version}
                    </dd>
                  </dl>
                </div>
                <div className="min-w-0">
                  <h2 className="nx-caps mb-3">
                    Station checks · {st.qc.filter((q) => q.passed).length} of {st.qc.length} passed
                  </h2>
                  <ul className="flex flex-col gap-1.5 text-[13.5px] leading-snug" data-testid="research-qc">
                    {st.qc.map((q) => (
                      <li key={q.name} className="flex gap-2">
                        <span className={`mt-px shrink-0 ${q.passed ? "text-n-ok" : "text-n-down"}`}>
                          <Icon name={q.passed ? "check" : "alert"} className="h-4 w-4" />
                        </span>
                        <span>
                          <b className="font-semibold">{q.name}</b> · <span className="text-n-ink-2">{q.detail}</span>
                        </span>
                      </li>
                    ))}
                  </ul>
                  <details className="nx-note mt-4 border-t border-n-line text-[13.5px] text-n-ink-2">
                    <summary className="nx-focus flex min-h-11 cursor-pointer items-center justify-between py-2 font-semibold text-n-ink">
                      Citation and licence <Icon name="chevronDown" className="nx-note-chev h-4 w-4 text-n-ink-3" />
                    </summary>
                    {observations.provenance.citation && <p className="mb-1">{observations.provenance.citation}</p>}
                    <p>{observations.provenance.license}</p>
                  </details>
                </div>
              </div>
            </TabPanel>

            <TabPanel idBase="r" id="grid" active={tab === "grid"}>
              <section id="r-tools" className="flex scroll-mt-4 flex-col gap-4" data-testid="research-grid">
                <div className="flex flex-col gap-1">
                  <h2 className="nx-display text-[22px] leading-[28px]" data-testid="grid-heading">
                    {source === "charm"
                      ? `Native C-HARM grid · ${VARIABLE_WORDS[gridVar].short} · ${forecastLayer?.time.valid_date ? `${lead === 0 ? "nowcast" : "forecast"} for ${dayLabel(forecastLayer.time.valid_date, true)}` : "—"}`
                      : `${source === "satellite" ? "Satellite chlorophyll" : "Surface currents"} · ${obsSensor(obsLayer)} · ${obsWhen(obsLayer)}`}
                  </h2>
                  <p className="nx-note-t">The published grid near {st.name}, cell by cell: read one, draw a transect, or count an area.</p>
                </div>
                <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
                  <Field label="Data source">
                    <Seg
                      scroll
                      label="Data source"
                      value={source}
                      onChange={pickSource}
                      options={(
                        [
                          ["charm", "C-HARM model", true],
                          ["satellite", "Satellite chlorophyll", hasSat],
                          ["radar", "Surface currents (HF radar)", hasRadar],
                        ] as const
                      )
                        .filter(([, , ok]) => ok)
                        .map(([id, label]) => ({ id, label, testId: `gsrc-${id}` }))}
                    />
                  </Field>
                  <Field label="Tool">
                    <Seg
                      scroll
                      label="Map tool"
                      value={tool}
                      onChange={(t) => {
                        setMapNear(true);
                        setTool(t);
                        setLine([]);
                        if (t === "area") setAreaArmed(true);
                      }}
                      options={(["cell", "transect", "area"] as const).map((t) => ({ id: t, testId: `tool-${t}`, label: t === "cell" ? (source === "satellite" ? "Inspect a pixel" : "Inspect a cell") : t === "transect" ? "Draw a transect" : "Select an area" }))}
                    />
                  </Field>
                </div>
                {source !== "charm" && (
                  <>
                    <div className="flex flex-wrap items-end gap-x-5 gap-y-3" data-testid="obs-controls">
                      <ObsControls manifest={manifest} source={source} sat={sat} onSat={setSat} radar={radar} onRadar={setRadar} now={now} />
                    </div>
                    <ObsNote source={source} />
                  </>
                )}
                <div className={`flex flex-wrap items-end gap-x-6 gap-y-3 ${source === "charm" ? "" : "hidden"}`} data-testid="grid-controls">
                  <Field label="C-HARM variable">
                    <Seg scroll label="C-HARM variable" value={gridVar} onChange={setGridVar} options={CHARM_VARIABLES.map((v) => ({ id: v, label: VARIABLE_WORDS[v].short, testId: `gvar-${v}` }))} />
                  </Field>
                  <Field label={`Day of this run (issued ${run ? `${dayLabel(run.issued_date)}${run.issued_date_derived ? ", inferred" : ""}` : "—"})`}>
                    <Seg
                      scroll
                      label="Day of the run"
                      value={lead}
                      onChange={setLead}
                      options={leads.map((l) => {
                        const d = charmLayer(manifest, gridVar, l)?.time.valid_date;
                        return {
                          id: l,
                          testId: `lead-${l}`,
                          label: (
                            <span className="flex flex-col items-center py-0.5 leading-tight">
                              <span className="text-[13.5px]">{d ? dayLabel(d, true) : "—"}</span>
                              <span className="text-[11.5px] font-medium">{l === 0 ? "nowcast" : `forecast +${l}`}</span>
                            </span>
                          ),
                        };
                      })}
                    />
                  </Field>
                  <label className="flex min-h-11 cursor-pointer items-center gap-2 text-[14px] font-semibold text-n-ink-2">
                    <input type="checkbox" checked={showEdges} onChange={(e) => setShowEdges(e.target.checked)} className="nx-focus h-4 w-4 accent-[var(--nx-accent)]" data-testid="grid-edges" />
                    Cell edges
                  </label>
                </div>
                {source === "charm" && <p className="nx-note-t">Only the days this run published; nothing beyond its horizon. Each value is model output for one 3 km cell, not a toxin measurement.</p>}
                <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
                  <div className="flex min-w-0 flex-col gap-2">
                    <div ref={mapBox} className="env-instrument relative h-[380px] overflow-hidden rounded-xl" data-testid="research-map" data-mounted={mapNear}>
                      {!mapNear && (
                        <p className="absolute inset-0 grid place-items-center text-[13.5px] text-n-ink-2" data-testid="research-map-waiting">
                          Map of the grid near {st.name}
                        </p>
                      )}
                      {mapNear && (
                        <MapCanvas
                          forecast={source === "charm" && forecastLayer?.image ? { id: `r-${forecastLayer.layer_id}`, url: artifactUrl(baseUrl, forecastLayer.image.url), corners: forecastLayer.image.corners_lnglat, grid: forecastLayer.grid ? { layer: forecastLayer, url: artifactUrl(baseUrl, forecastLayer.grid.url) } : null } : null}
                          satellite={satRaster}
                          satelliteAge={null}
                          currents={currentsOverlay}
                          imagery={null}
                          ports={ports}
                          showPorts
                          officialGeometry={labelledGeometry(official)}
                          showOfficial
                          selectedPort={null}
                          inspect={null}
                          chrome="none"
                          highlight={highlight}
                          cellEdges={source === "charm" && showEdges ? edges : null}
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
                      )}
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
                          <p className="text-[14px] text-n-ink-2">Click two points on the map. The profile lists every published {source === "satellite" ? "pixel" : "cell"} the line crosses; {source === "charm" ? "cells without a value stay gaps" : "those without an observation stay gaps, never joined"}.</p>
                        ) : source !== "charm" ? (
                          <ObsTransect manifest={manifest} read={obsRead} cells={obsProfile} a={line[0]} b={line[1]} />
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
                      source !== "charm" &&
                      (cellAtPt ? (
                        <ObsPointPanel manifest={manifest} baseUrl={baseUrl} read={obsRead} px={obsPx} point={cellAtPt} sat={sat} now={now} />
                      ) : (
                        <p className="text-[14px] text-n-ink-2">{source === "satellite" ? "Click the water to read one published pixel: its own observation date, sensor, native size and footprint. An empty pixel is no observation, never low chlorophyll." : "Click the water to read one published 2 km radar cell: speed and direction at this hour, and every published hour at the same cell. An empty cell is no observation, never calm water."}</p>
                      ))}
                    {tool === "cell" &&
                      source === "charm" &&
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
                          {source !== "charm" ? (
                            <ObsArea manifest={manifest} baseUrl={baseUrl} read={obsRead} cells={obsCells} bbox={area} now={now} />
                          ) : (
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
                          )}
                          <section className="flex flex-col gap-1 border-t border-n-line pt-3">
                            <h3 className="nx-caps">Shore stations in the box</h3>
                            {stationsIn.length ? (
                              <ul className="text-[14px]">
                                {stationsIn.map((s) => (
                                  <li key={s.station_id}>
                                    <button type="button" onClick={() => chooseStation(s.station_id)} className="nx-focus min-h-11 text-left font-semibold text-n-accent underline-offset-2 hover:underline">
                                      {s.name}
                                    </button>
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <p className="text-[14px] text-n-ink-2">No CalHABMAP station in this box: no sample is not a clean result.</p>
                            )}
                          </section>
                          <button type="button" onClick={() => setAreaArmed(true)} className={`${QUIET} self-start`} data-testid="area-redraw">
                            Draw another box
                          </button>
                        </div>
                      ) : (
                        <p className="text-[14px] text-n-ink-2">
                          {source === "charm"
                            ? "Drag a box across the water. It counts the cells with and without a published value and gives the spread of the published probabilities; gaps are counted, never filled."
                            : `Drag a box across the water. It counts the ${source === "satellite" ? "pixels" : "radar cells"} with and without an observation, their dates${source === "satellite" ? " and sensors" : ""}, and the spread of the observed values; gaps are counted, never filled.`}
                        </p>
                      ))}
                  </div>
                </div>
              </section>
            </TabPanel>

            <TabPanel idBase="r" id="lineage" active={tab === "lineage"}>
              {source === "charm" ? forecastLayer && <Lineage layer={forecastLayer} manifest={manifest} /> : obsLayer && <Lineage layer={obsLayer} manifest={manifest} />}
            </TabPanel>

            <TabPanel idBase="r" id="exports" active={tab === "exports"}>
              <div className="flex max-w-[78ch] flex-col">
                <p className="text-[14.5px] leading-6 text-n-ink-2">Every file starts with lines beginning with #: its source, the pipeline run, the licence and what kind of data it is. An empty value is never a zero.</p>
                <ul className="mt-4 flex flex-col border-t border-n-line">
                  <ExportRow title="This window (CSV)" what={`${vmeta.label} at ${st.name}: every visit in the ${windowWords} window, with its QC qualifier.`} action={<button type="button" onClick={exportWindow} className={QUIET}><Icon name="download" className="h-4 w-4" /> Download</button>} />
                  {hasHistory && <ExportRow title="Model history (CSV)" what="The C-HARM nowcast near the station, every day Neritic keeps: model probability, never a toxin measurement." action={<button type="button" onClick={exportModel} className={QUIET}><Icon name="download" className="h-4 w-4" /> Download</button>} />}
                  {csv && <ExportRow title="Original data (SCCOOS)" what="The station's own file, as the data provider publishes it." action={<a href={csv} target="_blank" rel="noreferrer" className={QUIET}>Open <Icon name="external" className="h-4 w-4" /></a>} />}
                  <ExportRow
                    title="A cell, a transect or an area (CSV)"
                    what="From the native grid: choose a tool, make a selection, and its export appears beside the result."
                    action={
                      <button type="button" onClick={() => openSection("grid")} className={QUIET}>
                        Native grid
                      </button>
                    }
                  />
                </ul>
              </div>
            </TabPanel>
          </div>
        </main>
      </div>
      <div className="env-instrument fixed inset-x-0 bottom-0 z-40 border-t border-n-line bg-n-surface lg:hidden" style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
        <NavTabs active="research" size="dock" />
      </div>
      <StationSheet open={sheet && !wide} onClose={closeSheet}>
        <div className="flex flex-col gap-4">
          <LocatorMap stations={ladder} selected={ladderSel} radiusKm={radiusKm} forecast={locatorRaster} onPick={pickStation} label={locatorLabel} height={180} />
          <div>
            <p className="nx-caps mb-1.5 flex items-baseline justify-between pr-1">
              <span>{stations.length} stations</span>
              <span>last HAB sample</span>
            </p>
            <CoastLadder stations={ladder} selected={st.station_id} onSelect={pickStation} height={Math.max(560, stations.length * 36 + 20)} />
          </div>
        </div>
      </StationSheet>
    </div>
  );
}

/** A quiet tool in the chart's bar (Table, Lineage, Grid): text, no outline; the Export menu is the same size. */
const TOOL = "nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 rounded-[4px] px-2 sm:px-2.5 text-[13.5px] font-medium text-n-ink-2 hover:text-n-ink aria-expanded:text-n-ink";

type TabId = "window" | "provenance" | "grid" | "lineage" | "exports";
const TABS: { id: TabId; label: string }[] = [
  { id: "window", label: "In this window" },
  { id: "provenance", label: "Provenance & QC" },
  { id: "grid", label: "Native grid" },
  { id: "lineage", label: "Lineage & citation" },
  { id: "exports", label: "Exports" },
];
const HASH_TAB: Record<string, TabId> = { "r-stats": "window", "r-tools": "grid", "r-provenance": "provenance", "r-lineage": "lineage", "r-exports": "exports" };

/** Short names for the quantity tabs (the full name is its title and the readout's). */
const SHORT: Record<ObsVariableId, string> = {
  pDA: "Domoic acid",
  dDA: "dDA",
  tDA: "tDA",
  pn_seriata: "P.-n. seriata",
  pn_delicatissima: "P.-n. delicatissima",
  chl_extracted: "Chlorophyll",
  temp: "Temperature",
};

/** ≥ 1024 px: the desktop instrument (after hydration; the server draws the phone). */
function useWide() {
  const [wide, setWide] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    const on = () => setWide(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return wide;
}

/** The window presets (90 d, 1 y, 3 y, All); a brushed window is "custom" until a preset is chosen. */
function RangePresets({ value, onChange }: { value: RangeId | null; onChange: (r: RangeId) => void }) {
  return (
    <div role="radiogroup" aria-label="Time range" className="flex items-center" data-testid="range-presets">
      {RANGES.map((r) => {
        const on = r.id === value;
        return (
          <button
            key={r.id}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => onChange(r.id)}
            className={`nx-focus relative min-h-11 whitespace-nowrap rounded-[4px] px-1.5 sm:px-2 font-data text-[12.5px] ${on ? "text-n-ink" : "text-n-ink-3 hover:text-n-ink"}`}
            data-testid={`range-${r.id}`}
          >
            {r.short}
            {on && <span aria-hidden className="absolute inset-x-2 bottom-2 h-px bg-n-ink" />}
          </button>
        );
      })}
      {value == null && (
        <span className="relative px-2 font-data text-[12.5px] text-n-ink" data-testid="range-custom">
          custom
          <span aria-hidden className="absolute inset-x-2 -bottom-[11px] h-px bg-n-ink" />
        </span>
      )}
    </div>
  );
}

/** What the marks mean, only the ones in view: a legend row under the chart. */
function Legend({ samples, hasModel, partial }: { samples: { value: number | null; q: string | null }[]; hasModel: boolean; partial: boolean }) {
  const zeros = samples.some((s) => s.q === "reported_zero" || s.value === 0);
  const rejected = samples.filter((s) => s.q === "rejected_negative").length;
  const flagged = samples.filter((s) => s.q === "flag_high" && s.value != null).length;
  const bare = samples.some((s) => s.value == null && s.q !== "rejected_negative");
  const item = "inline-flex items-center gap-1.5 whitespace-nowrap";
  return (
    <p className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[12px] text-n-ink-3" data-testid="series-legend">
      <span className={item}>
        <svg width="18" height="8" aria-hidden>
          <line x1="1" x2="17" y1="4" y2="4" stroke="var(--nx-measured)" strokeWidth="1.6" />
          <circle cx="9" cy="4" r="2.4" fill="var(--nx-measured)" />
        </svg>
        sample (joined when under 21 days apart)
      </span>
      {zeros && (
        <span className={item}>
          <svg width="8" height="8" aria-hidden>
            <rect x="1" y="5" width="6" height="3" fill="var(--nx-ink-3)" />
          </svg>
          <span className="whitespace-normal">on the floor: reported 0 (not quantified; the line breaks there)</span>
        </span>
      )}
      {flagged > 0 && (
        <span className={item}>
          <svg width="14" height="14" aria-hidden>
            <circle cx="7" cy="7" r="5.5" fill="none" stroke="var(--nx-down)" strokeWidth="1.3" />
          </svg>
          flagged high by QC, kept as published ({flagged})
        </span>
      )}
      {rejected > 0 && (
        <span className={item}>
          <svg width="10" height="10" aria-hidden>
            <path d="M1.5,1.5 l7,7 m0,-7 l-7,7" stroke="var(--nx-down)" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          rejected (QC), not a measurement ({rejected})
        </span>
      )}
      <span className={item}>
        <svg width="10" height="10" aria-hidden>
          <line x1="2" x2="2" y1="2" y2="8" stroke="var(--nx-measured)" strokeOpacity="0.6" />
          <line x1="7" x2="7" y1="2" y2="8" stroke="var(--nx-gap)" />
        </svg>
        rug: every visit{bare ? " (grey: this quantity not measured)" : ""}
      </span>
      {hasModel && partial && (
        <span className={item}>
          <svg width="10" height="10" aria-hidden>
            <circle cx="5" cy="5" r="2.8" fill="none" stroke="var(--nx-model)" strokeWidth="1.1" />
          </svg>
          model day with fewer than 60% of the usual cells (not joined)
        </span>
      )}
    </p>
  );
}

function Stat({ label, value, unit, note }: { label: string; value: React.ReactNode; unit?: string; note?: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <dt className="nx-caps">{label}</dt>
      <dd className="nx-value !text-[26px]">
        {value}
        {unit && <small>{unit}</small>}
        {note && <span className="mt-1 block font-sans text-[12.5px] font-normal tracking-normal text-n-ink-3">{note}</span>}
      </dd>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1 max-sm:w-full">
      <span className="nx-caps">{label}</span>
      {children}
    </div>
  );
}

function ExportRow({ title, what, action }: { title: string; what: string; action: React.ReactNode }) {
  return (
    <li className="flex flex-col gap-2 border-b border-n-line py-3.5 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
      <div className="min-w-0">
        <p className="text-[15px] font-semibold">{title}</p>
        <p className="text-[13.5px] leading-snug text-n-ink-2">{what}</p>
      </div>
      <div className="shrink-0">{action}</div>
    </li>
  );
}
