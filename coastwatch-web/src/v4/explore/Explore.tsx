"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import type { Manifest, PortsCollection, SourceStatus } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { StationLite } from "@/lib/data";
import { CHARM_VARIABLES, artifactUrl, charmRun, satelliteLatest, isFixture, sourceStatus, type CharmVariable } from "@/lib/layers";
import { loadField, fieldFeatures, type CurrentField } from "@/lib/currents";
import { cellEdges, cellLabels, gapAt, loadGrid } from "@/lib/grid";
import { gapFeatures, lastHabSample, stationFeatures } from "@/lib/provenance";
import { clipLimitsToWater, labelledGeometry } from "@/lib/officialGeometry";
import { loadWaterFallback, type WaterGeometry } from "@/lib/waterFallback";
import { journeyTo } from "@/lib/camera";
import { forecastFreshness, openingLayer, type Opening } from "@/lib/opening";
import { classifyTime, type Freshness } from "@/lib/freshness";
import { defaultLead } from "@/lib/charmDays";
import { pacificToday } from "@/lib/time";
import { useNow } from "@/lib/useNow";
import type { CurrentsOverlay, MapPoint } from "@/components/map/MapCanvas";
import { MobileSheet, type SnapId } from "@/components/MobileSheet";
import { HoverReadout, type HoverSource } from "@/components/map/HoverReadout";
import { useReportSource, type SourceState } from "@/v4/readiness/store";
import { SourceList } from "@/v4/readiness/StatusCenter";
import { TopBar, NavTabs } from "@/v4/shell/TopBar";
import { NoticesButton, type LocalNotices } from "@/v4/shell/Notices";
import { RegistryDisclosure } from "@/v4/shell/Disclosure";
import { associationsAt } from "@/lib/applicability";
import { SkipLink } from "@/v4/shell/SkipLink";
import { useCountyPrefs } from "@/components/map/Counties";
import { Icon } from "@/v4/ui/Icon";
import { dayLabel, kmLabel, lastHabWords } from "@/v4/ui/time";
import { decodeSelection, encodeSelection, sameSelection, type Camera, type HistoryEntry, type Selection } from "@/v4/state/selection";
import { choiceParams, parseChoice, resolve, runDays, SAT_WORDS, VARIABLE_WORDS, type LayerChoice } from "./layers";
import { LayerChips, LayerList, LayerOptions, Legend, Provenance, ReferenceLines } from "./LayerPanel";
import { Timebar, timeSteps } from "./Timebar";
import { PlaceSearch, type SearchHit } from "./Search";
import { AreaModel, CellBody, HarborView, Header, PointView, StationView } from "./Inspect";
import { boxFeature, useAreaDrag } from "./AreaDrag";
import { useCellRead } from "./reads";
import { PACKS } from "@/v4/offline/packs";
import { useOfflinePack } from "@/v4/offline/useOfflinePack";
import { OfflinePanel, useRecordSource } from "@/v4/offline/OfflinePanel";
import { OfflineBanner } from "@/v4/offline/OfflineBanner";
import { readHomePort } from "@/v4/state/homePort";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), { ssr: false, loading: () => null });

type Props = {
  manifest: Manifest;
  ports: PortsCollection | null;
  official: OfficialDataset | null;
  portIntel: PortIntelCollection | null;
  stations: StationLite[] | null;
  stationsError: string | null;
  baseUrl: string;
  initialParams: Record<string, string>;
};

const STATEWIDE = { id: "california", label: "All California", bounds: [[-125.3, 32.45], [-117.1, 42.05]] as [[number, number], [number, number]] };
const VIEW: Record<string, [[number, number], [number, number]]> = {
  north_coast: [[-124.72, 39.95], [-123.6, 42.02]],
  mendocino_sonoma: [[-124.05, 38.2], [-122.85, 39.6]],
  sf_bay_farallones: [[-123.15, 37.4], [-122.3, 38.15]],
  monterey_bay: [[-122.38, 36.48], [-121.66, 37.13]],
  central_coast: [[-121.45, 34.75], [-120.4, 35.85]],
  southern_california: [[-120.55, 32.55], [-117.05, 34.55]],
};
const PANEL = 404; // inspection panel width (desktop)
const DOCK = 64; // phone navigation dock height
const CREDITS = 24; // phone: the one-line map credits above the sheet
const PEEK = 196; // phone sheet: the timebar or the selection's heading
const LAYERS = 340; // layer deck width (desktop)

function useBreakpoint(): "phone" | "tablet" | "desktop" | null {
  const [bp, setBp] = useState<"phone" | "tablet" | "desktop" | null>(null);
  useEffect(() => {
    const f = () => setBp(window.innerWidth >= 1100 ? "desktop" : window.innerWidth >= 700 ? "tablet" : "phone");
    f();
    window.addEventListener("resize", f);
    return () => window.removeEventListener("resize", f);
  }, []);
  return bp;
}

/** The newest sample date (YYYY-MM-DD) among a station's latest values, of any variable. */
function newestSample(latest: { date: string }[] | undefined): string | null {
  let best: string | null = null;
  for (const l of latest ?? []) if (l.date && (!best || l.date.slice(0, 10) > best)) best = l.date.slice(0, 10);
  return best;
}

export function Explore({ manifest, ports, official, portIntel, stations, stationsError, baseUrl, initialParams }: Props) {
  const now = useNow();
  const bp = useBreakpoint();
  // offline with a verified pack: the map opens on the saved basemap (decided before it mounts)
  const pack = PACKS.monterey;
  const off = useOfflinePack(pack.id);
  useRecordSource(off.record, pack.name, off.online, off.evicted, off.problem);
  const [basemapPm, setBasemapPm] = useState<string | null | undefined>(undefined);
  useEffect(() => {
    if (basemapPm !== undefined || off.record === undefined) return;
    setBasemapPm(!navigator.onLine && off.record?.complete ? `/offline/${pack.id}/basemap.pmtiles` : null);
  }, [off.record, basemapPm, pack.id]);
  const [packOpen, setPackOpen] = useState(false);
  // the address as the server saw it; read again in the browser before the map mounts (below)
  const [q0, setQ0] = useState(() => new URLSearchParams(initialParams));
  const run = charmRun(manifest);
  const days = runDays(manifest, now);
  const firstLead = now ? defaultLead(days, pacificToday(now)) : (run?.leads_available.includes(1) ? 1 : (run?.leads_available[0] ?? 0));

  // ------------------------------------------------------------------ layer (URL: layer, var, lead)
  const linked = useMemo(() => parseChoice(q0), [q0]);
  // a forecast link without a day waits for the clock, then opens on the day valid today (lib/charmDays)
  const [choice, setChoiceRaw] = useState<LayerChoice | null>(linked);
  // no layer in the link: chosen once, in the browser, from the data's freshness (lib/opening)
  // when nothing is current, the reason is said on the map's panel, never left as a bare seafloor
  const [opening, setOpening] = useState<Opening | null>(null);
  useEffect(() => {
    if (choice || !now) return;
    const o = openingLayer(manifest, now, { regionId: q0.get("region"), variable: "particulate_domoic", lead: firstLead });
    setOpening(o);
    setChoiceRaw(o.group === "satellite" ? { kind: "satellite", product: o.product, day: null } : o.group === "forecast" ? { kind: "forecast", variable: "particulate_domoic", lead: firstLead } : { kind: "none" });
  }, [choice, now, manifest, q0, firstLead]);
  useEffect(() => {
    if (choice?.kind === "forecast" && choice.lead < 0 && now) setChoiceRaw({ ...choice, lead: firstLead });
  }, [choice, firstLead, now]);
  const active: LayerChoice = useMemo(() => choice ?? { kind: "none" }, [choice]);
  const resolved = useMemo(() => resolve(manifest, active, baseUrl), [manifest, active, baseUrl]);
  const variable: CharmVariable = active.kind === "forecast" ? active.variable : (CHARM_VARIABLES as readonly string[]).includes(q0.get("var") ?? "") ? (q0.get("var") as CharmVariable) : "particulate_domoic";
  const lead = active.kind === "forecast" ? active.lead : firstLead;

  // ------------------------------------------------------------------ selection + history
  const [selection, setSelectionRaw] = useState<Selection | null>(() => decodeSelection(q0));
  const [stack, setStack] = useState<HistoryEntry[]>([]);
  const mapRef = useRef<MlMap | null>(null);
  const camera = (): Camera | null => {
    const m = mapRef.current;
    if (!m) return null;
    const c = m.getCenter();
    return { center: [c.lng, c.lat], zoom: m.getZoom() };
  };
  const select = useCallback((s: Selection | null, opts: { push?: boolean } = {}) => {
    setSelectionRaw((prev) => {
      if (sameSelection(prev, s)) return prev;
      if (opts.push !== false) setStack((st) => [...st.slice(-20), { selection: prev, camera: camera() }]);
      return s;
    });
  }, []);
  const back = useCallback(() => {
    setStack((st) => {
      const last = st[st.length - 1];
      if (!last) return st;
      setSelectionRaw(last.selection);
      if (last.camera && mapRef.current) mapRef.current.easeTo({ center: last.camera.center, zoom: last.camera.zoom, duration: 380 });
      return st.slice(0, -1);
    });
  }, []);

  // focus: a place picked in search moves focus to its panel; closing a panel returns focus to what
  // opened it (the search box, the map, a button), never to the top of the page
  const panelRef = useRef<HTMLDivElement | null>(null);
  const opener = useRef<HTMLElement | null>(null);
  const focusPanel = useRef(false);
  const prevSel = useRef<Selection | null>(selection);
  useEffect(() => {
    const was = prevSel.current;
    prevSel.current = selection;
    if (!was && selection) {
      const a = document.activeElement as HTMLElement | null;
      opener.current = a && a !== document.body ? a : null;
    } else if (was && !selection) {
      const o = opener.current;
      opener.current = null;
      const a = document.activeElement;
      // only when focus was lost with the panel (or is on nothing): never pulled from where the reader went
      // (to the search field's frame, not the field itself, so its list does not open again)
      const to = o?.closest<HTMLElement>("#place-search") ?? o;
      if (to?.isConnected && (!a || a === document.body || !a.isConnected)) to.focus({ preventScroll: true });
    }
    if (selection && focusPanel.current) {
      focusPanel.current = false;
      requestAnimationFrame(() => panelRef.current?.focus({ preventScroll: true }));
    }
  }, [selection]);

  // URL: layer and selection; the camera is written when the view settles (replaceState, not history)
  const [region, setRegion] = useState<string>(q0.get("region") ?? "california");
  // In the browser, the address is the truth: offline, the service worker answers with the saved
  // page (rendered for the pack's harbour), whatever was asked for. And with a home port kept on
  // this device, a bare Explore opens on its region. Decided once, before the map mounts.
  const [synced, setSynced] = useState(false);
  useEffect(() => {
    const live = new URLSearchParams(window.location.search);
    if (!decodeSelection(live) && !live.has("region") && !live.has("at")) {
      const code = readHomePort();
      const r = code != null ? ports?.features.find((f) => f.properties.port_code === code)?.properties.region : null;
      if (r) live.set("region", r);
    }
    if (live.toString() !== q0.toString()) {
      setQ0(live);
      setChoiceRaw(parseChoice(live));
      setSelectionRaw(decodeSelection(live));
      setRegion(live.get("region") ?? "california");
    }
    setSynced(true);
    // once, on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const writeUrl = useCallback(() => {
    if (!synced) return;
    const q = new URLSearchParams();
    if (choice) for (const [k, v] of Object.entries(choiceParams(choice))) q.set(k, v);
    if (q0.get("from")?.startsWith("/coast")) q.set("from", q0.get("from")!);
    const sel = encodeSelection(selection);
    if (sel) q.set("sel", sel);
    const m = mapRef.current;
    if (m) {
      const c = m.getCenter();
      q.set("at", `${c.lat.toFixed(4)},${c.lng.toFixed(4)},${m.getZoom().toFixed(2)}`);
    } else if (q0.get("at")) q.set("at", q0.get("at")!);
    else if (region !== "california") q.set("region", region);
    window.history.replaceState(window.history.state, "", `?${q.toString()}`);
  }, [choice, selection, region, q0, synced]);
  useEffect(writeUrl, [writeUrl]);

  // ------------------------------------------------------------------ map data
  const [edges, setEdges] = useState<{ id: string; fc: GeoJSON.FeatureCollection; codes: Uint16Array } | null>(null);
  useEffect(() => {
    let live = true;
    const l = resolved.forecastLayer;
    if (!l?.grid || edges?.id === l.layer_id) return;
    loadGrid(artifactUrl(baseUrl, l.grid.url))
      .then((codes) => live && setEdges({ id: l.layer_id, fc: cellEdges(l.grid!, codes), codes }))
      .catch(() => live && setEdges(null));
    return () => {
      live = false;
    };
  }, [resolved.forecastLayer, baseUrl, edges?.id]);
  const [mapReady, setMapReady] = useState(false);
  const [zoom, setZoom] = useState<number | null>(null);
  const [harbour, setHarbour] = useState<{ id: string; labels: GeoJSON.FeatureCollection; gap: GeoJSON.FeatureCollection | null } | null>(null);
  useEffect(() => {
    const m = mapRef.current;
    const l = resolved.forecastLayer;
    if (!m || !l?.grid || !edges || edges.id !== l.layer_id) return setHarbour(null);
    const update = () => {
      if (m.getZoom() < 10.5) return setHarbour((h) => (h ? null : h));
      const b = m.getBounds();
      const c = m.getCenter();
      setHarbour({ id: l.layer_id, labels: cellLabels(l.grid!, edges.codes, [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]), gap: gapFeatures({ lat: c.lat, lon: c.lng }, gapAt(l.grid!, edges.codes, c.lat, c.lng)) });
    };
    update();
    m.on("moveend", update);
    return () => {
      m.off("moveend", update);
    };
  }, [edges, resolved.forecastLayer, mapReady]);
  const harbourOn = active.kind === "forecast" && harbour != null && harbour.id === resolved.forecastLayer?.layer_id;

  const [curField, setCurField] = useState<{ id: string; field: CurrentField } | null>(null);
  const [curState, setCurState] = useState<"idle" | "loading" | "ready" | "failed">("idle");
  // "Try again" for the currents grids (a failed grid is never kept in lib/grid's cache)
  const [curRetry, setCurRetry] = useState(0);
  const curLayer = resolved.curLayer;
  useEffect(() => {
    if (!curLayer) return setCurState("idle");
    let live = true;
    setCurState("loading");
    loadField(baseUrl, curLayer)
      .then((f) => {
        if (!live) return;
        setCurField({ id: curLayer.layer_id, field: f });
        setCurState("ready");
      })
      .catch(() => live && setCurState("failed"));
    return () => {
      live = false;
    };
  }, [curLayer, baseUrl, curRetry]);
  const currents: CurrentsOverlay | null = useMemo(
    () => (curLayer && curField?.id === curLayer.layer_id ? { id: curField.id, field: curField.field, features: fieldFeatures(curField.field), mode: "arrows" } : null),
    [curLayer, curField],
  );

  const stationPoints = useMemo(() => (stations?.length ? stationFeatures(stations, now, now ? pacificToday(now) : null) : null), [stations, now]);
  const [water, setWater] = useState<WaterGeometry | null>(null);
  useEffect(() => {
    let live = true;
    loadWaterFallback().then((g) => live && g && setWater(g));
    return () => {
      live = false;
    };
  }, []);
  const officialGeometry = useMemo(() => {
    const fc = labelledGeometry(official);
    return fc && water ? clipLimitsToWater(fc, water) : fc;
  }, [official, water]);

  // ------------------------------------------------------------------ the selected thing on the map
  const cellAtSel = selection?.kind === "cell" ? { lat: selection.lat, lon: selection.lon } : null;
  const cellRead = useCellRead(manifest, baseUrl, ports, cellAtSel, variable, lead);
  const station = selection?.kind === "station" ? (stations?.find((s) => s.station_id === selection.id) ?? null) : null;
  // the station's newest sample of any variable (its HAB sample may be older: the panel lists each)
  const stationLast = useMemo(() => newestSample(station?.latest), [station]);
  const harborF = selection?.kind === "harbor" ? (ports?.features.find((f) => f.properties.port_code === selection.port) ?? null) : null;
  // a harbour selected: the notices control leads with the notices covering its waters
  const harbourNotices = useMemo<LocalNotices | null>(() => {
    if (!harborF || !official) return null;
    const [lon, lat] = harborF.geometry.coordinates as number[];
    const a = associationsAt(official, `port-${harborF.properties.port_code}`, lat, lon);
    return { name: harborF.properties.display_name ?? harborF.properties.name, covering: a.filter((x) => x.tier === "geographic").map((x) => x.record.id), unsettled: a.filter((x) => x.tier === "unresolved").map((x) => x.record.id) };
  }, [harborF, official]);
  const highlight = useMemo<GeoJSON.FeatureCollection | null>(() => {
    const feats: GeoJSON.Feature[] = [];
    if (selection?.kind === "cell" && cellRead?.status === "ready" && cellRead.polygon) feats.push({ type: "Feature", geometry: cellRead.polygon, properties: { kind: "cell" } });
    if (selection?.kind === "cell") feats.push({ type: "Feature", geometry: { type: "Point", coordinates: [selection.lon, selection.lat] }, properties: { kind: "point" } });
    if (station) feats.push({ type: "Feature", geometry: { type: "Point", coordinates: [station.lon, station.lat] }, properties: { kind: "station", label: station.name } });
    if (harborF) feats.push({ type: "Feature", geometry: harborF.geometry as unknown as GeoJSON.Point, properties: { kind: "harbor", label: harborF.properties.display_name ?? harborF.properties.name } });
    if (selection?.kind === "point") feats.push({ type: "Feature", geometry: { type: "Point", coordinates: [selection.lon, selection.lat] }, properties: { kind: "point" } });
    if (selection?.kind === "area") {
      const [w, s, e, n] = selection.bbox;
      feats.push({ type: "Feature", geometry: { type: "Polygon", coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] }, properties: { kind: "area" } });
    }
    return feats.length ? { type: "FeatureCollection", features: feats } : null;
  }, [selection, cellRead, station, harborF]);

  // fly to a selection made outside the map (search, a claim's "Show on map", a link)
  const panelOpen = selection != null;
  // desktop: while a selection is open the layer deck folds to a capsule (reopened on demand), so
  // the map between the panel and the left edge stays open; a new selection folds it again
  const [deckOpen, setDeckOpen] = useState(false);
  useEffect(() => setDeckOpen(false), [selection]);
  const deckCompact = bp === "desktop" && panelOpen && !deckOpen;
  // `open`: frame for the panel being open (a camera move made for a new selection, in the same
  // step that selects it, before panelOpen has caught up)
  const padding = useCallback((open: boolean = panelOpen) => {
    if (bp === "desktop") return { top: 84, bottom: 150, left: open && !deckOpen ? 24 : LAYERS + 28, right: (open ? PANEL + 24 : 0) + 24 };
    if (bp === "tablet") return { top: 84, bottom: 200, left: 24, right: (open ? 380 : 0) + 24 };
    // phone: a selection opens the sheet to half height, so the camera centres it in the map strip
    // that stays visible above the sheet (not behind it)
    const sheet = open ? Math.round((window.innerHeight - DOCK) * 0.5) : PEEK;
    return { top: 128, bottom: sheet + DOCK + 24, left: 20, right: 20 };
  }, [bp, panelOpen, deckOpen]);
  // the whole state: on a wide window the map cannot zoom out past its pannable area, so the fit
  // keeps the north coast clear of the top bar and lets the far south run under the timebar
  const fitState = useCallback(() => {
    const m = mapRef.current;
    if (!m) return;
    const p = padding();
    m.fitBounds(STATEWIDE.bounds, { padding: bp === "phone" ? p : { ...p, top: 112, bottom: 24 }, duration: 500 });
  }, [padding, bp]);
  const focusOn = useCallback(
    (s: Selection) => {
      const m = mapRef.current;
      if (!m) return;
      const pt = s.kind === "station" ? stations?.find((x) => x.station_id === s.id) : null;
      const pf = s.kind === "harbor" ? ports?.features.find((f) => f.properties.port_code === s.port) : null;
      const at: [number, number] | null = pt ? [pt.lon, pt.lat] : pf ? (pf.geometry.coordinates as [number, number]) : s.kind === "cell" || s.kind === "point" ? [s.lon, s.lat] : null;
      if (s.kind === "area") return m.fitBounds([[s.bbox[0], s.bbox[1]], [s.bbox[2], s.bbox[3]]], { padding: padding(true), duration: 500 });
      if (at) journeyTo(m, { center: at, zoom: Math.max(m.getZoom(), s.kind === "harbor" ? 10.6 : 10.2), padding: padding(true) });
    },
    [stations, ports, padding],
  );
  const selectAndGo = useCallback(
    (s: Selection) => {
      select(s);
      focusOn(s);
    },
    [select, focusOn],
  );

  // ------------------------------------------------------------------ area tool
  const [areaMode, setAreaMode] = useState(false);
  const { dragBox, handlers: areaHandlers } = useAreaDrag(() => mapRef.current, (b) => select({ kind: "area", bbox: b }), () => setAreaMode(false));
  const shownHighlight = useMemo<GeoJSON.FeatureCollection | null>(() => {
    if (!dragBox) return highlight;
    return { type: "FeatureCollection", features: [boxFeature(dragBox)] };
  }, [dragBox, highlight]);

  // county reference lines: the remembered v2 choice, now with its control back in the layer panel
  const [counties, setCounties] = useCountyPrefs();

  // ------------------------------------------------------------------ readiness
  const [mapStatus, setMapStatus] = useState<{ basemap: "ok" | "slow" | "down"; field: "none" | "loading" | "ready" | "failed" } | null>(null);
  const onStatus = useCallback((s: { basemap: "ok" | "slow" | "down"; field: "none" | "loading" | "ready" | "failed" }) => setMapStatus(s), []);
  const reload = useCallback(() => window.location.reload(), []);
  // "Try again" for a data field whose tiles failed: the map asks for its sources afresh
  const [fieldRetry, setFieldRetry] = useState(0);
  const retryField = useCallback(() => setFieldRetry((n) => n + 1), []);
  const retryCurrents = useCallback(() => setCurRetry((n) => n + 1), []);
  useReportSource({
    id: "map",
    label: "Coastline and map",
    state: !mapReady ? "loading" : mapStatus?.basemap === "down" ? "partial" : mapStatus?.basemap === "slow" ? "partial" : "ready",
    detail: !mapReady ? "Loading the coastline, seafloor relief and map style" : mapStatus?.basemap === "ok" ? "Seafloor relief, coastline and harbours" : "Basemap tiles are slow or unreachable: the bundled coastline is drawn instead",
    actions: mapStatus?.basemap === "down" ? [{ label: "Reload the map", run: reload }] : undefined,
  });
  useReportSource({ id: "data", label: "Neritic dataset", state: "ready", detail: `Published ${manifest.generated_at.slice(0, 16).replace("T", " ")} UTC`, asOf: manifest.generated_at });
  const officialStatus = sourceStatus(manifest, "official");
  useReportSource({
    id: "notices",
    label: "Official notices",
    state: official ? "ready" : "unavailable",
    detail: official ? "Neritic's copy; not verified by a person. The agencies' pages are the authority." : "Neritic's copy could not be loaded; check CDFW and CDPH directly",
    asOf: officialStatus?.last_success_at ?? null,
  });
  // a field that loaded is "ready" only while it is current: stale and historical data, and a failed
  // update, are said in the status centre too (the same policy as the layer panel)
  const fieldState = (fresh: Freshness | null, status: SourceStatus | null): { state: SourceState; note: string | null } => {
    if (fresh?.state === "historical") return { state: "stale", note: `Out of date (${fresh.ageDays} days): does not describe current conditions` };
    if (fresh?.state === "stale") return { state: "stale", note: `Older than usual (${fresh.ageDays} days)` };
    if (status && status.outcome !== "updated") return { state: "partial", note: "The last update did not complete; the newest published data is shown" };
    return { state: "ready", note: null };
  };
  const modelStatus = sourceStatus(manifest, "charm");
  const modelFresh = now ? forecastFreshness(manifest, now) : null;
  const modelField = fieldState(modelFresh, modelStatus);
  useReportSource(
    active.kind === "forecast"
      ? {
          id: "model",
          label: "NOAA C-HARM forecast",
          state: mapStatus?.field === "failed" ? "unavailable" : mapStatus?.field === "ready" ? modelField.state : "loading",
          detail: [run ? `${VARIABLE_WORDS[active.variable].short}, run issued ${run.issued_date}` : "No run published", modelField.note].filter(Boolean).join(". "),
          asOf: run?.issued_date ?? null,
          actions: mapStatus?.field === "failed" ? [{ label: "Try again", run: retryField }] : undefined,
        }
      : null,
  );
  const obsFresh = (id: string, l: typeof resolved.satLayer) => {
    const st = sourceStatus(manifest, id);
    // each layer carries its own freshness policy; a latest-clear view is as fresh as its newest pixel
    return { st, fresh: now && l ? classifyTime(l.freshness, l.composite ? { ...l.time, observed_date: l.composite.newest_observed_date } : l.time, now) : null };
  };
  const satF = obsFresh("satellite_chl", resolved.satLayer);
  const satField = fieldState(satF.fresh, satF.st);
  useReportSource(
    active.kind === "satellite" || (active.kind === "currents" && active.chl)
      ? { id: "satellite", label: "Satellite chlorophyll", state: mapStatus?.field === "failed" ? "unavailable" : mapStatus?.field === "ready" ? satField.state : "loading", detail: [resolved.satLayer ? `${resolved.satLayer.layer_id.split("_")[0].toUpperCase()} tiles` : "No layer", mapStatus?.field === "failed" ? "Some tiles could not be loaded" : satField.note].filter(Boolean).join(". "), asOf: resolved.satLayer?.time.observed_date ?? null, actions: mapStatus?.field === "failed" ? [{ label: "Try again", run: retryField }] : undefined }
      : null,
  );
  const curF = obsFresh("hf_radar", curLayer);
  const curReady = fieldState(curF.fresh, curF.st);
  useReportSource(
    active.kind === "currents"
      ? { id: "currents", label: "Surface currents (HF radar)", state: curState === "failed" ? "unavailable" : curState === "ready" ? curReady.state : "loading", detail: ["Hourly, 2 km cells", curState === "failed" ? "This hour could not be loaded" : curReady.note].filter(Boolean).join(". "), asOf: curLayer?.time.valid_time ?? null, actions: curState === "failed" ? [{ label: "Try again", run: retryCurrents }] : undefined }
      : null,
  );
  const placeName = harborF?.properties.display_name ?? station?.name ?? null;
  useReportSource(stationsError && !stations ? { id: "conditions", label: "Shore stations", state: "unavailable", detail: stationsError } : null);

  // ------------------------------------------------------------------ hover readout (fine pointers): the exact published value
  // The pointer's position lives in a small child (HoverLayer), not here: a pointer move re-renders
  // only the readout, never the layer deck, the timebar or the selected panel.
  const hoverSink = useRef<((h: HoverAt | null) => void) | null>(null);
  const hoverRaf = useRef<number | null>(null);
  const hoverNext = useRef<HoverAt | null>(null);
  const [canHover, setCanHover] = useState(false);
  useEffect(() => setCanHover(window.matchMedia("(hover: hover) and (pointer: fine)").matches), []);
  const onHover = useCallback((pt: MapPoint | null, scr: { x: number; y: number } | null) => {
    hoverNext.current = pt && scr ? { p: pt, s: scr } : null;
    if (hoverRaf.current == null)
      hoverRaf.current = requestAnimationFrame(() => {
        hoverRaf.current = null;
        hoverSink.current?.(hoverNext.current);
      });
  }, []);
  const hoverSource: HoverSource = useMemo(() => {
    if (active.kind === "forecast" && resolved.forecastLayer) return { kind: "forecast", layer: resolved.forecastLayer };
    if (active.kind === "satellite" && resolved.satLayer) return { kind: "satellite", layer: resolved.satLayer };
    if (active.kind === "currents" && curField && curLayer && curField.id === curLayer.layer_id) return { kind: "currents", layer: curLayer, field: curField.field };
    return null;
  }, [active.kind, resolved.forecastLayer, resolved.satLayer, curField, curLayer]);

  // ------------------------------------------------------------------ keyboard
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (e.key === "Escape") {
        if (areaMode) setAreaMode(false);
        else if (selection) select(null);
        // the area tool's letter works only with the map focused: typed elsewhere, "a" is just a letter
      } else if (e.key === "a" && !e.metaKey && !e.ctrlKey && !e.altKey && t?.closest('[data-testid="map"]')) setAreaMode((v) => !v);
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [areaMode, selection, select]);

  // ------------------------------------------------------------------ pieces
  const regions = useMemo(() => [STATEWIDE, ...(ports?.regions ?? []).map((r) => ({ id: r.id, label: r.label, bounds: (VIEW[r.id] ?? r.bounds) as [[number, number], [number, number]] }))], [ports]);
  const onSearch = (h: SearchHit) => {
    if (h.kind !== "region") focusPanel.current = true;
    if (h.kind === "harbor") selectAndGo({ kind: "harbor", port: Number(h.id) });
    else if (h.kind === "station") selectAndGo({ kind: "station", id: h.id });
    else {
      const r = regions.find((x) => x.id === h.id);
      setRegion(h.id);
      select(null);
      if (r && mapRef.current) journeyTo(mapRef.current, { bounds: r.bounds, padding: padding() });
    }
  };
  // what a point can be read for: the forecast for the chosen variable and day, the satellite view on the map
  // (or the newest clear view), and the currents hour once its field has loaded
  const pointReads = useMemo(
    () => ({
      forecast: resolve(manifest, { kind: "forecast", variable, lead }, baseUrl).forecastLayer ?? null,
      satellite: (active.kind === "satellite" ? resolved.satLayer : null) ?? satelliteLatest(manifest, "olci300") ?? satelliteLatest(manifest, "viirs750") ?? null,
      currents: hoverSource?.kind === "currents" ? { layer: hoverSource.layer, field: hoverSource.field } : null,
    }),
    [manifest, variable, lead, baseUrl, active.kind, resolved.satLayer, hoverSource],
  );
  const ctx = { manifest, baseUrl, ports, portIntel, stations, now, variable, lead, onSelect: selectAndGo, official, reads: pointReads };
  // opened from a briefing ("Show on map"): a way back to it, only to a My Coast address
  const backTo = useMemo(() => {
    const f = q0.get("from");
    if (!f || !f.startsWith("/coast")) return null;
    const pk = new URLSearchParams(f.split("?")[1] ?? "").get("place");
    const name = pk?.startsWith("port-") ? (ports?.features.find((x) => `port-${x.properties.port_code}` === pk)?.properties.display_name ?? "harbour") : "place";
    return { href: f, name };
  }, [q0, ports]);
  const initialAt = q0.get("at")?.split(",").map(Number);
  const initialBounds = (regions.find((x) => x.id === (q0.get("region") ?? "")) ?? STATEWIDE).bounds;

  // offline with a saved pack: a place outside it is said to be outside, with the way to the saved one
  const selAt: [number, number] | null = !selection
    ? null
    : selection.kind === "harbor"
      ? harborF
        ? [(harborF.geometry.coordinates as number[])[0], (harborF.geometry.coordinates as number[])[1]]
        : null
      : selection.kind === "station"
        ? station
          ? [station.lon, station.lat]
          : null
        : selection.kind === "area"
          ? [(selection.bbox[0] + selection.bbox[2]) / 2, (selection.bbox[1] + selection.bbox[3]) / 2]
          : [selection.lon, selection.lat];
  const outsidePack = !off.online && !!off.record?.complete && selAt != null && !(selAt[0] >= pack.bbox[0] && selAt[0] <= pack.bbox[2] && selAt[1] >= pack.bbox[1] && selAt[1] <= pack.bbox[3]);
  const packNote = outsidePack ? (
    <p className="rounded-xl border border-n-line-strong bg-n-surface-2 p-3 text-[14px] leading-snug text-n-ink-2" data-testid="outside-pack">
      <span className="font-semibold text-n-ink">Offline, and this place is outside the saved {pack.name} pack.</span> What follows may be missing or older than it looks.{" "}
      <button type="button" onClick={() => selectAndGo({ kind: "harbor", port: pack.port })} className="nx-focus inline-flex min-h-11 items-center font-semibold text-n-accent underline underline-offset-2" data-testid="outside-pack-go">
        Go to {pack.name}
      </button>
    </p>
  ) : null;
  // on a phone the briefing is a link on the title row, so the evidence starts higher in the sheet
  const briefingLink =
    selection?.kind === "harbor" && bp === "phone" ? (
      <a href={`/coast?place=port-${selection.port}`} className="nx-press nx-focus mt-1 inline-flex min-h-11 shrink-0 items-center gap-0.5 rounded-full border border-n-line-strong pl-3 pr-1.5 text-[14px] font-semibold text-n-ink" data-testid="open-briefing-link">
        Briefing <Icon name="chevronRight" className="h-4 w-4 shrink-0" />
      </a>
    ) : undefined;
  const inspection = selection ? (
    <div ref={panelRef} tabIndex={-1} className="flex flex-col gap-4 outline-none" data-testid="inspect-panel" data-kind={selection.kind}>
      {packNote}
      {selection.kind === "harbor" && (
        <>
          <Header icon="harbor" kind="Harbour" title={placeName ?? "Harbour"} sub={harborF?.properties.county ? `${harborF.properties.county} County · CDFW port` : undefined} onClose={() => select(null)} onBack={back} canBack={stack.length > 0} action={briefingLink} />
          <HarborView code={selection.port} ctx={ctx} />
        </>
      )}
      {selection.kind === "station" && (
        <>
          <Header icon="station" kind="Shore station" title={station?.name ?? selection.id} sub={`CalHABMAP · SCCOOS${stationLast ? ` · last sampled ${dayLabel(stationLast, false, now)}` : ""}`} onClose={() => select(null)} onBack={back} canBack={stack.length > 0} />
          <StationView id={selection.id} ctx={ctx} />
        </>
      )}
      {selection.kind === "cell" && (
        <>
          <Header icon="cell" kind="C-HARM model cell" title={VARIABLE_WORDS[variable].name} sub={cellRead?.status === "ready" && cellRead.harbor ? `${kmLabel(cellRead.harbor.km)} from ${cellRead.harbor.name}` : undefined} onClose={() => select(null)} onBack={back} canBack={stack.length > 0} />
          <CellBody read={cellRead} ctx={ctx} variable={variable} />
        </>
      )}
      {selection.kind === "point" && (
        <>
          <Header icon="locate" kind="A point in the water" title="This point" onClose={() => select(null)} onBack={back} canBack={stack.length > 0} />
          <PointView lat={selection.lat} lon={selection.lon} ctx={ctx} />
        </>
      )}
      {selection.kind === "area" && (
        <>
          <Header icon="area" kind="Selected area" title="What covers this area" onClose={() => select(null)} onBack={back} canBack={stack.length > 0} />
          <AreaView bbox={selection.bbox} ctx={ctx} />
        </>
      )}
    </div>
  ) : null;

  const hasTime = active.kind !== "none" && !!timeSteps(manifest, active, now)?.steps.length;
  const nothingCurrent =
    active.kind === "none" && opening?.reason === "nothing-current" ? (
      <div className="flex flex-col gap-2 rounded-xl border border-n-stale/50 bg-n-surface-2 p-3 text-[13.5px] leading-snug text-n-ink-2" data-testid="nothing-current">
        <p className="font-semibold text-n-ink">No current forecast or recent satellite view</p>
        <p>
          {run ? `The newest C-HARM forecast was issued ${dayLabel(run.issued_date)}. ` : "No C-HARM forecast could be loaded. "}
          {opening.satellite?.layer.time.observed_date ? `The newest satellite view is from ${dayLabel(opening.satellite.layer.composite?.newest_observed_date ?? opening.satellite.layer.time.observed_date)}. ` : "No recent satellite view was published. "}
          The map shows the coast and seafloor only. This does not mean conditions are normal.
        </p>
        {run && (
          <button type="button" onClick={() => setChoiceRaw({ kind: "forecast", variable: "particulate_domoic", lead: firstLead < 0 ? 0 : firstLead })} className="nx-press nx-focus inline-flex min-h-11 w-fit items-center rounded-lg border border-n-line-strong px-3 font-semibold text-n-ink" data-testid="show-latest-forecast">
            Show the latest forecast, marked out of date
          </button>
        )}
      </div>
    ) : null;
  // a data layer at state scale: its 3 km cells are a few pixels wide, so say how to read one
  const readHint =
    (active.kind === "forecast" || active.kind === "satellite") && !selection && zoom != null && zoom < 7 ? (
      <p className="flex flex-wrap items-center gap-x-2 text-[13.5px] text-n-ink-2" data-testid="read-hint">
        Zoom in and tap the water to read a {active.kind === "forecast" ? "model cell" : "pixel"}.
        <button
          type="button"
          onClick={() => mapRef.current && journeyTo(mapRef.current, { bounds: (VIEW.monterey_bay ?? [[-122.3, 36.5], [-121.7, 37.05]]) as [[number, number], [number, number]], padding: padding() })}
          className="nx-focus inline-flex min-h-11 items-center font-semibold text-n-accent underline underline-offset-2"
          data-testid="zoom-monterey"
        >
          Go to Monterey Bay
        </button>
      </p>
    ) : null;
  const [snap, setSnap] = useState<SnapId>("peek");
  useEffect(() => setSnap(selection ? "half" : "peek"), [selection]);
  // on a phone, choosing a layer from the peeking sheet opens it to half: its time and options are then in view
  const pickFromSheet = (c: LayerChoice) => {
    setChoiceRaw(c);
    if (bp === "phone") setSnap((x) => (x === "peek" ? "half" : x));
  };
  // phone: the layer chips as one compact row (44 px targets), so the sheet's first lines carry content
  const compactChips = "[&_[role=radio]]:min-h-11 [&_[role=radio]]:rounded-full [&_[role=radio]]:px-3 [&_[role=radio]]:text-[14px]";
  const activeCard = (
    <div className="flex flex-col gap-3" data-testid="active-layer">
      <div className={compactChips}>
        <LayerChips manifest={manifest} choice={active} onChoice={pickFromSheet} defaults={{ lead: firstLead }} />
      </div>
      {nothingCurrent}
      {readHint}
      <p className="nx-eyebrow">On the map</p>
      {/* the time control first, so it stays within reach at the sheet's peek; then what it shows */}
      {hasTime && <Timebar manifest={manifest} choice={active} onChoice={setChoiceRaw} now={now} compact />}
      <Provenance manifest={manifest} choice={active} resolved={resolved} now={now} regions={ports?.regions ?? []} />
      <Legend choice={active} resolved={resolved} />
    </div>
  );
  const layerDeck = (
    <div className="flex flex-col gap-4">
      {nothingCurrent}
      <LayerList manifest={manifest} choice={active} onChoice={setChoiceRaw} now={now} defaults={{ lead: firstLead }} />
      <LayerOptions manifest={manifest} choice={active} onChoice={setChoiceRaw} />
      {readHint}
      {/* what the map shows and when, before how to read its colours: the stamps stay above the fold */}
      <Provenance manifest={manifest} choice={active} resolved={resolved} now={now} regions={ports?.regions ?? []} />
      <div className="border-t border-n-line pt-3">
        <Legend choice={active} resolved={resolved} />
      </div>
      <div className="border-t border-n-line pt-3">
        <ReferenceLines prefs={counties} onChange={setCounties} />
      </div>
    </div>
  );

  // the folded deck (desktop, with a selection open): what the map shows, in one line, and the way back to the deck
  const steps = hasTime ? timeSteps(manifest, active, now) : null;
  const step = steps?.steps.find((x) => x.key === steps.current);
  const capsuleWords = [
    active.kind === "forecast" ? "C-HARM forecast" : active.kind === "satellite" ? "Satellite chlorophyll" : active.kind === "currents" ? "Surface currents" : "Seafloor and coast",
    active.kind === "forecast" ? VARIABLE_WORDS[active.variable].short : active.kind === "satellite" ? SAT_WORDS[active.product]?.name : null,
    step ? [step.top, step.label].filter(Boolean).join(" ") : null,
  ].filter(Boolean);
  const capsule = (
    <div className="absolute left-4 top-[76px] z-20 flex max-w-[480px] flex-col items-start gap-1.5">
      <button
        type="button"
        aria-expanded={false}
        aria-label={`Map layers: ${capsuleWords.join(", ")}. Show the layers and legend`}
        onClick={() => setDeckOpen(true)}
        className="nx-press nx-focus flex min-h-11 max-w-full items-center gap-2 rounded-full border border-n-line bg-n-surface pl-3 pr-2.5 text-[14px] font-semibold text-n-ink shadow-[var(--nx-shadow)]"
        data-testid="layer-capsule"
      >
        <Icon name="layers" className="h-[18px] w-[18px] shrink-0 text-n-accent" />
        <span className="truncate">{capsuleWords.join(" · ")}</span>
        <Icon name="chevronDown" className="h-4 w-4 shrink-0 text-n-ink-3" />
      </button>
      {/* the deck carried the credits: folded, they stay on the map */}
      <MapCredits />
    </div>
  );

  // a data layer that could not be drawn: said on the map, with a way to try again (never a silent gap)
  const satFailed = mapStatus?.field === "failed" && (active.kind === "satellite" || (active.kind === "currents" && !!active.chl));
  const fieldProblem =
    mapStatus?.field === "failed" || (active.kind === "currents" && curState === "failed") ? (
      <div role="status" className="flex w-full max-w-[420px] items-center gap-2.5 rounded-2xl border border-n-line-strong bg-n-surface py-1 pl-3.5 pr-1.5 text-n-ink shadow-[var(--nx-shadow)]" data-testid="field-failed">
        <Icon name="alert" className="h-5 w-5 shrink-0 text-n-down" />
        <span className="min-w-0 flex-1 text-[14px] leading-snug">
          {active.kind === "currents" && curState === "failed" ? "These currents could not be loaded." : satFailed ? "Part of the satellite view could not be loaded." : "Part of the C-HARM forecast could not be loaded."} Nothing is drawn in its place.
        </span>
        <button type="button" onClick={active.kind === "currents" && curState === "failed" ? retryCurrents : retryField} className="nx-press nx-focus inline-flex min-h-11 shrink-0 items-center rounded-xl px-3 text-[14px] font-semibold text-n-accent hover:bg-n-surface-2" data-testid="field-retry">
          Try again
        </button>
      </div>
    ) : null;

  // the map's tools, in two attached stacks: the zoom pair, then recentre, area, share and the field pack
  const stackCls = "pointer-events-auto flex flex-col divide-y divide-n-line rounded-xl border border-n-line bg-n-surface shadow-[var(--nx-shadow)]";
  const toolCls = "nx-press nx-focus grid h-12 w-12 place-items-center text-n-ink first:rounded-t-xl last:rounded-b-xl max-md:h-14 max-md:w-14";
  const recenter = (
    <button type="button" aria-label={selection ? "Centre on the selection" : "Show all California"} onClick={() => (selection ? focusOn(selection) : fitState())} className={toolCls} data-testid="recenter">
      <Icon name="locate" />
    </button>
  );
  const packButton = (
    <button type="button" aria-label={`Field pack: save ${pack.name} for offline use`} onClick={() => setPackOpen(true)} className={toolCls} data-testid="pack-open">
      <Icon name={off.record?.complete ? "check" : "download"} />
    </button>
  );
  const mapControls = (
    <div className="pointer-events-auto flex flex-col gap-2" data-testid="map-controls">
      <div className={`${stackCls} [@media(max-height:560px)]:hidden`}>
        <button type="button" aria-label="Zoom in" onClick={() => mapRef.current?.zoomIn({ duration: 200 })} className={toolCls}>
          <Icon name="plus" />
        </button>
        <button type="button" aria-label="Zoom out" onClick={() => mapRef.current?.zoomOut({ duration: 200 })} className={toolCls}>
          <Icon name="minus" />
        </button>
      </div>
      <div className={stackCls}>
        {recenter}
        <button type="button" aria-label="Select an area" aria-pressed={areaMode} onClick={() => setAreaMode((v) => !v)} className={`${toolCls} ${areaMode ? "bg-n-accent text-n-on-accent" : ""}`} data-testid="area-tool">
          <Icon name="area" />
        </button>
        <ShareButton className={toolCls} />
        {packButton}
      </div>
    </div>
  );

  const map = (
    <MapCanvas
      forecast={resolved.forecast}
      satellite={resolved.satellite}
      satelliteAge={resolved.satelliteAge}
      imagery={null}
      currents={currents}
      ports={ports}
      showPorts
      officialGeometry={officialGeometry}
      showOfficial
      selectedPort={selection?.kind === "harbor" ? selection.port : null}
      inspect={null}
      chrome="none"
      counties={counties}
      basemapPmtiles={basemapPm ?? null}
      clusterStations
      highlight={shownHighlight}
      onStatus={onStatus}
      retry={fieldRetry}
      initialBounds={initialBounds}
      initialPadding={{ top: 84, bottom: 150, left: 24, right: 24 }}
      cellEdges={active.kind === "forecast" && edges && edges.id === resolved.forecastLayer?.layer_id ? edges.fc : null}
      cellLabels={harbourOn ? harbour!.labels : null}
      gap={harbourOn && !selection ? harbour!.gap : null}
      stations={stationPoints}
      fade={false}
      onHover={bp !== "phone" && canHover && !areaMode ? onHover : undefined}
      onPort={(code) => select({ kind: "harbor", port: code })}
      onStationPick={(id) => select({ kind: "station", id })}
      onPoint={(p: MapPoint) => select(active.kind === "forecast" ? { kind: "cell", lat: p.lat, lon: p.lon } : { kind: "point", lat: p.lat, lon: p.lon })}
      onReady={(m) => {
        mapRef.current = m;
        setMapReady(true);
        if (initialAt && initialAt.length === 3 && initialAt.every(Number.isFinite)) m.jumpTo({ center: [initialAt[1], initialAt[0]], zoom: initialAt[2] });
        else {
          const p = padding();
          m.fitBounds(initialBounds, { padding: initialBounds === STATEWIDE.bounds && bp !== "phone" ? { ...p, top: 112, bottom: 24 } : p, duration: 0 });
          const s = decodeSelection(q0);
          if (s) m.once("idle", () => focusOn(s));
        }
        // through the ref: a handler captured here would write the selection of the first render
        m.on("moveend", () => writeUrlRef.current());
        m.on("zoomend", () => setZoom(m.getZoom()));
        setZoom(m.getZoom());
      }}
    />
  );
  const writeUrlRef = useRef(writeUrl);
  writeUrlRef.current = writeUrl;

  // phone sheet
  const [vh, setVh] = useState(800);
  const sheetHalf = Math.round((vh - DOCK) * 0.5);
  useEffect(() => {
    const f = () => setVh(window.innerHeight);
    f();
    window.addEventListener("resize", f);
    return () => window.removeEventListener("resize", f);
  }, []);

  const starting = !mapReady;
  // under the search box (desktop and tablet in the top bar, phones in their own stack): the way back to
  // a briefing, the offline state and a layer that failed, one under another and never over each other
  const offlineBanner = !off.online ? <OfflineBanner record={off.record ?? null} problem={off.problem} name={pack.name} inline /> : null;
  // clip, not hidden: a hidden overflow can still be scrolled by focus or scrollIntoView, which slid
  // the whole workspace (and the dock) up the screen on phones
  return (
    <div className="env-instrument relative h-dvh w-full overflow-clip" data-testid="explore">
      <SkipLink to="place-search" label="Skip to search places" />
      <h1 className="sr-only">Explore: harmful algal bloom evidence on the California coast{placeName ? `, ${placeName}` : ""}</h1>
      <div className="absolute inset-0" data-testid="map">
        {basemapPm !== undefined && synced ? map : null}
      </div>
      {hoverSource && bp !== "phone" && <HoverLayer sink={hoverSink} manifest={manifest} baseUrl={baseUrl} source={hoverSource} panelOpen={panelOpen} />}
      {packOpen && (
        <div className="absolute inset-0 z-50 grid place-items-center bg-[rgba(4,16,22,0.55)] p-3" onClick={(e) => e.target === e.currentTarget && setPackOpen(false)}>
          <div role="dialog" aria-modal="true" aria-label={`Field pack: ${pack.name}`} className="env-chart nx-deck nx-rise max-h-[92dvh] w-[min(520px,100%)] overflow-y-auto p-5 text-n-ink">
            <OfflinePanel pack={pack} manifest={manifest} official={official} baseUrl={baseUrl} onClose={() => setPackOpen(false)} />
          </div>
        </div>
      )}
      {/* the start card: what is loading, what can be done meanwhile (never a bare spinner) */}
      {starting && <StartCard now={now} />}
      {areaMode && (
        <div
          {...areaHandlers}
          className="absolute inset-0 z-[5] cursor-crosshair touch-none"
          data-testid="area-overlay"
        >
          <p className="pointer-events-none absolute left-1/2 top-24 -translate-x-1/2 rounded-full bg-n-surface px-4 py-2 text-[14px] font-semibold text-n-ink shadow-[var(--nx-shadow)]">Drag across the water to select an area · Esc to cancel</p>
        </div>
      )}

      <TopBar
        active="explore"
        now={now}
        floating
        slot={
          bp && bp !== "phone" ? (
            <div className="flex flex-col items-start gap-1.5">
              <PlaceSearch ports={ports} stations={stations} regions={regions} onPick={onSearch} current={placeName ?? ""} />
              <RegistryDisclosure className="w-fit rounded-full border border-n-line bg-n-surface px-3 py-1" fixture={isFixture(manifest)} compact />
              {offlineBanner && <div className="w-full max-w-[420px]">{offlineBanner}</div>}
              {fieldProblem}
              {backTo && <BackToBriefing {...backTo} className="w-fit max-w-full" />}
            </div>
          ) : null
        }
        notices={<NoticesButton official={official} status={officialStatus} compact={bp === "phone"} local={harbourNotices} />}
      />

      {bp === "desktop" && (
        <>
          {deckCompact ? (
            capsule
          ) : (
            <aside aria-label="Map layers" className="nx-deck absolute bottom-4 left-4 top-[76px] z-20 flex flex-col overflow-hidden" style={{ width: LAYERS }} data-testid="layer-deck">
              {panelOpen && (
                <button type="button" aria-expanded onClick={() => setDeckOpen(false)} className="nx-focus flex min-h-11 shrink-0 items-center justify-between border-b border-n-line px-3 text-[14px] font-semibold text-n-ink-2 hover:text-n-ink" data-testid="layer-deck-fold">
                  <span className="flex items-center gap-2">
                    <Icon name="layers" className="h-[18px] w-[18px] shrink-0" /> Layers and legend
                  </span>
                  <Icon name="chevronDown" className="h-4 w-4 shrink-0 rotate-180" />
                </button>
              )}
              <div className="flex-1 overflow-y-auto p-3 [scrollbar-width:thin]">{layerDeck}</div>
              <div className="border-t border-n-line px-3 py-2">
                <Attribution />
              </div>
            </aside>
          )}
          <div className="absolute bottom-4 z-20" style={{ left: deckCompact ? 16 : LAYERS + 28, right: (panelOpen ? PANEL + 28 : 0) + 84 }}>
            {/* a deck only for a layer that has published frames (the multi-sensor view has none) */}
            {hasTime && (
              <div className="nx-deck mx-auto max-w-[760px] p-3" data-testid="timebar-deck">
                <Timebar manifest={manifest} choice={active} onChoice={setChoiceRaw} now={now} />
              </div>
            )}
          </div>
          <div className="absolute bottom-4 z-20" style={{ right: (panelOpen ? PANEL + 28 : 0) + 16 }}>
            {mapControls}
          </div>
          {inspection && (
            <aside aria-label="Selected" className="env-chart nx-deck nx-panel absolute bottom-4 right-4 top-[76px] z-20 overflow-y-auto overscroll-contain p-5 text-n-ink [scrollbar-width:thin]" style={{ width: PANEL }}>
              {inspection}
            </aside>
          )}
        </>
      )}

      {bp === "tablet" && (
        <>
          {/* the deck and timebar end where the map's tools begin: the tools never cover the timebar */}
          <div className="absolute bottom-4 left-4 z-20" style={{ width: `min(420px, calc(100% - ${(panelOpen ? 396 : 0) + 16 + 56 + 24 + 16}px))` }} data-testid="tablet-deck">
            <details className="nx-deck group mb-2 max-h-[55vh] overflow-y-auto p-3" data-testid="layer-deck">
              <summary className="nx-focus flex min-h-12 cursor-pointer list-none items-center justify-between text-[15px] font-semibold">
                <span className="flex items-center gap-2">
                  <Icon name="layers" /> Layers and legend
                </span>
                <Icon name="chevronDown" className="h-5 w-5 transition-transform group-open:rotate-180" />
              </summary>
              <div className="mt-3">{layerDeck}</div>
              <div className="mt-3">
                <Attribution />
              </div>
            </details>
            {hasTime && (
              <div className="nx-deck p-3">
                <Timebar manifest={manifest} choice={active} onChoice={setChoiceRaw} now={now} compact />
              </div>
            )}
            {/* the deck carries the full credits once opened; closed, the short line stays on the map */}
            <div className="mt-1.5">
              <MapCredits />
            </div>
          </div>
          <div className="absolute bottom-4 z-20" style={{ right: (panelOpen ? 396 : 0) + 16 }}>
            {mapControls}
          </div>
          {inspection && (
            <aside aria-label="Selected" className="env-chart nx-deck nx-panel absolute bottom-4 right-4 top-[76px] z-20 w-[380px] overflow-y-auto p-4 text-n-ink">
              {inspection}
            </aside>
          )}
        </>
      )}

      {bp === "phone" && (
        <>
          <div className="absolute inset-x-3 top-[68px] z-20 flex flex-col gap-1.5">
            <PlaceSearch ports={ports} stations={stations} regions={regions} onPick={onSearch} current={placeName ?? ""} />
            <RegistryDisclosure className="w-fit max-w-full rounded-full border border-n-line bg-n-surface px-3 py-1" fixture={isFixture(manifest)} compact />
            {/* clear of the map controls' column on the right */}
            {offlineBanner && <div className="pr-[68px]">{offlineBanner}</div>}
            {fieldProblem}
            {backTo && <BackToBriefing {...backTo} className="w-fit max-w-full" />}
          </div>
          {/* the sheet ends above the dock; the map stays live above it (non-modal) */}
          <div className="pointer-events-none absolute inset-x-0 top-0 z-30 overflow-clip" style={{ bottom: `calc(${DOCK}px + env(safe-area-inset-bottom, 0px))` }}>
            {snap === "peek" && (
              <div className="pointer-events-auto absolute right-3" style={{ bottom: PEEK + CREDITS + 10 }} data-testid="map-controls">
                <div className={stackCls}>
                  {recenter}
                  <ShareButton className={toolCls} />
                  {packButton}
                </div>
              </div>
            )}
            {/* the map's credits (ODbL) on the map's edge, one line just above the sheet at every height
                the map shows; at full height (the map all but covered) they ride on the sheet's edge */}
            {snap !== "full" && (
              <div className="pointer-events-auto absolute left-2 right-2 transition-[bottom] duration-200" style={{ bottom: (snap === "peek" ? PEEK : sheetHalf) + 4 }}>
                <MapCredits short />
              </div>
            )}
            <div className={`env-chart pointer-events-auto ${bp === "phone" ? "[&_[data-testid=open-briefing]]:hidden" : ""}`}>
              <MobileSheet
                snaps={[
                  { id: "peek", height: PEEK },
                  { id: "half", height: sheetHalf },
                  { id: "full", height: Math.round((vh - DOCK) * 0.9) },
                ]}
                snap={snap}
                onSnap={setSnap}
                label={selection ? "Selected" : "Layers and time"}
                header={null}
                field
                edge={snap === "full" ? <MapCredits short /> : null}
              >
                <div className="px-1 pb-6 text-n-ink">
                  {/* with a place open, the layers stay one tap away: no need to close the place to change the map */}
                  {inspection && (
                    <div className={`mb-3 border-b border-n-line pb-3 ${compactChips}`} data-testid="sheet-layer-chips">
                      <LayerChips manifest={manifest} choice={active} onChoice={setChoiceRaw} defaults={{ lead: firstLead }} />
                    </div>
                  )}
                  {inspection ?? (
                    <div className="flex flex-col gap-4">
                      {activeCard}
                      <div className="border-t border-n-line pt-4">
                        <p className="nx-eyebrow mb-2">Change what the map shows</p>
                        <LayerList manifest={manifest} choice={active} onChoice={setChoiceRaw} now={now} defaults={{ lead: firstLead }} />
                        <div className="mt-3">
                          <LayerOptions manifest={manifest} choice={active} onChoice={setChoiceRaw} />
                        </div>
                      </div>
                      <Attribution />
                    </div>
                  )}
                </div>
              </MobileSheet>
            </div>
          </div>
          <div className="absolute inset-x-0 bottom-0 z-40 border-t border-n-line bg-n-surface" style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
            <NavTabs active="explore" size="dock" />
          </div>
        </>
      )}
    </div>
  );
}

/**
 * The map's credits in one line (ODbL asks that OpenStreetMap be credited wherever its map is seen):
 * on the map's edge on phones, under the folded deck on desktop, under the tablet's deck.
 */
function MapCredits({ short = false }: { short?: boolean }) {
  return (
    <p className="w-fit max-w-full truncate rounded-md bg-[rgba(6,17,30,0.72)] px-2 py-0.5 text-[11px] leading-[16px] text-[#c9d6e2] [&_a]:underline [&_a]:underline-offset-2" data-testid={short ? "attribution-short" : "attribution-line"}>
      <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">© OpenStreetMap contributors</a> · <a href="https://www.openmaptiles.org/" target="_blank" rel="noreferrer">© OpenMapTiles</a> · NOAA relief · not for navigation
    </p>
  );
}

/** Basemap and relief credits, in the interface (the map's own control is off in v4). */
function Attribution() {
  return (
    <p className="text-[11.5px] leading-snug text-n-ink-3" data-testid="attribution">
      Basemap{" "}
      <a className="underline" href="https://openfreemap.org" target="_blank" rel="noreferrer">OpenFreeMap</a> ·{" "}
      <a className="underline" href="https://www.openmaptiles.org/" target="_blank" rel="noreferrer">© OpenMapTiles</a> ·{" "}
      <a className="underline" href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">© OpenStreetMap contributors</a>. Relief{" "}
      <a className="underline" href="https://doi.org/10.25921/fd45-gt74" target="_blank" rel="noreferrer">NOAA NCEI ETOPO 2022</a> and{" "}
      <a className="underline" href="https://www.ncei.noaa.gov/products/coastal-relief-model" target="_blank" rel="noreferrer">Coastal Relief Model</a>. Counties: CA Dept. of Technology / CDTFA. Not for navigation.
    </p>
  );
}

function ShareButton({ className }: { className: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      aria-label={done ? "Link copied" : "Copy a link to this view"}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(window.location.href);
          setDone(true);
          setTimeout(() => setDone(false), 1800);
        } catch {
          setDone(false);
        }
      }}
      className={className}
      data-testid="share-view"
    >
      <Icon name={done ? "check" : "share"} />
      <span className="sr-only" aria-live="polite">
        {done ? "Link copied" : ""}
      </span>
    </button>
  );
}

function StartCard({ now }: { now: Date | null }) {
  return (
    <div className="pointer-events-none absolute inset-0 z-[15] grid place-items-center p-4" data-testid="start-card">
      <svg className="absolute inset-0 h-full w-full text-n-accent opacity-25" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice" aria-hidden>
        {[0, 1, 2, 3, 4].map((i) => (
          <path key={i} className={i === 2 ? "nx-sweep" : undefined} d={`M-20 ${70 + i * 38} C 80 ${40 + i * 40}, 160 ${120 + i * 30}, 240 ${80 + i * 38} S 380 ${110 + i * 34}, 430 ${90 + i * 38}`} fill="none" stroke="currentColor" strokeWidth={i === 2 ? 1.6 : 0.8} />
        ))}
      </svg>
      <div className="nx-deck pointer-events-auto relative w-[min(440px,100%)] p-5">
        <p className="text-[18px] font-semibold">Preparing the coast</p>
        <p className="mt-1 text-[14px] text-n-ink-2">The map is loading. Search a harbour, read the notices or open a briefing meanwhile.</p>
        <div className="mt-3">
          <SourceList now={now} compact />
        </div>
      </div>
    </div>
  );
}

function AreaView({ bbox, ctx }: { bbox: [number, number, number, number]; ctx: { manifest: Manifest; baseUrl: string; stations: StationLite[] | null; ports: PortsCollection | null; variable: CharmVariable; lead: number; onSelect: (s: Selection) => void } }) {
  const layer = useMemo(() => resolve(ctx.manifest, { kind: "forecast", variable: ctx.variable, lead: ctx.lead }, ctx.baseUrl).forecastLayer, [ctx.manifest, ctx.variable, ctx.lead, ctx.baseUrl]);
  const [w, s, e, n] = bbox;
  const inBox = (lat: number, lon: number) => lat >= s && lat <= n && lon >= w && lon <= e;
  const sts = (ctx.stations ?? []).filter((x) => inBox(x.lat, x.lon));
  const prts = (ctx.ports?.features ?? []).filter((f) => {
    const [lon, lat] = f.geometry.coordinates as number[];
    return inBox(lat, lon);
  });
  return (
    <div className="flex flex-col gap-4" data-testid="inspect-area">
      <p className="nx-data text-[12.5px] text-n-ink-3">
        {s.toFixed(3)}–{n.toFixed(3)}°N, {Math.abs(e).toFixed(3)}–{Math.abs(w).toFixed(3)}°W
      </p>
      <AreaModel layer={layer} baseUrl={ctx.baseUrl} bbox={bbox} variable={ctx.variable} />
      <section className="flex flex-col gap-2 border-t border-n-line pt-4">
        <h3 className="nx-eyebrow">Shore stations in the box</h3>
        {sts.length ? (
          <ul>
            {sts.map((x) => (
              <li key={x.station_id}>
                <button type="button" onClick={() => ctx.onSelect({ kind: "station", id: x.station_id })} className="nx-focus flex min-h-12 w-full items-center justify-between rounded-lg px-1 text-left hover:bg-n-surface-2">
                  <span className="text-[14.5px] font-semibold">{x.name}</span>
                  <span className="text-[12.5px] text-n-ink-3">{lastHabWords(lastHabSample(x))}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-[14px] text-n-ink-2">No CalHABMAP station in this box: no sample is not a clean result.</p>
        )}
      </section>
      {!!prts.length && (
        <section className="flex flex-col gap-2 border-t border-n-line pt-4">
          <h3 className="nx-eyebrow">Harbours in the box</h3>
          <ul>
            {prts.map((f) => (
              <li key={f.properties.port_code}>
                <button type="button" onClick={() => ctx.onSelect({ kind: "harbor", port: f.properties.port_code })} className="nx-focus flex min-h-12 w-full items-center rounded-lg px-1 text-left text-[14.5px] font-semibold hover:bg-n-surface-2">
                  {f.properties.display_name ?? f.properties.name}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
      <p className="text-[12.5px] text-n-ink-3">Satellite and current statistics for an area are not computed in this version; use their layers and a point.</p>
    </div>
  );
}

/** The way back to the briefing that opened this map. */
function BackToBriefing({ href, name, className }: { href: string; name: string; className: string }) {
  return (
    <a href={href} className={`nx-press nx-focus env-chart flex min-h-11 items-center gap-2 rounded-full border border-n-line bg-n-surface px-4 text-[14.5px] font-semibold text-n-ink shadow-[var(--nx-shadow)] ${className}`} data-testid="back-to-briefing">
      <Icon name="back" className="h-[18px] w-[18px] shrink-0" /> Back to the {name} briefing
    </a>
  );
}

type HoverAt = { p: MapPoint; s: { x: number; y: number } };

/** The hover readout and the pointer state behind it: the workspace hands it positions through `sink`. */
function HoverLayer({ sink, manifest, baseUrl, source, panelOpen }: { sink: React.RefObject<((h: HoverAt | null) => void) | null>; manifest: Manifest; baseUrl: string; source: HoverSource; panelOpen: boolean }) {
  const [hover, setHover] = useState<HoverAt | null>(null);
  useEffect(() => {
    sink.current = setHover;
    return () => {
      sink.current = null;
    };
  }, [sink]);
  if (!hover || !source) return null;
  return (
    // the readout takes the dark tokens but must never paint over the map: transparent wrapper
    <div className="theme-dark pointer-events-none absolute inset-0 z-[12] !bg-transparent">
      <HoverReadout manifest={manifest} baseUrl={baseUrl} source={source} point={hover.p} screen={hover.s} bounds={{ w: window.innerWidth - (panelOpen ? PANEL + 24 : 0), h: window.innerHeight }} />
    </div>
  );
}
