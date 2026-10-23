"use client";

import { toast } from "@/v4/ui/toast";
import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { StationLite } from "@/lib/data";
import { CHARM_VARIABLES, artifactUrl, charmLayer, charmRun, satelliteLatest, isFixture, sourceStatus, type CharmVariable } from "@/lib/layers";
import { chooseHour, currentsHourly, fieldFeatures, loadFieldCached, type CurrentField } from "@/lib/currents";
import { decodeFootprint, encodeFootprint, footprintCell, footprintPolygon, productOf, type FootprintRef, type Lattice } from "@/lib/footprint";
import { satelliteSource } from "@/lib/readout";
import { clockLabel } from "@/v4/ui/time";
import type { LayerArtifact } from "@/generated/schema";
import { cellEdges, cellLabels, distanceKm, gapAt, loadGrid } from "@/lib/grid";
import { gapFeatures, lastHabSample, stationFeatures } from "@/lib/provenance";
import { clipLimitsToWater, labelledGeometry } from "@/lib/officialGeometry";
import { loadWaterFallback, type WaterGeometry } from "@/lib/waterFallback";
import { COAST_BOUNDS, coastCamera, journeyTo, seaward, waterSampler, type Insets } from "@/lib/camera";
import { forecastFreshness, openingLayer, type Opening } from "@/lib/opening";
import { ageInDays } from "@/lib/freshness";
import { defaultLead } from "@/lib/charmDays";
import { pacificToday } from "@/lib/time";
import { useNow } from "@/lib/useNow";
import type { CurrentsOverlay, FootprintOverlay, MapPoint } from "@/components/map/MapCanvas";
import { MobileSheet, type SnapId } from "@/components/MobileSheet";
import { HoverReadout, type HoverSource } from "@/components/map/HoverReadout";
import { useReportSource } from "@/v4/readiness/store";
import { currentsEntry, fieldState, layerFreshness, modelEntry, newestSatellite, satelliteEntry } from "@/v4/readiness/shown";
import { SourceList } from "@/v4/readiness/StatusCenter";
import { TopBar, NavTabs } from "@/v4/shell/TopBar";
import { NoticesButton, type LocalNotices } from "@/v4/shell/Notices";
import { RegistryDisclosure } from "@/v4/shell/Disclosure";
import { associationsAt, localNotices } from "@/lib/applicability";
import { bearingName, coastHref, pointKey, sitePlaces } from "@/lib/coast";
import { readHome } from "@/v4/coast/home";
import { SkipLink } from "@/v4/shell/SkipLink";
import { Icon } from "@/v4/ui/Icon";
import { dayLabel, kmLabel, lastHabWords } from "@/v4/ui/time";
import { decodeSelection, encodeSelection, sameSelection, type Camera, type HistoryEntry, type Selection } from "@/v4/state/selection";
import { choiceParams, parseChoice, resolve, runDays, useDerivedIndex, VARIABLE_WORDS, type LayerChoice } from "./layers";
import { derivedEstimate, derivedOutlook, derivedSst, nearshoreCheck } from "@/lib/derived";
import { LayerList, LayerOptions, Legend, LegendLine, Provenance, ReferenceLines, layerWords } from "./LayerPanel";
import { TimeRail, timeSteps } from "./Timebar";
import "./explore.css";
import { FlowControls, useFlowPrefs, useReducedMotion } from "./FlowControls";
import { PlaceSearch, type SearchHit } from "./Search";
import { AreaModel, CellBody, CrossLinks, EvidenceFold, HarborView, Header, PointView, StationView, tripOf, type InspectCtx } from "./Inspect";
import { boxFeature, useAreaDrag } from "./AreaDrag";
import { useCellRead } from "./reads";
import { PACKS } from "@/v4/offline/packs";
import { useOfflinePack } from "@/v4/offline/useOfflinePack";
import { useRecordSource } from "@/v4/offline/OfflinePanel";
import { PackDialog } from "@/v4/offline/PackDialog";
import { OfflineBanner } from "@/v4/offline/OfflineBanner";
import { readHomePort } from "@/v4/state/homePort";
import { AnimatePresence, motion, useIsPresent } from "motion/react";
import { EASE_IN, EASE_OUT, T, markAppReady, useSwapFade } from "@/v4/motion";
import { ContourLoader } from "@/v4/ui/ContourLoader";

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
const DOCK = 64; // phone navigation dock height
const PEEK = 196; // phone sheet at its peek: the dock (layer button, ramp, the time rail)
// at rest with nothing current: the layer button, the one status line and its two dated choices (taller on a narrow phone, where they wrap)
const peekRest = (w: number) => (w < 360 ? 296 : 262);


/** The height of an element as laid out (the dock and the map's bottom margin): the camera's insets follow it. */
function useHeight(ref: React.RefObject<HTMLElement | null>, fallback: number, key: unknown): number {
  const [h, setH] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const f = () => setH(Math.round(el.getBoundingClientRect().height));
    f();
    const ro = new ResizeObserver(f);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref, key]);
  return h;
}

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

/** The published region (ports.regions bounds) containing a point, or null. */
function regionAt(ports: PortsCollection | null, lat: number, lon: number): string | null {
  for (const r of ports?.regions ?? []) {
    const [[w, s], [e, n]] = r.bounds;
    if (lat >= s && lat <= n && lon >= w && lon <= e) return r.id;
  }
  return null;
}
/** Below this zoom the view is statewide: the currents hour follows the whole domain, not one region. */
const REGION_ZOOM = 7.5;
/** The time step key of a currents layer ("20261008T12Z", or "mean"). */
const hourKey = (l: LayerArtifact) => (l.layer_id.endsWith("mean24h") ? "mean" : (l.layer_id.split("_").pop() ?? ""));
/** The lattice a footprint product is published on (for a footprint carried in the URL). */
function latticeOf(m: Manifest, p: FootprintRef["product"]): Lattice | null {
  if (p === "olci300" || p === "viirs750") return satelliteLatest(m, p)?.grid ?? null;
  if (p === "hfr2km") return currentsHourly(m)[0]?.vectors?.u_grid ?? null;
  return charmLayer(m, "particulate_domoic", 0)?.grid ?? charmLayer(m, "pseudo_nitzschia", 0)?.grid ?? null;
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
  // the instrument at rest: nothing current, no layer drawn (the status and its dated choices are in the dock)
  const atRest = active.kind === "none" && opening?.reason === "nothing-current";
  // derived chlorophyll products (lib/derived): offered only when the dataset publishes their index
  const derivedRaw = useDerivedIndex(baseUrl);
  const derived = derivedRaw ?? null;
  const check = nearshoreCheck(derived);
  useEffect(() => {
    // a link to a derived layer (outlook, sea temperature) the dataset does not have: the default layer instead
    if ((choice?.kind === "outlook" && !derivedOutlook(derivedRaw)) || (choice?.kind === "sst" && !derivedSst(derivedRaw))) if (derivedRaw !== undefined) setChoiceRaw(null);
  }, [choice, derivedRaw]);
  // the region in view or selected: the currents open on the newest hour that observed it (lib/currents chooseHour)
  const [selection, setSelectionRaw] = useState<Selection | null>(() => decodeSelection(q0));
  const [viewRegion, setViewRegion] = useState<string | null>(() => {
    const r = q0.get("region");
    return r && r !== "california" ? r : null;
  });
  const selRegion = useMemo(() => {
    if (!selection || !ports) return null;
    if (selection.kind === "harbor") return ports.features.find((f) => f.properties.port_code === selection.port)?.properties.region ?? null;
    const st = selection.kind === "station" ? stations?.find((x) => x.station_id === selection.id) : null;
    const at = st ? { lat: st.lat, lon: st.lon } : selection.kind === "area" ? { lat: (selection.bbox[1] + selection.bbox[3]) / 2, lon: (selection.bbox[0] + selection.bbox[2]) / 2 } : selection.kind === "cell" || selection.kind === "point" ? selection : null;
    return at ? regionAt(ports, at.lat, at.lon) : null;
  }, [selection, ports, stations]);
  const hourRegion = selRegion ?? viewRegion;
  const resolved = useMemo(() => resolve(manifest, active, baseUrl, derived, { region: hourRegion }), [manifest, active, baseUrl, derived, hourRegion]);
  // the legend swaps with a short fade when the layer changes (a new day keeps the same legend)
  const legendKey = JSON.stringify({ ...active, day: undefined, lead: undefined });
  const legendRef = useSwapFade<HTMLDivElement>(legendKey);
  const legendPhoneRef = useSwapFade<HTMLDivElement>(legendKey);
  const variable: CharmVariable = active.kind === "forecast" ? active.variable : (CHARM_VARIABLES as readonly string[]).includes(q0.get("var") ?? "") ? (q0.get("var") as CharmVariable) : "particulate_domoic";
  const lead = active.kind === "forecast" ? active.lead : firstLead;

  // ------------------------------------------------------------------ selection + history
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
      // (the closing panel stays a moment for its exit, inert: focus inside it counts as lost)
      if (to?.isConnected && (!a || a === document.body || !a.isConnected || a.closest("[inert]"))) to.focus({ preventScroll: true });
    }
    if (selection && focusPanel.current) {
      focusPanel.current = false;
      requestAnimationFrame(() => panelRef.current?.focus({ preventScroll: true }));
    }
  }, [selection]);

  // a new selection in an open panel: the body cross-fades (the panel itself stays put)
  const selKey = selection ? JSON.stringify(selection) : null;
  const lastKey = useRef(selKey);
  useEffect(() => {
    const was = lastKey.current;
    lastKey.current = selKey;
    // from dimmed, never from empty: the panel never reads as a blank box between two places
    if (was && selKey && was !== selKey) panelRef.current?.animate?.([{ opacity: 0.45 }, { opacity: 1 }], { duration: 160, easing: "cubic-bezier(0.23, 1, 0.32, 1)" });
  }, [selKey]);

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
  // the selection's footprint (computed further down, once the layers are resolved): fp=product:row:col
  const fpEncRef = useRef<string | null>(null);
  const writeUrl = useCallback(() => {
    if (!synced) return;
    const q = new URLSearchParams();
    if (choice) for (const [k, v] of Object.entries(choiceParams(choice))) q.set(k, v);
    if (q0.get("from")?.startsWith("/coast")) q.set("from", q0.get("from")!);
    const sel = encodeSelection(selection);
    if (sel) q.set("sel", sel);
    if (sel && fpEncRef.current) q.set("fp", fpEncRef.current);
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
  // the launch screen (shown once a session) steps aside when the map has loaded
  useEffect(() => {
    if (mapReady) markAppReady();
  }, [mapReady]);
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

  const [curField, setCurField] = useState<{ id: string; layer: LayerArtifact; field: CurrentField } | null>(null);
  const curFieldRef = useRef(curField);
  curFieldRef.current = curField;
  const [curState, setCurState] = useState<"idle" | "loading" | "ready" | "failed">("idle");
  // "Try again" for the currents grids (a failed grid is never kept in lib/grid's cache)
  const [curRetry, setCurRetry] = useState(0);
  const curLayer = resolved.curLayer;
  useEffect(() => {
    if (!curLayer) return setCurState("idle");
    if (curFieldRef.current?.id === curLayer.layer_id) return setCurState("ready");
    let live = true;
    setCurState("loading");
    loadFieldCached(baseUrl, curLayer)
      .then((f) => {
        if (!live) return;
        setCurField({ id: curLayer.layer_id, layer: curLayer, field: f });
        setCurState("ready");
      })
      .catch(() => live && setCurState("failed"));
    return () => {
      live = false;
    };
  }, [curLayer, baseUrl, curRetry]);
  // v10 playback honesty: the map, its time label and every readout stay on the hour whose data has arrived
  // until the next one's has (a cut, never a blend); the time bar marks the hour asked for as loading
  const shownCur = active.kind === "currents" && curLayer ? curField : null;
  const curPending = !!curLayer && !!shownCur && shownCur.id !== curLayer.layer_id && curState === "loading";
  const shownResolved = useMemo(() => (shownCur ? { ...resolved, curLayer: shownCur.layer } : resolved), [resolved, shownCur]);
  // the next real hour, asked for during playback so each step cuts to data already loaded
  const prefetch = useCallback(
    (c: LayerChoice) => {
      if (c.kind !== "currents") return;
      const l = resolve(manifest, c, baseUrl, derived, { region: hourRegion }).curLayer;
      if (l) loadFieldCached(baseUrl, l).catch(() => undefined);
    },
    [manifest, baseUrl, derived, hourRegion],
  );
  // v10: moving particles by default (the map's own WebGL context), static arrows on request; paused and
  // coloured by speed on request; all kept per device. Reduced motion: one still frame.
  const [flowPrefs, setFlowPrefs, prefsReady] = useFlowPrefs();
  const reduced = useReducedMotion();
  const compareChl = active.kind === "currents" && !!active.chl;
  const flowOpts = useMemo(() => ({ paused: flowPrefs.paused, still: reduced, colour: flowPrefs.colour, coarser: compareChl }), [flowPrefs.paused, flowPrefs.colour, reduced, compareChl]);
  const arrows = prefsReady && flowPrefs.view === "arrows";
  const arrowFeatures = useMemo(() => (shownCur && arrows ? fieldFeatures(shownCur.field) : null), [shownCur, arrows]);
  const currents: CurrentsOverlay | null = useMemo(() => {
    if (!shownCur || !prefsReady) return null;
    if (arrows) return arrowFeatures ? { id: shownCur.id, field: shownCur.field, mode: "arrows", features: arrowFeatures } : null;
    return { id: shownCur.id, field: shownCur.field, mode: "flow", flow: flowOpts };
  }, [shownCur, prefsReady, arrows, arrowFeatures, flowOpts]);
  const flowLegend = active.kind === "currents" && !arrows ? { colour: flowPrefs.colour } : null;
  // no hour chosen and the newest hour did not observe the region in view: said, with the hour shown
  const hourPick = active.kind === "currents" && !active.hour && !active.mean ? chooseHour(manifest, hourRegion) : null;
  const hourNote =
    hourPick?.newestMissed && hourPick.newest?.time.valid_time && hourPick.layer?.time.valid_time ? (
      <p className="flex items-start gap-2 text-[12.5px] leading-snug text-n-stale" data-testid="currents-hour-note">
        <Icon name="clock" className="mt-0.5 h-4 w-4 shrink-0" />
        <span>
          The newest radar hour ({clockLabel(hourPick.newest.time.valid_time)}) {hourPick.newestFraction ? "saw little of" : "has no coverage in"} {hourPick.regionLabel ?? "this area"}. Showing {clockLabel(hourPick.layer.time.valid_time)}, the newest hour with coverage here.
        </span>
      </p>
    ) : null;

  const stationPoints = useMemo(() => (stations?.length ? stationFeatures(stations, now, now ? pacificToday(now) : null) : null), [stations, now]);
  const [water, setWater] = useState<WaterGeometry | null>(null);
  useEffect(() => {
    let live = true;
    loadWaterFallback().then((g) => live && g && setWater(g));
    return () => {
      live = false;
    };
  }, []);
  const sampler = useMemo(() => waterSampler(water), [water]);
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
  // a point or model cell selected: the notices whose geography covers that water, as its panel lists them
  const pointNotices = useMemo<LocalNotices | null>(() => {
    if (!official || (selection?.kind !== "point" && selection?.kind !== "cell")) return null;
    const { lat, lon } = selection;
    return localNotices(official, pointKey(lat, lon), "this point", lat, lon, "water");
  }, [selection, official]);
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
  // the layer list (a popover from the dock's layer button; in the sheet on a phone) closes on a new selection
  const [deckOpen, setDeckOpen] = useState(false);
  useEffect(() => setDeckOpen(false), [selection]);
  // what the floating surfaces occlude, from the layout itself: the camera frames a selection in the open map
  const panelW = bp === "desktop" ? 380 : 360;
  const dockRef = useRef<HTMLDivElement | null>(null);
  const marginRef = useRef<HTMLDivElement | null>(null);
  const dockH = useHeight(dockRef, 112, bp);
  const marginH = useHeight(marginRef, 30, bp);
  // `open`: frame for the panel being open (a camera move made for a new selection, in the same
  // step that selects it, before panelOpen has caught up)
  const padding = useCallback(
    (open: boolean = panelOpen) => {
      if (bp === "desktop" || bp === "tablet") return { top: 84, bottom: marginH + dockH + 28, left: 24, right: open ? panelW + 28 + 24 : 16 + 44 + 24 };
      // phone: a selection opens the sheet to half height, so the camera centres it in the map strip
      // that stays visible above the sheet (not behind it)
      const sheet = open ? Math.round((window.innerHeight - DOCK) * 0.5) : atRest ? peekRest(window.innerWidth) : PEEK;
      return { top: 112, bottom: sheet + DOCK + 24, left: 20, right: 20 };
    },
    [bp, panelOpen, panelW, marginH, dockH, atRest],
  );
  // v10 rework: the opening frames the coastal ocean, not the state. The coast (Crescent City to San Diego)
  // fills the height between the header and the time rail; its east edge sits just left of the map tools,
  // so the land is a narrow margin and the ocean fills the rest (under and beside the layer index)
  const coastInsets = useCallback((): Insets => {
    if (bp === "phone") return { top: 108, bottom: (atRest ? peekRest(window.innerWidth) : PEEK) + DOCK + 48, left: 12, right: 14 };
    return { top: 64, bottom: marginH + dockH + 12, left: (bp === "desktop" ? 264 : 236) + 40, right: 16 + 44 + 56 };
  }, [bp, marginH, dockH, atRest]);
  const coastView = useCallback(() => {
    const c = mapRef.current?.getContainer();
    return coastCamera(c?.clientWidth ?? window.innerWidth, c?.clientHeight ?? window.innerHeight, coastInsets());
  }, [coastInsets]);
  const fitState = useCallback(() => {
    const m = mapRef.current;
    if (!m) return;
    journeyTo(m, coastView());
  }, [coastView]);
  const focusOn = useCallback(
    (s: Selection) => {
      const m = mapRef.current;
      if (!m) return;
      const pt = s.kind === "station" ? stations?.find((x) => x.station_id === s.id) : null;
      const pf = s.kind === "harbor" ? ports?.features.find((f) => f.properties.port_code === s.port) : null;
      const at: [number, number] | null = pt ? [pt.lon, pt.lat] : pf ? (pf.geometry.coordinates as [number, number]) : s.kind === "cell" || s.kind === "point" ? [s.lon, s.lat] : null;
      if (s.kind === "area") return m.fitBounds([[s.bbox[0], s.bbox[1]], [s.bbox[2], s.bbox[3]]], { padding: padding(true), duration: 500 });
      if (!at) return;
      const zoom = Math.max(m.getZoom(), s.kind === "harbor" ? 10.6 : 10.2);
      const pad = padding(true);
      // a harbour or station sits on the shore: the camera leans toward its water, so the place and the sea
      // in front of it fill the open map (on a phone, the strip above the sheet), not the land behind it
      const lean = s.kind === "harbor" || s.kind === "station" ? seaward(sampler, at[1], at[0]) : null;
      let center = at;
      if (lean) {
        const c = m.getContainer();
        const stripW = c.clientWidth - pad.left - pad.right;
        const stripH = c.clientHeight - pad.top - pad.bottom;
        const px = 0.3 * Math.min(stripW, stripH * 1.6);
        const wp = 512 * 2 ** zoom;
        const x = (at[0] + 180) / 360 + (lean[0] * px) / wp;
        const s0 = Math.sin((at[1] * Math.PI) / 180);
        const y = 0.5 - (0.25 * Math.log((1 + s0) / (1 - s0))) / Math.PI + (lean[1] * Math.min(px, stripH * 0.3)) / wp;
        center = [x * 360 - 180, (Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180) / Math.PI];
      }
      journeyTo(m, { center, zoom, padding: pad });
    },
    [stations, ports, padding, sampler],
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
  // a field that loaded is "ready" only while it is current: the same rule on every route (v4/readiness/shown)
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
      : // not on the map: still reported with its freshness, so an out-of-date run never leaves the status "Ready"
        modelEntry(manifest, now, "Not on the map"),
  );
  const obsFresh = (id: string, l: typeof resolved.satLayer) => ({ st: sourceStatus(manifest, id), fresh: layerFreshness(l ?? null, now) });
  const satOn = active.kind === "satellite" || (active.kind === "currents" && !!active.chl);
  // the satellite's newest view, reported even when another layer is on the map
  const satNewest = satOn ? resolved.satLayer : newestSatellite(manifest);
  const satF = obsFresh("satellite_chl", satNewest);
  const satField = fieldState(satF.fresh, satF.st);
  useReportSource(
    satOn
      ? { id: "satellite", label: "Satellite chlorophyll", state: mapStatus?.field === "failed" ? "unavailable" : mapStatus?.field === "ready" ? satField.state : "loading", detail: [resolved.satLayer ? `${resolved.satLayer.layer_id.split("_")[0].toUpperCase()} tiles` : "No layer", mapStatus?.field === "failed" ? "Some tiles could not be loaded" : satField.note].filter(Boolean).join(". "), asOf: resolved.satLayer?.time.observed_date ?? null, actions: mapStatus?.field === "failed" ? [{ label: "Try again", run: retryField }] : undefined }
      : satelliteEntry(manifest, now, "Not on the map"),
  );
  // the radar's newest hour, reported even when the currents are not on the map
  const curNewest = active.kind === "currents" ? curLayer : (currentsHourly(manifest).at(-1) ?? null);
  const curF = obsFresh("hf_radar", curNewest);
  const curReady = fieldState(curF.fresh, curF.st);
  useReportSource(
    active.kind === "currents"
      ? { id: "currents", label: "Surface currents (HF radar)", state: curState === "failed" ? "unavailable" : curState === "ready" ? curReady.state : "loading", detail: ["Hourly, 2 km cells", curState === "failed" ? "This hour could not be loaded" : curReady.note].filter(Boolean).join(". "), asOf: curLayer?.time.valid_time ?? null, actions: curState === "failed" ? [{ label: "Try again", run: retryCurrents }] : undefined }
      : currentsEntry(manifest, now, "Not on the map"),
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
    if (active.kind === "forecast" && resolved.forecastLayer) return { kind: "forecast", layer: resolved.forecastLayer, context: !!resolved.forecast?.grid, check };
    if (active.kind === "satellite" && resolved.satLayer) return { kind: "satellite", layer: resolved.satLayer, estimate: resolved.estimate ? resolved.estimateLayer : null };
    if (active.kind === "outlook" && resolved.outlookLayer) return { kind: "outlook", layer: resolved.outlookLayer };
    if (active.kind === "sst" && resolved.sstLayer) return { kind: "sst", layer: resolved.sstLayer };
    if (active.kind === "currents" && shownCur) return { kind: "currents", layer: shownCur.layer, field: shownCur.field };
    return null;
  }, [active.kind, resolved, shownCur, check]);

  // ------------------------------------------------------------------ keyboard
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      // the field-pack dialog is modal: its own Escape closes it, never the panel behind
      // a panel or dialog above it that handled Escape marks the key as used (preventDefault)
      if (packOpen || e.defaultPrevented) return;
      if (e.key === "Escape") {
        if (areaMode) setAreaMode(false);
        else if (selection) select(null);
        // the area tool's letter works only with the map focused: typed elsewhere, "a" is just a letter
      } else if (e.key === "a" && !e.metaKey && !e.ctrlKey && !e.altKey && t?.closest('[data-testid="map"]')) setAreaMode((v) => !v);
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [areaMode, selection, select, packOpen]);

  // ------------------------------------------------------------------ pieces
  const regions = useMemo(() => [STATEWIDE, ...(ports?.regions ?? []).map((r) => ({ id: r.id, label: r.label, bounds: (VIEW[r.id] ?? r.bounds) as [[number, number], [number, number]] }))], [ports]);
  const onSearch = (h: SearchHit) => {
    // a species: My Coast for it, at the harbour on the map or the reader's home port, else its species-first landing
    if (h.kind === "species") {
      const home = readHome();
      const trip = tripOf(q0.get("from"), home);
      const place = selection?.kind === "harbor" ? `port-${selection.port}` : (home?.place ?? null);
      window.location.assign(coastHref({ place, for: place ? trip.for : undefined, species: h.id, date: place ? trip.date : undefined }));
      return;
    }
    if (h.kind !== "region") focusPanel.current = true;
    if (h.kind === "harbor") selectAndGo({ kind: "harbor", port: Number(h.id) });
    else if (h.kind === "station") selectAndGo({ kind: "station", id: h.id });
    else if (h.kind === "site" && h.lat != null && h.lon != null) selectAndGo({ kind: "point", lat: h.lat, lon: h.lon });
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
  // is this point water on the map (the basemap's own water, or the bundled water)? null when unknown
  const waterAt = useCallback((scr: { x: number; y: number }): boolean | null => {
    const m = mapRef.current;
    if (!m) return null;
    const ids = ["water", "water-fallback"].filter((id) => m.getLayer(id) && m.getLayoutProperty(id, "visibility") !== "none");
    if (!ids.length) return null;
    if (m.queryRenderedFeatures([scr.x, scr.y], { layers: ids }).length) return true;
    // no water drawn here: land, unless the basemap has drawn nothing at all yet
    return m.isSourceLoaded("omt") ? false : null;
  }, []);
  const waterAtPoint = useCallback((lat: number, lon: number) => {
    const m = mapRef.current;
    if (!m) return null;
    const p = m.project([lon, lat]);
    const c = m.getCanvas();
    // only a point in view can be checked against the drawn water
    if (p.x < 0 || p.y < 0 || p.x > c.clientWidth || p.y > c.clientHeight) return null;
    return waterAt({ x: p.x, y: p.y });
  }, [waterAt]);
  const derivedCtx = { outlook: derivedOutlook(derived), estimate: derivedEstimate(derived), sst: derivedSst(derived), check };

  // ------------------------------------------------------------------ the selection's true footprint
  // the cell or pixel a selected point is read from, at its native size: the 3 km C-HARM cell, the 2 km radar
  // cell, the satellite pixel (for the multi-sensor view, the pixel of the sensor whose observation is shown)
  const selPoint = selection?.kind === "point" || selection?.kind === "cell" ? selection : null;
  const satForFp = active.kind === "satellite" && (active.view ?? "chl") === "chl" ? resolved.satLayer : null;
  const multiKey = selPoint && satForFp?.multisensor ? `${satForFp.layer_id}@${selPoint.lat},${selPoint.lon}` : null;
  const [multiSrc, setMultiSrc] = useState<{ key: string; layer: LayerArtifact | null } | null>(null);
  useEffect(() => {
    if (!multiKey || !selPoint || !satForFp) return;
    let live = true;
    satelliteSource(manifest, baseUrl, satForFp, selPoint.lat, selPoint.lon)
      .then((l) => live && setMultiSrc({ key: multiKey, layer: l }))
      .catch(() => live && setMultiSrc({ key: multiKey, layer: null }));
    return () => {
      live = false;
    };
    // the key names the layer and the point
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [multiKey, manifest, baseUrl]);
  const fpFromUrl = useMemo(() => decodeFootprint(q0.get("fp")), [q0]);
  const fp = useMemo<{ ref: FootprintRef; g: Lattice } | null>(() => {
    if (!selPoint) return null;
    const at = (product: FootprintRef["product"], g: Lattice | null | undefined) => {
      const c = g ? footprintCell(g, selPoint.lat, selPoint.lon) : null;
      return c && g ? { ref: { product, ...c }, g } : null;
    };
    if (selection?.kind === "cell") return at("charm", pointReads.forecast?.grid);
    if (active.kind === "currents") return at("hfr2km", (shownCur?.layer ?? curLayer)?.vectors?.u_grid);
    if (satForFp) {
      const src = satForFp.multisensor ? (multiSrc?.key === multiKey ? multiSrc.layer : undefined) : satForFp;
      const p = src ? productOf(src.layer_id) : null;
      if (src && p) return at(p, src.grid);
      // the multi-sensor pick is still being read: the footprint the link carried, if it holds this point
      if (src === undefined && fpFromUrl && (fpFromUrl.product === "olci300" || fpFromUrl.product === "viirs750")) {
        const u = at(fpFromUrl.product, latticeOf(manifest, fpFromUrl.product));
        if (u && u.ref.row === fpFromUrl.row && u.ref.col === fpFromUrl.col) return u;
      }
    }
    return null;
  }, [selPoint, selection?.kind, pointReads.forecast, active.kind, shownCur, curLayer, satForFp, multiSrc, multiKey, fpFromUrl, manifest]);
  const fpEnc = encodeFootprint(fp?.ref ?? null);
  const footprint = useMemo<FootprintOverlay | null>(
    () => (fp ? { key: `${encodeFootprint(fp.ref)}@${encodeSelection(selection)}`, polygon: footprintPolygon(fp.g, fp.ref.row, fp.ref.col), tone: fp.ref.product === "charm" ? "model" : "observation" } : null),
    [fp, selection],
  );
  useEffect(() => {
    fpEncRef.current = fpEnc;
    writeUrlRef.current();
  }, [fpEnc]);

  // this view's own address (layer and selection, no camera): where Research's back link returns
  const here = useMemo(() => {
    const q = new URLSearchParams();
    if (choice) for (const [k, v] of Object.entries(choiceParams(choice))) q.set(k, v);
    const sel = encodeSelection(selection);
    if (sel) q.set("sel", sel);
    if (sel && fpEnc) q.set("fp", fpEnc);
    const f = q0.get("from");
    if (f?.startsWith("/coast")) q.set("from", f);
    const s = q.toString().replace(/%3A/g, ":").replace(/%2C/g, ",");
    return s ? `/?${s}` : "/";
  }, [choice, selection, q0, fpEnc]);
  const ctx: InspectCtx = { manifest, baseUrl, ports, portIntel, stations, now, variable, lead, layer: active.kind, onSelect: selectAndGo, official, reads: pointReads, onLayer: setChoiceRaw, derived: derivedCtx, waterAt: waterAtPoint, from: q0.get("from"), here };
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
  // a harvest site chosen in search: its point, named
  const siteHere = selection?.kind === "point" ? (sitePlaces().find((p) => Math.abs(p.lat - selection.lat) < 1e-4 && Math.abs(p.lon - selection.lon) < 1e-4) ?? null) : null;
  // v10: where a point is, in words (direction A's probe header): its region, its coordinates and the nearest harbour
  const pointWhere = useMemo(() => {
    if (selection?.kind !== "point" || !ports) return null;
    const { lat, lon } = selection;
    const rid = regionAt(ports, lat, lon);
    const region = rid ? ports.regions?.find((r) => r.id === rid)?.label : null;
    let best: { name: string; km: number; lat: number; lon: number } | null = null;
    for (const f of ports.features) {
      const [plon, plat] = f.geometry.coordinates as number[];
      const km = distanceKm(lat, lon, plat, plon);
      if (!best || km < best.km) best = { name: f.properties.display_name ?? f.properties.name, km, lat: plat, lon: plon };
    }
    const at = `${lat.toFixed(3)}°N ${Math.abs(lon).toFixed(3)}°W`;
    return { title: region ? `A point in ${region}` : "A point in the water", sub: best && best.km < 80 ? `${at} · ${kmLabel(best.km)} ${bearingName(best.lat, best.lon, lat, lon)} of ${best.name}` : at };
  }, [selection, ports]);
  // v7: the panel is concise (title, one primary action, at most three facts); the rest of each view is
  // in its "Evidence and sources" fold, mounted
  const inspection = selection ? (
    <div ref={panelRef} tabIndex={-1} className="flex flex-col gap-4 outline-none" data-testid="inspect-panel" data-kind={selection.kind}>
      {packNote}
      {selection.kind === "harbor" && (
        <>
          <Header icon="harbor" kind="Harbour" title={placeName ?? "Harbour"} sub={harborF?.properties.county ? `${harborF.properties.county} County` : undefined} onClose={() => select(null)} onBack={back} canBack={stack.length > 0} />
          <HarborView code={selection.port} ctx={ctx} concise evidenceFirst={bp === "phone"} />
        </>
      )}
      {selection.kind === "station" && (
        <>
          <Header icon="station" kind="Shore station" title={station?.name ?? selection.id} sub={`CalHABMAP · SCCOOS${stationLast ? ` · last sampled ${dayLabel(stationLast, false, now)}` : ""}`} onClose={() => select(null)} onBack={back} canBack={stack.length > 0} />
          <StationView id={selection.id} ctx={ctx} concise />
        </>
      )}
      {selection.kind === "cell" && (
        <>
          <Header icon="cell" kind="C-HARM model cell" title={VARIABLE_WORDS[variable].name} sub={cellRead?.status === "ready" && cellRead.harbor ? `${kmLabel(cellRead.harbor.km)} from ${cellRead.harbor.name}` : undefined} onClose={() => select(null)} onBack={back} canBack={stack.length > 0} />
          <CellBody read={cellRead} ctx={ctx} variable={variable} at={{ lat: selection.lat, lon: selection.lon }} concise />
        </>
      )}
      {selection.kind === "point" && (
        <>
          <Header icon="locate" kind={siteHere ? "Harvest site" : "A point in the water"} title={siteHere?.name ?? pointWhere?.title ?? "This point"} sub={siteHere?.note ?? pointWhere?.sub} onClose={() => select(null)} onBack={back} canBack={stack.length > 0} />
          <PointView lat={selection.lat} lon={selection.lon} ctx={ctx} site={siteHere} concise />
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

  const steps = timeSteps(manifest, active, now, derived, hourRegion);
  const hasTime = active.kind !== "none" && !!steps?.steps.length;
  // nothing current: say so, then offer what there is, each dated and marked old, never shown as current
  const satDate = opening?.reason === "nothing-current" && opening.satellite ? (opening.satellite.layer.composite?.newest_observed_date ?? opening.satellite.layer.time.observed_date ?? null) : null;
  // v10 rework: one line in the time rail's status row, and the two dated alternatives beside it as choices.
  // Neither is ever shown as current: each says its own date and age, and opens marked old.
  const satAge = satDate && now ? ageInDays(satDate.slice(0, 10), now) : null;
  const runAge = run && now ? ageInDays(run.issued_date, now) : null;
  const restCls = "nx-press nx-focus inline-flex min-h-10 items-center gap-2 rounded-[3px] border px-3 py-1 text-left text-[13px] leading-tight text-n-ink";
  const nothingCurrent =
    active.kind === "none" && opening?.reason === "nothing-current" ? (
      <div className="flex min-w-0 flex-col gap-2" data-testid="nothing-current">
        <p className="flex min-w-0 items-start gap-2 text-[12.5px] leading-snug text-n-ink-2" data-testid="nothing-current-line">
          <Icon name="clock" className="mt-px h-4 w-4 shrink-0 text-n-stale" />
          <span>
            <span className="font-semibold text-n-ink">No current forecast or recent satellite view</span>
            {" · newest: "}
            {run ? `C-HARM ${dayLabel(run.issued_date)}` : "no C-HARM run"}
            {" · "}
            {satDate ? `satellite ${dayLabel(satDate)}` : "no satellite view"}
            <span className="text-n-ink-3"> · this does not mean conditions are normal</span>
          </span>
        </p>
        {(run || (opening.satellite && satDate)) && (
          <div className="flex flex-wrap items-center gap-2" role="group" aria-label="The newest views, each dated">
            {opening.satellite && satDate && (
              <button type="button" onClick={() => setChoiceRaw({ kind: "satellite", product: opening.satellite!.product, day: null })} className={`${restCls} border-[color-mix(in_srgb,var(--nx-measured)_55%,transparent)] hover:bg-n-surface-2`} data-testid="show-latest-satellite">
                <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-[1px] bg-n-measured" />
                <span>
                  Show satellite chlorophyll · {dayLabel(satDate)}
                  {satAge != null ? <span className="text-n-stale"> · {satAge === 1 ? "1 day" : `${satAge} days`} old</span> : null}
                </span>
              </button>
            )}
            {run && (
              <button type="button" onClick={() => setChoiceRaw({ kind: "forecast", variable: "particulate_domoic", lead: firstLead < 0 ? 0 : firstLead })} className={`${restCls} border-[color-mix(in_srgb,var(--nx-model)_55%,transparent)] hover:bg-n-surface-2`} data-testid="show-latest-forecast">
                <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-[1px] bg-n-model" />
                <span>
                  Show the {dayLabel(run.issued_date)} forecast
                  <span className="text-n-stale"> · out of date{runAge != null ? `, ${runAge === 1 ? "1 day" : `${runAge} days`} old` : ""}</span>
                </span>
              </button>
            )}
          </div>
        )}
      </div>
    ) : null;
  // a data layer at state scale: its 3 km cells are a few pixels wide, so say how to read one (the first
  // thing a new reader needs: it sits on the dock, and goes once the map is zoomed in or a place is open)
  const readHint =
    (active.kind === "forecast" || active.kind === "satellite" || active.kind === "outlook" || active.kind === "sst") && !selection && zoom != null && zoom < 7 ? (
      <p className="flex flex-wrap items-center gap-x-2 text-[13px] leading-snug text-n-ink-2" data-testid="read-hint">
        <span>
          Tap a harbour for its briefing, or zoom in and tap the water to read a {active.kind === "forecast" ? "model cell" : "pixel"}.
        </span>
        <button
          type="button"
          onClick={() => mapRef.current && journeyTo(mapRef.current, { bounds: (VIEW.monterey_bay ?? [[-122.3, 36.5], [-121.7, 37.05]]) as [[number, number], [number, number]], padding: padding() })}
          className="nx-focus -my-2 inline-flex min-h-11 items-center font-semibold text-n-accent underline underline-offset-2"
          data-testid="zoom-monterey"
        >
          Go to Monterey Bay
        </button>
      </p>
    ) : null;

  // ------------------------------------------------------------------ the layer index (v10)
  // phone sheet
  const [snap, setSnap] = useState<SnapId>("peek");
  const [vh, setVh] = useState(800);
  const sheetHalf = Math.round((vh - DOCK) * 0.5);
  const [vw, setVw] = useState(390);
  useEffect(() => {
    const f = () => setVw(window.innerWidth);
    f();
    window.addEventListener("resize", f);
    return () => window.removeEventListener("resize", f);
  }, []);
  const peekH = atRest && !selection ? peekRest(vw) : PEEK;
  useEffect(() => {
    const f = () => setVh(window.innerHeight);
    f();
    window.addEventListener("resize", f);
    return () => window.removeEventListener("resize", f);
  }, []);
  // phone: the credits chip on the map edge, just above the sheet
  const creditsRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => setSnap(selection ? "half" : "peek"), [selection]);
  const wide = bp === "desktop" || bp === "tablet";
  // a short wide window (a landscape phone, a small laptop window): the index folds behind the layer
  // button, which opens it over the map; everywhere else on a wide screen it is always in view
  const shortWide = wide && vh < 700;
  const folded = bp === "phone" || shortWide;
  const deckRef = useRef<HTMLElement | null>(null);
  const capsuleRef = useRef<HTMLButtonElement | null>(null);
  const toggleDeck = () => {
    setDeckOpen((o) => {
      // on a phone the list opens in the sheet, which rises to show it
      if (!o && bp === "phone") setSnap("full");
      return !o;
    });
  };
  // folded: a tap outside the list, or Escape, closes it (Escape is taken here, before it would close the place)
  useEffect(() => {
    if (!deckOpen || !folded) return;
    const k = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      e.preventDefault();
      setDeckOpen(false);
      capsuleRef.current?.focus({ preventScroll: true });
    };
    const c = (e: PointerEvent) => {
      const t = e.target as Node;
      if (deckRef.current?.contains(t) || capsuleRef.current?.contains(t)) return;
      setDeckOpen(false);
    };
    window.addEventListener("keydown", k, true);
    window.addEventListener("pointerdown", c);
    return () => {
      window.removeEventListener("keydown", k, true);
      window.removeEventListener("pointerdown", c);
    };
  }, [deckOpen, folded]);
  // choosing another kind of layer: on a phone the list closes and the sheet shows the layer's time and legend
  const pickLayer = (c: LayerChoice) => {
    setChoiceRaw(c);
    if (bp === "phone" && c.kind !== active.kind) {
      setDeckOpen(false);
      setSnap("half");
    }
  };
  // how the field is drawn, one level in: kept open once opened (per page view)
  const [adjustOpen, setAdjustOpen] = useState(false);
  const [opacity, setOpacity] = useState(1);
  const lw = layerWords(active);
  const layerButton = (
    <button
      ref={capsuleRef}
      type="button"
      aria-expanded={deckOpen}
      aria-controls="layer-deck"
      aria-label={`Map layer: ${[lw.title, lw.detail].filter(Boolean).join(", ")} (${lw.kind}). ${deckOpen ? "Close the layer list" : "Change the layer"}`}
      onClick={toggleDeck}
      className={`nx-press nx-focus flex min-h-12 w-full min-w-0 items-center gap-3 rounded-[4px] px-2.5 py-1 text-left hover:bg-n-surface-2 ${deckOpen ? "bg-n-surface-2" : ""}`}
      data-testid="layer-capsule"
    >
      <span aria-hidden className={`h-2.5 w-2.5 shrink-0 rounded-[1px] ${lw.hatch ? "nx-index-sw nx-index-sw-hatch" : "nx-index-sw"}`} style={{ ["--tone" as string]: lw.sw }} />
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[15px] leading-tight text-n-ink">{lw.title}</span>
        <span className="block truncate text-[12px] leading-snug text-n-ink-2">
          <span className={lw.tone}>{lw.kind}</span>
          {lw.detail ? ` · ${lw.detail}` : ""}
        </span>
      </span>
      {bp !== "phone" && <Icon name="chevronDown" className={`h-4 w-4 shrink-0 text-n-ink-3 transition-transform ${deckOpen ? "rotate-180" : ""}`} />}
    </button>
  );
  const flowPause = active.kind === "currents" ? <FlowControls part="pause" prefs={flowPrefs} onPrefs={setFlowPrefs} reduced={reduced} /> : null;
  const flowView = active.kind === "currents" ? <FlowControls part="view" prefs={flowPrefs} onPrefs={setFlowPrefs} reduced={reduced} /> : null;
  const adjustable = active.kind !== "none";
  const indexBody = (
    <div className="flex flex-col gap-3">
      <LayerList manifest={manifest} choice={active} onChoice={pickLayer} now={now} defaults={{ lead: firstLead }} derived={derived} />
      <LayerOptions manifest={manifest} choice={active} onChoice={setChoiceRaw} estimate={resolved.estimateLayer} part="primary" />
      {adjustable && (
        <details className="nx-note group border-t border-n-line" data-testid="layer-adjust" open={adjustOpen} onToggle={(e) => setAdjustOpen(e.currentTarget.open)}>
          <summary className="nx-focus flex min-h-11 cursor-pointer items-center gap-2">
            <span className="min-w-0 flex-1 text-[13.5px] text-n-ink">Display</span>
            <span className="nx-data text-[11px] text-n-ink-3">{active.kind === "currents" && !active.chl ? (arrows ? "arrows" : "streaks") : `${Math.round(opacity * 100)}%`}</span>
            <Icon name="chevronDown" className="nx-note-chev h-4 w-4 shrink-0 text-n-ink-3" />
          </summary>
          <div className="nx-note-body pb-2 pt-1">
            <LayerOptions manifest={manifest} choice={active} onChoice={setChoiceRaw} estimate={resolved.estimateLayer} part="adjust" opacity={opacity} onOpacity={setOpacity} flow={flowView} />
          </div>
        </details>
      )}
      {/* the science and the full credits, on demand: provenance, coverage, method and skill, reference lines */}
      <details className="nx-note group border-t border-n-line" data-testid="layer-about">
        <summary className="nx-focus flex min-h-11 cursor-pointer items-center gap-2">
          <span className="min-w-0 flex-1 text-[13.5px] text-n-ink">About this layer</span>
          <Icon name="chevronDown" className="nx-note-chev h-4 w-4 shrink-0 text-n-ink-3" />
        </summary>
        <div className="nx-note-body flex flex-col gap-3 pb-2">
          <Provenance manifest={manifest} choice={active} resolved={shownResolved} now={now} regions={ports?.regions ?? []} part="about" />
          <div className="border-t border-n-line pt-3">
            <ReferenceLines />
          </div>
          <div className="border-t border-n-line pt-3">
            <Attribution />
          </div>
        </div>
      </details>
    </div>
  );

  // a data layer that could not be drawn: said on the map, with a way to try again (never a silent gap)
  const satFailed = mapStatus?.field === "failed" && (active.kind === "satellite" || (active.kind === "currents" && !!active.chl));
  const fieldProblem =
    mapStatus?.field === "failed" || (active.kind === "currents" && curState === "failed") ? (
      <div role="status" className="nx-iplate flex w-full max-w-[420px] items-center gap-2.5 py-1 pl-3.5 pr-1.5 text-n-ink" data-testid="field-failed">
        <Icon name="alert" className="h-5 w-5 shrink-0 text-n-down" />
        <span className="min-w-0 flex-1 text-[14px] leading-snug">
          {active.kind === "currents" && curState === "failed" ? "These currents could not be loaded." : satFailed ? "Part of the satellite view could not be loaded." : "Part of the C-HARM forecast could not be loaded."} Nothing is drawn in its place.
        </span>
        <button type="button" onClick={active.kind === "currents" && curState === "failed" ? retryCurrents : retryField} className="nx-press nx-focus inline-flex min-h-11 shrink-0 items-center rounded-[3px] px-3 text-[14px] font-semibold text-n-accent hover:bg-n-surface-2" data-testid="field-retry">
          Try again
        </button>
      </div>
    ) : null;
  // the field's tiles still arriving (after the map itself is up): said in place, beside its legend
  const fieldLoading = mapReady && mapStatus?.field === "loading" ? (
    <p role="status" className="cw-late nx-data text-[11px] text-n-ink-3" data-testid="field-loading">
      Reading the published tiles…
    </p>
  ) : null;

  // the map's tools, one column: the zoom pair, then recentre, area, share and the field pack
  const stackCls = "pointer-events-auto flex flex-col divide-y divide-n-line rounded-[4px] border border-n-line bg-[rgba(5,9,13,0.88)]";
  const toolCls = "nx-press nx-focus grid h-11 w-11 place-items-center text-n-ink-2 hover:text-n-ink max-md:h-14 max-md:w-14";
  const recenter = (
    <button type="button" aria-label={selection ? "Centre on the selection" : "Show the whole coast"} onClick={() => (selection ? focusOn(selection) : fitState())} className={toolCls} data-testid="recenter">
      <Icon name="locate" />
    </button>
  );
  const packButton = (
    <button type="button" aria-label={`Field pack: save ${pack.name} for offline use`} onClick={() => setPackOpen(true)} className={toolCls} data-testid="pack-open">
      <Icon name={off.record?.complete ? "check" : "download"} />
    </button>
  );
  // with the inspector open on a wide screen the tools lie in one row under it, off the coast beside it
  // (they sat over the shore and its labels, between the field and the panel)
  const toolRow = panelOpen && (bp === "desktop" || bp === "tablet");
  const rowCls = toolRow ? "flex-row divide-x divide-y-0" : "";
  const mapControls = (
    <div className={`pointer-events-auto flex gap-2 ${toolRow ? "flex-row" : "flex-col"}`} data-testid="map-controls" data-row={toolRow ? "true" : undefined}>
      <div className={`${stackCls} ${rowCls} [@media(max-height:560px)]:hidden`}>
        <button type="button" aria-label="Zoom in" onClick={() => mapRef.current?.zoomIn({ duration: 200 })} className={toolCls}>
          <Icon name="plus" />
        </button>
        <button type="button" aria-label="Zoom out" onClick={() => mapRef.current?.zoomOut({ duration: 200 })} className={toolCls}>
          <Icon name="minus" />
        </button>
      </div>
      <div className={`${stackCls} ${rowCls}`}>
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
      outlook={resolved.outlook}
      sst={resolved.sst}
      estimate={resolved.estimate}
      contextFill
      currents={currents}
      ports={ports}
      showPorts
      officialGeometry={officialGeometry}
      showOfficial
      selectedPort={selection?.kind === "harbor" ? selection.port : null}
      inspect={null}
      chrome="none"
      basemapPmtiles={basemapPm ?? null}
      clusterStations
      highlight={shownHighlight}
      footprint={dragBox ? null : footprint}
      onStatus={onStatus}
      retry={fieldRetry}
      fieldOpacity={opacity}
      initialBounds={initialBounds === STATEWIDE.bounds ? COAST_BOUNDS : initialBounds}
      initialPadding={{ top: 84, bottom: 150, left: 24, right: 24 }}
      cellEdges={active.kind === "forecast" && edges && edges.id === resolved.forecastLayer?.layer_id ? edges.fc : null}
      cellLabels={harbourOn ? harbour!.labels : null}
      gap={harbourOn && !selection ? harbour!.gap : null}
      stations={stationPoints}
      onHover={bp !== "phone" && canHover && !areaMode ? onHover : undefined}
      onPort={(code) => select({ kind: "harbor", port: code })}
      onStationPick={(id) => select({ kind: "station", id })}
      onPoint={(p: MapPoint) => select(active.kind === "forecast" ? { kind: "cell", lat: p.lat, lon: p.lon } : { kind: "point", lat: p.lat, lon: p.lon })}
      onReady={(m) => {
        mapRef.current = m;
        setMapReady(true);
        if (initialAt && initialAt.length === 3 && initialAt.every(Number.isFinite)) m.jumpTo({ center: [initialAt[1], initialAt[0]], zoom: initialAt[2] });
        else {
          if (initialBounds === STATEWIDE.bounds) m.jumpTo(coastView());
          else m.fitBounds(initialBounds, { padding: padding(), duration: 0 });
          const s = decodeSelection(q0);
          if (s) m.once("idle", () => focusOn(s));
        }
        // through the ref: a handler captured here would write the selection of the first render
        m.on("moveend", () => writeUrlRef.current());
        // the region in view (regional zoom): the currents open on the newest hour that observed it
        const inView = () => {
          const c = m.getCenter();
          const r = m.getZoom() >= REGION_ZOOM ? regionAt(ports, c.lat, c.lng) : null;
          setViewRegion((was) => (was === r ? was : r));
        };
        m.on("moveend", inView);
        inView();
        m.on("zoomend", () => setZoom(m.getZoom()));
        setZoom(m.getZoom());
      }}
    />
  );
  const writeUrlRef = useRef(writeUrl);
  writeUrlRef.current = writeUrl;

  const starting = !mapReady;
  // under the search box (desktop and tablet in the top bar; phones under the disclosure line): the way
  // back to a briefing, the offline state and a layer that failed, one under another and never over each other
  const offlineBanner = !off.online ? <OfflineBanner record={off.record ?? null} problem={off.problem} name={pack.name} inline /> : null;
  const fixture = isFixture(manifest);
  // the tools' right edge: clear of the panel when one is open
  const toolsRight = 16;
  // the layer's credit beside the map's (Copernicus asks that modified Sentinel data be said wherever it is shown)
  const sentinel = (active.kind === "satellite" && active.product !== "viirs750") || (active.kind === "currents" && !!active.chl) ? (resolved.satLayer?.time.observed_date ?? manifest.generated_at).slice(0, 4) : null;
  const layerCredit = sentinel ? `Contains modified Copernicus Sentinel data ${sentinel}` : null;
  // the time rail: the layer's own time, the other layers' frames as context, and its date line under it
  // the status row keeps the tallest height it has had for this layer: playing the currents adds and removes
  // lines (the hour note, a region the radar missed), and the rail must not jump with them. For the currents
  // three lines are reserved from the start (the stamp, a region the radar missed, an older-than-usual hour):
  // an hour that turns stale during playback folds into the reserved row (v10 fix 2)
  const railCaption = (
    <StableCaption key={active.kind} min={active.kind === "currents" ? 56 : 0}>
      {nothingCurrent}
      <div className="flex min-w-0 flex-wrap items-start gap-x-5 gap-y-0.5 empty:hidden [&_[data-testid=provenance]]:gap-0.5 [&_[data-testid=provenance]_p]:text-[11.5px] [&_[data-testid=provenance]_p]:leading-[16px] [&_[data-testid=currents-hour-note]]:text-[11.5px] [&_[data-testid=currents-hour-note]]:leading-[16px]">
        <Provenance manifest={manifest} choice={active} resolved={shownResolved} now={now} regions={ports?.regions ?? []} part="dock" />
        {hourNote}
      </div>
    </StableCaption>
  );
  const rail = (compact: boolean) => (
    <TimeRail
      manifest={manifest}
      choice={active}
      onChoice={setChoiceRaw}
      now={now}
      derived={derived}
      region={hourRegion}
      shown={shownCur ? hourKey(shownCur.layer) : null}
      pending={curPending && curLayer ? hourKey(curLayer) : null}
      onPrefetch={prefetch}
      compact={compact}
      caption={compact ? null : railCaption}
      variable={variable}
      lead={firstLead}
    />
  );
  const legendBody = (
    <div ref={legendRef} className="flex min-w-0 flex-col gap-2.5" data-testid="map-legend">
      <Legend choice={active} resolved={shownResolved} check={check} flow={flowLegend} />
      {flowPause}
      {fieldLoading}
      {readHint}
    </div>
  );
  const colW = bp === "desktop" ? 264 : 236;
  // clip, not hidden: a hidden overflow can still be scrolled by focus or scrollIntoView, which slid
  // the whole workspace (and the dock) up the screen on phones
  return (
    <div className="env-instrument relative h-dvh w-full overflow-clip" data-testid="explore">
      <SkipLink to="place-search" label="Skip to search places" />
      <h1 className="sr-only">Explore: harmful algal bloom evidence on the California coast{placeName ? `, ${placeName}` : ""}</h1>
      <div className="absolute inset-0" data-testid="map">
        {basemapPm !== undefined && synced ? map : null}
      </div>
      {/* the field darkens toward the instrument's edges (its words keep their contrast over any raster) */}
      {wide && <div aria-hidden className="nx-field-shade pointer-events-none absolute inset-0 z-[4]" />}
      {hoverSource && bp !== "phone" && <HoverLayer sink={hoverSink} manifest={manifest} baseUrl={baseUrl} source={hoverSource} inset={panelOpen ? panelW + 32 : 0} waterAt={waterAt} />}
      {packOpen && <PackDialog pack={pack} manifest={manifest} baseUrl={baseUrl} official={official} onClose={() => setPackOpen(false)} />}
      {/* the start state: the seafloor first, and what is being read (never a bare spinner) */}
      <AnimatePresence>{starting && <StartCard key="start" now={now} manifest={manifest} phone={bp === "phone"} />}</AnimatePresence>
      {areaMode && (
        <div {...areaHandlers} className="absolute inset-0 z-[5] cursor-crosshair touch-none" data-testid="area-overlay">
          <p className="nx-iplate pointer-events-none absolute left-1/2 top-24 -translate-x-1/2 px-4 py-2 text-[14px] text-n-ink">Drag across the water to select an area · Esc to cancel</p>
        </div>
      )}

      {/* the header row: brand and workspaces, the search, the notices, status and more; nothing else floats up here */}
      <TopBar
        active="explore"
        now={now}
        floating
        slot={
          bp ? (
            <div className="flex flex-col items-start gap-1.5">
              <PlaceSearch ports={ports} stations={stations} regions={regions} onPick={onSearch} current={placeName ?? ""} />
              {wide && offlineBanner && <div className="w-full max-w-[420px]">{offlineBanner}</div>}
              {wide && fieldProblem}
              {wide && backTo && <BackToBriefing {...backTo} className="w-fit max-w-full" />}
            </div>
          ) : null
        }
        notices={<NoticesButton official={official} status={officialStatus} compact={bp === "phone"} local={harbourNotices ?? pointNotices} />}
      />

      {wide && (
        <>
          {/* the instrument's left margin: the layer index (always in view) and, under it, the legend */}
          <div className="pointer-events-none absolute left-4 z-20 flex flex-col gap-3" style={{ top: 72, bottom: marginH + dockH + 14, width: colW }}>
            {shortWide ? (
              <div className="nx-iplate pointer-events-auto relative shrink-0 p-1">
                {layerButton}
                <section
                  ref={deckRef}
                  id="layer-deck"
                  aria-label="Map layers"
                  hidden={!deckOpen}
                  className="nx-iplate absolute left-0 top-[calc(100%+8px)] z-30 w-[min(320px,calc(100vw-32px))] overflow-y-auto overscroll-contain p-3 [scrollbar-width:thin]"
                  style={{ maxHeight: `calc(100dvh - ${72 + 60 + marginH + dockH + 24}px)` }}
                  data-testid="layer-deck"
                >
                  {indexBody}
                  <div className="mt-3 border-t border-n-line pt-3">{legendBody}</div>
                </section>
              </div>
            ) : (
              <section ref={deckRef} id="layer-deck" aria-label="Map layers" className="nx-iplate pointer-events-auto min-h-0 shrink overflow-y-auto overscroll-contain px-3.5 pb-1.5 pt-3 [scrollbar-width:thin]" data-testid="layer-deck">
                <p className="nx-caps pb-1.5">Field</p>
                {indexBody}
              </section>
            )}
            {!shortWide && (
              <section aria-label="Legend" className="nx-iplate pointer-events-auto mt-auto shrink-0 px-3.5 py-3">
                {legendBody}
              </section>
            )}
          </div>

          {/* the selection: right, top-aligned under the header, never down over the time rail */}
          <AnimatePresence>
            {inspection && (
              <InspectAside key="inspect" className="nx-iplate absolute right-4 top-[68px] z-20 overflow-y-auto overscroll-contain p-5 text-n-ink [scrollbar-width:thin]" style={{ width: panelW, maxHeight: `calc(100% - ${68 + marginH + dockH + 18 + 56}px)` }}>
                {inspection}
              </InspectAside>
            )}
          </AnimatePresence>

          <div className="absolute z-20" style={{ right: toolsRight, bottom: marginH + dockH + 14 }}>
            {mapControls}
          </div>

          {/* the time rail: full width along the bottom, a first-class instrument */}
          <div ref={dockRef} className="pointer-events-none absolute inset-x-4 z-20" style={{ bottom: marginH + 2 }}>
            <section aria-label="Map time" className="pointer-events-auto relative pt-1" data-testid={hasTime ? "timebar-deck" : "instrument-dock"}>
              {rail(false)}
            </section>
          </div>

          {/* the map's bottom margin: the one disclosure line and the credits, on the map, never covered */}
          <div ref={marginRef} className="pointer-events-none absolute inset-x-0 bottom-0 z-20 flex flex-wrap items-center justify-between gap-x-6 px-5 pb-1.5 pt-1" data-testid="map-margin">
            <RegistryDisclosure tone="map" fixture={fixture} className="pointer-events-auto" />
            <MapCredits className="pointer-events-auto" extra={layerCredit} />
          </div>
        </>
      )}

      {bp === "phone" && (
        <>
          {/* one line under the header: the disclosure; then the offline state, a failed layer, the way back */}
          <div className="pointer-events-none absolute inset-x-0 top-[72px] z-20 flex flex-col items-start gap-1.5 px-4">
            <RegistryDisclosure tone="map" fixture={fixture} className="pointer-events-auto max-w-full" />
            {offlineBanner && <div className="pointer-events-auto w-full">{offlineBanner}</div>}
            {fieldProblem && <div className="pointer-events-auto w-full">{fieldProblem}</div>}
            {backTo && <BackToBriefing {...backTo} className="pointer-events-auto w-fit max-w-full" />}
            {/* a place open: the sheet leads with it, and the layer is a small chip on the map, still one tap away */}
            {selection && (
              <button
                ref={capsuleRef}
                type="button"
                aria-expanded={deckOpen}
                aria-controls="layer-deck"
                aria-label={`Map layer: ${[lw.title, lw.detail].filter(Boolean).join(", ")} (${lw.kind}). ${deckOpen ? "Close the layer list" : "Change the layer"}`}
                onClick={toggleDeck}
                className="nx-press nx-focus nx-iplate pointer-events-auto inline-flex min-h-11 max-w-full items-center gap-2 px-3 text-[13px] text-n-ink"
                data-testid="layer-capsule"
              >
                <span aria-hidden className={`h-2.5 w-2.5 shrink-0 rounded-[1px] ${lw.hatch ? "nx-index-sw nx-index-sw-hatch" : "nx-index-sw"}`} style={{ ["--tone" as string]: lw.sw }} />
                <span className="truncate">{lw.title}</span>
                <Icon name="layers" className="h-4 w-4 shrink-0 text-n-ink-2" />
              </button>
            )}
          </div>
          {/* the sheet ends above the dock; the map stays live above it (non-modal) */}
          <div className="pointer-events-none absolute inset-x-0 top-0 z-30 overflow-clip" style={{ bottom: `calc(${DOCK}px + env(safe-area-inset-bottom, 0px))` }}>
            {/* the map's credits (ODbL) on the map's edge, one line just above the sheet at every height
                the map shows; at full height (the map all but covered) they ride on the sheet's edge */}
            {snap !== "full" && (
              <div ref={creditsRef} className="pointer-events-none absolute left-2 right-2 flex items-end justify-between gap-2 transition-[bottom] duration-200" style={{ bottom: (snap === "peek" ? peekH + 8 : sheetHalf + 4) }}>
                <div className="pointer-events-none min-w-0 flex-1 [&>*]:pointer-events-auto">
                  <CreditsChip extra={layerCredit} />
                </div>
                {/* the tools docked at the right edge just above the sheet, beside the credits (v10 fix 2): off the
                    coast the frame is about, which runs from the upper left to the lower right of a phone. Locate
                    stays one tap away; sharing and the field pack fold behind "More" */}
                {snap === "peek" && <PhoneTools recenter={recenter} packButton={packButton} stackCls={stackCls} toolCls={toolCls} />}
              </div>
            )}
            <div className="pointer-events-auto">
              <MobileSheet
                tone="instrument"
                snaps={[
                  { id: "peek", height: peekH },
                  { id: "half", height: sheetHalf },
                  { id: "full", height: Math.round((vh - DOCK) * 0.9) },
                ]}
                snap={snap}
                onSnap={setSnap}
                label={selection ? "Selected" : "Layer, legend and time"}
                header={null}
                field
                edge={snap === "full" ? <MapCredits short extra={layerCredit} /> : null}
              >
                <div className="pb-6 text-n-ink">
                  {/* the peek is the dock: the layer button, the ramp, the time rail */}
                  {/* a selection open: the place leads, the layer row steps aside (one header, not two stacked) */}
                  {!selection && <div className="-mx-1.5 pr-[52px]">{layerButton}</div>}
                  <section ref={deckRef} id="layer-deck" aria-label="Map layers" hidden={!deckOpen} className="mt-2 border-t border-n-line pt-3" data-testid="layer-deck">
                    {indexBody}
                  </section>
                  {inspection ? (
                    // the sheet's own resize handle sits at its top right: the place's close button keeps clear of it
                    <div className="pt-1 [&_header]:pr-12">{inspection}</div>
                  ) : (
                    <div ref={legendPhoneRef} className="mt-1 flex flex-col gap-2" data-testid="active-layer">
                      {nothingCurrent ? (
                        <div className="px-1">{nothingCurrent}</div>
                      ) : (
                        <div className="px-1">
                          <LegendLine choice={active} resolved={shownResolved} flow={flowLegend} />
                        </div>
                      )}
                      {rail(true)}
                      <div className="mt-2 flex flex-col gap-3 border-t border-n-line pt-3">
                        {readHint}
                        {flowPause}
                        {fieldLoading}
                        <Legend choice={active} resolved={shownResolved} check={check} inline noRamp flow={flowLegend} />
                        <Provenance manifest={manifest} choice={active} resolved={shownResolved} now={now} regions={ports?.regions ?? []} part="dock" />
                        {hourNote}
                      </div>
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
 * on the map's bottom margin (desktop, tablet), on the map's edge above the sheet (phones), with the
 * layer's own credit when its licence asks for one (Copernicus Sentinel). The full list is in the
 * layer index's "About this layer".
 */
function MapCredits({ short = false, className = "", extra = null }: { short?: boolean; className?: string; extra?: string | null }) {
  return (
    <p
      // the short line keeps a backing (its contrast must hold over any raster); smaller, so it stays on one line
      className={`max-w-full [&_a]:underline [&_a]:decoration-white/40 [&_a]:underline-offset-2 ${short ? "w-fit rounded-[3px] bg-[rgba(5,9,13,0.78)] px-1.5 text-[10.5px] leading-[15px] text-[#c9d6e2]" : "nx-on-map text-[11px] leading-[16px]"} ${className}`}
      data-testid={short ? "attribution-short" : "attribution-line"}
    >
      <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">© OpenStreetMap contributors</a> · <a href="https://www.openmaptiles.org/" target="_blank" rel="noreferrer">© OpenMapTiles</a> · NOAA relief{extra ? ` · ${extra}` : ""} · not for navigation
    </p>
  );
}

/**
 * Phones (v10 rework): the map's credits fold to a chip on the map's edge that still names OpenStreetMap;
 * a tap opens the whole line (every credit, its links and "not for navigation"), a second tap folds it.
 */
function CreditsChip({ extra = null }: { extra?: string | null }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex w-fit max-w-full flex-col items-start gap-1" data-testid="credits-chip-wrap">
      {open && <MapCredits short extra={extra} />}
      <button
        type="button"
        aria-expanded={open}
        aria-label={open ? "Hide the map credits" : "Map credits: © OpenStreetMap contributors, © OpenMapTiles, NOAA relief"}
        onClick={() => setOpen((o) => !o)}
        className="nx-focus inline-flex min-h-10 shrink-0 items-center"
        data-testid="credits-chip"
      >
        <span className="inline-flex items-center gap-1 rounded-full bg-[rgba(5,9,13,0.78)] px-1.5 text-[10.5px] leading-[18px] text-[#c9d6e2]">
          <Icon name={open ? "close" : "info"} className="h-3 w-3" />
          {open ? "" : "© OpenStreetMap"}
        </span>
      </button>
    </div>
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

/** The phone's map tools: locate, and "More" holding share and the field pack (56 px targets). */
function PhoneTools({ recenter, packButton, stackCls, toolCls }: { recenter: React.ReactNode; packButton: React.ReactNode; stackCls: string; toolCls: string }) {
  const [open, setOpen] = useState(false);
  const moreRef = useRef<HTMLButtonElement | null>(null);
  return (
    <div
      className="pointer-events-auto shrink-0"
      data-testid="map-controls"
      data-row="true"
      data-docked="true"
      onKeyDown={(e) => {
        if (e.key === "Escape" && open) {
          e.stopPropagation();
          setOpen(false);
          moreRef.current?.focus();
        }
      }}
    >
      <div className={`${stackCls} flex-row divide-x divide-y-0`}>
        {open && (
          <div id="map-tools-more" className="flex flex-row divide-x divide-n-line" data-testid="map-tools-more">
            <ShareButton className={toolCls} />
            {packButton}
          </div>
        )}
        <button ref={moreRef} type="button" aria-label={open ? "Fewer map tools" : "More map tools: share, field pack"} aria-expanded={open} aria-controls="map-tools-more" onClick={() => setOpen((o) => !o)} className={toolCls} data-testid="map-tools-toggle">
          <Icon name={open ? "close" : "more"} />
        </button>
        {recenter}
      </div>
    </div>
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
          toast("Link copied");
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

function StartCard({ now, manifest, phone }: { now: Date | null; manifest: Manifest; phone: boolean }) {
  // v10 (direction A's loading screen): the seafloor and coast first, then "Reading the published layers",
  // each source in place with its own state and date. It appears only after 400 ms of real waiting and
  // steps back (a fade) as the map arrives; while leaving it is inert and has no test id. Behind it the
  // canyon's isobaths sound (loading), never a coloured sea: an empty sea here means "not loaded yet".
  const present = useIsPresent();
  const gen = manifest.generated_at;
  return (
    <motion.div
      className="pointer-events-none absolute inset-0 z-[15]"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1, transition: { delay: 0.4, duration: 0.2, ease: EASE_OUT } }}
      exit={{ opacity: 0, transition: { duration: 0.22, ease: EASE_IN } }}
      data-testid={present ? "start-card" : undefined}
      inert={!present}
      aria-hidden={!present || undefined}
    >
      <ContourLoader className="absolute inset-0 h-full w-full text-n-accent opacity-35" />
      <div className={`nx-iplate pointer-events-auto absolute overflow-y-auto overscroll-contain [scrollbar-width:thin] ${phone ? "inset-x-3 top-[112px] max-h-[calc(100dvh-112px-260px)] px-4 py-3.5" : "right-[84px] top-24 max-h-[calc(100dvh-96px-200px)] w-[400px] px-5 py-[18px]"}`}>
        <p className={`${phone ? "text-[21px]" : "text-[24px]"} font-light leading-tight text-n-ink`}>Reading the published layers</p>
        <p className="nx-data mt-1 text-[11px] text-n-ink-3">
          Data run {manifest.pipeline_run_id.slice(0, 8)} · generated {dayLabel(gen.slice(0, 10))}, {gen.slice(11, 16)} UTC
        </p>
        <p className="nx-margin mt-2">The seafloor and the coast come first. Nothing is coloured until its values are read: an empty sea here means &ldquo;not loaded yet&rdquo;, never &ldquo;low&rdquo;.</p>
        <div className="mt-2 border-t border-n-line">
          <SourceList now={now} compact />
        </div>
        <p className="mt-2 text-[12.5px] text-n-ink-2">Search a harbour, read the notices or open a briefing meanwhile.</p>
      </div>
    </motion.div>
  );
}

/**
 * The selection's panel: slides in 12 px and fades (240 ms); leaves 8 px and fades faster (160 ms),
 * inert and hidden from assistive tech while it goes. A new selection keeps the panel and only
 * cross-fades its body (Explore). Reduced motion: fades only (MotionConfig).
 */
function InspectAside({ className, style, children }: { className: string; style?: React.CSSProperties; children: React.ReactNode }) {
  const present = useIsPresent();
  return (
    <motion.aside
      aria-label="Selected"
      className={className}
      style={style}
      initial={{ opacity: 0, x: 12 }}
      animate={{ opacity: 1, x: 0, transition: { duration: T.panel, ease: EASE_OUT } }}
      exit={{ opacity: 0, x: 8, transition: { duration: T.panelExit, ease: EASE_IN } }}
      inert={!present}
      aria-hidden={!present || undefined}
    >
      {children}
    </motion.aside>
  );
}

function AreaView({ bbox, ctx }: { bbox: [number, number, number, number]; ctx: InspectCtx }) {
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
      <CrossLinks ctx={ctx} place={pointKey((s + n) / 2, (w + e) / 2)} lat={(s + n) / 2} lon={(w + e) / 2} name="the middle of this area" variant="actions" primary="brief" briefLabel="Open the briefing" />
      <AreaModel layer={layer} baseUrl={ctx.baseUrl} bbox={bbox} variable={ctx.variable} />
      <EvidenceFold summary="Shore stations and harbours in the box, the box's bounds">
      <p className="nx-data text-[12.5px] text-n-ink-3">
        {s.toFixed(3)}–{n.toFixed(3)}°N, {Math.abs(e).toFixed(3)}–{Math.abs(w).toFixed(3)}°W
      </p>
      <section className="flex flex-col gap-2 border-t border-n-line pt-4">
        <h3 className="nx-eyebrow">Shore stations in the box</h3>
        {sts.length ? (
          <ul>
            {sts.map((x) => (
              <li key={x.station_id}>
                <button type="button" onClick={() => ctx.onSelect({ kind: "station", id: x.station_id })} className="nx-focus flex min-h-12 w-full items-center justify-between rounded-lg px-1 text-left nx-row hover:bg-n-surface-2">
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
                <button type="button" onClick={() => ctx.onSelect({ kind: "harbor", port: f.properties.port_code })} className="nx-focus flex min-h-12 w-full items-center rounded-lg px-1 text-left text-[14.5px] font-semibold nx-row hover:bg-n-surface-2">
                  {f.properties.display_name ?? f.properties.name}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
      <p className="text-[12.5px] text-n-ink-3">Satellite and current statistics for an area are not computed in this version; use their layers and a point.</p>
      </EvidenceFold>
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
function HoverLayer({ sink, manifest, baseUrl, source, inset, waterAt }: { sink: React.RefObject<((h: HoverAt | null) => void) | null>; manifest: Manifest; baseUrl: string; source: HoverSource; inset: number; waterAt: (s: { x: number; y: number }) => boolean | null }) {
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
      <HoverReadout manifest={manifest} baseUrl={baseUrl} source={source} point={hover.p} screen={hover.s} waterAt={waterAt} bounds={{ w: window.innerWidth - inset, h: window.innerHeight }} />
    </div>
  );
}

/**
 * The time rail's status row at a stable height: it keeps the tallest height its content has had since the
 * layer (the key) changed, so lines that come and go during playback never move the rail above it.
 */
function StableCaption({ children, min = 0 }: { children: React.ReactNode; min?: number }) {
  const inner = useRef<HTMLDivElement | null>(null);
  const [h, setH] = useState(min);
  useEffect(() => {
    const el = inner.current;
    if (!el) return;
    const f = () => setH((was) => Math.max(was, Math.ceil(el.getBoundingClientRect().height)));
    f();
    const ro = new ResizeObserver(f);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return (
    <div className="min-w-0" style={{ minHeight: h || undefined }} data-testid="rail-status">
      <div ref={inner} className="flex min-w-0 flex-col gap-1.5">
        {children}
      </div>
    </div>
  );
}
