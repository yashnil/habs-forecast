"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import {
  CHARM_LEADS,
  CHARM_VARIABLES,
  artifactUrl,
  charmLayer,
  charmRun,
  chlorophyllLayers,
  satelliteDays,
  satelliteLatest,
  sourceStatus,
  tileIndexUrl,
  type CharmVariable,
  type SatProduct,
} from "@/lib/layers";
import { officialVerification } from "@/lib/official";
import { openingLayer, type Opening } from "@/lib/opening";
import { cellEdges, loadGrid } from "@/lib/grid";
import type { StationLite } from "@/lib/data";
import { useNow } from "@/lib/useNow";
import { PortPanel } from "@/components/port/PortPanel";
import { Inspector, type InspectPoint } from "@/components/map/Inspector";
import { MobileSheet, type SnapId, type SnapPoint } from "@/components/MobileSheet";
import type { CurrentsOverlay, ImageRaster, MapPoint, TileRaster } from "@/components/map/MapCanvas";
import { NavCard, type RegionDef } from "@/components/map/NavCard";
import { PlaceChip } from "@/components/map/PlaceChip";
import { HoverReadout, type HoverSource } from "@/components/map/HoverReadout";
import { MeasuredNearby } from "@/components/map/MeasuredNearby";
import { LayerDock, selectedCurrents, type CurChoice, type FlowMode, type LayerGroup, type SatChoice } from "@/components/map/LayerDock";
import { ControlRail } from "@/components/map/ControlRail";
import { SatelliteNear } from "@/components/map/SatelliteNear";
import { CurrentsNear } from "@/components/map/CurrentsNear";
import { stampLines } from "@/lib/stamp";
import { currentsHourly, fieldFeatures, loadField, type CurrentField } from "@/lib/currents";
import { OFFICIAL_STATUS } from "@/content/copy";
import { Icon } from "@/components/ui/Icon";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-[#0b1d33]" aria-label="Loading map" />,
});

type Props = {
  manifest: Manifest;
  ports: PortsCollection | null;
  portsError: string | null;
  official: OfficialDataset | null;
  officialError: string | null;
  portIntel: PortIntelCollection | null;
  portIntelError: string | null;
  stations: StationLite[] | null;
  stationsError: string | null;
  baseUrl: string;
  /** search params from the server so the first render matches on server and client */
  initialParams: Record<string, string>;
};

const STATEWIDE: RegionDef = { id: "california", label: "All California", bounds: [[-125.9, 32.45], [-117.1, 42.05]] };
// Hand-set views tight on the coast that matters: every port in the region and the shelf off
// it, with a little sea. Monterey Bay runs Año Nuevo to Point Sur.
const VIEW: Record<string, [[number, number], [number, number]]> = {
  north_coast: [[-124.72, 39.95], [-123.6, 42.02]],
  mendocino_sonoma: [[-124.05, 38.2], [-122.85, 39.6]],
  sf_bay_farallones: [[-123.15, 37.4], [-122.3, 38.15]],
  monterey_bay: [[-122.38, 36.48], [-121.66, 37.13]],
  central_coast: [[-121.45, 34.75], [-120.4, 35.85]],
  southern_california: [[-120.55, 32.55], [-117.05, 34.55]],
};
const DEFAULT_REGION = "monterey_bay";

function parseCurrents(v: string | null): CurChoice | null {
  if (!v || !v.startsWith("currents")) return null;
  const rest = v.split(":")[1] ?? null;
  return rest === "mean" ? { hour: null, mean: true } : { hour: rest && /^\d{8}T\d{2}Z$/.test(rest) ? rest : null, mean: false };
}

function parseLayer(v: string | null): { group: LayerGroup | null; sat: SatChoice } | null {
  if (v === "none") return { group: null, sat: { product: "olci300", day: null } };
  if (!v || v === "forecast") return v ? { group: "forecast", sat: { product: "olci300", day: null } } : null;
  if (v.startsWith("currents")) return { group: "currents", sat: { product: "olci300", day: null } };
  if (v.startsWith("imagery:")) return { group: "satellite", sat: { imagery: v.slice(8) } };
  const [prod, day] = v.split(":");
  if (prod === "olci300" || prod === "viirs750" || prod === "multi") return { group: "satellite", sat: { product: prod, day: prod === "multi" ? null : (day ?? null) } };
  if (v.startsWith("gibs_")) return { group: "satellite", sat: { imagery: v } }; // M1-M3 links
  return null;
}

export function LiveOceanMap({ manifest, ports, portsError, official, officialError, portIntel, portIntelError, stations, stationsError, baseUrl, initialParams }: Props) {
  const now = useNow();
  const run = charmRun(manifest);
  const mapRef = useRef<MlMap | null>(null);
  const pendingFly = useRef<number | null>(null);
  const dockRef = useRef<HTMLDivElement | null>(null);
  // null until the first client render, so a phone never flashes the desktop layout
  const [wide, setWide] = useState<boolean | null>(null);
  const [navOpen, setNavOpen] = useState(false);

  const regions = useMemo<RegionDef[]>(
    () => [STATEWIDE, ...(ports?.regions ?? []).map((r) => ({ id: r.id, label: r.label, bounds: (VIEW[r.id] ?? r.bounds) as [[number, number], [number, number]] }))],
    [ports],
  );
  const initialQ = useMemo(() => new URLSearchParams(initialParams), [initialParams]);
  const initialRegion = regions.find((r) => r.id === initialQ.get("region")) ?? regions.find((r) => r.id === DEFAULT_REGION) ?? STATEWIDE;

  const firstLead = run?.leads_available.includes(1) ? 1 : (run?.leads_available[0] ?? 0);
  const initialLayer = parseLayer(initialQ.get("layer"));
  // The opening layer (lib/opening): a layer named in the link wins; otherwise it is chosen
  // once, in the browser, from the data's freshness and coverage. Until then nothing is drawn.
  const explicit = useRef(initialLayer != null);
  const [decided, setDecided] = useState(initialLayer != null);
  const [opening, setOpening] = useState<Opening | null>(null);
  const [group, setGroupRaw] = useState<LayerGroup | null>(initialLayer ? initialLayer.group : null);
  const [sat, setSatRaw] = useState<SatChoice>(initialLayer?.sat ?? { product: satelliteLatest(manifest, "olci300") ? "olci300" : "viirs750", day: null });
  const [cur, setCurRaw] = useState<CurChoice>(parseCurrents(initialQ.get("layer")) ?? { hour: null, mean: false });
  const setGroup = useCallback((g: LayerGroup | null) => {
    explicit.current = true;
    setDecided(true);
    setGroupRaw(g);
  }, []);
  const setSat = useCallback((c: SatChoice) => {
    explicit.current = true;
    setSatRaw(c);
  }, []);
  const setCur = useCallback((c: CurChoice) => {
    explicit.current = true;
    setCurRaw(c);
  }, []);
  useEffect(() => {
    if (decided || !now) return;
    const v = initialQ.get("var");
    const o = openingLayer(manifest, now, {
      regionId: initialRegion.id === STATEWIDE.id ? null : initialRegion.id,
      variable: v && (CHARM_VARIABLES as readonly string[]).includes(v) ? v : "particulate_domoic",
      lead: initialQ.has("lead") ? Number(initialQ.get("lead")) : firstLead,
    });
    setOpening(o);
    if (o.group === "satellite") setSatRaw({ product: o.product, day: null });
    setGroupRaw(o.group);
    setDecided(true);
  }, [decided, now, manifest, initialQ, initialRegion.id, firstLead]);
  const [flow, setFlow] = useState<FlowMode>(initialQ.get("flow") === "particles" ? "particles" : "arrows");
  // combined view (opt-in): satellite chlorophyll under the observed currents
  const [underlay, setUnderlay] = useState(initialQ.get("chl") === "1");
  const [reducedMotion, setReducedMotion] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const on = () => setReducedMotion(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  const [showAge, setShowAgeRaw] = useState(initialQ.get("age") === "1");
  const [showSensor, setShowSensorRaw] = useState(initialQ.get("sensor") === "1" && initialQ.get("age") !== "1");
  // the age and sensor views replace the chlorophyll colours, so only one at a time
  const setShowAge = (v: boolean) => {
    setShowAgeRaw(v);
    if (v) setShowSensorRaw(false);
  };
  const setShowSensor = (v: boolean) => {
    setShowSensorRaw(v);
    if (v) setShowAgeRaw(false);
  };
  const [variable, setVariable] = useState<CharmVariable>("particulate_domoic");
  const [lead, setLead] = useState<number>(firstLead);
  const [region, setRegion] = useState<string>(initialRegion.id);
  const [port, setPort] = useState<number | null>(null);
  const [inspect, setInspect] = useState<InspectPoint | null>(null);

  useEffect(() => {
    const onResize = () => setWide(window.innerWidth >= 1024);
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // initial state from the URL (shareable links; deterministic e2e tests)
  useEffect(() => {
    const q = initialQ;
    const v = q.get("var");
    if (v && (CHARM_VARIABLES as readonly string[]).includes(v)) setVariable(v as CharmVariable);
    const l = Number(q.get("lead"));
    if (q.has("lead") && (CHARM_LEADS as readonly number[]).includes(l)) setLead(l);
    const p = Number(q.get("port"));
    if (q.has("port") && ports?.features.some((f) => f.properties.port_code === p)) {
      setPort(p);
      pendingFly.current = p;
    }
    const ins = q.get("inspect")?.split(",").map(Number);
    if (ins && ins.length === 2 && ins.every(Number.isFinite)) setInspect({ lat: ins[0], lon: ins[1] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const layerParam =
    group == null ? "none" : group === "currents" ? (cur.mean ? "currents:mean" : cur.hour ? `currents:${cur.hour}` : "currents") : group === "forecast" ? "forecast" : "imagery" in sat ? `imagery:${sat.imagery}` : sat.day ? `${sat.product}:${sat.day}` : sat.product;
  useEffect(() => {
    if (!decided) return;
    const q = new URLSearchParams();
    q.set("region", region);
    if (port != null) q.set("port", String(port));
    q.set("var", variable);
    q.set("lead", String(lead));
    // the layer is written only once someone picked it (or the link named it), so a bookmark
    // of the plain map keeps choosing by freshness
    if (explicit.current) q.set("layer", layerParam);
    if (showAge && group === "satellite") q.set("age", "1");
    if (showSensor && group === "satellite") q.set("sensor", "1");
    if (flow === "particles" && group === "currents") q.set("flow", "particles");
    if (underlay && group === "currents") q.set("chl", "1");
    if (inspect) q.set("inspect", `${inspect.lat.toFixed(4)},${inspect.lon.toFixed(4)}`);
    window.history.replaceState(null, "", `?${q.toString()}`);
  }, [decided, region, port, variable, lead, layerParam, inspect, showAge, showSensor, group, flow, underlay]);

  const detailOpen = port != null || inspect != null;
  const [dockOpen, setDockOpen] = useState(false);
  const [placesOpen, setPlacesOpen] = useState(false);
  const [sheetSnap, setSheetSnap] = useState<SnapId>("peek");
  const [mapH, setMapH] = useState(700);
  const rootRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = rootRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setMapH(el.getBoundingClientRect().height));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // phone sheet: the peek is just tall enough for the layer tabs, what is shown and the whole
  // legend (it grows when the stamp wraps), so most of the screen stays map (≥ 55 % at 390 × 844)
  const [peekH, setPeekH] = useState(152);
  const legendBox = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const box = legendBox.current;
    if (!box || typeof ResizeObserver === "undefined") return;
    const fit = () => {
      // the legend's ramp (or arrow key) and its numbers; notes below it show on expanding
      const lg = box.querySelector("[data-peek=end]") ?? box.querySelector("[data-testid$=-legend]");
      if (!lg) return;
      const need = 24 + (lg.getBoundingClientRect().bottom - box.getBoundingClientRect().top) + 4; // handle + content + a hair of margin
      setPeekH(Math.round(Math.min(184, Math.max(136, need))));
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(box);
    return () => ro.disconnect();
  }, [wide, detailOpen, group]);
  const snaps: SnapPoint[] = detailOpen
    ? [
        { id: "peek", height: 196 },
        { id: "full", height: Math.max(260, mapH - 12) },
      ]
    : [
        { id: "peek", height: peekH },
        { id: "half", height: Math.round(mapH * 0.58) },
        { id: "full", height: Math.max(260, mapH - 12) },
      ];
  const sheetH = snaps.find((x) => x.id === sheetSnap)?.height ?? snaps[0].height;
  useEffect(() => setSheetSnap("peek"), [detailOpen, port, inspect]);

  const [dockH, setDockH] = useState(200);
  useEffect(() => {
    const el = dockRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => setDockH(el.getBoundingClientRect().height));
    ro.observe(el);
    return () => ro.disconnect();
  }, [wide]);
  const RAIL = 64;
  const INSPECTOR = 380;
  /** Free map area for fitting a region: right of the rail, below the place chip, clear of the
   *  inspector, and either beside the dock (and the place list, when open) or above the dock,
   *  whichever frames the region larger. */
  const padding = useCallback(() => {
    const m = mapRef.current;
    if (window.innerWidth >= 1024) {
      const right = (detailOpen ? INSPECTOR + 16 : 0) + 28;
      const r = dockRef.current?.getBoundingClientRect();
      const dH = r?.height ?? 200;
      const dW = r?.width ?? 560;
      const beside = { top: 64, bottom: 28, left: RAIL + 12 + Math.max(dW, placesOpen ? 320 : 0) + 20, right };
      const above = { top: 64, bottom: dH + 28, left: RAIL + 28, right };
      if (!m || placesOpen) return beside;
      const b = regions.find((x) => x.id === region)?.bounds ?? STATEWIDE.bounds;
      const z = (q: typeof beside) => m.cameraForBounds(b, { padding: q })?.zoom ?? 0;
      return z(above) > z(beside) ? above : beside;
    }
    return { top: 64, bottom: sheetH + 16, left: 16, right: 16 };
  }, [detailOpen, placesOpen, regions, region, sheetH]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape" || e.defaultPrevented) return;
      if (document.querySelector("[role=dialog][aria-modal=true]")) return; // the drawer handles its own
      if (placesOpen) setPlacesOpen(false);
      else if (port != null) setPort(null);
      else if (inspect) setInspect(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [placesOpen, port, inspect]);

  const goRegion = useCallback(
    (id: string) => {
      setRegion(id);
      setPort(null);
      setInspect(null);
      setNavOpen(false);
      const r = regions.find((x) => x.id === id);
      if (r && mapRef.current) mapRef.current.fitBounds(r.bounds, { padding: { ...padding(), right: (window.innerWidth >= 1024 ? 0 : 16) + 28 }, duration: reducedMotion ? 0 : 800 });
    },
    [regions, padding, reducedMotion],
  );

  const selectPort = useCallback(
    (code: number) => {
      setPort(code);
      setInspect(null);
      setNavOpen(false);
      setPlacesOpen(false);
      const f = ports?.features.find((x) => x.properties.port_code === code);
      if (f) {
        if (f.properties.region) setRegion(f.properties.region);
        const r = regions.find((x) => x.id === f.properties.region);
        // the inspector opens with the port: fit beside it (desktop) or above its peek (phone)
        const pad = window.innerWidth >= 1024 ? { ...padding(), right: INSPECTOR + 44 } : { ...padding(), bottom: 196 + 16 };
        if (r && mapRef.current) mapRef.current.fitBounds(r.bounds, { padding: pad, duration: reducedMotion ? 0 : 800 });
        else {
          const [lon, lat] = f.geometry.coordinates as number[];
          mapRef.current?.flyTo({ center: [lon, lat], zoom: Math.max(mapRef.current.getZoom(), 9), padding: pad, duration: reducedMotion ? 0 : 800 });
        }
        pendingFly.current = mapRef.current ? null : code;
      }
    },
    [ports, padding, regions, reducedMotion],
  );

  // ---------------------------------------------------------------- what the map draws
  const forecastLayer = charmLayer(manifest, variable, lead);
  const forecast: ImageRaster | null = useMemo(
    () => (group === "forecast" && forecastLayer?.image ? { id: forecastLayer.layer_id, url: artifactUrl(baseUrl, forecastLayer.image.url), corners: forecastLayer.image.corners_lnglat } : null),
    [group, forecastLayer, baseUrl],
  );
  const combined = group === "currents" && underlay && !!satelliteLatest(manifest, "olci300");
  const satLayer =
    group === "satellite" && !("imagery" in sat)
      ? sat.day
        ? (satelliteDays(manifest, sat.product as SatProduct).find((d) => d.time.observed_date === sat.day) ?? null)
        : satelliteLatest(manifest, sat.product)
      : combined
        ? satelliteLatest(manifest, "olci300")
        : null;
  const satellite: TileRaster | null = useMemo(() => {
    const t = satLayer?.tiles;
    if (!satLayer || !t || (!combined && ((showAge && (satLayer.composite || satLayer.multisensor)) || (showSensor && satLayer.multisensor)))) return null;
    return { id: satLayer.layer_id, template: t.relative ? artifactUrl(baseUrl, t.url_template) : t.url_template, minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat, indexUrl: tileIndexUrl(baseUrl, t) };
  }, [satLayer, baseUrl, showAge, showSensor, combined]);
  // categorical overlays that replace the chlorophyll colours: observation age, or which sensor
  const satelliteAge: TileRaster | null = useMemo(() => {
    const t = combined ? null : showAge ? (satLayer?.composite?.age_tiles ?? satLayer?.multisensor?.age_tiles) : showSensor ? satLayer?.multisensor?.sensor_tiles : null;
    if (!t || !satLayer) return null;
    return { id: `${satLayer.layer_id}-${showAge ? "age" : "sensor"}`, template: artifactUrl(baseUrl, t.url_template), minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat, indexUrl: tileIndexUrl(baseUrl, t) };
  }, [satLayer, baseUrl, showAge, showSensor, combined]);
  // observed currents: the selected hour (or mean) decoded from its u/v grids
  const curLayer = currentsHourly(manifest).length ? selectedCurrents(manifest, cur) : null;
  const [curField, setCurField] = useState<{ id: string; field: CurrentField } | null>(null);
  useEffect(() => {
    let cancelled = false;
    if (group !== "currents" || !curLayer) return;
    loadField(baseUrl, curLayer)
      .then((f) => !cancelled && setCurField({ id: curLayer.layer_id, field: f }))
      .catch(() => !cancelled && setCurField(null));
    return () => {
      cancelled = true;
    };
  }, [group, curLayer, baseUrl]);
  const currentsOverlay: CurrentsOverlay | null = useMemo(() => {
    if (group !== "currents" || !curField || curField.id !== curLayer?.layer_id) return null;
    return { id: curField.id, field: curField.field, features: fieldFeatures(curField.field), mode: flow === "particles" && !reducedMotion ? "particles" : "arrows", coarser: combined };
  }, [group, curField, curLayer, flow, reducedMotion, combined]);
  const imageryLayer = group === "satellite" && "imagery" in sat ? (chlorophyllLayers(manifest).find((l) => l.layer_id === sat.imagery) ?? chlorophyllLayers(manifest)[0]) : null;
  const imagery: TileRaster | null = imageryLayer?.tiles ? { id: imageryLayer.layer_id, template: imageryLayer.tiles.url_template, minzoom: 0, maxzoom: imageryLayer.tiles.max_native_zoom } : null;

  // C-HARM's own cell edges (faint, from z8.5): its 3 km cells read as model cells, not blur
  const [edges, setEdges] = useState<{ id: string; fc: GeoJSON.FeatureCollection } | null>(null);
  useEffect(() => {
    let cancelled = false;
    const g = group === "forecast" ? forecastLayer?.grid : null;
    if (!g || !forecastLayer || edges?.id === forecastLayer.layer_id) return;
    loadGrid(artifactUrl(baseUrl, g.url))
      .then((codes) => !cancelled && setEdges({ id: forecastLayer.layer_id, fc: cellEdges(g, codes) }))
      .catch(() => !cancelled && setEdges(null));
    return () => {
      cancelled = true;
    };
  }, [group, forecastLayer, baseUrl, edges?.id]);

  // switching layer fades the new data in; a new time step of the same layer cuts (lib note in MapCanvas)
  const fadeRef = useRef<{ group: LayerGroup | null; until: number }>({ group: null, until: 0 });
  // performance.now: the page clock (Date) may be frozen in tests and kiosks; this keeps counting
  if (fadeRef.current.group !== group) fadeRef.current = { group, until: performance.now() + 700 };
  const fade = !reducedMotion && performance.now() < fadeRef.current.until;

  // desktop hover readout: the exact value under the pointer for the layer on the map
  const [hover, setHover] = useState<{ p: MapPoint; s: { x: number; y: number } } | null>(null);
  const hoverRaf = useRef<number | null>(null);
  const hoverNext = useRef<{ p: MapPoint; s: { x: number; y: number } } | null>(null);
  const [canHover, setCanHover] = useState(false);
  useEffect(() => setCanHover(window.matchMedia("(hover: hover) and (pointer: fine)").matches), []);
  const onHover = useCallback((pt: MapPoint | null, scr: { x: number; y: number } | null) => {
    hoverNext.current = pt && scr ? { p: pt, s: scr } : null;
    if (hoverRaf.current == null)
      hoverRaf.current = requestAnimationFrame(() => {
        hoverRaf.current = null;
        setHover(hoverNext.current);
      });
  }, []);
  const hoverSource: HoverSource = useMemo(() => {
    if (group === "forecast" && forecastLayer) return { kind: "forecast", layer: forecastLayer };
    if (group === "satellite" && satellite && satLayer) return { kind: "satellite", layer: satLayer };
    if (group === "currents" && curField && curLayer && curField.id === curLayer.layer_id) return { kind: "currents", layer: curLayer, field: curField.field };
    return null;
  }, [group, forecastLayer, satellite, satLayer, curField, curLayer]);

  const stamp = stampLines({
    group, forecast: forecastLayer, run, satellite: satLayer, imagery: imageryLayer, currents: group === "currents" ? curLayer : null, combined, now,
    region: region === STATEWIDE.id ? null : { id: region, label: regions.find((r) => r.id === region)?.label ?? region },
  });
  const verification = useMemo(() => (now ? officialVerification(official, sourceStatus(manifest, "official"), now) : null), [official, manifest, now]);
  const intel = port != null ? (portIntel?.ports.find((p) => p.port_code === port) ?? null) : null;
  const regionNotices = useMemo(() => {
    if (!portIntel || region === STATEWIDE.id) return null;
    const ids = new Set<string>();
    const active = new Set((official?.registry.records ?? []).filter((r) => r.status === "active").map((r) => r.id));
    for (const p of portIntel.ports) if (p.region === region) for (const rel of p.official_relations) if (active.has(rel.record_id)) ids.add(rel.record_id);
    return ids.size;
  }, [portIntel, region, official]);
  const regionLabel = regions.find((r) => r.id === region)?.label ?? STATEWIDE.label;

  const near = (lat: number, lon: number, place: string) => (
    <>
      <SatelliteNear manifest={manifest} baseUrl={baseUrl} lat={lat} lon={lon} place={place} />
      {curLayer && <CurrentsNear layer={curLayer} baseUrl={baseUrl} lat={lat} lon={lon} place={place} />}
      <MeasuredNearby stations={stations} error={stationsError} lat={lat} lon={lon} now={now} />
    </>
  );
  const detail =
    port != null ? (
      intel && portIntel ? (
        <PortPanel port={intel} coll={portIntel} manifest={manifest} official={official} verification={verification} lead={lead} onLead={setLead} variable={variable} onVariable={setVariable} now={now} onClose={() => setPort(null)}>
          {near(intel.lat, intel.lon, intel.display_name)}
        </PortPanel>
      ) : (
        <div className="rounded-lg border border-hairline-strong p-4 text-[13px] text-ink-2" data-testid="port-unavailable">
          Port summaries are unavailable{portIntelError ? ` (${portIntelError})` : ""}. {OFFICIAL_STATUS.missingNotOpen}
          <button onClick={() => setPort(null)} className="ml-2 underline">
            Close
          </button>
        </div>
      )
    ) : inspect ? (
      <div className="space-y-5">
        <Inspector manifest={manifest} baseUrl={baseUrl} point={inspect} lead={lead} onLead={setLead} variable={variable} onClose={() => setInspect(null)} official={official} verification={verification} />
        {near(inspect.lat, inspect.lon, "this point")}
      </div>
    ) : null;

  const navProps = {
    regions,
    region,
    onRegion: goRegion,
    ports: portIntel?.ports ?? [],
    portsError: portIntel ? null : (portIntelError ?? "unavailable"),
    port,
    onPort: selectPort,
    variable,
    lead,
    palette: forecastLayer?.palette,
    regionNotices,
    showForecast: group === "forecast",
  };
  const nav = <NavCard {...navProps} />;
  const dockProps = {
    manifest,
    group,
    onGroup: setGroup,
    opening,
    run,
    charmStatus: sourceStatus(manifest, "charm"),
    satStatus: sourceStatus(manifest, "satellite_chl"),
    gibsStatus: sourceStatus(manifest, "gibs_chl"),
    variable,
    onVariable: setVariable,
    lead,
    onLead: setLead,
    sat,
    onSat: setSat,
    showAge,
    onShowAge: setShowAge,
    showSensor,
    onShowSensor: setShowSensor,
    cur,
    onCur: setCur,
    flow,
    onFlow: setFlow,
    curStatus: sourceStatus(manifest, "hf_radar"),
    reducedMotion,
    underlay,
    // the combined view's four statements are part of turning it on
    onUnderlay: (b: boolean) => {
      setUnderlay(b);
      if (b) setDockOpen(true);
    },
    regionId: region === STATEWIDE.id ? "monterey_bay" : region,
    regionLabel: region === STATEWIDE.id ? "Monterey Bay" : regionLabel,
    now,
    stamp,
  };

  const map = (
    <MapCanvas
      forecast={forecast}
      satellite={satellite}
      satelliteAge={satelliteAge}
      imagery={imagery}
      currents={currentsOverlay}
      ports={ports}
      showPorts
      officialGeometry={(official?.geometry as unknown as GeoJSON.FeatureCollection) ?? null}
      showOfficial
      selectedPort={port}
      inspect={inspect}
      initialBounds={initialRegion.bounds}
      initialPadding={wide ? { top: 64, bottom: 28, left: 660, right: 28 } : { top: 64, bottom: 168, left: 16, right: 16 }}
      cellEdges={group === "forecast" && edges && edges.id === forecastLayer?.layer_id ? edges.fc : null}
      fade={fade}
      onHover={wide && canHover ? onHover : undefined}
      onPort={selectPort}
      onPoint={(pt) => {
        setPort(null);
        setInspect(pt);
        setPlacesOpen(false);
        // keep the chosen point in view beside the inspector that opens for it
        const m = mapRef.current;
        if (m && window.innerWidth >= 1024) {
          const x = m.project([pt.lon, pt.lat]).x;
          const free = m.getContainer().clientWidth - INSPECTOR - 16;
          if (x > free - 40) m.easeTo({ center: m.unproject([m.getContainer().clientWidth / 2 + (x - (RAIL + free) / 2), m.getContainer().clientHeight / 2]), duration: reducedMotion ? 0 : 500 });
        }
      }}
      onReady={(m) => {
        mapRef.current = m;
        m.fitBounds(initialRegion.bounds, { padding: padding(), duration: 0 });
        const code = pendingFly.current;
        if (code != null) {
          pendingFly.current = null;
          m.once("idle", () => selectPort(code));
        }
      }}
    />
  );

  const verWord = verification ? OFFICIAL_STATUS.verification[verification.state] : "Checking…";
  const activeCount = official ? official.registry.records.filter((r) => r.status === "active").length : null;

  // The map keeps the same position in the tree for both layouts so switching between
  // desktop and mobile never re-creates it.
  return (
    <div ref={rootRef} className="relative min-h-0 flex-1 overflow-hidden" data-testid="map">
      <div className="absolute inset-0">{map}</div>
      {wide && hover && !detailOpen && <HoverReadout manifest={manifest} baseUrl={baseUrl} source={hoverSource} point={hover.p} screen={hover.s} bounds={{ w: rootRef.current?.clientWidth ?? 1440, h: mapH }} />}
      {wide && hover && detailOpen && hover.s.x < (rootRef.current?.clientWidth ?? 1440) - INSPECTOR - 16 && <HoverReadout manifest={manifest} baseUrl={baseUrl} source={hoverSource} point={hover.p} screen={hover.s} bounds={{ w: (rootRef.current?.clientWidth ?? 1440) - INSPECTOR - 16, h: mapH }} />}
      {wide == null || !decided ? null : wide ? (
        <>
          <ControlRail manifest={manifest} group={group} onGroup={setGroup} placesOpen={placesOpen} onPlaces={() => setPlacesOpen(!placesOpen)} />
          <div className="absolute top-3 z-20" style={{ left: RAIL + 12, maxWidth: `calc(100% - ${RAIL + 24}px - ${detailOpen ? INSPECTOR + 16 : 0}px)` }}>
            <PlaceChip label={port != null ? (intel?.display_name ?? "Port") : regionLabel} open={placesOpen} onToggle={() => setPlacesOpen(!placesOpen)} regionNotices={regionNotices} statewide={region === STATEWIDE.id} />
          </div>
          {placesOpen && (
            <div className="cw-pop-in absolute z-20 flex w-[320px]" style={{ left: RAIL + 12, top: 60, maxHeight: `calc(100% - ${Math.round(dockH) + 84}px)` }}>
              {/* the place chip above already shows the notices, so the list starts with places */}
              <NavCard {...navProps} showOfficial={false} />
            </div>
          )}
          <div
            ref={dockRef}
            className="absolute bottom-3 z-10 flex max-h-[calc(100%-24px)] flex-col"
            style={{ left: RAIL + 12, width: `min(${dockOpen ? 600 : 560}px, calc(100% - ${RAIL + 24}px - ${detailOpen ? INSPECTOR + 16 : 0}px))` }}
          >
            <LayerDock {...dockProps} variant="desktop" expanded={dockOpen} onExpanded={setDockOpen} />
          </div>
          {detail && (
            <aside
              aria-label="Selected place"
              data-testid="detail-panel"
              className="theme-paper cw-sheet-in absolute bottom-3 right-3 top-3 z-20 overflow-y-auto overscroll-contain rounded-xl bg-surface p-5 text-ink shadow-[0_1px_2px_rgba(6,17,30,0.14),0_10px_30px_rgba(6,17,30,0.25)] [scrollbar-width:thin]"
              style={{ width: INSPECTOR - 12 }}
            >
              {detail}
            </aside>
          )}
        </>
      ) : (
        <>
          <button
            onClick={() => setNavOpen(true)}
            data-testid="place-button"
            className="theme-paper absolute left-3 top-2.5 z-10 flex h-10 max-w-[calc(100%-24px)] items-center gap-2 rounded-full bg-surface pl-3.5 pr-1 text-left text-[15px] font-medium text-ink shadow-[0_6px_20px_rgba(6,17,30,0.3)]"
          >
            <Icon name="search" className="h-4 w-4 text-ink-3" />
            <span className="min-w-0 truncate pr-1">{port != null ? (intel?.display_name ?? "Port") : regionLabel}</span>
            <span data-testid="mobile-official" className="flex h-8 shrink-0 items-center gap-1 rounded-full border border-official-line bg-official-bg px-2.5 text-[12.5px] font-semibold text-official-ink">
              <Icon name="shield" className="h-3.5 w-3.5 text-official" />
              {region !== STATEWIDE.id && regionNotices != null ? regionNotices : (activeCount ?? "?")}
              <span className="sr-only"> official notices{region !== STATEWIDE.id ? ` may apply in ${regionLabel}` : " in California"},</span>
              {verification && verification.state !== "verified" && <span className="font-medium">{verification.state === "aging" ? "· ageing" : "· unverified"}</span>}
              <span className="sr-only"> {verWord}</span>
            </span>
          </button>
          {navOpen && (
            <div className="theme-paper cw-sheet-up absolute inset-0 z-30 flex flex-col overflow-hidden bg-surface text-ink" role="dialog" aria-modal="true" aria-label="Places">
              <div className="flex items-center justify-between px-4 pb-1 pt-2.5">
                <span className="font-display text-[20px] font-medium">Places</span>
                <button onClick={() => setNavOpen(false)} aria-label="Close places" data-testid="places-close" className="grid h-11 w-11 place-items-center text-ink-3">
                  <Icon name="close" className="h-5 w-5" />
                </button>
              </div>
              <div className="flex min-h-0 flex-1 flex-col [&>section]:rounded-none [&>section]:shadow-none">{nav}</div>
            </div>
          )}
          {!navOpen &&
            (detail ? (
              <MobileSheet
                key="inspect"
                snaps={snaps}
                snap={sheetSnap}
                onSnap={setSheetSnap}
                label="Place details"
                header={null}
              >
                <div className="pt-1">{detail}</div>
              </MobileSheet>
            ) : (
              <MobileSheet key="explore" snaps={snaps} snap={sheetSnap} onSnap={setSheetSnap} label="Map layers" header={null}>
                <div data-testid="mobile-legend" ref={legendBox}>
                  <LayerDock {...dockProps} variant="sheet" expanded={sheetSnap !== "peek"} />
                </div>
              </MobileSheet>
            ))}
        </>
      )}
      {officialError && !official && <p className="sr-only">Official notices unavailable: {officialError}</p>}
      {portsError && !ports && <p className="sr-only">Ports unavailable: {portsError}</p>}
    </div>
  );
}
