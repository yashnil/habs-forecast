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
  type CharmVariable,
  type SatProduct,
} from "@/lib/layers";
import { officialVerification } from "@/lib/official";
import { useNow } from "@/lib/useNow";
import { PortPanel } from "@/components/port/PortPanel";
import { Inspector, type InspectPoint } from "@/components/map/Inspector";
import { MobileSheet } from "@/components/MobileSheet";
import type { CurrentsOverlay, ImageRaster, TileRaster } from "@/components/map/MapCanvas";
import { NavCard, type RegionDef } from "@/components/map/NavCard";
import { LayerDock, selectedCurrents, type CurChoice, type FlowMode, type LayerGroup, type SatChoice } from "@/components/map/LayerDock";
import { SatelliteNear } from "@/components/map/SatelliteNear";
import { CurrentsNear } from "@/components/map/CurrentsNear";
import { MapStamp } from "@/components/map/MapStamp";
import { stampLines } from "@/lib/stamp";
import { DEMO } from "@/lib/demo";
import { currentsHourly, fieldFeatures, loadField, type CurrentField } from "@/lib/currents";
import { OFFICIAL_STATUS } from "@/content/copy";
import { useOfficialDrawer } from "@/components/shell/OfficialShell";
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
  baseUrl: string;
  /** search params from the server so the first render matches on server and client */
  initialParams: Record<string, string>;
};

const STATEWIDE: RegionDef = { id: "california", label: "All California", bounds: [[-125.9, 32.45], [-117.1, 42.05]] };
// Hand-set views tight on the coast that matters (design reset rev. 2): Monterey Bay runs
// Año Nuevo to Point Sur; other regions use their published bounds with a little sea.
const VIEW: Record<string, [[number, number], [number, number]]> = {
  monterey_bay: [[-122.38, 36.48], [-121.66, 37.13]],
};
const DEFAULT_REGION = "monterey_bay";

function parseCurrents(v: string | null): CurChoice | null {
  if (DEMO || !v || !v.startsWith("currents")) return null;
  const rest = v.split(":")[1] ?? null;
  return rest === "mean" ? { hour: null, mean: true } : { hour: rest && /^\d{8}T\d{2}Z$/.test(rest) ? rest : null, mean: false };
}

function parseLayer(v: string | null): { group: LayerGroup; sat: SatChoice } | null {
  if (!v || v === "forecast") return v ? { group: "forecast", sat: { product: "olci300", day: null } } : null;
  if (v.startsWith("currents")) return DEMO ? null : { group: "currents", sat: { product: "olci300", day: null } };
  if (v.startsWith("imagery:")) return { group: "satellite", sat: { imagery: v.slice(8) } };
  const [prod, day] = v.split(":");
  if (prod === "olci300" || prod === "viirs750" || prod === "multi") return { group: "satellite", sat: { product: prod, day: prod === "multi" ? null : (day ?? null) } };
  if (v.startsWith("gibs_")) return { group: "satellite", sat: { imagery: v } }; // M1-M3 links
  return null;
}

export function LiveOceanMap({ manifest, ports, portsError, official, officialError, portIntel, portIntelError, baseUrl, initialParams }: Props) {
  const now = useNow();
  const run = charmRun(manifest);
  const mapRef = useRef<MlMap | null>(null);
  const pendingFly = useRef<number | null>(null);
  const dockRef = useRef<HTMLDivElement | null>(null);
  const [wide, setWide] = useState(true);
  const [navOpen, setNavOpen] = useState(false);
  const { openDrawer } = useOfficialDrawer();

  const regions = useMemo<RegionDef[]>(
    () => [STATEWIDE, ...(ports?.regions ?? []).map((r) => ({ id: r.id, label: r.label, bounds: (VIEW[r.id] ?? r.bounds) as [[number, number], [number, number]] }))],
    [ports],
  );
  const initialQ = useMemo(() => new URLSearchParams(initialParams), [initialParams]);
  const initialRegion = regions.find((r) => r.id === initialQ.get("region")) ?? regions.find((r) => r.id === DEFAULT_REGION) ?? STATEWIDE;

  const firstLead = run?.leads_available.includes(1) ? 1 : (run?.leads_available[0] ?? 0);
  const initialLayer = parseLayer(initialQ.get("layer"));
  const [group, setGroup] = useState<LayerGroup>(initialLayer?.group ?? "forecast");
  const [sat, setSat] = useState<SatChoice>(initialLayer?.sat ?? { product: satelliteLatest(manifest, "olci300") ? "olci300" : "viirs750", day: null });
  const [cur, setCur] = useState<CurChoice>(parseCurrents(initialQ.get("layer")) ?? { hour: null, mean: false });
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
    group === "currents" ? (cur.mean ? "currents:mean" : cur.hour ? `currents:${cur.hour}` : "currents") : group === "forecast" ? "forecast" : "imagery" in sat ? `imagery:${sat.imagery}` : sat.day ? `${sat.product}:${sat.day}` : sat.product;
  useEffect(() => {
    const q = new URLSearchParams();
    q.set("region", region);
    if (port != null) q.set("port", String(port));
    q.set("var", variable);
    q.set("lead", String(lead));
    q.set("layer", layerParam);
    if (showAge && group === "satellite") q.set("age", "1");
    if (showSensor && group === "satellite") q.set("sensor", "1");
    if (flow === "particles" && group === "currents") q.set("flow", "particles");
    if (underlay && group === "currents") q.set("chl", "1");
    if (inspect) q.set("inspect", `${inspect.lat.toFixed(4)},${inspect.lon.toFixed(4)}`);
    window.history.replaceState(null, "", `?${q.toString()}`);
  }, [region, port, variable, lead, layerParam, inspect, showAge, showSensor, group, flow, underlay]);

  const detailOpen = port != null || inspect != null;
  const padding = useCallback(() => {
    if (window.innerWidth >= 1024) {
      const dockH = dockRef.current?.getBoundingClientRect().height ?? 240;
      return { top: 40, bottom: dockH + 40, left: 368, right: detailOpen ? 432 : 56 };
    }
    return { top: 80, bottom: Math.round(window.innerHeight * 0.42), left: 16, right: 48 };
  }, [detailOpen]);

  const goRegion = useCallback(
    (id: string) => {
      setRegion(id);
      setPort(null);
      setInspect(null);
      setNavOpen(false);
      const r = regions.find((x) => x.id === id);
      if (r && mapRef.current) mapRef.current.fitBounds(r.bounds, { padding: padding(), duration: matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 800 });
    },
    [regions, padding],
  );

  const selectPort = useCallback(
    (code: number) => {
      setPort(code);
      setInspect(null);
      setNavOpen(false);
      const f = ports?.features.find((x) => x.properties.port_code === code);
      if (f) {
        if (f.properties.region) setRegion(f.properties.region);
        const r = regions.find((x) => x.id === f.properties.region);
        const pad = { ...padding(), right: window.innerWidth >= 1024 ? 432 : 16 };
        if (r && mapRef.current) mapRef.current.fitBounds(r.bounds, { padding: pad, duration: 800 });
        else {
          const [lon, lat] = f.geometry.coordinates as number[];
          mapRef.current?.flyTo({ center: [lon, lat], zoom: Math.max(mapRef.current.getZoom(), 9), padding: pad, duration: 800 });
        }
        pendingFly.current = mapRef.current ? null : code;
      }
    },
    [ports, padding, regions],
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
    return { id: satLayer.layer_id, template: t.relative ? artifactUrl(baseUrl, t.url_template) : t.url_template, minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat };
  }, [satLayer, baseUrl, showAge, showSensor, combined]);
  // categorical overlays that replace the chlorophyll colours: observation age, or which sensor
  const satelliteAge: TileRaster | null = useMemo(() => {
    const t = combined ? null : showAge ? (satLayer?.composite?.age_tiles ?? satLayer?.multisensor?.age_tiles) : showSensor ? satLayer?.multisensor?.sensor_tiles : null;
    if (!t || !satLayer) return null;
    return { id: `${satLayer.layer_id}-${showAge ? "age" : "sensor"}`, template: artifactUrl(baseUrl, t.url_template), minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat };
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

  const stamp = stampLines({ group, forecast: forecastLayer, run, satellite: satLayer, imagery: imageryLayer, currents: group === "currents" ? curLayer : null, combined, now });
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

  const detail =
    port != null ? (
      intel && portIntel ? (
        <div className="space-y-5">
          <PortPanel port={intel} coll={portIntel} manifest={manifest} official={official} verification={verification} lead={lead} onLead={setLead} variable={variable} onVariable={setVariable} now={now} onClose={() => setPort(null)} />
          <SatelliteNear manifest={manifest} baseUrl={baseUrl} lat={intel.lat} lon={intel.lon} place={intel.display_name} />
          {curLayer && <CurrentsNear layer={curLayer} baseUrl={baseUrl} lat={intel.lat} lon={intel.lon} place={intel.display_name} />}
        </div>
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
        <Inspector manifest={manifest} baseUrl={baseUrl} point={inspect} lead={lead} onClose={() => setInspect(null)} official={official} verification={verification} />
        <SatelliteNear manifest={manifest} baseUrl={baseUrl} lat={inspect.lat} lon={inspect.lon} place="this point" />
        {curLayer && <CurrentsNear layer={curLayer} baseUrl={baseUrl} lat={inspect.lat} lon={inspect.lon} place="this point" />}
      </div>
    ) : null;

  const nav = (
    <NavCard
      regions={regions}
      region={region}
      onRegion={goRegion}
      ports={portIntel?.ports ?? []}
      portsError={portIntel ? null : (portIntelError ?? "unavailable")}
      port={port}
      onPort={selectPort}
      variable={variable}
      lead={lead}
      palette={forecastLayer?.palette}
      regionNotices={regionNotices}
      showForecast={group === "forecast"}
    />
  );
  const dock = (
    <LayerDock
      manifest={manifest}
      group={group}
      onGroup={setGroup}
      run={run}
      charmStatus={sourceStatus(manifest, "charm")}
      satStatus={sourceStatus(manifest, "satellite_chl")}
      gibsStatus={sourceStatus(manifest, "gibs_chl")}
      variable={variable}
      onVariable={setVariable}
      lead={lead}
      onLead={setLead}
      sat={sat}
      onSat={setSat}
      showAge={showAge}
      onShowAge={setShowAge}
      showSensor={showSensor}
      onShowSensor={setShowSensor}
      cur={cur}
      onCur={setCur}
      flow={flow}
      onFlow={setFlow}
      curStatus={sourceStatus(manifest, "hf_radar")}
      reducedMotion={reducedMotion}
      underlay={underlay}
      onUnderlay={setUnderlay}
      regionId={region === STATEWIDE.id ? "monterey_bay" : region}
      regionLabel={region === STATEWIDE.id ? "Monterey Bay" : regionLabel}
      now={now}
      compact={!wide}
    />
  );

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
      initialPadding={wide ? { top: 40, bottom: 300, left: 368, right: 56 } : { top: 80, bottom: 360, left: 16, right: 48 }}
      onPort={selectPort}
      onPoint={(pt) => {
        setPort(null);
        setInspect(pt);
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

  // The map keeps the same position in the tree for both layouts so switching between
  // desktop and mobile never re-creates it.
  return (
    <div className="relative min-h-0 flex-1" data-testid="map">
      <div className="absolute inset-0">{map}</div>
      {wide ? (
        <>
          <div className="absolute left-4 top-4 z-10 flex w-[320px]" style={{ maxHeight: "calc(100% - 300px)" }}>
            {nav}
          </div>
          <MapStamp lines={stamp} className="absolute top-4 z-10 max-w-[calc(100%-760px)]" style={{ left: 352 }} />
          <div ref={dockRef} className="absolute bottom-4 left-4 z-10" style={{ width: detailOpen ? "min(640px, calc(100% - 32px - 400px - 16px))" : "min(640px, calc(100% - 32px))" }}>
            {dock}
          </div>
          {detail && (
            <aside aria-label="Selected place" data-testid="detail-panel" className="theme-paper absolute bottom-4 right-4 top-4 z-10 w-[384px] overflow-y-auto rounded-2xl bg-surface p-5 text-ink shadow-[0_1px_2px_rgba(6,17,30,0.12),0_8px_24px_rgba(6,17,30,0.18)] [scrollbar-width:thin]">
              {detail}
            </aside>
          )}
        </>
      ) : (
        <>
          <button
            onClick={() => setNavOpen(true)}
            data-testid="place-button"
            className="theme-paper absolute left-3 right-3 top-3 z-10 flex h-11 items-center gap-2 rounded-full bg-surface px-4 text-left text-[15px] font-medium text-ink shadow-[0_8px_24px_rgba(6,17,30,0.25)]"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4 text-ink-3" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden>
              <circle cx="11" cy="11" r="6.5" />
              <path d="m20 20-4.2-4.2" />
            </svg>
            <span className="flex-1 truncate">{port != null ? (intel?.display_name ?? "Port") : regionLabel}</span>
            <span aria-hidden className="text-ink-3">▾</span>
          </button>
          {!navOpen && <MapStamp lines={stamp} className="absolute left-3 top-[60px] z-10 max-w-[calc(100%-24px)]" />}
          {navOpen && (
            <div className="theme-paper absolute inset-x-0 bottom-0 top-3 z-30 flex flex-col overflow-hidden rounded-t-2xl bg-surface text-ink shadow-[0_-8px_24px_rgba(6,17,30,0.35)]" role="dialog" aria-label="Places">
              <div className="flex items-center justify-between px-4 pb-1 pt-2.5">
                <span className="font-display text-[20px] font-medium">Places</span>
                <button onClick={() => setNavOpen(false)} aria-label="Close places" data-testid="places-close" className="grid h-10 w-10 place-items-center text-ink-3">
                  <Icon name="close" className="h-5 w-5" />
                </button>
              </div>
              <div className="flex min-h-0 flex-1 flex-col [&>section]:rounded-none [&>section]:shadow-none">{nav}</div>
            </div>
          )}
          {navOpen ? null : detail ? (
            <MobileSheet
              peek={port != null ? `${intel?.display_name ?? "Port"} · ${intel?.official_relations.length ?? 0} official notices · ${OFFICIAL_STATUS.verification[verification?.state ?? "unavailable"].toLowerCase()}` : "Point details"}
              expandTo="half"
              key={`d-${port}-${inspect?.lat}`}
            >
              <div className="theme-paper space-y-4 rounded-lg bg-surface p-3 text-ink">{detail}</div>
            </MobileSheet>
          ) : (
            <div ref={dockRef} className="absolute inset-x-0 bottom-0 z-20" data-testid="mobile-legend">
              <button
                type="button"
                onClick={(e) => openDrawer(e.currentTarget)}
                data-testid="mobile-official"
                className="theme-paper mx-3 mb-2 flex w-[calc(100%-24px)] items-center gap-2 rounded-xl border border-official-line bg-official-bg px-3 py-2 text-left text-[14px] shadow-[0_4px_16px_rgba(6,17,30,0.3)]"
              >
                <Icon name="shield" className="h-4 w-4 text-official" />
                <span className="flex-1 text-ink">
                  <b className="font-semibold text-official-ink">{official ? official.registry.records.filter((r) => r.status === "active").length : "?"} official notices</b> in California
                </span>
                <span className="text-[12px] font-medium text-official-ink">{verification ? OFFICIAL_STATUS.verification[verification.state] : "Checking…"} ›</span>
              </button>
              <div className="[&>section]:rounded-b-none">{dock}</div>
            </div>
          )}
        </>
      )}
      {officialError && !official && <p className="sr-only">Official notices unavailable: {officialError}</p>}
      {portsError && !ports && <p className="sr-only">Ports unavailable: {portsError}</p>}
    </div>
  );
}
