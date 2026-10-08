"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import { CHARM_LEADS, CHARM_VARIABLES, artifactUrl, charmLayer, charmRun, chlorophyllLayers, leadLabel, sourceStatus, type CharmVariable } from "@/lib/layers";
import { officialVerification } from "@/lib/official";
import { useNow } from "@/lib/useNow";
import { OfficialSummary } from "@/components/official/Official";
import { ForecastPanel } from "@/components/panels/ForecastPanel";
import { ObservationPanel } from "@/components/panels/ObservationPanel";
import { PortPanel } from "@/components/port/PortPanel";
import { Inspector, type InspectPoint } from "@/components/map/Inspector";
import { MobileSheet } from "@/components/MobileSheet";
import type { Raster } from "@/components/map/MapCanvas";
import { OFFICIAL_STATUS } from "@/content/copy";
import { colorAt } from "@/components/ui/ProbabilityLegend";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-[var(--cw-water)]" aria-label="Loading map" />,
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

type RasterChoice = { kind: "forecast" } | { kind: "observation"; layerId: string } | { kind: "none" };

const STATEWIDE = { id: "california", label: "All California", bounds: [[-125.0, 32.3], [-116.9, 42.1]] as [[number, number], [number, number]] };
const DEFAULT_REGION = "monterey_bay";

export function LiveOceanMap({ manifest, ports, portsError, official, officialError, portIntel, portIntelError, baseUrl, initialParams }: Props) {
  const now = useNow();
  const run = charmRun(manifest);
  const chl = chlorophyllLayers(manifest);
  const mapRef = useRef<MlMap | null>(null);
  const pendingFly = useRef<number | null>(null);
  const [wide, setWide] = useState(true);

  const regions = useMemo(
    () => [STATEWIDE, ...(ports?.regions ?? []).map((r) => ({ id: r.id, label: r.label, bounds: r.bounds as [[number, number], [number, number]] }))],
    [ports],
  );
  const initialQ = useMemo(() => new URLSearchParams(initialParams), [initialParams]);
  const initialRegion = regions.find((r) => r.id === initialQ.get("region")) ?? regions.find((r) => r.id === DEFAULT_REGION) ?? STATEWIDE;

  const firstLead = run?.leads_available.includes(1) ? 1 : (run?.leads_available[0] ?? 0);
  const [variable, setVariable] = useState<CharmVariable>("particulate_domoic");
  const [lead, setLead] = useState<number>(firstLead);
  const [choice, setChoice] = useState<RasterChoice>({ kind: "forecast" });
  const [opacity, setOpacity] = useState(0.7);
  const [showPorts, setShowPorts] = useState(true);
  const [showOfficial, setShowOfficial] = useState(true);
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
    const layer = q.get("layer");
    if (layer === "none") setChoice({ kind: "none" });
    else if (layer && chl.some((c) => c.layer_id === layer)) setChoice({ kind: "observation", layerId: layer });
    const p = Number(q.get("port"));
    if (q.has("port") && ports?.features.some((f) => f.properties.port_code === p)) {
      setPort(p);
      pendingFly.current = p; // fly once the map is ready
    }
    const ins = q.get("inspect")?.split(",").map(Number);
    if (ins && ins.length === 2 && ins.every(Number.isFinite)) setInspect({ lat: ins[0], lon: ins[1] });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const q = new URLSearchParams();
    q.set("region", region);
    if (port != null) q.set("port", String(port));
    q.set("var", variable);
    q.set("lead", String(lead));
    q.set("layer", choice.kind === "forecast" ? "forecast" : choice.kind === "none" ? "none" : choice.layerId);
    if (inspect) q.set("inspect", `${inspect.lat.toFixed(4)},${inspect.lon.toFixed(4)}`);
    window.history.replaceState(null, "", `?${q.toString()}`);
  }, [region, port, variable, lead, choice, inspect]);

  const padding = useCallback(
    () => (window.innerWidth >= 1024 ? { top: 56, bottom: 40, left: 400, right: port != null || inspect ? 452 : 40 } : { top: 56, bottom: Math.round(window.innerHeight * 0.56), left: 24, right: 24 }),
    [port, inspect],
  );

  const goRegion = useCallback(
    (id: string) => {
      setRegion(id);
      const r = regions.find((x) => x.id === id);
      if (r && mapRef.current) mapRef.current.fitBounds(r.bounds, { padding: padding(), duration: 900 });
    },
    [regions, padding],
  );

  const selectPort = useCallback(
    (code: number) => {
      setPort(code);
      setInspect(null);
      const f = ports?.features.find((x) => x.properties.port_code === code);
      if (f) {
        if (f.properties.region) setRegion(f.properties.region);
        const [lon, lat] = f.geometry.coordinates as number[];
        mapRef.current?.flyTo({ center: [lon, lat], zoom: Math.max(mapRef.current.getZoom(), 9.2), padding: { ...padding(), right: window.innerWidth >= 1024 ? 452 : 24 }, duration: 900 });
        pendingFly.current = mapRef.current ? null : code;
      }
    },
    [ports, padding],
  );

  const forecastLayer = charmLayer(manifest, variable, lead);
  const raster: Raster = useMemo(() => {
    if (choice.kind === "forecast" && forecastLayer?.image) {
      return { kind: "image", id: forecastLayer.layer_id, url: artifactUrl(baseUrl, forecastLayer.image.url), corners: forecastLayer.image.corners_lnglat };
    }
    if (choice.kind === "observation") {
      const l = chl.find((c) => c.layer_id === choice.layerId);
      if (l?.tiles) return { kind: "tiles", id: l.layer_id, template: l.tiles.url_template, maxzoom: l.tiles.max_native_zoom };
    }
    return null;
  }, [choice, forecastLayer, chl, baseUrl]);

  const verification = useMemo(() => (now ? officialVerification(official, sourceStatus(manifest, "official"), now) : null), [official, manifest, now]);
  const intel = port != null ? (portIntel?.ports.find((p) => p.port_code === port) ?? null) : null;
  const regionPorts = (portIntel?.ports ?? []).filter((p) => region === "california" || p.region === region);
  const initialBounds = initialRegion.bounds;

  const detail =
    port != null ? (
      intel && portIntel ? (
        <PortPanel
          port={intel}
          coll={portIntel}
          manifest={manifest}
          official={official}
          verification={verification}
          lead={lead}
          onLead={setLead}
          variable={variable}
          onVariable={setVariable}
          now={now}
          onClose={() => setPort(null)}
        />
      ) : (
        <div className="rounded-lg border border-hairline-strong bg-surface p-4 text-[12.5px] text-ink-2" data-testid="port-unavailable">
          Port summaries are unavailable{portIntelError ? ` (${portIntelError})` : ""}. {OFFICIAL_STATUS.missingNotOpen}
          <button onClick={() => setPort(null)} className="ml-2 underline">
            Close
          </button>
        </div>
      )
    ) : inspect ? (
      <Inspector manifest={manifest} baseUrl={baseUrl} point={inspect} lead={lead} onClose={() => setInspect(null)} official={official} verification={verification} />
    ) : null;

  const rail = (
    <>
      <OfficialSummary ds={official} v={verification} now={now} error={officialError} />
      <section className="space-y-2.5 rounded-lg border border-hairline bg-surface p-3.5" aria-labelledby="ports-h" data-testid="region-ports">
        <div className="flex items-baseline justify-between gap-2">
          <h2 id="ports-h" className="text-[13.5px] font-semibold tracking-tight text-ink">
            Ports · {regions.find((r) => r.id === region)?.label}
          </h2>
          <span className="text-[11px] text-ink-3">
            {leadLabel(lead)} · {variable === "pseudo_nitzschia" ? "bloom" : variable === "particulate_domoic" ? "particulate DA" : "cellular DA"}
          </span>
        </div>
        {!portIntel ? (
          <p className="text-[12px] text-ink-3">Port summaries unavailable{portIntelError ? `: ${portIntelError}` : "."}</p>
        ) : (
          <ul className="divide-y divide-[var(--cw-hairline)]">
            {regionPorts.map((p) => {
              const s = p.charm?.leads.find((l) => l.lead_days === lead)?.variables[variable];
              const nNotices = p.official_relations.length;
              const pal = forecastLayer?.palette;
              return (
                <li key={p.port_code}>
                  <button
                    onClick={() => selectPort(p.port_code)}
                    data-testid={`port-row-${p.port_code}`}
                    aria-current={port === p.port_code ? "true" : undefined}
                    className={`flex w-full items-center gap-2 px-1 py-1.5 text-left text-[12.5px] transition-colors hover:bg-surface-2 ${port === p.port_code ? "bg-surface-2" : ""}`}
                  >
                    <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: s?.median != null && pal ? colorAt(pal, s.median) : "var(--cw-surface-3)" }} aria-hidden />
                    <span className="flex-1 truncate text-ink">{p.display_name}</span>
                    <span className="text-[11px] text-[#ffcf85]" title="Official notices that may cover nearby waters">
                      {nNotices} notice{nNotices === 1 ? "" : "s"}
                    </span>
                    <span className="w-12 text-right font-semibold text-ink tabular">{s?.median != null ? `${Math.round(s.median * 100)}%` : "—"}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        <p className="text-[11px] text-ink-3">Median forecast probability within 15 km of each port (not at the dock).</p>
      </section>
      <ForecastPanel
        manifest={manifest}
        run={run}
        status={sourceStatus(manifest, "charm")}
        variable={variable}
        lead={lead}
        onVariable={setVariable}
        onLead={setLead}
        shown={choice.kind === "forecast"}
        onShow={(on) => setChoice(on ? { kind: "forecast" } : { kind: "none" })}
        opacity={opacity}
        onOpacity={setOpacity}
        now={now}
      />
      <ObservationPanel
        layers={chl}
        status={sourceStatus(manifest, "gibs_chl")}
        selected={choice.kind === "observation" ? choice.layerId : null}
        onSelect={(id) => setChoice(id ? { kind: "observation", layerId: id } : { kind: "forecast" })}
        now={now}
      />
      <section className="space-y-2 rounded-lg border border-hairline bg-surface p-3.5" aria-labelledby="ov-h" data-testid="overlays">
        <h2 id="ov-h" className="text-[13.5px] font-semibold tracking-tight text-ink">
          Overlays
        </h2>
        <label className="flex cursor-pointer items-start gap-2 text-[12.5px] text-ink-2">
          <input type="checkbox" className="mt-0.5" checked={showOfficial} onChange={(e) => setShowOfficial(e.target.checked)} disabled={!official} data-testid="toggle-official" />
          <span>
            <span className="font-medium text-ink">Official notice areas</span>
            <span className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-3">
              <span className="flex items-center gap-1">
                <span className="inline-block h-0 w-5 border-t-2 border-dashed border-[#ffb547]" /> county (official polygon)
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-0 w-5 border-t-2 border-[#ffb547]" /> official latitude limit
              </span>
            </span>
            <span className="mt-1 block text-[11px] text-ink-3">{OFFICIAL_STATUS.areaNote} Statewide notices are not drawn.</span>
            {official && official.geometry_errors.length > 0 && (
              <span className="mt-1 block text-[11px] text-warning">Some areas could not be drawn: {official.geometry_errors.join("; ")}</span>
            )}
          </span>
        </label>
        <label className="flex cursor-pointer items-start gap-2 text-[12.5px] text-ink-2">
          <input type="checkbox" className="mt-0.5" checked={showPorts} onChange={(e) => setShowPorts(e.target.checked)} disabled={!ports} />
          <span>
            <span className="font-medium text-ink">Landing ports</span>
            <span className="block text-[11px] text-ink-3">
              {ports ? `${ports.features.length} ports from CDFW (ds3081). ${ports.caveats[0]}` : `Ports unavailable${portsError ? `: ${portsError}` : "."}`}
            </span>
          </span>
        </label>
      </section>
    </>
  );

  const regionBar = (
    <nav aria-label="Regions" className="flex gap-1 overflow-x-auto" data-testid="region-bar">
      {regions.map((r) => (
        <button
          key={r.id}
          onClick={() => goRegion(r.id)}
          aria-pressed={region === r.id}
          data-testid={`region-${r.id}`}
          className={`whitespace-nowrap rounded-md border px-2.5 py-1 text-[12px] font-medium transition-colors ${
            region === r.id ? "border-accent/60 bg-surface-3 text-ink" : "border-hairline bg-surface text-ink-2 hover:text-ink"
          }`}
        >
          {r.label}
        </button>
      ))}
    </nav>
  );

  const map = (
    <MapCanvas
      raster={raster}
      opacity={opacity}
      ports={ports}
      showPorts={showPorts}
      officialGeometry={(official?.geometry as unknown as GeoJSON.FeatureCollection) ?? null}
      showOfficial={showOfficial}
      selectedPort={port}
      inspect={inspect}
      initialBounds={initialBounds}
      onPort={selectPort}
      onPoint={(p) => {
        setPort(null);
        setInspect(p);
      }}
      onReady={(m) => {
        mapRef.current = m;
        const code = pendingFly.current;
        if (code != null) {
          pendingFly.current = null;
          // let the initial camera settle before moving it
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
          <aside aria-label="Official notices, ports and map layers" className="absolute bottom-3 left-3 top-3 z-10 w-[372px] space-y-2.5 overflow-y-auto rounded-lg pr-1 [scrollbar-width:thin]">
            {rail}
          </aside>
          <div className="absolute left-[396px] right-3 top-3 z-10 flex justify-start">
            <div className="max-w-full rounded-lg border border-hairline bg-page/90 p-1">{regionBar}</div>
          </div>
          {detail ? (
            <aside aria-label="Details" className="absolute bottom-3 right-3 top-14 z-10 w-[424px] overflow-y-auto rounded-lg border border-hairline-strong bg-surface p-4 shadow-2xl [scrollbar-width:thin]" data-testid="detail-panel">
              {detail}
            </aside>
          ) : (
            <p className="pointer-events-none absolute right-3 top-16 z-10 rounded-md border border-hairline bg-surface px-3 py-1.5 text-[11.5px] text-ink-2">
              Select a port, or click the ocean to read forecast values
            </p>
          )}
        </>
      ) : (
        <>
          <div className="absolute left-2 right-2 top-2 z-10">{regionBar}</div>
          <MobileSheet
            peek={
              detail
                ? port != null
                  ? `${intel?.display_name ?? "Port"} · ${intel?.official_relations.length ?? 0} official notices · ${OFFICIAL_STATUS.verification[verification?.state ?? "unavailable"].toLowerCase()}`
                  : "Point details"
                : `${official ? official.registry.records.filter((r) => r.status === "active").length : "?"} official notices · ${verification ? OFFICIAL_STATUS.verification[verification.state] : "checking"}`
            }
            expandTo={detail ? "half" : undefined}
            key={detail ? `d-${port}-${inspect?.lat}` : "rail"}
          >
            <div className="space-y-3">
              {detail}
              {rail}
            </div>
          </MobileSheet>
        </>
      )}
    </div>
  );
}
