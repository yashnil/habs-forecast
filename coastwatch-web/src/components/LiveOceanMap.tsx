"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import { CHARM_LEADS, CHARM_VARIABLES, artifactUrl, charmLayer, charmRun, chlorophyllLayers, sourceStatus, type CharmVariable } from "@/lib/layers";
import { useNow } from "@/lib/useNow";
import { OfficialStatusCard } from "@/components/panels/OfficialStatusCard";
import { ForecastPanel } from "@/components/panels/ForecastPanel";
import { ObservationPanel } from "@/components/panels/ObservationPanel";
import { Inspector, type InspectPoint } from "@/components/map/Inspector";
import type { Raster } from "@/components/map/MapCanvas";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), {
  ssr: false,
  loading: () => <div className="h-full w-full animate-pulse bg-[var(--cw-water)]" aria-label="Loading map" />,
});

type Props = { manifest: Manifest; ports: PortsCollection | null; portsError: string | null; baseUrl: string };

/** One raster on the map at a time: the official forecast, one satellite product, or none. */
type RasterChoice = { kind: "forecast" } | { kind: "observation"; layerId: string } | { kind: "none" };

function readUrl() {
  if (typeof window === "undefined") return new URLSearchParams();
  return new URLSearchParams(window.location.search);
}

export function LiveOceanMap({ manifest, ports, portsError, baseUrl }: Props) {
  const now = useNow();
  const run = charmRun(manifest);
  const chl = chlorophyllLayers(manifest);

  const firstLead = run?.leads_available.includes(1) ? 1 : (run?.leads_available[0] ?? 0);
  const [variable, setVariable] = useState<CharmVariable>("particulate_domoic");
  const [lead, setLead] = useState<number>(firstLead);
  const [choice, setChoice] = useState<RasterChoice>({ kind: "forecast" });
  const [opacity, setOpacity] = useState(0.85);
  const [showPorts, setShowPorts] = useState(true);
  const [inspect, setInspect] = useState<InspectPoint | null>(null);

  // initial state from the URL (shareable links, and deterministic e2e tests)
  useEffect(() => {
    const q = readUrl();
    const v = q.get("var");
    if (v && (CHARM_VARIABLES as readonly string[]).includes(v)) setVariable(v as CharmVariable);
    const l = Number(q.get("lead"));
    if (q.has("lead") && (CHARM_LEADS as readonly number[]).includes(l)) setLead(l);
    const layer = q.get("layer");
    if (layer === "none") setChoice({ kind: "none" });
    else if (layer && chl.some((c) => c.layer_id === layer)) setChoice({ kind: "observation", layerId: layer });
    const ins = q.get("inspect")?.split(",").map(Number);
    if (ins && ins.length === 2 && ins.every(Number.isFinite)) setInspect({ lat: ins[0], lon: ins[1], port: null });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const q = new URLSearchParams();
    q.set("var", variable);
    q.set("lead", String(lead));
    q.set("layer", choice.kind === "forecast" ? "forecast" : choice.kind === "none" ? "none" : choice.layerId);
    if (inspect) q.set("inspect", `${inspect.lat.toFixed(4)},${inspect.lon.toFixed(4)}`);
    window.history.replaceState(null, "", `?${q.toString()}`);
  }, [variable, lead, choice, inspect]);

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

  const onInspect = useCallback((p: InspectPoint) => setInspect(p), []);

  return (
    <div className="flex min-h-0 flex-1 flex-col lg:relative">
      <div className="relative h-[58vh] min-h-[360px] w-full lg:absolute lg:inset-0 lg:h-auto" data-testid="map">
        <MapCanvas raster={raster} opacity={opacity} ports={ports} showPorts={showPorts} inspect={inspect} onInspect={onInspect} />
        {inspect && (
          <div className="absolute right-3 top-3 z-10 hidden w-[320px] lg:block">
            <Inspector manifest={manifest} baseUrl={baseUrl} point={inspect} lead={lead} onClose={() => setInspect(null)} />
          </div>
        )}
        {!inspect && run && (
          <p className="pointer-events-none absolute right-3 top-3 z-10 hidden rounded-lg border border-hairline bg-surface/90 px-3 py-1.5 text-[11.5px] text-ink-2 backdrop-blur lg:block">
            Click the ocean or a port to read forecast values
          </p>
        )}
      </div>

      <aside
        aria-label="Map layers and sources"
        className="z-10 space-y-3 p-3 lg:absolute lg:bottom-3 lg:left-3 lg:top-3 lg:w-[392px] lg:overflow-y-auto lg:rounded-2xl lg:border lg:border-hairline lg:bg-page/80 lg:p-3 lg:shadow-2xl lg:backdrop-blur-md"
      >
        {inspect && (
          <div className="lg:hidden">
            <Inspector manifest={manifest} baseUrl={baseUrl} point={inspect} lead={lead} onClose={() => setInspect(null)} />
          </div>
        )}
        <OfficialStatusCard />
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
        <section className="space-y-2 rounded-xl border border-hairline bg-surface p-4" aria-labelledby="ports-h">
          <div className="flex items-center justify-between">
            <h2 id="ports-h" className="text-[13px] font-semibold text-ink">
              Landing ports
            </h2>
            <label className="flex cursor-pointer items-center gap-2 text-[12px] text-ink-2">
              <input type="checkbox" checked={showPorts} onChange={(e) => setShowPorts(e.target.checked)} disabled={!ports} />
              Show
            </label>
          </div>
          <p className="text-[11.5px] leading-snug text-ink-3">
            {ports
              ? `${ports.features.length} ports from CDFW's landing-port reference layer (ds3081). ${ports.caveats[0]}`
              : `Ports unavailable${portsError ? `: ${portsError}` : "."}`}
          </p>
        </section>
      </aside>
    </div>
  );
}
