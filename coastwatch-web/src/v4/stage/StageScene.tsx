"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Layer, Marker, Source } from "react-map-gl/maplibre";
import type { Map as MlMap } from "maplibre-gl";
import MapCanvas, { type CurrentsOverlay } from "@/components/map/MapCanvas";
import { journeyTo } from "@/lib/camera";
import { loadFieldCached, type CurrentField } from "@/lib/currents";
import type { LayerArtifact } from "@/generated/schema";
import type { OceanStageProps } from "./OceanStage";
import type { StageData, StageMark } from "./types";

type SceneProps = OceanStageProps & {
  data: StageData;
  field: NonNullable<OceanStageProps["field"]>;
  currents: boolean;
  focus: string | null;
  port: number | null;
  interactive: boolean;
  visible: boolean;
  arrive: boolean;
  onShown: () => void;
  onFail: () => void;
  onFieldState: (s: "ready" | "failed" | "loading") => void;
  onFlow: (on: boolean) => void;
};

/** How much further out the map starts when it arrives. */
const ARRIVAL_ZOOM_OUT = 0.5;
const ARRIVAL_PITCH = 6;
/** A phone frame is narrow: it stands further back so the same water is in view. */
const PHONE_ZOOM_OUT = 0.6;
/** A map that has not drawn by this long stays hidden: the poster keeps standing in. */
const GIVE_UP_MS = 12000;
const IDLE_CAP_MS = 3500;

function bezier(x1: number, y1: number, x2: number, y2: number) {
  const f = (a: number, b: number, t: number) => 3 * a * t * (1 - t) ** 2 + 3 * b * t * t * (1 - t) + t ** 3;
  return (x: number) => {
    let lo = 0;
    let hi = 1;
    for (let i = 0; i < 20; i++) {
      const m = (lo + hi) / 2;
      if (f(x1, x2, m) < x) lo = m;
      else hi = m;
    }
    return f(y1, y2, (lo + hi) / 2);
  };
}
// softer than --nx-ease-emph on purpose: that curve finishes its travel inside the map's 480 ms fade-in, so the
// settle would happen while nothing can see it. This one leaves about a third of the travel for after the fade.
const easeSettle = bezier(0.25, 0.46, 0.2, 1);

function stageMs(): number {
  const v = getComputedStyle(document.documentElement).getPropertyValue("--nx-t-stage").trim();
  return v.endsWith("ms") ? parseFloat(v) : 1400;
}

/**
 * The map behind OceanStage (its own lazy chunk): the field, the notice lines, marks, and the measured currents
 * once the camera has settled. The camera arrives once, then stands; a new centre flies there.
 */
export default function StageScene(p: SceneProps) {
  const { data, field, currents, focus, port, interactive, visible, arrive, center, zoom, onShown, onFail, onFieldState, onFlow } = p;
  const pitch = Math.max(0, Math.min(50, p.pitch ?? 0));
  const bearing = p.bearing ?? 0;
  const [map, setMap] = useState<MlMap | null>(null);
  const [settled, setSettled] = useState(false);
  const [cur, setCur] = useState<CurrentField | null>(null);
  const reduced = useRef(false);
  const callbacks = useRef({ onShown, onFail, onFieldState, onFlow, onMap: p.onMap });
  useEffect(() => {
    callbacks.current = { onShown, onFail, onFieldState, onFlow, onMap: p.onMap };
  });

  const [camera] = useState(() => {
    reduced.current = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const on = arrive && !reduced.current;
    const box = document.querySelector<HTMLElement>('[data-testid="' + (p.testid ?? "ocean-stage") + '"]');
    const w = box?.clientWidth ?? window.innerWidth;
    const h = box?.clientHeight ?? window.innerHeight;
    const targetZoom = w < 768 ? (p.phoneZoom ?? zoom - PHONE_ZOOM_OUT) : zoom;
    // the subject sits in the clear part of the frame, above and right of the words over the stage's lower left
    const padding = { top: 0, right: 0, left: w >= 1024 ? Math.round(w * 0.14) : 0, bottom: Math.round(h * 0.3) };
    return { center, zoom: on ? targetZoom - ARRIVAL_ZOOM_OUT : targetZoom, targetZoom, pitch: on ? Math.max(0, pitch - ARRIVAL_PITCH) : pitch, bearing, padding };
  });

  // shown once the map has drawn, then the camera settles (once), then the currents begin: one thing moves at a time
  useEffect(() => {
    if (!map) return;
    let done = false;
    const giveUp = window.setTimeout(() => !done && callbacks.current.onFail(), GIVE_UP_MS);
    const start = () => {
      if (done) return;
      done = true;
      window.clearTimeout(giveUp);
      window.clearTimeout(cap);
      callbacks.current.onShown();
      const moved = arrive && !reduced.current;
      if (!moved) return setSettled(true);
      map.easeTo({ zoom: camera.targetZoom, pitch, duration: stageMs(), easing: easeSettle, essential: false });
      map.once("moveend", () => setSettled(true));
    };
    const cap = window.setTimeout(start, IDLE_CAP_MS);
    if (map.loaded() && !map.isMoving()) start();
    else map.once("idle", start);
    const lost = () => callbacks.current.onFail();
    map.getCanvas().addEventListener("webglcontextlost", lost);
    return () => {
      done = true;
      window.clearTimeout(giveUp);
      window.clearTimeout(cap);
      map.off("idle", start);
      map.getCanvas().removeEventListener("webglcontextlost", lost);
    };
    // once per map
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map]);

  // a new place: fly there (reduced motion jumps; lib/camera)
  const targetZoom = camera.targetZoom;
  const first = useRef(true);
  const [lon, lat] = center;
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    if (map) journeyTo(map, { center: [lon, lat], zoom: targetZoom });
  }, [map, lon, lat, targetZoom]);

  const curLayer = currents ? data.currents?.layer : null;
  useEffect(() => {
    if (!curLayer) return;
    let live = true;
    loadFieldCached(data.baseUrl, curLayer as LayerArtifact)
      .then((f) => live && setCur(f))
      .catch(() => live && setCur(null));
    return () => {
      live = false;
    };
  }, [curLayer, data.baseUrl]);

  const flow: CurrentsOverlay | null = useMemo(
    () =>
      cur && curLayer && settled
        ? { id: curLayer.layer_id, field: cur, mode: "flow", flow: { paused: !visible, still: reduced.current, colour: false, coarser: false } }
        : null,
    [cur, curLayer, settled, visible],
  );
  useEffect(() => {
    callbacks.current.onFlow(!!flow);
  }, [flow]);

  const fieldRaster = field === "none" ? null : data[field];
  const lines = p.lines ?? data.official;
  const lit = focus ?? "";
  const litFilter = useMemo(() => ["any", ["==", ["get", "id"], lit], ["in", lit, ["to-string", ["coalesce", ["get", "record_ids"], ""]]]] as unknown as import("maplibre-gl").FilterSpecification, [lit]);

  return (
    <MapCanvas
      // a field the dataset does not have is simply not drawn (the caption comes from the same place, so it goes too)
      forecast={field === "charm" ? ((fieldRaster as StageData["charm"])?.raster ?? null) : null}
      satellite={field === "chlorophyll" ? ((fieldRaster as StageData["chlorophyll"])?.raster ?? null) : null}
      satelliteAge={null}
      imagery={null}
      outlook={field === "outlook" ? ((fieldRaster as StageData["outlook"])?.raster ?? null) : null}
      sst={field === "sst" ? ((fieldRaster as StageData["sst"])?.raster ?? null) : null}
      currents={flow}
      ports={data.ports}
      showPorts
      officialGeometry={lines}
      showOfficial
      selectedPort={port}
      inspect={null}
      chrome="none"
      initialBounds={[[center[0] - 0.5, center[1] - 0.5], [center[0] + 0.5, center[1] + 0.5]]}
      initialPadding={0}
      initialCamera={camera}
      interactive={interactive}
      smoothField
      onPort={p.onPort ?? (() => {})}
      onPoint={p.onPoint ?? (() => {})}
      onReady={(m) => {
        setMap(m);
        callbacks.current.onMap?.(m);
      }}
      onStatus={({ field: f }) => callbacks.current.onFieldState(f === "failed" ? "failed" : f === "ready" ? "ready" : "loading")}
    >
      {lines && lit ? (
        <Source id="stage-lit" type="geojson" data={lines}>
          <Layer id="stage-lit-halo" type="line" filter={litFilter} layout={{ "line-cap": "round", "line-join": "round" }} paint={{ "line-color": "#06111e", "line-width": 6, "line-opacity": 0.55 }} />
          <Layer id="stage-lit-line" type="line" filter={litFilter} layout={{ "line-cap": "round", "line-join": "round" }} paint={{ "line-color": "#f6bb5c", "line-width": 3.2 }} />
        </Source>
      ) : null}
      {(p.marks ?? []).map((m: StageMark) => (
        <Marker key={m.id} longitude={m.lon} latitude={m.lat} anchor="center">
          <span className="nx-stage-mark" data-kind={m.kind} data-lit={focus === m.id || undefined} aria-hidden>
            <i />
            {m.label ? <b>{m.label}</b> : null}
          </span>
        </Marker>
      ))}
    </MapCanvas>
  );
}
