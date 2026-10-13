"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Map, { Layer, Marker, NavigationControl, ScaleControl, Source, type MapLayerMouseEvent } from "react-map-gl/maplibre";
import { setWorkerUrl, type ExpressionSpecification, type GeoJSONSource, type Map as MlMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { PortsCollection } from "@/generated/schema";
import type { LayerArtifact } from "@/generated/schema";
import {
  ARROW_LENGTHS,
  arrowImage,
  BASEMAP_STYLE,
  BEFORE_OVERLAY_ID,
  CA_BOUNDS,
  NODATA_FALLBACK_LAYER_ID,
  NODATA_LAYER_ID,
  noDataImage,
  SPEED_CLASSES,
  TINT_LAYER_IDS,
  WATER_FALLBACK_LAYER_IDS,
  WATER_FALLBACK_SOURCE,
  transformRequest,
} from "@/lib/basemap";
import type { CurrentField } from "@/lib/currents";
import { GRID_TILE_SIZE } from "@/lib/gridTiles";
import { gridTiles } from "@/lib/gridTileProtocol";
import { indexedTiles } from "@/lib/tileIndex";
import { loadWaterFallback, waterGeoJSON } from "@/lib/waterFallback";
import { onBasemapSlow } from "@/lib/waterMask";
import FlowParticles from "./FlowParticles";
import { CountyControl, CountyLayers, useCountyPrefs } from "./Counties";

// MapLibre's module worker is copied to public/ by scripts/copy-maplibre-worker.mjs
setWorkerUrl("/vendor/maplibre/maplibre-gl-worker.mjs");

/** C-HARM: one Mercator image placed by its corners. */
export type ImageRaster = {
  id: string;
  url: string;
  corners: number[][];
  /** drawn from the value grid when present (exact cell edges); the published image otherwise */
  grid?: { layer: LayerArtifact; url: string } | null;
};
/** XYZ tiles: CoastWatch-rendered satellite tiles or third-party imagery. */
export type TileRaster = { id: string; template: string; minzoom: number; maxzoom: number; bounds?: number[] | null; indexUrl?: string | null };

export type MapPoint = { lat: number; lon: number };
/** Observed currents for one hour (or the 24 h mean): arrows from the published grids, or particles. */
export type CurrentsOverlay = { id: string; features: GeoJSON.FeatureCollection; field: CurrentField; mode: "arrows" | "particles"; coarser?: boolean };

type Props = {
  forecast: ImageRaster | null;
  satellite: TileRaster | null;
  satelliteAge: TileRaster | null;
  imagery: TileRaster | null;
  currents?: CurrentsOverlay | null;
  ports: PortsCollection | null;
  showPorts: boolean;
  officialGeometry: GeoJSON.FeatureCollection | null;
  showOfficial: boolean;
  selectedPort: number | null;
  inspect: MapPoint | null;
  initialBounds: [[number, number], [number, number]];
  initialPadding: { top: number; bottom: number; left: number; right: number } | number;
  onPort: (portCode: number) => void;
  onPoint: (p: MapPoint) => void;
  onReady: (map: MlMap) => void;
  /** pointer over the map (desktop hover readout); null when it leaves */
  onHover?: (p: MapPoint | null, screen: { x: number; y: number } | null) => void;
  /** C-HARM model cell edges, drawn faintly from z8.5 so its 3 km cells read as cells */
  cellEdges?: GeoJSON.FeatureCollection | null;
  /** fade new data rasters in (switching layer); time steps always cut, never cross-fade */
  fade?: boolean;
  /**
   * My Coast: where each fact was obtained. Features carry properties.kind: "target" (the chosen
   * waters), "station" (a monitoring station, with name), "cell" (a C-HARM cell with a value) and
   * "link" (from the waters to that cell). Drawn above the data, below the ports.
   */
  overlay?: GeoJSON.FeatureCollection | null;
};

const OFFICIAL = "#f6bb5c";
const MODEL_MARK = "#b58cf0";
const MEASURED_MARK = "#4fd1ae";
// provenance threads: one layer per source so they draw out one after another (model, measured,
// official), 520 ms each with a 90 ms stagger; white, never a data colour
const THREADS = [
  { to: "cell", color: "rgba(233,239,246,0.9)" },
  { to: "station", color: "rgba(233,239,246,0.9)" },
  { to: "limit", color: "rgba(246,187,92,0.95)" },
] as const;
const THREAD_MS = 520;
const THREAD_STAGGER = 90;
/** A line drawn from its start up to `p` (0..1) of its length. */
function threadGradient(p: number, color: string): ExpressionSpecification {
  if (p >= 1) return ["interpolate", ["linear"], ["line-progress"], 0, color, 1, color];
  const q = Math.max(0.0001, p);
  return ["interpolate", ["linear"], ["line-progress"], 0, color, q, color, Math.min(1, q + 0.0001), "rgba(0,0,0,0)"];
}
// [level, minzoom, maxzoom]: one arrow per observed 2 km radar cell from z8.5 (M5); zoomed
// out, only cells on every 2nd, 4th or 8th row and column carry one. Arrows never move
// between zooms, more appear.
const ARROW_ZOOMS: [number, number, number][] = [
  [8, 0, 6.3],
  [4, 6.3, 7.3],
  [2, 7.3, 8.5],
  [1, 8.5, 24],
];
// one glyph per speed class (length grows with speed, same line width), matching the legend
const ARROW_IMAGE = ["step", ["get", "speed"], "cw-arrow-0", ...SPEED_CLASSES.slice(1).flatMap((v, i) => [v, `cw-arrow-${i + 1}`])] as unknown as ExpressionSpecification;
function addArrows(m: MlMap) {
  ARROW_LENGTHS.forEach((len, i) => {
    if (!m.hasImage(`cw-arrow-${i}`)) m.addImage(`cw-arrow-${i}`, arrowImage(len), { pixelRatio: 2 });
  });
}
// Every data raster is opaque and nearest-sampled: a pixel is a real source cell, and the
// legend colours are exactly the colours on the map (design reset rev. 2, §5.1).
const RASTER_PAINT = { "raster-opacity": 1, "raster-resampling": "nearest", "raster-fade-duration": 0 } as const;
const RASTER_FADE = { ...RASTER_PAINT, "raster-fade-duration": 220 } as const;

export default function MapCanvas(p: Props) {
  const [wide] = useState(() => typeof window !== "undefined" && window.innerWidth >= 1024);
  const { onPort, onPoint } = p;
  const [map, setMap] = useState<MlMap | null>(null);
  const hasRaster = !!(p.forecast || p.satellite || p.satelliteAge);
  // the fade is decided once per raster, when it first appears (a layer switch fades in; a new
  // time step cuts) and kept for that raster's life, so a later re-render never changes it
  const fadeOf = useRef(new globalThis.Map<string, boolean>());
  const paintFor = (id: string | undefined) => {
    if (!id) return RASTER_PAINT;
    if (!fadeOf.current.has(id)) fadeOf.current.set(id, !!p.fade);
    return fadeOf.current.get(id) ? RASTER_FADE : RASTER_PAINT;
  };
  const fg = p.forecast?.grid;
  const forecastTiles = useMemo(() => (fg ? gridTiles(fg.layer, fg.url) : null), [fg]);
  const [counties, setCounties] = useCountyPrefs();

  // When the basemap's tiles fail (tile server down, offline) or are late (a data tile stopped
  // waiting for them), draw the bundled water instead, so the sea, the coastline and the no-data
  // dots still read (lib/waterFallback); data rasters are clipped to the same water by
  // lib/waterMask. "slow" returns to "ok" once the basemap's tiles for the view have arrived.
  const [basemap, setBasemap] = useState<"ok" | "slow" | "down">("ok");
  const [fallbackReady, setFallbackReady] = useState(false);
  useEffect(() => onBasemapSlow(() => setBasemap((b) => (b === "ok" ? "slow" : b))), []);
  useEffect(() => {
    if (!map || basemap !== "slow") return;
    const arrived = (e: { sourceId?: string }) => {
      if (e.sourceId === "omt" && map.isSourceLoaded("omt")) setBasemap((b) => (b === "slow" ? "ok" : b));
    };
    map.on("sourcedata", arrived);
    return () => {
      map.off("sourcedata", arrived);
    };
  }, [map, basemap]);
  const fallbackOn = basemap !== "ok";
  useEffect(() => {
    if (!map || !fallbackOn) return;
    let live = true;
    loadWaterFallback().then((g) => {
      const src = map.getSource(WATER_FALLBACK_SOURCE) as GeoJSONSource | undefined;
      if (!live || !g || !src) return;
      if (!fallbackReady) src.setData(waterGeoJSON(g));
      setFallbackReady(true);
    });
    return () => {
      live = false;
    };
  }, [map, fallbackOn, fallbackReady]);
  const showFallback = fallbackOn && fallbackReady;

  // Has the data field arrived? Until a raster's first tiles in view have loaded there is nothing
  // for a gap to differ from: the hatch would cover the whole sea for a moment and read as "no
  // data" (signature P11, the hatch flash). The basemap stays calm until then; loading words
  // appear only after 400 ms. Keyed by the raster ids, so a new day starts unloaded again (the
  // swap dips through the basemap; two days are never blended).
  const rasterKey = [p.forecast?.id, p.satellite?.id, p.satelliteAge?.id].filter(Boolean).join("|");
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const [failedKey, setFailedKey] = useState<string | null>(null);
  const fieldReady = hasRaster && loadedKey === rasterKey;
  const fieldFailed = hasRaster && !fieldReady && failedKey === rasterKey;
  useEffect(() => {
    if (!map || !hasRaster) return;
    const ids = ["forecast", "satellite", "satellite-age"].filter((id) => (id === "forecast" ? p.forecast : id === "satellite" ? p.satellite : p.satelliteAge));
    const check = () => {
      if (ids.every((id) => map.getSource(id) && map.isSourceLoaded(id))) setLoadedKey(rasterKey);
    };
    // a failure is said in words, never hidden behind a spinner (signature handoff, rule 8)
    const failed = (e: unknown) => {
      const id = (e as { sourceId?: string }).sourceId;
      if (id && ids.includes(id)) setFailedKey(rasterKey);
    };
    check();
    map.on("sourcedata", check);
    map.on("idle", check);
    map.on("error", failed);
    return () => {
      map.off("sourcedata", check);
      map.off("idle", check);
      map.off("error", failed);
    };
  }, [map, hasRaster, rasterKey, p.forecast, p.satellite, p.satelliteAge]);

  // The no-data hatch shows only under a data field that has arrived: with currents alone (or
  // nothing) there is no "value" for a gap to be confused with. The bundled-water layers show
  // only while the basemap is slow or down.
  useEffect(() => {
    if (!map) return;
    // idempotent: every style change fires "styledata" again
    const apply = () => {
      const want: [string, boolean][] = [
        [NODATA_LAYER_ID, fieldReady],
        [NODATA_FALLBACK_LAYER_ID, fieldReady && showFallback],
        ...WATER_FALLBACK_LAYER_IDS.map((id): [string, boolean] => [id, showFallback]),
      ];
      for (const [id, on] of want) {
        if (!map.getLayer(id)) continue;
        const v = on ? "visible" : "none";
        if (map.getLayoutProperty(id, "visibility") !== v) map.setLayoutProperty(id, "visibility", v);
      }
      // depth-tinted seafloor only when no data colours are on the map: a cross-fade, not a cut
      for (const id of TINT_LAYER_IDS) {
        if (!map.getLayer(id)) continue;
        const o = hasRaster ? 0 : 1;
        if (map.getPaintProperty(id, "raster-opacity") !== o) map.setPaintProperty(id, "raster-opacity", o);
      }
    };
    apply();
    map.on("styledata", apply);
    return () => {
      map.off("styledata", apply);
    };
  }, [map, hasRaster, fieldReady, showFallback]);
  const { onHover } = p;

  // threads draw out once per selection (reduced motion: drawn at once); a new selection restarts
  const overlay = p.overlay;
  useEffect(() => {
    if (!map || !overlay) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    const set = (t: number) => {
      THREADS.forEach(({ to, color }, i) => {
        const id = `coast-thread-${to}`;
        if (!map.getLayer(id)) return;
        const k = reduce ? 1 : Math.min(1, Math.max(0, (t - i * THREAD_STAGGER) / THREAD_MS));
        // ease-out, so each thread lands softly at its source
        map.setPaintProperty(id, "line-gradient", threadGradient(1 - (1 - k) ** 3, color));
      });
    };
    if (reduce) {
      set(Infinity);
      return;
    }
    const t0 = performance.now();
    const step = (now: number) => {
      const t = now - t0;
      set(t);
      if (t < THREAD_MS + THREAD_STAGGER * (THREADS.length - 1)) raf = requestAnimationFrame(step);
    };
    // the layers exist after this render; start on the next frame
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [map, overlay]);

  const onClick = useCallback(
    (e: MapLayerMouseEvent) => {
      const f = e.features?.find((x) => x.layer.id === "ports-circle" || x.layer.id === "ports-hit");
      if (f) {
        onPort(Number((f.properties as Record<string, unknown>).port_code));
        return;
      }
      onPoint({ lat: e.lngLat.lat, lon: e.lngLat.lng });
    },
    [onPort, onPoint],
  );

  const portsData = useMemo(
    () =>
      p.ports
        ? ({
            type: "FeatureCollection",
            features: p.ports.features.map((f) => ({ type: "Feature", geometry: f.geometry as unknown as GeoJSON.Point, properties: f.properties })),
          } as GeoJSON.FeatureCollection)
        : null,
    [p.ports],
  );
  const interactive = p.showPorts && portsData ? ["ports-hit", "ports-circle"] : [];
  const inited = useRef<MlMap | null>(null);
  const init = (m: MlMap) => {
    // the style's layers exist (isStyleLoaded would also wait for the tiles)
    if (inited.current === m || !m.getLayer("land")) return;
    inited.current = m;
    if (!m.hasImage("nodata")) m.addImage("nodata", noDataImage(), { pixelRatio: 2 });
    addArrows(m);
    m.on("styleimagemissing", (ev: { id: string }) => {
      if (ev.id === "nodata" && !m.hasImage("nodata")) m.addImage("nodata", noDataImage(), { pixelRatio: 2 });
      if (ev.id.startsWith("cw-arrow-")) addArrows(m);
    });
    setMap(m);
    // exposed for end-to-end tests and debugging
    (window as unknown as { __cwMap?: unknown }).__cwMap = m;
    p.onReady(m);
  };

  return (
    <Map
      mapStyle={BASEMAP_STYLE}
      transformRequest={transformRequest}
      initialViewState={{ bounds: p.initialBounds, fitBoundsOptions: { padding: p.initialPadding } }}
      maxBounds={[CA_BOUNDS[0][0], CA_BOUNDS[0][1], CA_BOUNDS[1][0], CA_BOUNDS[1][1]]}
      minZoom={4.2}
      maxZoom={12.5}
      renderWorldCopies={false}
      attributionControl={{ compact: true }}
      interactiveLayerIds={interactive}
      onClick={onClick}
      onMouseMove={onHover ? (e) => onHover({ lat: e.lngLat.lat, lon: e.lngLat.lng }, { x: e.point.x, y: e.point.y }) : undefined}
      onMouseOut={onHover ? () => onHover(null, null) : undefined}
      // any other failed request (a tile server error) is reported without a repaint; ask for
      // one so the map still reaches its load event and renders what did arrive
      onError={(e) => {
        console.error(e.error);
        if ((e as { sourceId?: string }).sourceId === "omt") setBasemap("down");
        e.target.triggerRepaint();
      }}
      // set up as soon as the style is ready, not at "load": that waits for every basemap tile,
      // and the data layers, no-data dots, bundled water and navigation must not wait for a slow
      // tile server
      onStyleData={(e) => init(e.target)}
      onLoad={(e) => init(e.target)}
      cursor="crosshair"
      style={{ width: "100%", height: "100%" }}
    >
      <NavigationControl position="bottom-right" showCompass={false} />
      {wide && <ScaleControl position="bottom-right" unit="metric" />}
      <CountyControl prefs={counties} onChange={setCounties} />
      {/* county reference lines and names: under town names, above nothing that holds data on land */}
      <CountyLayers prefs={counties} beforeId="place-city" />

      {p.forecast && forecastTiles && (
        <Source key={`${p.forecast.id}-grid`} id="forecast" type="raster" tiles={[forecastTiles.template]} tileSize={GRID_TILE_SIZE} bounds={forecastTiles.bounds} minzoom={0} maxzoom={13}>
          <Layer id="forecast-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor(p.forecast?.id)} />
        </Source>
      )}
      {p.forecast && !forecastTiles && (
        <Source key={p.forecast.id} id="forecast" type="image" url={p.forecast.url} coordinates={p.forecast.corners as [[number, number], [number, number], [number, number], [number, number]]}>
          <Layer id="forecast-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor(p.forecast?.id)} />
        </Source>
      )}
      {p.forecast && p.cellEdges && (
        <Source key={`${p.forecast.id}-edges`} id="forecast-edges" type="geojson" data={p.cellEdges}>
          <Layer
            id="forecast-cell-edges"
            type="line"
            minzoom={8.5}
            beforeId={BEFORE_OVERLAY_ID}
            paint={{ "line-color": "#06111e", "line-width": 0.5, "line-opacity": ["interpolate", ["linear"], ["zoom"], 8.5, 0, 9.6, 0.1, 11, 0.16] }}
          />
        </Source>
      )}
      {p.satellite && (
        <Source
          key={p.satellite.id}
          id="satellite"
          type="raster"
          tiles={[indexedTiles(p.satellite.template, p.satellite.indexUrl)]}
          tileSize={256}
          minzoom={p.satellite.minzoom}
          maxzoom={p.satellite.maxzoom}
          bounds={p.satellite.bounds as [number, number, number, number] | undefined}
        >
          <Layer id="satellite-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor(p.satellite?.id)} />
        </Source>
      )}
      {p.satelliteAge && (
        <Source key={p.satelliteAge.id} id="satellite-age" type="raster" tiles={[indexedTiles(p.satelliteAge.template, p.satelliteAge.indexUrl)]} tileSize={256} minzoom={p.satelliteAge.minzoom} maxzoom={p.satelliteAge.maxzoom} bounds={p.satelliteAge.bounds as [number, number, number, number] | undefined}>
          <Layer id="satellite-age-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor(p.satelliteAge?.id)} />
        </Source>
      )}
      {p.imagery && (
        <Source key={p.imagery.id} id="observation" type="raster" tiles={[p.imagery.template]} tileSize={256} maxzoom={p.imagery.maxzoom}>
          <Layer id="observation-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={{ "raster-opacity": 1, "raster-fade-duration": 0 }} />
        </Source>
      )}

      {/* after load, once the arrow glyphs are registered */}
      {map && p.currents && p.currents.mode === "arrows" && (
        <Source key={p.currents.id} id="currents" type="geojson" data={p.currents.features}>
          {ARROW_ZOOMS.map(([level, minzoom, maxzoom]) => (
            <Layer
              key={level}
              id={`currents-arrows-${level}`}
              type="symbol"
              minzoom={minzoom}
              maxzoom={maxzoom}
              // over chlorophyll (combined view) one level sparser, so the imagery stays readable
              filter={[">=", ["get", "level"], p.currents?.coarser ? Math.min(16, level * 2) : level]}
              layout={{
                "icon-image": ARROW_IMAGE,
                "icon-rotate": ["get", "dir"],
                "icon-rotation-alignment": "map",
                "icon-allow-overlap": true,
                "icon-ignore-placement": true,
                // smaller when zoomed out, and at one per cell small enough not to touch
                "icon-size": ["interpolate", ["linear"], ["zoom"], 5, 0.5, 8.5, 0.78, 11, 1],
              }}
              paint={{ "icon-opacity": 1 }}
            />
          ))}
        </Source>
      )}
      {p.currents && p.currents.mode === "particles" && <FlowParticles key={p.currents.id} field={p.currents.field} count={wide ? 1400 : 600} />}

      {p.showOfficial && p.officialGeometry && (
        <Source id="official" type="geojson" data={p.officialGeometry}>
          <Layer id="official-casing" type="line" paint={{ "line-color": "#06111e", "line-width": 3.6, "line-opacity": 0.55 }} />
          <Layer id="official-county" type="line" filter={["!=", ["get", "kind"], "lat_limit"]} paint={{ "line-color": OFFICIAL, "line-width": 1.6, "line-dasharray": [2, 1.5] }} />
          <Layer id="official-limit" type="line" filter={["==", ["get", "kind"], "lat_limit"]} paint={{ "line-color": OFFICIAL, "line-width": 2 }} />
          <Layer
            id="official-limit-label"
            type="symbol"
            minzoom={7.4}
            filter={["==", ["get", "kind"], "lat_limit"]}
            layout={{ "symbol-placement": "line-center", "text-field": ["get", "label"], "text-font": ["Noto Sans Regular"], "text-size": 11, "text-offset": [0, -1.15], "text-allow-overlap": false }}
            paint={{ "text-color": OFFICIAL, "text-halo-color": "#06111e", "text-halo-width": 1.2, "text-halo-blur": 0.4 }}
          />
        </Source>
      )}

      {p.overlay && (
        // provenance (lib/provenance): exact cell edges, threads from the point to each source, a dot
        // in the source's mark colour at each end. Threads are connectors, never data colours.
        <Source id="coast-overlay" type="geojson" data={p.overlay} lineMetrics>
          <Layer id="coast-cell-fill" type="fill" filter={["==", ["get", "kind"], "cell"]} paint={{ "fill-color": MODEL_MARK, "fill-opacity": 0.12 }} />
          <Layer id="coast-cell-line" type="line" filter={["==", ["get", "kind"], "cell"]} paint={{ "line-color": "#d9ccff", "line-width": 1.6 }} />
          {THREADS.map(({ to, color }) => (
            <Layer
              key={to}
              id={`coast-thread-${to}`}
              type="line"
              filter={["all", ["==", ["get", "kind"], "thread"], ["==", ["get", "to"], to]]}
              layout={{ "line-cap": "round" }}
              paint={{ "line-width": 1.4, "line-opacity": ["case", ["boolean", ["get", "far"], false], 0.5, 1], "line-gradient": threadGradient(1, color) }}
            />
          ))}
          <Layer
            id="coast-thread-end"
            type="circle"
            filter={["==", ["get", "kind"], "end"]}
            paint={{
              "circle-radius": 4,
              "circle-color": ["match", ["get", "to"], "cell", MODEL_MARK, "station", MEASURED_MARK, OFFICIAL],
              "circle-stroke-color": "#06111e",
              "circle-stroke-width": 1.5,
              "circle-opacity": ["case", ["boolean", ["get", "far"], false], 0.55, 1],
              "circle-stroke-opacity": ["case", ["boolean", ["get", "far"], false], 0.55, 1],
            }}
          />
          <Layer id="coast-station" type="circle" filter={["==", ["get", "kind"], "station"]} paint={{ "circle-radius": 6, "circle-color": MEASURED_MARK, "circle-stroke-color": "#06111e", "circle-stroke-width": 2 }} />
          <Layer
            id="coast-station-label"
            type="symbol"
            filter={["==", ["get", "kind"], "station"]}
            layout={{ "text-field": ["get", "name"], "text-font": ["Noto Sans Regular"], "text-size": 12, "text-offset": [0.9, 0], "text-anchor": "left", "text-optional": true }}
            paint={{ "text-color": "#cdeee6", "text-halo-color": "rgba(6,17,30,0.7)", "text-halo-width": 1.6 }}
          />
          <Layer id="coast-target-halo" type="circle" filter={["==", ["get", "kind"], "target"]} paint={{ "circle-radius": 11, "circle-color": "rgba(255,255,255,0.08)", "circle-stroke-color": "#ffffff", "circle-stroke-width": 2 }} />
          <Layer id="coast-target" type="circle" filter={["==", ["get", "kind"], "target"]} paint={{ "circle-radius": 3.5, "circle-color": "#ffffff", "circle-stroke-color": "#06111e", "circle-stroke-width": 1.5 }} />
        </Source>
      )}

      {p.showPorts && portsData && (
        <Source id="ports" type="geojson" data={portsData}>
          <Layer id="ports-hit" type="circle" paint={{ "circle-radius": 14, "circle-opacity": 0 }} />
          <Layer
            id="ports-selected"
            type="circle"
            filter={["==", ["get", "port_code"], p.selectedPort ?? -1]}
            paint={{ "circle-radius": 13, "circle-color": "rgba(255,255,255,0.22)", "circle-stroke-color": "#ffffff", "circle-stroke-width": 1.5 }}
          />
          <Layer id="ports-circle" type="circle" paint={{ "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 3, 10, 6], "circle-color": "#ffffff", "circle-stroke-color": "#06111e", "circle-stroke-width": 1.6 }} />
          <Layer
            id="ports-label"
            type="symbol"
            minzoom={7.4}
            layout={{ "text-field": ["get", "display_name"], "text-font": ["Noto Sans Regular"], "text-size": 12.5, "text-offset": [0.9, 0], "text-anchor": "left", "text-optional": true }}
            paint={{ "text-color": "#ffffff", "text-halo-color": "#06111e", "text-halo-width": 1.8 }}
          />
        </Source>
      )}

      {hasRaster && !fieldReady && map && (
        <div className="pointer-events-none absolute inset-x-0 top-[72px] z-[1] flex justify-center" role="status" aria-live="polite">
          {fieldFailed ? (
            <span className="rounded-full bg-[rgba(11,26,43,0.92)] px-3 py-1.5 text-[13px] text-[#e9eff6] shadow-[0_6px_20px_rgba(6,17,30,0.3)]" data-testid="map-field-failed">
              Part of this layer could not be loaded. Nothing is drawn in its place.
            </span>
          ) : (
            <span className="cw-late rounded-full bg-[rgba(11,26,43,0.88)] px-3 py-1.5 text-[13px] text-[#e9eff6] shadow-[0_6px_20px_rgba(6,17,30,0.3)]" data-testid="map-field-loading">
              Drawing the data…
            </span>
          )}
        </div>
      )}

      {p.inspect && !p.overlay && (
        <Marker longitude={p.inspect.lon} latitude={p.inspect.lat} anchor="center">
          <span className="block h-4 w-4 rounded-full border-2 border-white shadow-[0_0_0_2px_rgba(4,11,23,0.85)]" aria-hidden />
        </Marker>
      )}
    </Map>
  );
}
