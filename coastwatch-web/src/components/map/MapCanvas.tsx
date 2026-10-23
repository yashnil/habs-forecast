"use client";

import { FORECAST_RASTER_OPACITY } from "@/lib/palette";
import { NEAR_MS, SHORT_MS, cameraTier } from "@/lib/camera";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Map, { Layer, Marker, NavigationControl, ScaleControl, Source, type MapLayerMouseEvent } from "react-map-gl/maplibre";
import { PMTiles, Protocol } from "pmtiles";
import { addProtocol, setWorkerUrl, type ExpressionSpecification, type GeoJSONSource, type Map as MlMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { PortsCollection } from "@/generated/schema";
import type { LayerArtifact, Palette } from "@/generated/schema";
import type { DerivedGrid } from "@/lib/derived";
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
import { derivedTiles, gridTiles } from "@/lib/gridTileProtocol";
import { indexedTiles } from "@/lib/tileIndex";
import { loadWaterFallback, waterGeoJSON } from "@/lib/waterFallback";
import { basemapArchive, onBasemapSlow, setBasemapArchive } from "@/lib/waterMask";
import FlowLayer, { type FlowOptions } from "./FlowLayer";
import { CountyLayers } from "./Counties";

/** Marks with a name: a pointer over one shows the name (and a pointer cursor). */
const MARK_LAYERS = ["ports-hit", "ports-circle", "map-station", "map-station-point", "map-station-cluster", "coast-station"];

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
/** A derived chlorophyll product (lib/derived), drawn from its chunked grid on the satellite ramp. */
export type DerivedRaster = { id: string; grid: DerivedGrid; baseUrl: string; palette: Palette; style?: "estimate" | "observed"; /** v6: the estimate's class grid (its shoreline strip is marked apart) */ classGrid?: DerivedGrid | null };
/** XYZ tiles: Neritic-rendered satellite tiles or third-party imagery. */
export type TileRaster = { id: string; template: string; minzoom: number; maxzoom: number; bounds?: number[] | null; indexUrl?: string | null };

export type MapPoint = { lat: number; lon: number };
/**
 * Observed currents for one hour (or the 24 h mean): moving particles in the map's own WebGL context
 * (components/map/FlowLayer, the default), or static arrows (one per observed cell; `features`).
 */
export type CurrentsOverlay = { id: string; field: CurrentField; coarser?: boolean } & ({ mode: "arrows"; features: GeoJSON.FeatureCollection } | { mode: "flow"; flow: FlowOptions });
/** The selection's true footprint (lib/footprint): an outline that draws in once after the camera settles. */
export type FootprintOverlay = { key: string; polygon: GeoJSON.Polygon; tone: "model" | "observation" };

type Props = {
  forecast: ImageRaster | null;
  satellite: TileRaster | null;
  satelliteAge: TileRaster | null;
  imagery: TileRaster | null;
  /** v5: the chlorophyll outlook as the map's field (hatched: an outlook, not an observation) */
  outlook?: DerivedRaster | null;
  /** v5: observed (or analysed) sea-surface temperature as the map's field */
  sst?: DerivedRaster | null;
  /** v5: the chlorophyll estimate, under the observed satellite pixels (stippled: estimated, not observed) */
  estimate?: DerivedRaster | null;
  /**
   * v5: water in the C-HARM grid without a value is drawn in one calm neutral tint, "NOAA publishes
   * no toxin probability here" (never with a probability); the workspace says so in its legend and readouts
   */
  contextFill?: boolean;
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
  /** a station of the overlay was chosen (My Coast) */
  onStation?: (stationId: string) => void;
  onPoint: (p: MapPoint) => void;
  onReady: (map: MlMap) => void;
  /** pointer over the map (desktop hover readout); null when it leaves */
  onHover?: (p: MapPoint | null, screen: { x: number; y: number } | null) => void;
  /** C-HARM model cell edges, drawn faintly from z8.5 so its 3 km cells read as cells */
  cellEdges?: GeoJSON.FeatureCollection | null;
  /** valued cell centres with their value, for labels from z 11 (lib/grid cellLabels) */
  cellLabels?: GeoJSON.FeatureCollection | null;
  /**
   * the nearshore gap at harbour zoom (lib/provenance gapFeatures): a note at the map's centre
   * when its cell is in the C-HARM domain without a value, and a dashed thread to the nearest cell
   * with one, carrying the distance
   */
  gap?: GeoJSON.FeatureCollection | null;
  /** the HAB shore stations (lib/provenance stationFeatures), drawn from z 9.5 with their last sample date */
  stations?: GeoJSON.FeatureCollection | null;
  /**
   * My Coast: where each fact was obtained. Features carry properties.kind: "target" (the chosen
   * waters), "station" (a monitoring station, with name), "cell" (a C-HARM cell with a value) and
   * "link" (from the waters to that cell). Drawn above the data, below the ports.
   */
  overlay?: GeoJSON.FeatureCollection | null;
  /** v4: "none" leaves the map's own controls to the workspace (zoom, scale) */
  chrome?: "v2" | "none";
  /** v4: a shore station was chosen on the map (the station itself is selected, not the water beside it) */
  onStationPick?: (stationId: string) => void;
  /** v4: stations as counted clusters below harbour zoom, so the regional view stays calm */
  clusterStations?: boolean;
  /**
   * v4: the selection, drawn above every data layer and label (it is never covered): features with
   * properties.kind "cell" or "area" (polygons, outlined), "station" or "point" (a ring), "pixel" (an
   * observation pixel, outlined) or "footprint" (the outline of observed pixels), optional label
   */
  highlight?: GeoJSON.FeatureCollection | null;
  /** v10: the selected cell or pixel at its native size, outlined (drawn in over 300 ms once the camera settles) */
  footprint?: FootprintOverlay | null;
  /** v4: readiness of the basemap and of the data field, for the status centre */
  onStatus?: (s: { basemap: "ok" | "slow" | "down"; field: "none" | "loading" | "ready" | "failed" }) => void;
  maxZoom?: number;
  /** v4 offline: the basemap from a saved regional PMTiles archive instead of OpenFreeMap */
  basemapPmtiles?: string | null;
  /** v4 "Try again": a new value asks for the data rasters afresh (a failed field is forgotten) */
  retry?: number;
  /** v10: the data field's opacity, 0.3–1 (the layer index's display control); 1 when absent */
  fieldOpacity?: number;
};

const OFFICIAL = "#f6bb5c";
const MODEL_MARK = "#b58cf0";
const MEASURED_MARK = "#4fd1ae";
const CLUSTER_SHIFT = ["interpolate", ["linear"], ["zoom"], 5, ["literal", [-17, 0]], 8, ["literal", [-13, 0]], 9.5, ["literal", [0, 0]]] as unknown as [number, number];
const CONDITIONS_MARK = "#74cfe6";
// provenance threads: one layer per source so they draw out one after another (model, measured,
// official), 520 ms each with a 90 ms stagger; white, never a data colour
const THREADS = [
  { to: "cell", color: "rgba(233,239,246,0.9)" },
  { to: "station", color: "rgba(233,239,246,0.9)" },
  { to: "pixel", color: "rgba(233,239,246,0.9)" },
  { to: "obs", color: "rgba(233,239,246,0.9)" },
  { to: "limit", color: "rgba(246,187,92,0.95)" },
] as const;
const THREAD_MS = 520;
/** v10: the selection's footprint outline draws in once, after the camera settles. */
export const FOOTPRINT_DRAW_MS = 300;
const FOOTPRINT_LINE = { model: "#e6dbff", observation: "#ffffff" } as const;
const FOOTPRINT_HALO = "rgba(4,18,26,0.7)";
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
// Data rasters start nearest-sampled (a pixel is a real source cell, and the legend colours are the
// colours on the map: design reset rev. 2, §5.1). v8 relaxes that below SHARP_ZOOM only, where the GPU
// blends neighbouring texels of each tile when it magnifies them (the effect further down): display
// only, about one texel wide; the tiles themselves, every readout and every legend value are untouched
// (verified on PR #79: readouts identical, gaps kept, satellite tiles unchanged). The forecast is drawn translucent over the
// relief (docs/coastwatch/v2/carto, T1): the seafloor stays legible under a field of 3 km cells,
// and the display classes were built for that opacity on the dark sea.
// v9: every data raster, observations included, fades its tiles in over 220 ms (--nx-t-layer) when the
// reader switches layer: the new layer's tiles come up over the basemap (the old source is removed, so two
// families never blend and no value is interpolated). A new day of the same layer cuts: its old day is
// removed at once, so a fade would dip to the bare sea on every step (and on every frame of playback).
// Reduced motion: a cut.
export const RASTER_FADE_MS = 220;
const RASTER_PAINT = { "raster-opacity": 1, "raster-resampling": "nearest", "raster-fade-duration": RASTER_FADE_MS } as const;
/** v8: data rasters are smoothed below this zoom and nearest-sampled (true cells) from it (see the effect in MapCanvas). */
const SHARP_ZOOM = 8;
const DATA_RASTERS = ["forecast-raster", "outlook-raster", "sst-raster", "satellite-raster", "estimate-raster", "satellite-age-raster"];
const FORECAST_RASTER = { ...RASTER_PAINT, "raster-opacity": FORECAST_RASTER_OPACITY } as const;
// a mild fade only: any lower and the no-value hatch under the field shows through valued cells,
// which would read as "no value" (the cells' outlines and printed values carry harbour zoom instead)
const V4_FORECAST_OPACITY = ["interpolate", ["linear"], ["zoom"], 9, FORECAST_RASTER_OPACITY, 12, 0.6] as unknown as number;

/** The selection ring (v4): 30 px, accent stroke, faint fill; drawn as a symbol so labels avoid it. */
function ringImage(): ImageData {
  const s = 60;
  const c = document.createElement("canvas");
  c.width = c.height = s;
  const g = c.getContext("2d")!;
  g.beginPath();
  g.arc(s / 2, s / 2, s / 2 - 4, 0, Math.PI * 2);
  g.fillStyle = "rgba(95,208,230,0.14)";
  g.fill();
  g.lineWidth = 4.4;
  g.strokeStyle = "#5fd0e6";
  g.stroke();
  return g.getImageData(0, 0, s, s);
}

let protocol: Protocol | null = null;
const archives = new globalThis.Map<string, PMTiles>();
/** The pmtiles:// protocol (once) and one shared archive per URL: the map and the water mask read
 *  the same file (lib/waterMask), so data are clipped at the shore the map draws. */
function registerPmtiles(url: string): PMTiles {
  if (!protocol) {
    protocol = new Protocol();
    addProtocol("pmtiles", protocol.tile);
  }
  let a = archives.get(url);
  if (!a) {
    a = new PMTiles(url);
    protocol.add(a);
    archives.set(url, a);
  }
  return a;
}

export default function MapCanvas(p: Props) {
  const [wide] = useState(() => typeof window !== "undefined" && window.innerWidth >= 1024);
  // one label point per latitude limit, at the line's eastern (coast) end
  const limitLabels = useMemo<GeoJSON.FeatureCollection | null>(() => {
    const fc = p.officialGeometry;
    if (!fc) return null;
    const features: GeoJSON.Feature[] = [];
    for (const f of fc.features) {
      const pr = (f.properties ?? {}) as { kind?: string; label?: string };
      if (pr.kind !== "lat_limit" || !pr.label) continue;
      const g = f.geometry;
      const lines = g.type === "LineString" ? [g.coordinates] : g.type === "MultiLineString" ? g.coordinates : [];
      let east: GeoJSON.Position | null = null;
      for (const l of lines) for (const c of l) if (!east || c[0] > east[0]) east = c;
      if (east) features.push({ type: "Feature", geometry: { type: "Point", coordinates: east }, properties: { label: pr.label } });
    }
    return { type: "FeatureCollection", features };
  }, [p.officialGeometry]);
  const { onPort, onPoint, onStation } = p;
  const [map, setMap] = useState<MlMap | null>(null);
  const hasRaster = !!(p.forecast || p.satellite || p.satelliteAge || p.outlook || p.estimate || p.sst);
  // read once for the map's life (reduced motion: data rasters cut instead of fading)
  const [fadeMs] = useState(() => (window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : RASTER_FADE_MS));
  // decided once per source, when it first appears, and kept for its life: it fades in when its raster layer
  // was not on the map in the last render (a layer switch), and cuts when it replaces the same layer (a new day)
  const shownLayers = useRef(new Set<string>());
  const rendering = new Set<string>();
  const fadeOf = useRef(new globalThis.Map<string, number>());
  useEffect(() => {
    shownLayers.current = new Set(rendering);
  });
  const fadeFor = (layer: string, key: string) => {
    rendering.add(layer);
    if (!fadeOf.current.has(key)) fadeOf.current.set(key, shownLayers.current.has(layer) ? 0 : fadeMs);
    return fadeOf.current.get(key)!;
  };
  const paintFor = (layer: string, key: string, forecast = false) => {
    const base = forecast ? FORECAST_RASTER : RASTER_PAINT;
    const ms = fadeFor(layer, key);
    const paint = ms === RASTER_FADE_MS ? base : { ...base, "raster-fade-duration": ms };
    // v4: the forecast recedes as the reader zooms to a harbour, so the coast and the harbour read
    // through 3 km cells (outlined and labelled with their values from z 10.5); values unchanged
    const out = forecast && p.chrome === "none" ? { ...paint, "raster-opacity": V4_FORECAST_OPACITY } : paint;
    // v10: the reader's field opacity (display only; the legend's colours are the full-opacity ones)
    const k = p.fieldOpacity ?? 1;
    if (k >= 1) return out;
    const o = out["raster-opacity"] as unknown;
    return { ...out, "raster-opacity": (typeof o === "number" ? o * k : ["interpolate", ["linear"], ["zoom"], 9, FORECAST_RASTER_OPACITY * k, 12, 0.6 * k]) as unknown as number };
  };
  // fixed for the map's life: an offline map keeps its saved basemap until the page reloads online
  // (the water that clips the data rasters is read from the same archive, set before the map asks
  // for any data tile, so an offline map never asks the tile server for it)
  const [archive] = useState(() => (p.basemapPmtiles ? registerPmtiles(`${window.location.origin}${p.basemapPmtiles}`) : null));
  const [style] = useState(() => {
    if (!archive) return BASEMAP_STYLE;
    setBasemapArchive(archive);
    const omt = BASEMAP_STYLE.sources.omt as { type: "vector"; attribution?: string };
    // glyphs from this site only (the pack holds every range the labels use): an offline map never asks a font server
    return { ...BASEMAP_STYLE, glyphs: `${window.location.origin}/fonts/{fontstack}/{range}.pbf`, sources: { ...BASEMAP_STYLE.sources, omt: { type: "vector" as const, url: `pmtiles://${archive.source.getKey()}`, attribution: omt.attribution } } };
  });
  useEffect(() => {
    if (!archive) return;
    setBasemapArchive(archive);
    return () => {
      if (basemapArchive() === archive) setBasemapArchive(null);
    };
  }, [archive]);
  const fg = p.forecast?.grid;
  const contextFill = !!p.contextFill;
  const forecastTiles = useMemo(() => (fg ? gridTiles(fg.layer, fg.url, { context: contextFill }) : null), [fg, contextFill]);
  const { outlook, estimate, sst } = p;
  const sstTiles = useMemo(() => (sst ? derivedTiles(sst.id, sst.grid, sst.baseUrl, sst.palette, "observed") : null), [sst]);

  const outlookTiles = useMemo(() => (outlook ? derivedTiles(outlook.id, outlook.grid, outlook.baseUrl, outlook.palette, "outlook") : null), [outlook]);
  const estimateTiles = useMemo(() => (estimate ? derivedTiles(estimate.id, estimate.grid, estimate.baseUrl, estimate.palette, "estimate", estimate.classGrid ?? null) : null), [estimate]);

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
  // "Try again" is part of the key: the sources are asked for again and start unloaded
  const retry = p.retry ?? 0;
  const rasterKey = `${[p.forecast?.id, p.satellite?.id, p.satelliteAge?.id, p.outlook?.id, p.estimate?.id, p.sst?.id].filter(Boolean).join("|")}#${retry}`;
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const [failedKey, setFailedKey] = useState<string | null>(null);
  // a source error beats "loaded": MapLibre counts an errored tile as loaded, so a field whose
  // tiles failed would otherwise read as drawn (and its gaps as the no-data hatch)
  const fieldFailed = hasRaster && failedKey === rasterKey;
  const fieldReady = hasRaster && loadedKey === rasterKey && !fieldFailed;
  const { onStatus } = p;
  useEffect(() => {
    onStatus?.({ basemap, field: !hasRaster ? "none" : fieldReady ? "ready" : fieldFailed ? "failed" : "loading" });
  }, [onStatus, basemap, hasRaster, fieldReady, fieldFailed]);
  useEffect(() => {
    if (!map || !hasRaster) return;
    const on: Record<string, unknown> = { forecast: p.forecast, satellite: p.satellite, "satellite-age": p.satelliteAge, outlook: p.outlook, estimate: p.estimate, sst: p.sst };
    const ids = Object.keys(on).filter((id) => on[id]);
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
  }, [map, hasRaster, rasterKey, p.forecast, p.satellite, p.satelliteAge, p.outlook, p.estimate, p.sst]);

  const tintOnly = !!(p.contextFill && forecastTiles && !p.satellite && !p.satelliteAge && !p.outlook && !p.estimate && !p.sst);
  // The no-data hatch shows only under a data field that has arrived: with currents alone (or
  // nothing) there is no "value" for a gap to be confused with. The bundled-water layers show
  // only while the basemap is slow or down.
  // v9: the depth-tinted seafloor stays until the first data field has drawn, then cross-fades to it (it
  // dropped to the bare relief for the second the new tiles took); once a field has shown, later swaps
  // (a day step, another layer) keep it off, so playback never pulses through it
  const fieldShown = useRef(false);
  if (!hasRaster) fieldShown.current = false;
  else if (fieldReady) fieldShown.current = true;
  const tintOff = hasRaster && fieldShown.current;
  useEffect(() => {
    if (!map) return;
    // idempotent: every style change fires "styledata" again
    const apply = () => {
      // the C-HARM layer with its no-value water tinted (contextFill) leaves nothing for the hatch to
      // mark inside its grid: the tint is the "NOAA publishes no probability here" mark, never hatched over
      const hatch = fieldReady && !tintOnly;
      const want: [string, boolean][] = [
        [NODATA_LAYER_ID, hatch],
        [NODATA_FALLBACK_LAYER_ID, hatch && showFallback],
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
        const o = tintOff ? 0 : 1;
        if (map.getPaintProperty(id, "raster-opacity") !== o) map.setPaintProperty(id, "raster-opacity", o);
      }
    };
    apply();
    map.on("styledata", apply);
    return () => {
      map.off("styledata", apply);
    };
  }, [map, hasRaster, fieldReady, showFallback, tintOnly, tintOff]);
  const { onHover } = p;
  // a harbour or station under the pointer: its name in a small tip at its mark, a pointer cursor,
  // and no value readout over it (fine pointers only; set only when the mark changes, not per move)
  const [tip, setTip] = useState<{ key: string; text: string; x: number; y: number } | null>(null);
  const tipKey = useRef<string | null>(null);
  const onMove = useCallback(
    (e: MapLayerMouseEvent) => {
      const f = e.features?.find((x) => MARK_LAYERS.includes(x.layer.id));
      const pr = (f?.properties ?? {}) as Record<string, unknown>;
      const text = !f
        ? null
        : f.layer.id === "map-station-cluster"
          ? `${pr.point_count} shore stations · click to zoom in`
          : String(pr.display_name ?? pr.name ?? "") || null;
      const key = f && text ? `${f.layer.id}:${String(pr.port_code ?? pr.id ?? pr.cluster_id ?? text)}` : null;
      if (key !== tipKey.current) {
        tipKey.current = key;
        if (f && text && key) {
          const at = f.geometry.type === "Point" ? e.target.project(f.geometry.coordinates as [number, number]) : e.point;
          setTip({ key, text, x: at.x, y: at.y });
        } else setTip(null);
      }
      if (onHover) {
        if (key) onHover(null, null);
        else onHover({ lat: e.lngLat.lat, lon: e.lngLat.lng }, { x: e.point.x, y: e.point.y });
      }
    },
    [onHover],
  );
  const clearTip = useCallback(() => {
    tipKey.current = null;
    setTip(null);
  }, []);

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

  // v10: the footprint outline draws in once per selection, after the camera has settled (a journey to the
  // selection, or the first load); reduced motion: drawn at once
  const fpKey = p.footprint?.key ?? null;
  const fpTone = p.footprint?.tone ?? "observation";
  useEffect(() => {
    if (!map || !fpKey) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let live = true;
    const set = (k: number) => {
      if (map.getLayer("v4-fp-halo")) map.setPaintProperty("v4-fp-halo", "line-gradient", threadGradient(k, FOOTPRINT_HALO));
      if (map.getLayer("v4-fp-line")) map.setPaintProperty("v4-fp-line", "line-gradient", threadGradient(k, FOOTPRINT_LINE[fpTone]));
      const el = map.getContainer();
      el.dataset.footprint = k >= 1 ? "drawn" : "drawing";
    };
    set(reduce ? 1 : 0);
    if (reduce) return;
    const start = () => {
      if (!live) return;
      if (!map.loaded()) return void map.once("idle", start);
      if (map.isMoving()) return void map.once("moveend", start);
      const t0 = performance.now();
      const step = (now: number) => {
        if (!live) return;
        const k = Math.min(1, (now - t0) / FOOTPRINT_DRAW_MS);
        set(1 - (1 - k) ** 3);
        if (k < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    };
    // a frame later: a camera journey begun in the same step as the selection is moving by then
    raf = requestAnimationFrame(start);
    return () => {
      live = false;
      cancelAnimationFrame(raf);
      map.off("idle", start);
      map.off("moveend", start);
    };
  }, [map, fpKey, fpTone]);
  const fpData = useMemo<GeoJSON.Feature | null>(() => (p.footprint ? { type: "Feature", geometry: { type: "LineString", coordinates: p.footprint.polygon.coordinates[0] }, properties: {} } : null), [p.footprint]);

  // v8: at state and regional scale a 3 km cell is a few screen pixels, and nearest sampling draws a
  // field of stair-steps; there the data rasters are smoothed. From harbour scale (SHARP_ZOOM) every
  // pixel is a real source cell again, outlined and readable, and the legend colours are exact.
  useEffect(() => {
    if (!map) return;
    // only a layer whose mode differs is touched, so this never feeds the style events it listens to
    const apply = () => {
      const want = map.getZoom() >= SHARP_ZOOM ? "nearest" : "linear";
      for (const id of DATA_RASTERS) if (map.getLayer(id) && map.getPaintProperty(id, "raster-resampling") !== want) map.setPaintProperty(id, "raster-resampling", want);
    };
    apply();
    map.on("zoomend", apply);
    // a raster that appears later (a layer switch, a new day) starts in the current mode
    map.on("styledata", apply);
    return () => {
      map.off("zoomend", apply);
      map.off("styledata", apply);
    };
  }, [map]);

  const { onStationPick } = p;
  const onClick = useCallback(
    (e: MapLayerMouseEvent) => {
      if (onStationPick) {
        const cl = e.features?.find((x) => x.layer.id === "map-station-cluster");
        if (cl && cl.geometry.type === "Point") {
          // a cluster opens up: zoom in on it, select nothing (a near journey; reduced motion jumps)
          const [lon, lat] = cl.geometry.coordinates as [number, number];
          const tier = cameraTier();
          e.target.easeTo({ center: [lon, lat], zoom: Math.min(10, e.target.getZoom() + 2), duration: tier === "none" ? 0 : tier === "short" ? SHORT_MS : NEAR_MS });
          return;
        }
        const ms = e.features?.find((x) => x.layer.id === "map-station" || x.layer.id === "map-station-point");
        if (ms) {
          onStationPick(String((ms.properties as Record<string, unknown>).id));
          return;
        }
      }
      const st = e.features?.find((x) => x.layer.id === "coast-station");
      if (st && onStation) {
        onStation(String((st.properties as Record<string, unknown>).id));
        return;
      }
      const f = e.features?.find((x) => x.layer.id === "ports-circle" || x.layer.id === "ports-hit");
      if (f) {
        onPort(Number((f.properties as Record<string, unknown>).port_code));
        return;
      }
      // a shore station opens the point inspector at the station itself (its measurements are
      // in "Measured nearby"), never at the pixel that was clicked beside it
      const ms = e.features?.find((x) => x.layer.id === "map-station");
      if (ms && ms.geometry.type === "Point") {
        const [lon, lat] = ms.geometry.coordinates as [number, number];
        onPoint({ lat, lon });
        return;
      }
      onPoint({ lat: e.lngLat.lat, lon: e.lngLat.lng });
    },
    [onPort, onPoint, onStation, onStationPick],
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
  const interactive = [...(p.showPorts && portsData ? ["ports-hit", "ports-circle"] : []), ...(p.onStation && p.overlay ? ["coast-station"] : []), ...(p.stations ? ["map-station"] : []), ...(p.stations && p.clusterStations ? ["map-station-cluster", "map-station-point"] : [])];
  // a station the provenance overlay already draws (the thread's end) is not drawn twice
  const overlayStationIds = useMemo(
    () => (p.overlay?.features ?? []).filter((f) => (f.properties as { kind?: string } | null)?.kind === "station").map((f) => String((f.properties as { id?: unknown }).id ?? "")),
    [p.overlay],
  );
  const inited = useRef<MlMap | null>(null);
  const init = (m: MlMap) => {
    // the style's layers exist (isStyleLoaded would also wait for the tiles)
    if (inited.current === m || !m.getLayer("land")) return;
    inited.current = m;
    if (!m.hasImage("nodata")) m.addImage("nodata", noDataImage(), { pixelRatio: 2 });
    if (!m.hasImage("v4-ring")) m.addImage("v4-ring", ringImage(), { pixelRatio: 2 });
    addArrows(m);
    m.on("styleimagemissing", (ev: { id: string }) => {
      if (ev.id === "nodata" && !m.hasImage("nodata")) m.addImage("nodata", noDataImage(), { pixelRatio: 2 });
      if (ev.id.startsWith("cw-arrow-")) addArrows(m);
      if (ev.id === "v4-ring" && !m.hasImage("v4-ring")) m.addImage("v4-ring", ringImage(), { pixelRatio: 2 });
    });
    setMap(m);
    // exposed for end-to-end tests and debugging
    (window as unknown as { __cwMap?: unknown }).__cwMap = m;
    p.onReady(m);
  };

  return (
    <Map
      mapStyle={style}
      transformRequest={transformRequest}
      initialViewState={{ bounds: p.initialBounds, fitBoundsOptions: { padding: p.initialPadding } }}
      maxBounds={[CA_BOUNDS[0][0], CA_BOUNDS[0][1], CA_BOUNDS[1][0], CA_BOUNDS[1][1]]}
      minZoom={4.2}
      maxZoom={p.maxZoom ?? 12.5}
      renderWorldCopies={false}
      // v4 states the attribution in its own interface (it stays visible beside the map)
      attributionControl={p.chrome === "none" ? false : { compact: true }}
      interactiveLayerIds={interactive}
      onClick={onClick}
      onMouseMove={onMove}
      onMouseOut={() => {
        clearTip();
        onHover?.(null, null);
      }}
      onMoveStart={clearTip}
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
      cursor={tip ? "pointer" : "crosshair"}
      style={{ width: "100%", height: "100%" }}
    >
      {p.chrome !== "none" && <NavigationControl position="bottom-right" showCompass={false} />}
      {wide && p.chrome !== "none" && <ScaleControl position="bottom-right" unit="metric" />}
      {/* county borders and names, always on: part of the basemap, under town names (which are placed
          first), above nothing that holds data on land */}
      <CountyLayers beforeId="place-city" />

      {p.forecast && forecastTiles && (
        <Source key={`${p.forecast.id}-grid-${retry}`} id="forecast" type="raster" tiles={[forecastTiles.template]} tileSize={GRID_TILE_SIZE} bounds={forecastTiles.bounds} minzoom={0} maxzoom={13}>
          <Layer id="forecast-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor("forecast-raster", `${p.forecast.id}-grid-${retry}`, true)} />
        </Source>
      )}
      {/* v7: the model's western edge, drawn and named, so the ocean beyond NOAA's grid reads as outside
          the model (never as a gap or a low value) */}
      {p.forecast && forecastTiles?.bounds && (
        <Source
          key={`${p.forecast.id}-edge`}
          id="forecast-domain-edge"
          type="geojson"
          data={{ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [[forecastTiles.bounds[0], forecastTiles.bounds[3]], [forecastTiles.bounds[0], forecastTiles.bounds[1]]] } }}
        >
          <Layer id="forecast-domain-edge" type="line" beforeId={BEFORE_OVERLAY_ID} paint={{ "line-color": "#9fc4d6", "line-opacity": 0.3, "line-width": 1, "line-dasharray": [1, 3] }} />
          <Layer
            id="forecast-domain-edge-label"
            type="symbol"
            layout={{ "symbol-placement": "line-center", "text-field": "Edge of NOAA's model", "text-font": ["Noto Sans Regular"], "text-size": 10.5, "text-offset": [0, 0.9], "text-letter-spacing": 0.04 }}
            paint={{ "text-color": "rgba(185,211,224,0.66)", "text-halo-color": "#061016", "text-halo-width": 1.2 }}
          />
        </Source>
      )}
      {p.forecast && !forecastTiles && (
        <Source key={`${p.forecast.id}-${retry}`} id="forecast" type="image" url={p.forecast.url} coordinates={p.forecast.corners as [[number, number], [number, number], [number, number], [number, number]]}>
          <Layer id="forecast-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor("forecast-raster", `${p.forecast.id}-${retry}`, true)} />
        </Source>
      )}
      {p.forecast && p.cellEdges && (
        <Source key={`${p.forecast.id}-edges`} id="forecast-edges" type="geojson" data={p.cellEdges}>
          <Layer
            id="forecast-cell-edges"
            type="line"
            minzoom={8.5}
            beforeId={BEFORE_OVERLAY_ID}
            paint={{ "line-color": "#06111e", "line-width": ["interpolate", ["linear"], ["zoom"], 8.5, 0.5, 11, 0.8], "line-opacity": ["interpolate", ["linear"], ["zoom"], 8.5, 0, 9.6, 0.35, 10.5, 0.6] }}
          />
        </Source>
      )}
      {/* at harbour zoom each cell says its value (carto T2): the field reads as cells of a
          class, never as a smooth surface; labels are white on the low classes, ink on the high */}
      {p.forecast && p.cellLabels && (
        <Source key={`${p.forecast.id}-labels`} id="forecast-labels" type="geojson" data={p.cellLabels}>
          <Layer
            id="forecast-cell-labels"
            type="symbol"
            minzoom={11}
            beforeId={BEFORE_OVERLAY_ID}
            layout={{ "text-field": ["get", "text"], "text-font": ["Noto Sans Regular"], "text-size": 11, "text-allow-overlap": false, "text-padding": 2 }}
            paint={{ "text-color": ["case", ["boolean", ["get", "dark"], false], "#1b1d2b", "#ffffff"], "text-opacity": 0.85 }}
          />
        </Source>
      )}
      {p.outlook && outlookTiles && (
        <Source key={`${p.outlook.id}-${retry}`} id="outlook" type="raster" tiles={[outlookTiles.template]} tileSize={GRID_TILE_SIZE} bounds={outlookTiles.bounds} minzoom={0} maxzoom={13}>
          <Layer id="outlook-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor("outlook-raster", `${p.outlook.id}-${retry}`)} />
        </Source>
      )}
      {p.sst && sstTiles && (
        <Source key={`${p.sst.id}-${retry}`} id="sst" type="raster" tiles={[sstTiles.template]} tileSize={GRID_TILE_SIZE} bounds={sstTiles.bounds} minzoom={0} maxzoom={13}>
          <Layer id="sst-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor("sst-raster", `${p.sst.id}-${retry}`)} />
        </Source>
      )}
      {p.satellite && (
        <Source
          key={`${p.satellite.id}-${retry}`}
          id="satellite"
          type="raster"
          tiles={[indexedTiles(p.satellite.template, p.satellite.indexUrl)]}
          tileSize={256}
          minzoom={p.satellite.minzoom}
          maxzoom={p.satellite.maxzoom}
          bounds={p.satellite.bounds as [number, number, number, number] | undefined}
        >
          <Layer id="satellite-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor("satellite-raster", `${p.satellite.id}-${retry}`)} />
        </Source>
      )}
      {/* the estimate lies under the observed pixels (added after them, placed before them): an observation is never covered by an estimate */}
      {p.estimate && estimateTiles && (
        <Source key={`${p.estimate.id}-${retry}`} id="estimate" type="raster" tiles={[estimateTiles.template]} tileSize={GRID_TILE_SIZE} bounds={estimateTiles.bounds} minzoom={0} maxzoom={13}>
          <Layer id="estimate-raster" type="raster" beforeId={p.satellite ? "satellite-raster" : BEFORE_OVERLAY_ID} paint={paintFor("estimate-raster", `${p.estimate.id}-${retry}`)} />
        </Source>
      )}
      {p.satelliteAge && (
        <Source key={`${p.satelliteAge.id}-${retry}`} id="satellite-age" type="raster" tiles={[indexedTiles(p.satelliteAge.template, p.satelliteAge.indexUrl)]} tileSize={256} minzoom={p.satelliteAge.minzoom} maxzoom={p.satelliteAge.maxzoom} bounds={p.satelliteAge.bounds as [number, number, number, number] | undefined}>
          <Layer id="satellite-age-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paintFor("satellite-age-raster", `${p.satelliteAge.id}-${retry}`)} />
        </Source>
      )}
      {p.imagery && (
        <Source key={p.imagery.id} id="observation" type="raster" tiles={[p.imagery.template]} tileSize={256} maxzoom={p.imagery.maxzoom}>
          <Layer id="observation-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={{ "raster-opacity": 1, "raster-fade-duration": fadeFor("observation-raster", p.imagery.id) }} />
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
      {/* not keyed by the hour: a new hour swaps the layer's buffers, it is never remounted */}
      {map && p.currents && p.currents.mode === "flow" && <FlowLayer map={map} id={p.currents.id} field={p.currents.field} options={p.currents.flow} beforeId={BEFORE_OVERLAY_ID} />}

      {p.showOfficial && p.officialGeometry && (
        <Source id="official" type="geojson" data={p.officialGeometry}>
          {/* v8: fine at state scale (a hairline that still reads as official amber), full weight from harbour zoom */}
          <Layer id="official-casing" type="line" paint={{ "line-color": "#06111e", "line-width": ["interpolate", ["linear"], ["zoom"], 5, 2.2, 9, 3.6], "line-opacity": 0.5 }} />
          <Layer id="official-county" type="line" filter={["!=", ["get", "kind"], "lat_limit"]} layout={{ "line-join": "round" }} paint={{ "line-color": OFFICIAL, "line-width": ["interpolate", ["linear"], ["zoom"], 5, 1, 9, 1.6], "line-opacity": ["interpolate", ["linear"], ["zoom"], 5, 0.7, 8, 1], "line-dasharray": [2, 1.5] }} />
          <Layer id="official-limit" type="line" filter={["==", ["get", "kind"], "lat_limit"]} layout={{ "line-cap": "round" }} paint={{ "line-color": OFFICIAL, "line-width": ["interpolate", ["linear"], ["zoom"], 5, 1.4, 9, 2], "line-opacity": ["interpolate", ["linear"], ["zoom"], 5, 0.75, 8, 1] }} />
        </Source>
      )}
      {p.showOfficial && limitLabels && (
        // each latitude limit is named where it meets the coast (its eastern end), the text running out to sea:
        // the map is centred on the coast, so a side panel never hides the name (it did, placed at the line's centre)
        <Source id="official-limit-labels" type="geojson" data={limitLabels}>
          <Layer
            id="official-limit-label"
            type="symbol"
            minzoom={7.4}
            layout={{ "text-field": ["get", "label"], "text-font": ["Noto Sans Regular"], "text-size": 11, "text-anchor": "bottom-right", "text-justify": "right", "text-offset": [-0.6, -0.35], "text-max-width": 14, "text-allow-overlap": false }}
            paint={{ "text-color": OFFICIAL, "text-halo-color": "#06111e", "text-halo-width": 1.2, "text-halo-blur": 0.4 }}
          />
        </Source>
      )}

      {p.overlay && (
        // provenance (lib/provenance): exact cell edges, threads from the point to each source, a dot
        // in the source's mark colour at each end. Threads are connectors, never data colours.
        <Source id="coast-overlay" type="geojson" data={p.overlay} lineMetrics>
          <Layer id="coast-cell-fill" type="fill" filter={["==", ["get", "kind"], "cell"]} paint={{ "fill-color": MODEL_MARK, "fill-opacity": 0.12, "fill-opacity-transition": { duration: 200, delay: 0 } }} />
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
              "circle-color": ["match", ["get", "to"], "cell", MODEL_MARK, "station", MEASURED_MARK, "pixel", MEASURED_MARK, "obs", CONDITIONS_MARK, OFFICIAL],
              "circle-stroke-color": "#06111e",
              "circle-stroke-width": 1.5,
              "circle-opacity": ["case", ["boolean", ["get", "far"], false], 0.55, 1],
              "circle-stroke-opacity": ["case", ["boolean", ["get", "far"], false], 0.55, 1],
            }}
          />
          {/* the distance to the nearest model cell, at the thread's end (carto T3); a point label, since
              the thread is often shorter than its words */}
          <Layer
            id="coast-thread-label"
            type="symbol"
            minzoom={9}
            filter={["all", ["==", ["get", "kind"], "end"], ["==", ["get", "to"], "cell"], ["has", "label"]]}
            layout={{ "text-field": ["get", "label"], "text-font": ["Noto Sans Regular"], "text-size": 11.5, "text-anchor": ["get", "anchor"], "text-radial-offset": 0.8, "text-justify": "auto", "text-max-width": 12, "text-allow-overlap": true, "text-ignore-placement": true }}
            paint={{ "text-color": "#e8e4ff", "text-halo-color": "#06111e", "text-halo-width": 1.4, "text-halo-blur": 0.4 }}
          />
          {/* the model's neighbourhood: the cells within it are what the briefing summarises */}
          <Layer id="coast-ring" type="line" filter={["==", ["get", "kind"], "ring"]} paint={{ "line-color": MODEL_MARK, "line-width": 1.4, "line-dasharray": [3, 2], "line-opacity": 0.9 }} />
          {/* stations: a point that carries its age (filled when sampled in the last 30 days, a
              ring when older); the station the briefing is describing is larger */}
          <Layer
            id="coast-station"
            type="circle"
            filter={["==", ["get", "kind"], "station"]}
            paint={{
              "circle-radius": ["case", ["boolean", ["get", "focus"], false], 8.5, ["boolean", ["get", "minor"], false], 4.5, 6.5],
              "circle-color": ["case", ["boolean", ["get", "aged"], false], "rgba(6,17,30,0.6)", MEASURED_MARK],
              "circle-stroke-color": ["case", ["boolean", ["get", "aged"], false], MEASURED_MARK, "#06111e"],
              "circle-stroke-width": ["case", ["boolean", ["get", "aged"], false], 2.2, 2],
              "circle-opacity": ["case", ["boolean", ["get", "minor"], false], 0.75, 1],
            }}
          />
          <Layer
            id="coast-station-label"
            type="symbol"
            filter={["all", ["==", ["get", "kind"], "station"], ["!", ["boolean", ["get", "minor"], false]]]}
            layout={{ "text-field": ["get", "name"], "text-font": ["Noto Sans Regular"], "text-size": 12, "text-offset": [0.9, 0], "text-anchor": "left", "text-optional": true }}
            paint={{ "text-color": "#cdeee6", "text-halo-color": "rgba(6,17,30,0.7)", "text-halo-width": 1.6 }}
          />
          <Layer id="coast-target-halo" type="circle" filter={["==", ["get", "kind"], "target"]} paint={{ "circle-radius": 11, "circle-color": "rgba(255,255,255,0.08)", "circle-stroke-color": "#ffffff", "circle-stroke-width": 2 }} />
          {/* NOAA stations behind the conditions card (context, never a field), in the conditions colour */}
          <Layer id="coast-obs" type="circle" filter={["==", ["get", "kind"], "obs"]} paint={{ "circle-radius": 6, "circle-color": CONDITIONS_MARK, "circle-stroke-color": "#06111e", "circle-stroke-width": 2 }} />
          <Layer
            id="coast-obs-label"
            type="symbol"
            filter={["==", ["get", "kind"], "obs"]}
            layout={{ "text-field": ["concat", ["case", ["==", ["get", "what"], "buoy"], "Buoy · ", "Tides · "], ["get", "name"]], "text-font": ["Noto Sans Regular"], "text-size": 12, "text-offset": [0.9, 0], "text-anchor": "left", "text-optional": true }}
            paint={{ "text-color": "#e9eff6", "text-halo-color": "#06111e", "text-halo-width": 1.4 }}
          />
          <Layer id="coast-target" type="circle" filter={["==", ["get", "kind"], "target"]} paint={{ "circle-radius": 3.5, "circle-color": "#ffffff", "circle-stroke-color": "#06111e", "circle-stroke-width": 1.5 }} />
        </Source>
      )}

      {p.stations && (
        // the HAB shore stations at harbour zoom (carto T3): filled when sampled in the last 30 days,
        // a ring when older; the label names the station and its last sample date, and moves
        // around the point rather than collide with a port's name
        <Source id="map-stations" type="geojson" data={p.stations}>
          <Layer
            id="map-station"
            type="circle"
            minzoom={9.5}
            filter={["!", ["in", ["get", "id"], ["literal", overlayStationIds]]]}
            paint={{
              "circle-radius": ["interpolate", ["linear"], ["zoom"], 9.5, 4.5, 11, 6],
              "circle-color": ["case", ["boolean", ["get", "recent"], false], MEASURED_MARK, "rgba(6,17,30,0.6)"],
              "circle-stroke-color": ["case", ["boolean", ["get", "recent"], false], "#06111e", MEASURED_MARK],
              "circle-stroke-width": ["case", ["boolean", ["get", "recent"], false], 1.6, 2],
              "circle-opacity": ["interpolate", ["linear"], ["zoom"], 9.5, 0, 10, 1],
              "circle-stroke-opacity": ["interpolate", ["linear"], ["zoom"], 9.5, 0, 10, 1],
            }}
          />
          <Layer
            id="map-station-label"
            type="symbol"
            minzoom={9.5}
            filter={["!", ["in", ["get", "id"], ["literal", overlayStationIds]]]}
            layout={{
              "text-field": ["get", "label"],
              "text-font": ["Noto Sans Regular"],
              "text-size": 11.5,
              "text-variable-anchor": ["left", "right", "top", "bottom"],
              "text-radial-offset": 0.9,
              "text-justify": "auto",
              "text-max-width": 14,
              "text-optional": true,
            }}
            paint={{ "text-color": "#cdeee6", "text-halo-color": "rgba(6,17,30,0.8)", "text-halo-width": 1.5, "text-opacity": ["interpolate", ["linear"], ["zoom"], 9.5, 0, 10, 1] }}
          />
        </Source>
      )}

      {p.stations && p.clusterStations && (
        // below harbour zoom the stations are counted clusters (a calm regional view); a single
        // station keeps its own mark and opens on tap
        <Source id="map-stations-clustered" type="geojson" data={p.stations} cluster clusterRadius={38} clusterMaxZoom={9}>
          <Layer
            id="map-station-cluster"
            type="circle"
            maxzoom={9.5}
            filter={["has", "point_count"]}
            // v8: a cluster sits just offshore of its stations (the coast runs north-south), so a harbour's
            // dot never lands on its count; it closes in on the stations as the reader zooms
            paint={{ "circle-radius": ["step", ["get", "point_count"], 9.5, 4, 11.5], "circle-color": "rgba(6,17,30,0.88)", "circle-stroke-color": MEASURED_MARK, "circle-stroke-width": 1.4, "circle-translate": CLUSTER_SHIFT }}
          />
          <Layer
            id="map-station-cluster-count"
            type="symbol"
            maxzoom={9.5}
            filter={["has", "point_count"]}
            layout={{ "text-field": ["get", "point_count_abbreviated"], "text-font": ["Noto Sans Regular"], "text-size": 10.5, "text-allow-overlap": true }}
            paint={{ "text-color": "#d6f3ec", "text-translate": CLUSTER_SHIFT }}
          />
          <Layer
            id="map-station-point"
            type="circle"
            maxzoom={9.5}
            filter={["!", ["has", "point_count"]]}
            paint={{
              "circle-radius": 4.5,
              "circle-color": ["case", ["boolean", ["get", "recent"], false], MEASURED_MARK, "rgba(6,17,30,0.7)"],
              "circle-stroke-color": ["case", ["boolean", ["get", "recent"], false], "#06111e", MEASURED_MARK],
              "circle-stroke-width": 1.6,
            }}
          />
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
          {/* circles take no part in label collision: an invisible symbol the size of the dot reserves its place, so a
              city or station label is never drawn into a harbour's marker ("San Dieg●San Diego") */}
          <Layer
            id="ports-collider"
            type="symbol"
            layout={{ "icon-image": "v4-ring", "icon-size": ["interpolate", ["linear"], ["zoom"], 4, 0.25, 10, 0.5], "icon-allow-overlap": true, "icon-padding": 1 }}
            paint={{ "icon-opacity": 0 }}
          />
          <Layer
            id="ports-label"
            type="symbol"
            minzoom={7.4}
            // a selected harbour is named by the selection's own label (v4), never twice
            filter={p.highlight ? ["!=", ["get", "port_code"], p.selectedPort ?? -1] : ["boolean", true]}
            layout={{ "text-field": ["get", "display_name"], "text-font": ["Noto Sans Regular"], "text-size": 12.5, "text-offset": [0.9, 0], "text-anchor": "left", "text-optional": true }}
            paint={{ "text-color": "#ffffff", "text-halo-color": "#06111e", "text-halo-width": 1.8 }}
          />
        </Source>
      )}

      {p.forecast && p.gap && (
        // the nearshore gap at harbour zoom (carto T3): the thread style of the provenance overlay,
        // dashed, from the map's centre to the nearest modelled cell; the note is the point of the
        // view, so it is always placed
        <Source id="forecast-gap" type="geojson" data={p.gap}>
          <Layer id="forecast-gap-thread" type="line" minzoom={10.5} filter={["==", ["get", "kind"], "thread"]} layout={{ "line-cap": "round" }} paint={{ "line-color": "rgba(233,239,246,0.9)", "line-width": 1.4, "line-dasharray": [3, 2.2] }} />
          <Layer id="forecast-gap-end" type="circle" minzoom={10.5} filter={["==", ["get", "kind"], "end"]} paint={{ "circle-radius": 4, "circle-color": MODEL_MARK, "circle-stroke-color": "#06111e", "circle-stroke-width": 1.5 }} />
          <Layer
            id="forecast-gap-distance"
            type="symbol"
            minzoom={10.5}
            filter={["==", ["get", "kind"], "end"]}
            // anchored on the side away from the harbour (lib/provenance gapFeatures), clear of its name
            layout={{ "text-field": ["get", "label"], "text-font": ["Noto Sans Regular"], "text-size": 11.5, "text-anchor": ["get", "anchor"], "text-radial-offset": 0.8, "text-justify": "auto", "text-max-width": 12, "text-allow-overlap": true }}
            paint={{ "text-color": "#e8e4ff", "text-halo-color": "#06111e", "text-halo-width": 1.4, "text-halo-blur": 0.4 }}
          />
          {/* the note is the top symbol layer, so it is placed first and always shown; it does not
              ignore placement, so the station and port labels at the harbour move out of its way */}
          <Layer
            id="forecast-gap-note"
            type="symbol"
            minzoom={10.5}
            filter={["==", ["get", "kind"], "note"]}
            layout={{ "text-field": ["get", "text"], "text-font": ["Noto Sans Regular"], "text-size": 12, "text-variable-anchor": ["top", "bottom", "left", "right"], "text-radial-offset": 0.9, "text-justify": "auto", "text-max-width": 16, "text-allow-overlap": true }}
            paint={{ "text-color": "#e9eff6", "text-halo-color": "#06111e", "text-halo-width": 1.5, "text-halo-blur": 0.4 }}
          />
          <Layer id="forecast-gap-centre" type="circle" minzoom={10.5} filter={["==", ["get", "kind"], "note"]} paint={{ "circle-radius": 3, "circle-color": "#ffffff", "circle-stroke-color": "#06111e", "circle-stroke-width": 1.5 }} />
        </Source>
      )}

      {p.highlight && (
        // v4: the selection, last in the stack so nothing covers it; its label is always placed
        // and the other labels move out of its way
        <Source id="v4-highlight" type="geojson" data={p.highlight}>
          <Layer id="v4-hl-fill" type="fill" filter={["in", ["get", "kind"], ["literal", ["cell", "area"]]]} paint={{ "fill-color": ["match", ["get", "kind"], "cell", MODEL_MARK, "#5fd0e6"], "fill-opacity": ["match", ["get", "kind"], "cell", 0.16, 0.07], "fill-opacity-transition": { duration: 200, delay: 0 } }} />
          {/* a model cell's outline is the footprint's (below), drawn in once; an area keeps its dashed frame */}
          <Layer id="v4-hl-halo" type="line" filter={["==", ["get", "kind"], "area"]} paint={{ "line-color": "#04121a", "line-width": 5, "line-opacity": 0.55, "line-opacity-transition": { duration: 200, delay: 0 } }} />
          <Layer id="v4-hl-line" type="line" filter={["==", ["get", "kind"], "area"]} paint={{ "line-color": ["match", ["get", "kind"], "cell", "#e6dbff", "#5fd0e6"], "line-width": 2, "line-dasharray": ["match", ["get", "kind"], "area", ["literal", [3, 2]], ["literal", [1, 0]]] }} />
          {/* observation pixels and footprints (Research): neutral white, never the model's colour */}
          <Layer id="v4-hl-pixel-halo" type="line" filter={["in", ["get", "kind"], ["literal", ["pixel", "footprint"]]]} paint={{ "line-color": "#04121a", "line-width": ["match", ["get", "kind"], "pixel", 4.5, 2.6], "line-opacity": 0.6 }} />
          <Layer id="v4-hl-pixel" type="line" filter={["in", ["get", "kind"], ["literal", ["pixel", "footprint"]]]} paint={{ "line-color": "#ffffff", "line-width": ["match", ["get", "kind"], "pixel", 2, 1], "line-opacity": 0.95 }} />
          {/* the ring is a symbol, so it takes part in label placement: other labels move off it */}
          <Layer id="v4-hl-transect-halo" type="line" filter={["==", ["get", "kind"], "transect"]} layout={{ "line-cap": "round" }} paint={{ "line-color": "#04121a", "line-width": 5, "line-opacity": 0.6 }} />
          <Layer id="v4-hl-transect" type="line" filter={["==", ["get", "kind"], "transect"]} layout={{ "line-cap": "round" }} paint={{ "line-color": "#5fd0e6", "line-width": 2.4 }} />
          <Layer id="v4-hl-ring" type="symbol" filter={["in", ["get", "kind"], ["literal", ["station", "point", "harbor"]]]} layout={{ "icon-image": "v4-ring", "icon-allow-overlap": true, "icon-padding": 4 }} />
          <Layer id="v4-hl-dot" type="circle" filter={["in", ["get", "kind"], ["literal", ["station", "point", "harbor"]]]} paint={{ "circle-radius": 4, "circle-color": "#ffffff", "circle-stroke-color": "#04121a", "circle-stroke-width": 1.5 }} />
          <Layer
            id="v4-hl-label"
            type="symbol"
            filter={["has", "label"]}
            layout={{ "text-field": ["get", "label"], "text-font": ["Noto Sans Regular"], "text-size": 12.5, "text-variable-anchor": ["top", "bottom", "left", "right"], "text-radial-offset": 1.5, "text-justify": "auto", "text-max-width": 16, "text-allow-overlap": true }}
            paint={{ "text-color": "#ffffff", "text-halo-color": "#04121a", "text-halo-width": 1.8 }}
          />
        </Source>
      )}

      {fpData && (
        // v10: the selected cell or pixel at its native size, a crisp outline over everything (it is the selection)
        <Source id="v4-footprint" type="geojson" data={fpData} lineMetrics>
          <Layer id="v4-fp-halo" type="line" layout={{ "line-join": "miter", "line-cap": "butt" }} paint={{ "line-width": 4.5, "line-gradient": threadGradient(0, FOOTPRINT_HALO) }} />
          <Layer id="v4-fp-line" type="line" layout={{ "line-join": "miter", "line-cap": "butt" }} paint={{ "line-width": 2, "line-gradient": threadGradient(0, FOOTPRINT_LINE[p.footprint?.tone ?? "observation"]) }} />
        </Source>
      )}

      {hasRaster && !fieldReady && map && p.chrome !== "none" && (
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

      {tip && (
        <div className="pointer-events-none absolute z-[2] -translate-x-1/2 whitespace-nowrap rounded-md bg-[rgba(6,17,30,0.9)] px-2 py-1 text-[12.5px] font-semibold text-[#e9eff6] shadow-[0_4px_14px_rgba(6,17,30,0.35)]" style={{ left: tip.x, top: tip.y - 36 }} data-testid="map-mark-tip">
          {tip.text}
        </div>
      )}

      {/* the chosen water: a ring settles onto the point and one pulse goes out from it, once */}
      {p.inspect && (
        <Marker key={`${p.inspect.lat},${p.inspect.lon}`} longitude={p.inspect.lon} latitude={p.inspect.lat} anchor="center">
          <span className="relative block h-5 w-5" aria-hidden data-testid="map-selection">
            <span className="cw-ping absolute inset-0 rounded-full border-2 border-white" />
            {!p.overlay && <span className="cw-settle absolute inset-0.5 rounded-full border-2 border-white shadow-[0_0_0_2px_rgba(4,11,23,0.85)]" />}
          </span>
        </Marker>
      )}
    </Map>
  );
}
