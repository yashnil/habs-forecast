import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, ImageSource, Map as MLMap } from "maplibre-gl";

maplibregl.setWorkerUrl("/vendor/maplibre/maplibre-gl-worker.mjs");
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { Grid } from "../data/store";
import type { Layer, Observations, Official, Port } from "../data/types";
import type { LngLat } from "../lib/geo";
import { daysBetween, TODAY } from "../lib/time";
import { classEdges, type FieldOptions, footprint, renderField } from "./field";
import { BEFORE_DATA, CA_COAST, hatchImage, signatureStyle } from "./style";
import { addThreadLayers } from "./threads";

/**
 * The map stage. Owns one MapLibre map; React passes *what* to show, the stage decides *how* it
 * changes. Every change of shown data is a hold-and-swap cross-fade (two image sources, A/B);
 * nothing is interpolated between two forecast days or two observations.
 */
export type Show = "base" | "forecast" | "satellite";
export interface StageProps {
  show: Show;
  grid?: Grid;
  layer?: Layer;
  field: FieldOptions;
  satellite?: Layer;
  official?: Official;
  observations?: Observations;
  ports?: Port[];
  highlightNotices?: string[] | null; // record ids to emphasise; others dim. null = all normal
  selection?: LngLat | null;
  cells?: { model?: [number, number, number, number] | null; modelGap?: boolean; pixel?: [number, number, number, number] | null; pixelGap?: boolean };
  researchGrid?: boolean;
  onPick?: (p: LngLat, how: "click" | "key") => void;
  onReady?: (map: MLMap) => void;
  onTiles?: (state: "ok" | "failed") => void;
  padding?: maplibregl.PaddingOptions;
  initial?: { center: LngLat; zoom: number } | null;
  label?: string;
}
export interface StageHandle { map: MLMap | null }

const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };
const prefersReduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

export const MapStage = forwardRef<StageHandle, StageProps>(function MapStage(props, ref) {
  const el = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);
  const canvases = useRef<[HTMLCanvasElement, HTMLCanvasElement]>([document.createElement("canvas"), document.createElement("canvas")]);
  const front = useRef<0 | 1>(0);
  const latest = useRef(props);
  latest.current = props;
  useImperativeHandle(ref, () => ({ get map() { return mapRef.current; } }), []);

  /* ------------------------------------------------------------ create */
  useEffect(() => {
    const map = new maplibregl.Map({
      container: el.current!,
      style: signatureStyle(),
      bounds: props.initial ? undefined : CA_COAST,
      center: props.initial?.center,
      zoom: props.initial?.zoom,
      attributionControl: { compact: true },
      dragRotate: false,
      pitchWithRotate: false,
      maxPitch: 0,
      fadeDuration: prefersReduced() ? 0 : 180,
      cancelPendingTileRequestsWhileZooming: true,
    });
    map.touchZoomRotate.disableRotation();
    map.keyboard.enable();
    mapRef.current = map;
    let tileErrors = 0;
    map.on("error", (e) => {
      const src = (e as unknown as { sourceId?: string }).sourceId;
      if (src === "omt" && ++tileErrors === 1) latest.current.onTiles?.("failed");
    });
    map.on("load", () => {
      const h = hatchImage();
      map.addImage("gap-hatch", h, { pixelRatio: 2 });
      // gap hatch on the water — only visible while a field is shown, under it
      map.addLayer({ id: "gap-hatch", type: "fill", source: "omt", "source-layer": "water", layout: { visibility: "none" }, paint: { "fill-pattern": "gap-hatch", "fill-opacity": 0.9 } }, BEFORE_DATA);
      map.addLayer({ id: "gap-hatch-fallback", type: "fill", source: "water-fallback", filter: ["==", ["geometry-type"], "Polygon"], layout: { visibility: "none" }, paint: { "fill-pattern": "gap-hatch", "fill-antialias": false } }, BEFORE_DATA);
      // two field image sources for hold-and-swap
      const blank = blankCanvasUrl();
      for (const k of ["a", "b"]) {
        map.addSource(`field-${k}`, { type: "image", url: blank, coordinates: [[-127, 43], [-117, 43], [-117, 31], [-127, 31]] });
        map.addLayer({ id: `field-${k}`, type: "raster", source: `field-${k}`, paint: { "raster-opacity": 0, "raster-opacity-transition": { duration: 160, delay: 0 }, "raster-resampling": "nearest", "raster-fade-duration": 0 } }, BEFORE_DATA);
      }
      // satellite tiles (observation, each pixel its own date)
      map.addSource("edges", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "edges", type: "line", source: "edges", layout: { visibility: "none", "line-join": "miter" }, paint: { "line-color": ["interpolate", ["linear"], ["get", "hi"], 1, "#8a73b8", 5, "#c3a6f5", 9, "#f1dcff"], "line-width": ["interpolate", ["linear"], ["zoom"], 6, 0.6, 10, 1.4], "line-opacity": ["interpolate", ["linear"], ["get", "hi"], 1, 0.45, 9, 0.95] } }, BEFORE_DATA);
      map.addSource("footprint", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "footprint", type: "line", source: "footprint", layout: { visibility: "none" }, paint: { "line-color": "#c3a6f5", "line-width": 1.2, "line-opacity": 0.8, "line-dasharray": [2, 2] } }, BEFORE_DATA);
      // research: cell grid lines near the selection
      map.addSource("cellgrid", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "cellgrid", type: "line", source: "cellgrid", minzoom: 8.5, paint: { "line-color": "rgba(195,166,245,0.35)", "line-width": 0.6 } });
      // official geometry: rules (lines) and areas (outlines) — amber, solid
      map.addSource("official", { type: "geojson", data: EMPTY, promoteId: "fid" });
      map.addLayer({ id: "official-area", type: "line", source: "official", filter: ["!=", ["geometry-type"], "LineString"], paint: { "line-color": "#f2b04c", "line-width": 1.2, "line-opacity": ["case", ["boolean", ["feature-state", "dim"], false], 0.18, ["boolean", ["feature-state", "hl"], false], 0.95, 0.5] } });
      map.addLayer({ id: "official-rule", type: "line", source: "official", filter: ["==", ["geometry-type"], "LineString"], layout: { "line-cap": "butt" }, paint: { "line-color": "#f2b04c", "line-width": ["case", ["boolean", ["feature-state", "hl"], false], 2.5, 1.6], "line-opacity": ["case", ["boolean", ["feature-state", "dim"], false], 0.25, 0.95] } });
      map.addLayer({ id: "official-rule-label", type: "symbol", source: "official", filter: ["==", ["geometry-type"], "LineString"], minzoom: 7, layout: { "symbol-placement": "line-center", "text-field": ["get", "label"], "text-font": ["Noto Sans Regular"], "text-size": 11, "text-offset": [0, -0.8], "text-allow-overlap": false }, paint: { "text-color": "#f6c477", "text-halo-color": "rgba(7,19,32,0.85)", "text-halo-width": 1.4, "text-opacity": ["case", ["boolean", ["feature-state", "dim"], false], 0.3, 1] } });
      // selected cells (exact edges)
      map.addSource("cells", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "cell-gap", type: "fill", source: "cells", filter: ["==", ["get", "gap"], true], paint: { "fill-pattern": "gap-hatch" } });
      map.addLayer({ id: "cells", type: "line", source: "cells", paint: { "line-color": ["get", "colour"], "line-width": 1.6, "line-dasharray": ["case", ["==", ["get", "gap"], true], ["literal", [2, 1.5]], ["literal", [1, 0]]] } });
      // stations: a dot that carries its date — filled when sampled in the last 14 days, a ring when older
      map.addSource("stations", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "stations", type: "circle", source: "stations", paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 3, 10, 5.5], "circle-color": ["case", ["get", "fresh"], "#4fd1ae", "rgba(79,209,174,0)"], "circle-stroke-color": "#4fd1ae", "circle-stroke-width": 1.6 } });
      map.addLayer({ id: "station-labels", type: "symbol", source: "stations", minzoom: 8.5, layout: { "text-field": ["get", "name"], "text-font": ["Noto Sans Regular"], "text-size": 11, "text-anchor": "left", "text-offset": [0.8, 0] }, paint: { "text-color": "#8fe3c9", "text-halo-color": "rgba(7,19,32,0.85)", "text-halo-width": 1.3 } });
      // ports: the only white labels
      map.addSource("ports", { type: "geojson", data: EMPTY });
      map.addLayer({ id: "ports", type: "circle", source: "ports", minzoom: 5.5, paint: { "circle-radius": 3, "circle-color": "#f4f7fb", "circle-stroke-color": "#071320", "circle-stroke-width": 1.2 } });
      map.addLayer({ id: "port-labels", type: "symbol", source: "ports", minzoom: 6.5, layout: { "text-field": ["get", "name"], "text-font": ["Noto Sans Regular"], "text-size": ["interpolate", ["linear"], ["zoom"], 6.5, 11, 10, 13.5], "text-anchor": "left", "text-offset": [0.7, 0], "text-optional": true }, paint: { "text-color": "#f4f7fb", "text-halo-color": "rgba(7,19,32,0.9)", "text-halo-width": 1.4 } });
      addThreadLayers(map);
      setReady(true);
      latest.current.onReady?.(map);
    });
    // picking: click on the water (keyboard picking is offered by the page: "Choose the map centre")
    map.on("click", (e) => latest.current.onPick?.([e.lngLat.lng, e.lngLat.lat], "click"));
    map.on("mouseenter", "stations", () => (map.getCanvas().style.cursor = "pointer"));
    map.on("mouseleave", "stations", () => (map.getCanvas().style.cursor = ""));
    return () => map.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ------------------------------------------------------------ sea: tint ⇄ neutral (cross-fade) */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const tint = props.show === "base" ? 1 : 0;
    map.setPaintProperty("relief-tint", "raster-opacity", tint);
    map.setPaintProperty("relief-tint-m", "raster-opacity", tint);
    const hatch = props.show === "forecast" ? "visible" : "none";
    map.setLayoutProperty("gap-hatch", "visibility", hatch);
    map.setLayoutProperty("gap-hatch-fallback", "visibility", hatch);
    const satOn = props.show === "satellite";
    if (map.getLayer("satellite")) map.setPaintProperty("satellite", "raster-opacity", satOn ? 0.92 : 0);
  }, [props.show, ready]);

  /* ------------------------------------------------------------ forecast field: hold-and-swap */
  const fieldKey = props.grid && props.layer ? `${props.layer.layer_id}|${props.field.mode}|${props.field.veil}|${props.field.lens ? props.field.lens.lon.toFixed(3) + props.field.lens.lat.toFixed(3) + props.field.lens.km : ""}` : "";
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const on = props.show === "forecast" && !!props.grid && !!props.layer?.palette;
    const ids = ["field-a", "field-b"];
    if (!on) { ids.forEach((id) => map.setPaintProperty(id, "raster-opacity", 0)); map.setLayoutProperty("edges", "visibility", "none"); return; }
    const next = (front.current === 0 ? 1 : 0) as 0 | 1;
    const cv = canvases.current[next];
    const corners = renderField(cv, props.grid!, props.layer!.palette!.stops, props.field);
    const src = map.getSource(`field-${next === 0 ? "a" : "b"}`) as ImageSource;
    src.updateImage({ url: cv.toDataURL(), coordinates: corners });
    // swap through the basemap ("dip"): the old field fades out, then the new one fades in, so no
    // pixel ever shows a blend of two forecast days (a cross-fade would briefly invent colours)
    const nextId = ids[next], prevId = ids[front.current];
    const rm = prefersReduced();
    map.setPaintProperty(prevId, "raster-opacity-transition", { duration: rm ? 0 : 100, delay: 0 });
    map.setPaintProperty(prevId, "raster-opacity", 0);
    map.setPaintProperty(nextId, "raster-opacity-transition", { duration: rm ? 0 : 140, delay: rm ? 0 : 100 });
    map.setPaintProperty(nextId, "raster-opacity", 1);
    front.current = next;
    const edgesOn = props.field.mode === "edges";
    map.setLayoutProperty("edges", "visibility", edgesOn ? "visible" : "none");
    if (edgesOn) (map.getSource("edges") as GeoJSONSource).setData(classEdges(props.grid!));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fieldKey, props.show, ready]);

  /* ------------------------------------------------------------ satellite tiles */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !props.satellite?.tiles) return;
    if (!map.getSource("satellite")) {
      const t = props.satellite.tiles;
      map.addSource("satellite", { type: "raster", tiles: [`${location.origin}/data/v1/${t.url_template}`], tileSize: t.tile_size, minzoom: t.min_zoom ?? 5, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat as [number, number, number, number] });
      map.addLayer({ id: "satellite", type: "raster", source: "satellite", paint: { "raster-opacity": props.show === "satellite" ? 0.92 : 0, "raster-opacity-transition": { duration: 160, delay: 0 }, "raster-resampling": "nearest", "raster-fade-duration": 120 } }, "field-a");
    }
  }, [props.satellite, ready, props.show]);

  /* ------------------------------------------------------------ official geometry */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !props.official) return;
    const recs = new Map(props.official.registry.records.map((r) => [r.id, r]));
    const feats = props.official.geometry.features.map((f, i) => {
      const ids = ((f.properties as { record_ids?: string[] })?.record_ids ?? []) as string[];
      const r = ids.map((id) => recs.get(id)).find(Boolean);
      const lat = f.geometry.type === "LineString" ? (f.geometry.coordinates[0] as number[])[1] : null;
      const label = lat != null ? `${Math.floor(lat)}°${((lat - Math.floor(lat)) * 60).toFixed(2)}′ N · ${r?.agency ?? ""}` : "";
      return { ...f, properties: { ...(f.properties ?? {}), fid: i, label, ids: ids.join(" ") } };
    });
    (map.getSource("official") as GeoJSONSource).setData({ type: "FeatureCollection", features: feats });
  }, [props.official, ready]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !props.official) return;
    const hl = props.highlightNotices;
    props.official.geometry.features.forEach((f, i) => {
      const ids = ((f.properties as { record_ids?: string[] })?.record_ids ?? []) as string[];
      const on = hl ? ids.some((id) => hl.includes(id)) : false;
      map.setFeatureState({ source: "official", id: i }, { hl: on, dim: hl != null && !on });
    });
  }, [props.highlightNotices, props.official, ready]);

  /* ------------------------------------------------------------ stations & ports */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !props.observations) return;
    (map.getSource("stations") as GeoJSONSource).setData({
      type: "FeatureCollection",
      features: props.observations.stations.map((s) => {
        const last = s.sample_times.at(-1)?.slice(0, 10);
        return { type: "Feature", properties: { id: s.station_id, name: s.name, fresh: last ? daysBetween(last, TODAY) <= 14 : false }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } };
      }),
    });
  }, [props.observations, ready]);
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready || !props.ports) return;
    (map.getSource("ports") as GeoJSONSource).setData({ type: "FeatureCollection", features: props.ports.map((p) => ({ type: "Feature", properties: { name: p.display_name }, geometry: { type: "Point", coordinates: [p.lon, p.lat] } })) });
  }, [props.ports, ready]);

  /* ------------------------------------------------------------ selected cells */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const f: GeoJSON.Feature[] = [];
    const box = (b: [number, number, number, number], colour: string, gap: boolean) =>
      f.push({ type: "Feature", properties: { colour, gap }, geometry: { type: "Polygon", coordinates: [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]] } });
    if (props.cells?.model) box(props.cells.model, props.cells.modelGap ? "rgba(176,196,220,0.9)" : "#d9c4ff", !!props.cells.modelGap);
    if (props.cells?.pixel) box(props.cells.pixel, props.cells.pixelGap ? "rgba(176,196,220,0.9)" : "#7ae3c4", !!props.cells.pixelGap);
    (map.getSource("cells") as GeoJSONSource).setData({ type: "FeatureCollection", features: f });
  }, [props.cells?.model?.join(), props.cells?.pixel?.join(), props.cells?.modelGap, props.cells?.pixelGap, ready]);

  /* ------------------------------------------------------------ research: cell grid around the selection */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const src = map.getSource("cellgrid") as GeoJSONSource;
    if (!props.researchGrid || !props.grid || !props.selection) { src.setData(EMPTY); return; }
    const s = props.grid.spec;
    const c = Math.round((props.selection[0] - s.lon_first) / s.lon_step), r = Math.round((props.selection[1] - s.lat_first) / s.lat_step);
    const lines: GeoJSON.Position[][] = [];
    const R = 6;
    for (let k = -R; k <= R + 1; k++) {
      const lon = s.lon_first + (c + k - 0.5) * s.lon_step, lat = s.lat_first + (r + k - 0.5) * s.lat_step;
      lines.push([[lon, s.lat_first + (r - R - 0.5) * s.lat_step], [lon, s.lat_first + (r + R + 0.5) * s.lat_step]]);
      lines.push([[s.lon_first + (c - R - 0.5) * s.lon_step, lat], [s.lon_first + (c + R + 0.5) * s.lon_step, lat]]);
    }
    src.setData({ type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: { type: "MultiLineString", coordinates: lines } }] });
  }, [props.researchGrid, props.grid, props.selection?.join(), ready]);

  /* ------------------------------------------------------------ model footprint (on demand) */
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !ready) return;
    const on = props.show === "forecast" && props.field.mode !== "veil" && !!props.grid;
    map.setLayoutProperty("footprint", "visibility", on ? "visible" : "none");
    if (on) (map.getSource("footprint") as GeoJSONSource).setData(footprint(props.grid!) as GeoJSON.FeatureCollection);
  }, [props.show, props.field.mode, props.grid, ready]);

  return <div ref={el} className="stage" role="region" aria-label={props.label ?? "Map of the California coast"} />;
});

let blank: string | null = null;
function blankCanvasUrl() {
  if (blank) return blank;
  const c = document.createElement("canvas");
  c.width = c.height = 1;
  blank = c.toDataURL();
  return blank;
}
