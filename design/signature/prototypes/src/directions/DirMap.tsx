import * as maplibregl from "maplibre-gl";
import type { GeoJSONSource, ImageSource, Map as MLMap } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { LngLat } from "../lib/geo";
import { daysBetween, TODAY } from "../lib/time";
import { classEdges, type FieldOptions } from "../map/field";
import { BEFORE_DATA, hatchImage } from "../map/style";
import { addThreadLayers, setThreads, type Thread } from "../map/threads";
import { renderFieldOn } from "./field";
import type { DirState } from "./shared";
import { FIELD_BASE, type Variant, variantStyle } from "./styles";

maplibregl.setWorkerUrl("/vendor/maplibre/maplibre-gl-worker.mjs");

/** One map per direction: variant basemap, the forecast field, official lines, stations, ports,
 *  selection, provenance threads and optional numbered HTML callouts. */
export interface Callout { id: string; at: LngLat; node: React.ReactNode }
const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

export function DirMap({ v, st, field, center, zoom, padding, threads = true, hatch = true, callouts = [], onMove, interactive = true, colours }: {
  v: Variant; st: DirState; field: FieldOptions; center: LngLat; zoom: number;
  padding?: maplibregl.PaddingOptions; threads?: boolean; hatch?: boolean; callouts?: Callout[];
  onMove?: (m: MLMap) => void; interactive?: boolean;
  colours?: { official: string; station: string; stationFill: string; port: string; portLabel: string; halo: string; edge: string; select: string };
}) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);
  const [hosts, setHosts] = useState<Record<string, HTMLElement>>({});
  const c = colours ?? { official: "#f2b04c", station: "#4fd1ae", stationFill: "#4fd1ae", port: "#f4f7fb", portLabel: "#f4f7fb", halo: "rgba(7,19,32,0.9)", edge: "#c3a6f5", select: "#ffffff" };

  useEffect(() => {
    const m = new maplibregl.Map({ container: el.current!, style: variantStyle(v), center, zoom, attributionControl: { compact: true }, dragRotate: false, interactive, fadeDuration: 0 });
    m.touchZoomRotate.disableRotation();
    map.current = m;
    m.on("load", () => {
      m.addImage("gap-hatch", hatchImage(), { pixelRatio: 2 });
      if (hatch) m.addLayer({ id: "gap-hatch", type: "fill", source: "omt", "source-layer": "water", paint: { "fill-pattern": "gap-hatch", "fill-opacity": v === "a" ? 0.9 : 0.5 } }, BEFORE_DATA);
      m.addSource("field", { type: "image", url: blank(), coordinates: [[-127, 43], [-117, 43], [-117, 31], [-127, 31]] });
      m.addLayer({ id: "field", type: "raster", source: "field", paint: { "raster-resampling": "nearest", "raster-fade-duration": 0 } }, BEFORE_DATA);
      m.addSource("edges", { type: "geojson", data: EMPTY });
      m.addLayer({ id: "edges", type: "line", source: "edges", paint: { "line-color": c.edge, "line-width": ["interpolate", ["linear"], ["zoom"], 6, 0.6, 10, 1.3], "line-opacity": ["interpolate", ["linear"], ["get", "hi"], 1, 0.4, 9, 0.95] } }, BEFORE_DATA);
      m.addSource("official", { type: "geojson", data: EMPTY });
      m.addLayer({ id: "official-area", type: "line", source: "official", filter: ["!=", ["geometry-type"], "LineString"], paint: { "line-color": c.official, "line-width": 1.1, "line-opacity": 0.45 } });
      m.addLayer({ id: "official-rule", type: "line", source: "official", filter: ["==", ["geometry-type"], "LineString"], paint: { "line-color": c.official, "line-width": 1.8 } });
      m.addSource("stations", { type: "geojson", data: EMPTY });
      m.addLayer({ id: "stations", type: "circle", source: "stations", paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 3, 10, 5.5], "circle-color": ["case", ["get", "fresh"], c.stationFill, "rgba(0,0,0,0)"], "circle-stroke-color": c.station, "circle-stroke-width": 1.6 } });
      m.addSource("ports", { type: "geojson", data: EMPTY });
      m.addLayer({ id: "ports", type: "circle", source: "ports", paint: { "circle-radius": 3, "circle-color": c.port, "circle-stroke-color": c.halo, "circle-stroke-width": 1.2 } });
      m.addLayer({ id: "port-labels", type: "symbol", source: "ports", layout: { "text-field": ["get", "name"], "text-font": ["Noto Sans Regular"], "text-size": 12.5, "text-anchor": "left", "text-offset": [0.7, 0] }, paint: { "text-color": c.portLabel, "text-halo-color": c.halo, "text-halo-width": 1.3 } });
      m.addSource("cell", { type: "geojson", data: EMPTY });
      m.addLayer({ id: "cell", type: "line", source: "cell", paint: { "line-color": c.edge, "line-width": 1.8 } });
      if (threads) addThreadLayers(m);
      // a field is always shown in these views: the depth tint is for the basemap alone (settled rule)
      if (v === "a") { m.setPaintProperty("relief-tint", "raster-opacity", 0); m.setPaintProperty("relief-tint-m", "raster-opacity", 0); }
      setReady(true);
    });
    if (onMove) { m.on("move", () => onMove(m)); m.on("load", () => onMove(m)); }
    return () => m.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [v]);

  // data
  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    if (st.official) (m.getSource("official") as GeoJSONSource).setData(st.official.geometry);
    if (st.observations) (m.getSource("stations") as GeoJSONSource).setData({ type: "FeatureCollection", features: st.observations.stations.map((s) => ({ type: "Feature", properties: { fresh: (s.sample_times.at(-1) ? daysBetween(s.sample_times.at(-1)!.slice(0, 10), TODAY) : 99) <= 14 }, geometry: { type: "Point", coordinates: [s.lon, s.lat] } })) });
    if (st.ports) (m.getSource("ports") as GeoJSONSource).setData({ type: "FeatureCollection", features: st.ports.map((p) => ({ type: "Feature", properties: { name: p.display_name }, geometry: { type: "Point", coordinates: [p.lon, p.lat] } })) });
  }, [ready, st.official, st.observations, st.ports]);

  // field
  useEffect(() => {
    const m = map.current;
    if (!m || !ready || !st.grid || !st.layer?.palette) return;
    const cv = document.createElement("canvas");
    const corners = renderFieldOn(cv, st.grid, st.layer.palette.stops, field, FIELD_BASE[v]);
    (m.getSource("field") as ImageSource).updateImage({ url: cv.toDataURL(), coordinates: corners });
    (m.getSource("edges") as GeoJSONSource).setData(field.mode === "edges" ? classEdges(st.grid) : EMPTY);
  }, [ready, st.grid, st.layer, field.mode, field.veil, field.outside, v]);

  // selection: model cell + threads
  useEffect(() => {
    const m = map.current;
    const r = st.reading;
    if (!m || !ready || !r) return;
    const mr = r.model;
    if ((mr.status === "value" || mr.status === "gap") && mr.cell) {
      const b = mr.cell;
      (m.getSource("cell") as GeoJSONSource).setData({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]] } });
    }
    if (!threads) return;
    const t: Thread[] = [];
    if (r.station) t.push({ id: "station", from: r.point, to: [r.station.station.lon, r.station.station.lat], colour: c.station });
    if (st.limit) t.push({ id: "notice", from: r.point, to: [r.point[0], st.limit.lat], colour: c.official });
    setThreads(m, t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, st.reading, st.limit]);

  // HTML callouts (portals into Marker elements)
  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    const markers: maplibregl.Marker[] = [];
    const h: Record<string, HTMLElement> = {};
    for (const co of callouts) {
      const div = document.createElement("div");
      div.className = `dir-callout dir-callout-${v}`;
      h[co.id] = div;
      markers.push(new maplibregl.Marker({ element: div }).setLngLat(co.at).addTo(m));
    }
    setHosts(h);
    return () => markers.forEach((mk) => mk.remove());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, callouts.map((x) => x.id + x.at.join()).join("|")]);

  useEffect(() => {
    if (map.current && ready && padding) map.current.jumpTo({ center, zoom, padding });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, JSON.stringify(padding)]);

  return (
    <div ref={el} className={`dir-map dir-map-${v}`} role="region" aria-label="Map: the waters off Santa Cruz">
      {callouts.map((co) => (hosts[co.id] ? createPortal(co.node, hosts[co.id], co.id) : null))}
    </div>
  );
}

let b: string | null = null;
function blank() { if (!b) { const cv = document.createElement("canvas"); cv.width = cv.height = 1; b = cv.toDataURL(); } return b; }
