"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Map, { Layer, Marker, NavigationControl, ScaleControl, Source, type MapLayerMouseEvent } from "react-map-gl/maplibre";
import { setWorkerUrl, type ExpressionSpecification, type Map as MlMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { PortsCollection } from "@/generated/schema";
import { ARROW_LENGTHS, arrowImage, BASEMAP_STYLE, BEFORE_OVERLAY_ID, CA_BOUNDS, NODATA_LAYER_ID, noDataImage, SPEED_CLASSES } from "@/lib/basemap";
import type { CurrentField } from "@/lib/currents";
import { indexedTiles } from "@/lib/tileIndex";
import FlowParticles from "./FlowParticles";

// MapLibre's module worker is copied to public/ by scripts/copy-maplibre-worker.mjs
setWorkerUrl("/vendor/maplibre/maplibre-gl-worker.mjs");

/** C-HARM: one Mercator image placed by its corners. */
export type ImageRaster = { id: string; url: string; corners: number[][] };
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
};

const OFFICIAL = "#f6bb5c";
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
  const paint = p.fade ? RASTER_FADE : RASTER_PAINT;

  // The no-data dots show only under a raster data layer: with currents alone (or nothing)
  // there is no "value" for a gap to be confused with.
  useEffect(() => {
    if (!map) return;
    // idempotent: every style change fires "styledata" again
    const apply = () => {
      if (!map.getLayer(NODATA_LAYER_ID)) return;
      const v = hasRaster ? "visible" : "none";
      if (map.getLayoutProperty(NODATA_LAYER_ID, "visibility") !== v) map.setLayoutProperty(NODATA_LAYER_ID, "visibility", v);
    };
    apply();
    map.on("styledata", apply);
    return () => {
      map.off("styledata", apply);
    };
  }, [map, hasRaster]);
  const { onHover } = p;

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

  return (
    <Map
      mapStyle={BASEMAP_STYLE}
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
      onLoad={(e) => {
        const m = e.target;
        if (!m.hasImage("nodata")) m.addImage("nodata", noDataImage(), { pixelRatio: 2 });
        addArrows(m);
        m.on("styleimagemissing", (ev: { id: string }) => {
          if (ev.id === "nodata" && !m.hasImage("nodata")) m.addImage("nodata", noDataImage(), { pixelRatio: 2 });
          if (ev.id.startsWith("cw-arrow-")) addArrows(m);
        });
        setMap(m);
        // exposed for end-to-end tests and debugging
        (window as unknown as { __cwMap?: unknown }).__cwMap = e.target;
        p.onReady(e.target);
      }}
      cursor="crosshair"
      style={{ width: "100%", height: "100%" }}
    >
      <NavigationControl position="bottom-right" showCompass={false} />
      {wide && <ScaleControl position="bottom-right" unit="metric" />}

      {p.forecast && (
        <Source key={p.forecast.id} id="forecast" type="image" url={p.forecast.url} coordinates={p.forecast.corners as [[number, number], [number, number], [number, number], [number, number]]}>
          <Layer id="forecast-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paint} />
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
          <Layer id="satellite-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paint} />
        </Source>
      )}
      {p.satelliteAge && (
        <Source key={p.satelliteAge.id} id="satellite-age" type="raster" tiles={[indexedTiles(p.satelliteAge.template, p.satelliteAge.indexUrl)]} tileSize={256} minzoom={p.satelliteAge.minzoom} maxzoom={p.satelliteAge.maxzoom} bounds={p.satelliteAge.bounds as [number, number, number, number] | undefined}>
          <Layer id="satellite-age-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={paint} />
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

      {p.inspect && (
        <Marker longitude={p.inspect.lon} latitude={p.inspect.lat} anchor="center">
          <span className="block h-4 w-4 rounded-full border-2 border-white shadow-[0_0_0_2px_rgba(4,11,23,0.85)]" aria-hidden />
        </Marker>
      )}
    </Map>
  );
}
