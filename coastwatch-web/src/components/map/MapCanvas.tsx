"use client";

import { useCallback, useMemo, useState } from "react";
import Map, { Layer, Marker, NavigationControl, ScaleControl, Source, type MapLayerMouseEvent } from "react-map-gl/maplibre";
import { setWorkerUrl, type Map as MlMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { PortsCollection } from "@/generated/schema";
import { BASEMAP_STYLE, BEFORE_OVERLAY_ID, CA_BOUNDS } from "@/lib/basemap";

// MapLibre's module worker is copied to public/ by scripts/copy-maplibre-worker.mjs
setWorkerUrl("/vendor/maplibre/maplibre-gl-worker.mjs");

export type Raster =
  | { kind: "image"; id: string; url: string; corners: number[][] }
  | { kind: "tiles"; id: string; template: string; maxzoom: number }
  | null;

export type MapPoint = { lat: number; lon: number };

type Props = {
  raster: Raster;
  opacity: number;
  ports: PortsCollection | null;
  showPorts: boolean;
  officialGeometry: GeoJSON.FeatureCollection | null;
  showOfficial: boolean;
  selectedPort: number | null;
  inspect: MapPoint | null;
  initialBounds: [[number, number], [number, number]];
  onPort: (portCode: number) => void;
  onPoint: (p: MapPoint) => void;
  onReady: (map: MlMap) => void;
};

const OFFICIAL = "#ffb547";

export default function MapCanvas({ raster, opacity, ports, showPorts, officialGeometry, showOfficial, selectedPort, inspect, initialBounds, onPort, onPoint, onReady }: Props) {
  const [wide] = useState(() => typeof window !== "undefined" && window.innerWidth >= 1024);

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
      ports
        ? ({
            type: "FeatureCollection",
            features: ports.features.map((f) => ({ type: "Feature", geometry: f.geometry as unknown as GeoJSON.Point, properties: f.properties })),
          } as GeoJSON.FeatureCollection)
        : null,
    [ports],
  );

  const interactive = showPorts && portsData ? ["ports-hit", "ports-circle"] : [];

  return (
    <Map
      mapStyle={BASEMAP_STYLE}
      initialViewState={{
        bounds: initialBounds,
        fitBoundsOptions: { padding: wide ? { top: 56, bottom: 140, left: 400, right: 440 } : 24 },
      }}
      maxBounds={[CA_BOUNDS[0][0], CA_BOUNDS[0][1], CA_BOUNDS[1][0], CA_BOUNDS[1][1]]}
      minZoom={4.2}
      maxZoom={11.5}
      renderWorldCopies={false}
      attributionControl={{ compact: true }}
      interactiveLayerIds={interactive}
      onClick={onClick}
      onLoad={(e) => {
        // exposed for end-to-end tests and debugging
        (window as unknown as { __cwMap?: unknown }).__cwMap = e.target;
        onReady(e.target);
      }}
      cursor="crosshair"
      style={{ width: "100%", height: "100%" }}
    >
      <NavigationControl position="bottom-right" showCompass={false} />
      {wide && <ScaleControl position="bottom-right" unit="metric" />}

      {raster?.kind === "image" && (
        <Source key={raster.id} id="forecast" type="image" url={raster.url} coordinates={raster.corners as [[number, number], [number, number], [number, number], [number, number]]}>
          <Layer
            id="forecast-raster"
            type="raster"
            beforeId={BEFORE_OVERLAY_ID}
            paint={{ "raster-opacity": opacity, "raster-resampling": "nearest", "raster-fade-duration": 150, "raster-opacity-transition": { duration: 200 } }}
          />
        </Source>
      )}
      {raster?.kind === "tiles" && (
        <Source key={raster.id} id="observation" type="raster" tiles={[raster.template]} tileSize={256} maxzoom={raster.maxzoom}>
          <Layer id="observation-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={{ "raster-opacity": opacity, "raster-fade-duration": 150 }} />
        </Source>
      )}

      {showOfficial && officialGeometry && (
        <Source id="official" type="geojson" data={officialGeometry}>
          <Layer
            id="official-county"
            type="line"
            filter={["!=", ["get", "kind"], "lat_limit"]}
            paint={{ "line-color": OFFICIAL, "line-width": 1.6, "line-dasharray": [2, 1.5], "line-opacity": 0.9 }}
          />
          <Layer
            id="official-limit"
            type="line"
            filter={["==", ["get", "kind"], "lat_limit"]}
            paint={{ "line-color": OFFICIAL, "line-width": 2, "line-opacity": 0.95 }}
          />
          <Layer
            id="official-limit-label"
            type="symbol"
            minzoom={6.4}
            filter={["==", ["get", "kind"], "lat_limit"]}
            layout={{
              "symbol-placement": "line-center",
              "text-field": ["get", "label"],
              "text-font": ["Noto Sans Regular"],
              "text-size": 11,
              "text-offset": [0, -0.8],
              "text-allow-overlap": false,
            }}
            paint={{ "text-color": "#ffd9a0", "text-halo-color": "#040b17", "text-halo-width": 1.6 }}
          />
        </Source>
      )}

      {showPorts && portsData && (
        <Source id="ports" type="geojson" data={portsData}>
          <Layer id="ports-hit" type="circle" paint={{ "circle-radius": 14, "circle-opacity": 0 }} />
          <Layer
            id="ports-selected"
            type="circle"
            filter={["==", ["get", "port_code"], selectedPort ?? -1]}
            paint={{ "circle-radius": 11, "circle-color": "rgba(111,211,238,0.18)", "circle-stroke-color": "#6fd3ee", "circle-stroke-width": 2 }}
          />
          <Layer
            id="ports-circle"
            type="circle"
            paint={{
              "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 3, 9, 6],
              "circle-color": "#eef3fa",
              "circle-stroke-color": "#040b17",
              "circle-stroke-width": 1.5,
            }}
          />
          <Layer
            id="ports-label"
            type="symbol"
            minzoom={6.6}
            layout={{
              "text-field": ["get", "display_name"],
              "text-font": ["Noto Sans Regular"],
              "text-size": 12,
              "text-offset": [0.9, 0],
              "text-anchor": "left",
              "text-optional": true,
            }}
            paint={{ "text-color": "#e4ecf7", "text-halo-color": "#040b17", "text-halo-width": 1.6 }}
          />
        </Source>
      )}

      {inspect && (
        <Marker longitude={inspect.lon} latitude={inspect.lat} anchor="center">
          <span className="block h-4 w-4 rounded-full border-2 border-white shadow-[0_0_0_2px_rgba(4,11,23,0.85)]" aria-hidden />
        </Marker>
      )}
    </Map>
  );
}
