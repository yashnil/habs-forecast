"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import Map, { Layer, Marker, NavigationControl, ScaleControl, Source, type MapLayerMouseEvent, type MapRef } from "react-map-gl/maplibre";
import { setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { PortsCollection } from "@/generated/schema";
import { BASEMAP_STYLE, BEFORE_OVERLAY_ID, CA_BOUNDS, CA_COAST_BOUNDS } from "@/lib/basemap";

// MapLibre's module worker is copied to public/ by scripts/copy-maplibre-worker.mjs
setWorkerUrl("/vendor/maplibre/maplibre-gl-worker.mjs");
import type { InspectPoint } from "./Inspector";

export type Raster =
  | { kind: "image"; id: string; url: string; corners: number[][] }
  | { kind: "tiles"; id: string; template: string; maxzoom: number }
  | null;

type Props = {
  raster: Raster;
  opacity: number;
  ports: PortsCollection | null;
  showPorts: boolean;
  inspect: InspectPoint | null;
  onInspect: (p: InspectPoint) => void;
};

export default function MapCanvas({ raster, opacity, ports, showPorts, inspect, onInspect }: Props) {
  const mapRef = useRef<MapRef>(null);
  const [wide] = useState(() => typeof window !== "undefined" && window.innerWidth >= 1024);

  const onClick = useCallback(
    (e: MapLayerMouseEvent) => {
      const f = e.features?.find((x) => x.layer.id === "ports-circle");
      if (f && f.geometry.type === "Point") {
        const [lon, lat] = f.geometry.coordinates as [number, number];
        const props = f.properties as Record<string, unknown>;
        onInspect({
          lat,
          lon,
          port: {
            port_code: Number(props.port_code),
            name: String(props.name),
            display_name: String(props.display_name),
            port_area: String(props.port_area),
            port_area_code: Number(props.port_area_code),
          },
        });
        return;
      }
      onInspect({ lat: e.lngLat.lat, lon: e.lngLat.lng, port: null });
    },
    [onInspect],
  );

  const portsData = useMemo(
    () =>
      ports
        ? {
            type: "FeatureCollection" as const,
            features: ports.features.map((f) => ({ type: "Feature" as const, geometry: f.geometry as unknown as GeoJSON.Point, properties: f.properties })),
          }
        : null,
    [ports],
  );

  return (
    <Map
      ref={mapRef}
      mapStyle={BASEMAP_STYLE}
      initialViewState={{
        bounds: CA_COAST_BOUNDS,
        fitBoundsOptions: { padding: wide ? { top: 40, bottom: 40, left: 430, right: 40 } : 16 },
      }}
      maxBounds={CA_BOUNDS}
      minZoom={4.2}
      maxZoom={11}
      renderWorldCopies={false}
      attributionControl={{ compact: true }}
      interactiveLayerIds={showPorts && portsData ? ["ports-circle"] : []}
      onClick={onClick}
      onLoad={(e) => {
        // exposed for end-to-end tests and debugging
        (window as unknown as { __cwMap?: unknown }).__cwMap = e.target;
      }}
      cursor="crosshair"
      style={{ width: "100%", height: "100%" }}
      aria-label="Map of the California coast"
    >
      <NavigationControl position="bottom-right" showCompass={false} />
      <ScaleControl position="bottom-left" unit="metric" />

      {raster?.kind === "image" && (
        <Source key={raster.id} id="forecast" type="image" url={raster.url} coordinates={raster.corners as [[number, number], [number, number], [number, number], [number, number]]}>
          <Layer
            id="forecast-raster"
            type="raster"
            beforeId={BEFORE_OVERLAY_ID}
            paint={{ "raster-opacity": opacity, "raster-resampling": "nearest", "raster-fade-duration": 0 }}
          />
        </Source>
      )}
      {raster?.kind === "tiles" && (
        <Source key={raster.id} id="observation" type="raster" tiles={[raster.template]} tileSize={256} maxzoom={raster.maxzoom}>
          <Layer id="observation-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={{ "raster-opacity": opacity, "raster-fade-duration": 0 }} />
        </Source>
      )}

      {showPorts && portsData && (
        <Source id="ports" type="geojson" data={portsData}>
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
            minzoom={6.2}
            layout={{
              "text-field": ["get", "display_name"],
              "text-font": ["Noto Sans Regular"],
              "text-size": 11.5,
              "text-offset": [0.8, 0],
              "text-anchor": "left",
              "text-optional": true,
            }}
            paint={{ "text-color": "#dbe5f2", "text-halo-color": "#040b17", "text-halo-width": 1.4 }}
          />
        </Source>
      )}

      {inspect && (
        <Marker longitude={inspect.lon} latitude={inspect.lat} anchor="center">
          <span className="block h-4 w-4 rounded-full border-2 border-white bg-transparent shadow-[0_0_0_2px_rgba(4,11,23,0.8)]" aria-hidden />
        </Marker>
      )}
    </Map>
  );
}
