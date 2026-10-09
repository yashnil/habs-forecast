"use client";

import { useCallback, useMemo, useState } from "react";
import Map, { Layer, Marker, NavigationControl, ScaleControl, Source, type MapLayerMouseEvent } from "react-map-gl/maplibre";
import { setWorkerUrl, type Map as MlMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import type { PortsCollection } from "@/generated/schema";
import { BASEMAP_STYLE, BEFORE_OVERLAY_ID, CA_BOUNDS, hatchImage } from "@/lib/basemap";

// MapLibre's module worker is copied to public/ by scripts/copy-maplibre-worker.mjs
setWorkerUrl("/vendor/maplibre/maplibre-gl-worker.mjs");

/** C-HARM: one Mercator image placed by its corners. */
export type ImageRaster = { id: string; url: string; corners: number[][] };
/** XYZ tiles: CoastWatch-rendered satellite tiles or third-party imagery. */
export type TileRaster = { id: string; template: string; minzoom: number; maxzoom: number; bounds?: number[] | null };

export type MapPoint = { lat: number; lon: number };

type Props = {
  forecast: ImageRaster | null;
  satellite: TileRaster | null;
  satelliteAge: TileRaster | null;
  imagery: TileRaster | null;
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
};

const OFFICIAL = "#f6bb5c";
// Every data raster is opaque and nearest-sampled: a pixel is a real source cell, and the
// legend colours are exactly the colours on the map (design reset rev. 2, §5.1).
const RASTER_PAINT = { "raster-opacity": 1, "raster-resampling": "nearest", "raster-fade-duration": 0 } as const;

export default function MapCanvas(p: Props) {
  const [wide] = useState(() => typeof window !== "undefined" && window.innerWidth >= 1024);
  const { onPort, onPoint } = p;

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
      onLoad={(e) => {
        const m = e.target;
        if (!m.hasImage("hatch")) m.addImage("hatch", hatchImage());
        m.on("styleimagemissing", (ev: { id: string }) => {
          if (ev.id === "hatch" && !m.hasImage("hatch")) m.addImage("hatch", hatchImage());
        });
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
          <Layer id="forecast-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={RASTER_PAINT} />
        </Source>
      )}
      {p.satellite && (
        <Source
          key={p.satellite.id}
          id="satellite"
          type="raster"
          tiles={[p.satellite.template]}
          tileSize={256}
          minzoom={p.satellite.minzoom}
          maxzoom={p.satellite.maxzoom}
          bounds={p.satellite.bounds as [number, number, number, number] | undefined}
        >
          <Layer id="satellite-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={RASTER_PAINT} />
        </Source>
      )}
      {p.satelliteAge && (
        <Source key={p.satelliteAge.id} id="satellite-age" type="raster" tiles={[p.satelliteAge.template]} tileSize={256} minzoom={p.satelliteAge.minzoom} maxzoom={p.satelliteAge.maxzoom} bounds={p.satelliteAge.bounds as [number, number, number, number] | undefined}>
          <Layer id="satellite-age-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={RASTER_PAINT} />
        </Source>
      )}
      {p.imagery && (
        <Source key={p.imagery.id} id="observation" type="raster" tiles={[p.imagery.template]} tileSize={256} maxzoom={p.imagery.maxzoom}>
          <Layer id="observation-raster" type="raster" beforeId={BEFORE_OVERLAY_ID} paint={{ "raster-opacity": 1, "raster-fade-duration": 0 }} />
        </Source>
      )}

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
            layout={{ "symbol-placement": "line-center", "text-field": ["get", "label"], "text-font": ["Noto Sans Regular"], "text-size": 11, "text-offset": [0, -0.8], "text-allow-overlap": false }}
            paint={{ "text-color": OFFICIAL, "text-halo-color": "#06111e", "text-halo-width": 1.6 }}
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
