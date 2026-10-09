"use client";

import { useEffect, useMemo, useRef } from "react";
import Map, { Layer, NavigationControl, Source, type MapLayerMouseEvent, type MapRef } from "react-map-gl/maplibre";
import { setWorkerUrl } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { BASEMAP_STYLE, CA_BOUNDS } from "@/lib/basemap";

setWorkerUrl("/vendor/maplibre/maplibre-gl-worker.mjs");

export type StationPoint = { id: string; name: string; lat: number; lon: number; recency: "current" | "stale" | "historical" | "unavailable" };

const RECENCY_COLOR = {
  current: "#eef3fa",
  stale: "#b4c2d6",
  historical: "#5d6f88",
  unavailable: "#5d6f88",
};

/** Shore stations as points. A point is a sampling location, not an area it represents. */
export default function StationMap({
  stations,
  selected,
  bounds,
  onSelect,
}: {
  stations: StationPoint[];
  selected: string;
  bounds: [[number, number], [number, number]];
  onSelect: (id: string) => void;
}) {
  const ref = useRef<MapRef>(null);
  const data = useMemo(
    () =>
      ({
        type: "FeatureCollection",
        features: stations.map((s) => ({
          type: "Feature",
          geometry: { type: "Point", coordinates: [s.lon, s.lat] },
          properties: { id: s.id, name: s.name, color: RECENCY_COLOR[s.recency], hollow: s.recency === "historical" || s.recency === "unavailable" ? 1 : 0 },
        })),
      }) as GeoJSON.FeatureCollection,
    [stations],
  );

  const sel = stations.find((s) => s.id === selected);
  useEffect(() => {
    const m = ref.current;
    if (!m || !sel) return;
    const b = m.getBounds();
    if (!b.contains([sel.lon, sel.lat])) m.flyTo({ center: [sel.lon, sel.lat], zoom: Math.max(m.getZoom(), 7.5), duration: 700 });
  }, [sel]);

  const onClick = (e: MapLayerMouseEvent) => {
    const f = e.features?.[0];
    if (f) onSelect(String((f.properties as Record<string, unknown>).id));
  };

  return (
    <Map
      ref={ref}
      mapStyle={BASEMAP_STYLE}
      initialViewState={{ bounds, fitBoundsOptions: { padding: 28 } }}
      maxBounds={[CA_BOUNDS[0][0], CA_BOUNDS[0][1], CA_BOUNDS[1][0], CA_BOUNDS[1][1]]}
      minZoom={4.2}
      maxZoom={12}
      renderWorldCopies={false}
      attributionControl={{ compact: true }}
      interactiveLayerIds={["stations-hit"]}
      onClick={onClick}
      cursor="pointer"
      onLoad={(e) => {
        (window as unknown as { __cwStationMap?: unknown }).__cwStationMap = e.target;
      }}
      style={{ width: "100%", height: "100%" }}
    >
      <NavigationControl position="bottom-right" showCompass={false} />
      <Source id="stations" type="geojson" data={data}>
        <Layer id="stations-hit" type="circle" paint={{ "circle-radius": 14, "circle-opacity": 0 }} />
        <Layer
          id="stations-selected"
          type="circle"
          filter={["==", ["get", "id"], selected]}
          paint={{ "circle-radius": 12, "circle-color": "rgba(111,211,238,0.16)", "circle-stroke-color": "#6fd3ee", "circle-stroke-width": 2 }}
        />
        <Layer
          id="stations-dot"
          type="circle"
          paint={{
            "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 4, 9, 7],
            "circle-color": ["case", ["==", ["get", "hollow"], 1], "#040b17", ["get", "color"]],
            "circle-stroke-color": ["case", ["==", ["get", "hollow"], 1], ["get", "color"], "#040b17"],
            "circle-stroke-width": ["case", ["==", ["get", "hollow"], 1], 2, 1.5],
          }}
        />
        <Layer
          id="stations-label"
          type="symbol"
          minzoom={7.2}
          layout={{ "text-field": ["get", "name"], "text-font": ["Noto Sans Regular"], "text-size": 12, "text-offset": [0.9, 0], "text-anchor": "left", "text-optional": true }}
          paint={{ "text-color": "#e4ecf7", "text-halo-color": "#040b17", "text-halo-width": 1.6 }}
        />
      </Source>
    </Map>
  );
}
