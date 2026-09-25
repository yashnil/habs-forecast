"use client";

import { useMemo, useState, useCallback } from "react";
import Map, {
  Layer,
  Source,
  NavigationControl,
  GeolocateControl,
  Marker,
} from "react-map-gl/mapbox";
import type { MapLayerMouseEvent } from "mapbox-gl";
import type { HarborsGeoJSON, HarborFeature } from "@/lib/types";
import {
  gibsChlorophyllTileUrlTemplate,
  GIBS_SATELLITE,
  GIBS_PACE_CHL,
} from "@/lib/gibs";
import { CA_COAST_INITIAL_VIEW, CA_COAST_MAX_BOUNDS } from "@/lib/caMapExtent";

import "mapbox-gl/dist/mapbox-gl.css";

type Props = {
  token: string;
  harbors: HarborsGeoJSON;
  viirsDate: string;
  paceDate: string;
  showGibs: boolean;
  gibsOpacity: number;
  showHarbors: boolean;
  onUserLocation: (lngLat: [number, number] | null) => void;
  onHarborSelect?: (harbor: HarborFeature) => void;
};

/** Native max tile zoom for GIBS GoogleMapsCompatible_Level7 (overscale above this in GL). */
const GIBS_RASTER_MAX_Z = 7;

export default function CoastMap({
  token,
  harbors,
  viirsDate,
  paceDate,
  showGibs,
  gibsOpacity,
  showHarbors,
  onUserLocation,
  onHarborSelect,
}: Props) {
  const [userPos, setUserPos] = useState<[number, number] | null>(null);

  const viirsTiles = useMemo(
    () => [
      gibsChlorophyllTileUrlTemplate(
        GIBS_SATELLITE.layerId,
        viirsDate,
        GIBS_SATELLITE.tileMatrixSet,
      ),
    ],
    [viirsDate],
  );

  const paceTiles = useMemo(
    () => [
      gibsChlorophyllTileUrlTemplate(
        GIBS_PACE_CHL.layerId,
        paceDate,
        GIBS_PACE_CHL.tileMatrixSet,
      ),
    ],
    [paceDate],
  );

  const onGeo = useCallback(
    (lng: number, lat: number) => {
      const p: [number, number] = [lng, lat];
      setUserPos(p);
      onUserLocation(p);
    },
    [onUserLocation],
  );

  const onMapClick = useCallback(
    (e: MapLayerMouseEvent) => {
      if (!onHarborSelect || !showHarbors) return;
      const raw = e.features?.[0];
      if (!raw || raw.geometry?.type !== "Point") return;
      const coords = raw.geometry.coordinates as [number, number];
      const name = raw.properties?.name;
      const region_key = raw.properties?.region_key;
      if (name == null || region_key == null) return;
      const harbor: HarborFeature = {
        type: "Feature",
        geometry: { type: "Point", coordinates: coords },
        properties: {
          name: String(name),
          region_key: String(region_key),
        },
      };
      onHarborSelect(harbor);
    },
    [onHarborSelect, showHarbors],
  );

  const onMapLoad = useCallback((e: { target: { resize: () => void } }) => {
    requestAnimationFrame(() => {
      try {
        e.target.resize();
      } catch {
        /* ignore */
      }
    });
  }, []);

  if (!token) {
    return (
      <div className="flex h-full min-h-[420px] items-center justify-center rounded-xl border border-amber-500/40 bg-slate-900/80 p-6 text-center text-sm text-slate-200">
        Add <code className="text-cyan-300">NEXT_PUBLIC_MAPBOX_TOKEN</code> in{" "}
        <code className="text-cyan-300">.env.local</code> to enable the interactive map.
      </div>
    );
  }

  return (
    <div className="relative min-h-[400px] h-[min(58vh,600px)] w-full overflow-hidden rounded-xl border border-slate-700/80 shadow-xl">
      <Map
        mapboxAccessToken={token}
        initialViewState={CA_COAST_INITIAL_VIEW}
        maxBounds={CA_COAST_MAX_BOUNDS}
        minZoom={4}
        maxZoom={9.5}
        renderWorldCopies={false}
        style={{ width: "100%", height: "100%" }}
        mapStyle="mapbox://styles/mapbox/dark-v11"
        reuseMaps={false}
        onLoad={onMapLoad}
        interactiveLayerIds={
          showHarbors && onHarborSelect ? ["harbors-circle"] : undefined
        }
        cursor={showHarbors && onHarborSelect ? "pointer" : undefined}
        onClick={onMapClick}
      >
        <NavigationControl position="top-right" showCompass={false} />
        <GeolocateControl
          position="top-left"
          trackUserLocation
          onGeolocate={(e) =>
            onGeo(e.coords.longitude, e.coords.latitude)
          }
        />

        {showGibs && (
          <>
            <Source
              id="gibs-pace-chl"
              type="raster"
              tiles={paceTiles}
              tileSize={256}
              maxzoom={GIBS_RASTER_MAX_Z}
            >
              <Layer
                id="gibs-pace-chl-layer"
                type="raster"
                minzoom={5.25}
                paint={{
                  "raster-opacity": gibsOpacity * 0.42,
                  "raster-fade-duration": 0,
                  "raster-resampling": "linear",
                }}
              />
            </Source>
            <Source
              id="gibs-viirs-chl"
              type="raster"
              tiles={viirsTiles}
              tileSize={256}
              maxzoom={GIBS_RASTER_MAX_Z}
            >
              <Layer
                id="gibs-viirs-chl-layer"
                type="raster"
                paint={{
                  "raster-opacity": gibsOpacity,
                  "raster-fade-duration": 0,
                  "raster-resampling": "linear",
                }}
              />
            </Source>
          </>
        )}

        {showHarbors && (
          <Source id="harbors" type="geojson" data={harbors}>
            <Layer
              id="harbors-circle"
              type="circle"
              paint={{
                "circle-radius": 6,
                "circle-color": "#38bdf8",
                "circle-stroke-width": 2,
                "circle-stroke-color": "#0f172a",
              }}
            />
            <Layer
              id="harbors-label"
              type="symbol"
              minzoom={6.5}
              layout={{
                "text-field": ["get", "name"],
                "text-size": 11,
                "text-offset": [0, 1.1],
                "text-anchor": "top",
              }}
              paint={{
                "text-color": "#e2e8f0",
                "text-halo-color": "#0f172a",
                "text-halo-width": 2,
              }}
            />
          </Source>
        )}

        {userPos && (
          <Marker longitude={userPos[0]} latitude={userPos[1]} anchor="center">
            <div className="h-4 w-4 rounded-full border-2 border-white bg-sky-400 shadow-lg ring-2 ring-sky-500/50" />
          </Marker>
        )}
      </Map>

      <div className="pointer-events-none absolute bottom-3 left-3 max-w-[280px] rounded-lg border border-slate-600/60 bg-slate-950/90 px-3 py-2 text-[10px] leading-snug text-slate-400">
        <span className="font-medium text-slate-300">Chlorophyll stack:</span> NASA PACE (under,
        {paceDate}) + VIIRS NOAA-20 (on top, {viirsDate}). PACE helps in{" "}
        <strong className="font-medium text-slate-300">turbid bays</strong> where VIIRS L3 is often
        masked. Zoom is capped so tiles stay stable; pan stays on the California coast.
      </div>
    </div>
  );
}
