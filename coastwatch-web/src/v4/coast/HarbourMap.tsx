"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import type { PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { StationLite } from "@/lib/data";
import { pointKey, type CoastPlace } from "@/lib/coast";
import { labelledGeometry } from "@/lib/officialGeometry";
import { distanceKm } from "@/lib/grid";
import type { EvidenceGraph } from "@/lib/harborEvidence";
import { ContourLoader } from "@/v4/ui/ContourLoader";
import { Icon } from "@/v4/ui/Icon";
import { Caps } from "./ui/Instrument";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), { ssr: false, loading: () => null });

/**
 * The harbour's waters, directly (v10): the place, the stations behind the briefing and the notices'
 * limits, with the routes to Explore and Research keeping the place. Its box has a fixed height and the
 * map loads once the box nears the screen. `basemapPmtiles`: offline with a saved field pack, the pack's
 * own map; undefined until the pack has been read (a map made before that would ask the tile server).
 */
export function HarbourMap({ place, g, ports, official, stations, exploreHref, researchHref, onPlace, basemapPmtiles }: { place: CoastPlace; g: EvidenceGraph | null; ports: PortsCollection | null; official: OfficialDataset | null; stations: StationLite[] | null; exploreHref: string; researchHref: string; onPlace: (key: string) => void; basemapPmtiles?: string | null }) {
  const feats = useMemo<GeoJSON.FeatureCollection>(() => {
    const out: GeoJSON.Feature[] = [{ type: "Feature", geometry: { type: "Point", coordinates: [place.lon, place.lat] }, properties: { kind: place.kind === "port" ? "harbor" : "point", label: place.name } }];
    for (const n of g?.nodes ?? []) {
      const sid = n.class === "measurement" ? n.id?.split(":")[1] : null;
      const st = sid ? stations?.find((s) => s.station_id === sid) : null;
      if (st && !out.some((f) => (f.properties as { id?: string }).id === sid)) out.push({ type: "Feature", geometry: { type: "Point", coordinates: [st.lon, st.lat] }, properties: { kind: "station", id: sid } });
    }
    return { type: "FeatureCollection", features: out };
  }, [place, g, stations]);
  const nStations = feats.features.length - 1;
  const geom = useMemo(() => labelledGeometry(official), [official]);
  // frame the place and what lies within 40 km (a station far up the coast is drawn, not framed)
  const pts = feats.features.map((f) => (f.geometry as GeoJSON.Point).coordinates).filter(([lo, la]) => distanceKm(place.lat, place.lon, la, lo) <= 40);
  const lons = pts.map((p) => p[0]);
  const lats = pts.map((p) => p[1]);
  const w = Math.min(...lons, place.lon - 0.2) - 0.03;
  const s = Math.min(...lats, place.lat - 0.14) - 0.03;
  const e = Math.max(...lons, place.lon + 0.2) + 0.03;
  const n = Math.max(...lats, place.lat + 0.14) + 0.03;
  const [map, setMap] = useState<import("maplibre-gl").Map | null>(null);
  const box = useRef<HTMLDivElement | null>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof IntersectionObserver === "undefined") return setNear(true);
    const io = new IntersectionObserver((es) => es.some((x) => x.isIntersecting) && setNear(true), { rootMargin: "200px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  useEffect(() => {
    map?.fitBounds([[w, s], [e, n]], { padding: 28, duration: 0, maxZoom: 11 });
  }, [map, w, s, e, n]);
  const linkCls = "nx-focus inline-flex min-h-11 items-center gap-1.5 text-[14px] text-n-ink underline decoration-n-line-strong underline-offset-4 hover:decoration-n-ink";
  return (
    <section aria-labelledby="coast-map-h" className="flex flex-col gap-2" data-testid="locator" id="coast-map">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h2 id="coast-map-h">
          <Caps>The waters · stations · notices&rsquo; limits</Caps>
        </h2>
      </div>
      <div ref={box} className="env-instrument relative h-[260px] overflow-hidden rounded-[4px] border border-n-line sm:h-[300px] lg:h-[320px]">
        {!map && (
          <div className="absolute inset-0 grid place-items-center" data-testid="locator-loading" role="status">
            <div className="absolute inset-0">
              <ContourLoader loading={near} className="h-full w-full text-n-accent opacity-30" />
            </div>
            <p className="relative text-[13px] text-n-ink-2">{near ? `Drawing ${place.name} and its stations…` : `Map of ${place.name}`}</p>
          </div>
        )}
        {near && basemapPmtiles !== undefined && (
          <MapCanvas
            forecast={null}
            satellite={null}
            satelliteAge={null}
            imagery={null}
            ports={ports}
            showPorts
            officialGeometry={geom}
            showOfficial
            selectedPort={place.kind === "port" ? Number(place.key.slice(5)) : null}
            inspect={null}
            chrome="none"
            basemapPmtiles={basemapPmtiles}
            highlight={feats}
            initialBounds={[[w, s], [e, n]]}
            initialPadding={24}
            onPort={(code) => onPlace(`port-${code}`)}
            onPoint={(pt) => onPlace(pointKey(pt.lat, pt.lon))}
            onReady={setMap}
          />
        )}
        <p className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-[rgba(5,9,13,0.9)] to-transparent px-2.5 pb-1.5 pt-5 text-right text-[10.5px] text-n-ink-2">© OpenStreetMap contributors · OpenMapTiles · NOAA relief · Not for navigation</p>
      </div>
      <p className="text-[12.5px] leading-snug text-n-ink-2" data-testid="locator-caption">
        {place.name}
        {nStations > 0 ? `, the ${nStations} ${nStations === 1 ? "station" : "stations"} behind this briefing` : ""} and the notices&rsquo; limits (amber), drawn from their official geography. <span className="text-n-ink">Tap a harbour or the water to brief those waters.</span>
      </p>
      <p className="flex flex-wrap gap-x-5">
        <Link href={exploreHref} className={linkCls} data-testid="locator-explore">
          <Icon name="map" className="h-4 w-4" /> {place.name} in Explore
        </Link>
        <Link href={researchHref} className={linkCls} data-testid="locator-research">
          <Icon name="research" className="h-4 w-4" /> Its record in Research
        </Link>
      </p>
    </section>
  );
}
