"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { StationLite } from "@/lib/data";
import { pointKey, type CoastPlace } from "@/lib/coast";
import { labelledGeometry } from "@/lib/officialGeometry";
import { distanceKm } from "@/lib/grid";
import type { EvidenceGraph } from "@/lib/harborEvidence";
import type { SatProduct } from "@/lib/layers";
import { coverageWhen, resolve } from "@/v4/explore/layers";
import { ContourLoader } from "@/v4/ui/ContourLoader";
import { Icon } from "@/v4/ui/Icon";
import { relTime } from "@/v4/ui/time";
import { Caps } from "./ui/Instrument";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), { ssr: false, loading: () => null });

/** The newest dated chlorophyll view the dataset holds (each pixel its own latest clear day): observed, never a forecast. */
export function chlorophyll(m: Manifest | null, baseUrl: string) {
  if (!m) return null;
  for (const product of ["multi", "olci300", "viirs750"] as SatProduct[]) {
    const r = resolve(m, { kind: "satellite", product, day: null, view: "chl", estimate: false }, baseUrl);
    if (r.satellite && r.satLayer) return { tiles: r.satellite, layer: r.satLayer };
  }
  return null;
}

/**
 * The trip's waters, live (v10 pass 2): the place, the newest satellite chlorophyll view (observed and
 * dated), the stations behind the briefing and the notices' limits: the briefing's ground on a wide
 * screen (the whole height of the window, beside the reading column), a band on a phone.
 * The map loads once its box nears the screen. `basemapPmtiles`: offline with a saved field pack, the
 * pack's own map; undefined until the pack has been read (a map made before that would ask the tile server).
 */
export function HarbourMap({
  place,
  g,
  ports,
  official,
  stations,
  exploreHref,
  researchHref,
  onPlace,
  basemapPmtiles,
  manifest = null,
  baseUrl = "",
  now = null,
}: {
  place: CoastPlace;
  g: EvidenceGraph | null;
  ports: PortsCollection | null;
  official: OfficialDataset | null;
  stations: StationLite[] | null;
  exploreHref: string;
  researchHref: string;
  onPlace: (key: string) => void;
  basemapPmtiles?: string | null;
  manifest?: Manifest | null;
  baseUrl?: string;
  now?: Date | null;
}) {
  const isPort = place.kind === "port";
  const feats = useMemo<GeoJSON.FeatureCollection>(() => {
    // a harbour is marked by the map's own selected-harbour ring; the highlight only names it (one marker, not two)
    const out: GeoJSON.Feature[] = [{ type: "Feature", geometry: { type: "Point", coordinates: [place.lon, place.lat] }, properties: { kind: isPort ? "name" : "point", label: place.name } }];
    for (const n of g?.nodes ?? []) {
      const sid = n.class === "measurement" ? n.id?.split(":")[1] : null;
      const st = sid ? stations?.find((s) => s.station_id === sid) : null;
      if (st && !out.some((f) => (f.properties as { id?: string }).id === sid)) out.push({ type: "Feature", geometry: { type: "Point", coordinates: [st.lon, st.lat] }, properties: { kind: "station", id: sid, name: st.name, recent: n.freshness?.state === "current" } });
    }
    return { type: "FeatureCollection", features: out };
  }, [place, g, stations, isPort]);
  // the stations in their own (measured, teal) marks; a station at the harbour itself is not ringed as
  // well, so the harbour reads as one marker, not two overlapping rings
  const stationMarks = useMemo<GeoJSON.FeatureCollection>(() => ({ type: "FeatureCollection", features: feats.features.filter((f) => (f.properties as { kind: string }).kind === "station").map((f) => ({ ...f, properties: { id: (f.properties as { id: string }).id, recent: (f.properties as { recent: boolean }).recent, label: (f.properties as { name: string }).name } })) }), [feats]);
  const marks = useMemo<GeoJSON.FeatureCollection>(() => ({ type: "FeatureCollection", features: feats.features.filter((f) => { const pr = f.properties as { kind: string }; if (pr.kind !== "station") return true; const [lo, la] = (f.geometry as GeoJSON.Point).coordinates; return distanceKm(place.lat, place.lon, la, lo) > 4; }) }), [feats, place]);
  const nStations = feats.features.length - 1;
  const geom = useMemo(() => labelledGeometry(official), [official]);
  const field = useMemo(() => chlorophyll(manifest, baseUrl), [manifest, baseUrl]);
  const fieldWhen = field ? coverageWhen(field.layer) : null;
  const newest = field?.layer.composite?.newest_observed_date ?? field?.layer.time.observed_date ?? null;
  // frame the place and what lies within 40 km (a station far up the coast is drawn, not framed)
  const pts = feats.features.map((f) => (f.geometry as GeoJSON.Point).coordinates).filter(([lo, la]) => distanceKm(place.lat, place.lon, la, lo) <= 40);
  const lons = pts.map((p) => p[0]);
  const lats = pts.map((p) => p[1]);
  // biased offshore (west and south on this coast), so the frame is mostly the trip's water, not the land behind it
  const w = Math.min(...lons, place.lon - 0.38) - 0.03;
  const s = Math.min(...lats, place.lat - 0.24) - 0.03;
  const e = Math.max(...lons, place.lon + 0.16) + 0.03;
  const n = Math.max(...lats, place.lat + 0.07) + 0.03;
  // a wide screen's map is the page's ground: room for the legend above and the links below
  const pad = () => (typeof window !== "undefined" && window.matchMedia("(min-width: 1024px)").matches ? { top: 130, bottom: 140, left: 60, right: 40 } : 24);
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
    if (!map) return;
    const fit = () => map.fitBounds([[w, s], [e, n]], { padding: pad(), duration: 0, maxZoom: 11 });
    fit();
    const mq = window.matchMedia("(min-width: 1024px)");
    mq.addEventListener("change", fit);
    return () => mq.removeEventListener("change", fit);
  }, [map, w, s, e, n]);
  const chip = "nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 text-[14px] text-n-ink underline decoration-n-line-strong underline-offset-4 hover:decoration-n-ink lg:min-h-9 lg:rounded-[3px] lg:border lg:border-n-line lg:bg-[rgba(5,9,13,0.86)] lg:px-3 lg:text-[13.5px] lg:no-underline lg:hover:border-n-line-strong";
  const credit = `© OpenStreetMap contributors · OpenMapTiles · NOAA relief${field ? " · NOAA CoastWatch" : ""} · Not for navigation`;
  return (
    // one map at every width: a band under the readings on a phone; on a wide screen the whole height of the
    // window beside the reading column (Briefing makes it sticky), with its legend and links set on the water
    <section aria-labelledby="coast-map-h" className="flex flex-col gap-2 lg:relative lg:block lg:h-full" data-testid="locator" id="coast-map">
      <h2 id="coast-map-h" className="lg:sr-only">
        <Caps>The waters · stations · notices&rsquo; limits</Caps>
      </h2>
      <div ref={box} className="env-instrument relative -mx-4 h-[210px] overflow-hidden border-y border-n-line sm:mx-0 sm:h-[300px] sm:rounded-[4px] sm:border lg:absolute lg:inset-0 lg:h-auto lg:rounded-none lg:border-0">
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
            satellite={field?.tiles ?? null}
            satelliteAge={null}
            imagery={null}
            ports={ports}
            showPorts
            officialGeometry={geom}
            showOfficial
            selectedPort={isPort ? Number(place.key.slice(5)) : null}
            inspect={null}
            chrome="none"
            limitLabelGuard={56}
            basemapPmtiles={basemapPmtiles}
            highlight={marks}
            stations={stationMarks}
            initialBounds={[[w, s], [e, n]]}
            initialPadding={pad()}
            onPort={(code) => onPlace(`port-${code}`)}
            onPoint={(pt) => onPlace(pointKey(pt.lat, pt.lon))}
            onReady={setMap}
          />
        )}
        {/* the reading column's edge fades into the water: the map is the page's ground, not a box */}
        <div aria-hidden className="pointer-events-none absolute inset-y-0 left-0 hidden w-12 bg-gradient-to-r from-[var(--nx-bg)] to-transparent lg:block" />
        <p className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-[rgba(5,9,13,0.9)] to-transparent px-2.5 pb-1.5 pt-5 text-right text-[10.5px] text-n-ink-2 lg:hidden">{credit}</p>
      </div>
      <p className="text-[12px] leading-snug text-n-ink-2 lg:pointer-events-none lg:absolute lg:right-4 lg:top-[92px] lg:max-w-[300px] lg:rounded-[4px] lg:bg-[rgba(5,9,13,0.84)] lg:px-3 lg:py-2 lg:text-right" data-testid="locator-legend">
        {field ? (
          <>
            <span className="nx-caps text-n-measured">Chlorophyll-a · satellite, observed</span>
            <br />
            {fieldWhen}
            {newest && now ? ` · newest ${relTime(newest, now)}` : ""}. Not toxin; no colour is no clear view, not low.
          </>
        ) : (
          <>No satellite chlorophyll view in this dataset: the coast, stations and notices&rsquo; limits only.</>
        )}
      </p>
      <div className="flex flex-col gap-1 lg:pointer-events-none lg:absolute lg:inset-x-0 lg:bottom-0 lg:gap-2 lg:bg-gradient-to-t lg:from-[rgba(5,9,13,0.95)] lg:via-[rgba(5,9,13,0.8)] lg:to-transparent lg:px-5 lg:pb-2 lg:pt-12">
        <p className="max-w-[52ch] text-[12.5px] leading-snug text-n-ink-2 max-sm:sr-only" data-testid="locator-caption">
          {place.name}
          {nStations > 0 ? `, the ${nStations} ${nStations === 1 ? "station" : "stations"} behind this briefing` : ""} and the notices&rsquo; limits (amber), from their official geography. <span className="text-n-ink">Tap a harbour or the water to brief those waters.</span>
        </p>
        <p className="flex flex-wrap gap-x-5 gap-y-0 lg:gap-2">
          <Link href={exploreHref} className={`${chip} lg:pointer-events-auto`} data-testid="locator-explore">
            <Icon name="map" className="h-4 w-4" /> {place.name} in Explore
          </Link>
          <Link href={researchHref} className={`${chip} lg:pointer-events-auto`} data-testid="locator-research">
            <Icon name="research" className="h-4 w-4" /> Its record in Research
          </Link>
        </p>
        <p className="hidden text-right text-[10.5px] text-n-ink-2 lg:block">{credit}</p>
      </div>
    </section>
  );
}
