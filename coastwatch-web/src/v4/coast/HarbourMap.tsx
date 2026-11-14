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

/** One zoom-driven width expression that makes a notice's own line heavier while it is lit. */
const litWidth = (ids: string[], base: [number, number], heavy: number) => {
  const isLit = ids.length ? ["any", ...ids.map((id) => ["in", id, ["coalesce", ["get", "record_ids"], ["literal", []]]])] : false;
  const at = (w: number) => (ids.length ? ["case", isLit, heavy, w] : w);
  return ["interpolate", ["linear"], ["zoom"], 5, at(base[0]), 9, at(base[1])];
};

/**
 * The trip's stage (v11): the place, the newest satellite chlorophyll view (observed, dated, never toxin),
 * the stations behind the briefing and the notices' limits. Beside the column on a wide screen, above it
 * on a phone. The notices' geography is the map's own interaction: tap a limit line, or a notice in its
 * sheet, and that notice's line lights and its words open. `basemapPmtiles`: offline with a saved field
 * pack, the pack's own map; undefined until the pack has been read.
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
  noticeCount = 0,
  lit = [],
  onNotices,
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
  /** the notices tied to these waters, counted once */
  noticeCount?: number;
  /** registry ids whose lines are lit */
  lit?: string[];
  /** open the notices sheet, focused on these records when a line was tapped */
  onNotices?: (recordIds: string[] | null) => void;
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
  // a phone's band keeps its top clear of the floating bar and the legend, so the waters are never under the chrome
  const pad = () => (typeof window !== "undefined" && window.matchMedia("(min-width: 1024px)").matches ? { top: 130, bottom: 140, left: 60, right: 40 } : { top: 84, bottom: 20, left: 20, right: 20 });
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
  // a notice's own line, lit (the map's official layers already exist once the map is ready)
  const litKey = lit.join("|");
  useEffect(() => {
    if (!map) return;
    const ids = litKey ? litKey.split("|") : [];
    const set = () => {
      try {
        if (map.getLayer("official-limit")) map.setPaintProperty("official-limit", "line-width", litWidth(ids, [1.4, 2], 4.2) as never);
        if (map.getLayer("official-county")) map.setPaintProperty("official-county", "line-width", litWidth(ids, [1, 1.6], 3.2) as never);
      } catch {
        // the style is not ready yet: the next change sets it
      }
    };
    if (map.isStyleLoaded()) set();
    else map.once("idle", set);
  }, [map, litKey]);
  // a tap on a notice's line reads that notice; anywhere else on the water briefs those waters
  const point = (pt: { lat: number; lon: number }) => {
    if (map && onNotices) {
      const q = map.project([pt.lon, pt.lat]);
      const near: [[number, number], [number, number]] = [[q.x - 4, q.y - 4], [q.x + 4, q.y + 4]];
      const layers = ["official-limit", "official-county"].filter((l) => map.getLayer(l));
      const hit = layers.length ? map.queryRenderedFeatures(near, { layers }) : [];
      const ids = [...new Set(hit.flatMap((f) => { const raw = (f.properties as { record_ids?: unknown }).record_ids; const list = typeof raw === "string" ? (JSON.parse(raw) as string[]) : (raw as string[] | undefined); return list ?? []; }))];
      if (ids.length) return onNotices(ids);
    }
    onPlace(pointKey(pt.lat, pt.lon));
  };
  const credit = `© OpenStreetMap contributors · OpenMapTiles · NOAA relief${field ? " · NOAA CoastWatch" : ""} · Not for navigation`;
  return (
    // one map at every width: a band above the column on a phone; beside it, the whole height of the window, on a wide screen
    <section aria-labelledby="coast-map-h" className="coast-stage" data-testid="locator" id="coast-map">
      <h2 id="coast-map-h" className="sr-only">
        The waters of this trip, the stations and the notices&rsquo; limits
      </h2>
      <div ref={box} className="env-instrument coast-stage-canvas">
        {!map && (
          <div className="absolute inset-0 grid place-items-center" data-testid="locator-loading" role="status">
            <div className="absolute inset-0">
              <ContourLoader loading={near} className="h-full w-full text-n-ink-2 opacity-30" />
            </div>
            <p className="relative text-[13.5px] text-n-ink-2">{near ? `Drawing ${place.name} and its stations…` : `Map of ${place.name}`}</p>
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
            onPoint={point}
            onReady={setMap}
          />
        )}
      </div>
      {/* the legend as the map's title: what it shows, when, and what it is not */}
      <p className="coast-stage-legend nx-deck" data-testid="locator-legend">
        {field ? (
          <>
            <b className="font-semibold text-n-ink">Chlorophyll · observed</b> · {fieldWhen?.split(", each")[0]}
            <span className="max-sm:hidden">
              {fieldWhen?.includes(", each") ? `, each${fieldWhen.split(", each")[1]}` : ""}
              {newest && now ? `, newest ${relTime(newest, now)}` : ""}
            </span>
            . Not toxin<span className="max-sm:hidden">; no colour is no clear view, not low</span>.
          </>
        ) : (
          <>No satellite chlorophyll view in this dataset: the coast, stations and notices&rsquo; limits only.</>
        )}
      </p>
      <div className="coast-stage-foot">
        <p className="coast-stage-actions max-lg:sr-only">
          <Link href={exploreHref} className="nx-link nx-focus inline-flex min-h-11 items-center gap-1.5" data-testid="locator-explore">
            <Icon name="map" className="h-4 w-4" /> {place.name} in Explore
          </Link>
          <Link href={researchHref} className="nx-link nx-focus inline-flex min-h-11 items-center gap-1.5" data-testid="locator-research">
            <Icon name="research" className="h-4 w-4" /> Its record in Research
          </Link>
        </p>
        <p className="coast-stage-hint" data-testid="locator-caption">
          {noticeCount > 0 ? (
            <button type="button" onClick={() => onNotices?.(null)} className="nx-focus nx-press inline-flex min-h-11 items-center gap-2 text-n-ink" data-testid="notices-open">
              <Icon name="shieldOff" className="h-4 w-4 text-n-official" /> {noticeCount} {noticeCount === 1 ? "notice" : "notices"} tied here, not verified · tap a line
            </button>
          ) : (
            <span>No notice line tied to these waters</span>
          )}
          <span className="max-lg:hidden"> · Tap a harbour or the water to brief those waters.</span>
          {nStations > 0 ? <span className="sr-only">, the {nStations} {nStations === 1 ? "station" : "stations"} behind this briefing</span> : null}
        </p>
        <p className="coast-stage-credit">{credit}</p>
      </div>
    </section>
  );
}
