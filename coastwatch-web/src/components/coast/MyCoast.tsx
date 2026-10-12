"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Map as MlMap } from "maplibre-gl";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { StationLite } from "@/lib/data";
import { COAST_COPY } from "@/content/coast";
import { associationsAt } from "@/lib/applicability";
import {
  ageDays,
  ageWords,
  cellPolygon,
  coastHref,
  parseAudience,
  pointKey,
  portPlaces,
  resolvePlace,
  sitePlaces,
  stationsNear,
  type Audience,
  type CoastPlace,
} from "@/lib/coast";
import { CHARM_LEADS, charmLayer, sourceStatus } from "@/lib/layers";
import { classifyTime } from "@/lib/freshness";
import { officialVerification } from "@/lib/official";
import { formatDate, pacificToday } from "@/lib/time";
import { useNow } from "@/lib/useNow";
import { useOfficialDrawer } from "@/components/shell/OfficialShell";
import { Icon } from "@/components/ui/Icon";
import { PlacePicker } from "./PlacePicker";
import { MeasuredSection, ModelSection, ObservedSection, OfficialSection, UnknownsSection, modelHeadline, useModel } from "./CoastSections";
import type { SatelliteResult } from "@/components/map/SatelliteNear";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-[#0b1d33]" role="status" aria-label="Loading map" />,
});

const HOME_KEY = "cw-home-port";
const CA: [[number, number], [number, number]] = [[-125.2, 32.5], [-117.1, 42.0]];
const EXAMPLES = ["port-201", "port-220", "port-593", "port-550", "port-770", "site-clam-beach"];

function readHome(): string | null {
  try {
    return window.localStorage.getItem(HOME_KEY);
  } catch {
    return null;
  }
}

type Props = {
  manifest: Manifest;
  ports: PortsCollection | null;
  official: OfficialDataset | null;
  stations: StationLite[] | null;
  stationsError: string | null;
  baseUrl: string;
};

/**
 * My Coast: one location's official information, nearby measurements, forecast availability and
 * recent observations, each traceable to its source and drawn on the map where it was obtained.
 * The location is the waters someone cares about (a port, a CDPH harvest site or a point on the
 * map); the home port is remembered separately and never assumed to be where anyone fishes.
 */
export function MyCoast({ manifest, ports, official, stations, stationsError, baseUrl }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const sp = useSearchParams();
  const now = useNow();
  const placeKey = sp.get("place");
  const audience = parseAudience(sp.get("for"));
  const homeParam = sp.get("home");
  const [homeStored, setHomeStored] = useState<string | null>(null);
  useEffect(() => setHomeStored(readHome()), []);
  const homeKey = homeParam ?? homeStored;

  const allPlaces = useMemo(() => [...portPlaces(ports), ...sitePlaces()], [ports]);
  const place = useMemo(() => resolvePlace(placeKey, ports), [placeKey, ports]);
  const home = useMemo(() => (homeKey ? resolvePlace(homeKey, ports) : null), [homeKey, ports]);

  const go = useCallback(
    (next: { place?: string | null; home?: string | null; for?: Audience }, mode: "push" | "replace" = "push") => {
      const href = coastHref({ place: next.place !== undefined ? next.place : placeKey, home: next.home !== undefined ? next.home : homeParam, for: next.for ?? audience });
      (mode === "push" ? router.push : router.replace)(href, { scroll: false });
    },
    [router, placeKey, homeParam, audience],
  );
  const pick = useCallback((p: CoastPlace) => go({ place: p.key }), [go]);

  const setHome = (key: string | null) => {
    try {
      if (key) window.localStorage.setItem(HOME_KEY, key);
      else window.localStorage.removeItem(HOME_KEY);
    } catch {
      // not remembered on this device
    }
    setHomeStored(key);
    go({ home: null }, "replace");
  };

  // ------------------------------------------------------------ facts for the chosen waters
  const verification = useMemo(() => (official && now ? officialVerification(official, sourceStatus(manifest, "official"), now) : null), [official, manifest, now]);
  const assocs = useMemo(() => (place ? associationsAt(official, place.key, place.lat, place.lon, undefined, place.kind === "point" ? "water" : "shore") : []), [official, place]);
  const st = useMemo(() => (place ? stationsNear(stations, place.lat, place.lon) : { near: [], nearestBeyond: null }), [stations, place]);
  const leads = CHARM_LEADS.map((l) => charmLayer(manifest, "particulate_domoic", l));
  const today = now ? pacificToday(now) : null;
  const [lead, setLead] = useState<number | null>(null);
  useEffect(() => {
    if (lead !== null || !today) return;
    const i = leads.findIndex((l) => l?.time.valid_date === today);
    const issued = leads.map((l, k) => (l ? k : -1)).filter((k) => k >= 0);
    setLead(i >= 0 ? i : (issued.at(-1) ?? 0));
  }, [today, lead, leads]);
  const leadNow = lead ?? 0;
  const model = useModel(manifest, baseUrl, place?.lat ?? null, place?.lon ?? null, leadNow);
  // the observations section reads the satellite grids once and reports what it found
  const [sat, setSat] = useState<SatelliteResult>(null);
  useEffect(() => setSat(null), [place?.key]);

  // ------------------------------------------------------------ the map shows where each fact came from
  const overlay = useMemo<GeoJSON.FeatureCollection | null>(() => {
    if (!place) return null;
    const f: GeoJSON.Feature[] = [{ type: "Feature", properties: { kind: "target" }, geometry: { type: "Point", coordinates: [place.lon, place.lat] } }];
    for (const s of st.near.slice(0, 4)) f.push({ type: "Feature", properties: { kind: "station", name: s.station.name }, geometry: { type: "Point", coordinates: [s.station.lon, s.station.lat] } });
    const g = model?.layer?.grid;
    const used = model?.pda?.here ?? model?.pda?.nearest ?? null;
    if (g && used) {
      f.push({ type: "Feature", properties: { kind: "cell" }, geometry: cellPolygon(g, used.cell) });
      if (!model?.pda?.here) f.push({ type: "Feature", properties: { kind: "link" }, geometry: { type: "LineString", coordinates: [[place.lon, place.lat], [used.cell.lon, used.cell.lat]] } });
    }
    return { type: "FeatureCollection", features: f };
  }, [place, st, model]);

  const mapRef = useRef<MlMap | null>(null);
  const [reducedMotion, setReducedMotion] = useState(false);
  useEffect(() => setReducedMotion(window.matchMedia("(prefers-reduced-motion: reduce)").matches), []);
  const fit = useCallback(
    (m: MlMap) => {
      if (!place) return m.fitBounds(CA, { padding: 24, duration: 0 });
      const pts: [number, number][] = [[place.lon, place.lat]];
      const used = model?.pda?.here ?? model?.pda?.nearest;
      if (used) pts.push([used.cell.lon, used.cell.lat]);
      for (const s of st.near.slice(0, 2)) if (s.km < 25) pts.push([s.station.lon, s.station.lat]);
      const lons = pts.map((p) => p[0]);
      const lats = pts.map((p) => p[1]);
      const pad = 0.06;
      m.fitBounds([[Math.min(...lons) - pad, Math.min(...lats) - pad], [Math.max(...lons) + pad, Math.max(...lats) + pad]], { padding: 48, maxZoom: 11, duration: reducedMotion ? 0 : 600 });
    },
    [place, model, st, reducedMotion],
  );
  useEffect(() => {
    if (mapRef.current) fit(mapRef.current);
  }, [fit]);

  // ------------------------------------------------------------ what is not known
  const unknowns = useMemo(() => {
    if (!place) return [];
    const out: string[] = [];
    const tied = assocs.filter((a) => a.tier === "geographic" && a.record.area.type !== "statewide");
    if (!tied.length) out.push("No notice in CoastWatch's list is tied to these waters by its geography. The list may be incomplete; check the agencies.");
    out.push("Whether each notice applies here was worked out from the notices' official geography; no person has reviewed these associations, and the notices themselves are not verified.");
    const nUnresolved = assocs.filter((a) => a.tier === "unresolved").length;
    if (nUnresolved) out.push(`Whether ${nUnresolved === 1 ? "one notice applies" : `${nUnresolved} notices apply`} to these waters is not settled by ${nUnresolved === 1 ? "its" : "their"} official geography.`);
    const recentToxin = st.near.some((s) => s.station.latest.some((m) => m.variable === "pDA" && now && ageDays(m.date, now) <= 30));
    if (!recentToxin) out.push("There is no toxin measurement from the last 30 days within 40 km.");
    if (model?.pda && !model.pda.here) out.push(`C-HARM gives no toxin probability for these waters${model.pda.nearest ? `; its nearest value is ${model.pda.nearest.km.toFixed(1)} km away and is not a forecast here` : ""}.`);
    const lyr = model?.layer;
    if (lyr && now && classifyTime(lyr.freshness, lyr.time, now).state !== "current") out.push("The latest C-HARM run is not current.");
    if (sat && !sat.date) out.push(`No clear satellite view of these waters in the last week (cloud, fog or no overpass)${sat.nearestKm != null ? `; the nearest clear pixel was ${sat.nearestKm.toFixed(1)} km away` : ""}.`);
    out.push("Toxin in seafood is not measured by CoastWatch. CDPH tests shellfish; paralytic shellfish poisoning (PSP) results are not shown here.");
    return out;
  }, [place, assocs, st, model, sat, now]);

  const { openDrawer } = useOfficialDrawer();

  // ------------------------------------------------------------ render
  const map = (
    <MapCanvas
      forecast={null}
      satellite={null}
      satelliteAge={null}
      imagery={null}
      ports={ports}
      showPorts
      officialGeometry={(official?.geometry as unknown as GeoJSON.FeatureCollection) ?? null}
      showOfficial
      selectedPort={place?.kind === "port" ? Number(place.key.slice(5)) : null}
      inspect={null}
      overlay={overlay}
      initialBounds={place ? [[place.lon - 0.3, place.lat - 0.2], [place.lon + 0.3, place.lat + 0.2]] : CA}
      initialPadding={32}
      onPort={(code) => go({ place: `port-${code}` })}
      onPoint={(p) => go({ place: pointKey(p.lat, p.lon) })}
      onReady={(m) => {
        mapRef.current = m;
        // on a small map the credits start folded (the (i) button opens them), as on Bloom's map
        if (!window.matchMedia("(min-width: 1024px)").matches) m.getContainer().querySelector(".maplibregl-ctrl-attrib")?.classList.remove("maplibregl-compact-show");
        fit(m);
      }}
    />
  );

  const header = (
    <header className="space-y-3 sm:space-y-4" data-testid="coast-header">
      <div>
        <p className="text-[12px] font-semibold uppercase tracking-[0.1em] text-accent">{COAST_COPY.title}</p>
        <div className="mt-1 flex items-start justify-between gap-3">
          <h1 className="text-balance font-display text-[30px] font-medium leading-[1.08] text-ink sm:text-[40px]" data-testid="coast-title">
            {place ? place.name : "Choose the waters you care about"}
          </h1>
          {place && <CopyLink />}
        </div>
        {place ? (
          <p className="mt-1.5 text-[15px] text-ink-2" data-testid="coast-subtitle">
            {place.note}
            {place.county ? ` · ${place.county} County` : ""}
            <span className={`tabular max-sm:hidden ${place.kind === "point" ? "hidden" : ""}`}>
              {" "}
              · {place.lat.toFixed(3)}°N {Math.abs(place.lon).toFixed(3)}°W
            </span>
          </p>
        ) : (
          <p className="mt-2 max-w-[60ch] text-[17px] leading-relaxed text-ink-2">{COAST_COPY.lede}</p>
        )}
        {!place && placeKey && (
          <p className="mt-2 rounded-lg bg-official-bg px-3 py-2 text-[14px] text-official-ink" role="status" data-testid="coast-unknown-place">
            The place in this link (“{placeKey}”) is not one CoastWatch knows. Choose a port, a harvest site or a point on the map.
          </p>
        )}
      </div>
      <PlacePicker places={allPlaces} current={place} onPick={pick} />
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px]" data-testid="coast-home">
        {home ? (
          <>
            <span className="text-ink-2">
              Home port: <b className="font-semibold text-ink">{home.name}</b>
            </span>
            {place?.key !== home.key && (
              <button type="button" onClick={() => pick(home)} className="min-h-[32px] font-medium text-accent hover:underline" data-testid="coast-home-open">
                Show home port
              </button>
            )}
            <button type="button" onClick={() => setHome(null)} className="min-h-[32px] text-ink-3 hover:text-ink" data-testid="coast-home-clear">
              Clear
            </button>
          </>
        ) : place?.kind === "port" ? (
          <button type="button" onClick={() => setHome(place.key)} className="min-h-[32px] font-medium text-accent hover:underline" data-testid="coast-home-set">
            Make {place.name} my home port
          </button>
        ) : (
          <span className="text-ink-3">Save a home port to return to it; it is kept on this device only.</span>
        )}
      </div>
    </header>
  );

  const tied = assocs.filter((a) => a.tier === "geographic" && a.record.area.type !== "statewide").length;
  const statewide = assocs.filter((a) => a.record.area.type === "statewide").length;
  const unresolved = assocs.filter((a) => a.tier === "unresolved").length;
  const lastToxin = st.near
    .flatMap((s) => s.station.latest.filter((m) => m.variable === "pDA").map((m) => ({ s, m })))
    .sort((a, b) => b.m.date.localeCompare(a.m.date))[0];

  const glance = place && (
    <nav aria-label="At a glance" data-testid="coast-glance" className="overflow-hidden rounded-2xl border border-hairline bg-surface">
      <dl className="divide-y divide-hairline text-[15px]">
        <GlanceRow href="#coast-official-h" dot="bg-official" label="Official" testid="glance-official">
          {official ? (
            <>
              {tied ? `${tied} notice${tied === 1 ? "" : "s"} tied to these waters` : "No notice tied to these waters"}
              {statewide ? `, ${statewide} statewide` : ""}
              {unresolved ? `; ${unresolved} may apply (unresolved)` : ""} · <span className="text-official-ink">not verified</span>
            </>
          ) : (
            "Unavailable: check CDFW and CDPH"
          )}
        </GlanceRow>
        <GlanceRow href="#coast-measured-h" dot="bg-measured" label="Measured" testid="glance-measured">
          {lastToxin && now ? (
            <>
              Toxin last sampled {ageWords(ageDays(lastToxin.m.date, now))}
              {ageDays(lastToxin.m.date, now) > 90 ? ` (${formatDate(lastToxin.m.date, { year: true })})` : ""} at {lastToxin.s.station.name}, {lastToxin.s.km.toFixed(lastToxin.s.km < 10 ? 1 : 0)} km away
            </>
          ) : stationsError ? (
            "Station data unavailable"
          ) : (
            "No toxin measurement within 40 km"
          )}
        </GlanceRow>
        <GlanceRow href="#coast-model-h" dot="bg-model" label="Forecast" testid="glance-model">
          {modelHeadline(model)}
          {model && !model.error && !model.pda?.here && !model.pn?.here && (model.pda?.nearest ? ` · nearest value ${model.pda.nearest.km.toFixed(1)} km ${model.pda.nearest.bearing}, not a forecast here` : " or within 60 km")}
          {model?.layer && now && classifyTime(model.layer.freshness, model.layer.time, now).state !== "current" && <span className="text-[#9b2c2c]"> · stale run</span>}
        </GlanceRow>
        <GlanceRow href="#coast-observed-h" dot="bg-measured" label="Satellite" testid="glance-satellite">
          {sat === null
            ? "Reading…"
            : sat.date && now
              ? `Clear view ${formatDate(sat.date)} (${ageWords(ageDays(sat.date, now))})`
              : sat.nearestKm != null
                ? `No clear pixel at these waters this week; the nearest is ${sat.nearestKm.toFixed(1)} km away`
                : "No clear view of these waters in the last week"}
        </GlanceRow>
      </dl>
      <p className="border-t border-hairline px-4 py-2.5 text-[13px] text-ink-3">
        A summary of what exists, not a rating. Nothing here says waters are open or seafood is safe.{" "}
        <button type="button" onClick={(e) => openDrawer(e.currentTarget)} className="font-medium text-accent hover:underline">
          All official notices
        </button>
      </p>
    </nav>
  );

  const landing = !place && (
    <section aria-labelledby="coast-start-h" className="space-y-3" data-testid="coast-landing">
      <h2 id="coast-start-h" className="text-[13px] font-semibold uppercase tracking-[0.08em] text-ink-3">
        Start with
      </h2>
      <ul className="grid gap-2 sm:grid-cols-2">
        {(home ? [home.key, ...EXAMPLES.filter((k) => k !== home.key)] : EXAMPLES).map((k) => {
          const p = resolvePlace(k, ports);
          if (!p) return null;
          return (
            <li key={k}>
              <Link
                href={coastHref({ place: k, home: homeParam, for: audience })}
                scroll={false}
                className="flex min-h-[52px] items-center justify-between gap-3 rounded-xl border border-hairline bg-surface px-4 py-2.5 hover:border-ink-3"
                data-testid={`coast-start-${k}`}
              >
                <span>
                  <span className="block text-[16px] font-medium text-ink">{p.name}</span>
                  <span className="block text-[13px] text-ink-3">{k === home?.key ? "Your home port" : p.note}</span>
                </span>
                <span aria-hidden className="text-ink-3">→</span>
              </Link>
            </li>
          );
        })}
      </ul>
      <p className="text-[14px] text-ink-3">Or click any water on the map to choose those waters.</p>
    </section>
  );

  return (
    <div className="lg:grid lg:grid-cols-[minmax(0,620px)_minmax(0,1fr)]" data-testid="my-coast">
      <div className="min-w-0 space-y-5 px-4 pb-24 pt-4 sm:space-y-8 sm:px-8 sm:pt-6 lg:pb-16">
        {header}
        {glance}
        <div className="h-[42vh] min-h-[260px] overflow-hidden rounded-2xl lg:hidden" data-testid="coast-map-mobile">
          {/* one map: phones get it inline, wide screens in the sticky column */}
          <MapSlot wide={false}>{map}</MapSlot>
        </div>
        {landing}
        {place && (
          <>
            <OfficialSection official={official} assocs={assocs} audience={audience} place={place} verification={verification} onAudience={(a) => go({ for: a }, "replace")} />
            <MeasuredSection near={st.near} nearestBeyond={st.nearestBeyond} now={now} />
            <ModelSection manifest={manifest} model={model} lead={leadNow} onLead={setLead} now={now} />
            <ObservedSection manifest={manifest} baseUrl={baseUrl} place={place} onSat={setSat} />
            <UnknownsSection items={unknowns} />
            <p className="border-t border-hairline pt-4 text-[13px] leading-relaxed text-ink-3">
              Sources: CDFW and CDPH notices (transcribed, not verified); CalHABMAP via SCCOOS ERDDAP; NOAA C-HARM v3.1; NOAA CoastWatch satellite chlorophyll; HF-radar
              currents. {pathname === "/coast" ? "This page's link reproduces this view." : ""}
            </p>
          </>
        )}
      </div>
      <div className="hidden lg:block">
        <div className="sticky top-[var(--cw-masthead-h)] h-[calc(100dvh-var(--cw-masthead-h))]" data-testid="coast-map-desktop">
          <MapSlot wide>{map}</MapSlot>
        </div>
      </div>
    </div>
  );
}

/** Renders the single map in whichever slot matches the screen, so it is never mounted twice. */
function MapSlot({ wide, children }: { wide: boolean; children: React.ReactNode }) {
  const [isWide, setIsWide] = useState<boolean | null>(null);
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    const on = () => setIsWide(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  if (isWide === null || isWide !== wide) return null;
  return <div className="relative h-full w-full">{children}</div>;
}

function GlanceRow({ href, dot, label, testid, children }: { href: string; dot: string; label: string; testid: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[92px_1fr] items-baseline gap-3 px-4 py-2 sm:grid-cols-[110px_1fr] sm:py-2.5" data-testid={testid}>
      <dt className="flex items-center gap-2 text-[13px] font-semibold uppercase tracking-[0.06em] text-ink-3">
        <span aria-hidden className={`h-2 w-2 rounded-full ${dot}`} />
        {label}
      </dt>
      <dd className="text-ink">
        <a href={href} className="hover:underline">
          {children}
        </a>
      </dd>
    </div>
  );
}

function CopyLink() {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(window.location.href);
          setDone(true);
          setTimeout(() => setDone(false), 2000);
        } catch {
          setDone(false);
        }
      }}
      aria-label={done ? "Link copied" : "Copy a link to these waters"}
      title="Copy a link to these waters"
      className="mt-1 flex min-h-[40px] shrink-0 items-center gap-1.5 rounded-full border border-hairline-strong bg-surface px-3 text-[14px] text-ink-2 hover:border-ink-3"
      data-testid="coast-copy-link"
    >
      <Icon name="link" className="h-4 w-4" />
      <span className="max-sm:sr-only">{done ? "Copied" : "Copy link"}</span>
      <span className="sr-only" aria-live="polite">{done ? "Link copied" : ""}</span>
    </button>
  );
}
