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
import { provenanceFeatures } from "@/lib/provenance";
import { journeyTo } from "@/lib/camera";
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
import { PurposeSwitch } from "@/components/welcome/PurposeSwitch";
import { defaultAudience, sectionOrder, usePurpose } from "@/lib/purpose";
import { MeasuredSection, ModelSection, ObservedSection, OfficialSection, UnknownsSection, useModel } from "./CoastSections";
import type { SatelliteResult } from "@/components/map/SatelliteNear";
import { AnimatePresence } from "motion/react";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { ImageRaster, TileRaster } from "@/components/map/MapCanvas";
import { HabBriefing, useHabEvidence, useOutlook, useSatellite, type Focus } from "./HabBriefing";
import { MapKey, StationPeek, WatersAndSpecies } from "./Waters";
import { fmtMeasured, fmtPct, ringPolygon, stationsAround, NEIGHBOURHOOD_KM, RECENT_DAYS } from "@/lib/hab";
import { artifactUrl, satelliteLatest, tileIndexUrl } from "@/lib/layers";
import { supportsGridTiles } from "@/lib/gridTiles";
import { gradientCss } from "@/lib/palette";
import { speciesStatements, type NearSample } from "@/lib/nextSteps";
import { AnswerLine, ConditionsCard, NextSteps, OfficialBrief, PlanBar, fmtWhen, officialAnswer, useConditions, type AnswerClause } from "./Briefing";
import { MEASURED_RECENT_DAYS } from "@/components/ui/Mark";
import { speciesById, type SpeciesId } from "@/content/species";
import { useMode } from "@/lib/mode";
import { ModeSwitch } from "@/components/shell/ModeNav";
import { nextSteps } from "@/lib/nextSteps";
import { currentRecords } from "@/lib/official";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-[#0b1d33]" role="status" aria-label="Loading map" />,
});

const HOME_KEY = "cw-home-port";
const CA: [[number, number], [number, number]] = [[-125.2, 32.5], [-117.1, 42.0]];
const EXAMPLES = ["port-201", "port-220", "port-593", "port-550", "port-770", "site-clam-beach"];

const reducedMotionNow = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

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
  portIntel?: PortIntelCollection | null;
};

/**
 * My Coast: one location's official information, nearby measurements, forecast availability and
 * recent observations, each traceable to its source and drawn on the map where it was obtained.
 * The location is the waters someone cares about (a port, a CDPH harvest site or a point on the
 * map); the home port is remembered separately and never assumed to be where anyone fishes.
 */
export function MyCoast({ manifest, ports, official, stations, stationsError, baseUrl, portIntel = null }: Props) {
  const router = useRouter();
  const pathname = usePathname();
  const sp = useSearchParams();
  const now = useNow();
  const placeKey = sp.get("place");
  // a link's own filter wins; otherwise the saved purpose's default (only a default: nothing is hidden)
  const [purpose] = usePurpose();
  const audience = sp.get("for") ? parseAudience(sp.get("for")) : defaultAudience(purpose);
  const homeParam = sp.get("home");
  const species = speciesById(sp.get("species"));
  const [viewMode, , modeRead] = useMode();
  const [homeStored, setHomeStored] = useState<string | null>(null);
  useEffect(() => setHomeStored(readHome()), []);
  const homeKey = homeParam ?? homeStored;

  const allPlaces = useMemo(() => [...portPlaces(ports), ...sitePlaces()], [ports]);
  const place = useMemo(() => resolvePlace(placeKey, ports), [placeKey, ports]);
  const home = useMemo(() => (homeKey ? resolvePlace(homeKey, ports) : null), [homeKey, ports]);

  const go = useCallback(
    (next: { place?: string | null; home?: string | null; for?: Audience; species?: SpeciesId }, how: "push" | "replace" = "push") => {
      const href = coastHref({
        place: next.place !== undefined ? next.place : placeKey,
        home: next.home !== undefined ? next.home : homeParam,
        for: next.for ?? audience,
        species: next.species ?? species.id,
      });
      (how === "push" ? router.push : router.replace)(href, { scroll: false });
    },
    [router, placeKey, homeParam, audience, species.id],
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
  const cond = useConditions(place?.lat ?? null, place?.lon ?? null);

  // ------------------------------------------------------------ HAB evidence (the briefing) and what the map shows of it
  const outlook = useOutlook(manifest, baseUrl, place?.lat ?? null, place?.lon ?? null);
  const satNow = useSatellite(manifest, baseUrl, place?.lat ?? null, place?.lon ?? null);
  const hab = useHabEvidence(stations, place?.lat ?? null, place?.lon ?? null, now);
  const portEntry = useMemo(() => (place?.kind === "port" ? (portIntel?.ports.find((x) => `port-${x.port_code}` === place.key) ?? null) : null), [place, portIntel]);
  const [focus, setFocus] = useState<Focus>(null);
  const [peek, setPeek] = useState<string | null>(null);
  useEffect(() => {
    setFocus(null);
    setPeek(null);
  }, [place?.key]);
  const around = useMemo(() => (place && now ? stationsAround(stations, place.lat, place.lon, now) : []), [stations, place, now]);
  const outDay = outlook && outlook !== "error" ? (outlook.find((d) => d.lead === leadNow) ?? outlook[0] ?? null) : null;
  const [quantity, setQuantity] = useState<"pda" | "pn">("pda");
  const modelLayer = charmLayer(manifest, quantity === "pda" ? "particulate_domoic" : "pseudo_nitzschia", leadNow);
  const forecastRaster: ImageRaster | null = useMemo(
    () =>
      focus === "model" && modelLayer?.image
        ? {
            id: `coast-${modelLayer.layer_id}`,
            url: artifactUrl(baseUrl, modelLayer.image.url),
            corners: modelLayer.image.corners_lnglat,
            grid: supportsGridTiles(modelLayer) ? { layer: modelLayer, url: artifactUrl(baseUrl, modelLayer.grid!.url) } : null,
          }
        : null,
    [focus, modelLayer, baseUrl],
  );
  const satLayer = satelliteLatest(manifest, "olci300") ?? satelliteLatest(manifest, "viirs750");
  const satelliteRaster: TileRaster | null = useMemo(() => {
    const t = satLayer?.tiles;
    if (focus !== "satellite" || !satLayer || !t) return null;
    return { id: `coast-${satLayer.layer_id}`, template: t.relative ? artifactUrl(baseUrl, t.url_template) : t.url_template, minzoom: t.min_zoom ?? 0, maxzoom: t.max_native_zoom, bounds: t.bounds_lnglat, indexUrl: tileIndexUrl(baseUrl, t) };
  }, [focus, satLayer, baseUrl]);
  const keyGradient = focus === "model" && modelLayer?.palette ? gradientCss(modelLayer.palette) : focus === "satellite" && satLayer?.palette ? gradientCss(satLayer.palette) : null;
  const focusOn = useCallback(
    (f: Focus) => {
      setFocus(f);
      setPeek(null);
      // on a phone the map is above: bring it into view so the reader sees what changed
      if (f && !window.matchMedia("(min-width: 1024px)").matches) {
        const el = document.querySelector('[data-testid="coast-map-mobile"]');
        const r = el?.getBoundingClientRect();
        if (el && r && (r.bottom < 80 || r.top > window.innerHeight - 120)) el.scrollIntoView({ behavior: reducedMotionNow() ? "auto" : "smooth", block: "center" });
      }
    },
    [],
  );

  // ------------------------------------------------------------ the map shows where each fact came from
  const overlay = useMemo<GeoJSON.FeatureCollection | null>(() => {
    if (!place) return null;
    const g = model?.layer?.grid;
    const used = model?.pda?.here ?? model?.pda?.nearest ?? null;
    // the stations the briefing names (Pseudo-nitzschia, toxin); one of them in focus
    const named = [hab?.cells, hab?.toxin].filter((e): e is NonNullable<typeof e> => !!e && e.km <= 120);
    const ids = new Set(named.map((e) => e.station.station_id));
    const focused = focus === "cells" ? hab?.cells?.station.station_id : focus === "toxin" ? hab?.toxin?.station.station_id : peek;
    const threadTo = focus === "cells" || focus === "toxin" ? named.filter((e) => e.station.station_id === focused) : focus ? [] : named;
    const pix = focus === "satellite" && satNow && satNow !== "error" && satNow.value != null && satNow.km > 0 ? { lat: satNow.pixelLat, lon: satNow.pixelLon } : null;
    return provenanceFeatures({
      target: { lat: place.lat, lon: place.lon },
      cell: g && used && (focus === null || focus === "model") ? { polygon: cellPolygon(g, used.cell), lat: used.cell.lat, lon: used.cell.lon, own: !!model?.pda?.here } : null,
      stations: threadTo.map((e) => ({ lat: e.station.lat, lon: e.station.lon, name: e.station.name, km: e.km, id: e.station.station_id, aged: e.ageDays > RECENT_DAYS, focus: e.station.station_id === focused })),
      others: around
        .filter((a) => !threadTo.some((e) => e.station.station_id === a.station.station_id))
        .map((a) => ({ lat: a.station.lat, lon: a.station.lon, name: a.station.name, id: a.station.station_id, aged: a.ageDays > RECENT_DAYS, minor: !ids.has(a.station.station_id) && a.station.station_id !== peek })),
      pixel: pix,
      ring: focus === "model" ? ringPolygon(place.lat, place.lon, NEIGHBOURHOOD_KM) : null,
      official: focus === null || focus === "official" ? ((official?.geometry as unknown as GeoJSON.FeatureCollection) ?? null) : null,
    });
  }, [place, model, official, hab, focus, peek, around, satNow]);

  const mapRef = useRef<MlMap | null>(null);
  const [headH, setHeadH] = useState<number | null>(null);
  useEffect(() => {
    const h = document.querySelector("header");
    if (!h) return;
    const on = () => setHeadH(h.getBoundingClientRect().height);
    on();
    const ro = new ResizeObserver(on);
    ro.observe(h);
    return () => ro.disconnect();
  }, []);
  const [reducedMotion, setReducedMotion] = useState(false);
  useEffect(() => setReducedMotion(window.matchMedia("(prefers-reduced-motion: reduce)").matches), []);
  const fit = useCallback(
    (m: MlMap) => {
      if (!place) return m.fitBounds(CA, { padding: 24, duration: 0 });
      const pts: [number, number][] = [[place.lon, place.lat]];
      const used = model?.pda?.here ?? model?.pda?.nearest;
      const ev = focus === "cells" ? hab?.cells : focus === "toxin" ? hab?.toxin : null;
      if (focus === "model") {
        // the whole neighbourhood the briefing summarises
        const r = ringPolygon(place.lat, place.lon, NEIGHBOURHOOD_KM).coordinates[0];
        for (const q of r) pts.push(q as [number, number]);
      } else if (ev) {
        pts.push([ev.station.lon, ev.station.lat]);
      } else if (focus === "satellite") {
        if (satNow && satNow !== "error" && satNow.value != null) pts.push([satNow.pixelLon, satNow.pixelLat]);
        pts.push([place.lon - 0.12, place.lat - 0.08], [place.lon + 0.12, place.lat + 0.08]);
      } else {
        if (used) pts.push([used.cell.lon, used.cell.lat]);
        for (const e of [hab?.cells, hab?.toxin]) if (e && e.km < 25) pts.push([e.station.lon, e.station.lat]);
      }
      const lons = pts.map((p) => p[0]);
      const lats = pts.map((p) => p[1]);
      const pad = focus === "model" ? 0.01 : 0.06;
      journeyTo(m, { bounds: [[Math.min(...lons) - pad, Math.min(...lats) - pad], [Math.max(...lons) + pad, Math.max(...lats) + pad]], padding: 48, maxZoom: focus === "satellite" ? 10.5 : 11 });
    },
    [place, model, hab, focus, satNow],
  );
  useEffect(() => {
    if (mapRef.current) fit(mapRef.current);
  }, [fit]);

  // ------------------------------------------------------------ what is not known
  const unknowns = useMemo(() => {
    if (!place) return [];
    const out: string[] = [];
    const tied = assocs.filter((a) => a.tier === "geographic" && !a.dispute && a.record.area.type !== "statewide");
    for (const a of assocs.filter((x) => x.dispute)) out.push(`${a.dispute!.agency} release ${a.dispute!.release} appears to lift “${a.record.title}”; no person has reviewed it yet.`);
    if (!tied.length) out.push("No notice in CoastWatch's list is tied to these waters by its geography. The list may be incomplete; check the agencies.");
    out.push("Whether each notice applies here was worked out from the notices' official geography; no person has reviewed these associations, and the notices themselves are not verified.");
    const nUnresolved = assocs.filter((a) => a.tier === "unresolved" && !a.dispute).length;
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
      forecast={forecastRaster}
      satellite={satelliteRaster}
      fade
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
      onStation={(id) => {
        // a station the briefing names takes its card into focus; any other opens its latest measurements
        if (hab?.cells?.station.station_id === id && focus !== "cells") return focusOn("cells");
        if (hab?.toxin?.station.station_id === id && focus !== "toxin") return focusOn("toxin");
        setPeek(id);
      }}
      onReady={(m) => {
        mapRef.current = m;
        // on a small map the credits start folded (the (i) button opens them), as on Bloom's map
        if (!window.matchMedia("(min-width: 1024px)").matches) m.getContainer().querySelector(".maplibregl-ctrl-attrib")?.classList.remove("maplibregl-compact-show");
        fit(m);
      }}
    />
  );

  const peekStation = peek && now ? (around.find((a) => a.station.station_id === peek) ?? null) : null;
  const mapExtras = place && (
    <>
      <MapKey focus={focus} gradient={keyGradient} />
      <AnimatePresence>{peekStation && now && <StationPeek key={peekStation.station.station_id} station={peekStation.station} km={peekStation.km} now={now} onClose={() => setPeek(null)} />}</AnimatePresence>
    </>
  );
  const [picking, setPicking] = useState(false);
  useEffect(() => setPicking(false), [place?.key]);
  const actionLink = "min-h-[32px] font-medium text-accent hover:underline";
  // a function: the answer line below it is computed further down
  const header = () => (
    <header className="space-y-3 sm:space-y-4" data-testid="coast-header">
      <div>
        <p className="text-[13px] font-medium text-accent">{COAST_COPY.title}</p>
        <div className="mt-1 flex items-start justify-between gap-3">
          <h1 className="text-balance font-display text-[30px] font-medium leading-[1.08] text-ink sm:text-[40px]" data-testid="coast-title">
            {place ? place.name : "Choose the waters you care about"}
          </h1>
          {place && <CopyLink />}
        </div>
        {place ? (
          <p className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[15px] text-ink-2" data-testid="coast-subtitle">
            <span>
              {place.note}
              {place.county ? ` · ${place.county} County` : ""}
            </span>
            <button
              type="button"
              aria-expanded={picking}
              onClick={() => {
                setPicking(!picking);
                // focus moves into the search it opens
                if (!picking) requestAnimationFrame(() => document.querySelector<HTMLInputElement>('[data-testid="place-input"]')?.focus());
              }}
              className={actionLink}
              data-testid="coast-change"
            >
              {picking ? "Close" : "Change place"}
            </button>
          </p>
        ) : (
          <p className="mt-2 max-w-[60ch] text-[17px] leading-relaxed text-ink-2">{COAST_COPY.lede}</p>
        )}
        {!place && placeKey && (
          <p className="mt-2 rounded-lg border border-hairline border-l-[3px] border-l-official-line bg-surface px-3 py-2 text-[14px] text-official-ink" role="status" data-testid="coast-unknown-place">
            The place in this link (“{placeKey}”) is not one CoastWatch knows. Choose a port, a harvest site or a point on the map.
          </p>
        )}
      </div>
      {(!place || picking) && <PlacePicker places={allPlaces} current={place} onPick={pick} />}
      {place && answer && <AnswerLine clauses={answer} />}
    </header>
  );
  const homeLine = (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[14px]" data-testid="coast-home">
      {home ? (
        <>
          <span className="text-ink-2">
            Home port: <b className="font-semibold text-ink">{home.name}</b>
          </span>
          {place?.key !== home.key && (
            <button type="button" onClick={() => pick(home)} className={actionLink} data-testid="coast-home-open">
              Show home port
            </button>
          )}
          <button type="button" onClick={() => setHome(null)} className="min-h-[32px] text-ink-3 hover:text-ink" data-testid="coast-home-clear">
            Clear
          </button>
        </>
      ) : place?.kind === "port" ? (
        <button type="button" onClick={() => setHome(place.key)} className={actionLink} data-testid="coast-home-set">
          Make {place.name} my home port
        </button>
      ) : (
        <span className="text-ink-3">Save a home port to return to it; it is kept on this device only.</span>
      )}
    </div>
  );

  const lastToxin = st.near
    .flatMap((s) => s.station.latest.filter((m) => m.variable === "pDA" && m.value != null).map((m) => ({ s, m })))
    .sort((a, b) => b.m.date.localeCompare(a.m.date))[0];
  const toxin =
    lastToxin && now
      ? { station: lastToxin.s.station.name, km: lastToxin.s.km, date: lastToxin.m.date, ageDays: ageDays(lastToxin.m.date, now), value: `${Number(lastToxin.m.value!.toPrecision(2))} ${lastToxin.m.units}` }
      : null;
  const stale = model?.layer && now ? classifyTime(model.layer.freshness, model.layer.time, now).state !== "current" : false;
  const forecastLine = !model
    ? "Reading…"
    : model.error
      ? "No C-HARM run available"
      : model.pda?.here
        ? `${Math.round(model.pda.here.value * 100)}% probability in the waters' own 3 km cell`
        : "No toxin forecast for these waters";
  const forecastSub = model?.layer
    ? `NOAA C-HARM, valid ${model.layer.time.valid_date ? formatDate(model.layer.time.valid_date) : "—"}${stale ? " · stale run" : ""}${
        model.pda && !model.pda.here && model.pda.nearest ? ` · nearest value ${model.pda.nearest.km.toFixed(1)} km ${model.pda.nearest.bearing} (not a forecast here)` : ""
      }`
    : "NOAA C-HARM";
  const satelliteLine =
    sat === null ? "Reading…" : sat.date && now ? `Clear view ${formatDate(sat.date)} (${ageWords(ageDays(sat.date, now))})` : "No clear view this week (cloud, fog or no overpass)";
  const official1 = officialAnswer(assocs, audience, species);
  const status = speciesStatements(official, species, audience);
  const seasonStatus = status.find((x) => /season status/i.test(x.topic));
  const answer: AnswerClause[] | null =
    place && modeRead
      ? [
          ...(seasonStatus ? [{ kind: "official" as const, testid: "answer-status", text: `${seasonStatus.agency}: “${seasonStatus.statement}”` }] : []),
          {
            kind: "official",
            testid: "answer-official",
            text:
              official1.headline + (official1.disputed.length ? ` · ${official1.disputed.length} more reported lifted, awaiting review` : ""),
          },
          measuredClause(hab, stationsError),
          modelClause(outlook, outDay?.valid ?? null, outDay?.pda ?? null),
        ]
      : null;
  const samples = useMemo<NearSample[]>(
    () =>
      [hab?.cells, hab?.toxin]
        .filter((e): e is NonNullable<typeof e> => !!e && e.near)
        .sort((a, b) => a.ageDays - b.ageDays)
        .map((e) => ({
          kind: e.kind === "toxin" ? "toxin" : "cells",
          station: e.station.name,
          stationId: e.station.station_id,
          km: e.km,
          date: e.date,
          ageDays: e.ageDays,
          text: e.kind === "cells" ? `Pseudo-nitzschia ${fmtMeasured("pn_seriata", e.values.reduce((a, v) => a + v.value, 0))}` : `domoic acid ${fmtMeasured("pDA", e.values[0].value)}`,
        })),
    [hab],
  );
  const steps = useMemo(
    () =>
      place
        ? nextSteps({ official, verification, assocs, audience, species, lat: place.lat, conditions: cond.data, lastToxin: toxin, fmt: fmtWhen, samples })
        : [],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [place, official, verification, assocs, audience, species, cond.data, toxin?.date, toxin?.station, samples],
  );
  const detailsRef = useRef<HTMLDetailsElement | null>(null);
  const openDetails = () => {
    const d = detailsRef.current;
    if (!d) return;
    d.open = true;
    d.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
  };

  const sections = (order: ReturnType<typeof sectionOrder>) =>
    place &&
    order.map((k) =>
      k === "official" ? (
        <OfficialSection key={k} official={official} assocs={assocs} audience={audience} place={place} verification={verification} />
      ) : k === "measured" ? (
        <MeasuredSection key={k} near={st.near} nearestBeyond={st.nearestBeyond} now={now} />
      ) : k === "model" ? (
        <ModelSection key={k} manifest={manifest} model={model} lead={leadNow} onLead={setLead} now={now} />
      ) : (
        <ObservedSection key={k} manifest={manifest} baseUrl={baseUrl} place={place} onSat={setSat} />
      ),
    );

  const landing = !place && (
    <section aria-labelledby="coast-start-h" className="space-y-3" data-testid="coast-landing">
      <h2 id="coast-start-h" className="text-[13.5px] font-semibold text-ink-3">
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
        {header()}
        {place && modeRead && viewMode === "coastal" && (
          // Coastal: your waters and species, then the HAB evidence at once (no chapters to open)
          <WatersAndSpecies
            audience={audience}
            onAudience={(a) => go({ for: a }, "replace")}
            species={species}
            onSpecies={(x) => go({ species: x }, "replace")}
            leads={leads}
            lead={leadNow}
            onLead={setLead}
            today={today}
          />
        )}
        {place && modeRead && viewMode === "research" && (
          <>
            <PlanBar audience={audience} onAudience={(a) => go({ for: a }, "replace")} species={species} onSpecies={(x) => go({ species: x }, "replace")} />
            <OfficialBrief
              assocs={assocs}
              audience={audience}
              species={species}
              verification={verification}
              totalActive={currentRecords(official).length}
              onAll={(el) => openDrawer(el)}
              onDetails={openDetails}
                statements={status}
            />
          </>
        )}
        {landing}
        {place && modeRead && (
          <HabBriefing
            manifest={manifest}
            placeName={place.name}
            stations={stations}
            outlook={outlook}
            sat={satNow}
            port={portEntry}
            lead={leadNow}
            focus={focus}
            onFocus={focusOn}
            pick={quantity}
            onPick={setQuantity}
            now={now}
            lat={place.lat}
            lon={place.lon}
            between={
          <div className="h-[42vh] min-h-[260px] overflow-hidden rounded-2xl lg:hidden" data-testid="coast-map-mobile">
            {/* one map: phones get it inline, below the summary it illustrates; wide screens in the sticky column */}
            <MapSlot wide={false}>
              {map}
              {mapExtras}
            </MapSlot>
          </div>
            }
          />
        )}
        {place && modeRead && viewMode === "coastal" && (
          <>
            <section aria-label="Official information and ocean conditions" className="space-y-5" data-testid="coast-official-conditions">
              <OfficialBrief
                assocs={assocs}
                audience={audience}
                species={species}
                verification={verification}
                totalActive={currentRecords(official).length}
                onAll={(el) => openDrawer(el)}
                onDetails={openDetails}
                statements={status}
              />
              <ConditionsCard conditions={cond.data} loading={cond.loading} error={cond.error} now={now} habTiles={false} day={leadNow > 0 ? (leads[leadNow]?.time.valid_date ?? null) : null} tidesFirst={audience === "shellfish"} />
            </section>
            <NextSteps
              steps={steps}
              conditionsShown
              sub={
                species.id === "any"
                  ? `A general overview for ${place.name}. Choose a species above for its notices, rules and history.`
                  : `For ${{ all: "", commercial: "commercial ", recreational: "recreational ", shellfish: "" }[audience]}${species.label.toLowerCase().replace(/^dungeness/, "Dungeness").replace(/^california/, "California")} at ${place.name}.`
              }
            />
          </>
        )}
        {place && modeRead && (
          <>
            {viewMode === "research" && (
              <ConditionsCard
                conditions={cond.data}
                loading={cond.loading}
                error={cond.error}
                toxin={toxin}
                toxinError={stationsError}
                forecastLine={forecastLine}
                forecastSub={forecastSub}
                satelliteLine={satelliteLine}
                now={now}
                leads={leads}
                lead={leadNow}
                onLead={setLead}
              />
            )}
            {viewMode === "coastal" ? (
              <details ref={detailsRef} className="group rounded-[18px] border border-hairline bg-surface" data-testid="coast-details">
                <summary className="flex min-h-[52px] cursor-pointer items-center justify-between gap-3 px-4 text-[15px] font-medium text-ink sm:px-5">
                  <span>
                    Scientific details and every notice
                    <span className="block text-[13px] font-normal text-ink-3">Measurements, forecast cells, satellite, currents, what is not known, sources</span>
                  </span>
                  <span aria-hidden className="text-ink-3 transition-transform group-open:rotate-90">›</span>
                </summary>
                <div className="space-y-8 border-t border-hairline px-4 pb-6 pt-5 sm:px-5">
                  {sections(sectionOrder(null))}
                  <UnknownsSection items={unknowns} />
                </div>
              </details>
            ) : (
              <>
                {/* research: the full sections first, in the researcher's order, open */}
                {sections(sectionOrder({ purpose: "research" }))}
                <NextSteps steps={steps} />
              </>
            )}
            {viewMode === "research" && <UnknownsSection items={unknowns} />}
            {homeLine}
            <div className="flex flex-wrap items-center gap-3">
              <ModeSwitch tone="paper" />
              <PurposeSwitch compact />
            </div>
            <p className="border-t border-hairline pt-4 text-[13px] leading-relaxed text-ink-3">
              Sources: CDFW and CDPH notices (transcribed, not verified); CalHABMAP via SCCOOS ERDDAP; NOAA C-HARM v3.1; NOAA CoastWatch satellite chlorophyll; HF-radar
              currents. {pathname === "/coast" ? "This page's link reproduces this view." : ""}
            </p>
          </>
        )}
      </div>
      <div className="hidden lg:block">
        {/* below the whole sticky header (masthead and any status banner), so nothing on the map hides under it */}
        <div className="sticky top-[var(--cw-masthead-h)] h-[calc(100dvh-var(--cw-masthead-h))]" style={headH ? { top: headH, height: `calc(100dvh - ${headH}px)` } : undefined} data-testid="coast-map-desktop">
          <MapSlot wide>
            {map}
            {mapExtras}
          </MapSlot>
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


type Hab = ReturnType<typeof useHabEvidence>;

/** The answer line's measured clause: the latest shore sample near these waters, with its age. */
function measuredClause(hab: Hab, error: string | null): AnswerClause {
  if (error) return { kind: "gap", testid: "answer-measured", text: "Shore measurements could not be loaded" };
  if (!hab) return { kind: "gap", testid: "answer-measured", text: "Reading the shore measurements" };
  const near = [hab.cells, hab.toxin].filter((e): e is NonNullable<typeof e> => !!e && e.near).sort((a, b) => a.ageDays - b.ageDays)[0];
  if (!near) return { kind: "gap", testid: "answer-measured", text: "No shore station within 60 km measures bloom algae or toxin" };
  const km = near.km < 10 ? near.km.toFixed(1) : String(Math.round(near.km));
  const what = near.kind === "cells" ? `Pseudo-nitzschia ${fmtMeasured("pn_seriata", near.values.reduce((a, v) => a + v.value, 0))}` : `Domoic acid ${fmtMeasured("pDA", near.values[0].value)}`;
  return { kind: "measured", testid: "answer-measured", aged: near.ageDays > MEASURED_RECENT_DAYS, text: `${what} at ${near.station.name} (${km} km), ${ageWords(near.ageDays)}` };
}

/** The answer line's model clause: the waters' own cell, or the real offshore cells near them (never as the waters' own). */
function modelClause(outlook: ReturnType<typeof useOutlook>, valid: string | null, nb: import("@/lib/hab").Neighbourhood | null): AnswerClause {
  if (outlook === null) return { kind: "gap", testid: "answer-model", text: "Reading the toxin model" };
  if (outlook === "error" || !outlook.length) return { kind: "gap", testid: "answer-model", text: "No toxin model run is available" };
  const day = valid ? `, ${formatDate(valid)}` : "";
  if (nb?.here != null) return { kind: "model", testid: "answer-model", text: `Model: ${fmtPct(nb.here)} chance toxin in the water exceeds its threshold (this 3 km cell${day})` };
  if (nb && nb.n > 0) return { kind: "model", testid: "answer-model", text: `Model: no value for these nearshore waters; ${nb.n} offshore cells within ${NEIGHBOURHOOD_KM} km, median ${fmtPct(nb.median)}${day}` };
  return { kind: "gap", testid: "answer-model", text: `Outside the toxin model: no cell within ${NEIGHBOURHOOD_KM} km${nb?.nearest ? ` (nearest ${Math.round(nb.nearest.km)} km away)` : ""}` };
}
