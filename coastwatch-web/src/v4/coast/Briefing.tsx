"use client";

import { toast } from "@/v4/ui/toast";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { StationLite } from "@/lib/data";
import { AUDIENCES, coastHref, parseAudience, resolvePlace, sitePlaces, type Audience, type CoastPlace } from "@/lib/coast";
import { SPECIES, speciesById, speciesPhrase, type SpeciesId } from "@/content/species";
import { defaultAudience, usePurpose } from "@/lib/purpose";
import { useNow } from "@/lib/useNow";
import { charmLayer } from "@/lib/layers";
import type { EvidenceNode } from "@/lib/harborEvidence";
import type { FieldKey } from "@/lib/tripEvaluation";
import { encodeSelection, type Selection } from "@/v4/state/selection";
import { useReportSource } from "@/v4/readiness/store";
import { modelEntry, satelliteEntry } from "@/v4/readiness/shown";
import { TopBar, NavTabs } from "@/v4/shell/TopBar";
import { NoticesButton, type LocalNotices } from "@/v4/shell/Notices";
import { RegistryDisclosure } from "@/v4/shell/Disclosure";
import { checklistKey } from "@/lib/trip";
import { SkipLink } from "@/v4/shell/SkipLink";
import type { Step } from "@/lib/nextSteps";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel, dayYearLabel, kmLabel, relTime } from "@/v4/ui/time";
import { nodeSelection } from "@/v4/explore/Inspect";
import { Matters } from "./Matters";
import { TripGlance, TripWorksheet, hasSharedScenario } from "@/v4/coast/economics";
import { SinceLastCheck } from "@/components/coast/SinceLastCheck";
import { useBriefing } from "./useBriefing";
import { PACKS, packForPort } from "@/v4/offline/packs";
import { isFixture, sourceStatus } from "@/lib/layers";
import { useOfflinePack } from "@/v4/offline/useOfflinePack";
import { useRecordSource } from "@/v4/offline/OfflinePanel";
import { OfflineBanner } from "@/v4/offline/OfflineBanner";
import { PackDialog } from "@/v4/offline/PackDialog";
import { savedSearch, savedSnapshot } from "@/v4/offline/snapshot";
import { homeHref, readHome, writeHome, type HomePort } from "./home";
import { Fold } from "./Fold";
import { Landing } from "./Landing";
import { OfficialContext, SeasonLinks } from "./Official";
import { DecisionRow } from "./Decision";
import { AGENCY_LINKS } from "@/content/coast";
import { econAudience } from "./economics/model";
import { TimeView } from "./Lanes";
import { HarbourMap, chlorophyll } from "./HarbourMap";
import { seasonStatus } from "./season";
import { Caps } from "./ui/Instrument";
import { SeaBody, SeasonsBody, WorthBody, useSeaLimits, worthLine } from "@/components/coast/FishingBrief";

type Props = {
  manifest: Manifest;
  ports: PortsCollection | null;
  official: OfficialDataset | null;
  stations: StationLite[] | null;
  stationsError: string | null;
  baseUrl: string;
  portIntel: PortIntelCollection | null;
};

function exploreHref(s: Selection, back: string): string {
  const q = new URLSearchParams({ sel: encodeSelection(s)!, from: back });
  return `/?${q.toString()}`;
}

const PT_STAMP = new Intl.DateTimeFormat("en-US", { timeZone: "America/Los_Angeles", weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });

/**
 * A trip choice in the header: a caps label over a native select drawn as a ruled line. `disabled` until
 * hydrated: a choice made before the page is live would be dropped silently on a slow phone.
 */
function Select({ id, label, value, onChange, options, disabled = false }: { id: string; label: string; value: string; onChange: (v: string) => void; options: { id: string; label: string }[]; disabled?: boolean }) {
  return (
    <label htmlFor={id} className="flex min-w-0 flex-col gap-0.5 sm:flex-none">
      <span className="nx-caps">{label}</span>
      <select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)} className="nx-select nx-focus min-h-11 w-full rounded-none border-0 border-b border-n-line-strong bg-transparent pl-0 text-[16px] font-medium text-n-ink hover:border-n-ink-2 sm:w-auto sm:max-w-[15rem]" data-testid={id}>
        {options.map((o) => (
          <option key={o.id} value={o.id}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Briefing(props: Props) {
  const { ports, official, stations } = props;
  const router = useRouter();
  const liveSp = useSearchParams();
  // offline, a saved briefing answers other trips and places: hydrate the HTML it holds, then render the
  // address bar's own (otherwise React throws #418 and the first render is thrown away)
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const sp = (!mounted && savedSearch("/coast")) || liveSp;
  const now = useNow();
  const [purpose] = usePurpose();
  const placeKey = sp.get("place");
  const audience: Audience = sp.get("for") ? parseAudience(sp.get("for")) : defaultAudience(purpose);
  const species = speciesById(sp.get("species"));
  const place = useMemo(() => resolvePlace(placeKey, ports), [placeKey, ports]);
  const b = useBriefing({ ...props, place, audience, species, dateParam: sp.get("date"), now });
  const [home, setHomeState] = useState<HomePort | null>(null);
  useEffect(() => setHomeState(readHome()), []);
  // a bare /coast with a home port opens the home briefing (the reader's own trip), after mount only
  const liveHasPlace = liveSp.has("place");
  useEffect(() => {
    if (liveHasPlace) return;
    const h = readHome();
    if (h && resolvePlace(h.place, ports)) router.replace(homeHref(h));
  }, [liveHasPlace, ports, router]);
  const setHome = useCallback((h: HomePort | null) => {
    writeHome(h);
    setHomeState(h);
  }, []);
  const go = useCallback(
    (next: { place?: string | null; for?: Audience; species?: SpeciesId; date?: string | null }) => {
      router.replace(coastHref({ place: next.place !== undefined ? next.place : placeKey, for: next.for ?? audience, species: next.species ?? species.id, date: next.date !== undefined ? next.date : sp.get("date") }), { scroll: false });
    },
    [router, placeKey, audience, species.id, sp],
  );

  // the worksheet is a view of this page (?sheet=trip), opened and closed in the history without a server
  // round trip (so it opens offline from the saved briefing too)
  const sheet = mounted && sp.get("sheet") === "trip";
  const focusParam = (sp.get("focus") as FieldKey | null) ?? null;
  const openSheet = useCallback((focus: FieldKey | null = null) => {
    const q = new URLSearchParams(window.location.search);
    q.set("sheet", "trip");
    if (focus) q.set("focus", focus);
    else q.delete("focus");
    window.history.pushState(null, "", `${window.location.pathname}?${q.toString()}${window.location.hash}`);
    window.scrollTo({ top: 0 });
  }, []);
  const closeSheet = useCallback(() => {
    const q = new URLSearchParams(window.location.search);
    q.delete("sheet");
    q.delete("focus");
    window.history.pushState(null, "", `${window.location.pathname}?${q.toString()}`);
    window.scrollTo({ top: 0 });
  }, []);
  // a shared scenario link ("#trip=…") opens the worksheet that holds it
  useEffect(() => {
    if (mounted && place && hasSharedScenario() && liveSp.get("sheet") !== "trip") {
      const q = new URLSearchParams(window.location.search);
      q.set("sheet", "trip");
      window.history.replaceState(null, "", `${window.location.pathname}?${q.toString()}${window.location.hash}`);
    }
  }, [mounted, place, liveSp]);

  const [tripOpen, setTripOpen] = useState(false);
  const [limits, setLimits] = useSeaLimits();
  const [placeOpen, setPlaceOpen] = useState(false);
  const covering = place?.kind === "port" ? packForPort(Number(place.key.slice(5))) : null;
  const pack = covering ?? PACKS.monterey;
  const off = useOfflinePack(pack.id);
  useRecordSource(off.record, pack.name, off.online, off.evicted, off.problem);
  const [packOpen, setPackOpen] = useState(false);
  const hasPack = !!covering;
  const snap = covering ? savedSnapshot(off.record, off.online) : null;
  const basemap = off.record === undefined ? undefined : typeof navigator !== "undefined" && !navigator.onLine && off.record?.complete ? `/offline/${pack.id}/basemap.pmtiles` : null;
  const back = place ? coastHref({ place: place.key, for: audience, species: species.id, date: sp.get("date") }) : "/coast";
  const placeSel: Selection | null = !place ? null : place.kind === "port" ? { kind: "harbor", port: Number(place.key.slice(5)) } : { kind: "point", lat: place.lat, lon: place.lon };

  useReportSource({ id: "data", label: "Neritic dataset", state: "ready", detail: `Published ${props.manifest.generated_at.slice(0, 16).replace("T", " ")} UTC`, asOf: props.manifest.generated_at });
  // the published sources this briefing shows, with their freshness (the same rule as Explore), so the
  // top bar never says "Ready" over a days-old model run or satellite view
  const mapLayer = useMemo(() => chlorophyll(props.manifest, props.baseUrl)?.layer ?? null, [props.manifest, props.baseUrl]);
  useReportSource(place ? modelEntry(props.manifest, now, "In this briefing's model lane and evidence") : null);
  useReportSource(place && mapLayer ? satelliteEntry(props.manifest, now, "On this briefing's map", mapLayer) : null);
  useReportSource(place ? { id: "conditions", label: "Harbour conditions", state: b.cond.loading ? "loading" : b.cond.error ? "unavailable" : b.cond.data ? "ready" : "waiting", detail: b.cond.error ?? "NOAA CO-OPS, NDBC and NWS, read live" } : null);

  // the reader's own ticks on "Before you go", per place, trip and day, on this device only
  const tickKey = place ? checklistKey(place.key, audience, species.id, sp.get("date")) : null;
  const [ticks, setTicks] = useState<{ k: string | null; ids: string[] }>({ k: null, ids: [] });
  useEffect(() => {
    if (!tickKey) return;
    try {
      setTicks({ k: tickKey, ids: JSON.parse(window.localStorage.getItem(tickKey) ?? "[]") as string[] });
    } catch {
      setTicks({ k: tickKey, ids: [] });
    }
  }, [tickKey]);
  const ticked = useMemo(() => (ticks.k === tickKey ? ticks.ids : []), [ticks, tickKey]);
  const toggleTick = useCallback(
    (id: string) => {
      if (!tickKey) return;
      const ids = ticked.includes(id) ? ticked.filter((x) => x !== id) : [...ticked, id];
      setTicks({ k: tickKey, ids });
      try {
        window.localStorage.setItem(tickKey, JSON.stringify(ids));
      } catch {
        // not remembered
      }
    },
    [tickKey, ticked],
  );
  const localNotices = useMemo<LocalNotices | null>(() => (place ? { name: place.name, covering: b.notices.hereAssocs.map((a) => a.record.id), unsettled: b.notices.unsettled } : null), [place, b.notices]);
  const goPlace = useCallback((key: string) => router.push(coastHref({ place: key, for: audience, species: species.id })), [router, audience, species.id]);
  // the statements appear together once the model and the harbour conditions have answered (or 6 s
  // have passed), so the answer does not reshuffle as each source arrives
  const settledNow = b.outlook !== null && !b.cond.loading;
  const [waited, setWaited] = useState(false);
  useEffect(() => {
    setWaited(false);
    const t = setTimeout(() => setWaited(true), 6000);
    return () => clearTimeout(t);
  }, [placeKey]);
  const settled = settledNow || waited;
  useReportSource(place ? { id: "evidence", label: "Briefing evidence", state: b.graph && settled ? (b.outlook === "error" ? "partial" : "ready") : "loading", detail: b.graph ? `${b.matters.length} statements, each with its source` : "Reading notices, shore samples, NOAA's model and the satellite" } : null);
  const g = b.graph;
  // the sea itself (NWS text, tides, the buoy's wind, waves and water) is read in the decision row and the
  // lanes; the evidence section holds what bears on a bloom, so nothing is said twice
  const measured = useMemo(() => (g ? g.nodes.filter((n) => n.variable !== "official notice" && !SEA_VARIABLES.test(n.variable)) : []), [g]);
  const near = measured.filter((n) => (n.where?.representative === "at" || n.where?.representative === "near") && (!n.freshness || n.freshness.state === "current"));
  const said = new Set(b.matters.filter((c) => c.history).flatMap((c) => c.supports));
  const far = measured.filter((n) => !near.includes(n) && !(n.id?.startsWith("record:") && said.has(n.id)));
  const notices = g ? g.nodes.filter((n) => n.variable === "official notice") : [];
  const mapBox = useRef<HTMLDivElement | null>(null);
  const jumpToMap = useCallback(() => {
    setPlaceOpen(false);
    window.setTimeout(() => mapBox.current?.scrollIntoView({ block: "center", behavior: "auto" }), 60);
  }, []);

  // ---------------------------------------------------------------- no place yet: one question, one control
  if (!place) {
    return (
      <div className="env-chart min-h-dvh bg-n-bg text-n-ink" data-testid="my-coast" data-view="landing">
        <SkipLink />
        <TopBar active="coast" now={now} notices={<NoticesButton official={official} status={sourceStatus(props.manifest, "official")} />} />
        <RegistryDisclosure className="mx-auto w-full max-w-[1320px] px-4 sm:px-8" fixture={isFixture(props.manifest)} />
        <main id="main" className="cw-fade-in mx-auto w-full max-w-[1320px] overflow-x-clip px-4 pb-32 pt-4 sm:px-8">
          <Landing ports={ports} home={home} audience={audience} unknownKey={placeKey} />
        </main>
        <MobileDock />
      </div>
    );
  }

  // ---------------------------------------------------------------- the briefing
  const activityOptions = AUDIENCES.map((a) => ({ id: a.id, label: a.id === "all" ? "Any activity" : a.label }));
  const speciesOptions = SPECIES.filter((s) => s.id === "any" || audience === "all" || s.activities.includes(audience as never)).map((s) => ({ id: s.id, label: s.id === "any" ? "Any species" : s.label }));
  const dayOptions = b.days.map((d) => ({ id: d.date, label: `${d.label}, ${d.day}` }));
  const asSaved = snap ? `as saved ${clockLabel(snap.savedAt)}` : null;
  const counts = { here: b.notices.here, namingSpecies: b.notices.namingSpecies };
  const unclear = new Set([...b.notices.unsettled, ...b.assocs.filter((a) => a.dispute).map((a) => a.record.id)]).size;
  const portCode = place.kind === "port" ? Number(place.key.slice(5)) : null;
  const otherHere = b.notices.hereAssocs.filter((a) => !b.notices.namingIds.has(a.record.id));
  const activityLabel = audience === "all" ? "Any activity" : AUDIENCES.find((x) => x.id === audience)?.label;
  const dayWords = b.tripDay ? (b.tripDay.label === "Today" ? "Today" : `${b.tripDay.label}, ${b.tripDay.day}`) : "Today";
  const checksN = b.checks.shown.length + b.checks.more.length;
  const tickedN = [...b.checks.shown, ...b.checks.more].filter((x) => ticked.includes(x.id)).length;
  const season = seasonStatus(place, species, audience, b.tripDate ?? b.today);
  const runLayer = charmLayer(props.manifest, "particulate_domoic", 0) ?? charmLayer(props.manifest, "pseudo_nitzschia", 0);
  const runIssued = runLayer?.time.issued_date ? dayLabel(runLayer.time.issued_date) : null;
  const stamp = asSaved ? `Briefing ${asSaved}` : now ? `Briefing · ${PT_STAMP.format(now).replace(/,(?= \d{1,2}:)/, " ·")}` : "Briefing";
  const exploreLink = exploreHref(placeSel!, back);
  const researchLink = `/research?${new URLSearchParams({ place: place.key, from: back }).toString()}`;
  const econProps = { place: { key: place.key, name: place.name, portCode }, species: species.id, audience, tripDate: b.tripDate, season };

  const header = (
    <header className="flex flex-col gap-2 pb-5 lg:pb-6" data-testid="coast-header">
      <Caps className="text-n-ink-2" as="p">
        <span data-testid="coast-stamp">{sheet ? `${place.name} · ${activityLabel} · ${species.id === "any" ? "any species" : species.label} · ${dayWords}` : stamp}</span>
      </Caps>
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-end gap-x-4 gap-y-2">
        <div className="min-w-0">
          {sheet ? (
            <h1 className="font-serif font-medium tracking-[-0.015em] text-[34px] leading-[1.04] text-n-ink sm:text-[56px]" data-testid="econ-title">
              Trip worksheet
            </h1>
          ) : (
            <>
              <h1 className="font-serif font-medium tracking-[-0.015em] text-[38px] leading-[1.04] text-n-ink sm:text-[56px] xl:text-[64px]" data-testid="coast-title">
                <button type="button" aria-expanded={placeOpen} aria-controls={placeOpen ? "coast-place-menu" : undefined} aria-describedby="coast-place-hint" onClick={() => setPlaceOpen((o) => !o)} className="nx-focus group max-w-full rounded-[3px] text-left hover:text-n-ink-2" data-testid="coast-place-button">
                  {place.name}
                  <Icon name="chevronDown" className={`ml-2 inline-block h-6 w-6 align-middle text-n-ink-3 transition-transform group-hover:text-n-ink ${placeOpen ? "rotate-180" : ""}`} />
                </button>
              </h1>
              <span id="coast-place-hint" hidden>
                Change the place, choose waters on the map, or keep it as your home port
              </span>
            </>
          )}
        </div>
        <div className="col-start-2 row-start-1 self-start">
          {sheet ? (
            <button type="button" onClick={closeSheet} className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 rounded-[3px] border border-n-line-strong px-3 text-[14px] text-n-ink" data-testid="econ-back">
              <Icon name="back" className="h-4 w-4" /> Briefing
            </button>
          ) : (
            <MoreMenu />
          )}
        </div>
        {sheet ? (
          <p className="col-span-2 max-w-[100ch] text-[14.5px] leading-snug text-n-ink" data-testid="econ-kicker-head">
            <SheetLine place={place} species={species} season={season} rec={econAudienceOf(audience)} />
          </p>
        ) : (
          <div className="col-span-2 flex flex-col gap-1 sm:flex-row sm:flex-wrap sm:items-end sm:justify-between sm:gap-x-6">
            <button type="button" aria-expanded={tripOpen} aria-controls="trip-controls" onClick={() => setTripOpen((o) => !o)} className="nx-focus -mx-1 flex min-h-12 items-center justify-between gap-3 rounded-[3px] px-1 text-left sm:hidden" data-testid="trip-summary">
              <span className="min-w-0 text-[15.5px] font-medium leading-snug text-n-ink">
                {activityLabel} · {species.id === "any" ? "Any species" : species.label} · {dayWords}
              </span>
              <span className="shrink-0 text-[14.5px] text-n-ink underline decoration-n-line-strong underline-offset-4" data-testid="trip-change">
                {tripOpen ? "Done" : "Change"}
              </span>
            </button>
            <div id="trip-controls" className={`flex-col gap-3 sm:flex sm:flex-row sm:flex-wrap sm:items-end sm:gap-5 ${tripOpen ? "flex" : "hidden"}`} data-testid="trip-controls">
              <Select id="trip-activity" disabled={!mounted} label="For" value={audience} onChange={(v) => go({ for: v as Audience, species: v === "all" || species.activities.includes(v as never) ? species.id : "any" })} options={activityOptions} />
              <Select id="trip-species" disabled={!mounted} label="Species" value={species.id} onChange={(v) => go({ species: v as SpeciesId })} options={speciesOptions} />
              <Select id="trip-day" disabled={!mounted} label="Day" value={b.tripDate ?? dayOptions[0]?.id ?? ""} onChange={(v) => go({ date: v === b.days[0]?.date ? null : v })} options={dayOptions.length ? dayOptions : [{ id: "", label: "Today" }]} />
            </div>
            <p className="flex flex-wrap items-center gap-x-3 text-[13.5px] text-n-ink-2 max-sm:hidden" data-testid="coast-subtitle">
              <span>
                {place.note}
                {place.county ? ` · ${place.county} County` : ""}
              </span>
              {home?.place === place.key ? (
                <span className="inline-flex items-center gap-1 text-n-ink" data-testid="coast-home">
                  <Icon name="harbor" className="h-4 w-4" /> Your home port
                </span>
              ) : home ? (
                <HomeLink home={home} ports={ports} />
              ) : null}
            </p>
          </div>
        )}
        {placeOpen && !sheet && (
          <div className="col-span-2">
            <PlaceMenu
              place={place}
              ports={ports}
              home={home}
              audience={audience}
              species={species.id}
              onHome={setHome}
              onPlace={(k) => {
                setPlaceOpen(false);
                goPlace(k);
              }}
              onMap={jumpToMap}
              onClose={() => setPlaceOpen(false)}
            />
          </div>
        )}
      </div>
    </header>
  );

  // ---------------------------------------------------------------- the worksheet view
  if (sheet)
    return (
      <div className="env-chart relative min-h-dvh bg-n-bg text-n-ink" data-testid="my-coast" data-view="worksheet">
        <SkipLink />
        <TopBar active="coast" now={now} notices={<NoticesButton official={official} status={sourceStatus(props.manifest, "official")} local={localNotices} />} />
        <RegistryDisclosure className="mx-auto w-full max-w-[1320px] px-4 sm:px-8" fixture={isFixture(props.manifest)} />
        {!off.online && (
          <div className="mx-auto w-full max-w-[1320px] px-4 pt-3 sm:px-8">
            <OfflineBanner record={off.record ?? null} problem={off.problem} name={pack.name} inline outside={covering ? null : { place: place.name, href: coastHref({ place: pack.place }) }} />
          </div>
        )}
        <main id="main" className="cw-fade-in mx-auto w-full max-w-[1320px] overflow-x-clip px-4 pb-32 pt-5 sm:px-8 lg:pt-8">
          {header}
          <TripWorksheet {...econProps} focus={focusParam} />
        </main>
        <MobileDock />
      </div>
    );

  // ---------------------------------------------------------------- the briefing: a reading column on the trip's waters
  const official1 = (
    <OfficialContext
      place={place}
      species={species}
      audience={audience}
      season={season}
      details={
        <Fold inline className="data-[open=true]:basis-full" id="seasons" title="Season details" summary={species.id === "any" ? "CDFW's list for these waters" : `CDFW's list, ${speciesPhrase(species)} first`}>
          <SeasonLinks place={place} season={season} />
          <SeasonsBody place={place} species={species} audience={audience} tripDate={b.tripDate} today={b.today} />
        </Fold>
      }
    />
  );
  const noticeRows = notices.length + otherHere.length;
  return (
    <div className="env-chart relative min-h-dvh bg-n-bg text-n-ink" data-testid="my-coast" data-view="trip">
      <SkipLink />
      <TopBar floating active="coast" now={now} notices={<NoticesButton official={official} status={sourceStatus(props.manifest, "official")} local={localNotices} />} />
      {packOpen && <PackDialog pack={pack} manifest={props.manifest} baseUrl={props.baseUrl} official={official} onClose={() => setPackOpen(false)} />}
      <main id="main" className="cw-fade-in grid grid-cols-1 overflow-x-clip pb-28 lg:grid-cols-[minmax(0,1fr)_minmax(420px,42vw)] lg:grid-rows-[auto_1fr] lg:pb-0">
        {/* the decision: place, trip, season, the readings that decide a trip */}
        <div className="flex min-w-0 flex-col px-4 pt-[64px] sm:px-8 lg:col-start-1 lg:row-start-1 lg:pl-10 lg:pr-10 xl:pl-12">
          <RegistryDisclosure className="pb-3 pt-1" fixture={isFixture(props.manifest)} />
          {!off.online && (
            <div className="pb-3">
              <OfflineBanner record={off.record ?? null} problem={off.problem} name={pack.name} inline outside={covering ? null : { place: place.name, href: coastHref({ place: pack.place }) }} />
            </div>
          )}
          {header}
          {/* what changed leads only when something did; the quiet "remembered" line goes under the readings */}
          <div data-testid="brief-changes" className="mb-5 empty:hidden has-[[data-state=first]]:order-last has-[[data-state=first]]:mb-0 has-[[data-state=first]]:mt-3 has-[[data-state=none]]:order-last has-[[data-state=none]]:mb-0 has-[[data-state=none]]:mt-3">
            <SinceLastCheck />
          </div>
          <div className="flex flex-col gap-5" data-testid="brief-glance">
            {official1}
            <DecisionRow cond={b.cond} now={now} today={b.today} tripDate={b.tripDate} asSaved={asSaved} notices={{ ...counts, unclear }} species={species} audience={audience} econ={<TripGlance {...econProps} onOpen={openSheet} />} />
          </div>
          <Fold inline className="mt-1" id="checks" title="Before you go" count={checksN} summary={`${tickedN} of ${checksN} ticked by you`}>
            <p className="text-[13.5px] text-n-ink-2" data-testid="checks-progress">
              <span className="text-n-ink">
                {b.checks.shown.filter((x) => ticked.includes(x.id)).length} of {b.checks.shown.length} ticked by you.
              </span>{" "}
              A tick is not a check by Neritic or by any agency.
            </p>
            <ol className="grid grid-cols-1 divide-y divide-n-line border-y border-n-line" data-testid="checks">
              {b.checks.shown.map((s) => (
                <CheckItem key={s.id} s={s} done={ticked.includes(s.id)} onToggle={() => toggleTick(s.id)} />
              ))}
            </ol>
            {b.checks.more.length > 0 && (
              <div data-testid="checks-more">
                <Caps as="h4">Also worth checking</Caps>
                <ol className="divide-y divide-n-line border-b border-n-line">
                  {b.checks.more.map((s) => (
                    <CheckItem key={s.id} s={s} done={ticked.includes(s.id)} onToggle={() => toggleTick(s.id)} />
                  ))}
                </ol>
              </div>
            )}
          </Fold>
        </div>

        {/* the trip's waters: the page's ground on a wide screen (sticky while the column scrolls), a band on a phone */}
        <div ref={mapBox} className="scroll-mt-16 px-4 pt-5 sm:px-8 lg:col-start-2 lg:row-span-2 lg:row-start-1 lg:p-0">
          <div className="lg:sticky lg:top-0 lg:h-dvh">
            <HarbourMap place={place} g={g} ports={ports} official={official} stations={stations} exploreHref={exploreLink} researchHref={researchLink} onPlace={goPlace} basemapPmtiles={basemap} manifest={props.manifest} baseUrl={props.baseUrl} now={now} />
          </div>
        </div>

        <div className="flex min-w-0 flex-col gap-8 px-4 pt-8 sm:gap-12 sm:px-8 sm:pt-10 lg:col-start-1 lg:row-start-2 lg:pb-16 lg:pl-10 lg:pr-10 xl:pl-12">
          {/* the next 60 hours, then every NWS period in the forecaster's words */}
          <section aria-label="The sea, the next 60 hours" className="flex flex-col gap-1" data-testid="brief-lanes">
            <TimeView cond={b.cond} now={now} today={b.today} tripDate={b.tripDate} limits={limits} onLimits={setLimits} asSaved={asSaved} outlook={b.outlook} runIssued={runIssued} />
            <Fold inline id="sea" title="Sea forecast detail" summary={asSaved ? `every NWS period, the buoy and the tides, ${asSaved}` : "every NWS period in the forecaster's words, the buoy and the tides"}>
              <SeaBody cond={b.cond} tripDate={b.tripDate} today={b.today} limits={limits} now={now} asSaved={asSaved} />
            </Fold>
          </section>

          {/* the one evidence section: the statements, then what each rests on (measured, official), then what is not known */}
          <section aria-labelledby="now-h" className="flex min-w-0 flex-col gap-3" data-testid="brief-now">
            <h2 id="now-h" className="flex flex-wrap items-baseline justify-between gap-x-4 border-t border-n-line-strong pt-4" data-testid="matters-heading">
              <span className="font-serif font-medium tracking-[-0.015em] leading-[1.1] text-[24px] text-n-ink">{snap ? "What mattered when saved" : "What the evidence says"}</span>
              {/* the separator is in the text, so the heading reads "… saved · Oct 22, 9:06 PM" aloud, not "savedOct" */}
              <span className="text-[13px] text-n-ink-2">
                <span className="sr-only"> · </span>
                {snap ? clockLabel(snap.savedAt) : "notices, shore samples, NOAA’s model, the satellite"}
              </span>
            </h2>
            <div className="min-h-[156px]">
              {!g || !settled ? (
                <p className="py-2 text-[15px] text-n-ink-2" role="status" data-testid="matters-reading">
                  Reading the notices, shore samples, NOAA&rsquo;s model and the satellite for {place.name}…
                </p>
              ) : (
                <Matters claims={b.matters} graph={g} stations={stations} onSelect={(s) => router.push(exploreHref(s, back))} fallback={placeSel} />
              )}
            </div>
            <Fold inline openFrom="md" className="border-t border-n-line pt-1 md:pt-4" id="evidence" title="What it rests on" summary={`${near.length + far.length} measured · ${noticeRows} official, not verified`}>
            <div className="grid grid-cols-1 gap-x-10 gap-y-6 md:grid-cols-2">
              <div className="flex min-w-0 flex-col gap-4" data-testid="brief-evidence-measured">
                <div className="flex flex-col gap-1" data-testid="brief-measured">
                  <Caps as="h3" family="measured">
                    Measured near these waters{asSaved ? `, ${asSaved}` : ""}
                  </Caps>
                  {!g ? (
                    <p className="text-[14px] text-n-ink-2">Reading…</p>
                  ) : near.length ? (
                    <ul className="divide-y divide-n-line">{near.map((n) => <NodeRow key={n.id ?? n.variable} n={n} now={now} back={back} />)}</ul>
                  ) : (
                    <p className="text-[14px] text-n-ink-2">Nothing measured recently in these waters themselves{far.length ? "; what exists is farther away or older, below." : "."}</p>
                  )}
                </div>
                {far.length > 0 && (
                  <Fold inline id="context" title="Farther away or older" count={far.length} summary="context only">
                    <ul className="divide-y divide-n-line">{far.map((n) => <NodeRow key={n.id ?? n.variable} n={n} now={now} back={back} />)}</ul>
                  </Fold>
                )}
              </div>
              <div className="flex min-w-0 flex-col gap-2" data-testid="brief-official">
                <h3 id="official-h" className="scroll-mt-16">
                  <Caps family="official">Official notices · not verified</Caps>
                </h3>
                {noticeRows ? (
                  <ul className="divide-y divide-n-line border-b border-n-line">
                    {notices.map((n) => (
                      <li key={n.id} className="py-2.5" data-testid={`notice-${n.id}`}>
                        <p className="text-[15px] font-medium leading-snug">{n.text}</p>
                        <p className="mt-0.5 text-[13px] text-n-ink-2">
                          <span className="nx-caps text-n-official">{n.source.name}</span>
                          {n.where?.name ? ` · ${n.where.name}` : ""}
                          {(n.time as { issued?: string | null }).issued ? ` · since ${dayLabel((n.time as { issued: string }).issued)}` : ""}
                          <span className="text-n-official"> · Not verified</span>
                        </p>
                        {n.freshness?.note && <p className="mt-1 text-[13px] text-n-official">{n.freshness.note}</p>}
                        {n.source.url && (
                          <a href={n.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 text-[13px] text-n-ink underline decoration-n-line-strong underline-offset-4 sm:min-h-8">
                            {n.source.name} notice <Icon name="external" className="h-3.5 w-3.5" />
                          </a>
                        )}
                      </li>
                    ))}
                    {otherHere.length > 0 && (
                      <li className="pt-2.5" data-testid="notices-other">
                        <Caps as="p">{species.id === "any" ? "Also tied to these waters" : `Tied to these waters, not naming ${speciesPhrase(species)}`}</Caps>
                        <ul className="divide-y divide-n-line">
                          {otherHere.map((a) => {
                            const url = a.record.sources.find((x) => x.url.startsWith("https://"))?.url ?? null;
                            return (
                              <li key={a.record.id} className="py-2" data-testid={`notice-official:${a.record.id}`}>
                                <p className="text-[14.5px] leading-snug">{a.record.title}</p>
                                <p className="flex flex-wrap items-center gap-x-3 text-[13px] text-n-ink-2">
                                  <span>
                                    {a.record.agency}
                                    {a.record.effective_date ? ` · since ${dayLabel(a.record.effective_date)}` : ""}
                                  </span>
                                  <span className="text-n-official">Not verified</span>
                                  {url && (
                                    <a href={url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 text-n-ink underline decoration-n-line-strong underline-offset-4 sm:min-h-8">
                                      {a.record.agency} notice <Icon name="external" className="h-3.5 w-3.5" />
                                    </a>
                                  )}
                                </p>
                              </li>
                            );
                          })}
                        </ul>
                      </li>
                    )}
                  </ul>
                ) : (
                  <p className="text-[14px] text-n-ink-2">No notice in Neritic&rsquo;s list is tied to these waters.</p>
                )}
                <p className="text-[13px] leading-snug text-n-ink-2" data-testid="official-caveat">
                  <span className="text-n-official">Not verified</span> · may be incomplete · no notice listed does not mean open or safe. The agency page is the authority:{" "}
                  <a href={`tel:${AGENCY_LINKS.cdfwClosureLine.tel}`} className="nx-focus whitespace-nowrap text-n-ink underline decoration-n-line-strong underline-offset-4">
                    CDFW {AGENCY_LINKS.cdfwClosureLine.phone}
                  </a>
                  {" · "}
                  <a href={`tel:${AGENCY_LINKS.cdphLine.tel}`} className="nx-focus whitespace-nowrap text-n-ink underline decoration-n-line-strong underline-offset-4">
                    CDPH {AGENCY_LINKS.cdphLine.phone}
                  </a>
                </p>
              </div>
            </div>
            </Fold>
            <div className="flex flex-col gap-0 border-t border-n-line pt-2">
              <Fold inline id="unknown" title="Not known here" count={b.unknowns.length} summary="missing evidence is said, never filled in">
                <ul className="flex list-disc flex-col gap-1 pl-5 text-[13.5px] leading-snug text-n-ink-2" data-testid="unknowns">
                  {b.unknowns.map((u) => (
                    <li key={u}>{u}</li>
                  ))}
                </ul>
              </Fold>
              <a href="#unknown-h" className="sr-only" data-testid="glance-unknown">
                {b.unknowns.length} {b.unknowns.length === 1 ? "thing" : "things"} Neritic cannot say here
              </a>
              <Fold inline id="worth" title="Dock value, history" summary="CDFW landings by year, not today's price">
                <p className="text-[14px] text-n-ink" data-testid="glance-worth">
                  {worthLine(place, ports, species)}
                </p>
                <WorthBody place={place} ports={ports} species={species} />
              </Fold>
            </div>
          </section>

          <footer className="flex max-w-[80ch] flex-col gap-3 text-[13px] leading-snug text-n-ink-2">
            <div className="flex flex-wrap gap-x-5 gap-y-1 text-[14px]">
              <Link href={exploreLink} className="nx-focus inline-flex min-h-11 items-center gap-1.5 text-n-ink-2 hover:text-n-ink">
                <Icon name="map" className="h-4 w-4" /> Open in Explore
              </Link>
              <Link href={researchLink} className="nx-focus inline-flex min-h-11 items-center gap-1.5 text-n-ink-2 hover:text-n-ink">
                <Icon name="research" className="h-4 w-4" /> Open in Research
              </Link>
              {hasPack && (
                <button type="button" onClick={() => setPackOpen(true)} className="nx-focus inline-flex min-h-11 items-center gap-1.5 text-n-ink-2 hover:text-n-ink" data-testid="pack-open">
                  <Icon name={off.record?.complete ? "check" : "download"} className="h-4 w-4" /> {off.record?.complete ? "Saved for offline use" : "Save for offline use"}
                </button>
              )}
            </div>
            <p className="nx-margin" data-testid="matters-provenance">
              Official items quote Neritic&rsquo;s copy of the CDFW and CDPH pages and notices. No person has reviewed that copy yet, so the agency pages are the authority. Your limits, ticks, trip numbers and home port stay on this device.
            </p>
            <p>Sources: CDFW and CDPH (Neritic&rsquo;s copy), CalHABMAP via SCCOOS, NOAA C-HARM, NOAA CoastWatch, NWS, NDBC and CO-OPS, CDFW By The Numbers, EIA. This link reproduces this trip. Neritic is an independent research project, not affiliated with or endorsed by NOAA, CDFW or CDPH.</p>
          </footer>
        </div>
      </main>
      <MobileDock />
    </div>
  );
}

/**
 * The worksheet's one framing line, above every number (integrity C.19): hypothetical, the list it quotes
 * and what it says, and that Neritic holds no commercial season. One line, said once.
 */
function SheetLine({ place, species, season, rec }: { place: CoastPlace; species: ReturnType<typeof speciesById>; season: ReturnType<typeof seasonStatus>; rec: boolean }) {
  const s = season.item;
  return (
    <span className="flex flex-col gap-0.5 border-l-2 border-n-official pl-3 sm:block" data-testid="econ-hypothetical">
      <b className="font-medium">Hypothetical: your numbers, not a forecast or permission to fish.</b>{" "}
      <span className="text-n-ink-2" data-testid="econ-official">
        {s ? (
          <span data-testid="econ-closed">
            {s.name} is {s.mark === "closed" ? "closed" : s.mark === "alert" ? "open, with an alert," : "open"} on CDFW&rsquo;s {season.commercial ? "sport " : ""}list for {place.name}&rsquo;s region (Neritic&rsquo;s copy, not verified)
            {season.commercial || season.commercialOnly ? "; " : ". "}
          </span>
        ) : (
          <>{species.id === "any" ? "No species chosen. " : `${species.label}: no entry of its own on CDFW's sport list here${season.commercial || season.commercialOnly ? "; " : ". "}`}</>
        )}
        {(season.commercial || season.commercialOnly) && "Neritic holds no commercial season. "}
        {rec ? <span data-testid="econ-rec-note">A recreational trip is worked out as its cost only: no price, gross, crew pay or net. </span> : ""}
        <span className="whitespace-nowrap">Kept on this device only.</span>
      </span>
    </span>
  );
}

const SEA_VARIABLES = /^(marine forecast|tide predictions|wind|significant wave height|water temperature)$/i;

const econAudienceOf = (a: Audience) => econAudience(a) === "recreational";


const CLASS_WORD: Record<string, { word: string; family: "measured" | "model" }> = {
  measurement: { word: "Measured", family: "measured" },
  satellite_retrieval: { word: "Satellite", family: "measured" },
  published_forecast: { word: "Model / forecast", family: "model" },
};

function whenOf(n: EvidenceNode): string | null {
  const t = n.time as { observed?: string | null; issued?: string | null; valid?: string | null };
  return t.observed ?? t.valid ?? t.issued ?? null;
}

function NodeRow({ n, now, back }: { n: EvidenceNode; now: Date | null; back: string }) {
  const k = CLASS_WORD[n.class] ?? CLASS_WORD.measurement;
  const sel = nodeSelection(n, null) ?? (n.where?.lat != null && n.where?.lon != null ? ({ kind: "point", lat: n.where.lat, lon: n.where.lon } as Selection) : null);
  const stationId = n.class === "measurement" ? n.id?.split(":")[1] : null;
  const target: Selection | null = stationId ? { kind: "station", id: stationId } : sel;
  const when = whenOf(n);
  const stale = n.freshness?.state && n.freshness.state !== "current";
  // a reading is a short value set as an instrument figure; a sentence (a comparison, a forecast's text) is read as text
  const long = (n.text?.length ?? 0) > 18;
  return (
    <li className={`grid gap-x-4 py-2.5 ${long ? "grid-cols-1" : "grid-cols-[minmax(0,1fr)_auto]"}`} data-testid={`node-${n.id ?? n.variable}`} data-node={n.id}>
      <div className="min-w-0">
        <p className="text-[14.5px] leading-snug text-n-ink">
          <Caps family={k.family} className="mr-2">
            {k.word}
          </Caps>
          {n.variable.charAt(0).toUpperCase() + n.variable.slice(1)}
        </p>
        {long && (
          <p className="mt-0.5 line-clamp-3 text-[13.5px] leading-snug text-n-ink" data-value>
            {n.text}
          </p>
        )}
        <p className="mt-0.5 flex flex-wrap items-center gap-x-3 text-[12.5px] leading-snug text-n-ink-2">
          <span>
            {n.where?.name ?? n.source.name}
            {n.where && n.where.km > 0 ? ` · ${kmLabel(n.where.km)}${n.where.bearing ? ` ${n.where.bearing}` : ""}` : ""}
            {when ? ` · ${dayYearLabel(when, now)}${now ? ` (${relTime(when, now)})` : ""}` : ""}
          </span>
          {stale && (
            <span className="inline-flex items-center gap-1 text-n-stale">
              <Icon name="clock" className="h-3.5 w-3.5" /> {n.freshness?.note ?? "older than usual"}
            </span>
          )}
          {n.source.url && (
            <a href={n.source.url} target="_blank" rel="noreferrer" aria-label={`${n.source.name} (source)`} className="nx-focus inline-flex min-h-11 items-center gap-1 text-n-ink underline decoration-n-line-strong underline-offset-4 sm:min-h-7">
              source <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          )}
          {target && (
            <Link href={exploreHref(target, back)} aria-label="Show on map" className="nx-focus inline-flex min-h-11 items-center gap-1 text-n-ink-2 hover:text-n-ink sm:min-h-7" data-testid="node-show">
              <Icon name="locate" className="h-3.5 w-3.5" /> map
            </Link>
          )}
        </p>
      </div>
      {long ? null : (
        <span className="nx-value self-start pt-0.5 text-[19px]" data-value>
          {n.text ?? "—"}
        </span>
      )}
    </li>
  );
}

/** The page's small menu: copy this briefing's link (the trip is in it). */
function MoreMenu() {
  const [open, setOpen] = useState(false);
  const [done, setDone] = useState(false);
  const box = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    const onDown = (e: PointerEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
    };
  }, [open]);
  return (
    <div ref={box} className="relative shrink-0">
      <button type="button" aria-expanded={open} aria-controls={open ? "coast-more-menu" : undefined} aria-label="More: copy a link to this briefing" onClick={() => setOpen((o) => !o)} className="nx-press nx-focus grid h-11 w-11 place-items-center rounded-[3px] border border-n-line text-n-ink-2 hover:text-n-ink" data-testid="coast-more">
        <Icon name="more" className="h-5 w-5" />
      </button>
      {open && (
        <div id="coast-more-menu" className="nx-pop absolute right-0 top-14 z-30 flex w-60 flex-col rounded-[4px] border border-n-line bg-n-surface p-1.5 shadow-[var(--nx-shadow)]">
          <button
            type="button"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(window.location.href);
                setDone(true);
                toast("Link copied");
                setTimeout(() => {
                  setDone(false);
                  setOpen(false);
                }, 1200);
              } catch {
                setDone(false);
              }
            }}
            className="nx-tint nx-focus flex min-h-12 items-center gap-2 rounded-[3px] px-3 text-left text-[15px]"
            data-testid="coast-copy-link"
          >
            <Icon name={done ? "check" : "link"} className="h-[18px] w-[18px]" /> {done ? "Link copied" : "Copy link to this trip"}
          </button>
          <span className="sr-only" aria-live="polite">
            {done ? "Link copied" : ""}
          </span>
        </div>
      )}
    </div>
  );
}

function MobileDock() {
  return (
    <div className="env-instrument fixed inset-x-0 bottom-0 z-40 border-t border-n-line bg-n-surface lg:hidden" style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
      <NavTabs active="coast" size="dock" />
    </div>
  );
}

/** One check before going out: the reader's own tick, what to do, why, and where (an agency page or a phone line). */
function CheckItem({ s, done, onToggle }: { s: Step; done: boolean; onToggle: () => void }) {
  return (
    <li className="grid grid-cols-[44px_minmax(0,1fr)] items-start gap-x-2 py-2" data-testid={`check-${s.id}`} data-done={done}>
      <label className="nx-focus grid h-11 w-11 cursor-pointer place-items-center rounded-[3px] hover:bg-n-surface-2" title="Tick when you have done this (your own note)">
        <input type="checkbox" checked={done} onChange={onToggle} className="h-5 w-5 accent-[var(--nx-accent)]" aria-label={`I have done this: ${s.title}`} data-testid={`check-${s.id}-box`} />
      </label>
      <div className="min-w-0 pt-2">
        <p className={`text-[15px] font-medium leading-snug ${done ? "text-n-ink-2" : "text-n-ink"}`}>{s.title}</p>
        <p className="mt-0.5 text-[13.5px] leading-snug text-n-ink-2" title={/\(HTTP \d+\)/.test(s.reason) ? s.reason : undefined}>
          {s.reason.replace(/\s*\(HTTP \d+\)/, "")}
        </p>
        <p className="flex flex-wrap gap-x-3 text-[13.5px]">
          {"url" in s.source ? (
            <a href={s.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 text-n-ink underline decoration-n-line-strong underline-offset-4">
              {s.source.label} <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          ) : (
            <a href={`tel:${s.source.tel.replace(/[^\d+]/g, "")}`} className="nx-focus inline-flex min-h-11 items-center gap-1 text-n-ink" data-testid="check-tel">
              <span className="font-data">{s.source.label}</span>
            </a>
          )}
          {s.when && <span className="inline-flex min-h-11 items-center text-n-ink-3">{s.when}</span>}
        </p>
      </div>
    </li>
  );
}

function HomeLink({ home, ports }: { home: HomePort; ports: PortsCollection | null }) {
  const p = resolvePlace(home.place, ports);
  if (!p) return null;
  return (
    <Link href={homeHref(home)} className="nx-focus inline-flex min-h-11 items-center gap-1 text-n-ink underline decoration-n-line-strong underline-offset-4" data-testid="coast-home-open">
      <Icon name="harbor" className="h-4 w-4" /> Home: {p.name}
    </Link>
  );
}

/**
 * Which waters, and the reader's home port, opened from the title: change the place (harbours by region,
 * CDPH harvest sites), choose waters on the map, keep this harbour as home on this device (with this
 * trip). A point chosen on the map is a place too, but not a home port.
 */
function PlaceMenu({ place, ports, home, audience, species, onHome, onPlace, onMap, onClose }: { place: CoastPlace; ports: PortsCollection | null; home: HomePort | null; audience: Audience; species: string; onHome: (h: HomePort | null) => void; onPlace: (k: string) => void; onMap: () => void; onClose: () => void }) {
  const homePlace = home ? resolvePlace(home.place, ports) : null;
  const regions = ports?.regions ?? [];
  const byRegion = regions.map((r) => ({ r, items: (ports?.features ?? []).filter((f) => f.properties.region === r.id).map((f) => ({ key: `port-${f.properties.port_code}`, name: f.properties.display_name })) }));
  const sites = sitePlaces();
  const isHome = home?.place === place.key;
  const canBeHome = place.kind !== "point";
  const box = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    box.current?.querySelector<HTMLElement>("select")?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);
  const trip = [audience !== "all" ? AUDIENCES.find((a) => a.id === audience)?.label : null, species !== "any" ? speciesById(species).label : null].filter(Boolean).join(" · ");
  return (
    <div ref={box} id="coast-place-menu" className="nx-pop nx-origin-left flex flex-col gap-3 rounded-[4px] border border-n-line bg-n-surface p-4 shadow-[var(--nx-shadow)]" data-testid="coast-place-bar">
      <label className="flex flex-col gap-1">
        <span className="nx-caps">Another place</span>
        <select value="" onChange={(e) => e.target.value && onPlace(e.target.value)} className="nx-select nx-focus min-h-12 w-full cursor-pointer rounded-[3px] border border-n-line-strong bg-n-surface pl-3 text-[16px] text-n-ink" data-testid="coast-place-picker">
          <option value="">Choose a harbour or harvest site…</option>
          {homePlace && homePlace.key !== place.key && (
            <optgroup label="Your home port">
              <option value={homePlace.key}>{homePlace.name}</option>
            </optgroup>
          )}
          {byRegion.map(({ r, items }) =>
            items.length ? (
              <optgroup key={r.id} label={`Harbours · ${r.label}`}>
                {items.map((i) => (
                  <option key={i.key} value={i.key} disabled={i.key === place.key}>
                    {i.name}
                  </option>
                ))}
              </optgroup>
            ) : null,
          )}
          <optgroup label="CDPH harvest monitoring sites">
            {sites.map((x) => (
              <option key={x.key} value={x.key} disabled={x.key === place.key}>
                {x.name}
              </option>
            ))}
          </optgroup>
        </select>
      </label>
      <button type="button" onClick={onMap} className="nx-press nx-focus inline-flex min-h-12 items-center gap-2 self-start rounded-[3px] px-1 text-[15px] text-n-ink underline decoration-n-line-strong underline-offset-4" data-testid="coast-map-jump">
        <Icon name="map" /> Choose waters on the map
      </button>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-n-line pt-3 text-[14.5px]">
        {isHome ? (
          <>
            <span className="inline-flex items-center gap-1.5 text-n-ink">
              <Icon name="harbor" className="h-4 w-4" /> {place.name} is your home port
            </span>
            <button type="button" onClick={() => onHome(null)} className="nx-focus inline-flex min-h-11 items-center px-1 text-n-ink underline underline-offset-4" data-testid="coast-home-clear">
              Forget
            </button>
          </>
        ) : canBeHome ? (
          <button type="button" onClick={() => onHome({ place: place.key, for: audience, species })} className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 rounded-[3px] border border-n-line-strong px-3.5 text-n-ink" data-testid="coast-home-set">
            <Icon name="harbor" className="h-4 w-4" /> Make {place.name} my home port{trip ? ` (${trip})` : ""}
          </button>
        ) : (
          <span className="text-n-ink-3">A point on the map cannot be a home port.</span>
        )}
      </div>
    </div>
  );
}
