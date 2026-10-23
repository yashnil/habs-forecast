"use client";

import { toast } from "@/v4/ui/toast";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Manifest, PortsCollection } from "@/generated/schema";
import type { OfficialDataset } from "@/generated/official";
import type { PortIntelCollection } from "@/generated/port_intel";
import type { StationLite } from "@/lib/data";
import { AUDIENCES, coastHref, parseAudience, pointKey, portPlaces, resolvePlace, sitePlaces, type Audience, type CoastPlace } from "@/lib/coast";
import { SPECIES, speciesById, type SpeciesId } from "@/content/species";
import { defaultAudience, usePurpose } from "@/lib/purpose";
import { useNow } from "@/lib/useNow";
import { labelledGeometry } from "@/lib/officialGeometry";
import type { EvidenceGraph, EvidenceNode } from "@/lib/harborEvidence";
import { encodeSelection, type Selection } from "@/v4/state/selection";
import { useReportSource } from "@/v4/readiness/store";
import { TopBar, NavTabs } from "@/v4/shell/TopBar";
import { NoticesButton, type LocalNotices } from "@/v4/shell/Notices";
import { RegistryDisclosure } from "@/v4/shell/Disclosure";
import { checklistKey } from "@/lib/trip";
import { SkipLink } from "@/v4/shell/SkipLink";
import { distanceKm } from "@/lib/grid";
import type { Step } from "@/lib/nextSteps";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel, dayYearLabel, kmLabel, relTime } from "@/v4/ui/time";
import { marineHeadline } from "@/lib/conditions";
import { marineMaxima, overLimits } from "@/lib/trip";
import { speciesPhrase } from "@/content/species";
import { nodeSelection } from "@/v4/explore/Inspect";
import { Matters } from "./Matters";
import { TripEconomics, useTripSummary } from "@/v4/coast/economics";
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
import { AtAGlance, SeaCard, SeasonsCard, WorthBody, noticeLine, useSeaLimits } from "@/components/coast/FishingBrief";

const MapCanvas = dynamic(() => import("@/components/map/MapCanvas"), { ssr: false, loading: () => null });

type Props = {
  manifest: Manifest;
  ports: PortsCollection | null;
  official: OfficialDataset | null;
  stations: StationLite[] | null;
  stationsError: string | null;
  baseUrl: string;
  portIntel: PortIntelCollection | null;
};

const EXAMPLES = ["port-550", "port-201", "port-220", "port-593", "port-880", "port-770", "site-clam-beach"];

function exploreHref(s: Selection, back: string): string {
  const q = new URLSearchParams({ sel: encodeSelection(s)!, from: back });
  return `/?${q.toString()}`;
}

/**
 * A section below "Before you go", folded: its title, how many items it holds and one line saying what
 * they are, so the page reads as an answer first and the evidence stays one tap away (never removed).
 */
function Fold({ id, title, count, summary, kicker, children }: { id: string; title: string; count: number; summary: string; kicker?: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  // it opens by height (.nx-fold, 240 ms) and is `hidden` again once it has closed
  const [bodyHidden, setBodyHidden] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const timer = useRef<number | null>(null);
  const toggle = () => {
    if (timer.current != null) window.clearTimeout(timer.current);
    if (!open) {
      setOpen(true);
      setBodyHidden(false);
      timer.current = window.setTimeout(() => setExpanded(true), 16);
    } else {
      setOpen(false);
      setExpanded(false);
      timer.current = window.setTimeout(() => setBodyHidden(true), 240);
    }
  };
  return (
    <section aria-labelledby={`${id}-h`} className="border-t border-n-line pt-3" data-testid={`brief-${id}`} data-open={open}>
      <h2 id={`${id}-h`} className="font-serif text-[21px] font-semibold leading-tight text-n-ink">
        <button type="button" aria-expanded={open} aria-controls={`${id}-body`} onClick={toggle} className="nx-tint nx-focus group flex min-h-14 w-full items-start gap-3 rounded-lg py-2 text-left" data-testid={`brief-${id}-toggle`}>
          <span className="min-w-0 flex-1">
            <span className="flex items-baseline gap-2">
              {title}
              <span className="nx-data rounded-full bg-n-surface-2 px-2 py-0.5 font-ui text-[13px] font-semibold text-n-ink-2">{count}</span>
            </span>
            <span className="mt-1 block font-ui text-[14.5px] font-normal leading-snug text-n-ink-2">{summary}</span>
          </span>
          <Icon name="chevronDown" className={`mt-1 text-n-ink-3 transition-transform duration-[240ms] ease-[var(--nx-ease-move)] ${open ? "rotate-180" : ""}`} />
        </button>
      </h2>
      <div className="nx-fold" data-open={expanded}>
        <div id={`${id}-body`} hidden={bodyHidden} className="flex flex-col gap-4 pb-4 pt-2">
          {kicker && <p className="text-[15px] text-n-ink-2">{kicker}</p>}
          {children}
        </div>
      </div>
    </section>
  );
}

const cap = (v: string) => v.charAt(0).toUpperCase() + v.slice(1);

/** "Water temperature, Pseudo-nitzschia cells and 2 more": what a list of evidence is, in one line. */
function whatList(ns: EvidenceNode[]): string {
  const names = [...new Set(ns.map((n) => cap(n.variable)))];
  if (names.length <= 3) return names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names.at(-1)}` : (names[0] ?? "");
  return `${names.slice(0, 2).join(", ")} and ${names.length - 2} more`;
}

const CLASS_WORD: Record<string, { word: string; dot: string }> = {
  measurement: { word: "Measured", dot: "bg-n-measured" },
  satellite_retrieval: { word: "Satellite", dot: "bg-n-measured" },
  published_forecast: { word: "Model / forecast", dot: "bg-n-model" },
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
  return (
    <li className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 py-3" data-testid={`node-${n.id ?? n.variable}`} data-node={n.id}>
      <span className={`mt-2 h-2.5 w-2.5 rounded-full ${k.dot}`} aria-hidden />
      <div className="min-w-0">
        <p className="flex flex-wrap items-baseline justify-between gap-x-3">
          <span className="text-[15.5px] font-semibold text-n-ink">{n.variable.charAt(0).toUpperCase() + n.variable.slice(1)}</span>
          <span className="nx-data text-[17px] font-semibold text-n-ink" data-value>
            {n.text ?? "—"}
          </span>
        </p>
        <p className="mt-0.5 text-[14px] leading-snug text-n-ink-2">
          {n.where?.name ?? n.source.name}
          {n.where && n.where.km > 0 ? ` · ${kmLabel(n.where.km)}${n.where.bearing ? ` ${n.where.bearing}` : ""}` : ""}
          {when ? ` · ${dayYearLabel(when, now)}${now ? ` (${relTime(when, now)})` : ""}` : ""}
        </p>
        <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
          <span className="font-semibold text-n-ink-3">{k.word}</span>
          {stale && (
            <span className="inline-flex items-center gap-1 font-semibold text-n-stale">
              <Icon name="clock" className="h-3.5 w-3.5" /> {n.freshness?.note ?? "older than usual"}
            </span>
          )}
          {n.source.url && (
            <a href={n.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
              {n.source.name} <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          )}
          {target && (
            <Link href={exploreHref(target, back)} className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-ink-2 hover:text-n-ink" data-testid="node-show">
              <Icon name="locate" className="h-3.5 w-3.5" /> Show on map
            </Link>
          )}
        </p>
      </div>
    </li>
  );
}

// `disabled` until hydrated: a choice made before the page is live would be dropped silently on a slow phone
function Select({ id, label, value, onChange, options, disabled = false }: { id: string; label: string; value: string; onChange: (v: string) => void; options: { id: string; label: string }[]; disabled?: boolean }) {
  return (
    <label htmlFor={id} className="flex min-w-0 flex-1 flex-col gap-1">
      <span className="nx-eyebrow">{label}</span>
      <select id={id} value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)} className="nx-select nx-focus min-h-[52px] w-full rounded-xl border border-n-line-strong bg-n-surface pl-3 text-[16px] font-semibold text-n-ink" data-testid={id}>
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
  // the home port (with the reader's own trip) is remembered on this device only, and never assumed to be where anyone fishes
  const setHome = useCallback((h: HomePort | null) => {
    writeHome(h);
    setHomeState(h);
  }, []);
  const go = useCallback(
    (next: { place?: string | null; for?: Audience; species?: SpeciesId; date?: string | null }) => {
      router.replace(
        coastHref({ place: next.place !== undefined ? next.place : placeKey, for: next.for ?? audience, species: next.species ?? species.id, date: next.date !== undefined ? next.date : sp.get("date") }),
        { scroll: false },
      );
    },
    [router, placeKey, audience, species.id, sp],
  );
  const [tripOpen, setTripOpen] = useState(false);
  // phone (< 1024 px): the map is a strip under the glance, opened on tap; wider, a sticky right rail.
  // null on the server and the first render, so neither map is drawn into the server's HTML
  const [wide, setWide] = useState<boolean | null>(null);
  const [mapOpen, setMapOpen] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    const on = () => setWide(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  const tripSummary = useTripSummary(placeKey, species.id);
  const [limits, setLimits] = useSeaLimits();
  // species first (v5): on the landing, the reader can say what they fish before choosing where
  const [pick, setPick] = useState<SpeciesId>(species.id);
  const [placeOpen, setPlaceOpen] = useState(false);
  // the field pack that covers this place, if one does (else Monterey's, to say what is saved)
  const covering = place?.kind === "port" ? packForPort(Number(place.key.slice(5))) : null;
  const pack = covering ?? PACKS.monterey;
  const off = useOfflinePack(pack.id);
  useRecordSource(off.record, pack.name, off.online, off.evicted, off.problem);
  const [packOpen, setPackOpen] = useState(false);
  const hasPack = !!covering;
  // served from the saved pack (offline, or the page itself came from it): headings say when it was saved,
  // never "now"; null on the server and the first render, so no device time meets server HTML
  const snap = covering ? savedSnapshot(off.record, off.online) : null;
  const basemap = off.record === undefined ? undefined : typeof navigator !== "undefined" && !navigator.onLine && off.record?.complete ? `/offline/${pack.id}/basemap.pmtiles` : null;
  const back = `/coast?${sp.toString()}`;
  const placeSel: Selection | null = !place ? null : place.kind === "port" ? { kind: "harbor", port: Number(place.key.slice(5)) } : { kind: "point", lat: place.lat, lon: place.lon };

  useReportSource(place ? { id: "conditions", label: "Harbour conditions", state: b.cond.loading ? "loading" : b.cond.error ? "unavailable" : b.cond.data ? "ready" : "waiting", detail: b.cond.error ?? "NOAA CO-OPS, NDBC and NWS, read live" } : null);

  const all = useMemo(() => [...portPlaces(ports), ...sitePlaces()], [ports]);
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
  // the notices control leads with this place's own notices (official geography; unsettled apart)
  const localNotices = useMemo<LocalNotices | null>(
    // the same count as the glance and the notices section (useBriefing's one source)
    () => (place ? { name: place.name, covering: b.notices.hereAssocs.map((a) => a.record.id), unsettled: b.notices.unsettled } : null),
    [place, b.notices],
  );
  // another place keeps the trip (activity, species) and is a new entry in the history
  // a new place is a new briefing: it opens at its top (on a phone the map is far below it)
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
  // the statements are shown once the sources settle; until then the evidence is still loading
  useReportSource(place ? { id: "evidence", label: "Briefing evidence", state: b.graph && settled ? (b.outlook === "error" ? "partial" : "ready") : "loading", detail: b.graph ? `${b.matters.length} statements, each with its source` : "Reading notices, shore samples, NOAA's model and the satellite" } : null);
  const g = b.graph;
  const measured = useMemo(() => (g ? g.nodes.filter((n) => n.variable !== "official notice") : []), [g]);
  const near = measured.filter((n) => (n.where?.representative === "at" || n.where?.representative === "near") && (!n.freshness || n.freshness.state === "current"));
  // a station's sample against its own record is already said, whole, by the statement it supports above
  const said = new Set(b.matters.filter((c) => c.history).flatMap((c) => c.supports));
  const far = measured.filter((n) => !near.includes(n) && !(n.id?.startsWith("record:") && said.has(n.id)));
  const notices = g ? g.nodes.filter((n) => n.variable === "official notice") : [];

  // ---------------------------------------------------------------- no place yet: choose one
  if (!place) {
    return (
      <div className="env-chart min-h-dvh bg-n-bg text-n-ink" data-testid="my-coast" data-view="landing">
        <SkipLink />
        <TopBar active="coast" now={now} notices={<NoticesButton official={official} status={sourceStatus(props.manifest, "official")} />} />
        <RegistryDisclosure className="mx-auto w-full max-w-[760px] px-4 sm:px-6" fixture={isFixture(props.manifest)} />
        <main id="main" className="cw-page-in mx-auto w-full max-w-[760px] px-4 pb-28 pt-6 sm:px-6">
          <p className="nx-eyebrow">My Coast</p>
          <h1 className="mt-2 text-balance font-serif text-[36px] font-semibold leading-[1.1] sm:text-[44px]" data-testid="coast-title">
            Where are you going next?
          </h1>
          <p className="mt-3 max-w-[60ch] text-[17px] leading-relaxed text-n-ink-2">One briefing for the trip: CDFW&rsquo;s season list for those waters, the sea forecast, the agencies&rsquo; notices, the latest bloom and toxin samples, and what to check before you leave. Every statement shows where it came from.</p>
          <div className="mt-6" data-testid="coast-species-first">
            <p className="nx-eyebrow" id="coast-species-first-h">What do you fish?</p>
            <div role="group" aria-labelledby="coast-species-first-h" className="-mx-4 mt-2 flex gap-1.5 overflow-x-auto px-4 pb-1 sm:mx-0 sm:flex-wrap sm:px-0">
              {SPECIES.map((x) => (
                <button key={x.id} type="button" aria-pressed={x.id === pick} onClick={() => setPick(x.id)} className={`nx-press nx-focus min-h-11 shrink-0 rounded-full border px-3.5 text-[14.5px] font-semibold ${x.id === pick ? "border-n-ink bg-n-ink text-n-bg" : "border-n-line-strong bg-n-surface text-n-ink-2 hover:text-n-ink"}`} data-testid={`coast-pick-${x.id}`}>
                  {x.label}
                </button>
              ))}
            </div>
            <p className="mt-2 text-[14px] text-n-ink-2">Then choose where. The briefing opens with CDFW&rsquo;s season list for those waters, the sea forecast and what the catch has been worth.</p>
          </div>
          {placeKey && (
            <p className="mt-4 rounded-xl border border-n-official-line bg-n-official-bg px-3 py-2 text-[15px]" role="status" data-testid="coast-unknown-place">
              The place in this link (&ldquo;{placeKey}&rdquo;) is not one Neritic knows. Choose a harbour or harvest site below.
            </p>
          )}
          <ul className="mt-6 grid gap-2 sm:grid-cols-2" data-testid="coast-landing">
            {[...(home ? [home.place] : []), ...EXAMPLES.filter((k) => k !== home?.place)].map((k) => {
              const p = resolvePlace(k, ports);
              if (!p) return null;
              const isHome = k === home?.place;
              return (
                <li key={k}>
                  <Link href={isHome && home && pick === "any" ? homeHref(home) : coastHref({ place: k, for: audience, species: pick })} className="nx-press nx-focus nx-lift flex min-h-[64px] items-center justify-between gap-3 rounded-2xl border border-n-line bg-n-surface px-4 hover:border-n-line-strong" data-testid={`coast-start-${k}`}>
                    <span>
                      <span className="block text-[17px] font-semibold">{p.name}</span>
                      <span className="block text-[13.5px] text-n-ink-3">{isHome ? "Your home port" : p.note}</span>
                    </span>
                    <Icon name="chevronRight" className="text-n-ink-3" />
                  </Link>
                </li>
              );
            })}
          </ul>
          <details className="mt-4">
            <summary className="nx-focus min-h-11 cursor-pointer py-2 text-[15px] font-semibold text-n-accent">All {all.length} places</summary>
            <ul className="mt-2 columns-2 gap-4 text-[15px]">
              {all.map((p) => (
                <li key={p.key}>
                  <Link className="nx-focus inline-flex min-h-11 items-center hover:underline" href={coastHref({ place: p.key, for: audience, species: pick })}>
                    {p.name}
                  </Link>
                </li>
              ))}
            </ul>
          </details>
        </main>
        <MobileDock />
      </div>
    );
  }

  // ---------------------------------------------------------------- the briefing
  const activityOptions = AUDIENCES.map((a) => ({ id: a.id, label: a.id === "all" ? "Any activity" : a.label }));
  const speciesOptions = SPECIES.filter((s) => s.id === "any" || audience === "all" || s.activities.includes(audience as never)).map((s) => ({ id: s.id, label: s.label }));
  const dayOptions = b.days.map((d) => ({ id: d.date, label: `${d.label}, ${d.day}` }));
  const asSaved = snap ? `as saved ${clockLabel(snap.savedAt)}` : null;
  const counts = { here: b.notices.here, namingSpecies: b.notices.namingSpecies };
  const condLine = b.tripPeriod ? `NWS ${b.tripPeriod.name}: ${marineHeadline(b.tripPeriod)}` : null;
  const max = b.tripPeriod ? marineMaxima(b.tripPeriod.text) : null;
  const hasLimits = limits.windKt != null || limits.seasFt != null;
  const overNow = max && hasLimits ? (() => { const o = overLimits(max, limits); return o.wind || o.seas; })() : null;
  const portCode = place.kind === "port" ? Number(place.key.slice(5)) : null;
  // the notices tied to these waters that the graph does not list for this trip (another species or activity)
  const otherHere = b.notices.hereAssocs.filter((a) => !b.notices.namingIds.has(a.record.id));
  const next = nextLink(near, b.matters, g, stations, back, placeSel);

  return (
    <div className="env-chart relative min-h-dvh bg-n-bg text-n-ink" data-testid="my-coast" data-view="trip">
      <SkipLink />
      <TopBar active="coast" now={now} notices={<NoticesButton official={official} status={sourceStatus(props.manifest, "official")} local={localNotices} />} />
      <RegistryDisclosure className="mx-auto w-full max-w-[1240px] px-4 sm:px-6" fixture={isFixture(props.manifest)} />
      {!off.online && (
        <div className="px-3 pt-3">
          {/* in the page's flow: on the briefing it must never cover the title. A place no pack covers says
              so, rather than showing the saved pack's banner over a briefing it does not hold */}
          <OfflineBanner record={off.record ?? null} problem={off.problem} name={pack.name} inline outside={covering ? null : { place: place.name, href: coastHref({ place: pack.place }) }} />
        </div>
      )}
      {packOpen && <PackDialog pack={pack} manifest={props.manifest} baseUrl={props.baseUrl} official={official} onClose={() => setPackOpen(false)} />}
      <div className="mx-auto grid w-full max-w-[1240px] gap-8 px-4 pb-32 pt-3 sm:px-6 lg:grid-cols-[minmax(0,700px)_minmax(0,1fr)] lg:pt-5">
        <main id="main" className="cw-page-in flex min-w-0 flex-col gap-6">
          {/* the compact header (v6): the place, then the trip as one tap target; the link in a small menu */}
          <header className="flex flex-col gap-2" data-testid="coast-header">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <h1 className="text-balance font-serif text-[30px] font-semibold leading-[1.1] sm:text-[38px]" data-testid="coast-title">
                  {/* the title is the place control: other harbours, waters on the map, the home port */}
                  <button type="button" aria-expanded={placeOpen} aria-controls={placeOpen ? "coast-place-menu" : undefined} aria-describedby="coast-place-hint" onClick={() => setPlaceOpen((o) => !o)} className="nx-tint nx-focus group inline-flex max-w-full items-center gap-2 rounded-lg text-left hover:text-n-accent min-h-11" data-testid="coast-place-button">
                    <span className="min-w-0">{place.name}</span>
                    <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-n-line-strong bg-n-surface text-n-ink-2 group-hover:text-n-accent">
                      <Icon name="chevronDown" className={`h-[18px] w-[18px] transition-transform ${placeOpen ? "rotate-180" : ""}`} />
                    </span>
                  </button>
                </h1>
                <span id="coast-place-hint" hidden>
                  Change the place, choose waters on the map, or keep it as your home port
                </span>
                <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[14px] text-n-ink-2" data-testid="coast-subtitle">
                  <span>
                    {place.note}
                    {place.county ? ` · ${place.county} County` : ""}
                  </span>
                  {home?.place === place.key ? (
                    <span className="inline-flex items-center gap-1 font-semibold text-n-ink" data-testid="coast-home">
                      <Icon name="harbor" className="h-4 w-4 text-n-accent" /> Your home port
                    </span>
                  ) : home ? (
                    <HomeLink home={home} ports={ports} />
                  ) : null}
                </p>
              </div>
              <MoreMenu />
            </div>
            {placeOpen && (
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
                onMap={() => {
                  setPlaceOpen(false);
                  setMapOpen(true);
                }}
                onClose={() => setPlaceOpen(false)}
              />
            )}
            <button type="button" aria-expanded={tripOpen} aria-controls="trip-controls" onClick={() => setTripOpen((o) => !o)} className="nx-press nx-focus flex min-h-12 items-center justify-between gap-3 rounded-xl border border-n-line bg-n-surface px-3.5 text-left sm:hidden" data-testid="trip-summary">
              <span className="min-w-0 text-[15px] font-semibold leading-snug">
                {audience === "all" ? "Any activity" : AUDIENCES.find((x) => x.id === audience)?.label} · {species.label} · {b.tripDay ? `${b.tripDay.label}, ${b.tripDay.day}` : "Today"}
              </span>
              <span className="shrink-0 text-[14.5px] font-semibold text-n-accent" data-testid="trip-change">
                {tripOpen ? "Done" : "Change"}
              </span>
            </button>
            <div id="trip-controls" className={`flex-col gap-2 sm:flex sm:flex-row ${tripOpen ? "flex" : "hidden"}`} data-testid="trip-controls">
              <Select id="trip-activity" disabled={!mounted} label="Activity" value={audience} onChange={(v) => go({ for: v as Audience, species: v === "all" || species.activities.includes(v as never) ? species.id : "any" })} options={activityOptions} />
              <Select id="trip-species" disabled={!mounted} label="Species" value={species.id} onChange={(v) => go({ species: v as SpeciesId })} options={speciesOptions} />
              <Select id="trip-day" disabled={!mounted} label="Day" value={b.tripDate ?? dayOptions[0]?.id ?? ""} onChange={(v) => go({ date: v === b.days[0]?.date ? null : v })} options={dayOptions.length ? dayOptions : [{ id: "", label: "Today" }]} />
            </div>
          </header>

          <AtAGlance place={place} ports={ports} species={species} audience={audience} cond={b.cond} tripDate={b.tripDate} today={b.today} notices={counts} limits={limits} onLimits={setLimits} asSaved={asSaved} trip={tripSummary} unknowns={b.unknowns.length} next={next} />

          {/* phone: the map is a strip under the glance, opened on tap (at 1440 it is the right rail) */}
          {wide === false && (
            <div id="coast-map" className="scroll-mt-4" data-testid="locator">
              <Locator place={place} g={g} ports={ports} official={official} stations={stations} back={back} onPlace={goPlace} basemapPmtiles={basemap} strip open={mapOpen} onOpen={setMapOpen} />
            </div>
          )}

          <section aria-labelledby="now-h" className="flex flex-col gap-1" data-testid="brief-now">
            <h2 id="now-h" className="font-serif text-[22px] font-semibold leading-tight" data-testid="matters-heading">
              {snap ? (
                <>
                  What mattered when saved <span className="font-ui text-[15px] font-semibold text-n-ink-2">· {clockLabel(snap.savedAt)}</span>
                </>
              ) : (
                "What matters now"
              )}
            </h2>
            {!g || !settled ? (
              <p className="text-[15px] text-n-ink-2" role="status" data-testid="matters-reading">
                Reading the notices, shore samples, NOAA&rsquo;s model and the satellite for {place.name}…
              </p>
            ) : (
              <Matters claims={b.matters} graph={g} stations={stations} onSelect={(s) => router.push(exploreHref(s, back))} fallback={placeSel} />
            )}
            {/* the one page-level disclosure: every official line, notice and season below is this unreviewed copy */}
            <p className="mt-1 text-[13px] leading-snug text-n-ink-3" data-testid="matters-provenance">
              Official items quote Neritic&rsquo;s copy of the CDFW and CDPH pages and notices. No person has reviewed that copy yet, so the agency pages are the authority. Your limits, ticks, trip numbers and home port stay on this device.
            </p>
          </section>

          <div data-testid="brief-changes">
            <SinceLastCheck />
          </div>

          {/* the sections, in the order a trip is planned (v6): money, sea, seasons, checks, then the evidence */}
          <div data-testid="brief-econ">
            <TripEconomics place={{ key: place.key, name: place.name, portCode }} species={species.id} audience={audience} tripDate={b.tripDate} conditions={condLine || asSaved ? { line: condLine ?? "No NWS period for this day", over: overNow, asSaved } : null} notices={counts} />
          </div>
          <SeaCard cond={b.cond} tripDate={b.tripDate} today={b.today} limits={limits} onLimits={setLimits} now={now} asSaved={asSaved} />
          <SeasonsCard place={place} species={species} audience={audience} tripDate={b.tripDate} today={b.today} />

          <section aria-labelledby="checks-h" className="flex flex-col gap-2 border-t border-n-line pt-6" data-testid="brief-checks">
            <h2 id="checks-h" className="font-serif text-[22px] font-semibold leading-tight text-n-ink">
              Before you go
            </h2>
            <p className="text-[13.5px] text-n-ink-2" data-testid="checks-progress">
              <span className="font-semibold text-n-ink">
                {b.checks.shown.filter((x) => ticked.includes(x.id)).length} of {b.checks.shown.length} ticked by you.
              </span>{" "}
              A tick is not a check by Neritic or by any agency.
            </p>
            <ol className="divide-y divide-n-line border-y border-n-line" data-testid="checks">
              {b.checks.shown.map((s) => (
                <CheckItem key={s.id} s={s} done={ticked.includes(s.id)} onToggle={() => toggleTick(s.id)} />
              ))}
            </ol>
            {/* at most five checks up front; the rest stay one tap away, never dropped */}
            {b.checks.more.length > 0 && (
              <details data-testid="checks-more">
                <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center font-semibold text-n-accent">
                  {b.checks.more.length} more {b.checks.more.length === 1 ? "check" : "checks"}
                </summary>
                <ol className="divide-y divide-n-line border-y border-n-line">
                  {b.checks.more.map((s) => (
                    <CheckItem key={s.id} s={s} done={ticked.includes(s.id)} onToggle={() => toggleTick(s.id)} />
                  ))}
                </ol>
              </details>
            )}
          </section>

          {/* the evidence behind the answer, folded: each says how much it holds and what, one tap away */}
          <div className="flex flex-col" data-testid="brief-evidence">
            <p className="nx-eyebrow pt-2">Evidence</p>
            <Fold id="worth" title="Dock value, history" count={2} summary="CDFW's statewide dock values for 2023 and 2024: not today's price">
              <WorthBody place={place} ports={ports} species={species} />
            </Fold>
            <Fold id="measured" title="The evidence near you" count={near.length} summary={!g ? "Reading…" : near.length ? `${whatList(near)}${asSaved ? `, ${asSaved}` : ""}` : "Nothing measured recently in these waters themselves"}>
              {!g ? (
                <p className="text-[15px] text-n-ink-2">Reading…</p>
              ) : near.length ? (
                <ul className="divide-y divide-n-line">{near.map((n) => <NodeRow key={n.id ?? n.variable} n={n} now={now} back={back} />)}</ul>
              ) : (
                <p className="text-[15px] text-n-ink-2">Nothing measured recently in these waters themselves. What exists is farther away or older, below.</p>
              )}
            </Fold>
            {far.length > 0 && (
              <Fold id="context" title="Farther away or older" count={far.length} summary={`Context only: ${whatList(far).toLowerCase()}`}>
                <ul className="divide-y divide-n-line">{far.map((n) => <NodeRow key={n.id ?? n.variable} n={n} now={now} back={back} />)}</ul>
              </Fold>
            )}

            <Fold id="official" title="Official restrictions" count={counts.here} summary={noticeLine(counts, species, audience)}>
              {notices.length + otherHere.length ? (
                <ul className="divide-y divide-n-line">
                  {notices.map((n) => (
                    <li key={n.id} className="py-3" data-testid={`notice-${n.id}`}>
                      <p className="text-[15.5px] font-semibold leading-snug">{n.text}</p>
                      <p className="mt-0.5 text-[14px] text-n-ink-2">
                        {n.source.name}
                        {n.where?.name ? ` · ${n.where.name}` : ""}
                        {(n.time as { issued?: string | null }).issued ? ` · since ${dayLabel((n.time as { issued: string }).issued)}` : ""}
                      </p>
                      {n.freshness?.note && <p className="mt-1 text-[14px] font-semibold text-n-official">{n.freshness.note}</p>}
                      <p className="flex flex-wrap gap-x-3 text-[13px]">
                        <span className="inline-flex min-h-11 items-center font-semibold text-n-official">Not verified</span>
                        {n.source.url && (
                          <a href={n.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
                            {n.source.name} notice <Icon name="external" className="h-3.5 w-3.5" />
                          </a>
                        )}
                      </p>
                    </li>
                  ))}
                  {otherHere.length > 0 && (
                    <li className="pt-3" data-testid="notices-other">
                      <p className="nx-eyebrow">{species.id === "any" ? "Also tied to these waters" : `Tied to these waters, not naming ${speciesPhrase(species)}`}</p>
                      <ul className="divide-y divide-n-line">
                        {otherHere.map((a) => {
                          const url = a.record.sources.find((x) => x.url.startsWith("https://"))?.url ?? null;
                          return (
                            <li key={a.record.id} className="py-2" data-testid={`notice-official:${a.record.id}`}>
                              <p className="text-[15px] font-semibold leading-snug">{a.record.title}</p>
                              <p className="flex flex-wrap items-center gap-x-3 text-[13px] text-n-ink-2">
                                <span>
                                  {a.record.agency}
                                  {a.record.effective_date ? ` · since ${dayLabel(a.record.effective_date)}` : ""}
                                </span>
                                <span className="font-semibold text-n-official">Not verified</span>
                                {url && (
                                  <a href={url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
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
                <p className="text-[15px] text-n-ink-2">No notice in Neritic&rsquo;s list is tied to these waters. The list may be incomplete: check CDFW and CDPH.</p>
              )}
            </Fold>

            <Fold id="unknown" title="What is not known" count={b.unknowns.length} summary="Missing evidence is said, never filled in">
              <ul className="flex list-disc flex-col gap-2 pl-5 text-[15.5px] leading-relaxed text-n-ink-2" data-testid="unknowns">
                {b.unknowns.map((u) => (
                  <li key={u}>{u}</li>
                ))}
              </ul>
            </Fold>
          </div>

          <footer className="flex flex-col gap-3 border-t border-n-line pt-6 text-[14px] text-n-ink-3">
            <div className="flex flex-wrap gap-2">
              <Link href={exploreHref(placeSel!, back)} className="nx-press nx-focus flex min-h-12 items-center gap-2 rounded-xl border border-n-line-strong px-4 font-semibold text-n-ink">
                <Icon name="map" /> Open in Explore
              </Link>
              <Link href={`/research?${new URLSearchParams({ place: place.key, from: back }).toString()}`} className="nx-press nx-focus flex min-h-12 items-center gap-2 rounded-xl border border-n-line-strong px-4 font-semibold text-n-ink">
                <Icon name="research" /> Open in Research
              </Link>
              {hasPack && (
                <button type="button" onClick={() => setPackOpen(true)} className="nx-press nx-focus flex min-h-12 items-center gap-2 rounded-xl border border-n-line-strong px-4 font-semibold text-n-ink" data-testid="pack-open">
                  <Icon name={off.record?.complete ? "check" : "download"} /> {off.record?.complete ? "Saved for offline use" : "Save for offline use"}
                </button>
              )}
            </div>
            <p>Sources: CDFW and CDPH (Neritic&rsquo;s copy), CalHABMAP via SCCOOS, NOAA C-HARM, NOAA CoastWatch, NWS, NDBC and CO-OPS, CDFW By The Numbers. This link reproduces this trip.</p>
            <p>Neritic is an independent research project and is not affiliated with or endorsed by NOAA, CDFW, or CDPH.</p>
          </footer>
        </main>

        {/* 1440: a sticky rail with the map and the trip's numbers */}
        {wide && (
          <aside id="coast-map" aria-label={`Map of ${place.name}`} className="flex flex-col gap-3 self-start lg:sticky lg:top-4" data-testid="locator">
            <Locator place={place} g={g} ports={ports} official={official} stations={stations} back={back} onPlace={goPlace} basemapPmtiles={basemap} />
            <a href="#trip-economics" className="nx-press nx-focus nx-lift flex flex-col gap-0.5 rounded-2xl border border-n-line bg-n-surface p-3.5 hover:border-n-line-strong" data-testid="rail-econ">
              <span className="nx-eyebrow">Trip economics</span>
              {tripSummary ? (
                <span className="nx-data text-[15px] font-semibold">
                  Cost {tripSummary.costUsd != null ? `$${Math.round(tripSummary.costUsd).toLocaleString("en-US")}` : "—"}
                  {tripSummary.breakEvenLb != null ? ` · break-even ${Math.round(tripSummary.breakEvenLb).toLocaleString("en-US")} lb` : ""}
                  {tripSummary.netUsd != null ? ` · net ${tripSummary.netUsd < 0 ? "−" : ""}$${Math.abs(Math.round(tripSummary.netUsd)).toLocaleString("en-US")}` : ""}
                </span>
              ) : (
                <span className="text-[15px] font-semibold">Work out cost and break-even from your own numbers</span>
              )}
              <span className="text-[13px] text-n-ink-2">{tripSummary ? `Your scenario “${tripSummary.name}”, not a forecast.` : "A scenario from your inputs, not a forecast."}</span>
            </a>
          </aside>
        )}
      </div>
      <MobileDock />
    </div>
  );
}

/**
 * "Inspect next": one deep link. The newest measurement near these waters, in Research with this
 * briefing as the way back; else the first statement's place in Explore; else the harbour in Explore.
 */
function nextLink(near: EvidenceNode[], matters: { supports: string[] }[], g: EvidenceGraph | null, stations: StationLite[] | null, back: string, placeSel: Selection | null): { label: string; href: string } | null {
  const st = near.find((n) => n.class === "measurement" && n.id && stations?.some((s) => s.station_id === n.id!.split(":")[1]));
  if (st) {
    const id = st.id!.split(":")[1];
    return { label: `${cap(st.variable)} at ${st.where?.name ?? "the nearest station"}, in Research`, href: `/research?${new URLSearchParams({ station: id, from: back }).toString()}` };
  }
  const nodes = new Map((g?.nodes ?? []).map((n) => [n.id, n]));
  const t = matters.flatMap((c) => c.supports).map((id) => nodeSelection(nodes.get(id), stations)).find(Boolean) ?? placeSel;
  return t ? { label: "These waters' evidence on the map, in Explore", href: exploreHref(t, back) } : null;
}

/** A small map that says where the evidence is; the full spatial view is Explore. */
/** `basemapPmtiles`: offline with a saved field pack, the locator draws the pack's own map, not the tile server's.
 *  Undefined until the pack has been read: the map waits for that decision (its basemap is fixed for its life,
 *  so a map made before it would ask the tile server offline).
 *  `strip` (phones): a one-line bar under the glance; the map loads only once it is opened. */
function Locator({ place, g, ports, official, stations, back, onPlace, basemapPmtiles, strip = false, open = true, onOpen }: { place: CoastPlace; g: EvidenceGraph | null; ports: PortsCollection | null; official: OfficialDataset | null; stations: StationLite[] | null; back: string; onPlace: (key: string) => void; basemapPmtiles?: string | null; strip?: boolean; open?: boolean; onOpen?: (o: boolean) => void }) {
  const feats = useMemo<GeoJSON.FeatureCollection>(() => {
    const out: GeoJSON.Feature[] = [{ type: "Feature", geometry: { type: "Point", coordinates: [place.lon, place.lat] }, properties: { kind: place.kind === "port" ? "harbor" : "point", label: place.name } }];
    for (const n of g?.nodes ?? []) {
      const sid = n.class === "measurement" ? n.id?.split(":")[1] : null;
      const st = sid ? stations?.find((s) => s.station_id === sid) : null;
      if (st && !out.some((f) => (f.properties as { id?: string }).id === sid)) out.push({ type: "Feature", geometry: { type: "Point", coordinates: [st.lon, st.lat] }, properties: { kind: "station", id: sid } });
    }
    return { type: "FeatureCollection", features: out };
  }, [place, g, stations]);
  const geom = useMemo(() => labelledGeometry(official), [official]);
  // frame the place and what lies within 40 km (a station far up the coast is drawn, not framed:
  // framing it zoomed the locator out to the whole coast)
  const pts = feats.features.map((f) => (f.geometry as GeoJSON.Point).coordinates).filter(([lo, la]) => distanceKm(place.lat, place.lon, la, lo) <= 40);
  const lons = pts.map((p) => p[0]);
  const lats = pts.map((p) => p[1]);
  const w = Math.min(...lons, place.lon - 0.2) - 0.03;
  const s = Math.min(...lats, place.lat - 0.14) - 0.03;
  const e = Math.max(...lons, place.lon + 0.2) + 0.03;
  const n = Math.max(...lats, place.lat + 0.14) + 0.03;
  const bounds: [[number, number], [number, number]] = [[w, s], [e, n]];
  // the evidence arrives after the map: frame the place and every station the briefing names
  const [map, setMap] = useState<import("maplibre-gl").Map | null>(null);
  // the map loads only when it comes near the screen (and, as a phone strip, once it is opened)
  const box = useRef<HTMLDivElement | null>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    if (!open) return;
    const el = box.current;
    if (!el || typeof IntersectionObserver === "undefined") return setNear(true);
    const io = new IntersectionObserver((es) => es.some((e) => e.isIntersecting) && setNear(true), { rootMargin: "300px" });
    io.observe(el);
    return () => io.disconnect();
  }, [open]);
  useEffect(() => {
    map?.fitBounds([[w, s], [e, n]], { padding: 28, duration: 0, maxZoom: 11 });
  }, [map, w, s, e, n]);
  const exploreLink = (
    <Link href={`/?${new URLSearchParams({ sel: place.kind === "port" ? `harbor:${place.key.slice(5)}` : `point:${place.lat.toFixed(4)},${place.lon.toFixed(4)}`, from: back }).toString()}`} className="nx-focus flex min-h-11 shrink-0 items-center gap-1 rounded-lg px-2 text-[14px] font-semibold text-n-accent" data-testid="locator-explore">
      Explore <Icon name="chevronRight" className="h-4 w-4" />
    </Link>
  );
  if (strip && !open)
    return (
      <div className="env-instrument flex items-center justify-between gap-2 rounded-2xl border border-n-line bg-n-surface pl-1 pr-1">
        <button type="button" aria-expanded={false} onClick={() => onOpen?.(true)} className="nx-tint nx-focus flex min-h-12 min-w-0 flex-1 items-center gap-2 rounded-xl px-2.5 text-left text-[14.5px] font-semibold text-n-ink" data-testid="locator-toggle">
          <Icon name="map" className="h-[18px] w-[18px] shrink-0 text-n-accent" />
          <span className="min-w-0 truncate">Map: {place.name}, its stations and notice limits</span>
        </button>
        {exploreLink}
      </div>
    );
  return (
    <div ref={box} className="env-instrument overflow-hidden rounded-2xl border border-n-line">
      <div className={`relative ${strip ? "h-[240px]" : "h-[360px]"}`}>
        {/* until the map has drawn: the same quiet bathymetric lines as Explore's start, never a blank box */}
        {!map && (
          <div className="absolute inset-0 grid place-items-center" data-testid="locator-loading" role="status">
            <svg className="absolute inset-0 h-full w-full text-n-accent opacity-25" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice" aria-hidden>
              {[0, 1, 2, 3, 4].map((i) => (
                <path key={i} className={i === 2 ? (near ? "nx-sweep" : undefined) : "nx-draw"} pathLength={i === 2 ? undefined : 1} style={{ "--i": i } as React.CSSProperties} d={`M-20 ${70 + i * 38} C 80 ${40 + i * 40}, 160 ${120 + i * 30}, 240 ${80 + i * 38} S 380 ${110 + i * 34}, 430 ${90 + i * 38}`} fill="none" stroke="currentColor" strokeWidth={i === 2 ? 1.6 : 0.8} />
              ))}
            </svg>
            <p className="nx-late relative text-[13.5px] text-n-ink-2">{near ? `Drawing ${place.name} and its stations…` : `Map of ${place.name}`}</p>
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
            initialBounds={bounds}
            initialPadding={24}
            // choose other waters on the map: a harbour, or any point in the water (briefed as that point)
            onPort={(code) => onPlace(`port-${code}`)}
            onPoint={(pt) => onPlace(pointKey(pt.lat, pt.lon))}
            onReady={setMap}
          />
        )}
      </div>
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <p className="text-[13px] text-n-ink-2" data-testid="locator-caption">
          {place.name}, the stations behind this briefing and the notices&rsquo; limits. <span className="font-semibold text-n-ink">Tap a harbour or the water to brief those waters.</span>
        </p>
        <div className="flex shrink-0 items-center">
          {strip && (
            <button type="button" aria-expanded onClick={() => onOpen?.(false)} className="nx-focus min-h-11 rounded-lg px-2 text-[14px] font-semibold text-n-ink-2" data-testid="locator-hide">
              Hide
            </button>
          )}
          {exploreLink}
        </div>
      </div>
      <p className="px-3 pb-2 text-[11px] text-n-ink-3">© OpenStreetMap contributors · OpenMapTiles · NOAA relief. Not for navigation.</p>
    </div>
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
      <button type="button" aria-expanded={open} aria-controls={open ? "coast-more-menu" : undefined} aria-label="More: copy a link to this briefing" onClick={() => setOpen((o) => !o)} className="nx-press nx-focus grid h-12 w-12 place-items-center rounded-full border border-n-line-strong bg-n-surface text-n-ink" data-testid="coast-more">
        <Icon name="more" className="h-5 w-5" />
      </button>
      {open && (
        <div id="coast-more-menu" className="nx-pop absolute right-0 top-14 z-30 flex w-60 flex-col rounded-2xl border border-n-line bg-n-surface p-1.5 shadow-[var(--nx-shadow)]">
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
            className="nx-tint nx-focus flex min-h-12 items-center gap-2 rounded-xl px-3 text-left text-[15px] font-semibold"
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

/** One check before going out, one line: the reader's own tick and what to do; why, and where (an agency page, or a phone line that dials), one tap away. */
function CheckItem({ s, done, onToggle }: { s: Step; done: boolean; onToggle: () => void }) {
  return (
    <li className={`grid grid-cols-[44px_minmax(0,1fr)] items-start gap-x-1 py-1 ${done ? "opacity-80" : ""}`} data-testid={`check-${s.id}`} data-done={done}>
      {/* the reader's own acknowledgement: never a verification of the step */}
      <label className="nx-focus grid h-11 w-11 cursor-pointer place-items-center rounded-xl hover:bg-n-surface-2" title="Tick when you have done this (your own note)">
        <input type="checkbox" checked={done} onChange={onToggle} className="h-5 w-5 accent-[var(--nx-accent)]" aria-label={`I have done this: ${s.title}`} data-testid={`check-${s.id}-box`} />
      </label>
      <details className="group min-w-0">
        <summary className={`nx-focus flex min-h-11 cursor-pointer list-none items-center gap-2 rounded-lg text-[15.5px] font-semibold leading-snug ${done ? "text-n-ink-2" : ""}`}>
          <span className="min-w-0 flex-1 py-1">{s.title}</span>
          <Icon name="chevronDown" className="h-4 w-4 shrink-0 text-n-ink-3 transition-transform group-open:rotate-180" />
        </summary>
        {/* a service's technical reason (an HTTP status) stays available on hover, not in the sentence */}
        <p className="text-[14.5px] leading-snug text-n-ink-2" title={/\(HTTP \d+\)/.test(s.reason) ? s.reason : undefined}>
          {s.reason.replace(/\s*\(HTTP \d+\)/, "")}
        </p>
        <p className="flex flex-wrap gap-x-3 text-[13.5px]">
          {"url" in s.source ? (
            <a href={s.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
              {s.source.label} <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          ) : (
            <a href={`tel:${s.source.tel.replace(/[^\d+]/g, "")}`} className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent" data-testid="check-tel">
              {/* the label is the number as the agency writes it; the dialled form is in the link */}
              <span className="nx-data">{s.source.label}</span>
            </a>
          )}
          {s.when && <span className="inline-flex min-h-11 items-center text-n-ink-3">{s.when}</span>}
        </p>
      </details>
    </li>
  );
}

/** One tap back to the home briefing, with the reader's own trip. */
function HomeLink({ home, ports }: { home: HomePort; ports: PortsCollection | null }) {
  const p = resolvePlace(home.place, ports);
  if (!p) return null;
  return (
    <Link href={homeHref(home)} className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent underline underline-offset-2" data-testid="coast-home-open">
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
    <div ref={box} id="coast-place-menu" className="nx-pop nx-origin-left flex flex-col gap-3 rounded-[18px] border border-n-line bg-n-surface p-4 shadow-[var(--nx-shadow)]" data-testid="coast-place-bar">
      <label className="flex flex-col gap-1">
        <span className="nx-eyebrow">Another place</span>
        <select
          value=""
          onChange={(e) => e.target.value && onPlace(e.target.value)}
          className="nx-select nx-focus min-h-12 w-full cursor-pointer rounded-xl border border-n-line-strong bg-n-surface pl-3 text-[16px] font-semibold text-n-ink"
          data-testid="coast-place-picker"
        >
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
      <a href="#coast-map" onClick={onMap} className="nx-press nx-focus inline-flex min-h-12 items-center gap-2 rounded-xl border border-n-line px-3.5 text-[15px] font-semibold text-n-accent lg:hidden" data-testid="coast-map-jump">
        <Icon name="map" /> Choose waters on the map
      </a>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-n-line pt-3 text-[14.5px]">
        {isHome ? (
          <>
            <span className="inline-flex items-center gap-1.5 font-semibold text-n-ink">
              <Icon name="harbor" className="h-4 w-4 text-n-accent" /> {place.name} is your home port
            </span>
            <button type="button" onClick={() => onHome(null)} className="nx-focus inline-flex min-h-11 items-center px-1 font-semibold text-n-accent underline underline-offset-2" data-testid="coast-home-clear">
              Forget
            </button>
          </>
        ) : canBeHome ? (
          <button type="button" onClick={() => onHome({ place: place.key, for: audience, species })} className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 rounded-full border border-n-line-strong px-3.5 font-semibold text-n-ink" data-testid="coast-home-set">
            <Icon name="harbor" className="h-4 w-4 text-n-accent" /> Make {place.name} my home port{trip ? ` (${trip})` : ""}
          </button>
        ) : (
          <span className="text-n-ink-3">A point on the map cannot be a home port.</span>
        )}
      </div>
    </div>
  );
}
