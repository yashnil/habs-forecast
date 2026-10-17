"use client";

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
import { dayLabel, kmLabel, relTime } from "@/v4/ui/time";
import { ClaimRow, nodeSelection } from "@/v4/explore/Inspect";
import { SinceLastCheck } from "@/components/coast/SinceLastCheck";
import { useBriefing } from "./useBriefing";
import { PACKS, packForPort } from "@/v4/offline/packs";
import { isFixture, sourceStatus } from "@/lib/layers";
import { useOfflinePack } from "@/v4/offline/useOfflinePack";
import { OfflinePanel, useRecordSource } from "@/v4/offline/OfflinePanel";
import { OfflineBanner } from "@/v4/offline/OfflineBanner";

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
const HOME_KEY = "cw-home-port";

function exploreHref(s: Selection, back: string): string {
  const q = new URLSearchParams({ sel: encodeSelection(s)!, from: back });
  return `/?${q.toString()}`;
}

function Section({ id, title, kicker, children }: { id: string; title: string; kicker?: string; children: React.ReactNode }) {
  return (
    <section aria-labelledby={`${id}-h`} className="flex flex-col gap-4 border-t border-n-line pt-7" data-testid={`brief-${id}`}>
      <div>
        <h2 id={`${id}-h`} className="font-serif text-[24px] font-semibold leading-tight text-n-ink">
          {title}
        </h2>
        {kicker && <p className="mt-1 text-[15px] text-n-ink-2">{kicker}</p>}
      </div>
      {children}
    </section>
  );
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
          {when ? ` · ${dayLabel(when)}${now ? ` (${relTime(when, now)})` : ""}` : ""}
        </p>
        <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-[13px]">
          <span className="font-semibold text-n-ink-3">{k.word}</span>
          {stale && (
            <span className="inline-flex items-center gap-1 font-semibold text-n-stale">
              <Icon name="clock" className="h-3.5 w-3.5" /> {n.freshness?.note ?? "older than usual"}
            </span>
          )}
          {n.source.url && (
            <a href={n.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-8 items-center gap-1 font-semibold text-n-accent">
              {n.source.name} <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          )}
          {target && (
            <Link href={exploreHref(target, back)} className="nx-focus inline-flex min-h-8 items-center gap-1 font-semibold text-n-ink-2 hover:text-n-ink" data-testid="node-show">
              <Icon name="locate" className="h-3.5 w-3.5" /> Show on map
            </Link>
          )}
        </p>
      </div>
    </li>
  );
}

function TripSummary({ audience: a, species: sp2, day, open, onToggle }: { audience: Audience; species: string; day: string; open: boolean; onToggle: () => void }) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-2xl border border-n-line bg-n-surface px-4 py-2 sm:hidden" data-testid="trip-summary">
      <p className="min-w-0 text-[15.5px] font-semibold leading-snug">
        {a === "all" ? "Any activity" : AUDIENCES.find((x) => x.id === a)?.label} · {sp2} · {day}
      </p>
      <button type="button" aria-expanded={open} onClick={onToggle} className="nx-press nx-focus min-h-12 shrink-0 rounded-xl px-3 text-[15px] font-semibold text-n-accent" data-testid="trip-change">
        {open ? "Done" : "Change"}
      </button>
    </div>
  );
}

function Select({ id, label, value, onChange, options }: { id: string; label: string; value: string; onChange: (v: string) => void; options: { id: string; label: string }[] }) {
  return (
    <label htmlFor={id} className="flex min-w-0 flex-1 flex-col gap-1">
      <span className="nx-eyebrow">{label}</span>
      <select id={id} value={value} onChange={(e) => onChange(e.target.value)} className="nx-focus min-h-[52px] w-full rounded-xl border border-n-line-strong bg-n-surface px-3 text-[16px] font-semibold text-n-ink" data-testid={id}>
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
  const sp = useSearchParams();
  const now = useNow();
  const [purpose] = usePurpose();
  const placeKey = sp.get("place");
  const audience: Audience = sp.get("for") ? parseAudience(sp.get("for")) : defaultAudience(purpose);
  const species = speciesById(sp.get("species"));
  const place = useMemo(() => resolvePlace(placeKey, ports), [placeKey, ports]);
  const b = useBriefing({ ...props, place, audience, species, dateParam: sp.get("date"), now });
  const [home, setHomeState] = useState<string | null>(null);
  useEffect(() => {
    try {
      setHomeState(window.localStorage.getItem(HOME_KEY));
    } catch {
      setHomeState(null);
    }
  }, []);
  // the home port is remembered on this device only, and never assumed to be where anyone fishes
  const setHome = useCallback((key: string | null) => {
    try {
      if (key) window.localStorage.setItem(HOME_KEY, key);
      else window.localStorage.removeItem(HOME_KEY);
    } catch {
      // not remembered on this device
    }
    setHomeState(key);
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
  const pack = packForPort(place?.kind === "port" ? Number(place.key.slice(5)) : null) ?? PACKS.monterey;
  const off = useOfflinePack(pack.id);
  useRecordSource(off.record, pack.name, off.online, off.evicted);
  const [packOpen, setPackOpen] = useState(false);
  const hasPack = place?.kind === "port" && !!packForPort(Number(place.key.slice(5)));
  const back = typeof window === "undefined" ? `/coast?${sp.toString()}` : `${window.location.pathname}${window.location.search}`;
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
    () => (place ? { name: place.name, covering: b.assocs.filter((a) => a.tier === "geographic" && !a.dispute).map((a) => a.record.id), unsettled: b.assocs.filter((a) => a.tier === "unresolved" && !a.dispute).map((a) => a.record.id) } : null),
    [place, b.assocs],
  );
  // another place keeps the trip (activity, species) and is a new entry in the history
  const goPlace = useCallback((key: string) => router.push(coastHref({ place: key, for: audience, species: species.id }), { scroll: false }), [router, audience, species.id]);
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
  const far = measured.filter((n) => !near.includes(n));
  const notices = g ? g.nodes.filter((n) => n.variable === "official notice") : [];

  // ---------------------------------------------------------------- no place yet: choose one
  if (!place) {
    return (
      <div className="env-chart min-h-dvh bg-n-bg text-n-ink" data-testid="my-coast" data-view="landing">
        <SkipLink />
        <TopBar active="coast" now={now} notices={<NoticesButton official={official} status={sourceStatus(props.manifest, "official")} />} />
        <RegistryDisclosure className="mx-auto w-full max-w-[760px] px-4 sm:px-6" fixture={isFixture(props.manifest)} />
        <main id="main" className="mx-auto w-full max-w-[760px] px-4 pb-28 pt-6 sm:px-6">
          <p className="nx-eyebrow">My Coast</p>
          <h1 className="mt-2 text-balance font-serif text-[36px] font-semibold leading-[1.1] sm:text-[44px]" data-testid="coast-title">
            Where are you going next?
          </h1>
          <p className="mt-3 max-w-[60ch] text-[17px] leading-relaxed text-n-ink-2">One briefing for the trip: the agencies&rsquo; notices, the latest bloom and toxin samples near you, NOAA&rsquo;s model, sea conditions and what to check before you leave. Every statement shows where it came from.</p>
          {placeKey && (
            <p className="mt-4 rounded-xl border border-n-official-line bg-n-official-bg px-3 py-2 text-[15px]" role="status" data-testid="coast-unknown-place">
              The place in this link (&ldquo;{placeKey}&rdquo;) is not one Neritic knows. Choose a harbour or harvest site below.
            </p>
          )}
          <ul className="mt-6 grid gap-2 sm:grid-cols-2" data-testid="coast-landing">
            {[...(home ? [home] : []), ...EXAMPLES.filter((k) => k !== home)].map((k) => {
              const p = resolvePlace(k, ports);
              if (!p) return null;
              return (
                <li key={k}>
                  <Link href={coastHref({ place: k, for: audience })} className="nx-press nx-focus flex min-h-[64px] items-center justify-between gap-3 rounded-2xl border border-n-line bg-n-surface px-4" data-testid={`coast-start-${k}`}>
                    <span>
                      <span className="block text-[17px] font-semibold">{p.name}</span>
                      <span className="block text-[13.5px] text-n-ink-3">{k === home ? "Your home port" : p.note}</span>
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
                  <Link className="nx-focus inline-flex min-h-10 items-center hover:underline" href={coastHref({ place: p.key, for: audience })}>
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

  return (
    <div className="env-chart relative min-h-dvh bg-n-bg text-n-ink" data-testid="my-coast" data-view="trip">
      <SkipLink />
      <TopBar active="coast" now={now} notices={<NoticesButton official={official} status={sourceStatus(props.manifest, "official")} local={localNotices} />} />
      <RegistryDisclosure className="mx-auto w-full max-w-[1240px] px-4 sm:px-6" fixture={isFixture(props.manifest)} />
      {!off.online && (
        <div className="px-3 pt-3">
          {/* in the page's flow: on the briefing it must never cover the title */}
          <OfflineBanner record={off.record ?? null} name={pack.name} inline />
        </div>
      )}
      {packOpen && (
        <div className="fixed inset-0 z-50 grid place-items-center bg-[rgba(4,16,22,0.55)] p-3" onClick={(e) => e.target === e.currentTarget && setPackOpen(false)}>
          <div role="dialog" aria-modal="true" aria-label={`Field pack: ${pack.name}`} className="nx-deck nx-rise max-h-[92dvh] w-[min(520px,100%)] overflow-y-auto p-5">
            <OfflinePanel pack={pack} manifest={props.manifest} baseUrl={props.baseUrl} onClose={() => setPackOpen(false)} />
          </div>
        </div>
      )}
      <div className="mx-auto grid w-full max-w-[1240px] gap-8 px-4 pb-32 pt-4 sm:px-6 lg:grid-cols-[minmax(0,720px)_minmax(0,1fr)] lg:pt-6">
        <main id="main" className="flex min-w-0 flex-col gap-7">
          <header className="flex flex-col gap-4" data-testid="coast-header">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="nx-eyebrow">Your trip briefing</p>
                <h1 className="mt-1 text-balance font-serif text-[38px] font-semibold leading-[1.05] sm:text-[48px]" data-testid="coast-title">
                  {place.name}
                </h1>
                <p className="mt-1 text-[15.5px] text-n-ink-2" data-testid="coast-subtitle">
                  {place.note}
                  {place.county ? ` · ${place.county} County` : ""}
                </p>
              </div>
              <CopyLink />
            </div>
            <PlaceBar place={place} ports={ports} home={home} onHome={setHome} onPlace={goPlace} />
            <TripSummary audience={audience} species={species.label} day={b.tripDay ? `${b.tripDay.label}, ${b.tripDay.day}` : "Today"} open={tripOpen} onToggle={() => setTripOpen((o) => !o)} />
            <div className={`flex-col gap-2 sm:flex sm:flex-row ${tripOpen ? "flex" : "hidden"}`} data-testid="trip-controls">
              <Select id="trip-activity" label="Activity" value={audience} onChange={(v) => go({ for: v as Audience, species: v === "all" || species.activities.includes(v as never) ? species.id : "any" })} options={activityOptions} />
              <Select id="trip-species" label="Species" value={species.id} onChange={(v) => go({ species: v as SpeciesId })} options={speciesOptions} />
              <Select id="trip-day" label="Day" value={b.tripDate ?? dayOptions[0]?.id ?? ""} onChange={(v) => go({ date: v === b.days[0]?.date ? null : v })} options={dayOptions.length ? dayOptions : [{ id: "", label: "Today" }]} />
            </div>
          </header>

          <section aria-labelledby="now-h" className="flex flex-col gap-4" data-testid="brief-now">
            <h2 id="now-h" className="font-serif text-[26px] font-semibold leading-tight">
              What matters now
            </h2>
            {!g || !settled ? (
              <p className="text-[15px] text-n-ink-2" role="status" data-testid="matters-reading">
                Reading the notices, shore samples, NOAA&rsquo;s model and the satellite for {place.name}…
              </p>
            ) : (
              <ul className="flex flex-col gap-5" data-testid="matters">
                {b.matters.map((c) => (
                  <ClaimRow key={c.id} c={c} graph={g as EvidenceGraph} stations={stations} onSelect={(s) => router.push(exploreHref(s, back))} fallback={placeSel} />
                ))}
              </ul>
            )}
            <p className="text-[13.5px] leading-snug text-n-ink-3" data-testid="matters-provenance">
              Official items quote Neritic&rsquo;s copy of the CDFW and CDPH pages. No person has reviewed that copy yet, so the agency pages are the authority.
            </p>
          </section>

          <div data-testid="brief-changes">
            <SinceLastCheck />
          </div>

          {/* what to do comes right after what matters and what changed; the evidence follows (v2 order) */}
          <Section id="checks" title="Before you go" kicker={`For ${species.id === "any" ? (audience === "all" ? "any activity" : AUDIENCES.find((a) => a.id === audience)?.label.toLowerCase()) : species.label.toLowerCase()} at ${place.name}${b.tripDay ? `, ${b.tripDay.label.toLowerCase() === "today" ? "today" : `${b.tripDay.label} ${b.tripDay.day}`}` : ""}.`}>
            <p className="text-[13.5px] text-n-ink-2" data-testid="checks-progress">
              <span className="font-semibold text-n-ink">
                {b.checks.shown.filter((x) => ticked.includes(x.id)).length} of {b.checks.shown.length} ticked by you
              </span>{" "}
              · your own notes, kept on this device. A tick is not a check by Neritic or by any agency.
            </p>
            <ol className="flex flex-col gap-3" data-testid="checks">
              {b.checks.shown.map((s, i) => (
                <CheckItem key={s.id} s={s} n={i + 1} done={ticked.includes(s.id)} onToggle={() => toggleTick(s.id)} />
              ))}
            </ol>
            {/* at most five checks up front; the rest stay one tap away, never dropped */}
            {b.checks.more.length > 0 && (
              <details className="mt-3" data-testid="checks-more">
                <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center font-semibold text-n-accent">
                  {b.checks.more.length} more {b.checks.more.length === 1 ? "check" : "checks"}
                </summary>
                <ol className="mt-2 flex flex-col gap-3">
                  {b.checks.more.map((s, i) => (
                    <CheckItem key={s.id} s={s} n={b.checks.shown.length + i + 1} done={ticked.includes(s.id)} onToggle={() => toggleTick(s.id)} />
                  ))}
                </ol>
              </details>
            )}
          </Section>

          <Section id="measured" title="The evidence near you" kicker="Measurements and observations that describe these waters, with where and when each was taken.">
            {!g ? (
              <p className="text-[15px] text-n-ink-2">Reading…</p>
            ) : near.length ? (
              <ul className="divide-y divide-n-line">{near.map((n) => <NodeRow key={n.id ?? n.variable} n={n} now={now} back={back} />)}</ul>
            ) : (
              <p className="text-[15px] text-n-ink-2">Nothing measured recently in these waters themselves. What exists is farther away or older, below.</p>
            )}
          </Section>

          {far.length > 0 && (
            <Section id="context" title="Farther away or older" kicker="Context only: useful to know, but not a reading of these waters on this day.">
              <ul className="divide-y divide-n-line">{far.map((n) => <NodeRow key={n.id ?? n.variable} n={n} now={now} back={back} />)}</ul>
            </Section>
          )}

          <Section id="official" title="Official restrictions" kicker="Neritic's transcription of the agencies' notices: not verified by a person. Check the agency before you rely on any of them.">
            {notices.length ? (
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
                    <p className="mt-1 flex flex-wrap gap-x-3 text-[13px]">
                      <span className="font-semibold text-n-official">Not verified</span>
                      {n.source.url && (
                        <a href={n.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-8 items-center gap-1 font-semibold text-n-accent">
                          {n.source.name} notice <Icon name="external" className="h-3.5 w-3.5" />
                        </a>
                      )}
                    </p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-[15px] text-n-ink-2">No notice in Neritic&rsquo;s list is tied to these waters. The list may be incomplete: check CDFW and CDPH.</p>
            )}
          </Section>

          <Section id="unknown" title="What is not known" kicker="Missing evidence is said, never filled in.">
            <ul className="flex list-disc flex-col gap-2 pl-5 text-[15.5px] leading-relaxed text-n-ink-2" data-testid="unknowns">
              {b.unknowns.map((u) => (
                <li key={u}>{u}</li>
              ))}
            </ul>
          </Section>

          <footer className="flex flex-col gap-3 border-t border-n-line pt-6 text-[14px] text-n-ink-3">
            <div className="flex flex-wrap gap-2">
              <Link href={exploreHref(placeSel!, back)} className="nx-press nx-focus flex min-h-12 items-center gap-2 rounded-xl border border-n-line-strong px-4 font-semibold text-n-ink">
                <Icon name="map" /> Open in Explore
              </Link>
              <Link href={`/research?place=${encodeURIComponent(place.key)}`} className="nx-press nx-focus flex min-h-12 items-center gap-2 rounded-xl border border-n-line-strong px-4 font-semibold text-n-ink">
                <Icon name="research" /> Open in Research
              </Link>
              {hasPack && (
                <button type="button" onClick={() => setPackOpen(true)} className="nx-press nx-focus flex min-h-12 items-center gap-2 rounded-xl border border-n-line-strong px-4 font-semibold text-n-ink" data-testid="pack-open">
                  <Icon name={off.record?.complete ? "check" : "download"} /> {off.record?.complete ? "Saved for offline use" : "Save for offline use"}
                </button>
              )}
            </div>
            <p>Sources: CDFW and CDPH (Neritic&rsquo;s copy), CalHABMAP via SCCOOS, NOAA C-HARM, NOAA CoastWatch, NDBC and CO-OPS. This link reproduces this trip.</p>
            <p>Neritic is an independent research project and is not affiliated with or endorsed by NOAA, CDFW, or CDPH.</p>
          </footer>
        </main>

        <aside aria-label={`Map of ${place.name}`} className="lg:sticky lg:top-4 lg:self-start" data-testid="locator">
          <Locator place={place} g={g} ports={ports} official={official} stations={stations} back={back} onPlace={goPlace} basemapPmtiles={off.record === undefined ? undefined : !navigator.onLine && off.record?.complete ? `/offline/${pack.id}/basemap.pmtiles` : null} />
        </aside>
      </div>
      <MobileDock />
    </div>
  );
}

/** A small map that says where the evidence is; the full spatial view is Explore. */
/** `basemapPmtiles`: offline with a saved field pack, the locator draws the pack's own map, not the tile server's.
 *  Undefined until the pack has been read: the map waits for that decision (its basemap is fixed for its life,
 *  so a map made before it would ask the tile server offline). */
function Locator({ place, g, ports, official, stations, back, onPlace, basemapPmtiles }: { place: CoastPlace; g: EvidenceGraph | null; ports: PortsCollection | null; official: OfficialDataset | null; stations: StationLite[] | null; back: string; onPlace: (key: string) => void; basemapPmtiles?: string | null }) {
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
  // the map loads only when it comes near the screen (on a phone it is below the answer)
  const box = useRef<HTMLDivElement | null>(null);
  const [near, setNear] = useState(false);
  useEffect(() => {
    const el = box.current;
    if (!el || typeof IntersectionObserver === "undefined") return setNear(true);
    const io = new IntersectionObserver((es) => es.some((e) => e.isIntersecting) && setNear(true), { rootMargin: "300px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  useEffect(() => {
    map?.fitBounds([[w, s], [e, n]], { padding: 28, duration: 0, maxZoom: 11 });
  }, [map, w, s, e, n]);
  return (
    <div ref={box} className="env-instrument overflow-hidden rounded-2xl border border-n-line">
      <div className="relative h-[220px] lg:h-[420px]">
        {/* until the map has drawn: the same quiet bathymetric lines as Explore's start, never a blank box */}
        {!map && (
          <div className="absolute inset-0 grid place-items-center" data-testid="locator-loading" role="status">
            <svg className="absolute inset-0 h-full w-full text-n-accent opacity-25" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice" aria-hidden>
              {[0, 1, 2, 3, 4].map((i) => (
                <path key={i} className={i === 2 && near ? "nx-sweep" : undefined} d={`M-20 ${70 + i * 38} C 80 ${40 + i * 40}, 160 ${120 + i * 30}, 240 ${80 + i * 38} S 380 ${110 + i * 34}, 430 ${90 + i * 38}`} fill="none" stroke="currentColor" strokeWidth={i === 2 ? 1.6 : 0.8} />
              ))}
            </svg>
            <p className="relative text-[13.5px] text-n-ink-2">{near ? `Drawing ${place.name} and its stations…` : `Map of ${place.name}`}</p>
          </div>
        )}
        {near && basemapPmtiles !== undefined && <MapCanvas
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
        />}
      </div>
      <div className="flex items-center justify-between gap-3 px-3 py-2.5">
        <p className="text-[13px] text-n-ink-2" data-testid="locator-caption">
          {place.name}, the stations behind this briefing and the notices&rsquo; limits. <span className="font-semibold text-n-ink">Tap a harbour or the water to brief those waters.</span>
        </p>
        <Link href={`/?${new URLSearchParams({ sel: place.kind === "port" ? `harbor:${place.key.slice(5)}` : `point:${place.lat.toFixed(4)},${place.lon.toFixed(4)}`, from: back }).toString()}`} className="nx-focus flex min-h-11 shrink-0 items-center gap-1 rounded-lg px-2 text-[14px] font-semibold text-n-accent" data-testid="locator-explore">
          Explore <Icon name="chevronRight" className="h-4 w-4" />
        </Link>
      </div>
      <p className="px-3 pb-2 text-[11px] text-n-ink-3">© OpenStreetMap contributors · OpenMapTiles · NOAA relief. Not for navigation.</p>
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
          setTimeout(() => setDone(false), 1800);
        } catch {
          setDone(false);
        }
      }}
      aria-label={done ? "Link copied" : "Copy a link to this briefing"}
      className="nx-press nx-focus flex h-12 shrink-0 items-center gap-2 rounded-full border border-n-line-strong bg-n-surface px-4 text-[14.5px] font-semibold"
      data-testid="coast-copy-link"
    >
      <Icon name={done ? "check" : "link"} className="h-[18px] w-[18px]" />
      <span className="max-sm:sr-only">{done ? "Copied" : "Copy link"}</span>
      <span className="sr-only" aria-live="polite">
        {done ? "Link copied" : ""}
      </span>
    </button>
  );
}

function MobileDock() {
  return (
    <div className="env-instrument fixed inset-x-0 bottom-0 z-40 border-t border-n-line bg-n-surface lg:hidden" style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}>
      <NavTabs active="coast" size="dock" />
    </div>
  );
}

/** One check before going out: what to do, why, and where (an agency page, or a phone line that dials). */
function CheckItem({ s, n, done, onToggle }: { s: Step; n: number; done: boolean; onToggle: () => void }) {
  return (
    <li className={`grid grid-cols-[28px_minmax(0,1fr)_auto] gap-3 rounded-2xl border bg-n-surface p-4 ${done ? "border-n-line opacity-80" : "border-n-line"}`} data-testid={`check-${s.id}`} data-done={done}>
      <span className="nx-data grid h-7 w-7 place-items-center rounded-full bg-n-surface-2 text-[13px] font-semibold">{n}</span>
      <div className="min-w-0">
        <p className={`text-[16px] font-semibold leading-snug ${done ? "text-n-ink-2" : ""}`}>{s.title}</p>
        {/* a service's technical reason (an HTTP status) stays available on hover, not in the sentence */}
        <p className="mt-0.5 text-[14.5px] leading-snug text-n-ink-2" title={/\(HTTP \d+\)/.test(s.reason) ? s.reason : undefined}>
          {s.reason.replace(/\s*\(HTTP \d+\)/, "")}
        </p>
        <p className="mt-1.5 flex flex-wrap gap-x-3 text-[13.5px]">
          {"url" in s.source ? (
            <a href={s.source.url} target="_blank" rel="noreferrer" className="nx-focus inline-flex min-h-9 items-center gap-1 font-semibold text-n-accent">
              {s.source.label} <Icon name="external" className="h-3.5 w-3.5" />
            </a>
          ) : (
            <a href={`tel:${s.source.tel.replace(/[^\d+]/g, "")}`} className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent" data-testid="check-tel">
              {s.source.label}: <span className="nx-data">{s.source.tel}</span>
            </a>
          )}
          {s.when && <span className="inline-flex min-h-9 items-center text-n-ink-3">{s.when}</span>}
        </p>
      </div>
      {/* the reader's own acknowledgement: never a verification of the step */}
      <label className="nx-focus -m-2 grid h-14 w-14 cursor-pointer place-items-center rounded-xl hover:bg-n-surface-2" title="Tick when you have done this (your own note)">
        <input type="checkbox" checked={done} onChange={onToggle} className="h-6 w-6 accent-[var(--nx-accent)]" aria-label={`I have done this: ${s.title}`} data-testid={`check-${s.id}-box`} />
      </label>
    </li>
  );
}

/**
 * Which waters, and the reader's home port: change the place (harbours by region, CDPH harvest sites),
 * keep one harbour as home on this device, and get back to it in one tap. A point chosen on the map is
 * a place too, but not a home port.
 */
function PlaceBar({ place, ports, home, onHome, onPlace }: { place: CoastPlace; ports: PortsCollection | null; home: string | null; onHome: (k: string | null) => void; onPlace: (k: string) => void }) {
  const homePlace = home ? resolvePlace(home, ports) : null;
  const regions = ports?.regions ?? [];
  const byRegion = regions.map((r) => ({ r, items: (ports?.features ?? []).filter((f) => f.properties.region === r.id).map((f) => ({ key: `port-${f.properties.port_code}`, name: f.properties.display_name })) }));
  const sites = sitePlaces();
  const isHome = home === place.key;
  const canBeHome = place.kind !== "point";
  return (
    <div className="flex flex-wrap items-center gap-2" data-testid="coast-place-bar">
      <label className="relative inline-flex min-h-11 items-center">
        <span className="sr-only">Change the place</span>
        <select
          value=""
          onChange={(e) => e.target.value && onPlace(e.target.value)}
          className="nx-focus min-h-11 cursor-pointer appearance-none rounded-full border border-n-line-strong bg-n-surface py-2 pl-3.5 pr-9 text-[14.5px] font-semibold text-n-ink"
          data-testid="coast-place-picker"
        >
          <option value="">Change place…</option>
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
        <Icon name="chevronDown" className="pointer-events-none absolute right-3 h-4 w-4 text-n-ink-3" />
      </label>
      {isHome ? (
        <span className="inline-flex min-h-11 items-center gap-2 text-[14px] text-n-ink-2" data-testid="coast-home">
          <span className="inline-flex items-center gap-1.5 font-semibold text-n-ink">
            <Icon name="harbor" className="h-4 w-4 text-n-accent" /> Your home port
          </span>
          <button type="button" onClick={() => onHome(null)} className="nx-focus inline-flex min-h-11 items-center px-1 font-semibold text-n-accent underline underline-offset-2" data-testid="coast-home-clear">
            Forget
          </button>
        </span>
      ) : (
        <>
          {canBeHome && (
            <button type="button" onClick={() => onHome(place.key)} className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 rounded-full border border-n-line-strong px-3.5 text-[14px] font-semibold text-n-ink" data-testid="coast-home-set">
              <Icon name="harbor" className="h-4 w-4 text-n-accent" /> Make my home port
            </button>
          )}
          {homePlace && (
            <button type="button" onClick={() => onPlace(homePlace.key)} className="nx-focus inline-flex min-h-11 items-center px-1 text-[14px] font-semibold text-n-accent underline underline-offset-2" data-testid="coast-home-open">
              Home: {homePlace.name}
            </button>
          )}
        </>
      )}
      <span className="w-full text-[12.5px] text-n-ink-3">The home port is kept on this device only.</span>
    </div>
  );
}
