"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { PortsCollection } from "@/generated/schema";
import type { Audience, CoastPlace } from "@/lib/coast";
import { compass, marineHeadline, stationName, NWS_MARINE_PAGE, type Conditions as SeaConditions } from "@/lib/conditions";
import { distanceKm } from "@/lib/grid";
import { LIMITS_KEY, datePeriods, marineMaxima, overLimits, parseLimits, periodFor, type DatedPeriod, type SeaLimits } from "@/lib/trip";
import { MARK_WORDS, dollars, groundfishOn, monthDay, perPound, portAreaValues, pounds, regionFor, regionSeasons, seasonsAgeDays, speciesValues } from "@/lib/speciesSeasons";
import { CDFW_COMMERCIAL, CDFW_GROUNDFISH, CDFW_INSEASON, GROUNDFISH_UPDATED, SALMON_LINES, SEASONS_READ, SEASONS_STALE_DAYS, type SeasonItem } from "@/content/seasons";
import { CDFW_MFDE, LANDINGS } from "@/content/landings";
import { speciesPhrase, type Species } from "@/content/species";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel, kmLabel } from "@/v4/ui/time";

/**
 * The fishing side of the trip briefing (v5 fishermen audit, v6 IA): is it on CDFW's list here, what is
 * the sea doing against the reader's own limits, what has the catch been worth. Everything is quoted from
 * its agency with its date: CDFW's regional sport-fishing summary (content/seasons), NWS, NDBC and CO-OPS
 * (lib/conditions), CDFW By The Numbers (content/landings). Neritic never says a fishery is open, that it
 * is safe to go, or where fish are.
 */

const ext = { target: "_blank", rel: "noreferrer" } as const;
const C_TO_F = (c: number) => (c * 9) / 5 + 32;
const COMMERCIAL_ONLY = new Set(["squid", "urchin"]);
const NO_NUMBER = "Neritic could not read a wind or seas number from this NWS period";

/** The CDFW port area for a place: its own (a port), else the nearest port's within 60 km, named. */
export function placePortArea(place: CoastPlace, ports: PortsCollection | null): { area: string; via: string | null } | null {
  const feats = ports?.features ?? [];
  if (place.kind === "port") {
    const f = feats.find((x) => `port-${x.properties.port_code}` === place.key);
    return f?.properties.port_area ? { area: f.properties.port_area, via: null } : null;
  }
  const near = feats
    .map((f) => ({ f, km: distanceKm(place.lat, place.lon, (f.geometry.coordinates as number[])[1], (f.geometry.coordinates as number[])[0]) }))
    .filter((x) => x.km <= 60 && x.f.properties.port_area)
    .sort((a, b) => a.km - b.km)[0];
  return near ? { area: near.f.properties.port_area!, via: `${near.f.properties.display_name}, ${kmLabel(near.km)} away` } : null;
}

const titleCase = (s: string) => s.toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());

function MarkDot({ mark }: { mark: SeasonItem["mark"] }) {
  // CDFW's own colours, as a shape too (filled, half, ring) so it never rests on colour alone
  const cls = mark === "open" ? "bg-n-ok border-n-ok" : mark === "alert" ? "border-n-official bg-[linear-gradient(90deg,var(--color-n-official)_50%,transparent_50%)]" : "border-n-down bg-transparent";
  return <span aria-hidden className={`mt-[5px] inline-block h-3 w-3 shrink-0 rounded-full border-2 ${cls}`} />;
}

// ---------------------------------------------------------------- notices: one count, one wording

/** The notices tied to a place (official geography, not disputed) and how many of them name the reader's species or activity. */
export type NoticeCounts = { here: number; namingSpecies: number };

const AUDIENCE_WORDS: Record<Audience, string> = { all: "", commercial: "commercial fishing", recreational: "recreational fishing", shellfish: "shellfish harvest" };

/**
 * "3 notices tied to these waters · none name Dungeness crab": the one wording for the top bar's count,
 * the glance and the notices section, so "here" and "naming your species" are never confused, and "none
 * here" is never said when notices are tied to the waters.
 */
export function noticeLine(n: NoticeCounts, species: Species, audience: Audience): string {
  if (!n.here) return "No notice in Neritic's list is tied to these waters";
  const head = `${n.here} ${n.here === 1 ? "notice" : "notices"} tied to these waters`;
  const m = n.namingSpecies ? String(n.namingSpecies) : "none";
  if (species.id !== "any") return `${head} · ${m} ${n.namingSpecies === 1 ? "names" : "name"} ${speciesPhrase(species)}`;
  if (audience !== "all") return `${head} · ${m} for ${AUDIENCE_WORDS[audience]}`;
  return head;
}

// ---------------------------------------------------------------- the reader's own limits

export function useSeaLimits(): [SeaLimits, (l: SeaLimits) => void] {
  const [l, setL] = useState<SeaLimits>({ windKt: null, seasFt: null });
  useEffect(() => {
    try {
      setL(parseLimits(window.localStorage.getItem(LIMITS_KEY)));
    } catch {
      // not available here
    }
  }, []);
  const set = (n: SeaLimits) => {
    setL(n);
    try {
      if (n.windKt == null && n.seasFt == null) window.localStorage.removeItem(LIMITS_KEY);
      else window.localStorage.setItem(LIMITS_KEY, JSON.stringify(n));
    } catch {
      // kept for this visit only
    }
  };
  return [l, set];
}

const limitWords = (l: SeaLimits) => [l.windKt && `wind ${l.windKt} kt`, l.seasFt && `seas ${l.seasFt} ft`].filter(Boolean).join(", ");

/** A dated label for a period: its NWS name live; once saved, its own date ("Mon, Oct 12, night"), never "Tonight". */
function periodLabel(p: DatedPeriod, saved: boolean): string {
  if (!saved) return p.name;
  return p.date ? `${dayLabel(p.date, true)}${p.night ? ", night" : ""}` : "Undated period";
}

// ---------------------------------------------------------------- the trip at a glance (v7)

const DIRS = "N|NNE|NE|ENE|E|ESE|SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW|North|South|East|West|Northwest|Northeast|Southwest|Southeast|Variable";
const DIR_SHORT: Record<string, string> = { north: "N", south: "S", east: "E", west: "W", northwest: "NW", northeast: "NE", southwest: "SW", southeast: "SE", variable: "Var." };
const range = (a: string, b?: string) => (b ? `${a}–${b}` : a);

/** The NWS period's wind as the forecaster's own range ("W 5–10"), else the highest number it states; null when none. */
export function windValue(text: string, max: number | null): string | null {
  const m = text.match(new RegExp(`\\b(${DIRS})?\\s*winds?\\s+(?:(?:around|up to)\\s+)?(\\d+)(?:\\s*to\\s*(\\d+))?\\s*kt`, "i"));
  if (m) {
    const d = m[1] ? (DIR_SHORT[m[1].toLowerCase()] ?? m[1].toUpperCase()) : "";
    return `${d ? `${d} ` : ""}${range(m[2], m[3])}`;
  }
  const lt = text.match(/\bwinds?\b[^.]*?\bless than (\d+) kt/i);
  if (lt) return `< ${lt[1]}`;
  return max != null ? `≤ ${max}` : null;
}

/** The NWS period's seas as its own range ("3–5"), else the highest number it states; null when none. */
export function seasValue(text: string, max: number | null): string | null {
  const m = text.match(/\bseas?\s+(?:(?:around|up to)\s+)?(\d+(?:\.\d+)?)(?:\s*to\s*(\d+(?:\.\d+)?))?\s*ft/i);
  if (m) return range(m[1], m[2]);
  return max != null ? `≤ ${max}` : null;
}


/** The dock value in one line: the species' own CDFW fishery, else the port area, history and labelled so. */
export function worthLine(place: CoastPlace, ports: PortsCollection | null, species: Species): string {
  const value = species.id !== "any" ? (speciesValues(species.id)[0] ?? null) : null;
  const pa = species.id === "any" ? placePortArea(place, ports) : null;
  const paValue = pa ? (portAreaValues(pa.area)[0] ?? null) : null;
  const head = `Dock value, ${value?.year ?? paValue?.year ?? LANDINGS[0].year} (history): `;
  const body = value ? `${value.fishery} ~${perPound(value.perLb)} statewide average, ${dollars(value.dollars)} landed` : species.id !== "any" ? "no species-level reference" : paValue ? `${dollars(paValue.dollars)} landed, ${titleCase(pa!.area)} port area, all species` : "no CDFW figure for these waters";
  return `${head}${body}${value || paValue ? ". Not today's price." : ""}`;
}

// ---------------------------------------------------------------- the sea

export function SeaBody({
  cond,
  tripDate,
  today,
  limits,
  now,
  asSaved,
}: {
  cond: { data: SeaConditions | null; loading: boolean; error: string | null };
  tripDate: string | null;
  today: string | null;
  limits: SeaLimits;
  now: Date | null;
  asSaved: string | null;
}) {
  const c = cond.data;
  const alerts = c?.alerts.status === "ok" ? c.alerts.data : [];
  const periods = c?.marine.status === "ok" ? datePeriods(c.marine.data.periods, c.marine.data.updated) : [];
  const chosen = periodFor(c, tripDate, today);
  const buoy = c?.buoy.status === "ok" ? c.buoy : null;
  const b = buoy?.data;
  const harbor = c?.harbor?.status === "ok" ? c.harbor : null;
  const day = tripDate ?? today;
  const tides = c?.tides.status === "ok" ? c.tides.data.filter((t) => (day ? t.at.startsWith(day) : true)).slice(0, 4) : [];
  const hm = (t: string) => {
    const [h, m] = t.split(" ")[1].split(":").map(Number);
    return `${((h + 11) % 12) + 1}:${String(m).padStart(2, "0")} ${h < 12 ? "AM" : "PM"}`;
  };
  const hasLimits = limits.windKt != null || limits.seasFt != null;
  const isChosen = (p: DatedPeriod) => chosen?.name === p.name && chosen?.date === p.date;
  // the trip day's period first, then the rest in the forecaster's order; every word shown (this note is the disclosure)
  const ordered = [...periods.filter(isChosen), ...periods.filter((p) => !isChosen(p))];
  return (
    <div className="flex flex-col gap-3">
      <p className="text-[13.5px] leading-snug text-n-ink-2" data-testid="sea-limits">
        {hasLimits ? <>Each period against your own limits ({limitWords(limits)}), set above. </> : "Set your own limits above to compare them with each period. "}
        Wind, waves and tides in the forecaster&rsquo;s words: never a go or no-go from Neritic.
      </p>
      {cond.loading && <p className="text-[15px] text-n-ink-2" role="status">Reading NWS, NDBC and CO-OPS…</p>}
      {cond.error && (
        <p className="text-[15px] text-n-ink-2">
          Conditions are unavailable right now.{" "}
          <a href={NWS_MARINE_PAGE} {...ext} className="nx-focus underline underline-offset-4">
            NWS marine forecasts
          </a>
        </p>
      )}
      {alerts.map((a) => (
        <p key={a.url} className="flex items-start gap-2 border-l-2 border-n-conditions pl-3 text-[15px]" data-testid="sea-alert">
          <Icon name="alert" className="mt-0.5 h-5 w-5 shrink-0 text-n-conditions" />
          <span>
            <span className="font-medium">NWS {a.event}</span>
            {a.ends ? <span className="text-n-ink-2"> · until {clockLabel(a.ends)}</span> : null}
            <span className="block text-[13.5px] text-n-ink-2">{a.headline}</span>
          </span>
        </p>
      ))}
      {c?.marine.status === "ok" && !chosen && (
        <p className="text-[14.5px] text-n-ink-2" data-testid="sea-no-period">
          {asSaved ? "No NWS period saved for this day." : "No NWS period for this day yet."}
        </p>
      )}
      {ordered.length > 0 && (
        <ol className="divide-y divide-n-line border-y border-n-line" data-testid="sea-periods">
          {ordered.map((p) => {
            const m = marineMaxima(p.text);
            const o = overLimits(m, limits);
            const mine = isChosen(p);
            return (
              <li key={`${p.name}-${p.date}`} className={`py-2.5 ${mine ? "border-l-2 border-n-ink pl-2.5" : ""}`} data-testid="sea-period" data-chosen={mine}>
                <p className="flex flex-wrap items-baseline justify-between gap-x-2">
                  <span className="nx-caps text-n-conditions">{periodLabel(p, !!asSaved)}{mine ? " · your trip" : ""}</span>
                  <span className="font-data text-[13px] text-n-ink-2">
                    {m.windKt != null ? `≤ ${m.windKt} kt` : ""}
                    {m.windKt != null && m.seasFt != null ? " · " : ""}
                    {m.seasFt != null ? `≤ ${m.seasFt} ft` : ""}
                  </span>
                </p>
                <p className="mt-0.5 text-[15px] font-medium leading-snug">{marineHeadline(p)}</p>
                {m.windKt == null && m.seasFt == null && <p className="mt-0.5 text-[13.5px] text-n-ink-2">{NO_NUMBER}.</p>}
                {(o.wind || o.seas) && (
                  <p className="mt-0.5 text-[13.5px] font-medium text-n-ink" data-testid="sea-over">
                    The forecast&rsquo;s {[o.wind && `${m.windKt} kt`, o.seas && `${m.seasFt} ft`].filter(Boolean).join(" and ")} {o.wind && o.seas ? "are" : "is"} above your own limit
                  </p>
                )}
                <p className="mt-1 text-[13.5px] leading-relaxed text-n-ink-2">{p.text}</p>
              </li>
            );
          })}
        </ol>
      )}
      {c?.marine.status === "ok" && (
        <p className="text-[13px] text-n-ink-2">
          NWS {c.marine.data.zoneName} ({c.marine.data.zone}){c.marine.data.updated ? `, issued ${clockLabel(c.marine.data.updated)}` : ""}.{" "}
          <a href={c.marine.url} {...ext} className="nx-focus underline underline-offset-4">
            NWS forecast <Icon name="external" className="inline h-3 w-3" />
          </a>
        </p>
      )}
      {c && c.marine.status !== "ok" && !cond.loading && (
        <p className="text-[14px] text-n-ink-2">
          No NWS zone forecast is read for these waters.{" "}
          <a href={c.marine.url ?? NWS_MARINE_PAGE} {...ext} className="nx-focus underline underline-offset-4">
            NWS marine forecasts
          </a>
        </p>
      )}

      {(b || harbor || tides.length > 0) && (
        <div data-testid="sea-measured">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
            {b?.windKt && (
              <div>
                <dt className="nx-caps text-n-conditions">Wind, measured</dt>
                <dd className="font-data text-[16px]">
                  {b.windDirDeg ? `${compass(b.windDirDeg.value)} ` : ""}
                  {Math.round(b.windKt.value)} kt{b.gustKt ? `, gusts ${Math.round(b.gustKt.value)}` : ""}
                </dd>
              </div>
            )}
            {b?.waveFt && (
              <div>
                <dt className="nx-caps text-n-conditions">Waves, measured</dt>
                <dd className="font-data text-[16px]">
                  {b.waveFt.value.toFixed(1)} ft{b.periodS ? ` at ${Math.round(b.periodS.value)} s` : ""}
                </dd>
              </div>
            )}
            {(harbor?.data.waterC ?? b?.waterC) && (
              <div>
                <dt className="nx-caps text-n-conditions">{harbor ? "Water, harbour" : "Water, at the buoy"}</dt>
                <dd className="font-data text-[16px]">{Math.round(C_TO_F((harbor?.data.waterC ?? b!.waterC!).value))}°F</dd>
              </div>
            )}
            {tides.length > 0 && (
              <div className="col-span-2 sm:col-span-1">
                <dt className="nx-caps text-n-conditions">Tides{day ? `, ${dayLabel(day)}` : ""} (predicted)</dt>
                <dd className="font-data text-[14px] leading-snug">{tides.map((t) => `${t.type === "H" ? "H" : "L"} ${t.ft.toFixed(1)} ft ${hm(t.at)}`).join(" · ")}</dd>
              </div>
            )}
          </dl>
          <p className="mt-2 text-[12.5px] leading-snug text-n-ink-2">
            {buoy?.station ? `${stationName(buoy.station.name)} (NDBC ${buoy.station.id}), ${kmLabel(buoy.station.km)} away` : ""}
            {b?.waveFt && now ? `, ${clockLabel(b.waveFt.at)}` : ""}
            {harbor?.station ? ` · harbour water at ${stationName(harbor.station.name)} (CO-OPS)${harbor.data.waterC ? `, ${clockLabel(harbor.data.waterC.at)}` : ""}` : ""}
            {c?.tides.status === "ok" && c.tides.station ? ` · tides at ${stationName(c.tides.station.name)}, feet above MLLW` : ""}
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- seasons

/** One CDFW entry: its mark (a shape too, never colour alone), its words, and CDFW's sentence when asked for. */
function SeasonRow({ i, quote = true, extra, commercial = false }: { i: SeasonItem; quote?: boolean; extra?: React.ReactNode; commercial?: boolean }) {
  return (
    <li className="flex items-start gap-2.5 py-2" data-testid={`season-${i.id}`} data-mark={i.mark}>
      <MarkDot mark={i.mark} />
      <span className="min-w-0 flex-1">
        <span className="text-[15px] font-medium">{i.name}</span>
        <span className="block text-[13px] text-n-ink-2">
          {MARK_WORDS[i.mark]}
          {commercial ? " (sport list)" : ""}
        </span>
        {quote && i.quote && <span className="mt-1 block border-l-2 border-n-official-line pl-3 text-[14px] leading-relaxed text-n-ink" data-testid="season-quote">&ldquo;{i.quote}&rdquo;</span>}
        {extra}
      </span>
    </li>
  );
}

export function SeasonsBody({ place, species, audience, tripDate, today }: { place: CoastPlace; species: Species; audience: Audience; tripDate: string | null; today: string | null }) {
  const region = regionFor(place.lat);
  if (!region) return <p className="text-[15px] text-n-ink-2">These waters are outside CDFW&rsquo;s five sport-fishing regions.</p>;
  const rs = regionSeasons(region, species.id);
  const day = tripDate ?? today;
  const gf = day ? groundfishOn(region.id, day) : null;
  const age = today ? seasonsAgeDays(today) : 0;
  const stale = age > SEASONS_STALE_DAYS;
  const commercialOnly = COMMERCIAL_ONLY.has(species.id);
  const commercial = audience === "commercial";
  const isGroundfish = (i: SeasonItem) => ["rockfish", "lingcod", "cabezon", "greenlings"].includes(i.id);
  const gfExtra = gf ? (
    <span className="mt-1 block bg-n-surface-2 px-3 py-2 text-[13.5px]" data-testid="season-groundfish-row">
      CDFW&rsquo;s {region.label} groundfish table, {monthDay(gf.from)}–{monthDay(gf.to)} {gf.year}: nearshore &ldquo;{gf.nearshore}&rdquo;; shelf, slope and lingcod &ldquo;{gf.shelf}&rdquo;.{" "}
      <a href={CDFW_GROUNDFISH.url} {...ext} className="nx-focus underline underline-offset-4">
        Table, updated {GROUNDFISH_UPDATED} <Icon name="external" className="inline h-3 w-3" />
      </a>
    </span>
  ) : null;
  return (
    <div className="flex flex-col gap-3">
      <p className="text-[13.5px] text-n-ink-2" data-testid="seasons-source">
        CDFW sport summary, <span className="text-n-ink">{region.label}</span> region ({region.bounds}), dated {region.updated}; Neritic&rsquo;s copy of {dayLabel(SEASONS_READ)}.
      </p>
      {stale && (
        <p className="border-l-2 border-n-official pl-3 text-[14.5px]" data-testid="seasons-stale">
          Neritic&rsquo;s copy is {age} days old. Seasons open and close in that time: read CDFW&rsquo;s page first.
        </p>
      )}
      {(commercial || commercialOnly) && (
        <p className="text-[14.5px] text-n-ink-2" data-testid="seasons-commercial">
          This is CDFW&rsquo;s list for sport fishing. Commercial seasons, openers, permits, quotas, and in-season gear, depth and toxin actions are set separately, and Neritic holds none of them:{" "}
          <a href={CDFW_COMMERCIAL.url} {...ext} className="nx-focus underline underline-offset-4">
            {CDFW_COMMERCIAL.label}
          </a>{" "}
          and{" "}
          <a href={CDFW_INSEASON.url} {...ext} className="nx-focus underline underline-offset-4">
            in-season changes
          </a>
          .
        </p>
      )}
      {rs.mine.length > 0 && (
        <div data-testid="seasons-mine">
          <p className="nx-caps">For {speciesPhrase(species)}</p>
          {/* a commercial reader's own row never quotes the sport list's dates as a season (v6 P0 2) */}
          <ul className="divide-y divide-n-line">
            {rs.mine.map((i) => (
              <SeasonRow key={i.id} i={i} quote={!commercial} commercial={commercial} extra={isGroundfish(i) ? gfExtra : null} />
            ))}
          </ul>
        </div>
      )}
      {species.id !== "any" && !rs.mine.length && !commercialOnly && <p className="text-[14.5px] text-n-ink-2">CDFW&rsquo;s {region.label} summary has no entry of its own for {speciesPhrase(species)}.</p>}
      {species.id === "salmon" && (
        <p className="flex flex-wrap gap-x-4 text-[14px]" data-testid="seasons-salmon-lines">
          {SALMON_LINES.map((l) => (
            <a key={l.tel} href={`tel:${l.tel}`} className="nx-focus inline-flex min-h-11 items-center gap-1 text-n-ink">
              {l.label}: <span className="font-data">{l.phone}</span>
            </a>
          ))}
        </p>
      )}
      <div data-testid="seasons-all">
        <p className="nx-caps">
          Every entry on CDFW&rsquo;s {region.label} page · {rs.counts.open} open, {rs.counts.alert} alert, {rs.counts.closed} closed
        </p>
        <ul className="grid grid-cols-1 gap-x-6 sm:grid-cols-2" data-testid="seasons-headline">
          {rs.all.map((i) => (
            <SeasonRow key={i.id} i={i} quote={false} extra={isGroundfish(i) && rs.mine.every((m) => m.id !== i.id) ? gfExtra : null} />
          ))}
        </ul>
      </div>
      <p className="flex flex-wrap gap-x-4 text-[14px]">
        <a href={region.url} {...ext} className="nx-focus inline-flex min-h-11 items-center gap-1 underline underline-offset-4">
          CDFW {region.label} region page <Icon name="external" className="h-3.5 w-3.5" />
        </a>
        <a href={CDFW_INSEASON.url} {...ext} className="nx-focus inline-flex min-h-11 items-center gap-1 underline underline-offset-4">
          {CDFW_INSEASON.label} <Icon name="external" className="h-3.5 w-3.5" />
        </a>
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- what it was worth

export function WorthBody({ place, ports, species }: { place: CoastPlace | null; ports: PortsCollection | null; species: Species }) {
  const pa = place ? placePortArea(place, ports) : null;
  const rows = pa ? portAreaValues(pa.area) : [];
  const sv = speciesValues(species.id);
  const top = LANDINGS[0];
  return (
    <div className="flex flex-col gap-4" data-testid="worth">
      {species.id !== "any" &&
        (sv.length ? (
          <div data-testid="worth-species">
            <p className="nx-eyebrow">{species.label}: dock value, statewide (history)</p>
            <ul className="mt-1 divide-y divide-n-line">
              {sv.map((v) => (
                <li key={`${v.year}-${v.fishery}`} className="flex flex-wrap items-baseline justify-between gap-x-3 py-2">
                  <span className="text-[15px]">
                    <span className="font-semibold">{v.year}</span> · {v.fishery} · #{v.rank} by value
                  </span>
                  <span className="nx-data text-[15.5px] font-semibold">
                    {dollars(v.dollars)} · {pounds(v.pounds)} · ~{perPound(v.perLb)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="text-[14.5px] text-n-ink-2" data-testid="worth-species-none">
            No species-level reference: {species.label} has no fishery of its own in CDFW&rsquo;s statewide top ten by value for 2023 or 2024, so Neritic has no figure for it (and never shows a neighbouring fishery&rsquo;s).{" "}
            <a href={CDFW_MFDE.url} {...ext} className="nx-focus font-semibold text-n-accent underline">
              {CDFW_MFDE.label}
            </a>
          </p>
        ))}
      {species.id === "any" && (
        <div data-testid="worth-top">
          <p className="nx-eyebrow">Top fisheries by value, statewide, {top.year}</p>
          <ol className="mt-1 divide-y divide-n-line">
            {top.topByValue.slice(0, 5).map((f, i) => (
              <li key={f.fishery} className="flex flex-wrap items-baseline justify-between gap-x-3 py-2">
                <span className="text-[15px]">
                  {i + 1}. {f.fishery}
                </span>
                <span className="nx-data text-[15.5px] font-semibold">
                  {dollars(f.dollars)} · ~{perPound(f.dollars / f.pounds)}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
      {pa && rows.length > 0 && (
        <div data-testid="worth-port-area">
          <p className="nx-eyebrow">
            {titleCase(pa.area)} port area, all species{pa.via ? ` (nearest port: ${pa.via})` : ""}
          </p>
          <ul className="mt-1 divide-y divide-n-line">
            {rows.map((r) => (
              <li key={r.year} className="flex flex-wrap items-baseline justify-between gap-x-3 py-2">
                <span className="text-[15px]">
                  <span className="font-semibold">{r.year}</span> · #{r.rank} of CDFW&rsquo;s {r.of} port areas
                </span>
                <span className="nx-data text-[15.5px] font-semibold">
                  {dollars(r.dollars)} · {pounds(r.pounds)} · {Math.round(r.share * 100)}% of the state
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
      <p className="text-[13px] leading-snug text-n-ink-3">
        Ex-vessel value is, in CDFW&rsquo;s words, &ldquo;the amount paid to the fishermen at the dock&rdquo;: calendar-year totals from{" "}
        {LANDINGS.map((r, i) => (
          <span key={r.year}>
            {i ? " and " : ""}
            <a href={r.url} {...ext} className="nx-focus font-semibold text-n-accent">
              {r.label}
            </a>{" "}
            ({r.asOf})
          </span>
        ))}
        . History, not this season&rsquo;s prices or today&rsquo;s dock quote. The per-pound figure is Neritic&rsquo;s arithmetic, CDFW&rsquo;s dollars divided by CDFW&rsquo;s pounds, an average over a year of different boats, grades and buyers.{" "}
        <Link href="/fisheries" className="nx-focus font-semibold text-n-accent">
          Ten years of statewide history
        </Link>
      </p>
    </div>
  );
}
