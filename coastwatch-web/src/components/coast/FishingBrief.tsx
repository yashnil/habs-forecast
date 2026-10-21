"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { PortsCollection } from "@/generated/schema";
import type { Audience, CoastPlace } from "@/lib/coast";
import { compass, marineHeadline, stationName, NWS_MARINE_PAGE, type Conditions as SeaConditions } from "@/lib/conditions";
import { distanceKm } from "@/lib/grid";
import { LIMITS_KEY, datePeriods, marineMaxima, overLimits, parseLimits, periodFor, type DatedPeriod, type SeaLimits } from "@/lib/trip";
import { MARK_SHORT, MARK_WORDS, dollars, groundfishOn, monthDay, perPound, portAreaValues, pounds, regionFor, regionSeasons, seasonsAgeDays, speciesValues } from "@/lib/speciesSeasons";
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

/** The two numbers, edited in place (the glance row and the Sea section share it). */
function LimitsEditor({ limits, onLimits, idPrefix }: { limits: SeaLimits; onLimits: (l: SeaLimits) => void; idPrefix: string }) {
  const num = (v: string) => (v.trim() === "" ? null : Number(v) > 0 ? Number(v) : null);
  const box = "nx-focus min-h-12 w-24 rounded-xl border border-n-line-strong bg-n-surface px-3 text-[16px] text-n-ink";
  return (
    <div className="flex flex-wrap gap-3">
      <label className="flex flex-col gap-1 text-[13.5px] font-semibold text-n-ink-2">
        Wind above (kt)
        <input type="number" inputMode="numeric" min={1} max={80} defaultValue={limits.windKt ?? ""} onChange={(e) => onLimits({ ...limits, windKt: num(e.target.value) })} className={box} data-testid={`${idPrefix}-limit-wind`} />
      </label>
      <label className="flex flex-col gap-1 text-[13.5px] font-semibold text-n-ink-2">
        Seas above (ft)
        <input type="number" inputMode="decimal" min={1} max={40} step="0.5" defaultValue={limits.seasFt ?? ""} onChange={(e) => onLimits({ ...limits, seasFt: num(e.target.value) })} className={box} data-testid={`${idPrefix}-limit-seas`} />
      </label>
    </div>
  );
}

/** A dated label for a period: its NWS name live; once saved, its own date ("Mon, Oct 12, night"), never "Tonight". */
function periodLabel(p: DatedPeriod, saved: boolean): string {
  if (!saved) return p.name;
  return p.date ? `${dayLabel(p.date, true)}${p.night ? ", night" : ""}` : "Undated period";
}

/** The water temperature a briefing reads, with where and when it was measured. */
function waterReading(c: SeaConditions | null): { f: number; at: string; where: string } | null {
  if (c?.harbor?.status === "ok") return { f: Math.round(C_TO_F(c.harbor.data.waterC.value)), at: c.harbor.data.waterC.at, where: c.harbor.station ? `${stationName(c.harbor.station.name)} (CO-OPS), ${kmLabel(c.harbor.station.km)} away` : "the harbour" };
  if (c?.buoy.status === "ok" && c.buoy.data.waterC) return { f: Math.round(C_TO_F(c.buoy.data.waterC.value)), at: c.buoy.data.waterC.at, where: c.buoy.station ? `NDBC ${c.buoy.station.id}, ${kmLabel(c.buoy.station.km)} away` : "the buoy" };
  return null;
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

/** The highest number of a value like "5–10", to say when the text goes higher (gusts, building seas). */
const top = (v: string | null) => (v ? Number(v.match(/(\d+(?:\.\d+)?)(?!.*\d)/)?.[1] ?? NaN) : NaN);

function Sounding({ label, value, unit, note, testId }: { label: string; value: string | null; unit?: string; note: React.ReactNode; testId?: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5" data-testid={testId}>
      <span className="text-[13px] font-semibold leading-4 text-n-ink-2">{label}</span>
      <span className={`nx-sounding-v whitespace-nowrap ${value ? "" : "!text-n-ink-3"}`}>
        {value ?? "—"}
        {value && unit && <span className="ml-1 text-[15px] font-semibold text-n-ink-2">{unit}</span>}
      </span>
      <span className="nx-note-t">{note}</span>
    </div>
  );
}

/**
 * Conditions (v7): four soundings, wind and seas from the NWS period for the trip day (a forecast, said
 * so), waves and water as measured, then one quiet source line and the reader's own limits compared with
 * the NWS numbers. Never a go or no-go.
 */
export function TripConditions({
  cond,
  tripDate,
  today,
  limits,
  onLimits,
  asSaved,
  now = null,
}: {
  cond: { data: SeaConditions | null; loading: boolean; error: string | null };
  tripDate: string | null;
  today: string | null;
  limits: SeaLimits;
  onLimits: (l: SeaLimits) => void;
  /** "as saved Oct 19, 3:07 PM PDT" when the page is a saved field-pack copy, else null */
  asSaved: string | null;
  now?: Date | null;
}) {
  const c = cond.data;
  // a reading from today says its time; an older one (a saved copy) says its date too
  const when = (iso: string) => {
    const full = clockLabel(iso);
    if (!now) return full;
    const [d, ...rest] = full.split(", ");
    return d === clockLabel(now.toISOString()).split(", ")[0] && rest.length ? rest.join(", ") : full;
  };
  const period = periodFor(c, tripDate, today);
  const alerts = c?.alerts.status === "ok" ? c.alerts.data : [];
  const max = period ? marineMaxima(period.text) : null;
  const over = max ? overLimits(max, limits) : { wind: false, seas: false };
  const wind = period ? windValue(period.text, max?.windKt ?? null) : null;
  const seas = period ? seasValue(period.text, max?.seasFt ?? null) : null;
  const buoy = c?.buoy.status === "ok" ? c.buoy : null;
  const wave = buoy?.data.waveFt ?? null;
  const water = waterReading(c);
  const hasLimits = limits.windKt != null || limits.seasFt != null;
  const [editing, setEditing] = useState(false);
  const fcNote = period ? "Forecast" : cond.loading ? "Reading…" : "No forecast";
  const higher = (v: string | null, m: number | null | undefined, unit: string) => (m != null && Number.isFinite(top(v)) && m > top(v) ? `, to ${m} ${unit}` : "");
  return (
    <div className="flex flex-col gap-3" data-testid="glance-sea">
      {alerts.length > 0 && (
        <p className="inline-flex items-center gap-1.5 text-[15px] font-semibold text-n-official" data-testid="glance-alert">
          <Icon name="alert" className="h-4 w-4" /> {alerts[0].event}
        </p>
      )}
      <div className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
        <Sounding label="Wind" value={wind} unit="kt" note={`${fcNote}${higher(wind, max?.windKt, "kt")}`} testId="glance-wind" />
        <Sounding label="Seas" value={seas} unit="ft" note={`${fcNote}${higher(seas, max?.seasFt, "ft")}`} testId="glance-seas" />
        <Sounding label="Waves" value={wave ? wave.value.toFixed(1) : null} unit={wave ? `ft${buoy?.data.periodS ? ` · ${Math.round(buoy.data.periodS.value)} s` : ""}` : undefined} note={wave ? "Measured" : cond.loading ? "Reading…" : "Not measured nearby"} testId="glance-waves" />
        <Sounding label="Water" value={water ? String(water.f) : null} unit="°F" note={water ? "Measured" : cond.loading ? "Reading…" : "Not measured nearby"} testId="glance-water-t" />
      </div>
      <p className="nx-note-t !text-[12.5px] leading-[18px] text-n-ink-3">
        <a href="#sea-h" className="nx-focus rounded font-semibold text-n-ink-2 underline decoration-n-line-strong underline-offset-2 hover:text-n-ink" data-testid="glance-sea-link">
          {period ? `Marine forecast (NWS), ${periodLabel(period, !!asSaved)}` : cond.loading ? "Reading the NWS forecast…" : c?.marine.status === "ok" ? (asSaved ? "No NWS period saved for this day" : "No NWS period for this day yet") : "NWS zone forecast not available here"}
        </a>
        {period && max && max.windKt == null && max.seasFt == null && <span data-testid="glance-sea-unread"> · {NO_NUMBER}</span>}
        {/* a phone keeps the one source link; where and when each value was measured is in the Sea note */}
        {wave && buoy && (
          <span className="max-sm:sr-only">
            {" "}
            · Waves measured {when(wave.at)} at buoy {buoy.station ? `${buoy.station.id}, ${kmLabel(buoy.station.km)}` : ""}
          </span>
        )}
        {water && (
          <span className="max-sm:sr-only" data-testid="glance-water">
            {" "}
            · Water measured {when(water.at)} at {water.where}
          </span>
        )}
      </p>
      <div className="flex flex-wrap items-center gap-x-2 text-[14px] text-n-ink-2">
        <span>
          <span className="font-semibold text-n-ink">Your limits</span>: {hasLimits ? limitWords(limits) : "not set"}
          {(over.wind || over.seas) && (
            <span className="font-semibold text-n-official" data-testid="glance-over">
              {" "}
              · NWS states more than your own {[over.wind && `${limits.windKt} kt`, over.seas && `${limits.seasFt} ft`].filter(Boolean).join(" and ")} limit
            </span>
          )}
        </span>
        <button type="button" aria-expanded={editing} onClick={() => setEditing((e) => !e)} className="nx-press nx-focus inline-flex min-h-11 items-center rounded-lg px-1.5 font-semibold text-n-accent" data-testid="glance-limits-set">
          {editing ? "Done" : hasLimits ? "Change" : "Set"}
        </button>
      </div>
      {editing && <LimitsEditor limits={limits} onLimits={onLimits} idPrefix="glance" />}
    </div>
  );
}

/** The unsettled count beside the one notice count: "+1 status unclear" (never folded into "here"). */
export function OfficialPosition({
  place,
  species,
  audience,
  tripDate,
  today,
  notices,
  unclear,
}: {
  place: CoastPlace;
  species: Species;
  audience: Audience;
  tripDate: string | null;
  today: string | null;
  notices: NoticeCounts;
  unclear: number;
}) {
  const region = regionFor(place.lat);
  const rs = region ? regionSeasons(region, species.id) : null;
  const mine = rs?.mine[0] ?? null;
  const day = tripDate ?? today;
  const gf = species.id === "groundfish" && region && day ? groundfishOn(region.id, day) : null;
  const commercialOnly = COMMERCIAL_ONLY.has(species.id);
  const commercial = audience === "commercial";
  // CDFW's own sentence, when its sport section gives a reopening date. Never for a commercial reader:
  // the sport list's date is not the commercial opener (v6 P0 2).
  const reopen = !commercial ? (mine?.quote.match(/scheduled to reopen on ([A-Z][a-z]+ \d{1,2}, \d{4})/)?.[1] ?? null) : null;
  const link = "nx-focus font-semibold text-n-accent underline decoration-n-accent/40 underline-offset-2";
  return (
    <div className="mt-3 flex gap-3" data-testid="glance-official">
      <span aria-hidden className="nx-tick" data-family="official" />
      <div className="min-w-0 flex-1">
        <p className="text-[16.5px] leading-[25px] text-n-ink">
          <a href="#seasons-h" className="nx-focus rounded font-semibold hover:text-n-accent" data-testid="glance-season">
            {mine ? (
              <>
                {mine.name}
                {commercial ? (
                  <span className="font-normal">: CDFW&rsquo;s sport list only.</span>
                ) : (
                  <span className="font-normal" data-testid="glance-season-list">
                    {" "}
                    is on CDFW&rsquo;s {MARK_SHORT[mine.mark].toLowerCase()}.
                  </span>
                )}
              </>
            ) : rs ? (
              commercialOnly ? `${species.label}: commercial only, not on CDFW's sport list.` : species.id === "any" ? `CDFW ${region!.label}: ${rs.counts.open + rs.counts.alert} on the open list, ${rs.counts.closed} closed.` : `${species.label}: no entry of its own on CDFW's ${region!.label} list.`
            ) : (
              "Outside CDFW's five sport-fishing regions."
            )}
          </a>
          {mine && !commercial && (reopen || gf) && <span className="text-n-ink-2"> {reopen ? `CDFW: “scheduled to reopen on ${reopen}”` : gf ? `Groundfish ${monthDay(gf.from)}–${monthDay(gf.to)}: ${gf.nearshore}` : null}</span>}
          {(commercial || commercialOnly) && rs && (
            <span className="text-n-ink-2" data-testid="glance-commercial">
              {" "}
              Commercial opener: see{" "}
              <a href={CDFW_COMMERCIAL.url} {...ext} className={link}>
                CDFW commercial
              </a>{" "}
              /{" "}
              <a href={CDFW_INSEASON.url} {...ext} className={link}>
                in-season page
              </a>
            </span>
          )}
        </p>
        <a href="#official-h" className="nx-focus mt-1 inline-flex min-h-8 items-center rounded text-[14px] text-n-ink-2 hover:text-n-ink" data-testid="glance-notices">
          {noticeLine(notices, species, audience)}
          {unclear > 0 && ` · +${unclear} status unclear`}
          <Icon name="chevronRight" className="ml-0.5 h-3.5 w-3.5 shrink-0" />
        </a>
      </div>
    </div>
  );
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
  onLimits,
  now,
  asSaved,
}: {
  cond: { data: SeaConditions | null; loading: boolean; error: string | null };
  tripDate: string | null;
  today: string | null;
  limits: SeaLimits;
  onLimits: (l: SeaLimits) => void;
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
  const [editing, setEditing] = useState(false);
  const hasLimits = limits.windKt != null || limits.seasFt != null;
  const isChosen = (p: DatedPeriod) => chosen?.name === p.name && chosen?.date === p.date;
  const chosenP = periods.find(isChosen) ?? null;
  const others = periods.filter((p) => !isChosen(p));
  const renderPeriod = (p: DatedPeriod) => {
    const m = marineMaxima(p.text);
    const o = overLimits(m, limits);
    const mine = isChosen(p);
    return (
      <li key={`${p.name}-${p.date}`} className={`py-2.5 ${mine ? "border-l-2 border-n-accent pl-2.5" : ""}`} data-testid="sea-period" data-chosen={mine}>
        <p className="flex flex-wrap items-baseline justify-between gap-x-2">
          <span className="text-[13px] font-semibold text-n-ink-2">{periodLabel(p, !!asSaved)}</span>
          <span className="nx-data text-[13px] text-n-ink-2">
            {m.windKt != null ? `≤ ${m.windKt} kt` : ""}
            {m.windKt != null && m.seasFt != null ? " · " : ""}
            {m.seasFt != null ? `≤ ${m.seasFt} ft` : ""}
          </span>
        </p>
        <p className="mt-0.5 text-[15.5px] font-semibold leading-snug">{marineHeadline(p)}</p>
        {m.windKt == null && m.seasFt == null && <p className="mt-0.5 text-[13.5px] text-n-ink-2">{NO_NUMBER}.</p>}
        {(o.wind || o.seas) && (
          <p className="mt-0.5 text-[13.5px] font-semibold text-n-official" data-testid="sea-over">
            The forecast&rsquo;s {[o.wind && `${m.windKt} kt`, o.seas && `${m.seasFt} ft`].filter(Boolean).join(" and ")} {o.wind && o.seas ? "are" : "is"} above your own limit
          </p>
        )}
        <details>
          <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center text-[13.5px] font-semibold text-n-accent">Full NWS text</summary>
          <p className="text-[14px] leading-relaxed text-n-ink-2">{p.text}</p>
        </details>
      </li>
    );
  };
  return (
    <div className="flex flex-col gap-3">
      <div>
        {/* the reader's own limits lead the section: the one comparison Neritic makes */}
        <div className="flex flex-wrap items-center gap-x-2 text-[14.5px]" data-testid="sea-limits">
          <span>
            <span className="font-semibold">Your own limits</span>
            <span className="text-n-ink-2">: {hasLimits ? limitWords(limits) : "not set"}</span>
          </span>
          <button type="button" onClick={() => setEditing((e) => !e)} aria-expanded={editing} className="nx-press nx-focus min-h-11 rounded-xl px-2 text-[14.5px] font-semibold text-n-accent" data-testid="sea-limits-edit">
            {editing ? "Done" : hasLimits ? "Change" : "Set"}
          </button>
        </div>
        {editing && <LimitsEditor limits={limits} onLimits={onLimits} idPrefix="sea" />}
        <p className="mt-1 text-[13px] leading-snug text-n-ink-3">Wind, waves and tides in the forecaster&rsquo;s words: never a go or no-go from Neritic. It only compares your limits with the highest numbers the NWS text states.</p>
      </div>
      {cond.loading && <p className="text-[15px] text-n-ink-2" role="status">Reading NWS, NDBC and CO-OPS…</p>}
      {cond.error && (
        <p className="text-[15px] text-n-ink-2">
          Conditions are unavailable right now.{" "}
          <a href={NWS_MARINE_PAGE} {...ext} className="nx-focus font-semibold text-n-accent underline">
            NWS marine forecasts
          </a>
        </p>
      )}
      {alerts.map((a) => (
        <p key={a.url} className="flex items-start gap-2 rounded-xl border border-n-official-line bg-n-official-bg px-3 py-2.5 text-[15px]" data-testid="sea-alert">
          <Icon name="alert" className="mt-0.5 h-5 w-5 shrink-0 text-n-official" />
          <span>
            <span className="font-semibold">{a.event}</span>
            {a.ends ? <span className="text-n-ink-2"> · until {clockLabel(a.ends)}</span> : null}
            <span className="block text-[13.5px] text-n-ink-2">{a.headline}</span>
          </span>
        </p>
      ))}
      {c?.marine.status === "ok" && !chosen && (
        <p className="text-[14.5px] font-semibold text-n-ink-2" data-testid="sea-no-period">
          {asSaved ? "No NWS period saved for this day." : "No NWS period for this day yet."}
        </p>
      )}
      {/* the trip day's period first; the forecast's other periods one tap away */}
      {chosenP && <ol className="border-y border-n-line" data-testid="sea-periods">{renderPeriod(chosenP)}</ol>}
      {others.length > 0 && (
        <details data-testid="sea-periods-more">
          <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center text-[14.5px] font-semibold text-n-accent">
            {chosenP ? `${others.length} more NWS ${others.length === 1 ? "period" : "periods"}` : `The ${others.length} NWS ${others.length === 1 ? "period" : "periods"} ${asSaved ? "saved" : "issued"}`}
          </summary>
          <ol className="divide-y divide-n-line border-y border-n-line">{others.map(renderPeriod)}</ol>
        </details>
      )}
      {c?.marine.status === "ok" && (
        <p className="text-[13px] text-n-ink-3">
          NWS {c.marine.data.zoneName} ({c.marine.data.zone}){c.marine.data.updated ? `, issued ${clockLabel(c.marine.data.updated)}` : ""}.{" "}
          <a href={c.marine.url} {...ext} className="nx-focus font-semibold text-n-accent">
            NWS forecast <Icon name="external" className="inline h-3 w-3" />
          </a>
        </p>
      )}
      {c && c.marine.status !== "ok" && !cond.loading && (
        <p className="text-[14px] text-n-ink-2">
          No NWS zone forecast is read for these waters.{" "}
          <a href={c.marine.url ?? NWS_MARINE_PAGE} {...ext} className="nx-focus font-semibold text-n-accent underline">
            NWS marine forecasts
          </a>
        </p>
      )}

      {(b || harbor || tides.length > 0) && (
        <div data-testid="sea-measured">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4">
            {b?.windKt && (
              <div>
                <dt className="text-[12.5px] font-semibold text-n-ink-3">Wind, measured</dt>
                <dd className="nx-data text-[16px] font-semibold">
                  {b.windDirDeg ? `${compass(b.windDirDeg.value)} ` : ""}
                  {Math.round(b.windKt.value)} kt{b.gustKt ? `, gusts ${Math.round(b.gustKt.value)}` : ""}
                </dd>
              </div>
            )}
            {b?.waveFt && (
              <div>
                <dt className="text-[12.5px] font-semibold text-n-ink-3">Waves, measured</dt>
                <dd className="nx-data text-[16px] font-semibold">
                  {b.waveFt.value.toFixed(1)} ft{b.periodS ? ` at ${Math.round(b.periodS.value)} s` : ""}
                </dd>
              </div>
            )}
            {(harbor?.data.waterC ?? b?.waterC) && (
              <div>
                <dt className="text-[12.5px] font-semibold text-n-ink-3">{harbor ? "Water, harbour" : "Water, at the buoy"}</dt>
                <dd className="nx-data text-[16px] font-semibold">{Math.round(C_TO_F((harbor?.data.waterC ?? b!.waterC!).value))}°F</dd>
              </div>
            )}
            {tides.length > 0 && (
              <div className="col-span-2 sm:col-span-1">
                <dt className="text-[12.5px] font-semibold text-n-ink-3">Tides{day ? `, ${dayLabel(day)}` : ""} (predicted)</dt>
                <dd className="nx-data text-[14px] font-semibold leading-snug">{tides.map((t) => `${t.type === "H" ? "H" : "L"} ${t.ft.toFixed(1)} ft ${hm(t.at)}`).join(" · ")}</dd>
              </div>
            )}
          </dl>
          <p className="mt-2 text-[12.5px] leading-snug text-n-ink-3">
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

function SeasonRow({ i, open = false, extra, commercial = false }: { i: SeasonItem; open?: boolean; extra?: React.ReactNode; commercial?: boolean }) {
  return (
    <li className="py-1.5" data-testid={`season-${i.id}`} data-mark={i.mark}>
      <details open={open} className="group">
        <summary className="nx-focus flex min-h-11 cursor-pointer list-none items-start gap-2.5 rounded-lg">
          <MarkDot mark={i.mark} />
          <span className="min-w-0 flex-1">
            <span className="text-[15.5px] font-semibold">{i.name}</span>
            <span className="block text-[13.5px] text-n-ink-2">
              {MARK_WORDS[i.mark]}
              {commercial ? " (sport)" : ""}
            </span>
          </span>
          <Icon name="chevronDown" className="mt-1 h-4 w-4 text-n-ink-3 transition-transform group-open:rotate-180" />
        </summary>
        <div className="ml-[22px] mt-1 flex flex-col gap-2">
          {i.quote && <blockquote className="border-l-2 border-n-official-line pl-3 text-[14.5px] leading-relaxed text-n-ink">&ldquo;{i.quote}&rdquo;</blockquote>}
          {extra}
        </div>
      </details>
    </li>
  );
}

export function SeasonsBody({ place, species, audience, tripDate, today }: { place: CoastPlace; species: Species; audience: Audience; tripDate: string | null; today: string | null }) {
  const region = regionFor(place.lat);
  if (!region)
    return (
      <p className="text-[15px] text-n-ink-2">These waters are outside CDFW&rsquo;s five sport-fishing regions.</p>
    );
  const rs = regionSeasons(region, species.id);
  const day = tripDate ?? today;
  const gf = day ? groundfishOn(region.id, day) : null;
  const age = today ? seasonsAgeDays(today) : 0;
  const stale = age > SEASONS_STALE_DAYS;
  const commercialOnly = COMMERCIAL_ONLY.has(species.id);
  const commercial = audience === "commercial";
  const isGroundfish = (i: SeasonItem) => ["rockfish", "lingcod", "cabezon", "greenlings"].includes(i.id);
  const gfExtra = gf ? (
    <p className="rounded-xl bg-n-surface-2 px-3 py-2 text-[14px]" data-testid="season-groundfish-row">
      CDFW&rsquo;s {region.label} groundfish table, {monthDay(gf.from)}–{monthDay(gf.to)} {gf.year}: nearshore &ldquo;{gf.nearshore}&rdquo;; shelf, slope and lingcod &ldquo;{gf.shelf}&rdquo;.{" "}
      <a href={CDFW_GROUNDFISH.url} {...ext} className="nx-focus font-semibold text-n-accent">
        Table, updated {GROUNDFISH_UPDATED} <Icon name="external" className="inline h-3 w-3" />
      </a>
    </p>
  ) : null;
  return (
    <div className="flex flex-col gap-3">
      <div>
        {/* one source mark for the section; the page-level disclosure says the copy is unreviewed */}
        <p className="text-[13.5px] text-n-ink-3" data-testid="seasons-source">
          CDFW sport summary, <span className="font-semibold text-n-ink-2">{region.label}</span> region ({region.bounds}), dated {region.updated}; Neritic&rsquo;s copy of {dayLabel(SEASONS_READ)}.
        </p>
      </div>
      {stale && (
        <p className="rounded-xl border border-n-official-line bg-n-official-bg px-3 py-2 text-[14.5px] font-semibold" data-testid="seasons-stale">
          Neritic&rsquo;s copy is {age} days old. Seasons open and close in that time: read CDFW&rsquo;s page first.
        </p>
      )}
      {(commercial || commercialOnly) && (
        <p className="text-[14.5px] text-n-ink-2" data-testid="seasons-commercial">
          This is CDFW&rsquo;s list for sport fishing. Commercial seasons, openers, permits, quotas, and in-season gear, depth and toxin actions are set separately:{" "}
          <a href={CDFW_COMMERCIAL.url} {...ext} className="nx-focus font-semibold text-n-accent underline">
            {CDFW_COMMERCIAL.label}
          </a>{" "}
          and{" "}
          <a href={CDFW_INSEASON.url} {...ext} className="nx-focus font-semibold text-n-accent underline">
            in-season changes
          </a>
          .
        </p>
      )}
      {rs.mine.length > 0 && (
        <div data-testid="seasons-mine">
          <p className="nx-eyebrow">For {speciesPhrase(species)}</p>
          <ul className="divide-y divide-n-line">
            {rs.mine.map((i) => (
              // a commercial reader's own row stays closed: its quote is the sport list's dates
              <SeasonRow key={i.id} i={i} open={!commercial} commercial={commercial} extra={isGroundfish(i) ? gfExtra : null} />
            ))}
          </ul>
        </div>
      )}
      {species.id !== "any" && !rs.mine.length && !commercialOnly && <p className="text-[14.5px] text-n-ink-2">CDFW&rsquo;s {region.label} summary has no entry of its own for {speciesPhrase(species)}.</p>}
      {species.id === "salmon" && (
        <p className="flex flex-wrap gap-x-4 text-[14px]" data-testid="seasons-salmon-lines">
          {SALMON_LINES.map((l) => (
            <a key={l.tel} href={`tel:${l.tel}`} className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
              {l.label}: <span className="nx-data">{l.phone}</span>
            </a>
          ))}
        </p>
      )}
      <details data-testid="seasons-all">
        <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center text-[15px] font-semibold text-n-accent">
          {rs.mine.length ? "Also often fished here, and" : "Often fished here, and"} all {rs.all.length} entries ({rs.counts.open} open, {rs.counts.alert} alert, {rs.counts.closed} closed)
        </summary>
        <ul className="divide-y divide-n-line" data-testid="seasons-headline">
          {rs.headline.map((i) => (
            <SeasonRow key={i.id} i={i} extra={isGroundfish(i) ? gfExtra : null} />
          ))}
        </ul>
        <p className="nx-eyebrow mt-3">Every entry on CDFW&rsquo;s {region.label} page</p>
        <ul className="divide-y divide-n-line">
          {rs.all.map((i) => (
            <SeasonRow key={i.id} i={i} />
          ))}
        </ul>
      </details>
      <p className="flex flex-wrap gap-x-4 text-[14px]">
        <a href={region.url} {...ext} className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
          CDFW {region.label} region page <Icon name="external" className="h-3.5 w-3.5" />
        </a>
        <a href={CDFW_INSEASON.url} {...ext} className="nx-focus inline-flex min-h-11 items-center gap-1 font-semibold text-n-accent">
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
