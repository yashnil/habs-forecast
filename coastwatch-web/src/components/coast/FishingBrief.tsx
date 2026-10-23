"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { PortsCollection } from "@/generated/schema";
import type { Audience, CoastPlace } from "@/lib/coast";
import { compass, marineHeadline, stationName, NWS_MARINE_PAGE, type Conditions, type MarinePeriod } from "@/lib/conditions";
import { distanceKm } from "@/lib/grid";
import { LIMITS_KEY, marineMaxima, overLimits, parseLimits, type SeaLimits } from "@/lib/trip";
import { MARK_SHORT, MARK_WORDS, dollars, groundfishOn, monthDay, perPound, portAreaValues, pounds, regionFor, regionSeasons, seasonsAgeDays, speciesValues } from "@/lib/speciesSeasons";
import { CDFW_COMMERCIAL, CDFW_GROUNDFISH, CDFW_INSEASON, GROUNDFISH_UPDATED, SALMON_LINES, SEASONS_READ, SEASONS_STALE_DAYS, type SeasonItem } from "@/content/seasons";
import { CDFW_MFDE, LANDINGS } from "@/content/landings";
import { speciesPhrase, type Species } from "@/content/species";
import { Icon } from "@/v4/ui/Icon";
import { clockLabel, dayLabel, kmLabel } from "@/v4/ui/time";

/**
 * The fishing side of the trip briefing (v5, fishermen audit): is it in season here, what is the sea
 * doing, and what has the catch been worth. Everything is quoted from its agency with its date:
 * CDFW's regional sport-fishing summary (content/seasons), NWS, NDBC and CO-OPS (lib/conditions),
 * CDFW By The Numbers (content/landings). Neritic never says a fishery is open, that it is safe to go,
 * or where fish are.
 */

const ext = { target: "_blank", rel: "noreferrer" } as const;
const C_TO_F = (c: number) => (c * 9) / 5 + 32;
const COMMERCIAL_ONLY = new Set(["squid", "urchin"]);

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

/** The trip day's daytime NWS period ("Tuesday", not "Tuesday Night"), else the first. */
export function periodFor(c: Conditions | null, day: string | null, today: string | null): MarinePeriod | null {
  if (c?.marine.status !== "ok") return null;
  const ps = c.marine.data.periods;
  if (!day || day === today) return ps[0] ?? null;
  const name = new Date(`${day}T12:00:00Z`).toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" }).toUpperCase();
  return ps.find((p) => p.name.toUpperCase().startsWith(name.slice(0, 3)) && !/NIGHT/i.test(p.name)) ?? null;
}

// ---------------------------------------------------------------- at a glance

/**
 * Four answers a skipper looks for first, each one tap from its evidence: CDFW's season list for the
 * species here, the forecaster's wind and seas, the notices tied to these waters, and what the catch
 * was worth. Each tile is a link to its section below; nothing here is Neritic's verdict.
 */
export function AtAGlance({
  place,
  ports,
  species,
  audience,
  cond,
  tripDate,
  today,
  notices,
  limits,
}: {
  place: CoastPlace;
  ports: PortsCollection | null;
  species: Species;
  audience: Audience;
  cond: { data: Conditions | null; loading: boolean; error: string | null };
  tripDate: string | null;
  today: string | null;
  notices: number;
  limits: SeaLimits;
}) {
  const region = regionFor(place.lat);
  const rs = region ? regionSeasons(region, species.id) : null;
  const mine = rs?.mine[0] ?? null;
  const gf = species.id === "groundfish" && region && (tripDate ?? today) ? groundfishOn(region.id, (tripDate ?? today)!) : null;
  const c = cond.data;
  const period = periodFor(c, tripDate, today);
  const alerts = c?.alerts.status === "ok" ? c.alerts.data : [];
  const max = period ? marineMaxima(period.text) : null;
  const over = max ? overLimits(max, limits) : { wind: false, seas: false };
  const value = speciesValues(species.id)[0] ?? null;
  const pa = placePortArea(place, ports);
  const paValue = pa ? portAreaValues(pa.area)[0] : null;
  const commercialOnly = COMMERCIAL_ONLY.has(species.id);
  // CDFW's own sentence, when its section gives a reopening date ("The season is scheduled to reopen on November 7, 2026.")
  const reopen = mine?.quote.match(/scheduled to reopen on ([A-Z][a-z]+ \d{1,2}, \d{4})/)?.[1] ?? null;
  const water = c?.harbor?.status === "ok" ? { c: c.harbor.data.waterC.value, where: "harbour" } : c?.buoy.status === "ok" && c.buoy.data.waterC ? { c: c.buoy.data.waterC.value, where: "buoy" } : null;

  const tile = "nx-press nx-focus flex min-h-[112px] flex-col gap-1 rounded-2xl border border-n-line bg-n-surface p-3.5 text-left hover:border-n-line-strong";
  return (
    <section aria-labelledby="glance-h" className="flex flex-col gap-3" data-testid="brief-glance">
      <h2 id="glance-h" className="nx-eyebrow">
        At a glance{tripDate && today && tripDate !== today ? ` · ${dayLabel(tripDate, true)}` : ""}
      </h2>
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <a href="#seasons-h" className={tile} data-testid="glance-season">
          <span className="text-[12.5px] font-semibold uppercase tracking-wide text-n-ink-3">Season · CDFW</span>
          {mine ? (
            <>
              <span className="flex items-start gap-1.5 text-[16px] font-semibold leading-snug">
                <MarkDot mark={mine.mark} />
                {mine.name}
              </span>
              <span className="text-[14px] font-semibold leading-snug" data-testid="glance-season-list">
                On CDFW&rsquo;s {MARK_SHORT[mine.mark].toLowerCase()}
              </span>
              <span className="text-[13px] leading-snug text-n-ink-2">
                {reopen ? `CDFW: “scheduled to reopen on ${reopen}”` : gf ? `${monthDay(gf.from)}–${monthDay(gf.to)}: ${gf.nearshore}` : `${region!.label} region, CDFW ${region!.updated}`}
              </span>
            </>
          ) : rs ? (
            <>
              <span className="text-[16px] font-semibold leading-snug">
                {commercialOnly ? `${species.label}: commercial` : `${rs.counts.open + rs.counts.alert} on the open list, ${rs.counts.closed} closed`}
              </span>
              <span className="text-[13px] leading-snug text-n-ink-2">{commercialOnly ? "Not on CDFW's sport summary: see CDFW commercial rules" : `${region!.label} region, CDFW sport summary of ${region!.updated}`}</span>
            </>
          ) : (
            <span className="text-[14px] text-n-ink-2">Outside CDFW&rsquo;s five sport-fishing regions</span>
          )}
          {audience === "commercial" && !commercialOnly && rs && <span className="text-[12.5px] text-n-ink-3">Sport list; commercial rules differ</span>}
        </a>
        <a href="#sea-h" className={tile} data-testid="glance-sea">
          <span className="text-[12.5px] font-semibold uppercase tracking-wide text-n-ink-3">Sea · NWS{period ? ` ${period.name.toLowerCase()}` : ""}</span>
          {alerts.length > 0 && (
            <span className="inline-flex items-center gap-1 text-[14px] font-semibold text-n-official" data-testid="glance-alert">
              <Icon name="alert" className="h-4 w-4" /> {alerts[0].event}
            </span>
          )}
          {period ? (
            <span className="text-[15px] font-semibold leading-snug">{marineHeadline(period)}</span>
          ) : (
            <span className="text-[14px] text-n-ink-2">{cond.loading ? "Reading NWS…" : c?.marine.status === "ok" ? "No NWS period for this day yet" : "NWS zone forecast not available here"}</span>
          )}
          {water && <span className="text-[13px] text-n-ink-2">Water {Math.round(C_TO_F(water.c))}°F at the {water.where}, measured</span>}
          {(over.wind || over.seas) && (
            <span className="text-[13px] font-semibold text-n-official" data-testid="glance-over">
              Above your own {[over.wind && `${limits.windKt} kt`, over.seas && `${limits.seasFt} ft`].filter(Boolean).join(" and ")} limit
            </span>
          )}
        </a>
        <a href="#official-h" className={tile} data-testid="glance-notices">
          <span className="text-[12.5px] font-semibold uppercase tracking-wide text-n-ink-3">Notices · CDFW, CDPH</span>
          <span className="text-[16px] font-semibold leading-snug">{notices ? `${notices} tied to these waters` : "None tied to these waters"}</span>
          <span className="text-[13px] font-semibold text-n-official">Not verified; the list may be incomplete</span>
        </a>
        <a href="#worth-h" className={tile} data-testid="glance-worth">
          <span className="text-[12.5px] font-semibold uppercase tracking-wide text-n-ink-3">Worth · CDFW {value?.year ?? paValue?.year ?? ""}</span>
          {value ? (
            <>
              <span className="text-[16px] font-semibold leading-snug">
                {value.fishery}: {dollars(value.dollars)}
              </span>
              <span className="text-[13px] leading-snug text-n-ink-2">statewide at the dock, about {perPound(value.perLb)} on average</span>
            </>
          ) : paValue ? (
            <>
              <span className="text-[16px] font-semibold leading-snug">{dollars(paValue.dollars)} landed</span>
              <span className="text-[13px] leading-snug text-n-ink-2">
                {titleCase(pa!.area)} port area, all species{pa!.via ? ` (nearest port ${pa!.via})` : ""}
              </span>
            </>
          ) : (
            <span className="text-[14px] text-n-ink-2">No CDFW landings figure for these waters</span>
          )}
        </a>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- the sea

export function SeaCard({
  cond,
  tripDate,
  today,
  limits,
  onLimits,
  now,
}: {
  cond: { data: Conditions | null; loading: boolean; error: string | null };
  tripDate: string | null;
  today: string | null;
  limits: SeaLimits;
  onLimits: (l: SeaLimits) => void;
  now: Date | null;
}) {
  const c = cond.data;
  const alerts = c?.alerts.status === "ok" ? c.alerts.data : [];
  const periods = c?.marine.status === "ok" ? c.marine.data.periods : [];
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
  const num = (v: string) => (v.trim() === "" ? null : Number(v) > 0 ? Number(v) : null);
  return (
    <section aria-labelledby="sea-h" className="flex flex-col gap-4 border-t border-n-line pt-7" data-testid="brief-sea">
      <div>
        <h2 id="sea-h" className="font-serif text-[24px] font-semibold leading-tight">
          Sea conditions
        </h2>
        <p className="mt-1 text-[15px] text-n-ink-2">The forecaster&rsquo;s words, the nearest buoy and the tides. Wind, waves and tides only: not a fishing or toxin forecast, and never a go or no-go from Neritic.</p>
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
      {periods.length > 0 && (
        <ol className="grid gap-2 sm:grid-cols-2" data-testid="sea-periods">
          {periods.map((p) => {
            const m = marineMaxima(p.text);
            const o = overLimits(m, limits);
            const isChosen = chosen?.name === p.name;
            return (
              <li key={p.name} className={`rounded-2xl border bg-n-surface p-3.5 ${isChosen ? "border-n-accent" : "border-n-line"}`} data-testid="sea-period" data-chosen={isChosen}>
                <p className="flex items-baseline justify-between gap-2">
                  <span className="text-[13px] font-semibold uppercase tracking-wide text-n-ink-3">{p.name}</span>
                  <span className="nx-data text-[13px] text-n-ink-2">
                    {m.windKt != null ? `≤ ${m.windKt} kt` : ""}
                    {m.windKt != null && m.seasFt != null ? " · " : ""}
                    {m.seasFt != null ? `≤ ${m.seasFt} ft` : ""}
                  </span>
                </p>
                <p className="mt-1 text-[16px] font-semibold leading-snug">{marineHeadline(p)}</p>
                {(o.wind || o.seas) && (
                  <p className="mt-1 text-[13.5px] font-semibold text-n-official" data-testid="sea-over">
                    The forecast&rsquo;s {[o.wind && `${m.windKt} kt`, o.seas && `${m.seasFt} ft`].filter(Boolean).join(" and ")} {o.wind && o.seas ? "are" : "is"} above your own limit
                  </p>
                )}
                <details className="mt-1">
                  <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center text-[13.5px] font-semibold text-n-accent">Full NWS text</summary>
                  <p className="text-[14px] leading-relaxed text-n-ink-2">{p.text}</p>
                </details>
              </li>
            );
          })}
        </ol>
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
        <div className="rounded-2xl border border-n-line bg-n-surface p-3.5" data-testid="sea-measured">
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
          <p className="mt-3 text-[12.5px] leading-snug text-n-ink-3">
            {buoy?.station ? `${stationName(buoy.station.name)} (NDBC ${buoy.station.id}), ${kmLabel(buoy.station.km)} away` : ""}
            {b?.waveFt && now ? `, ${clockLabel(b.waveFt.at)}` : ""}
            {harbor?.station ? ` · harbour water at ${stationName(harbor.station.name)} (CO-OPS)` : ""}
            {c?.tides.status === "ok" && c.tides.station ? ` · tides at ${stationName(c.tides.station.name)}, feet above MLLW` : ""}
          </p>
        </div>
      )}

      <div className="rounded-2xl border border-dashed border-n-line-strong p-3.5" data-testid="sea-limits">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p className="text-[14.5px]">
            <span className="font-semibold">Your own limits</span>
            {limits.windKt || limits.seasFt ? (
              <span className="text-n-ink-2">
                : {[limits.windKt && `wind ${limits.windKt} kt`, limits.seasFt && `seas ${limits.seasFt} ft`].filter(Boolean).join(", ")}
              </span>
            ) : (
              <span className="text-n-ink-2">: not set</span>
            )}
          </p>
          <button type="button" onClick={() => setEditing((e) => !e)} aria-expanded={editing} className="nx-press nx-focus min-h-11 rounded-xl px-3 text-[14.5px] font-semibold text-n-accent" data-testid="sea-limits-edit">
            {editing ? "Done" : limits.windKt || limits.seasFt ? "Change" : "Set"}
          </button>
        </div>
        {editing && (
          <div className="mt-2 flex flex-wrap gap-3">
            <label className="flex flex-col gap-1 text-[13.5px] font-semibold text-n-ink-2">
              Wind above (kt)
              <input type="number" inputMode="numeric" min={1} max={80} defaultValue={limits.windKt ?? ""} onChange={(e) => onLimits({ ...limits, windKt: num(e.target.value) })} className="nx-focus min-h-12 w-28 rounded-xl border border-n-line-strong bg-n-surface px-3 text-[16px] text-n-ink" data-testid="sea-limit-wind" />
            </label>
            <label className="flex flex-col gap-1 text-[13.5px] font-semibold text-n-ink-2">
              Seas above (ft)
              <input type="number" inputMode="decimal" min={1} max={40} step="0.5" defaultValue={limits.seasFt ?? ""} onChange={(e) => onLimits({ ...limits, seasFt: num(e.target.value) })} className="nx-focus min-h-12 w-28 rounded-xl border border-n-line-strong bg-n-surface px-3 text-[16px] text-n-ink" data-testid="sea-limit-seas" />
            </label>
          </div>
        )}
        <p className="mt-1 text-[12.5px] leading-snug text-n-ink-3">Kept on this device. Neritic only compares them with the highest wind and seas the NWS text states; your boat, your crew and the skipper decide.</p>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- seasons

function SeasonRow({ i, open = false, extra }: { i: SeasonItem; open?: boolean; extra?: React.ReactNode }) {
  return (
    <li className="py-2.5" data-testid={`season-${i.id}`} data-mark={i.mark}>
      <details open={open} className="group">
        <summary className="nx-focus flex min-h-11 cursor-pointer list-none items-start gap-2.5 rounded-lg">
          <MarkDot mark={i.mark} />
          <span className="min-w-0 flex-1">
            <span className="text-[16px] font-semibold">{i.name}</span>
            <span className="block text-[13.5px] text-n-ink-2">{MARK_WORDS[i.mark]}</span>
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

export function SeasonsCard({ place, species, audience, tripDate, today }: { place: CoastPlace; species: Species; audience: Audience; tripDate: string | null; today: string | null }) {
  const region = regionFor(place.lat);
  if (!region)
    return (
      <section aria-labelledby="seasons-h" className="flex flex-col gap-3 border-t border-n-line pt-7" data-testid="brief-seasons">
        <h2 id="seasons-h" className="font-serif text-[24px] font-semibold leading-tight">
          Seasons in these waters
        </h2>
        <p className="text-[15px] text-n-ink-2">These waters are outside CDFW&rsquo;s five sport-fishing regions.</p>
      </section>
    );
  const rs = regionSeasons(region, species.id);
  const day = tripDate ?? today;
  const gf = day ? groundfishOn(region.id, day) : null;
  const age = today ? seasonsAgeDays(today) : 0;
  const stale = age > SEASONS_STALE_DAYS;
  const commercialOnly = COMMERCIAL_ONLY.has(species.id);
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
    <section aria-labelledby="seasons-h" className="flex flex-col gap-3 border-t border-n-line pt-7" data-testid="brief-seasons">
      <div>
        <h2 id="seasons-h" className="font-serif text-[24px] font-semibold leading-tight">
          Seasons in these waters
        </h2>
        <p className="mt-1 text-[15px] text-n-ink-2">
          CDFW&rsquo;s sport-fishing summary for its <span className="font-semibold text-n-ink">{region.label}</span> region ({region.bounds}), which CDFW dates {region.updated}. Quoted from Neritic&rsquo;s copy of {dayLabel(SEASONS_READ)}; no person has reviewed it, and CDFW&rsquo;s page is the authority.
        </p>
      </div>
      {stale && (
        <p className="rounded-xl border border-n-official-line bg-n-official-bg px-3 py-2 text-[14.5px] font-semibold" data-testid="seasons-stale">
          Neritic&rsquo;s copy is {age} days old. Seasons open and close in that time: read CDFW&rsquo;s page first.
        </p>
      )}
      {(audience === "commercial" || commercialOnly) && (
        <p className="text-[14.5px] text-n-ink-2" data-testid="seasons-commercial">
          This summary is CDFW&rsquo;s for sport fishing. Commercial seasons, permits, quotas, and in-season gear, depth and toxin actions are set separately:{" "}
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
              <SeasonRow key={i.id} i={i} open extra={isGroundfish(i) ? gfExtra : null} />
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
      <div>
        <p className="nx-eyebrow">{rs.mine.length ? "Also often fished here" : "Often fished here"}</p>
        <ul className="divide-y divide-n-line" data-testid="seasons-headline">
          {rs.headline.map((i) => (
            <SeasonRow key={i.id} i={i} extra={isGroundfish(i) ? gfExtra : null} />
          ))}
        </ul>
      </div>
      <details data-testid="seasons-all">
        <summary className="nx-focus inline-flex min-h-11 cursor-pointer items-center text-[15px] font-semibold text-n-accent">
          All {rs.all.length} entries on CDFW&rsquo;s {region.label} page ({rs.counts.open} open, {rs.counts.alert} marked alert, {rs.counts.closed} closed)
        </summary>
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
    </section>
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
            <p className="nx-eyebrow">{species.label}, statewide</p>
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
            {species.label} is not in CDFW&rsquo;s statewide top ten by value for 2023 or 2024, so Neritic has no figure for it.{" "}
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
        . History, not this season&rsquo;s prices. The per-pound figure is Neritic&rsquo;s arithmetic, CDFW&rsquo;s dollars divided by CDFW&rsquo;s pounds, an average over a year of different boats, grades and buyers.{" "}
        <Link href="/fisheries" className="nx-focus font-semibold text-n-accent">
          Ten years of statewide history
        </Link>
      </p>
    </div>
  );
}
