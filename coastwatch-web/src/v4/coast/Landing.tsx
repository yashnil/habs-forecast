"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useId, useMemo, useState, type ReactNode } from "react";
import type { PortsCollection } from "@/generated/schema";
import { coastHref, portPlaces, resolvePlace, sitePlaces, type Audience, type CoastPlace } from "@/lib/coast";
import { EXAMPLES } from "@/content/coast";
import { Icon } from "@/v4/ui/Icon";
import { homeHref, type HomePort } from "./home";
import { compass } from "@/lib/conditions";
import { placeReadings, type Reading as Obs } from "@/lib/placeReadings";
import { useConditions } from "@/v4/explore/reads";
import { relTime } from "@/v4/ui/time";
import { COAST_LINES } from "./coastIsobaths";

/**
 * My Coast, first visit (v11): the question, the search, and the coast itself as the selector (real depth
 * contours, the ports on their true latitude and longitude). The names are a short list, home port first.
 * Pointing at a place (or focusing it) shows what is being measured there now, dated and sourced; a phone
 * reads the same on the briefing. The readings are observations, never a forecast and never a verdict.
 */
export function Landing({ ports, home, audience, unknownKey }: { ports: PortsCollection | null; home: HomePort | null; audience: Audience; unknownKey: string | null }) {
  const all = useMemo(() => [...portPlaces(ports), ...sitePlaces()], [ports]);
  const starters = [...(home ? [home.place] : []), ...EXAMPLES.filter((k) => k !== home?.place)]
    .map((k) => resolvePlace(k, ports))
    .filter((p): p is CoastPlace => !!p)
    .slice(0, 6);
  const hrefFor = (p: CoastPlace) => (home && p.key === home.place ? homeHref(home) : coastHref({ place: p.key, for: audience }));
  const [hot, setHot] = useState<string | null>(null);
  const hotPlace = starters.find((p) => p.key === hot) ?? null;
  return (
    <div className="coast-land" data-testid="coast-land">
      <section className="coast-land-ask" aria-labelledby="coast-title-h">
        <h1 id="coast-title-h" className="nx-display text-balance text-[clamp(46px,7vw,92px)] leading-[0.98]" data-testid="coast-title">
          Where are you fishing?
        </h1>
        <p className="nx-lede max-w-[34ch] text-n-ink-2">One trip: what the agencies say, the sea against your own limits, and what it would cost.</p>
        {unknownKey && (
          <p className="text-[15.5px] text-n-ink" role="status" data-testid="coast-unknown-place">
            The place in this link (&ldquo;{unknownKey}&rdquo;) is not one Neritic knows. Search for a harbour or harvest site.
          </p>
        )}
        <div className="max-w-[560px]">
          <PlaceSearch places={all} hrefFor={hrefFor} />
        </div>
        <nav aria-label="Start with" className="flex flex-col gap-1">
          <h2 className="nx-caps">Start with</h2>
          <ul className="coast-land-names" data-testid="coast-landing">
            {starters.map((p) => (
              <li key={p.key}>
                <Link href={hrefFor(p)} data-hot={hot === p.key || undefined} data-testid={`coast-start-${p.key}`} onPointerEnter={() => setHot(p.key)} onPointerLeave={() => setHot(null)} onFocus={() => setHot(p.key)} onBlur={() => setHot(null)} className="nx-focus">
                  {home?.place === p.key && <Icon name="harbor" className="h-4 w-4 text-n-ink-2" />}
                  {p.name}
                  {home?.place === p.key && <span className="sr-only"> (your home port)</span>}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </section>
      <CoastPicker places={starters} hot={hot} onHot={setHot} hrefFor={hrefFor} reading={hotPlace ? <PlaceReadings place={hotPlace} /> : null} />
    </div>
  );
}

const C_TO_F = (c: number) => (c * 9) / 5 + 32;

/** The readings of the place under the pointer, in place: what was measured, when, and by what. Never a forecast. */
function PlaceReadings({ place }: { place: CoastPlace }) {
  const [r] = useConditions({ lat: place.lat, lon: place.lon });
  const now = useNow();
  const v = r?.status === "ready" ? placeReadings(r.data) : null;
  const loading = !r || r.status === "loading";
  return (
    <div className="coast-land-read nx-deck flex flex-col gap-2" data-testid="landing-read">
      <p className="font-serif text-[20px] leading-6 text-n-ink">{place.name}</p>
      <span className="grid grid-cols-3 gap-2" data-testid="landing-readings">
        <Reading label="Water" testId="landing-water" loading={loading} r={v?.water ?? null} now={now} value={(w) => ({ v: `${Math.round(C_TO_F(w.c))}`, unit: "°F" })} />
        <Reading label="Waves" testId="landing-waves" loading={loading} r={v?.waves ?? null} now={now} value={(w) => ({ v: w.ft.toFixed(1), unit: w.s ? `ft · ${Math.round(w.s)} s` : "ft" })} />
        <Reading label="Wind" testId="landing-wind" loading={loading} r={v?.wind ?? null} now={now} value={(w) => ({ v: `${Math.round(w.kt)}`, unit: `kt${w.dir != null ? ` ${compass(w.dir)}` : ""}`, dir: w.dir })} />
      </span>
      <span className="min-h-[17px] text-[13px] leading-[18px] text-n-ink-2" data-testid="landing-sources">
        {loading ? "\u00a0" : v && v.sources.length ? `Measured by ${v.sources.map((x) => x.full).join(" · ")}` : "No reading near here just now"}
      </span>
    </div>
  );
}

function Reading<T>({
  label,
  r,
  value,
  loading,
  now,
  testId,
}: {
  label: string;
  r: Obs<T> | null;
  value: (r: T) => { v: string; unit: string; dir?: number | null };
  loading: boolean;
  now: Date | null;
  testId: string;
}) {
  const shown = r?.ok ? value(r as unknown as T) : null;
  return (
    <span className="min-w-0" data-testid={testId}>
      <span className="nx-caps block">{label}</span>
      {loading || !r ? (
        <span aria-hidden className="mt-1.5 block h-6 w-12 rounded-[2px] bg-n-surface-2" />
      ) : r.ok && shown ? (
        <>
          <span className="mt-0.5 flex items-baseline gap-1 whitespace-nowrap">
            {/* the arrow points where the wind blows toward (from its direction, plus 180 degrees) */}
            {shown.dir != null && <svg aria-hidden viewBox="0 0 12 12" className="h-3 w-3 self-center text-n-ink-2" style={{ transform: `rotate(${shown.dir + 180}deg)` }}><path d="M6 1.5v9M2.8 4.7 6 1.5l3.2 3.2" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>}
            <span className="nx-value text-[24px] leading-7">{shown.v}</span>
            <span className="truncate text-[13px] text-n-ink-2">{shown.unit}</span>
          </span>
          {/* this reading's own observation time and instrument, never the card's newest */}
          <span className="block truncate text-[13px] leading-[18px] text-n-ink-2" data-testid={`${testId}-age`}>
            {now ? relTime(r.at, now) : "\u00a0"}
          </span>
          <span className="block truncate text-[13px] leading-[18px] text-n-ink-2">{r.source.short}</span>
        </>
      ) : !r.ok ? (
        <>
          <span className="mt-0.5 flex min-h-7 items-center text-[14px] font-medium leading-[18px] text-n-ink-2" data-testid={`${testId}-none`}>
            {r.why}
          </span>
          {r.source && <span className="block truncate text-[13px] leading-[18px] text-n-ink-2">{r.source.short}</span>}
        </>
      ) : null}
    </span>
  );
}

/**
 * The coast as the selector: the real 200, 1,000 and 3,000 m depth contours (the 200 m shelf edge runs right
 * along the shore), with the starting places at their true latitude and longitude. Each point is a link.
 */
function CoastPicker({ places, hot, onHot, hrefFor, reading }: { places: CoastPlace[]; hot: string | null; onHot: (k: string | null) => void; hrefFor: (p: CoastPlace) => string; reading: ReactNode }) {
  const W = 480;
  const H = 720;
  const [LON0, LON1, LAT0, LAT1] = [-126.5, -116.6, 31.9, 42.3];
  const midLat = (LAT0 + LAT1) / 2;
  const kx = Math.cos((midLat * Math.PI) / 180);
  const scale = Math.min(W / ((LON1 - LON0) * kx), H / (LAT1 - LAT0));
  const ox = (W - (LON1 - LON0) * kx * scale) / 2;
  const x = (lon: number) => ox + (lon - LON0) * kx * scale;
  const y = (lat: number) => (LAT1 - lat) * scale;
  const line = (pts: [number, number][]) => pts.map(([lo, la], i) => `${i ? "L" : "M"}${x(lo).toFixed(1)},${y(la).toFixed(1)}`).join(" ");
  return (
    <div className="coast-land-map" data-testid="coast-picker">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" className="coast-land-svg" aria-label="California coast by depth contours; the starting places are links">
        {COAST_LINES.map((c) => (
          <path key={c.depth} d={line(c.points)} fill="none" stroke="var(--nx-ink)" strokeOpacity={c.depth === 200 ? 0.75 : c.depth === 1000 ? 0.28 : 0.14} strokeWidth={c.depth === 200 ? 1.5 : 1} strokeLinejoin="round" />
        ))}
        {places.map((p) => {
          const on = hot === p.key;
          const px = x(p.lon);
          const py = y(p.lat);
          const name = p.name.replace(/ \/.*$/, "");
          return (
            <g key={p.key} className="coast-land-point" onPointerEnter={() => onHot(p.key)} onPointerLeave={() => onHot(null)}>
              <a href={hrefFor(p)} data-testid={`coast-point-${p.key}`} onFocus={() => onHot(p.key)} onBlur={() => onHot(null)} aria-label={`${name}: open the briefing`}>
                {/* the 44 px tap target */}
                <circle cx={px} cy={py} r={22} fill="transparent" />
                <circle cx={px} cy={py} r={on ? 10 : 7} fill="none" stroke="var(--nx-ink)" strokeOpacity={on ? 1 : 0.35} strokeWidth="1.2" className="coast-land-ring" />
                <circle cx={px} cy={py} r={3.4} fill="var(--nx-ink)" />
                <text x={px > W - 96 ? px - 15 : px + 15} y={py + 4} textAnchor={px > W - 96 ? "end" : "start"} fontSize="15" fontWeight={on ? 600 : 500} fill="var(--nx-ink)" fillOpacity={on ? 1 : 0.8} style={{ paintOrder: "stroke", stroke: "var(--nx-bg)", strokeWidth: 3 }}>
                  {name}
                </text>
              </a>
            </g>
          );
        })}
        <text x={W - 6} y={H - 6} textAnchor="end" fontSize="11" fill="var(--nx-ink-2)">
          Depth contours 200, 1,000 and 3,000 m · true positions
        </text>
      </svg>
      {reading}
    </div>
  );
}

/** A client clock for "measured 20 min ago", set after hydration (never rendered on the server). */
function useNow(): Date | null {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const t = window.setInterval(() => setNow(new Date()), 60_000);
    return () => window.clearInterval(t);
  }, []);
  return now;
}

/** The one control: type-ahead over every place by name or county (an ARIA combobox; results are links). */
function PlaceSearch({ places, hrefFor }: { places: CoastPlace[]; hrefFor: (p: CoastPlace) => string }) {
  const router = useRouter();
  const id = useId();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const results = useMemo(() => {
    const t = q.trim().toLowerCase();
    if (!t) return [];
    const score = (p: CoastPlace) => {
      const n = p.name.toLowerCase();
      if (n.startsWith(t)) return 0;
      if (n.split(/[\s/(),-]+/).some((w) => w.startsWith(t))) return 1;
      if (n.includes(t)) return 2;
      if ((p.county ?? "").toLowerCase().startsWith(t)) return 3;
      return 9;
    };
    return places
      .map((p) => ({ p, s: score(p) }))
      .filter((x) => x.s < 9)
      .sort((a, b) => a.s - b.s || a.p.name.localeCompare(b.p.name))
      .slice(0, 8)
      .map((x) => x.p);
  }, [q, places]);
  const shown = open && q.trim().length > 0;
  const listId = `${id}-list`;
  const optId = (i: number) => `${id}-opt-${i}`;
  return (
    <div className="relative max-w-[640px]" data-testid="coast-place-search">
      <label htmlFor={`${id}-input`} className="sr-only">
        Search harbours and harvest sites
      </label>
      <div className="flex min-h-[60px] items-center gap-3 rounded-[4px] border border-n-line-strong bg-n-surface px-4 focus-within:border-n-accent">
        <Icon name="search" className="h-5 w-5 shrink-0 text-n-ink-3" />
        <input
          id={`${id}-input`}
          type="text"
          role="combobox"
          aria-expanded={shown}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={shown && results.length ? optId(active) : undefined}
          autoComplete="off"
          spellCheck={false}
          enterKeyHint="go"
          placeholder="Harbour, beach or town"
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
            setActive(0);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") {
              e.preventDefault();
              setOpen(true);
              setActive((a) => Math.min(a + 1, Math.max(results.length - 1, 0)));
            } else if (e.key === "ArrowUp") {
              e.preventDefault();
              setActive((a) => Math.max(a - 1, 0));
            } else if (e.key === "Enter" && shown && results[active]) {
              e.preventDefault();
              router.push(hrefFor(results[active]));
            } else if (e.key === "Escape") {
              if (q) setQ("");
              setOpen(false);
            }
          }}
          className="min-h-12 w-full min-w-0 bg-transparent text-[18px] font-medium text-n-ink outline-none placeholder:font-normal placeholder:text-n-ink-3"
          data-testid="coast-place-search-input"
        />
      </div>
      <div id={listId} role="listbox" aria-label="Places" hidden={!shown} className="nx-pop absolute inset-x-0 top-[calc(100%+6px)] z-30 overflow-hidden rounded-[4px] border border-n-line bg-n-surface py-1.5 shadow-[var(--nx-shadow)]" data-testid="coast-place-results">
        {results.length ? (
          results.map((p, i) => (
            <Link
              key={p.key}
              id={optId(i)}
              role="option"
              aria-selected={i === active}
              href={hrefFor(p)}
              onMouseEnter={() => setActive(i)}
              className={`flex min-h-12 items-center justify-between gap-3 px-4 text-left ${i === active ? "bg-n-surface-2" : ""}`}
              data-testid={`coast-result-${p.key}`}
            >
              <span className="min-w-0">
                <span className="block truncate text-[16px] font-semibold text-n-ink">{p.name}</span>
                <span className="block truncate text-[13px] text-n-ink-2">
                  {p.kind === "port" ? "Harbour" : "Harvest monitoring site"}
                  {p.county ? ` · ${p.county} County` : ""}
                </span>
              </span>
              <Icon name="chevronRight" className="h-4 w-4 shrink-0 text-n-ink-3" />
            </Link>
          ))
        ) : (
          <p className="px-4 py-3 text-[15px] text-n-ink-2" role="status">
            No harbour or harvest site matches &ldquo;{q.trim()}&rdquo;.
          </p>
        )}
      </div>
    </div>
  );
}
