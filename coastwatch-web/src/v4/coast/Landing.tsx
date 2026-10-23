"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { PortsCollection } from "@/generated/schema";
import { coastHref, portPlaces, resolvePlace, sitePlaces, type Audience, type CoastPlace } from "@/lib/coast";
import { EXAMPLES } from "@/content/coast";
import { Icon } from "@/v4/ui/Icon";
import { homeHref, type HomePort } from "./home";
import { compass } from "@/lib/conditions";
import { placeReadings, type Reading as Obs } from "@/lib/placeReadings";
import { useConditions } from "@/v4/explore/reads";
import { relTime } from "@/v4/ui/time";

/**
 * My Coast, first visit (v8): the front door. A dark hero holds the one question and the one control
 * ("Where are you fishing?", a place search over every harbour and harvest site) beside a drawing of the
 * coast with the starting places on it; under it, the starting places as cards, each with what is being
 * measured there now (water, waves, wind), dated and sourced, or nothing where nothing is measured. The
 * readings are observations, never a forecast and never a verdict on the water; the briefing holds the
 * rest. Activity and species are chosen in the briefing's trip bar.
 */
export function Landing({ ports, home, audience, unknownKey }: { ports: PortsCollection | null; home: HomePort | null; audience: Audience; unknownKey: string | null }) {
  const all = useMemo(() => [...portPlaces(ports), ...sitePlaces()], [ports]);
  const starters = [...(home ? [home.place] : []), ...EXAMPLES.filter((k) => k !== home?.place)]
    .map((k) => resolvePlace(k, ports))
    .filter((p): p is CoastPlace => !!p)
    .slice(0, 6);
  const hrefFor = (p: CoastPlace) => (home && p.key === home.place ? homeHref(home) : coastHref({ place: p.key, for: audience }));
  const [hot, setHot] = useState<string | null>(null);
  return (
    <div className="relative isolate">
      {/* the front door (v10): one question and one control over the coast, no card stack */}
      <section className="relative isolate overflow-hidden border-b border-n-line-strong pb-10 pt-6 sm:pb-12 sm:pt-10" aria-labelledby="coast-title-h">
        <ContourField />
        <div className="grid items-center gap-8 md:grid-cols-[minmax(0,1fr)_260px] lg:grid-cols-[minmax(0,1fr)_300px]">
          <div className="min-w-0">
            <p className="nx-caps">My Coast · one place, one trip</p>
            <h1 id="coast-title-h" className="mt-3 text-balance font-serif text-[40px] font-medium leading-[1.04] tracking-[-0.015em] text-n-ink sm:text-[60px]" data-testid="coast-title">
              Where are you fishing?
            </h1>
            <p className="mt-4 max-w-[52ch] text-pretty text-[16px] leading-6 text-n-ink-2">What the agencies say, the sea against your own limits, what the trip would cost and bring, and the waters on the map, for one place, each with its source.</p>
            {unknownKey && (
              <p className="mt-4 border-l-2 border-n-official pl-3 text-[15px] text-n-ink" role="status" data-testid="coast-unknown-place">
                The place in this link (&ldquo;{unknownKey}&rdquo;) is not one Neritic knows. Search for a harbour or harvest site.
              </p>
            )}
            <div className="mt-7 sm:mt-9">
              <PlaceSearch places={all} hrefFor={hrefFor} />
            </div>
          </div>
          <CoastPicker places={starters} hot={hot} onHot={setHot} hrefFor={hrefFor} />
        </div>
      </section>
      <nav aria-labelledby="coast-start-h" className="mt-8 sm:mt-10">
        <div className="mb-3 flex flex-col gap-0.5 sm:flex-row sm:items-baseline sm:justify-between sm:gap-4">
          <h2 id="coast-start-h" className="font-serif text-[24px] font-medium tracking-[-0.015em] text-n-ink">
            Start with
          </h2>
          <p className="text-[13px] text-n-ink-2">Latest measurements, from the harbour tide gauge and the nearest buoy, each with its own time</p>
        </div>
        <ul className="grid grid-cols-1 gap-px overflow-hidden rounded-[4px] border border-n-line bg-n-line sm:grid-cols-2 lg:grid-cols-3" data-testid="coast-landing">
          {starters.map((p) => (
            <PlaceCard key={p.key} place={p} href={hrefFor(p)} home={home?.place === p.key} hot={hot === p.key} onHot={setHot} />
          ))}
        </ul>
      </nav>
    </div>
  );
}

const C_TO_F = (c: number) => (c * 9) / 5 + 32;

function PlaceCard({ place, href, home, hot, onHot }: { place: CoastPlace; href: string; home: boolean; hot: boolean; onHot: (k: string | null) => void }) {
  const ref = useRef<HTMLLIElement | null>(null);
  // the readings are asked for once the card is near the screen (six places, each its own small request)
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;
    if (typeof IntersectionObserver === "undefined") return setSeen(true);
    const io = new IntersectionObserver((e) => e.some((x) => x.isIntersecting) && setSeen(true), { rootMargin: "200px" });
    io.observe(el);
    return () => io.disconnect();
  }, [seen]);
  const [r] = useConditions(seen ? { lat: place.lat, lon: place.lon } : null);
  const now = useNow();
  const v = r?.status === "ready" ? placeReadings(r.data) : null;
  const loading = !r || r.status === "loading";
  return (
    <li ref={ref} onPointerEnter={() => onHot(place.key)} onPointerLeave={() => onHot(null)}>
      <Link
        href={href}
        onFocus={() => onHot(place.key)}
        onBlur={() => onHot(null)}
        className={`nx-focus nx-tint group flex h-full flex-col gap-4 p-4 sm:p-5 ${hot ? "bg-n-surface-2" : "bg-n-surface"}`}
        data-testid={`coast-start-${place.key}`}
      >
        <span className="flex items-start justify-between gap-3">
          <span className="min-w-0">
            <span className="flex items-center gap-1.5 font-serif text-[22px] font-medium leading-7 tracking-[-0.01em] text-n-ink">
              {home && <Icon name="harbor" className="h-4 w-4 text-n-accent" />}
              <span className="truncate">{place.name}</span>
              {home && <span className="sr-only"> (your home port)</span>}
            </span>
            <span className="block truncate text-[13px] text-n-ink-2">
              {home ? "Your home port" : place.kind === "port" ? "Harbour" : "Harvest site"}
              {place.county ? ` · ${place.county} County` : ""}
            </span>
          </span>
          <Icon name="chevronRight" className="mt-1 h-4 w-4 text-n-ink-3 transition-transform group-hover:translate-x-0.5 group-hover:text-n-accent" />
        </span>
        <span className="grid grid-cols-3 gap-2" data-testid="landing-readings">
          <Reading label="Water" testId="landing-water" loading={loading} r={v?.water ?? null} now={now} value={(w) => ({ v: `${Math.round(C_TO_F(w.c))}`, unit: "°F" })} />
          <Reading label="Waves" testId="landing-waves" loading={loading} r={v?.waves ?? null} now={now} value={(w) => ({ v: w.ft.toFixed(1), unit: w.s ? `ft · ${Math.round(w.s)} s` : "ft" })} />
          <Reading label="Wind" testId="landing-wind" loading={loading} r={v?.wind ?? null} now={now} value={(w) => ({ v: `${Math.round(w.kt)}`, unit: `kt${w.dir != null ? ` ${compass(w.dir)}` : ""}`, dir: w.dir })} />
        </span>
        {/* each instrument named once; every reading above carries its own age and which of these measured it */}
        <span className="mt-auto min-h-[17px] text-[12.5px] leading-[17px] text-n-ink-3" data-testid="landing-sources">
          {loading ? "\u00a0" : v && v.sources.length ? `Measured by ${v.sources.map((x) => x.full).join(" · ")}` : "No reading near here just now"}
        </span>
      </Link>
    </li>
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
      <span className="nx-caps block text-n-conditions">{label}</span>
      {loading || !r ? (
        <span aria-hidden className="nx-skeleton mt-1.5 block h-6 w-12 rounded-[2px]" />
      ) : r.ok && shown ? (
        <>
          <span className="mt-0.5 flex items-baseline gap-1 whitespace-nowrap">
            {/* the arrow points where the wind blows toward (from its direction, plus 180 degrees) */}
            {shown.dir != null && <svg aria-hidden viewBox="0 0 12 12" className="h-3 w-3 self-center text-n-ink-2" style={{ transform: `rotate(${shown.dir + 180}deg)` }}><path d="M6 1.5v9M2.8 4.7 6 1.5l3.2 3.2" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" /></svg>}
            <span className="nx-value text-[26px] leading-7">{shown.v}</span>
            <span className="truncate text-[12.5px] text-n-ink-2">{shown.unit}</span>
          </span>
          {/* this reading's own observation time and instrument, never the card's newest */}
          <span className="block truncate text-[12px] leading-4 text-n-ink-2" data-testid={`${testId}-age`}>
            {now ? relTime(r.at, now) : "\u00a0"}
          </span>
          <span className="block truncate text-[12px] leading-4 text-n-ink-3">{r.source.short}</span>
        </>
      ) : !r.ok ? (
        <>
          <span className="mt-0.5 flex min-h-7 items-center text-[14px] font-medium leading-[18px] text-n-ink-3" data-testid={`${testId}-none`}>
            {r.why}
          </span>
          {r.source && <span className="block truncate text-[12px] leading-4 text-n-ink-3">{r.source.short}</span>}
        </>
      ) : null}
    </span>
  );
}

/**
 * The coast with the starting places on it (true latitude, longitude compressed as the Research coast
 * ladder draws it): a picture that answers "where", in step with the cards (pointing at one lights the
 * other). Decorative for assistive technology: the cards are the accessible way in.
 */
function CoastPicker({ places, hot, onHot, hrefFor }: { places: CoastPlace[]; hot: string | null; onHot: (k: string | null) => void; hrefFor: (p: CoastPlace) => string }) {
  const router = useRouter();
  const W = 320;
  const H = 380;
  const y = (lat: number) => 14 + ((42.2 - lat) / (42.2 - 32.3)) * (H - 28);
  const x = (lon: number) => 40 + ((lon + 124.6) / 7.7) * (W - 120);
  const coast = COAST.map(([la, lo], i) => `${i ? "L" : "M"}${x(lo).toFixed(1)},${y(la).toFixed(1)}`).join(" ");
  const land = `${coast} L${W},${y(32.3)} L${W},${y(42.2)} Z`;
  return (
    <svg aria-hidden viewBox={`0 0 ${W} ${H}`} className="mx-auto hidden w-full max-w-[320px] md:block" data-testid="coast-picker">
      <defs>
        <linearGradient id="nx-landing-land" x1="0" x2="1">
          <stop offset="0" stopColor="#1d2a35" />
          <stop offset="1" stopColor="#0c141b" stopOpacity="0" />
        </linearGradient>
        {/* the land fades out at the top and bottom of the drawing, never a hard edge */}
        <linearGradient id="nx-landing-fade" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="#fff" stopOpacity="0" />
          <stop offset="0.12" stopColor="#fff" stopOpacity="1" />
          <stop offset="0.88" stopColor="#fff" stopOpacity="1" />
          <stop offset="1" stopColor="#fff" stopOpacity="0" />
        </linearGradient>
        <mask id="nx-landing-mask">
          <rect width={W} height={H} fill="url(#nx-landing-fade)" />
        </mask>
      </defs>
      {[0, 1, 2, 3].map((i) => (
        <path key={i} d={coast} transform={`translate(${-10 - i * 12} 0)`} fill="none" stroke="#5fd0e6" strokeOpacity={0.16 - i * 0.03} strokeWidth="1" />
      ))}
      <path d={land} fill="url(#nx-landing-land)" mask="url(#nx-landing-mask)" />
      <path d={coast} fill="none" stroke="#bfe4ec" strokeOpacity="0.7" strokeWidth="1.3" strokeLinejoin="round" className="nx-landing-coast" pathLength={1} />
      {places.map((p) => {
        const on = hot === p.key;
        const px = x(p.lon);
        const py = y(p.lat);
        return (
          <g key={p.key} className="cursor-pointer" onPointerEnter={() => onHot(p.key)} onPointerLeave={() => onHot(null)} onClick={() => router.push(hrefFor(p))}>
            <circle cx={px} cy={py} r={14} fill="transparent" />
            {on && <circle cx={px} cy={py} r={9} fill="rgba(95,208,230,0.16)" stroke="#5fd0e6" strokeWidth="1.2" />}
            <circle cx={px} cy={py} r={on ? 4.2 : 3.4} fill={on ? "#5fd0e6" : "#eaf3f7"} stroke="#061016" strokeWidth="1.4" />
            <text x={px + 12} y={py + 4} textAnchor="start" fontSize="12" fontWeight={on ? 600 : 500} fill={on ? "#eaf6fa" : "rgba(214,232,240,0.72)"} style={{ paintOrder: "stroke", stroke: "#071119", strokeWidth: 3 }}>
              {p.name.replace(/ \/.*$/, "")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** A simplified California shoreline, north to south: (lat, lon). Drawn, not measured (as the Research coast ladder). */
const COAST: [number, number][] = [
  [42.0, -124.21], [41.75, -124.2], [41.06, -124.14], [40.8, -124.2], [40.44, -124.41], [39.8, -123.85],
  [38.95, -123.74], [38.3, -123.07], [38.0, -123.02], [37.81, -122.52], [37.5, -122.49], [37.1, -122.33],
  [36.96, -122.02], [36.8, -121.79], [36.6, -121.89], [36.3, -121.9], [35.67, -121.28], [35.37, -120.86],
  [35.17, -120.74], [34.58, -120.64], [34.45, -120.47], [34.41, -119.69], [34.1, -119.1], [34.03, -118.75],
  [34.01, -118.5], [33.75, -118.41], [33.6, -117.93], [33.46, -117.71], [33.0, -117.28], [32.87, -117.25],
  [32.53, -117.12],
];

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

/** Isobaths behind the question: static, decorative, faint cyan on the instrument ground. */
function ContourField() {
  return (
    <svg aria-hidden className="pointer-events-none absolute inset-0 -z-10 h-full w-full text-[#5fd0e6] opacity-[0.13]" viewBox="0 0 760 340" preserveAspectRatio="xMidYMid slice" fill="none">
      {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
        <path key={i} d={`M${-40 + i * 26} -10 C ${80 + i * 30} ${60 + i * 8}, ${30 + i * 34} ${140 + i * 10}, ${160 + i * 36} ${190 + i * 12} S ${400 + i * 30} ${250 + i * 10}, ${520 + i * 26} ${300 + i * 8} S ${700 + i * 20} ${330 + i * 6}, ${800} ${320 + i * 10}`} stroke="currentColor" strokeWidth={i === 4 ? 1.4 : 0.8} />
      ))}
    </svg>
  );
}
