"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useId, useMemo, useState } from "react";
import type { PortsCollection } from "@/generated/schema";
import { coastHref, portPlaces, resolvePlace, sitePlaces, type Audience, type CoastPlace } from "@/lib/coast";
import { EXAMPLES } from "@/content/coast";
import { Icon } from "@/v4/ui/Icon";
import { homeHref, type HomePort } from "./home";

/**
 * My Coast, first visit (v7): one question, one control. "Where are you fishing?" over a faint static
 * contour field, a place search over every harbour and harvest site, a short row of starting places and
 * one line on what a briefing holds. Activity and species are chosen in the briefing's trip bar.
 */
export function Landing({ ports, home, audience, unknownKey }: { ports: PortsCollection | null; home: HomePort | null; audience: Audience; unknownKey: string | null }) {
  const all = useMemo(() => [...portPlaces(ports), ...sitePlaces()], [ports]);
  const starters = [...(home ? [home.place] : []), ...EXAMPLES.filter((k) => k !== home?.place)]
    .map((k) => resolvePlace(k, ports))
    .filter((p): p is CoastPlace => !!p)
    .slice(0, 6);
  const hrefFor = (p: CoastPlace) => (home && p.key === home.place ? homeHref(home) : coastHref({ place: p.key, for: audience }));
  return (
    <div className="relative isolate">
      <ContourField />
      <h1 className="nx-display text-balance pt-6 sm:pt-14" data-testid="coast-title">
        Where are you fishing?
      </h1>
      {unknownKey && (
        <p className="mt-4 border-l-[3px] border-n-official pl-3 text-[15px] text-n-ink" role="status" data-testid="coast-unknown-place">
          The place in this link (&ldquo;{unknownKey}&rdquo;) is not one Neritic knows. Search for a harbour or harvest site.
        </p>
      )}
      <div className="mt-6 sm:mt-8">
        <PlaceSearch places={all} hrefFor={hrefFor} />
      </div>
      <nav aria-labelledby="coast-start-h" className="mt-6 flex flex-col gap-1 sm:mt-7 sm:flex-row sm:items-baseline sm:gap-4">
        <span id="coast-start-h" className="shrink-0 text-[13px] font-semibold text-n-ink-3">
          Start with
        </span>
        <ul className="flex flex-wrap gap-x-5 text-[15.5px]" data-testid="coast-landing">
          {starters.map((p) => {
            const isHome = home?.place === p.key;
            return (
              <li key={p.key}>
                <Link href={hrefFor(p)} className="nx-focus inline-flex min-h-11 items-center gap-1 rounded font-semibold text-n-ink underline decoration-n-line-strong underline-offset-[5px] hover:text-n-accent hover:decoration-n-accent" data-testid={`coast-start-${p.key}`}>
                  {isHome && <Icon name="harbor" className="h-4 w-4 text-n-accent" />}
                  {p.name}
                  {isHome && <span className="sr-only"> (your home port)</span>}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>
      <p className="mt-8 max-w-[56ch] text-[15px] leading-6 text-n-ink-2 sm:mt-10">A briefing holds the sea forecast, CDFW&rsquo;s season list, the agencies&rsquo; notices, the latest bloom samples and your own trip numbers, each with its source.</p>
    </div>
  );
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
      <div className="flex min-h-[60px] items-center gap-3 rounded-2xl border border-n-line-strong bg-n-surface px-4 shadow-[0_1px_2px_rgba(11,24,32,0.06)] focus-within:border-n-accent focus-within:shadow-[0_0_0_3px_var(--nx-accent-bg)]">
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
      <div id={listId} role="listbox" aria-label="Places" hidden={!shown} className="nx-pop absolute inset-x-0 top-[calc(100%+6px)] z-30 overflow-hidden rounded-2xl border border-n-line bg-n-surface py-1.5 shadow-[var(--nx-shadow)]" data-testid="coast-place-results">
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
                <span className="block truncate text-[13px] text-n-ink-3">
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

/** Faint isobaths behind the question: static, decorative, about 6 % ink. */
function ContourField() {
  return (
    <svg aria-hidden className="pointer-events-none absolute -right-4 top-2 -z-10 h-[340px] w-[min(760px,100%)] text-n-ink opacity-[0.07] sm:-right-10" viewBox="0 0 760 340" preserveAspectRatio="xMaxYMin slice" fill="none">
      {[0, 1, 2, 3, 4, 5, 6].map((i) => (
        <path key={i} d={`M${120 + i * 26} -10 C ${200 + i * 30} ${60 + i * 8}, ${150 + i * 34} ${140 + i * 10}, ${260 + i * 36} ${190 + i * 12} S ${470 + i * 30} ${250 + i * 10}, ${560 + i * 26} ${300 + i * 8} S ${700 + i * 20} ${330 + i * 6}, ${800} ${320 + i * 10}`} stroke="currentColor" strokeWidth={i === 3 ? 1.4 : 0.9} />
      ))}
    </svg>
  );
}
