"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { PortsCollection } from "@/generated/schema";
import type { StationLite } from "@/lib/data";
import { Icon } from "@/v4/ui/Icon";
import { readHomePort } from "@/v4/state/homePort";
import { SPECIES } from "@/content/species";
import { sitePlaces } from "@/lib/coast";

/**
 * A search result. v6: species (choosing one opens My Coast for that species) and My Coast's
 * shellfish harvest sites (CDPH monitoring sites, e.g. Clam Beach) are found too; `aliases` are other
 * words a result is found by (a species' own match words: "rockfish" finds groundfish).
 */
export type SearchHit = { kind: "harbor" | "station" | "region" | "species" | "site"; id: string; label: string; sub: string; aliases?: string[]; lat?: number; lon?: number };

/** Offered when the box is empty: the harbours and piers most of Neritic's questions are about
 *  (harbours by CDFW port code, so a renamed display name never drops one; stations by name). */
const FEATURED: [SearchHit["kind"], string][] = [
  ["harbor", "550"],
  ["harbor", "592"],
  ["harbor", "593"],
  ["station", "Santa Cruz Wharf"],
  ["station", "Monterey Wharf"],
  ["harbor", "201"],
  ["harbor", "452"],
  ["harbor", "606"],
];

/** Other names people use for a CDFW port (by port code): found by search, said on the result. */
const ALIASES: Record<string, string[]> = {
  "452": ["Pillar Point", "Princeton", "Half Moon Bay"],
  "602": ["Port San Luis", "Avila Beach"],
  "220": ["Humboldt Bay"],
  "223": ["Noyo Harbor"],
  "473": ["Bodega"],
};

const norm = (s: string) => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
const KIND_ORDER: Record<SearchHit["kind"], number> = { harbor: 0, species: 1, site: 2, station: 3, region: 4 };

/** The species and harvest sites search offers (pure, so it is tested with the ranking). */
export function speciesHits(): SearchHit[] {
  return SPECIES.filter((s) => s.id !== "any").map((s) => ({ kind: "species" as const, id: s.id, label: s.label, sub: "Species · opens My Coast for it", aliases: s.match.filter((m) => m.length > 3) }));
}
export function siteHits(): SearchHit[] {
  return sitePlaces().map((p) => ({ kind: "site" as const, id: p.key, label: p.name, sub: `Harvest site · ${p.note}`, lat: p.lat, lon: p.lon }));
}

/**
 * How well a place matches what was typed (lower is better; null: no match): the exact name, then a
 * name that starts with it, then a word of the name (or another name for the place) that starts with
 * it, then a name that contains it, then its county or kind.
 */
export function matchRank(h: SearchHit, q: string): number | null {
  const n = norm(q.trim());
  if (!n) return null;
  const label = norm(h.label);
  const aliases = (h.kind === "harbor" ? (ALIASES[h.id] ?? []) : (h.aliases ?? [])).map(norm);
  if (label === n || aliases.includes(n)) return 0;
  if (label.startsWith(n) || aliases.some((a) => a.startsWith(n))) return 1;
  const words = (t: string) => t.split(/[\s/·,()-]+/).filter(Boolean);
  if ([label, ...aliases].some((t) => words(t).some((w) => w.startsWith(n)))) return 2;
  if (label.includes(n) || aliases.some((a) => a.includes(n))) return 3;
  if (norm(h.sub).includes(n)) return 4;
  return null;
}

/** The places matching a query, best first (rank, then harbours before stations before regions, then name). */
export function rankHits(all: SearchHit[], q: string, limit = 12): SearchHit[] {
  return all
    .map((h) => ({ h, r: matchRank(h, q) }))
    .filter((x): x is { h: SearchHit; r: number } => x.r != null)
    .sort((a, b) => a.r - b.r || KIND_ORDER[a.h.kind] - KIND_ORDER[b.h.kind] || a.h.label.localeCompare(b.h.label))
    .slice(0, limit)
    .map((x) => x.h);
}

/** The other name a hit was found by, when its own name does not show the query. */
export function aliasMatched(h: SearchHit, q: string): string | null {
  const n = norm(q.trim());
  if (!n || norm(h.label).includes(n)) return null;
  return (h.kind === "harbor" ? (ALIASES[h.id] ?? []) : (h.aliases ?? [])).find((a) => norm(a).includes(n)) ?? null;
}

/** Places to go: harbours, shore stations, harvest sites and regions, and species. "/" focuses it from anywhere in the workspace. */
export function PlaceSearch({ ports, stations, regions, onPick, current }: { ports: PortsCollection | null; stations: StationLite[] | null; regions: { id: string; label: string }[]; onPick: (h: SearchHit) => void; current: string }) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [i, setI] = useState(0);
  const input = useRef<HTMLInputElement | null>(null);
  const id = useId();
  const all = useMemo<SearchHit[]>(
    () => [
      ...(ports?.features ?? []).map((f) => ({ kind: "harbor" as const, id: String(f.properties.port_code), label: f.properties.display_name ?? f.properties.name, sub: `Harbour · ${f.properties.county ?? ""}`.trim() })),
      ...(stations ?? []).map((s) => ({ kind: "station" as const, id: s.station_id, label: s.name, sub: "Shore station · CalHABMAP" })),
      ...regions.map((r) => ({ kind: "region" as const, id: r.id, label: r.label, sub: "Region" })),
      ...siteHits(),
      ...speciesHits(),
    ],
    [ports, stations, regions],
  );
  // the reader's home port (kept on this device by My Coast) leads the empty list; read when the list opens
  const [home, setHome] = useState<number | null>(null);
  useEffect(() => {
    if (open) setHome(readHomePort());
  }, [open]);
  const hits = useMemo(() => {
    // an empty box offers the places most looked at (then the regions), so the common question is two taps
    if (!norm(q.trim())) {
      const mine = home != null ? all.find((h) => h.kind === "harbor" && h.id === String(home)) : undefined;
      const featured = FEATURED.map(([kind, key]) => all.find((h) => h.kind === kind && (kind === "harbor" ? h.id === key : h.label === key))).filter((h): h is SearchHit => !!h && h !== mine);
      return [...(mine ? [{ ...mine, sub: `Your home port · ${mine.sub}` }] : []), ...featured, ...all.filter((h) => h.kind === "region")].slice(0, 12);
    }
    return rankHits(all, q);
  }, [q, all, home]);
  useEffect(() => setI(0), [q]);
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.key === "/" && !(t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable))) {
        e.preventDefault();
        input.current?.focus();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, []);
  // a phone: the search shares the header row, so its prompt is one word and its list spans the screen
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 699px)");
    const f = () => setNarrow(mq.matches);
    f();
    mq.addEventListener("change", f);
    return () => mq.removeEventListener("change", f);
  }, []);
  // the workspace moves focus to what was picked (and back here when it is closed)
  const pick = (h: SearchHit) => {
    onPick(h);
    setQ("");
    setOpen(false);
  };
  return (
    // a phone: the field shares the header row; focused, it spans the row (over the brand and the notices)
    <div className="@container relative w-full max-w-[420px] max-[699px]:focus-within:fixed max-[699px]:focus-within:inset-x-3 max-[699px]:focus-within:top-3 max-[699px]:focus-within:z-[60] max-[699px]:focus-within:w-auto max-[699px]:focus-within:max-w-none" id="place-search" tabIndex={-1} data-testid="place-search">
      <div className="relative flex h-12 items-center gap-2 rounded-full border border-n-line-strong bg-n-surface pl-3.5 pr-2 focus-within:ring-2 focus-within:ring-n-accent md:h-11 @max-[96px]:justify-center @max-[96px]:px-0">
        <Icon name="search" className="h-[18px] w-[18px] shrink-0 text-n-ink-3" />
        <input
          ref={input}
          id={`${id}-in`}
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-list`}
          aria-activedescendant={open && hits[i] ? `${id}-o${i}` : undefined}
          aria-label="Search harbours, stations, harvest sites, regions and species"
          placeholder={current || (narrow ? "Search" : "Search harbours, stations, species…")}
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 120)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") setI((x) => Math.min(hits.length - 1, x + 1));
            else if (e.key === "ArrowUp") setI((x) => Math.max(0, x - 1));
            else if (e.key === "Enter" && hits[i]) pick(hits[i]);
            else if (e.key === "Escape") {
              setOpen(false);
              input.current?.blur();
            } else return;
            e.preventDefault();
          }}
          className="h-full min-w-0 flex-1 bg-transparent text-[15px] text-n-ink outline-none placeholder:text-n-ink-2 @max-[96px]:absolute @max-[96px]:inset-0 @max-[96px]:rounded-full @max-[96px]:opacity-0"
          data-ring="container"
          data-testid="place-search-input"
        />
        <kbd className="hidden rounded border border-n-line px-1.5 text-[12px] text-n-ink-3 md:block">/</kbd>
      </div>
      {open && hits.length > 0 && (
        <ul id={`${id}-list`} role="listbox" aria-label="Places" className="nx-pop nx-origin-left absolute left-0 right-0 top-[52px] z-50 max-h-[60vh] overflow-auto max-[699px]:fixed max-[699px]:inset-x-3 max-[699px]:top-[64px] rounded-[18px] border border-n-line bg-n-surface p-1.5 shadow-[var(--nx-shadow)]" data-testid="place-search-list">
          {hits.map((h, k) => (
            <li
              key={`${h.kind}-${h.id}`}
              id={`${id}-o${k}`}
              role="option"
              aria-selected={k === i}
              onMouseDown={(e) => {
                e.preventDefault();
                pick(h);
              }}
              onMouseEnter={() => setI(k)}
              className={`nx-row flex min-h-12 cursor-pointer items-center gap-3 rounded-lg px-3 ${k === i ? "bg-n-surface-2" : ""}`}
            >
              <Icon name={h.kind === "harbor" ? "harbor" : h.kind === "station" ? "station" : h.kind === "species" ? "book" : h.kind === "site" ? "locate" : "map"} className="h-[18px] w-[18px] shrink-0 text-n-ink-3" />
              <span className="min-w-0">
                <span className="block truncate text-[15px] font-semibold text-n-ink">{h.label}</span>
                <span className="block truncate text-[12.5px] text-n-ink-3">
                  {h.sub}
                  {aliasMatched(h, q) ? ` · also called ${aliasMatched(h, q)}` : ""}
                </span>
              </span>
            </li>
          ))}
        </ul>
      )}
      {open && q.trim() !== "" && hits.length === 0 && (
        // nothing matched: say so, and offer the next step instead of an empty box
        <div role="status" className="nx-pop nx-origin-left absolute left-0 right-0 top-[52px] z-50 max-[699px]:fixed max-[699px]:inset-x-3 max-[699px]:top-[64px] rounded-[18px] border border-n-line bg-n-surface p-3 text-n-ink shadow-[var(--nx-shadow)]" data-testid="place-search-empty">
          <p className="text-[14.5px] font-semibold">No harbour, station or region matches “{q.trim()}”</p>
          <p className="mt-0.5 text-[13.5px] leading-snug text-n-ink-2">Try a harbour or town name, a county, a harvest site or a species. Or tap the water on the map to read any point.</p>
          <button
            type="button"
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => {
              setQ("");
              input.current?.focus();
            }}
            className="nx-focus mt-1 inline-flex min-h-11 items-center text-[14px] font-semibold text-n-accent underline underline-offset-2"
            data-testid="place-search-clear"
          >
            Show the common harbours
          </button>
        </div>
      )}
    </div>
  );
}
