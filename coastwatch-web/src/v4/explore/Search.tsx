"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { PortsCollection } from "@/generated/schema";
import type { StationLite } from "@/lib/data";
import { Icon } from "@/v4/ui/Icon";

export type SearchHit = { kind: "harbor" | "station" | "region"; id: string; label: string; sub: string };

/** Offered when the box is empty: the harbours and piers most of Neritic's questions are about. */
const FEATURED: [SearchHit["kind"], string][] = [
  ["harbor", "Monterey"],
  ["harbor", "Moss Landing"],
  ["harbor", "Santa Cruz"],
  ["station", "Santa Cruz Wharf"],
  ["station", "Monterey Wharf"],
  ["harbor", "Crescent City"],
  ["harbor", "Half Moon Bay"],
  ["harbor", "Morro Bay"],
];

const norm = (s: string) => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");

/** Places to go: harbours, shore stations and regions. "/" focuses it from anywhere in the workspace. */
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
    ],
    [ports, stations, regions],
  );
  const hits = useMemo(() => {
    const n = norm(q.trim());
    // an empty box offers the places most looked at (then the regions), so the common question is two taps
    if (!n) {
      const featured = FEATURED.map(([kind, label]) => all.find((h) => h.kind === kind && h.label === label)).filter((h): h is SearchHit => !!h);
      return [...featured, ...all.filter((h) => h.kind === "region")].slice(0, 12);
    }
    return all.filter((h) => norm(h.label).includes(n) || norm(h.sub).includes(n)).slice(0, 12);
  }, [q, all]);
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
  const pick = (h: SearchHit) => {
    onPick(h);
    setQ("");
    setOpen(false);
    input.current?.blur();
  };
  return (
    <div className="relative w-full max-w-[420px]" id="place-search" tabIndex={-1} data-testid="place-search">
      <div className="flex h-12 items-center gap-2 rounded-full md:h-11 border border-n-line-strong bg-n-surface pl-3.5 pr-2 focus-within:ring-2 focus-within:ring-n-accent">
        <Icon name="search" className="h-[18px] w-[18px] shrink-0 text-n-ink-3" />
        <input
          ref={input}
          id={`${id}-in`}
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-list`}
          aria-activedescendant={open && hits[i] ? `${id}-o${i}` : undefined}
          aria-label="Search harbours, stations and regions"
          placeholder={current || "Search harbours, stations…"}
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
          className="h-full min-w-0 flex-1 bg-transparent text-[15px] text-n-ink outline-none placeholder:text-n-ink-2"
          data-ring="container"
          data-testid="place-search-input"
        />
        <kbd className="hidden rounded border border-n-line px-1.5 text-[12px] text-n-ink-3 md:block">/</kbd>
      </div>
      {open && hits.length > 0 && (
        <ul id={`${id}-list`} role="listbox" aria-label="Places" className="nx-rise absolute left-0 right-0 top-[52px] z-50 max-h-[60vh] overflow-auto rounded-2xl border border-n-line bg-n-surface p-1.5 shadow-[var(--nx-shadow)]" data-testid="place-search-list">
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
              className={`flex min-h-12 cursor-pointer items-center gap-3 rounded-lg px-3 ${k === i ? "bg-n-surface-2" : ""}`}
            >
              <Icon name={h.kind === "harbor" ? "harbor" : h.kind === "station" ? "station" : "map"} className="h-[18px] w-[18px] shrink-0 text-n-ink-3" />
              <span className="min-w-0">
                <span className="block truncate text-[15px] font-semibold text-n-ink">{h.label}</span>
                <span className="block truncate text-[12.5px] text-n-ink-3">{h.sub}</span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
