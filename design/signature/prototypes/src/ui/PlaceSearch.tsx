import { useId, useMemo, useRef, useState } from "react";
import type { Port, Station } from "../data/types";
import type { LngLat } from "../lib/geo";

/**
 * Place search: a combobox over ports and sampling stations (no geocoder dependency). Suggestions
 * show *what kind of place* each is with the same marks as the map (port = white dot, station =
 * measured dot). Arrow keys, Enter, Escape; results are announced. Typing nothing shows the user's
 * recent places first ("returning to previously selected waters").
 */
export interface PlaceHit { kind: "port" | "station" | "recent"; id: string; label: string; sub: string; at: LngLat }

export function PlaceSearch({ ports, stations, recent, current, onPick, onFocusChange }: {
  ports?: Port[];
  stations?: Station[];
  recent: PlaceHit[];
  current: string;
  onPick: (h: PlaceHit) => void;
  onFocusChange?: (on: boolean) => void;
}) {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const hits = useMemo<PlaceHit[]>(() => {
    const all: PlaceHit[] = [
      ...(ports ?? []).map((p) => ({ kind: "port" as const, id: `port-${p.port_code}`, label: p.display_name, sub: `Port · ${p.county} County`, at: [p.lon, p.lat] as LngLat })),
      ...(stations ?? []).map((s) => ({ kind: "station" as const, id: s.station_id, label: s.name, sub: "Shore sampling station", at: [s.lon, s.lat] as LngLat })),
    ];
    const t = q.trim().toLowerCase();
    if (!t) return [...recent.slice(0, 3), ...all.filter((h) => h.kind === "port").slice(0, 6)];
    return all.filter((h) => h.label.toLowerCase().includes(t)).sort((a, b) => Number(!a.label.toLowerCase().startsWith(t)) - Number(!b.label.toLowerCase().startsWith(t))).slice(0, 8);
  }, [q, ports, stations, recent]);
  const pick = (h: PlaceHit) => { onPick(h); setQ(""); setOpen(false); input.current?.blur(); };
  return (
    <div className="search" data-open={open && hits.length > 0}>
      <svg className="search-i" width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="5" fill="none" stroke="currentColor" strokeWidth="1.5" /><path d="M11 11l3.5 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>
      <input ref={input} role="combobox" aria-expanded={open} aria-controls={`${id}-list`} aria-autocomplete="list"
        aria-activedescendant={open && hits[active] ? `${id}-${active}` : undefined}
        placeholder={current ? `${current} — search ports and piers` : "Search ports and piers, or tap the water"}
        value={q}
        onChange={(e) => { setQ(e.target.value); setOpen(true); setActive(0); }}
        onFocus={() => { setOpen(true); onFocusChange?.(true); }}
        onBlur={() => { setTimeout(() => setOpen(false), 120); onFocusChange?.(false); }}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown") { e.preventDefault(); setOpen(true); setActive((a) => Math.min(hits.length - 1, a + 1)); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
          else if (e.key === "Enter" && hits[active]) { e.preventDefault(); pick(hits[active]); }
          else if (e.key === "Escape") { setOpen(false); }
        }} />
      {open && hits.length > 0 && (
        <ul className="search-list enter-fade" id={`${id}-list`} role="listbox" aria-label="Places">
          {hits.map((h, i) => (
            <li key={h.kind + h.id} id={`${id}-${i}`} role="option" aria-selected={i === active} onPointerDown={(e) => { e.preventDefault(); pick(h); }} onPointerEnter={() => setActive(i)}>
              <span className={`search-mark search-mark-${h.kind}`} aria-hidden="true" />
              <span><span className="search-l">{h.label}</span><span className="t-meta search-s">{h.kind === "recent" ? `Recent · ${h.sub}` : h.sub}</span></span>
            </li>
          ))}
        </ul>
      )}
      <span className="sr-only" aria-live="polite">{open && q ? `${hits.length} places` : ""}</span>
    </div>
  );
}
