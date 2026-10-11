"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import type { ObsStation } from "@/generated/observations";
import type { FreshnessPolicy } from "@/generated/schema";
import { REGION_LABEL } from "@/content/copy";
import { fmtDate } from "@/components/charts/StationChart";
import { pdaLine } from "@/lib/bloom";
import { CALIFORNIA_BOUNDS, REGION_BOUNDS, stationFreshness } from "@/lib/observations";
import type { StationPoint } from "./StationMap";
import { FreshGlyph } from "./FreshDot";

const StationMap = dynamic(() => import("./StationMap"), {
  ssr: false,
  loading: () => <div className="h-full w-full bg-[#0b1d33]" aria-label="Loading map" />,
});

/** North to south (spec §3.2 item 2). */
export const REGION_ORDER = ["north_coast", "mendocino_sonoma", "sf_bay_farallones", "monterey_bay", "central_coast", "southern_california"];

export function groupStations(stations: ObsStation[]): { region: string; stations: ObsStation[] }[] {
  const order = [...REGION_ORDER, "other"];
  const byRegion = new Map<string, ObsStation[]>();
  for (const s of stations) {
    const r = order.includes(s.region ?? "") ? s.region! : "other";
    byRegion.set(r, [...(byRegion.get(r) ?? []), s]);
  }
  return order.filter((r) => byRegion.has(r)).map((r) => ({ region: r, stations: byRegion.get(r)!.sort((a, b) => (b.lat ?? 0) - (a.lat ?? 0)) }));
}

export function StationRail({
  stations,
  selected,
  onSelect,
  policy,
  now,
  showMap = true,
  testid = "station-list",
  scrollBox = "rail",
}: {
  stations: ObsStation[];
  selected: ObsStation;
  onSelect: (id: string) => void;
  policy: FreshnessPolicy | null;
  now: Date | null;
  showMap?: boolean;
  testid?: string;
  /** which box scrolls: the desktop rail's list, or the phone picker dialog */
  scrollBox?: "rail" | "dialog";
}) {
  const groups = groupStations(stations);
  // The rail (and its map) is only displayed from 1280 px. MapLibre must not start inside a hidden
  // rail: on a throttled phone that cost ~340 ms of main-thread time for a map nobody sees.
  const [mapVisible, setMapVisible] = useState(false);
  useEffect(() => {
    if (!showMap) return;
    const mq = window.matchMedia("(min-width: 1280px)");
    const on = () => setMapVisible(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [showMap]);
  // bring the selected station into view inside the rail's own scroll box (never the page)
  const navRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const nav = navRef.current;
    const box = scrollBox === "dialog" ? (nav?.closest<HTMLElement>('[role="dialog"]') ?? null) : (nav?.closest<HTMLElement>("[data-rail-scroll]") ?? null);
    const el = nav?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!box || !el) return;
    // align the selected station's region heading with the top of the list, never mid-row
    const group = el.closest<HTMLElement>("[data-region]") ?? el;
    const top = group.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
    const bottom = el.getBoundingClientRect().bottom - box.getBoundingClientRect().top + box.scrollTop;
    if (top < box.scrollTop || bottom > box.scrollTop + box.clientHeight) box.scrollTop = Math.max(0, top - 4);
  }, [selected.station_id, scrollBox]);
  const points: StationPoint[] = stations
    .filter((s) => s.lat != null && s.lon != null)
    .map((s) => ({ id: s.station_id, name: s.name, lat: s.lat!, lon: s.lon!, recency: stationFreshness(s, policy, now)?.state ?? "unavailable" }));
  return (
    <div className="flex min-h-0 flex-col gap-3">
      {showMap && (
        <div className="shrink-0 overflow-hidden rounded-lg border border-hairline bg-[#0b1d33]" data-testid="station-map">
          <div className="h-[220px]">
            {mapVisible && <StationMap stations={points} selected={selected.station_id} bounds={REGION_BOUNDS[selected.region ?? ""] ?? CALIFORNIA_BOUNDS} onSelect={onSelect} />}
          </div>
        </div>
      )}
      <nav ref={navRef} aria-label="Shore stations" data-testid={testid} data-rail-scroll className="min-h-0 overflow-y-auto pb-4 pr-1 [scrollbar-width:thin]">
        <p className="px-1 pb-2 text-[13px] text-ink-2">
          <b className="font-semibold text-ink">{stations.length} shore stations</b>, north to south. pDA: particulate domoic acid.
        </p>
        <p className="flex flex-wrap gap-x-3 px-1 pb-2 text-[13px] text-ink-3" aria-label="Key">
          <span className="inline-flex items-center gap-1">
            <FreshGlyph state="current" /> sampled recently
          </span>
          <span className="inline-flex items-center gap-1">
            <FreshGlyph state="stale" /> stale
          </span>
          <span className="inline-flex items-center gap-1">
            <FreshGlyph state="historical" /> not current
          </span>
        </p>
        <div className="space-y-3">
          {groups.map((g) => (
            <div key={g.region} data-region={g.region}>
              <p className="px-1 pb-1 text-[12px] font-semibold uppercase tracking-[0.08em] text-ink-3">{REGION_LABEL[g.region] ?? "Other"}</p>
              <ul>
                {g.stations.map((s) => {
                  const f = stationFreshness(s, policy, now);
                  const active = s.station_id === selected.station_id;
                  const last = s.sample_times.at(-1);
                  return (
                    <li key={s.station_id}>
                      <button
                        type="button"
                        onClick={() => onSelect(s.station_id)}
                        aria-current={active ? "page" : undefined}
                        data-testid={`station-row-${s.station_id}`}
                        className={`w-full rounded-md px-2 py-1.5 text-left transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${active ? "bg-navy-900 text-on-navy" : "hover:bg-surface-2"}`}
                      >
                        <span className="flex items-center justify-between gap-2">
                          <span className={`truncate text-[14px] font-medium ${active ? "text-on-navy" : "text-ink"}`}>{s.name}</span>
                          {f && <FreshGlyph state={f.state} />}
                          {f && <span className="sr-only">{f.state}</span>}
                        </span>
                        <span className={`block truncate text-[13px] ${active ? "text-on-navy-2" : "text-ink-3"}`}>
                          {s.status === "failed" ? "Unavailable in this update" : `${last ? fmtDate(Date.parse(last), { month: "short", day: "numeric", year: last.slice(0, 4) === String(now?.getFullYear() ?? "") ? undefined : "numeric" }) : "No samples"} · ${pdaLine(s, policy, now)}`}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      </nav>
    </div>
  );
}
