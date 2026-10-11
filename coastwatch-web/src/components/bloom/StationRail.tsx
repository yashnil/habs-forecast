"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef } from "react";
import type { ObsStation } from "@/generated/observations";
import type { FreshnessPolicy } from "@/generated/schema";
import { REGION_LABEL } from "@/content/copy";
import { fmtDate } from "@/components/charts/StationChart";
import { headline } from "@/lib/bloom";
import { CALIFORNIA_BOUNDS, REGION_BOUNDS, monthYear, stationFreshness, summaryOf } from "@/lib/observations";
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

/** Two significant figures, trailing zeros kept (0.20, not 0.2), as spec §5.2. */
const sig2 = (v: number) => (v >= 100 ? v.toFixed(0) : v.toPrecision(2));

/** One plain line about particulate domoic acid at a station (spec §3.2 item 2). Never implies absence. */
export function pdaLine(s: ObsStation, policy: FreshnessPolicy | null, now: Date | null): string {
  const sm = summaryOf(s, "pDA");
  if (!sm || sm.n_measured === 0) return "pDA not measured here";
  const h = policy && now ? headline(sm, policy, now) : null;
  if (h?.kind === "old") return `pDA last measured ${monthYear(sm.last_date!)}`;
  if (sm.last_qualifier === "reported_zero") return `pDA reported 0 · ${fmtDate(Date.parse(`${sm.last_date}T12:00:00Z`), { month: "short", day: "numeric" })}`;
  return `pDA ${sm.last_value != null ? sig2(sm.last_value) : "—"} ng/mL`;
}

export function StationRail({
  stations,
  selected,
  onSelect,
  policy,
  now,
  showMap = true,
  testid = "station-list",
}: {
  stations: ObsStation[];
  selected: ObsStation;
  onSelect: (id: string) => void;
  policy: FreshnessPolicy | null;
  now: Date | null;
  showMap?: boolean;
  testid?: string;
}) {
  const groups = groupStations(stations);
  // bring the selected station into view inside the rail's own scroll box (never the page)
  const navRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const nav = navRef.current;
    const box = nav?.closest<HTMLElement>("[data-rail-scroll]") ?? null;
    const el = nav?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!box || !el) return;
    const top = el.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
    if (top < box.scrollTop || top > box.scrollTop + box.clientHeight - 60) box.scrollTop = Math.max(0, top - box.clientHeight / 3);
  }, [selected.station_id]);
  const points: StationPoint[] = stations
    .filter((s) => s.lat != null && s.lon != null)
    .map((s) => ({ id: s.station_id, name: s.name, lat: s.lat!, lon: s.lon!, recency: stationFreshness(s, policy, now)?.state ?? "unavailable" }));
  return (
    <div className="flex min-h-0 flex-col gap-3">
      {showMap && (
        <div className="shrink-0 overflow-hidden rounded-lg border border-hairline bg-[#0b1d33]" data-testid="station-map">
          <div className="h-[220px]">
            <StationMap stations={points} selected={selected.station_id} bounds={REGION_BOUNDS[selected.region ?? ""] ?? CALIFORNIA_BOUNDS} onSelect={onSelect} />
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
            <div key={g.region}>
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
