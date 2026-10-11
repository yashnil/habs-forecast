"use client";

import Link from "next/link";
import type { ObsStation } from "@/generated/observations";
import type { FreshnessPolicy } from "@/generated/schema";
import { REGION_LABEL } from "@/content/copy";
import { fmtDate } from "@/components/charts/StationChart";
import { samplingStrip } from "@/lib/bloom";
import { stationFreshness, summaryOf, cadenceLabel } from "@/lib/observations";
import { formatDateTimePT } from "@/lib/time";
import { FreshDot } from "./FreshDot";

/**
 * The title block (M6 §0.2 item 3): the station, then when it was last sampled and how regularly,
 * read as one statement. The 120-day strip has one tick per sampling visit, shades the gap since the
 * last visit, and marks today, so "how fresh is this" is visible before any number.
 */
export function StationHeader({ station, policy, now, portCode }: { station: ObsStation; policy: FreshnessPolicy | null; now: Date | null; portCode: number | null }) {
  const f = stationFreshness(station, policy, now);
  const last = station.sample_times.at(-1) ?? null;
  const strip = now ? samplingStrip(station.sample_times, now) : null;
  // cadence from the most-sampled quantity this year
  const best = station.summaries.reduce((a, s) => (s.n_last_365d > (a?.n_last_365d ?? -1) ? s : a), summaryOf(station, "pDA"));
  const region = REGION_LABEL[station.region ?? "other"] ?? station.region;
  const mapHref = `/?region=${station.region ?? "california"}${portCode != null ? `&port=${portCode}` : ""}`;

  return (
    <header className="grid gap-x-10 gap-y-5 lg:grid-cols-[minmax(0,1fr)_340px]" data-testid="station-header">
      <div className="min-w-0 space-y-2">
        <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-measured">Shore station · {region} · CalHABMAP</p>
        <h1 id="station-h" className="font-display text-[34px] font-medium leading-[1.08] text-ink sm:text-[44px]">
          {station.name}
        </h1>
        <p className="flex flex-wrap gap-x-2 text-[13px] text-ink-3">
          {station.lat != null && station.lon != null && (
            <span className="font-mono">
              {station.lat.toFixed(3)}°N {Math.abs(station.lon).toFixed(3)}°W
            </span>
          )}
          {station.location_code && <span className="font-mono">· {station.location_code}</span>}
          {station.nearest_port_name && (
            <span>
              · {station.nearest_port_km} km from {station.nearest_port_name}
            </span>
          )}

        </p>
        <Link href={mapHref} className="inline-block text-[14px] font-medium text-accent underline-offset-2 hover:underline" data-testid="station-map-link">
          View on the Ocean Map →
        </Link>
      </div>

      <div className="space-y-2 border-t-2 pt-2 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0" style={{ borderColor: f?.state === "current" ? "#17704a" : f?.state === "stale" ? "#86590a" : "var(--cw-hairline-strong)" }} data-testid="freshness">
        <p className="text-[12px] font-semibold uppercase tracking-[0.08em] text-ink-3">Latest sample</p>
        <p className="font-display text-[26px] font-medium leading-tight text-ink" data-testid="last-sample">
          {last ? fmtDate(Date.parse(last), { month: "long", day: "numeric", year: "numeric" }) : "No samples"}
        </p>
        {f && (
          <p className="flex items-center gap-2 text-[13.5px] text-ink-2">
            <FreshDot f={f} />
            {f.ageDays != null && <span>{f.ageDays === 0 ? "today" : f.ageDays === 1 ? "yesterday" : `${f.ageDays} days ago`}</span>}
          </p>
        )}
        {strip && <SamplingStrip strip={strip} />}
        <p className="text-[13px] leading-snug text-ink-3">
          {best && best.n_last_365d ? `Sampled ${cadenceLabel(best.median_interval_days_365d)} (${best.n_last_365d} samples in 12 months). ` : "No samples in the last 12 months. "}
          Lab results arrive days to weeks after sampling.
        </p>
      </div>

      {station.status === "carried_forward" && (
        <p className="rounded-md border border-warning/40 bg-[#fff7e6] px-3 py-2 text-[13px] text-ink-2 lg:col-span-2" data-testid="station-carried">
          <b className="font-semibold text-ink">The latest update could not reach this station&apos;s data.</b> Showing what was retrieved {station.retrieved_at ? formatDateTimePT(station.retrieved_at) : "earlier"}, with the original sample dates; newer samples may exist.
        </p>
      )}
      {f?.state === "historical" && (
        <p className="rounded-md border border-serious/40 bg-[#fdeeee] px-3 py-2 text-[13px] text-ink-2 lg:col-span-2" data-testid="station-historical">
          <b className="font-semibold text-ink">No recent samples.</b> The newest is {f.ageDays} days old, so these values do not describe current conditions.
        </p>
      )}
    </header>
  );
}

function SamplingStrip({ strip }: { strip: ReturnType<typeof samplingStrip> }) {
  const W = 300;
  const H = 34;
  const x = (t: number) => Math.round(((t - strip.t0) / (strip.t1 - strip.t0)) * (W - 2) + 1);
  const gapFrom = strip.last != null && strip.last > strip.t0 ? strip.last : strip.t0;
  const months: number[] = [];
  const d = new Date(strip.t0);
  for (let k = 1; k <= 5; k++) {
    const t = Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + k, 1);
    if (t < strip.t1 - 14 * 86_400_000) months.push(t); // leave room for "today"
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="block max-w-[340px]" role="img" aria-label={`${strip.visits.length} sampling visits in the last 120 days${strip.daysSinceLast != null ? `; the latest ${strip.daysSinceLast} days ago` : ""}.`} data-testid="sampling-strip" data-visits={strip.visits.length}>
      <rect x={x(gapFrom)} y={2} width={Math.max(0, x(strip.t1) - x(gapFrom))} height={18} fill="var(--cw-surface-3)" />
      {strip.visits.map((t) => (
        <line key={t} x1={x(t)} x2={x(t)} y1={3} y2={19} stroke="var(--cw-chl)" strokeWidth={2} />
      ))}
      <line x1={W - 1} x2={W - 1} y1={0} y2={22} stroke="var(--cw-ink)" strokeWidth={1.5} />
      {months.map((t) => (
        <text key={t} x={x(t)} y={32} textAnchor="middle" className="fill-[var(--cw-ink-3)] text-[11px]">
          {new Date(t).toLocaleDateString("en-US", { month: "short", timeZone: "UTC" })}
        </text>
      ))}
      <text x={W - 2} y={32} textAnchor="end" className="fill-[var(--cw-ink)] text-[11px] font-semibold">
        today
      </text>
    </svg>
  );
}
