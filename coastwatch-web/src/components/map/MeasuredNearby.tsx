"use client";

import type { StationLite } from "@/lib/data";
import { ageInDays } from "@/lib/freshness";
import { NEARBY_KM, nearbyStations } from "@/lib/stations";
import { formatNumber, unitLabel } from "@/lib/observations";
import { formatDate } from "@/lib/time";

function fmt(l: StationLite["latest"][number]): string {
  if (l.qualifier === "rejected_negative") return "rejected (QC)";
  if (l.value == null) return "not measured";
  if (l.qualifier === "reported_zero") return "reported 0 (not quantified)";
  return `${formatNumber({ kind: l.kind as never }, l.value)} ${unitLabel(l.units)}`;
}

/**
 * Measured nearby (M5 inspector): the latest water samples at CalHABMAP stations within
 * 30 km, each with its own sample date and age. Measurements, not forecasts; a long gap since
 * the last sample is said plainly. Full series are on the Bloom page.
 */
export function MeasuredNearby({ stations, error, lat, lon, now }: { stations: StationLite[] | null; error: string | null; lat: number; lon: number; now: Date | null }) {
  const near = stations ? nearbyStations(stations, lat, lon) : [];
  return (
    <section data-testid="measured-nearby" aria-labelledby="measured-h" className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <h3 id="measured-h" className="text-[13px] font-semibold text-ink">
          Measured nearby
        </h3>
        <span className="text-[11px] text-ink-3">CalHABMAP water samples</span>
      </div>
      {!stations ? (
        <p className="text-[12px] text-ink-3">Station measurements are unavailable{error && error !== "not published" ? ` (${error})` : ""}.</p>
      ) : !near.length ? (
        <p className="text-[12px] text-ink-2" data-testid="measured-none">
          No monitoring station within {NEARBY_KM} km.
        </p>
      ) : (
        <ul className="space-y-2">
          {near.map(({ s, km }) => {
            const pda = s.latest.find((l) => l.variable === "pDA");
            const pn = s.latest.filter((l) => l.variable.startsWith("pn_"));
            const age = pda && now ? ageInDays(pda.date, now) : null;
            return (
              <li key={s.station_id} data-testid={`measured-${s.station_id}`} className="rounded-lg border border-hairline px-2.5 py-2">
                <div className="flex items-baseline justify-between gap-2">
                  <a href={`/bloom?station=${encodeURIComponent(s.station_id)}`} className="text-[13px] font-medium text-accent hover:underline">
                    {s.name}
                  </a>
                  <span className="text-[11px] text-ink-3 tabular">{km < 1 ? "< 1" : km.toFixed(0)} km</span>
                </div>
                {pda ? (
                  <p className="text-[12px] text-ink-2">
                    Particulate domoic acid <span className="font-semibold text-ink tabular">{fmt(pda)}</span> · sampled {formatDate(pda.date, { year: true })}
                    {age != null && age > 30 && <span className="text-warning"> ({age} days ago)</span>}
                  </p>
                ) : (
                  <p className="text-[12px] text-ink-3">No domoic acid measurement published.</p>
                )}
                {pn.length > 0 && (
                  <p className="text-[11.5px] text-ink-3">
                    {pn.map((l, i) => (
                      <span key={l.variable}>
                        {i ? " · " : ""}
                        {l.label.replace(/^Pseudo-nitzschia /, "P.-n. ")} {fmt(l)} ({formatDate(l.date).replace(/^\w+, /, "")})
                      </span>
                    ))}
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}
      <p className="text-[11.5px] text-ink-3">Measured at the station on the date shown, not today and not at this point.</p>
    </section>
  );
}

