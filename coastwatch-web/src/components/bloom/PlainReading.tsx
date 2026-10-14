"use client";

import type { ObsStation, ObsVariable } from "@/generated/observations";
import { summaryOf } from "@/lib/observations";
import { ageWords, ageDays } from "@/lib/coast";
import { formatDate } from "@/lib/time";
import { Mark } from "@/components/ui/Mark";

/**
 * Bloom Intelligence in Coastal mode: what this station's latest samples say, in plain words,
 * against the station's own last 12 months. No thresholds are invented and nothing is rated;
 * the full analysis (all quantities, charts, seasons, methods) is one tap below.
 */
const ROWS: { id: "pDA" | "pn_seriata" | "pn_delicatissima"; what: string }[] = [
  { id: "pDA", what: "Toxin (domoic acid) in seawater" },
  { id: "pn_seriata", what: "Pseudo-nitzschia, the algae that can make it (large cells)" },
  { id: "pn_delicatissima", what: "Pseudo-nitzschia (small cells)" },
];

const fmt = (v: number) => (v >= 100 ? Math.round(v).toLocaleString("en-US") : Number(v.toPrecision(2)).toString());

export function PlainReading({ station, vars, now }: { station: ObsStation; vars: Record<string, ObsVariable>; now: Date | null }) {
  const rows = ROWS.map((r) => ({ ...r, s: summaryOf(station, r.id), v: vars[r.id] })).filter((r) => r.s && r.s.n_measured > 0 && r.s.last_date && r.v);
  return (
    <section aria-labelledby="plain-h" className="rounded-[18px] border border-hairline bg-surface px-4 py-4 sm:px-5" data-testid="plain-reading">
      <p className="cw-kind text-measured"><Mark kind="measured" />Measured at this pier, about weekly</p>
      <h2 id="plain-h" className="mt-1 font-display text-[22px] font-medium leading-snug text-ink">
        What the latest samples show
      </h2>
      {rows.length === 0 ? (
        <p className="mt-2 text-[15px] text-ink-2">This station has no toxin or Pseudo-nitzschia measurements.</p>
      ) : (
        <ul className="mt-3 divide-y divide-hairline">
          {rows.map(({ id, what, s, v }) => {
            const last = s!.last_date as string;
            const age = now ? ageDays(last, now) : null;
            const old = age != null && age > 30;
            return (
              <li key={id} className="py-2.5" data-testid={`plain-${id}`}>
                <p className="text-[13px] text-ink-3">{what}</p>
                <p className="text-[17px] font-medium text-ink">
                  {s!.last_value == null ? "Not detected or not reported" : `${fmt(s!.last_value)} ${v!.units}`}
                  <span className={`ml-2 text-[14px] font-normal ${old ? "text-warning" : "text-ink-2"}`}>
                    {formatDate(last, { year: true })}
                    {age != null ? ` · ${ageWords(age)}` : ""}
                  </span>
                </p>
                <p className="text-[13px] text-ink-3">
                  {s!.n_last_365d > 0 && s!.max_last_365d != null
                    ? `Highest here in the last 12 months: ${fmt(s!.max_last_365d)} ${v!.units} (${s!.n_last_365d} samples).`
                    : "No samples here in the last 12 months."}
                </p>
              </li>
            );
          })}
        </ul>
      )}
      <p className="mt-2 border-t border-hairline pt-2.5 text-[13px] text-ink-3">
        A pier sample describes the water at this pier on that day. It is not a measurement of seafood; CDPH tests shellfish, and the agencies decide closures.
      </p>
    </section>
  );
}
