"use client";

import { marineMaxima, periodFor, windChangeNote, windSteps } from "@/lib/trip";
import { seasValue, windValue } from "@/components/coast/FishingBrief";
import { Figure } from "@/v4/ui/Figure";
import { clock, instantMin, localMin, periodName, type Cond } from "./Lanes";

/**
 * The trip's three figures (v11): the NWS wind and seas for the trip period and NOAA's next tide turn,
 * each with its own source; what is measured and the notices live in the lanes and the status line, once.
 * Never a go or no-go.
 */
export function TripFigures({ cond, now, today, tripDate, asSaved }: { cond: Cond; now: Date | null; today: string | null; tripDate: string | null; asSaved: string | null }) {
  const c = cond.data;
  const period = periodFor(c, tripDate, today);
  const max = period ? marineMaxima(period.text) : null;
  const wind = period ? windValue(period.text, max?.windKt ?? null) : null;
  const seas = period ? seasValue(period.text, max?.seasFt ?? null) : null;
  const gust = period?.text.match(/\bgusts?\s+(?:up\s+)?to\s+(\d+)\s*kt/i)?.[1];
  const windNote = period ? (windChangeNote(windSteps(period.text)) ?? (gust ? `gusts to ${gust} kt` : undefined)) : undefined;

  let tide: { ft: number; type: "H" | "L"; m: number } | null = null;
  if (c?.tides.status === "ok" && today && now) {
    const nowMin = instantMin(now.toISOString(), today) ?? 0;
    const tripK = tripDate ? Math.round((Date.parse(`${tripDate}T00:00:00Z`) - Date.parse(`${today}T00:00:00Z`)) / 86_400_000) : 0;
    const from = tripK > 0 ? tripK * 1440 : nowMin;
    const turns = c.tides.data.map((t) => ({ ft: t.ft, type: t.type as "H" | "L", m: localMin(t.at, today) })).sort((a, b) => a.m - b.m);
    tide = turns.find((t) => t.m > from) ?? null;
  }
  const tripOther = !!(tripDate && today && tripDate !== today);
  const seaLink = period ? `Marine forecast (NWS), ${periodName(period, !!asSaved)}` : cond.loading ? "Reading the NWS forecast…" : c?.marine.status === "ok" ? (asSaved ? "No NWS period saved for this day" : "No NWS period for this day yet") : "NWS zone forecast not available here";
  const reading = cond.loading && !c;

  return (
    <section aria-labelledby="glance-h" className="coast-figures-wrap" data-testid="decision-row">
      <h2 id="glance-h" className="sr-only">
        The trip at a glance{asSaved ? `, ${asSaved}` : ""}
      </h2>
      <div className="coast-figures">
        <Figure
          testId="glance-wind"
          label="Wind"
          value={wind ?? "—"}
          unit={wind ? "kt" : undefined}
          age={
            <a href="#sea-h" className="nx-link nx-focus" data-testid="glance-sea-link">
              {seaLink}
            </a>
          }
        />
        <Figure testId="glance-seas" label="Seas" value={seas ?? "—"} unit={seas ? "ft" : undefined} age={period && max?.seasFt != null ? `up to ${max.seasFt} ft · NWS` : reading ? "reading…" : "NWS"} />
        <Figure
          testId="glance-tide"
          label="Tide"
          value={tide ? tide.ft.toFixed(1) : "—"}
          unit={tide ? `ft ${tide.type === "H" ? "high" : "low"}` : undefined}
          age={tide ? `NOAA, next turn ${tripOther ? `${tripDate!.slice(5).replace("-", "/")} ` : ""}${clock(tide.m)}${tripOther ? "" : ` · ${tide.type === "H" ? "rising" : "falling"} now`}` : cond.loading ? "reading…" : "no predictions here"}
        />
      </div>
      {windNote && (
        <p className="coast-wind-note" data-testid="glance-wind-changes">
          Wind {windNote}
        </p>
      )}
    </section>
  );
}
