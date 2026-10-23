"use client";

import type { ReactNode } from "react";
import type { Audience } from "@/lib/coast";
import { marineMaxima, periodFor, windChangeNote, windSteps } from "@/lib/trip";
import { speciesPhrase, type Species } from "@/content/species";
import { seasValue, windValue } from "@/components/coast/FishingBrief";
import { clockLabel, kmLabel } from "@/v4/ui/time";
import { Icon } from "@/v4/ui/Icon";
import { Caps, Reading, type Family } from "./ui/Instrument";
import { clock, instantMin, localMin, periodName, waterReading, type Cond } from "./Lanes";

/**
 * The trip decision row (v10 pass 2): one line of instrument readings above the fold, each with its
 * family colour, source and time: the NWS wind and seas for the trip period, NOAA's next tide turn,
 * the official notices tied to these waters (a count, not verified) and the worksheet's result or the
 * one line of what it still needs. Never a go or no-go. Readings with no source nearby are not drawn as
 * empty tiles: one small line says what is not measured.
 */
export function DecisionRow({
  cond,
  now,
  today,
  tripDate,
  asSaved,
  notices,
  species,
  audience,
  econ,
}: {
  cond: Cond;
  now: Date | null;
  today: string | null;
  tripDate: string | null;
  asSaved: string | null;
  notices: { here: number; namingSpecies: number; unclear: number };
  species: Species;
  audience: Audience;
  econ: ReactNode;
}) {
  const c = cond.data;
  const period = periodFor(c, tripDate, today);
  const max = period ? marineMaxima(period.text) : null;
  const wind = period ? windValue(period.text, max?.windKt ?? null) : null;
  const seas = period ? seasValue(period.text, max?.seasFt ?? null) : null;
  const gust = period?.text.match(/\bgusts?\s+(?:up\s+)?to\s+(\d+)\s*kt/i)?.[1];
  const windNote = period ? (windChangeNote(windSteps(period.text)) ?? (gust ? `gusts to ${gust} kt` : undefined)) : undefined;
  const fc = period ? "NWS" : cond.loading ? "reading NWS…" : "no NWS period";
  // a reading from today says its time; an older one (a saved copy) says its date too
  const when = (iso: string) => {
    const full = clockLabel(iso);
    if (!now || asSaved) return full;
    const [d, ...rest] = full.split(", ");
    return d === clockLabel(now.toISOString()).split(", ")[0] && rest.length ? rest.join(", ") : full;
  };

  // NOAA's next predicted turn after now (or the trip day's first), and which way the water is going
  let tide: { ft: number; type: "H" | "L"; m: number } | null = null;
  if (c?.tides.status === "ok" && today && now) {
    const nowMin = instantMin(now.toISOString(), today) ?? 0;
    const tripK = tripDate ? Math.round((Date.parse(`${tripDate}T00:00:00Z`) - Date.parse(`${today}T00:00:00Z`)) / 86_400_000) : 0;
    const from = tripK > 0 ? tripK * 1440 : nowMin;
    const turns = c.tides.data.map((t) => ({ ft: t.ft, type: t.type as "H" | "L", m: localMin(t.at, today) })).sort((a, b) => a.m - b.m);
    tide = turns.find((t) => t.m > from) ?? null;
  }
  const tripOther = tripDate && today && tripDate !== today;

  const buoy = c?.buoy.status === "ok" ? c.buoy : null;
  const wave = buoy?.data.waveFt ?? null;
  const water = waterReading(c);
  const noticeNote = species.id !== "any" ? `${notices.namingSpecies || "none"} ${notices.namingSpecies === 1 ? "names" : "name"} ${speciesPhrase(species)}` : audience !== "all" ? `${notices.namingSpecies || "none"} for this activity` : "tied to these waters";

  return (
    <section aria-labelledby="glance-h" className="flex flex-col gap-2" data-testid="decision-row">
      <h2 id="glance-h" className="sr-only">
        The trip at a glance{asSaved ? `, ${asSaved}` : ""}
      </h2>
      <dl className="grid grid-cols-2 border-y border-n-line-strong max-[359px]:grid-cols-1 sm:grid-cols-4 xl:grid-cols-[1.2fr_0.8fr_0.95fr_0.8fr_1.45fr]" data-testid="glance-grid">
        <Cell pos={0} family="conditions" caps={`Wind · ${fc}`} testId="glance-wind" value={wind} unit="kt" note={windNote ? <span className="max-sm:hidden">then {windNote.replace(/^then /, "")}</span> : period ? periodName(period, !!asSaved) : undefined} />
        <Cell pos={1} family="conditions" caps={`Seas · ${fc}`} testId="glance-seas" value={seas} unit="ft" note={period && max?.seasFt != null ? `up to ${max.seasFt} ft` : undefined} />
        <Cell
          pos={2}
          family="conditions"
          caps="Tide · NOAA"
          testId="glance-tide"
          value={tide ? tide.ft.toFixed(1) : null}
          unit={tide ? `ft ${tide.type === "H" ? "high" : "low"}` : undefined}
          note={tide ? `next turn ${tripOther ? `${tripDate!.slice(5).replace("-", "/")} ` : ""}${clock(tide.m)}${tripOther ? "" : ` · ${tide.type === "H" ? "rising" : "falling"} now`}` : cond.loading ? "reading…" : "no predictions here"}
        />
        <div className={CELL[3]} data-testid="glance-notices-cell">
          <dt>
            <Caps family="official">Notices</Caps>
          </dt>
          <dd className={DD}>
            <a href="#official-h" className="nx-focus self-start rounded-[2px] hover:underline hover:decoration-n-official-line hover:underline-offset-4" data-testid="glance-notices">
              <Reading value={String(notices.here)} unit="tied here" className="text-[22px] min-[360px]:text-[26px] xl:text-[28px]" />
            </a>
            <span className="text-[12.5px] leading-snug text-n-ink-2">
              <span className="text-n-official">not verified</span> · {notices.here ? noticeNote : "none listed"}
              {notices.unclear > 0 && ` · +${notices.unclear} unclear`}
            </span>
          </dd>
        </div>
        <div className={CELL[4]} data-testid="brief-econ">
          {econ}
        </div>
      </dl>
      {/* one small line: the advisory, the forecast it comes from, and what is (or is not) measured nearby */}
      {/* its height is held while the sources answer, so late data never moves the page (CLS) */}
      <p className="flex min-h-[104px] flex-wrap content-start items-baseline gap-x-3 gap-y-1 text-[13px] leading-snug text-n-ink-2 sm:min-h-[40px]" data-testid="glance-source">
        {windNote && (
          <span className="basis-full sm:hidden" data-testid="glance-wind-changes">
            Wind {windNote}
          </span>
        )}
        {c?.alerts.status === "ok" && c.alerts.data.length > 0 && (
          <span className="inline-flex items-baseline gap-1.5 text-n-conditions" data-testid="glance-alert">
            <Icon name="alert" className="h-3.5 w-3.5 self-center" /> NWS {c.alerts.data[0].event}
            {c.alerts.data[0].ends ? <span className="text-n-ink-2"> until {clockLabel(c.alerts.data[0].ends)}</span> : null}
          </span>
        )}
        <a href="#sea-h" className="nx-focus rounded-[2px] text-n-ink underline decoration-n-line-strong underline-offset-4 hover:decoration-n-ink" data-testid="glance-sea-link">
          {period ? `Marine forecast (NWS), ${periodName(period, !!asSaved)}` : cond.loading ? "Reading the NWS forecast…" : c?.marine.status === "ok" ? (asSaved ? "No NWS period saved for this day" : "No NWS period for this day yet") : "NWS zone forecast not available here"}
        </a>
        {c?.marine.status === "ok" && <span>{c.marine.data.zone}{c.marine.data.updated ? `, issued ${when(c.marine.data.updated)}` : ""}</span>}
        <span data-testid="glance-measured">
          {wave || water ? (
            <>
              Measured
              {wave && buoy ? ` · waves ${wave.value.toFixed(1)} ft${buoy.data.periodS ? ` at ${Math.round(buoy.data.periodS.value)} s` : ""}, buoy ${buoy.station ? `${buoy.station.id} ${kmLabel(buoy.station.km)} out` : ""} (${when(wave.at)}), not the harbour` : ""}
              {water ? (
                <span data-testid="glance-water">
                  {" "}
                  · water {water.f} °F, measured {when(water.at)} at {water.where}
                </span>
              ) : null}
            </>
          ) : cond.loading ? null : (
            "Waves and water temperature: not measured nearby."
          )}
        </span>
        {tripOther && <span>Measured readings are the latest, not the trip day&rsquo;s.</span>}
      </p>
    </section>
  );
}

// under 360 px the four readings are one column of compact rows (label · value · note), not a 2×2 grid
// of half-empty tiles
const ROW = "max-[359px]:grid max-[359px]:grid-cols-[96px_minmax(0,1fr)] max-[359px]:items-baseline max-[359px]:gap-x-3 max-[359px]:border-l-0 max-[359px]:py-2 max-[359px]:pl-0";
const CELL = [
  `flex min-w-0 flex-col gap-1 border-n-line py-3 pr-3 ${ROW}`,
  `flex min-w-0 flex-col gap-1 border-l border-n-line py-3 pl-3 pr-3 sm:pl-4 ${ROW} max-[359px]:border-t`,
  `flex min-w-0 flex-col gap-1 border-t border-n-line py-3 pr-3 sm:border-l sm:border-t-0 sm:pl-4 ${ROW}`,
  `flex min-w-0 flex-col gap-1 border-l border-t border-n-line py-3 pl-3 pr-3 sm:border-t-0 sm:pl-4 ${ROW}`,
  "col-span-2 flex min-w-0 flex-col gap-1 border-t border-n-line py-3 max-[359px]:col-span-1 sm:col-span-4 xl:col-span-1 xl:border-l xl:border-t-0 xl:pl-4",
];
const DD = "flex min-h-[74px] min-w-0 flex-col gap-1 max-[359px]:min-h-0 max-[359px]:gap-0.5 sm:min-h-[84px]";

function Cell({ pos, family, caps, value, unit, note, testId }: { pos: number; family: Family; caps: string; value: string | null; unit?: string; note?: ReactNode; testId: string }) {
  return (
    <div className={CELL[pos]} data-testid={testId}>
      <dt>
        <Caps family={family}>{caps}</Caps>
      </dt>
      <dd className={DD}>
        <Reading value={value ?? "—"} unit={value ? unit : undefined} muted={!value} className="text-[22px] min-[360px]:text-[26px] xl:text-[28px]" />
        {note && <span className="line-clamp-2 text-[12.5px] leading-snug text-n-ink-2 xl:line-clamp-3">{note}</span>}
      </dd>
    </div>
  );
}
