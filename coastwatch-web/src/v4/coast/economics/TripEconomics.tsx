"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import type { Audience } from "@/lib/coast";
import type { SpeciesId } from "@/content/species";
import type { FieldKey } from "@/lib/tripEvaluation";
import { decodeShare, SHARE_PARAM, type Draft } from "@/lib/tripEconomicsStore";
import { num } from "@/lib/tripEconomics";
import type { SeasonStatus } from "@/v4/coast/season";
import { Caps } from "@/v4/coast/ui/Instrument";
import { Icon } from "@/v4/ui/Icon";
import { useMounted, useWorkingDraft } from "./useTripData";
import { evaluateDraft, hasInput, money, needLabel } from "./model";

// the worksheet's code loads when it is opened (and is held by the offline pack like every lazy chunk)
const Worksheet = dynamic(() => import("./Worksheet"), { ssr: false, loading: () => <div aria-hidden className="h-[900px]" data-testid="econ-placeholder" /> });

export type TripEconomicsProps = {
  place: { key: string; name: string; portCode?: string | number | null };
  species: SpeciesId;
  audience: Audience;
  tripDate: string | null;
  season: SeasonStatus;
};

/** "#trip=…" (a shared scenario), read once after mount. */
function sharedFromHash(): Draft | null {
  try {
    const m = window.location.hash.match(new RegExp(`[#&]${SHARE_PARAM}=([A-Za-z0-9_-]+)`));
    return m ? decodeShare(m[1]) : null;
  } catch {
    return null;
  }
}

export const hasSharedScenario = () => typeof window !== "undefined" && new RegExp(`[#&]${SHARE_PARAM}=`).test(window.location.hash);

/**
 * The briefing's glance at the trip economics (v10): what the user's own numbers already work out to,
 * and exactly which inputs the rest waits for, each a link straight to its field in the worksheet. The
 * box has a fixed height before hydration (device storage is read only after mount), so nothing shifts.
 */
export function TripGlance({ place, species, audience, season, onOpen }: TripEconomicsProps & { onOpen: (focus?: FieldKey | null) => void }) {
  const mounted = useMounted();
  const w = useWorkingDraft(place.key, species);
  const ev = useMemo(() => (w ? evaluateDraft(w.draft, audience) : null), [w, audience]);
  const rec = ev?.audience === "recreational";
  const entered = w ? hasInput(w.draft) : false;
  const rows = !ev
    ? []
    : rec
      ? ([
          ["Fuel used", ev.fuelGallons, (v: number) => `${num(v, 1)} gal`],
          ["Cost of the trip", ev.tripCost, (v: number) => money(v)],
        ] as const)
      : ([
          ["Gross", ev.gross, (v: number) => money(v)],
          ["Trip cost", ev.tripCost, (v: number) => money(v)],
          ["Net to boat", ev.netToBoat, (v: number) => money(v, true)],
        ] as const);
  return (
    <section id="trip-economics" aria-labelledby="trip-economics-h" className="scroll-mt-4 flex flex-col gap-3" data-testid="trip-economics-section">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h2 id="trip-economics-h">
          <Caps>{rec ? "Cost of the trip" : "Trip economics"} · hypothetical</Caps>
        </h2>
        <span className="text-[12.5px] text-n-ink-2" data-testid="econ-kicker">
          Scenario from your inputs, not a forecast.
        </span>
      </div>
      <div className="min-h-[188px]" data-testid="econ-glance">
        {!mounted || !ev || !w ? (
          <div aria-hidden className="h-[188px]" />
        ) : (
          <>
            {season.closed && season.item && (
              <p className="mb-2 text-[13.5px] leading-snug text-n-ink" data-testid="econ-glance-closed">
                {season.item.name} is on CDFW&rsquo;s closed sport list here: your numbers, not permission to fish.
              </p>
            )}
            {!entered ? (
              <p className="py-2 text-[14.5px] leading-snug text-n-ink-2" data-testid="econ-glance-empty">
                Nothing entered yet. {rec ? "The worksheet works out fuel and the cost of the trip from your own numbers." : "The worksheet works out gross, fuel, costs, crew pay and net from your own numbers, step by step, and says what each result still needs."}
              </p>
            ) : (
              <dl className="flex flex-col">
                {rows.map(([label, o, fmt]) => (
                  <div key={label} className="grid grid-cols-[7.5rem_minmax(0,1fr)] items-baseline gap-3 border-t border-n-line py-2" data-testid={`econ-glance-${label.toLowerCase().replace(/\s+/g, "-")}`}>
                    <dt className="text-[13.5px] text-n-ink-2">{label}</dt>
                    <dd className="min-w-0">
                      {o.status === "value" ? (
                        <span className={`nx-value text-[24px] ${o.value < 0 ? "text-n-down" : ""}`}>{fmt(o.value)}</span>
                      ) : o.status === "missing" ? (
                        <span className="text-[13.5px] leading-snug text-n-ink-2">
                          <span className="nx-value mr-2 text-[24px] text-n-ink-3" aria-hidden>
                            —
                          </span>
                          needs {o.missing.map((k) => needLabel(k, w.draft)).join(" · ")}
                        </span>
                      ) : (
                        <span className="text-[13.5px] text-n-ink-2">{o.status === "unreachable" ? o.reason : "—"}</span>
                      )}
                    </dd>
                  </div>
                ))}
              </dl>
            )}
            {entered && ev.missingFields.length > 0 && (
              <p className="mt-1 text-[13.5px] leading-snug text-n-ink-2" data-testid="econ-glance-needs">
                Still needed:{" "}
                {ev.missingFields.map((k, i) => (
                  <span key={k}>
                    {i > 0 && <span aria-hidden> · </span>}
                    <button type="button" onClick={() => onOpen(k)} className="nx-focus inline-flex min-h-11 items-center rounded-[2px] font-medium text-n-ink underline decoration-dashed decoration-n-ink-3 underline-offset-[3px] hover:decoration-n-ink" data-testid="econ-glance-need" data-field={k}>
                      {needLabel(k, w.draft)}
                    </button>
                  </span>
                ))}
              </p>
            )}
            {w.from === "saved" && w.name && <p className="mt-1 text-[12.5px] text-n-ink-2">From your saved scenario &ldquo;{w.name}&rdquo;.</p>}
          </>
        )}
      </div>
      <button type="button" aria-expanded={false} aria-controls="trip-worksheet" onClick={() => onOpen(null)} className="nx-press nx-focus inline-flex min-h-11 items-center justify-between gap-3 self-stretch rounded-[3px] border border-n-line-strong px-4 text-[14.5px] font-medium text-n-ink sm:self-start" data-testid="econ-open">
        {entered ? "Open the worksheet" : "Start the worksheet"} <Icon name="chevronRight" className="h-4 w-4" />
      </button>
    </section>
  );
}

/** The worksheet view: the working draft (or a shared link's numbers) worked through. */
export function TripWorksheet({ place, species, audience, tripDate, season, focus }: TripEconomicsProps & { focus: FieldKey | null }) {
  const mounted = useMounted();
  const w = useWorkingDraft(place.key, species);
  const [shared] = useState<Draft | null>(() => (typeof window === "undefined" ? null : sharedFromHash()));
  return (
    <section id="trip-worksheet" aria-label="Trip worksheet" className="scroll-mt-4" data-testid="trip-economics-section">
      {mounted && w ? <Worksheet key={`${place.key}|${species}`} place={place} species={species} audience={audience} tripDate={tripDate} shared={shared} initial={w.draft} focus={focus} season={season} /> : <div aria-hidden className="h-[900px]" data-testid="econ-placeholder" />}
    </section>
  );
}
