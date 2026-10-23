"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import type { Audience } from "@/lib/coast";
import type { SpeciesId } from "@/content/species";
import type { FieldKey } from "@/lib/tripEvaluation";
import { decodeShare, SHARE_PARAM, type Draft } from "@/lib/tripEconomicsStore";
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
 * The briefing's reading of the trip economics (v10 pass 2), one cell of the decision row: the result the
 * user's own numbers already reach (net once it is worked out, else gross as soon as catch and price
 * exist, else the trip cost), and one line of what the rest still needs, each a link straight to its
 * field in the worksheet. The cell has a fixed height before hydration (storage is read after mount).
 */
export function TripGlance({ place, species, audience, onOpen }: TripEconomicsProps & { onOpen: (focus?: FieldKey | null) => void }) {
  const mounted = useMounted();
  const w = useWorkingDraft(place.key, species);
  const ev = useMemo(() => (w ? evaluateDraft(w.draft, audience) : null), [w, audience]);
  const rec = ev?.audience === "recreational";
  const entered = w ? hasInput(w.draft) : false;
  // the headline: the furthest result the inputs reach
  const head: { label: string; value: number; signed: boolean } | null = !ev
    ? null
    : rec
      ? ev.tripCost.status === "value"
        ? { label: "trip cost", value: ev.tripCost.value, signed: false }
        : null
      : ev.netToBoat.status === "value"
        ? { label: "net to boat", value: ev.netToBoat.value, signed: true }
        : ev.gross.status === "value"
          ? { label: "gross", value: ev.gross.value, signed: false }
          : ev.tripCost.status === "value"
            ? { label: "trip cost", value: ev.tripCost.value, signed: false }
            : null;
  const cost = ev && !rec && head?.label !== "trip cost" && ev.tripCost.status === "value" ? ev.tripCost.value : null;
  return (
    <>
      <dt className="flex flex-wrap items-baseline justify-between gap-x-3">
        <Caps>
          <span id="trip-economics-h">{rec ? "Trip cost" : "Trip worksheet"}</span> · hypothetical
        </Caps>
        <span className="sr-only" data-testid="econ-kicker">
          Scenario from your inputs, not a forecast.
        </span>
      </dt>
      <dd id="trip-economics" className="flex min-h-[64px] min-w-0 scroll-mt-4 flex-col gap-1" data-testid="econ-glance">
        {!mounted || !ev || !w ? null : head ? (
          <>
            <button type="button" onClick={() => onOpen(null)} className="nx-focus self-start rounded-[2px] text-left hover:underline hover:decoration-n-line-strong hover:underline-offset-4" data-testid="econ-open">
              <span className={`nx-value whitespace-nowrap text-[20px] min-[360px]:text-[26px] xl:text-[28px] ${head.value < 0 ? "text-n-down" : ""}`} data-testid="econ-glance-result" data-label={head.label}>
                {money(head.value, head.signed)}
                <small>{head.label}</small>
              </span>
            </button>
            <span className="text-[12.5px] leading-snug text-n-ink-2">
              {cost !== null && <span data-testid="econ-glance-cost">cost {money(cost)} · </span>}
              {ev.missingFields.length > 0 ? (
                <span data-testid="econ-glance-needs">
                  still needs{" "}
                  {ev.missingFields.map((k, i) => (
                    <span key={k}>
                      {i > 0 && <span aria-hidden> · </span>}
                      <button type="button" onClick={() => onOpen(k)} className="nx-focus rounded-[2px] font-medium text-n-ink underline decoration-dashed decoration-n-ink-3 underline-offset-[3px] hover:decoration-n-ink" data-testid="econ-glance-need" data-field={k}>
                        {needLabel(k, w.draft)}
                      </button>
                    </span>
                  ))}
                </span>
              ) : (
                <>your numbers, not a forecast{w.from === "saved" && w.name ? ` · “${w.name}”` : ""}</>
              )}
            </span>
          </>
        ) : (
          <>
            <button type="button" onClick={() => onOpen(entered ? (ev.missingFields[0] ?? null) : null)} className="nx-press nx-focus inline-flex min-h-11 items-center gap-2 self-start rounded-[3px] border border-n-line-strong px-3.5 text-[14.5px] font-medium text-n-ink" data-testid="econ-open">
              {entered ? "Open the worksheet" : "Start the worksheet"} <Icon name="chevronRight" className="h-4 w-4" />
            </button>
            <span className="text-[12.5px] leading-snug text-n-ink-2" data-testid={entered ? "econ-glance-needs" : "econ-glance-empty"}>
              {entered ? (
                <>
                  still needs{" "}
                  {ev.missingFields.map((k, i) => (
                    <span key={k}>
                      {i > 0 && <span aria-hidden> · </span>}
                      <button type="button" onClick={() => onOpen(k)} className="nx-focus rounded-[2px] font-medium text-n-ink underline decoration-dashed decoration-n-ink-3 underline-offset-[3px] hover:decoration-n-ink" data-testid="econ-glance-need" data-field={k}>
                        {needLabel(k, w.draft)}
                      </button>
                    </span>
                  ))}
                </>
              ) : rec ? (
                "Nothing entered yet: fuel and cost from your own numbers."
              ) : (
                "Nothing entered yet: gross, costs, crew pay and net from your own numbers."
              )}
            </span>
          </>
        )}
      </dd>
    </>
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
