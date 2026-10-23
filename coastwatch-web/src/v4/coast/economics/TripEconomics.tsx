"use client";

import { useEffect, useState } from "react";
import type { Audience } from "@/lib/coast";
import type { SpeciesId } from "@/content/species";
import { decodeShare, draftKey, SHARE_PARAM, type Draft } from "@/lib/tripEconomicsStore";
import { useMounted, useTripData } from "./useTripData";
import { initialDraft } from "./draft";
import Worksheet from "./Worksheet";

export type TripEconomicsProps = {
  place: { key: string; name: string; portCode?: string | number | null };
  species: SpeciesId;
  audience: Audience;
  tripDate: string | null;
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

/**
 * Trip economics (v7): the briefing's one interactive instrument, open. A live result line, the four
 * numbers a skipper changes between trips (fishery or vessel, fuel price, catch, price), then the full
 * worksheet folded in the same panel. Everything comes from the reader's own inputs and stays on this
 * device. Rendered only after mount: the server (and a page restored from saved HTML) shows a fixed-height
 * placeholder, so nothing from this device's storage can mismatch hydration.
 */
export function TripEconomics(props: TripEconomicsProps) {
  const mounted = useMounted();
  return (
    <section id="trip-economics" aria-labelledby="trip-economics-h" className="scroll-mt-4 rounded-[20px] border border-n-line bg-n-surface px-4 pb-3 pt-4 shadow-[0_1px_2px_rgba(11,24,32,0.05)] sm:px-5 sm:pt-5" data-testid="trip-economics-section">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3">
        <h2 id="trip-economics-h" className="nx-title text-n-ink">
          Trip economics
        </h2>
        <span className="text-[12.5px] text-n-ink-3">Kept on this device only</span>
      </div>
      <p className="mt-0.5 text-[14px] text-n-ink-2" data-testid="econ-kicker">
        Scenario from your inputs, not a forecast.
      </p>
      {mounted ? <Body {...props} /> : <div aria-hidden className="h-[460px]" data-testid="econ-placeholder" />}
    </section>
  );
}

function Body({ place, species, audience, tripDate }: TripEconomicsProps) {
  const data = useTripData();
  const key = draftKey(place.key, species);
  const [shared] = useState<Draft | null>(() => sharedFromHash());
  useEffect(() => {
    if (shared) document.getElementById("trip-economics")?.scrollIntoView({ block: "start" });
  }, [shared]);
  const working = data?.meta.drafts[key]?.draft ?? null;
  return <Worksheet key={key} place={place} species={species} audience={audience} tripDate={tripDate} shared={shared} initial={working ?? initialDraft(species)} />;
}
