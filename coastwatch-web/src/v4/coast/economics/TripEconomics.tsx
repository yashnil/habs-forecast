"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import type { Audience } from "@/lib/coast";
import { speciesById, speciesPhrase, type SpeciesId } from "@/content/species";
import { computeTrip, num, usd } from "@/lib/tripEconomics";
import { decodeShare, draftInputs, draftKey, SHARE_PARAM, type Draft } from "@/lib/tripEconomicsStore";
import { Icon } from "@/v4/ui/Icon";
import { useMounted, useTripData } from "./useTripData";
import { initialDraft } from "./draft";

export type TripEconomicsProps = {
  place: { key: string; name: string; portCode?: string | number | null };
  species: SpeciesId;
  audience: Audience;
  tripDate: string | null;
  /** M's one-line sea reading for the trip day, e.g. "NWS for Thu: wind to 25 kt, above your 20 kt limit (your limit, not a safety check)" */
  conditions: { line: string; over: boolean | null; asSaved: string | null } | null;
  notices: { here: number; namingSpecies: number };
};

const Worksheet = dynamic(() => import("./Worksheet"), {
  ssr: false,
  loading: () => <div aria-hidden className="nx-skeleton h-[420px] w-full" />,
});

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
 * Trip economics inside My Coast: a one-line reading of the sea and the notices for context, a compact
 * summary of the trip's numbers, and the worksheet behind one button (loaded only when opened).
 * Everything comes from the user's own inputs and stays on this device. Rendered only after mount: the
 * server (and an offline page restored from saved HTML) shows a fixed-height placeholder, so nothing
 * from this device's storage can mismatch hydration.
 */
export function TripEconomics(props: TripEconomicsProps) {
  const mounted = useMounted();
  return (
    <section id="trip-economics" aria-labelledby="trip-economics-h" className="flex scroll-mt-24 flex-col gap-4 border-t border-n-line pt-7" data-testid="brief-trip-economics">
      <div>
        <h2 id="trip-economics-h" className="font-serif text-[24px] font-semibold leading-tight text-n-ink">
          Trip economics
        </h2>
        <p className="mt-1 text-[15px] text-n-ink-2" data-testid="econ-kicker">
          Scenario from your inputs, not a forecast.
        </p>
      </div>
      {mounted ? <Body {...props} /> : <div aria-hidden className="h-[132px]" data-testid="econ-placeholder" />}
    </section>
  );
}

function Body({ place, species, audience, tripDate, conditions, notices }: TripEconomicsProps) {
  const data = useTripData();
  const key = draftKey(place.key, species);
  const [shared] = useState<Draft | null>(() => sharedFromHash());
  const [open, setOpen] = useState<boolean>(() => shared !== null);
  // the worksheet's numbers as they are typed (the stored draft lags only for a shared link, which is not saved)
  const [live, setLive] = useState<Draft | null>(null);
  const sp = speciesById(species);

  useEffect(() => {
    if (shared) document.getElementById("trip-economics")?.scrollIntoView({ block: "start" });
  }, [shared]);

  const working = data?.meta.drafts[key]?.draft ?? null;
  const summaryDraft = live ?? shared ?? working;
  const r = useMemo(() => (summaryDraft ? computeTrip(draftInputs(summaryDraft)) : null), [summaryDraft]);
  const hasNumbers = r !== null && (r.tripCostUsd !== null || r.grossUsd !== null || r.breakEvenLb.value !== null);

  return (
    <div className="flex flex-col gap-4">
      <ul className="flex flex-col gap-1.5 text-[15px] leading-snug text-n-ink-2" data-testid="econ-context">
        {conditions && (
          <li className="flex gap-2" data-testid="econ-conditions" data-over={conditions.over === true}>
            <Icon name={conditions.over ? "alert" : "info"} className={`mt-0.5 h-[18px] w-[18px] shrink-0 ${conditions.over ? "text-n-official" : "text-n-ink-3"}`} />
            <span className="min-w-0">
              {conditions.line}
              {conditions.asSaved && <span className="text-n-ink-3"> · as saved {conditions.asSaved}</span>}
            </span>
          </li>
        )}
        <li className="flex gap-2" data-testid="econ-notices">
          <Icon name="shield" className="mt-0.5 h-[18px] w-[18px] shrink-0 text-n-ink-3" />
          <span className="min-w-0">
            Notices: {notices.here} tied to these waters
            {sp.id !== "any" && `, ${notices.namingSpecies} naming ${speciesPhrase(sp)}`}
          </span>
        </li>
      </ul>

      <div className="rounded-2xl border border-n-line bg-n-surface px-4 py-3" data-testid="econ-summary">
        {hasNumbers && r ? (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 min-[560px]:grid-cols-4">
            <Fig label="Trip cost" value={usd(r.tripCostUsd)} testId="econ-sum-cost" />
            <Fig label="Gross" value={usd(r.grossUsd)} testId="econ-sum-gross" />
            <Fig label="Net to the boat" value={usd(r.netUsd, { signed: true })} testId="econ-sum-net" />
            <Fig
              label="Break-even"
              value={r.breakEvenLb.note === "none" ? "None at this price" : r.breakEvenLb.value !== null ? `${num(r.breakEvenLb.value)} lb` : r.breakEvenUsdPerLb.value !== null ? `${usd(r.breakEvenUsdPerLb.value, { cents: true })}/lb` : "—"}
              testId="econ-sum-breakeven"
            />
          </dl>
        ) : (
          <p className="text-[15px] text-n-ink-2" data-testid="econ-empty">
            Enter this trip&rsquo;s fuel, costs, crew and your own catch and price to see the cost and the break-even. Nothing is pre-filled for catch or price.
          </p>
        )}
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1">
          <button
            type="button"
            aria-expanded={open}
            aria-controls="econ-worksheet"
            onClick={() => setOpen((o) => !o)}
            className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 rounded-xl bg-n-ink px-4 text-[15px] font-semibold text-n-bg"
            data-testid="econ-toggle"
          >
            {open ? "Close the worksheet" : hasNumbers ? "Open the worksheet" : "Work out this trip"}
            <Icon name="chevronDown" className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`} />
          </button>
          <span className="text-[13.5px] text-n-ink-3">Kept on this device only.</span>
        </div>
      </div>

      <div id="econ-worksheet" hidden={!open}>
        {open && <Worksheet key={key} place={place} species={species} audience={audience} tripDate={tripDate} shared={shared} initial={working ?? initialDraft(species)} onLive={setLive} />}
      </div>
    </div>
  );
}

function Fig({ label, value, testId }: { label: string; value: string; testId: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[13px] font-semibold text-n-ink-3">{label}</dt>
      <dd className="nx-data text-[19px] font-semibold text-n-ink" data-testid={testId}>
        {value}
      </dd>
    </div>
  );
}
