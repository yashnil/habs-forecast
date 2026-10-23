"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { Audience } from "@/lib/coast";
import { speciesById, type SpeciesId } from "@/content/species";
import { presetById, type FisheryPreset, type PresetReference } from "@/content/fisheryPresets";
import { DIESEL_REF, NO_PRICE_REF, dieselAge, priceRef } from "@/content/referenceValues";
import { LB_PER_KG, LB_PER_TON, num } from "@/lib/tripEconomics";
import type { CrewChoice, FieldKey, Output, TripEvaluation } from "@/lib/tripEvaluation";
import { MAX_VESSELS, draftKey, type Draft } from "@/lib/tripEconomicsStore";
import { useNow } from "@/lib/useNow";
import { toast } from "@/v4/ui/toast";
import { Icon } from "@/v4/ui/Icon";
import { dayLabel } from "@/v4/ui/time";
import { Action, Caps, Choice, NumIn, TextIn, fieldDomId } from "@/v4/coast/ui/Instrument";
import type { SeasonStatus } from "@/v4/coast/season";
import { notifyTrip, tripStore, useTripData } from "./useTripData";
import { FISHERY_OPTIONS, fieldReferences, setField, sourceShort, switchFishery } from "./draft";
import { TRIP_FUEL_KEYS, econAudience, evaluateDraft, money } from "./model";
import { BreakEven, Headline, Needs, Sensitivity, Waterfall, type WU } from "./Results";
import { Scenarios, ShareLink, StoreStatus, TripDataPanel } from "./Scenarios";

export type WorksheetProps = {
  place: { key: string; name: string };
  species: SpeciesId;
  audience: Audience;
  tripDate: string | null;
  shared: Draft | null;
  initial: Draft;
  /** a field to bring into view and focus on opening (a "Needs …" link on the briefing) */
  focus: FieldKey | null;
  season: SeasonStatus;
};

const ext = { target: "_blank", rel: "noreferrer" } as const;
const clone = (d: Draft): Draft => JSON.parse(JSON.stringify(d));

const CREW_OPTIONS: { value: CrewChoice; label: string; hint: string }[] = [
  { value: "none", label: "No hired crew", hint: "Nothing is paid to crew." },
  { value: "afterExpenses", label: "Share after expenses", hint: "Crew get a % of what is left after shared costs." },
  { value: "ofGross", label: "Share of gross", hint: "Crew get a % of the gross, before any cost." },
  { value: "dayRate", label: "Day rate", hint: "Each crew member is paid per day, whatever the catch." },
];

const CREW_EFFECT: Record<CrewChoice, string> = {
  none: "Net is the gross less fees and every cost.",
  afterExpenses: "The share is taken after fuel and shared costs come off, so it never makes the boat pay crew from a loss; a larger share lowers net, not the catch that covers the costs.",
  ofGross: "The share is taken from every dollar of the gross, so it raises the catch needed to break even.",
  dayRate: "Crew pay is a fixed cost: it adds to what the catch must cover, whatever the catch.",
};

/**
 * The trip worksheet (v10): the user's own numbers in the order money moves (what the catch would bring,
 * fuel, other costs, crew pay), each step with its status and the engine's worked sum, results beside
 * them that appear the moment their own inputs exist. References sit beside fields, tinted, with their
 * source, and are copied in only on a tap. Recreational trips are cost-only. Device-only storage.
 */
export default function Worksheet({ place, species, audience, tripDate, shared, initial, focus }: WorksheetProps) {
  const data = useTripData();
  const key = draftKey(place.key, species);
  const [draft, setDraft] = useState<Draft>(() => clone(shared ?? initial));
  const [fromLink, setFromLink] = useState(shared !== null);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const [fuelMode, setFuelMode] = useState<"trip" | "gallons">(() => ((shared ?? initial).inputs.fuelGal != null ? "gallons" : "trip"));
  const readOnly = data?.status.readOnly ?? false;
  const unitPref = data?.meta.unit ?? "lb";
  const preset = presetById(draft.preset);
  const wu: WU = preset?.unit === "ton" ? { unit: "ton", w: 1 / LB_PER_TON, p: LB_PER_TON } : unitPref === "kg" ? { unit: "kg", w: 1 / LB_PER_KG, p: LB_PER_KG } : { unit: "lb", w: 1, p: 1 };
  const rec = econAudience(audience) === "recreational";
  const ev = useMemo(() => evaluateDraft(draft, audience), [draft, audience]);
  const needed = useMemo(() => new Set<string>(ev.missingFields), [ev]);
  // complete (net, or a recreational trip's cost, is worked out): the inputs fold to one line and the results
  // fill the page. Only a worksheet opened complete starts folded; one completed while typing stays open.
  const complete = rec ? ev.tripCost.status === "value" : ev.netToBoat.status === "value";
  const [editing, setEditing] = useState(() => {
    const e = evaluateDraft(shared ?? initial, audience);
    return focus !== null || (econAudience(audience) === "recreational" ? e.tripCost.status !== "value" : e.netToBoat.status !== "value");
  });
  const folded = complete && !editing;

  const update = (fn: (d: Draft) => void) => {
    const d = clone(draft);
    fn(d);
    setDraft(d);
    if (!fromLink && !readOnly) {
      tripStore().saveDraft(key, d);
      notifyTrip();
    }
  };
  const setNum = (k: string) => (v: number | null) =>
    update((d) => {
      setField(d, k, v);
      delete d.basis[k];
    });

  // "Needs …": bring the exact field into view and focus it (switching the fuel route when it belongs to the other one)
  const focusField = useCallback(
    (k: FieldKey) => {
      let target: string = k;
      if (k === "fuelGal" && fuelMode === "trip") target = TRIP_FUEL_KEYS.find((x) => draft.inputs[x as "roundTripNm"] == null) ?? "roundTripNm";
      if (TRIP_FUEL_KEYS.includes(k) && fuelMode === "gallons") setFuelMode("trip");
      if (k === "crew.kind") target = "crew.kind";
      window.setTimeout(() => {
        const el = document.getElementById(fieldDomId(target));
        if (!el) return;
        el.scrollIntoView({ block: "center", behavior: "auto" });
        el.focus({ preventScroll: true });
      }, 30);
    },
    [fuelMode, draft],
  );
  const focusDone = useRef(false);
  useEffect(() => {
    if (focus && !focusDone.current) {
      focusDone.current = true;
      focusField(focus);
    }
  }, [focus, focusField]);

  // phones: the result line follows the reader through the inputs once its own place has scrolled away
  const headRef = useRef<HTMLDivElement | null>(null);
  const [headSeen, setHeadSeen] = useState(true);
  useEffect(() => {
    const el = headRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setHeadSeen(e.isIntersecting || e.boundingClientRect.top > 0), { rootMargin: "-56px 0px 0px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  const lines = draft.inputs.catch ?? [];
  const sp = speciesById(species);

  return (
    <div className="flex flex-col gap-8" data-testid="econ-worksheet">
      {fromLink && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-l-2 border-n-accent pl-3 text-[15px]" role="status" data-testid="econ-shared">
          <span className="min-w-0">From a shared link, not saved.</span>
          <Action
            disabled={readOnly}
            onClick={() => {
              tripStore().saveDraft(key, draft);
              notifyTrip();
              setFromLink(false);
              toast("Saved to this device");
            }}
            testId="econ-shared-save"
          >
            Save to this device
          </Action>
        </div>
      )}
      <StoreStatus />

      {folded ? (
        <div className="flex flex-col gap-8" data-testid="econ-results" data-folded="true">
          <div className="flex flex-col gap-3 border-y border-n-line-strong py-4">
            <h3 id="econ-results-h" className="flex items-baseline justify-between gap-3">
              <Caps>Result · hypothetical</Caps>
              <span className="text-[12.5px] text-n-ink-2">from the inputs below</span>
            </h3>
            <Headline ev={ev} draft={draft} onFocus={focusField} wu={wu} />
          </div>
          <InputsSummary draft={draft} rec={rec} wu={wu} onEdit={() => setEditing(true)} />
          <div className="grid grid-cols-1 gap-x-14 gap-y-8 lg:grid-cols-2">
            <div className="flex min-w-0 flex-col gap-7">
              {!rec && <Waterfall ev={ev} draft={draft} onFocus={focusField} />}
              {!rec && <BreakEven ev={ev} draft={draft} onFocus={focusField} wu={wu} />}
            </div>
            <div className="flex min-w-0 flex-col gap-7">
              <Sensitivity ev={ev} draft={draft} onFocus={focusField} wu={wu} />
              {ev.warnings.map((w) => (
                <p key={w} className="flex gap-2 text-[14px] text-n-ink" data-testid="econ-warning">
                  <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0 text-n-down" />
                  {w}
                </p>
              ))}
            </div>
          </div>
        </div>
      ) : (
      <div className="flex flex-col gap-8 lg:grid lg:grid-cols-[minmax(0,1fr)_minmax(400px,460px)] lg:items-start lg:gap-x-14">
        {/* results: first on a phone (then the inputs, then the charts); a sticky column on a wide screen */}
        <aside aria-labelledby="econ-results-h" className="max-lg:contents lg:sticky lg:top-4 lg:col-start-2 lg:row-start-1 lg:flex lg:flex-col lg:gap-7" data-testid="econ-results">
          <div ref={headRef} className="order-1 flex flex-col gap-3 border-y border-n-line-strong py-4">
            <h3 id="econ-results-h" className="flex items-baseline justify-between gap-3">
              <Caps>Result · hypothetical</Caps>
              {complete ? (
                <button type="button" onClick={() => setEditing(false)} className="nx-focus inline-flex min-h-11 items-center gap-1 rounded-[2px] text-[13px] text-n-ink underline decoration-n-line-strong underline-offset-4 sm:min-h-8" data-testid="econ-fold">
                  Fold the inputs
                </button>
              ) : (
                <span className="text-[12.5px] text-n-ink-2">updates as you type</span>
              )}
            </h3>
            <Headline ev={ev} draft={draft} onFocus={focusField} wu={wu} />
            {ev.missingFields.length > 0 && (
              <p className="border-t border-n-line pt-2" data-testid="econ-needs">
                <Needs fields={ev.missingFields} draft={draft} onFocus={focusField} lead={rec ? "Still needed:" : "Still needed for net:"} />
              </p>
            )}
          </div>
          <div className="order-3 flex flex-col gap-7">
            {!rec && <Waterfall ev={ev} draft={draft} onFocus={focusField} />}
            {!rec && <BreakEven ev={ev} draft={draft} onFocus={focusField} wu={wu} />}
            <Sensitivity ev={ev} draft={draft} onFocus={focusField} wu={wu} />
            {ev.warnings.map((w) => (
              <p key={w} className="flex gap-2 text-[14px] text-n-ink" data-testid="econ-warning">
                <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0 text-n-down" />
                {w}
              </p>
            ))}
          </div>
        </aside>

        <div className="order-2 flex min-w-0 flex-col lg:col-start-1 lg:row-start-1" data-testid="econ-steps">
          {!rec && (
            <Step n={1} title="What the catch would bring" o={ev.gross} draft={draft} onFocus={focusField} testId="econ-g-catch" sums={[ev.gross]}>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-[minmax(0,1fr)_auto]">
                <label className="flex min-w-0 flex-col gap-1 text-[14px] text-n-ink-2">
                  Fishery (names the catch lines and the references offered)
                  <select
                    value={(draft.preset ?? "") as string}
                    onChange={(e) => update((d) => switchFishery(d, presetById(e.target.value)))}
                    className="nx-select nx-focus min-h-11 w-full rounded-[3px] border border-n-line-strong bg-n-surface pl-3 text-[15px] text-n-ink"
                    data-testid="econ-fishery"
                  >
                    {FISHERY_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.value === "" && species !== "any" ? `${sp.label}: no preset, start blank` : o.label}
                      </option>
                    ))}
                  </select>
                </label>
                {wu.unit !== "ton" ? (
                  <label className="flex flex-col gap-1 text-[14px] text-n-ink-2">
                    Weigh in
                    <select
                      value={unitPref}
                      onChange={(e) => {
                        tripStore().setUnit(e.target.value as "lb" | "kg");
                        notifyTrip();
                      }}
                      className="nx-select nx-focus min-h-11 rounded-[3px] border border-n-line-strong bg-n-surface pl-3 text-[15px] text-n-ink"
                      data-testid="econ-unit"
                    >
                      <option value="lb">Pounds (lb)</option>
                      <option value="kg">Kilograms (kg)</option>
                    </select>
                  </label>
                ) : (
                  <p className="self-end pb-2 text-[13px] text-n-ink-2">Market squid is sold by the short ton (2,000 lb).</p>
                )}
              </div>
              {preset?.note && <p className="nx-margin">{preset.note}</p>}
              {lines.map((c, i) => (
                <CatchLine key={i} i={i} draft={draft} wu={wu} needed={needed} update={update} canRemove={i > 0} />
              ))}
              {lines.length < 10 && (
                <TextAction onClick={() => update((d) => (d.inputs.catch = [...(d.inputs.catch ?? []), { species: `Catch ${(d.inputs.catch ?? []).length + 1}`, ref: null, lb: null, usdPerLb: null }]))} testId="econ-add-catch">
                  <Icon name="plus" className="h-4 w-4" /> Add a catch line
                </TextAction>
              )}
            </Step>
          )}

          <Step n={rec ? 1 : 2} title="Fuel" o={ev.fuelCost} draft={draft} onFocus={focusField} testId="econ-g-vessel" sums={[ev.fuelGallons, ev.fuelCost]}>
            <Choice
              name="fuel-mode"
              legend="How you know the fuel"
              value={fuelMode}
              options={[
                { value: "trip", label: "Work it out from the trip", hint: "distance, speed and burn; hours and burn" },
                { value: "gallons", label: "I know the gallons", hint: "one number for the whole trip" },
              ]}
              onChange={(m) => {
                setFuelMode(m);
                if (m === "trip" && draft.inputs.fuelGal != null) update((d) => (d.inputs.fuelGal = null));
              }}
              testId="econ-fuel-mode"
            />
            {fuelMode === "trip" ? (
              <div className="grid grid-cols-1 gap-x-8 gap-y-4 sm:grid-cols-2">
                <div className="flex flex-col gap-4">
                  <Caps>Steaming</Caps>
                  <NumIn id={fieldDomId("roundTripNm")} label="Round trip" unit="nm" kind="nm" decimals={1} value={draft.inputs.roundTripNm ?? null} onChange={setNum("roundTripNm")} needed={needed.has("roundTripNm")} testId="econ-nm" />
                  <NumIn id={fieldDomId("cruiseKt")} label="Cruising speed" unit="kt" kind="kt" decimals={1} value={draft.inputs.cruiseKt ?? null} onChange={setNum("cruiseKt")} needed={needed.has("cruiseKt")} testId="econ-kt" />
                  <NumIn id={fieldDomId("steamGph")} label="Burn steaming" unit="gal/h" kind="gph" decimals={1} value={draft.inputs.steamGph ?? null} onChange={setNum("steamGph")} needed={needed.has("steamGph")} testId="econ-steam-gph" />
                  <FisheryRefs field="steamGph" draft={draft} update={update} />
                </div>
                <div className="flex flex-col gap-4">
                  <Caps>Fishing</Caps>
                  <NumIn
                    id={fieldDomId("fishingHours")}
                    label={preset?.effort === "dive day" ? "Hours diving" : "Hours fishing"}
                    unit="h"
                    kind="hours"
                    decimals={1}
                    value={draft.inputs.fishingHours ?? null}
                    onChange={setNum("fishingHours")}
                    needed={needed.has("fishingHours")}
                    testId="econ-hours"
                    note={preset && preset.effort !== "trip" ? `One scenario is one ${preset.effort}.` : tripDate ? `For the trip on ${dayLabel(tripDate, true)}.` : undefined}
                  />
                  <NumIn id={fieldDomId("fishingGph")} label="Burn fishing" unit="gal/h" kind="gph" decimals={1} value={draft.inputs.fishingGph ?? null} onChange={setNum("fishingGph")} needed={needed.has("fishingGph")} testId="econ-fish-gph" />
                  <FisheryRefs field="fishingGph" draft={draft} update={update} />
                </div>
              </div>
            ) : (
              <NumIn id={fieldDomId("fuelGal")} label="Gallons for the whole trip" unit="gal" kind="gal" decimals={1} value={draft.inputs.fuelGal ?? null} onChange={setNum("fuelGal")} needed={needed.has("fuelGal")} testId="econ-fuel-gal" />
            )}
            <div className="grid grid-cols-1 gap-3">
              <NumIn id={fieldDomId("fuelUsdPerGal")} label="Fuel price" unit="$/gal" kind="usdPerGal" decimals={3} value={draft.inputs.fuelUsdPerGal ?? null} onChange={setNum("fuelUsdPerGal")} needed={needed.has("fuelUsdPerGal")} testId="econ-fuel-price" />
              <DieselRef used={draft.basis.fuelUsdPerGal === "eia"} onUse={() => update((d) => ((d.inputs.fuelUsdPerGal = DIESEL_REF.usdPerGal), (d.basis.fuelUsdPerGal = "eia")))} />
            </div>
            {!rec && (
              <label className="flex min-h-11 cursor-pointer items-center gap-3 text-[14.5px] text-n-ink">
                <input type="checkbox" checked={draft.inputs.fuelShared !== false} onChange={(e) => update((d) => (d.inputs.fuelShared = e.target.checked))} className="nx-focus h-5 w-5 shrink-0 accent-[var(--nx-accent)]" data-testid="econ-fuel-shared" />
                Fuel comes off the top before crew shares
              </label>
            )}
            <Vessels draft={draft} readOnly={readOnly} update={update} />
          </Step>

          <Step n={rec ? 2 : 3} title={rec ? "Other trip costs" : "Other costs and fees"} o={ev.otherCosts} draft={draft} onFocus={focusField} testId="econ-g-costs" sums={rec ? [ev.otherCosts] : [ev.otherCosts, ev.fees]} optional>
            <ul className="flex flex-col gap-3">
              {(draft.inputs.costs ?? []).map((c, i) => (
                <li key={i} className="grid grid-cols-[minmax(0,1fr)_minmax(0,8.5rem)_auto] items-end gap-3" data-testid={`econ-cost-${i}`}>
                  <TextIn label={`Cost ${i + 1}`} value={c.label} onChange={(v) => update((d) => (d.inputs.costs![i].label = v))} />
                  <NumIn id={fieldDomId(`costs.${i}.usd`)} label={`${c.label || `Cost ${i + 1}`}, amount`} labelHidden unit="$" kind="usd" decimals={0} value={c.usd} onChange={(v) => update((d) => (d.inputs.costs![i].usd = v))} testId={`econ-cost-${i}-usd`} />
                  <button type="button" aria-label={`Remove ${c.label || `cost ${i + 1}`}`} title={`Remove ${c.label || `cost ${i + 1}`}`} onClick={() => update((d) => d.inputs.costs!.splice(i, 1))} className="nx-press nx-focus grid h-11 w-11 place-items-center rounded-[3px] text-n-ink-3 hover:text-n-ink">
                    <Icon name="close" className="h-4 w-4" />
                  </button>
                  {!rec && (
                    <label className="col-span-3 -mt-1 flex min-h-11 cursor-pointer items-center gap-3 text-[13.5px] text-n-ink-2">
                      <input type="checkbox" checked={c.shared} onChange={(e) => update((d) => (d.inputs.costs![i].shared = e.target.checked))} className="nx-focus h-5 w-5 shrink-0 accent-[var(--nx-accent)]" />
                      Shared: comes off before crew shares
                    </label>
                  )}
                </li>
              ))}
            </ul>
            {(draft.inputs.costs ?? []).length < 20 && (
              <TextAction onClick={() => update((d) => (d.inputs.costs = [...(d.inputs.costs ?? []), { label: "", usd: null, shared: true }]))} testId="econ-add-cost">
                <Icon name="plus" className="h-4 w-4" /> Add a cost
              </TextAction>
            )}
            {preset?.potTiers && !rec && <PotFields draft={draft} preset={preset} update={update} />}
            {!rec && (
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="flex flex-col gap-2">
                  <NumIn id={fieldDomId("feesPctOfGross")} label="Fees, share of gross" unit="%" kind="pct" value={draft.inputs.feesPctOfGross ?? null} onChange={setNum("feesPctOfGross")} testId="econ-fees-pct" />
                  <FisheryRefs field="feesPctOfGross" draft={draft} update={update} />
                </div>
                <NumIn id={fieldDomId("feesUsdPerLb")} label={preset?.perLbLabel ?? "Fees, $ per lb"} unit="$/lb" kind="usdPerLb" value={draft.inputs.feesUsdPerLb ?? null} onChange={setNum("feesUsdPerLb")} testId="econ-fees-lb" />
              </div>
            )}
          </Step>

          {rec ? (
            <Step n={3} title="Shared between" o={ev.perPerson} draft={draft} onFocus={focusField} testId="econ-g-split" sums={[ev.perPerson]} optional>
              <NumIn id={fieldDomId("anglers")} label="People sharing the cost" unit="people" kind="people" decimals={0} value={draft.inputs.anglers ?? null} onChange={(v) => update((d) => (d.inputs.anglers = v))} testId="econ-anglers" />
            </Step>
          ) : (
            <Step n={4} title="Crew pay" o={ev.crewPay} draft={draft} onFocus={focusField} testId="econ-g-crew" sums={[ev.crewPay, ev.perCrew]} last>
              <Choice
                name="crew-kind"
                legend="How crew is paid (nothing is chosen for you)"
                value={draft.inputs.crew?.kind ?? null}
                options={CREW_OPTIONS}
                onChange={(k) =>
                  update((d) => {
                    d.inputs.crew = { ...(d.inputs.crew ?? {}), kind: k };
                    if (k === "none") d.inputs.crew.pct = null;
                    if (d.basis["crew.pct"] === "preset") {
                      d.inputs.crew.pct = null;
                      delete d.basis["crew.pct"];
                    }
                  })
                }
                testId="econ-crew"
                firstId={fieldDomId("crew.kind")}
              />
              <CrewParams draft={draft} preset={preset} needed={needed} setNum={setNum} update={update} />
            </Step>
          )}
        </div>
      </div>
      )}

      <div className="grid grid-cols-1 gap-8 border-t border-n-line-strong pt-6 lg:grid-cols-[minmax(0,1fr)_minmax(320px,420px)] lg:gap-x-14">
        <Scenarios place={place} species={species} audience={audience} draft={draft} scenarioId={scenarioId} setScenarioId={setScenarioId} readOnly={readOnly} onOpen={(d) => update((x) => Object.assign(x, clone(d)))} />
        <div className="flex flex-col gap-4">
          <ShareLink draft={draft} />
          <TripDataPanel />
        </div>
      </div>

      {/* phones: a result line above the dock once the result block has scrolled away (in a portal: an
          ancestor transform or size container would pin a fixed bar to itself) */}
      {!headSeen &&
        typeof document !== "undefined" &&
        createPortal(
          <a href="#econ-results-h" className="nx-focus fixed inset-x-0 z-30 block border-t border-n-line-strong bg-n-surface px-4 py-2 lg:hidden" style={{ bottom: "calc(60px + env(safe-area-inset-bottom, 0px))" }} data-testid="econ-bar" aria-label={barLabel(ev)}>
            <span className="mx-auto flex max-w-[640px] items-baseline justify-between gap-3" aria-hidden>
              <span className="nx-caps text-n-ink-2">Hypothetical</span>
              {rec ? (
                <span className="font-data text-[15px] text-n-ink">cost {short(ev.tripCost)}</span>
              ) : (
                <>
                  <span className="font-data text-[15px] text-n-ink">net {short(ev.netToBoat, true)}</span>
                  <span className="font-data text-[14px] text-n-ink-2">cost {short(ev.tripCost)}</span>
                </>
              )}
              <Icon name="chevronRight" className="h-4 w-4 -rotate-90 self-center text-n-ink-3" />
            </span>
          </a>,
          document.body,
        )}
    </div>
  );
}

const short = (o: Output, signed = false) => (o.status === "value" ? money(o.value, signed) : "—");
const barLabel = (ev: TripEvaluation) => (ev.audience === "recreational" ? `Hypothetical trip cost ${short(ev.tripCost)}` : `Hypothetical net to boat ${short(ev.netToBoat, true)}, trip cost ${short(ev.tripCost)}`);

// ---------------------------------------------------------------- pieces

/** One step of the worksheet: number, title, status, its fields, then the engine's worked sums. */
function Step({ n, title, o, draft, onFocus, testId, sums, children, optional = false, last = false }: { n: number; title: string; o: Output; draft: Draft; onFocus: (k: FieldKey) => void; testId: string; sums: Output[]; children: ReactNode; optional?: boolean; last?: boolean }) {
  const status = o.status === "value" ? (optional ? "Counted" : "Worked out") : o.status === "missing" ? null : o.status === "unreachable" ? "No value" : "Not used";
  const formulas = sums.filter((s): s is Extract<Output, { status: "value" }> => s.status === "value");
  const notes = [...new Set(sums.flatMap((s) => (s.status === "value" ? s.assumptions.map((a) => a.note) : [])))];
  return (
    <fieldset className={`min-w-0 border-t border-n-line-strong pb-7 pt-4 ${last ? "" : ""}`} data-testid={testId}>
      <legend className="sr-only">
        {n}. {title}
      </legend>
      <div aria-hidden className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <span className="flex items-baseline gap-3">
          <span className="font-data text-[13px] text-n-ink-3">{String(n).padStart(2, "0")}</span>
          <span className="font-serif font-medium tracking-[-0.015em] text-[22px] leading-tight text-n-ink">{title}</span>
        </span>
        <span className="text-[13px] text-n-ink-2" data-testid={`${testId}-status`}>
          {status ?? (o.status === "missing" ? `Needs ${o.missing.length === 1 ? "1 input" : `${o.missing.length} inputs`}` : "")}
          {optional && o.status !== "value" ? " · optional" : ""}
        </span>
      </div>
      <div className="flex flex-col gap-5">{children}</div>
      <div className="mt-5 flex flex-col gap-1 border-l border-n-line-strong pl-3" data-testid={`${testId}-sum`}>
        {formulas.map((f) => (
          <p key={f.formula} className="font-data text-[13.5px] leading-snug text-n-ink" data-testid="econ-formula">
            {f.formula}
          </p>
        ))}
        {o.status === "missing" && (
          <p>
            <Needs fields={o.missing} draft={draft} onFocus={onFocus} />
          </p>
        )}
        {o.status === "unreachable" && <p className="text-[13.5px] text-n-ink-2">{o.reason}</p>}
        {notes.map((t) => (
          <p key={t} className="nx-margin text-[12.5px]" data-testid="econ-assumption">
            {t}
          </p>
        ))}
        {!formulas.length && o.status !== "missing" && o.status !== "unreachable" && optional && <p className="text-[13px] text-n-ink-2">Blank lines are not counted.</p>}
      </div>
    </fieldset>
  );
}

function TextAction({ onClick, children, testId }: { onClick: () => void; children: ReactNode; testId?: string }) {
  return (
    <button type="button" onClick={onClick} className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 self-start rounded-[3px] px-1 text-[14.5px] text-n-ink underline decoration-n-line-strong underline-offset-4 hover:decoration-n-ink" data-testid={testId}>
      {children}
    </button>
  );
}

/**
 * A reference beside a field: tinted and set apart from anything typed, with its source, period and
 * caveat, and a tap to use it. Once used, the field says "basis: …" until the user types over it.
 */
function RefRow({ testId, label, value, source, note, used, onUse, useLabel, useTestId, stale }: { testId: string; label: ReactNode; value: ReactNode; source: ReactNode; note?: ReactNode; used: boolean; onUse?: () => void; useLabel?: string; useTestId?: string; stale?: ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5 rounded-[3px] bg-n-surface-2 px-3 py-2.5 shadow-[inset_2px_0_0_var(--nx-line-strong)]" data-testid={testId}>
      <p className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="nx-caps text-n-ink-2">Reference · {label}</span>
        <span className="font-data text-[15px] text-n-ink">{value}</span>
      </p>
      <p className="text-[12.5px] leading-snug text-n-ink-2">
        {source}
        {note ? <span className="nx-margin text-[12.5px]"> {note}</span> : null}
      </p>
      {stale}
      {used ? (
        <p className="text-[12.5px] font-medium text-n-ink">In the field above as your starting assumption ({useLabel ? `basis: ${useLabel}` : "basis: reference"}). Type over it to use your own.</p>
      ) : onUse ? (
        <button type="button" onClick={onUse} className="nx-press nx-focus inline-flex min-h-11 items-center self-start rounded-[3px] border border-n-line-strong bg-n-surface px-3 text-[13.5px] font-medium text-n-ink" data-testid={useTestId}>
          Use {value}
        </button>
      ) : null}
    </div>
  );
}

function FisheryRefs({ field, draft, update }: { field: "steamGph" | "fishingGph" | "feesPctOfGross" | "crew.pct"; draft: Draft; update: (fn: (d: Draft) => void) => void }) {
  const refs = fieldReferences(draft, field);
  if (!refs.length) return null;
  const cur = field === "crew.pct" ? (draft.inputs.crew?.pct ?? null) : ((draft.inputs[field as "steamGph"] as number | null | undefined) ?? null);
  const unit = field === "steamGph" || field === "fishingGph" ? " gal/h" : " %";
  return (
    <>
      {refs.map((r: PresetReference) => {
        const used = draft.basis[field] === "preset" && cur === r.value;
        return (
          <RefRow
            key={`${r.source}-${r.value}`}
            testId={`econ-ref-${field.replace(".", "-")}`}
            label={r.label}
            value={`${r.value}${unit}`}
            source={
              <a href={r.sourceUrl} {...ext} className="nx-focus underline decoration-n-line-strong underline-offset-2">
                {sourceShort(r)}
              </a>
            }
            note={r.note ? `${r.note}.` : undefined}
            used={used}
            useLabel={sourceShort(r)}
            useTestId={`econ-use-${field.replace(".", "-")}`}
            onUse={() =>
              update((d) => {
                if (field === "crew.pct") d.inputs.crew = { ...(d.inputs.crew ?? { kind: null }), pct: r.value };
                else (d.inputs as Record<string, unknown>)[field] = r.value;
                d.basis[field] = "preset";
              })
            }
          />
        );
      })}
    </>
  );
}

function DieselRef({ used, onUse }: { used: boolean; onUse: () => void }) {
  const now = useNow();
  const age = dieselAge(now);
  return (
    <RefRow
      testId="econ-diesel-ref"
      label="EIA California diesel"
      value={`$${DIESEL_REF.usdPerGal.toFixed(3)}`}
      source={
        <>
          Week of Oct 19, 2026; a year earlier ${DIESEL_REF.prior.usdPerGal.toFixed(3)}.{" "}
          <a href={DIESEL_REF.url} {...ext} className="nx-focus underline decoration-n-line-strong underline-offset-2">
            EIA weekly series
          </a>
        </>
      }
      note={`${DIESEL_REF.label}.`}
      stale={
        age.stale ? (
          <p className="inline-flex items-center gap-1 text-[12.5px] font-medium text-n-ink" data-testid="econ-diesel-stale">
            <Icon name="clock" className="h-3.5 w-3.5" /> {age.days} days old: check your dock&rsquo;s price
          </p>
        ) : null
      }
      used={used}
      useLabel="EIA"
      onUse={onUse}
      useTestId="econ-use-diesel"
    />
  );
}

function CatchLine({ i, draft, wu, needed, update, canRemove }: { i: number; draft: Draft; wu: WU; needed: Set<string>; update: (fn: (d: Draft) => void) => void; canRemove: boolean }) {
  const c = draft.inputs.catch![i];
  const basisKey = `catch.${i}.usdPerLb`;
  const ref = priceRef(c.ref);
  const refShown = ref ? (wu.unit === "ton" ? `$${num(ref.usdPerLb * LB_PER_TON)}/ton` : wu.unit === "kg" ? `$${(ref.usdPerLb * LB_PER_KG).toFixed(2)}/kg` : `$${ref.usdPerLb.toFixed(2)}/lb`) : null;
  return (
    <div className="flex flex-col gap-3" data-testid={`econ-catch-${i}`}>
      <div className="flex items-end gap-2">
        <div className="min-w-0 flex-1">
          <TextIn label={i === 0 ? "Main catch" : `Catch line ${i + 1}`} value={c.species} onChange={(v) => update((d) => (d.inputs.catch![i].species = v))} testId={`econ-catch-${i}-name`} />
        </div>
        {canRemove && (
          <button type="button" aria-label={`Remove ${c.species || `catch line ${i + 1}`}`} onClick={() => update((d) => d.inputs.catch!.splice(i, 1))} className="nx-press nx-focus grid h-11 w-11 place-items-center rounded-[3px] text-n-ink-3 hover:text-n-ink">
            <Icon name="close" className="h-4 w-4" />
          </button>
        )}
      </div>
      <div className="grid grid-cols-1 gap-x-8 gap-y-3 sm:grid-cols-2">
        <NumIn id={fieldDomId(`catch.${i}.lb`)} label="Landed" unit={wu.unit} kind="lb" decimals={wu.unit === "ton" ? 2 : 0} scale={wu.w} value={c.lb} onChange={(v) => update((d) => (d.inputs.catch![i].lb = v))} needed={needed.has(`catch.${i}.lb`)} testId={`econ-catch-${i}-lb`} />
        <NumIn
          id={fieldDomId(`catch.${i}.usdPerLb`)}
          label="Your price"
          unit={`$/${wu.unit}`}
          kind="usdPerLb"
          decimals={wu.unit === "ton" ? 0 : 2}
          scale={wu.p}
          value={c.usdPerLb}
          needed={needed.has(`catch.${i}.usdPerLb`)}
          onChange={(v) =>
            update((d) => {
              d.inputs.catch![i].usdPerLb = v;
              delete d.basis[basisKey];
            })
          }
          testId={`econ-catch-${i}-price`}
        />
      </div>
      {ref ? (
        <RefRow
          testId={`econ-catch-${i}-ref`}
          label="CDFW history, not today’s price"
          value={refShown}
          source={
            <>
              {ref.fishery}, statewide, all ports and gears, 2024{ref.prior ? ` (${ref.prior.year}: $${(ref.prior.usdPerLb * wu.p).toFixed(2)})` : ""}.{" "}
              <a href={ref.url} {...ext} className="nx-focus underline decoration-n-line-strong underline-offset-2">
                CDFW report
              </a>
            </>
          }
          note={`Past average, not today’s dock price: $ ÷ lb computed by Neritic${ref.caveat ? `; ${ref.caveat}` : ""}. Ask your buyer.`}
          used={draft.basis[basisKey] === "cdfw"}
          useLabel="CDFW history"
          useTestId={`econ-catch-${i}-use`}
          onUse={() =>
            update((d) => {
              d.inputs.catch![i].usdPerLb = ref.usdPerLb;
              d.basis[basisKey] = "cdfw";
            })
          }
        />
      ) : (
        <p className="text-[12.5px] leading-snug text-n-ink-2" data-testid={`econ-catch-${i}-ref`}>
          {NO_PRICE_REF} for {c.species || "this catch"}: CDFW publishes none Neritic can quote, and no preset holds a catch or a price. Ask your buyer.
        </p>
      )}
    </div>
  );
}

function CrewParams({ draft, preset, needed, setNum, update }: { draft: Draft; preset: FisheryPreset | null; needed: Set<string>; setNum: (k: string) => (v: number | null) => void; update: (fn: (d: Draft) => void) => void }) {
  const k = draft.inputs.crew?.kind ?? null;
  if (k === null)
    return (
      <p className="text-[13.5px] leading-snug text-n-ink-2" data-testid="econ-crew-unset">
        Choose one: net to boat, the break-evens and the sensitivity wait for it. {preset && !preset.crewPct ? `Neritic has no sourced crew share for ${preset.label.toLowerCase()}, so nothing is assumed.` : "Nothing is assumed."}
      </p>
    );
  return (
    <div className="flex flex-col gap-4">
      <p className="nx-margin" data-testid="econ-crew-effect">
        {CREW_EFFECT[k]}
      </p>
      {k === "dayRate" ? (
        <div className="grid grid-cols-1 gap-x-8 gap-y-4 sm:grid-cols-2">
          <NumIn id={fieldDomId("crew.crew")} label="Crew paid" unit="people" kind="people" decimals={0} value={draft.inputs.crew?.crew ?? null} onChange={setNum("crew.crew")} needed={needed.has("crew.crew")} testId="econ-crew-n" />
          <NumIn id={fieldDomId("crew.usdPerDay")} label="Day rate each" unit="$/day" kind="usdPerDay" decimals={0} value={draft.inputs.crew?.usdPerDay ?? null} onChange={setNum("crew.usdPerDay")} needed={needed.has("crew.usdPerDay")} testId="econ-crew-rate" />
          <NumIn id={fieldDomId("crew.days")} label="Days" unit="days" kind="days" decimals={1} value={draft.inputs.crew?.days ?? null} onChange={setNum("crew.days")} needed={needed.has("crew.days")} testId="econ-crew-days" />
        </div>
      ) : k === "none" ? null : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-2">
            <NumIn id={fieldDomId("crew.pct")} label={k === "ofGross" ? "Crew share of gross, all crew" : "Crew share after expenses, all crew"} unit="%" kind="pct" decimals={1} value={draft.inputs.crew?.pct ?? null} onChange={setNum("crew.pct")} needed={needed.has("crew.pct")} testId="econ-crew-pct" />
            <FisheryRefs field="crew.pct" draft={draft} update={update} />
            {preset && !preset.crewPct && <p className="text-[12.5px] text-n-ink-2">No sourced crew share for {preset.label.toLowerCase()}: nothing is assumed.</p>}
          </div>
          <NumIn id={fieldDomId("crew.crew")} label="Crew sharing it (for pay each)" unit="people" kind="people" decimals={0} value={draft.inputs.crew?.crew ?? null} onChange={setNum("crew.crew")} testId="econ-crew-n" />
        </div>
      )}
    </div>
  );
}

function PotFields({ draft, preset, update }: { draft: Draft; preset: FisheryPreset; update: (fn: (d: Draft) => void) => void }) {
  const tiers = preset.potTiers!;
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <label className="flex flex-col gap-1 text-[14px] text-n-ink-2">
        Pots fished (trap tier)
        <select value={draft.pots === null ? "" : String(draft.pots)} onChange={(e) => update((d) => (d.pots = e.target.value === "" ? null : Number(e.target.value)))} className="nx-select nx-focus min-h-11 rounded-[3px] border border-n-line-strong bg-n-surface pl-3 text-[15px] text-n-ink" data-testid="econ-pots">
          <option value="">Not set</option>
          {tiers.tiers.map((t) => (
            <option key={t} value={String(t)}>
              {t} pots
            </option>
          ))}
        </select>
        <a href="https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=FGC&sectionNum=8276.5" {...ext} className="nx-focus text-[12.5px] underline decoration-n-line-strong underline-offset-2">
          Tiers from Fish and Game Code §8276.5; your allocation is on your CDFW permit
        </a>
      </label>
      <NumIn id={fieldDomId("baitUsdPerPot")} label="Bait per pot" unit="$" kind="usdPerDay" value={draft.baitUsdPerPot} onChange={(v) => update((d) => (d.baitUsdPerPot = v))} testId="econ-bait" note={draft.pots !== null && draft.baitUsdPerPot !== null ? `Bait ${money(draft.pots * draft.baitUsdPerPot)}, shared` : undefined} />
    </div>
  );
}

/** A saved vessel fills speed, burn, crew and pots; the current numbers can be kept as one. */
function Vessels({ draft, readOnly, update }: { draft: Draft; readOnly: boolean; update: (fn: (d: Draft) => void) => void }) {
  const data = useTripData();
  const vessels = data?.vessels ?? [];
  const [name, setName] = useState("");
  const [naming, setNaming] = useState(false);
  return (
    <div className="flex flex-col gap-3">
      {vessels.length > 0 && (
        <label className="flex flex-col gap-1 text-[14px] text-n-ink-2">
          Your vessel
          <select
            value=""
            onChange={(e) => {
              const v = vessels.find((x) => x.id === e.target.value);
              if (!v) return;
              update((d) => {
                for (const [k, val] of [["cruiseKt", v.cruiseKt], ["steamGph", v.steamGph], ["fishingGph", v.fishingGph], ["crew.crew", v.crew], ["pots", v.pots]] as const) {
                  if (val === null) continue;
                  setField(d, k, val);
                  delete d.basis[k];
                }
              });
            }}
            className="nx-select nx-focus min-h-11 rounded-[3px] border border-n-line-strong bg-n-surface pl-3 text-[15px] text-n-ink"
            data-testid="econ-vessel"
          >
            <option value="">Fill speed, burn, crew from…</option>
            {vessels.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
        </label>
      )}
      {naming ? (
        <div className="flex flex-wrap items-end gap-3">
          <div className="min-w-0 flex-1">
            <TextIn label="Vessel name (kept on this device)" value={name} onChange={setName} testId="econ-vessel-name" />
          </div>
          <Action
            primary
            testId="econ-vessel-save"
            onClick={() => {
              const res = tripStore().saveVessel({ name: name || "My vessel", cruiseKt: draft.inputs.cruiseKt ?? null, steamGph: draft.inputs.steamGph ?? null, fishingGph: draft.inputs.fishingGph ?? null, crew: draft.inputs.crew?.crew ?? null, pots: draft.pots });
              notifyTrip();
              toast(res.ok ? "Vessel saved" : res.reason === "full" ? `${MAX_VESSELS} vessels kept: delete one first` : "Not saved on this device");
              if (res.ok) setNaming(false);
            }}
          >
            Save
          </Action>
        </div>
      ) : (
        !readOnly && (
          <TextAction onClick={() => setNaming(true)} testId="econ-vessel-new">
            <Icon name="plus" className="h-4 w-4" /> Keep speed and burn as a vessel
          </TextAction>
        )
      )}
    </div>
  );
}


/**
 * The inputs, folded to one line once the result is worked out (v10 pass 2): each number as typed (or the
 * reference it came from, named), then "Edit inputs". Nothing here is a new number: it reads the draft.
 */
function InputsSummary({ draft, rec, wu, onEdit }: { draft: Draft; rec: boolean; wu: WU; onEdit: () => void }) {
  const i = draft.inputs;
  const n = (v: number | null | undefined, d = 1) => (v == null ? null : num(v, d));
  const basis = (k: string) => (draft.basis[k] === "eia" ? "EIA" : draft.basis[k] === "cdfw" ? "CDFW" : draft.basis[k] === "preset" ? "reference" : null);
  const parts: { text: string; note?: string | null }[] = [];
  if (!rec)
    for (const c of i.catch ?? [])
      if (c.lb != null || c.usdPerLb != null) parts.push({ text: `${c.species ? `${c.species} ` : ""}${n(c.lb != null ? c.lb * wu.w : null, wu.unit === "ton" ? 2 : 0) ?? "—"} ${wu.unit} at $${c.usdPerLb != null ? (c.usdPerLb * wu.p).toFixed(wu.unit === "ton" ? 0 : 2) : "—"}/${wu.unit}` });
  if (i.fuelGal != null) parts.push({ text: `${n(i.fuelGal)} gal` });
  else if (i.roundTripNm != null) parts.push({ text: `${n(i.roundTripNm)} nm at ${n(i.cruiseKt) ?? "—"} kt, ${n(i.steamGph) ?? "—"} gal/h · ${n(i.fishingHours) ?? "—"} h at ${n(i.fishingGph) ?? "—"} gal/h` });
  if (i.fuelUsdPerGal != null) parts.push({ text: `fuel $${i.fuelUsdPerGal.toFixed(3)}/gal`, note: basis("fuelUsdPerGal") });
  for (const c of i.costs ?? []) if (c.usd != null) parts.push({ text: `${(c.label || "cost").toLowerCase()} $${num(c.usd, 0)}`, note: rec ? null : c.shared ? "shared" : "boat-paid" });
  if (i.feesPctOfGross) parts.push({ text: `fees ${n(i.feesPctOfGross)} %` });
  if (i.feesUsdPerLb) parts.push({ text: `fees $${i.feesUsdPerLb.toFixed(2)}/lb` });
  const crew = i.crew;
  if (!rec && crew?.kind)
    parts.push({
      text:
        crew.kind === "none"
          ? "no hired crew"
          : crew.kind === "dayRate"
            ? `crew $${n(crew.usdPerDay, 0) ?? "—"}/day × ${n(crew.crew, 0) ?? "—"} × ${n(crew.days, 0) ?? "—"} d`
            : `crew ${n(crew.pct) ?? "—"} % ${crew.kind === "ofGross" ? "of gross" : "after expenses"}${crew.crew ? `, ${n(crew.crew, 0)} sharing` : ""}`,
    });
  if (rec && i.anglers) parts.push({ text: `${n(i.anglers, 0)} sharing` });
  return (
    <div className="flex flex-col gap-2 border-b border-n-line pb-4 sm:flex-row sm:items-baseline sm:gap-6" data-testid="econ-inputs-summary">
      <Caps className="shrink-0">Your inputs</Caps>
      <p className="min-w-0 flex-1 font-data text-[13.5px] leading-relaxed text-n-ink">
        {parts.map((p, k) => (
          <span key={k}>
            {k > 0 && <span className="text-n-ink-3"> · </span>}
            {p.text}
            {p.note && <span className="font-ui text-[12px] text-n-ink-2"> {p.note}</span>}
          </span>
        ))}
      </p>
      <Action onClick={onEdit} testId="econ-edit-inputs">
        Edit inputs <Icon name="chevronDown" className="h-4 w-4" />
      </Action>
    </div>
  );
}
