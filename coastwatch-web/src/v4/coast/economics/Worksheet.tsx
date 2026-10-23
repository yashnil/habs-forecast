"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import type { Audience } from "@/lib/coast";
import type { SpeciesId } from "@/content/species";
import { presetById, type FisheryPreset } from "@/content/fisheryPresets";
import { DIESEL_REF, NO_PRICE_REF, PRICE_REF_LABEL, dieselAge, priceRef } from "@/content/referenceValues";
import { LB_PER_KG, LB_PER_TON, MISSING_LABEL, computeTrip, num, usd, type CrewKind, type TripResult } from "@/lib/tripEconomics";
import { MAX_SCENARIOS, MAX_VESSELS, SHARE_PARAM, draftInputs, draftKey, encodeShare, type Draft, type ScenarioRecord } from "@/lib/tripEconomicsStore";
import { useNow } from "@/lib/useNow";
import { toast } from "@/v4/ui/toast";
import { Icon } from "@/v4/ui/Icon";
import { dayLabel } from "@/v4/ui/time";
import { Check, NumberField, Select, TextField } from "@/v4/ui/Fields";
import { notifyTrip, tripStore, useTripData } from "./useTripData";
import { FISHERY_OPTIONS, applyPreset, presetSource, refsUsed, setField } from "./draft";

type Props = {
  place: { key: string; name: string };
  species: SpeciesId;
  audience: Audience;
  tripDate: string | null;
  shared: Draft | null;
  initial: Draft;
  onLive: (d: Draft) => void;
};

type WUnit = "lb" | "kg" | "ton";
const W_SCALE: Record<WUnit, number> = { lb: 1, kg: 1 / LB_PER_KG, ton: 1 / LB_PER_TON };
const P_SCALE: Record<WUnit, number> = { lb: 1, kg: LB_PER_KG, ton: LB_PER_TON };

const CREW_OPTIONS: { value: CrewKind; label: string }[] = [
  { value: "afterExpenses", label: "Share after expenses" },
  { value: "ofGross", label: "Share of gross" },
  { value: "dayRate", label: "Day rate" },
];

const ext = { target: "_blank", rel: "noreferrer" } as const;
const clone = (d: Draft): Draft => JSON.parse(JSON.stringify(d));
const pct = (x: number) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.round(Math.abs(x) * 100)} %`;

export default function Worksheet({ place, species, tripDate, shared, initial, onLive }: Props) {
  const data = useTripData();
  const key = draftKey(place.key, species);
  const [draft, setDraft] = useState<Draft>(() => clone(shared ?? initial));
  const [fromLink, setFromLink] = useState(shared !== null);
  const [scenarioId, setScenarioId] = useState<string | null>(null);
  const readOnly = data?.status.readOnly ?? false;
  const unitPref = data?.meta.unit ?? "lb";
  const preset = presetById(draft.preset);
  const wUnit: WUnit = preset?.unit === "ton" ? "ton" : unitPref;
  const r = useMemo(() => computeTrip(draftInputs(draft)), [draft]);

  useEffect(() => onLive(draft), [draft, onLive]);

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

  const src = (k: string) => {
    const s = presetSource(draft, k);
    return s ? <SourceNote text={s.text} url={s.url} note={s.note} /> : null;
  };

  // the results bar on phones shows only while the worksheet is on screen
  const root = useRef<HTMLDivElement | null>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = root.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { rootMargin: "0px 0px -120px 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div ref={root} className="@container flex flex-col gap-6" data-testid="econ-worksheet">
      {fromLink && (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-xl border border-n-official-line bg-n-official-bg px-4 py-3 text-[15px]" role="status" data-testid="econ-shared">
          <span className="min-w-0 font-semibold">From a shared link, not saved.</span>
          <button
            type="button"
            disabled={readOnly}
            onClick={() => {
              tripStore().saveDraft(key, draft);
              notifyTrip();
              setFromLink(false);
              toast("Saved to this device");
            }}
            className="nx-press nx-focus min-h-11 rounded-xl border border-n-line-strong bg-n-surface px-4 font-semibold disabled:opacity-60"
            data-testid="econ-shared-save"
          >
            Save to this device
          </button>
        </div>
      )}
      <StoreStatus />

      <div className="grid gap-6 @[760px]:grid-cols-[minmax(0,1fr)_300px]">
        <div className="flex min-w-0 flex-col gap-6">
          <Group title="Trip" testId="econ-g-trip">
            <Select
              label="Fishery"
              value={(draft.preset ?? "") as string}
              options={FISHERY_OPTIONS}
              testId="econ-fishery"
              hint={preset?.note}
              onChange={(v) =>
                update((d) => {
                  const p = presetById(v);
                  if (p) applyPreset(d, p);
                  else d.preset = null;
                })
              }
            />
            {wUnit !== "ton" ? (
              <Select
                label="Weigh in"
                value={unitPref}
                options={[
                  { value: "lb", label: "Pounds (lb)" },
                  { value: "kg", label: "Kilograms (kg)" },
                ]}
                testId="econ-unit"
                onChange={(v) => {
                  tripStore().setUnit(v);
                  notifyTrip();
                }}
              />
            ) : (
              <p className="self-end pb-3 text-[14px] text-n-ink-3">Market squid is sold by the short ton (2,000 lb).</p>
            )}
            <NumberField label="Round trip" unit="nm" kind="nm" decimals={1} value={draft.inputs.roundTripNm ?? null} onChange={setNum("roundTripNm")} testId="econ-nm" />
            <NumberField label="Cruising speed" unit="kt" kind="kt" decimals={1} value={draft.inputs.cruiseKt ?? null} onChange={setNum("cruiseKt")} testId="econ-kt" />
            <NumberField
              label={preset?.effort === "dive day" ? "Hours diving" : "Hours fishing"}
              unit="h"
              kind="hours"
              decimals={1}
              value={draft.inputs.fishingHours ?? null}
              onChange={setNum("fishingHours")}
              testId="econ-hours"
              hint={preset && preset.effort !== "trip" ? `One scenario is one ${preset.effort}.` : tripDate ? `For the trip on ${dayLabel(tripDate, true)}.` : undefined}
            />
          </Group>

          <Group title="Vessel and fuel" testId="econ-g-vessel">
            <VesselPicker draft={draft} update={update} readOnly={readOnly} />
            <NumberField label="Fuel burn steaming" unit="gal/h" kind="gph" decimals={1} value={draft.inputs.steamGph ?? null} onChange={setNum("steamGph")} testId="econ-steam-gph" hint={src("steamGph")} />
            <NumberField label="Fuel burn fishing" unit="gal/h" kind="gph" decimals={1} value={draft.inputs.fishingGph ?? null} onChange={setNum("fishingGph")} testId="econ-fish-gph" hint={src("fishingGph")} />
            <NumberField
              label="Or gallons for the whole trip"
              unit="gal"
              kind="gal"
              decimals={1}
              value={draft.inputs.fuelGal ?? null}
              onChange={setNum("fuelGal")}
              testId="econ-fuel-gal"
              hint={r.fuelGal !== null && draft.inputs.fuelGal == null ? `Worked out: ${num(r.fuelGal, 1)} gal` : "Replaces the burn estimate when filled."}
            />
            <NumberField label="Fuel price" unit="$/gal" kind="usdPerGal" decimals={3} value={draft.inputs.fuelUsdPerGal ?? null} onChange={setNum("fuelUsdPerGal")} testId="econ-fuel-price" hint={<DieselRef used={draft.basis.fuelUsdPerGal === "eia"} onUse={() => update((d) => ((d.inputs.fuelUsdPerGal = DIESEL_REF.usdPerGal), (d.basis.fuelUsdPerGal = "eia")))} />} />
            <div className="@[480px]:col-span-2">
              <Check label="Fuel comes off the top before crew shares" checked={draft.inputs.fuelShared !== false} onChange={(v) => update((d) => (d.inputs.fuelShared = v))} testId="econ-fuel-shared" />
            </div>
            {preset?.potTiers && <PotFields draft={draft} preset={preset} update={update} />}
          </Group>

          <Group title="Costs and crew" testId="econ-g-costs">
            <div className="flex flex-col gap-3 @[480px]:col-span-2">
              {(draft.inputs.costs ?? []).map((c, i) => (
                <div key={i} className="grid grid-cols-[minmax(0,1fr)_minmax(0,9rem)_auto] items-end gap-2" data-testid={`econ-cost-${i}`}>
                  <TextField label={`Cost ${i + 1}`} value={c.label} onChange={(v) => update((d) => (d.inputs.costs![i].label = v))} />
                  <NumberField label={<span className="sr-only">{c.label || `Cost ${i + 1}`} amount</span>} unit="$" kind="usd" value={c.usd} onChange={(v) => update((d) => (d.inputs.costs![i].usd = v))} testId={`econ-cost-${i}-usd`} />
                  <IconButton label={`Remove ${c.label || `cost ${i + 1}`}`} icon="close" onClick={() => update((d) => d.inputs.costs!.splice(i, 1))} />
                  <div className="col-span-3 -mt-1">
                    <Check label="Shared (comes off before crew shares)" checked={c.shared} onChange={(v) => update((d) => (d.inputs.costs![i].shared = v))} />
                  </div>
                </div>
              ))}
              {(draft.inputs.costs ?? []).length < 20 && (
                <TextButton onClick={() => update((d) => (d.inputs.costs = [...(d.inputs.costs ?? []), { label: "", usd: null, shared: true }]))} testId="econ-add-cost">
                  <Icon name="plus" className="h-4 w-4" /> Add a cost
                </TextButton>
              )}
            </div>
            <NumberField label="Fees, share of gross" unit="%" kind="pct" value={draft.inputs.feesPctOfGross ?? null} onChange={setNum("feesPctOfGross")} testId="econ-fees-pct" hint={src("feesPctOfGross")} />
            <NumberField label={preset?.perLbLabel ?? "Fees, $ per lb"} unit="$/lb" kind="usdPerLb" value={draft.inputs.feesUsdPerLb ?? null} onChange={setNum("feesUsdPerLb")} testId="econ-fees-lb" />
            <Select<CrewKind> label="Crew paid by" value={draft.inputs.crew?.kind ?? "afterExpenses"} options={CREW_OPTIONS} onChange={(v) => update((d) => (d.inputs.crew = { ...d.inputs.crew, kind: v }))} testId="econ-crew-kind" />
            {draft.inputs.crew?.kind === "dayRate" ? (
              <>
                <NumberField label="Crew paid a day rate" unit="people" kind="people" decimals={0} value={draft.inputs.crew?.crew ?? null} onChange={setNum("crew.crew")} testId="econ-crew-n" />
                <NumberField label="Day rate each" unit="$/day" kind="usdPerDay" value={draft.inputs.crew?.usdPerDay ?? null} onChange={setNum("crew.usdPerDay")} testId="econ-crew-rate" />
                <NumberField label="Days" unit="days" kind="days" decimals={1} value={draft.inputs.crew?.days ?? null} onChange={setNum("crew.days")} testId="econ-crew-days" />
              </>
            ) : (
              <NumberField
                label={draft.inputs.crew?.kind === "ofGross" ? "Crew share of gross (all crew)" : "Crew share after expenses (all crew)"}
                unit="%"
                kind="pct"
                decimals={1}
                value={draft.inputs.crew?.pct ?? null}
                onChange={setNum("crew.pct")}
                testId="econ-crew-pct"
                hint={
                  <>
                    {src("crew.pct")}
                    {preset?.crewPctAlt && draft.inputs.crew?.kind === "afterExpenses" && draft.inputs.crew?.pct !== preset.crewPctAlt.value && (
                      <button type="button" className="nx-focus mt-1 min-h-11 text-left font-semibold text-n-accent underline-offset-2 hover:underline" onClick={() => update((d) => ((d.inputs.crew = { ...d.inputs.crew!, pct: preset.crewPctAlt!.value }), (d.basis["crew.pct"] = "preset")))} data-testid="econ-crew-alt">
                        Use {preset.crewPctAlt.value} % ({preset.crewPctAlt.label})
                      </button>
                    )}
                  </>
                }
              />
            )}
          </Group>

          <Group title="Catch and price" testId="econ-g-catch" note="Your own pounds and price. Neritic never estimates a catch or a price.">
            <div className="flex flex-col gap-5 @[480px]:col-span-2">
              {(draft.inputs.catch ?? []).map((c, i) => (
                <CatchRow key={i} i={i} draft={draft} wUnit={wUnit} update={update} canRemove={i > 0} />
              ))}
              {(draft.inputs.catch ?? []).length < 10 && (
                <TextButton onClick={() => update((d) => (d.inputs.catch = [...(d.inputs.catch ?? []), { species: `Catch ${(d.inputs.catch ?? []).length + 1}`, ref: null, lb: null, usdPerLb: null }]))} testId="econ-add-catch">
                  <Icon name="plus" className="h-4 w-4" /> Add a catch line
                </TextButton>
              )}
            </div>
          </Group>
        </div>

        <div className="flex min-w-0 flex-col gap-6 @[760px]:sticky @[760px]:top-20 @[760px]:self-start">
          <Results r={r} wUnit={wUnit} />
          <Sensitivity r={r} />
        </div>
      </div>

      <Scenarios place={place} species={species} draft={draft} scenarioId={scenarioId} setScenarioId={setScenarioId} readOnly={readOnly} onOpen={(d) => update((x) => Object.assign(x, clone(d)))} />
      <ShareLink draft={draft} />
      <TripDataPanel />

      {/* in a portal: the worksheet is a size container and the page may carry a transform, either of which would pin a fixed bar to them */}
      {inView &&
        createPortal(
        <div className="env-chart fixed inset-x-0 z-30 border-t border-n-line bg-n-surface px-4 py-2 shadow-[0_-6px_16px_-12px_rgba(11,24,32,0.4)] lg:hidden" style={{ bottom: "calc(60px + env(safe-area-inset-bottom, 0px))" }} data-testid="econ-bar" aria-hidden>
          <div className="nx-data mx-auto flex max-w-[640px] items-baseline justify-between gap-3 text-[15px]">
            <span>
              <span className="font-ui text-[12.5px] text-n-ink-3">Net </span>
              <span className="font-semibold">{usd(r.netUsd, { signed: true })}</span>
            </span>
            <span>
              <span className="font-ui text-[12.5px] text-n-ink-3">Cost </span>
              {usd(r.tripCostUsd)}
            </span>
            <span>
              <span className="font-ui text-[12.5px] text-n-ink-3">B/E </span>
              {beLb(r, wUnit)}
            </span>
          </div>
        </div>,
          document.body,
        )}
    </div>
  );
}

// ---------------------------------------------------------------- pieces

function Group({ title, testId, note, children }: { title: string; testId: string; note?: string; children: ReactNode }) {
  return (
    <fieldset className="min-w-0 border-t border-n-line pt-4" data-testid={testId}>
      <legend className="float-left mb-3 w-full text-[17px] font-semibold text-n-ink">{title}</legend>
      {note && <p className="clear-left mb-3 text-[14px] text-n-ink-2">{note}</p>}
      <div className="clear-left grid grid-cols-1 gap-4 @[480px]:grid-cols-2">{children}</div>
    </fieldset>
  );
}

function SourceNote({ text, url, note }: { text: string; url: string; note?: string }) {
  return (
    <span className="block">
      Starting value, edit freely:{" "}
      <a href={url} {...ext} className="nx-focus underline underline-offset-2">
        {text}
      </a>
      {note && `. ${note}.`}
    </span>
  );
}

function TextButton({ onClick, children, testId }: { onClick: () => void; children: ReactNode; testId?: string }) {
  return (
    <button type="button" onClick={onClick} className="nx-press nx-focus inline-flex min-h-11 items-center gap-1.5 self-start rounded-xl px-2 text-[15px] font-semibold text-n-accent" data-testid={testId}>
      {children}
    </button>
  );
}

function IconButton({ label, icon, onClick }: { label: string; icon: string; onClick: () => void }) {
  return (
    <button type="button" aria-label={label} title={label} onClick={onClick} className="nx-press nx-focus grid h-12 w-11 place-items-center rounded-xl text-n-ink-3 hover:text-n-ink">
      <Icon name={icon} className="h-5 w-5" />
    </button>
  );
}

function UseButton({ onClick, testId, children }: { onClick: () => void; testId: string; children?: ReactNode }) {
  return (
    <button type="button" onClick={onClick} className="nx-press nx-focus mt-1 inline-flex min-h-11 items-center rounded-xl border border-n-line-strong bg-n-surface px-3 text-[14px] font-semibold text-n-ink" data-testid={testId}>
      {children ?? "Use as a starting assumption"}
    </button>
  );
}

function DieselRef({ used, onUse }: { used: boolean; onUse: () => void }) {
  const now = useNow();
  const age = dieselAge(now);
  return (
    <span className="block" data-testid="econ-diesel-ref">
      <span className="font-semibold text-n-ink-2">
        EIA California diesel, week of Oct 19, 2026: <span className="nx-data">${DIESEL_REF.usdPerGal.toFixed(3)}</span>/gal
      </span>{" "}
      (a year earlier <span className="nx-data">${DIESEL_REF.prior.usdPerGal.toFixed(3)}</span>). {DIESEL_REF.label}.{" "}
      <a href={DIESEL_REF.url} {...ext} className="nx-focus underline underline-offset-2">
        EIA
      </a>
      {age.stale && (
        <span className="ml-2 inline-flex items-center gap-1 rounded-full border border-n-official-line bg-n-official-bg px-2 py-0.5 text-[12.5px] font-semibold text-n-official" data-testid="econ-diesel-stale">
          <Icon name="clock" className="h-3.5 w-3.5" /> {age.days} days old
        </span>
      )}
      {used ? <span className="mt-1 block font-semibold text-n-ink-2">In use as your starting assumption (basis: EIA).</span> : <UseButton onClick={onUse} testId="econ-use-diesel" />}
    </span>
  );
}

function PotFields({ draft, preset, update }: { draft: Draft; preset: FisheryPreset; update: (fn: (d: Draft) => void) => void }) {
  const tiers = preset.potTiers!;
  return (
    <>
      <Select
        label="Pots fished (trap tier)"
        value={draft.pots === null ? "" : String(draft.pots)}
        options={[{ value: "", label: "Not set" }, ...tiers.tiers.map((t) => ({ value: String(t), label: `${t} pots` }))]}
        onChange={(v) => update((d) => (d.pots = v === "" ? null : Number(v)))}
        testId="econ-pots"
        hint={
          <a href={`https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=FGC&sectionNum=8276.5`} {...ext} className="nx-focus underline underline-offset-2">
            Tiers from Fish and Game Code §8276.5; your allocation and any in-season reduction are on your CDFW permit
          </a>
        }
      />
      <NumberField label="Bait per pot" unit="$" kind="usdPerDay" value={draft.baitUsdPerPot} onChange={(v) => update((d) => (d.baitUsdPerPot = v))} testId="econ-bait" hint={draft.pots !== null && draft.baitUsdPerPot !== null ? `Bait: ${usd(draft.pots * draft.baitUsdPerPot)}, shared` : undefined} />
    </>
  );
}

function VesselPicker({ draft, update, readOnly }: { draft: Draft; update: (fn: (d: Draft) => void) => void; readOnly: boolean }) {
  const data = useTripData();
  const vessels = data?.vessels ?? [];
  const [name, setName] = useState("");
  const [naming, setNaming] = useState(false);
  return (
    <div className="flex flex-col gap-2 @[480px]:col-span-2">
      {vessels.length > 0 && (
        <Select
          label="Your vessel"
          value=""
          options={[{ value: "", label: "Choose to fill speed, burn, crew…" }, ...vessels.map((v) => ({ value: v.id, label: v.name }))]}
          testId="econ-vessel"
          onChange={(id) => {
            const v = vessels.find((x) => x.id === id);
            if (!v) return;
            update((d) => {
              for (const [k, val] of [["cruiseKt", v.cruiseKt], ["steamGph", v.steamGph], ["fishingGph", v.fishingGph], ["crew.crew", v.crew], ["pots", v.pots]] as const) {
                if (val === null) continue;
                setField(d, k, val);
                delete d.basis[k];
              }
            });
          }}
        />
      )}
      {naming ? (
        <div className="flex flex-wrap items-end gap-2">
          <div className="min-w-0 flex-1">
            <TextField label="Vessel name (kept on this device)" value={name} onChange={setName} testId="econ-vessel-name" />
          </div>
          <button
            type="button"
            className="nx-press nx-focus min-h-12 rounded-xl bg-n-ink px-4 text-[15px] font-semibold text-n-bg"
            data-testid="econ-vessel-save"
            onClick={() => {
              const res = tripStore().saveVessel({ name: name || "My vessel", cruiseKt: draft.inputs.cruiseKt ?? null, steamGph: draft.inputs.steamGph ?? null, fishingGph: draft.inputs.fishingGph ?? null, crew: draft.inputs.crew?.crew ?? null, pots: draft.pots });
              notifyTrip();
              toast(res.ok ? "Vessel saved" : res.reason === "full" ? `${MAX_VESSELS} vessels kept: delete one first` : "Not saved on this device");
              if (res.ok) setNaming(false);
            }}
          >
            Save
          </button>
        </div>
      ) : (
        !readOnly && (
          <TextButton onClick={() => setNaming(true)} testId="econ-vessel-new">
            <Icon name="plus" className="h-4 w-4" /> Save speed and burn as a vessel
          </TextButton>
        )
      )}
    </div>
  );
}

function CatchRow({ i, draft, wUnit, update, canRemove }: { i: number; draft: Draft; wUnit: WUnit; update: (fn: (d: Draft) => void) => void; canRemove: boolean }) {
  const c = draft.inputs.catch![i];
  const ref = priceRef(c.ref);
  const basisKey = `catch.${i}.usdPerLb`;
  const priceUnit = wUnit === "ton" ? "$/ton" : wUnit === "kg" ? "$/kg" : "$/lb";
  const refShown = ref ? (wUnit === "ton" ? `$${num(ref.usdPerLb * LB_PER_TON)}/ton` : wUnit === "kg" ? `$${(ref.usdPerLb * LB_PER_KG).toFixed(2)}/kg` : `$${ref.usdPerLb.toFixed(2)}/lb`) : null;
  const priorShown = ref ? (wUnit === "ton" ? `$${num(ref.prior.usdPerLb * LB_PER_TON)}` : wUnit === "kg" ? `$${(ref.prior.usdPerLb * LB_PER_KG).toFixed(2)}` : `$${ref.prior.usdPerLb.toFixed(2)}`) : null;
  return (
    <div className="flex flex-col gap-3 border-l-2 border-n-line pl-3" data-testid={`econ-catch-${i}`}>
      <div className="flex items-end gap-2">
        <div className="min-w-0 flex-1">
          <TextField label={i === 0 ? "Main catch" : `Catch line ${i + 1}`} value={c.species} onChange={(v) => update((d) => (d.inputs.catch![i].species = v))} testId={`econ-catch-${i}-name`} />
        </div>
        {canRemove && <IconButton label={`Remove ${c.species || `catch line ${i + 1}`}`} icon="close" onClick={() => update((d) => d.inputs.catch!.splice(i, 1))} />}
      </div>
      <div className="grid grid-cols-1 gap-3 min-[360px]:grid-cols-2">
        <NumberField label="Landed" unit={wUnit} kind="lb" decimals={wUnit === "ton" ? 2 : 0} scale={W_SCALE[wUnit]} value={c.lb} onChange={(v) => update((d) => (d.inputs.catch![i].lb = v))} testId={`econ-catch-${i}-lb`} />
        <NumberField
          label="Your price"
          unit={priceUnit}
          kind="usdPerLb"
          decimals={wUnit === "ton" ? 0 : 2}
          scale={P_SCALE[wUnit]}
          value={c.usdPerLb}
          onChange={(v) =>
            update((d) => {
              d.inputs.catch![i].usdPerLb = v;
              delete d.basis[basisKey];
            })
          }
          testId={`econ-catch-${i}-price`}
        />
      </div>
      <div className="text-[13px] leading-snug text-n-ink-3" data-testid={`econ-catch-${i}-ref`}>
        {ref ? (
          <>
            <span className="font-semibold text-n-ink-2">
              CDFW {ref.fishery}: <span className="nx-data">{refShown}</span> in 2024
            </span>{" "}
            ({ref.prior.year}: <span className="nx-data">{priorShown}</span>). {PRICE_REF_LABEL.replace("calendar year,", `calendar year${ref.caveat ? ` (${ref.fishery.split(" ")[0]}: ${ref.caveat})` : ""},`)}{" "}
            <a href={ref.url} {...ext} className="nx-focus underline underline-offset-2">
              Report
            </a>
            {draft.basis[basisKey] === "cdfw" ? (
              <span className="mt-1 block font-semibold text-n-ink-2">In use as your starting assumption (basis: CDFW history).</span>
            ) : (
              <UseButton
                testId={`econ-catch-${i}-use`}
                onClick={() =>
                  update((d) => {
                    d.inputs.catch![i].usdPerLb = ref.usdPerLb;
                    d.basis[basisKey] = "cdfw";
                  })
                }
              />
            )}
          </>
        ) : (
          <span>{NO_PRICE_REF}. Ask your buyer.</span>
        )}
      </div>
    </div>
  );
}

function beLb(r: TripResult, w: WUnit): string {
  const b = r.breakEvenLb;
  if (b.note === "none") return "none";
  if (b.note === "covered") return `0 ${w}`;
  if (b.value === null) return "—";
  return `${num(b.value * W_SCALE[w], w === "ton" ? 2 : 0)} ${w}`;
}

function bePrice(r: TripResult, w: WUnit): string {
  const b = r.breakEvenUsdPerLb;
  if (b.note === "none") return "None at this catch";
  if (b.note === "covered") return "Covered by other lines";
  if (b.value === null) return "—";
  return `${usd(b.value * P_SCALE[w], { cents: w !== "ton" })}/${w}`;
}

function Results({ r, wUnit }: { r: TripResult; wUnit: WUnit }) {
  const rows: [string, string, string][] = [
    ["Trip cost (fuel and costs)", usd(r.tripCostUsd), "econ-r-cost"],
    ["Gross", usd(r.grossUsd), "econ-r-gross"],
    ...(r.feesUsd ? ([["Fees", usd(r.feesUsd), "econ-r-fees"]] as [string, string, string][]) : []),
    ["Crew", usd(r.crewUsd), "econ-r-crew"],
    ["Net to the boat", usd(r.netUsd, { signed: true }), "econ-r-net"],
    ["Break-even catch", r.breakEvenLb.note === "none" ? "None at this price" : r.breakEvenLb.note === "covered" ? "Covered by other lines" : beLb(r, wUnit), "econ-r-be-lb"],
    ["Break-even price", bePrice(r, wUnit), "econ-r-be-price"],
  ];
  return (
    <div className="rounded-2xl border border-n-line bg-n-surface p-4" data-testid="econ-results" aria-live="polite" aria-atomic="false">
      <h3 className="text-[17px] font-semibold">Results</h3>
      <dl className="mt-2 divide-y divide-n-line">
        {rows.map(([k, v, id]) => (
          <div key={id} className={`flex items-baseline justify-between gap-3 py-2 ${id === "econ-r-net" ? "font-semibold" : ""}`}>
            <dt className="min-w-0 text-[15px] text-n-ink-2">{k}</dt>
            <dd className="nx-data shrink-0 text-right text-[16px] text-n-ink" data-testid={id}>
              {v}
            </dd>
          </div>
        ))}
      </dl>
      {r.missing.length > 0 && (
        <p className="mt-2 text-[14px] text-n-ink-2" data-testid="econ-missing">
          Still needed: {r.missing.map((m) => MISSING_LABEL[m]).join("; ")}.
        </p>
      )}
      {r.warnings.map((w) => (
        <p key={w} className="mt-2 flex gap-2 text-[14px] font-semibold text-n-official" data-testid="econ-warning">
          <Icon name="alert" className="mt-0.5 h-4 w-4 shrink-0" />
          {w}
        </p>
      ))}
      <p className="mt-3 text-[13px] text-n-ink-3">Net to the boat is after fuel, costs, fees and crew. Your numbers worked through, not a forecast.</p>
    </div>
  );
}

function Sensitivity({ r }: { r: TripResult }) {
  if (!r.sensitivity) return null;
  const cell = (p: number, c: number) => r.sensitivity!.find((x) => x.priceDelta === p && x.catchDelta === c)!.netUsd;
  return (
    <div className="rounded-2xl border border-n-line bg-n-surface p-4" data-testid="econ-sensitivity">
      <table className="w-full table-fixed border-collapse text-right">
        <caption className="mb-2 text-left text-[15px] font-semibold text-n-ink">
          Net to the boat if price and catch move <span className="block text-[13px] font-normal text-n-ink-3">all lines together; columns are catch, rows are price</span>
        </caption>
        <thead>
          <tr className="text-[12.5px] text-n-ink-3">
            <th scope="col" className="w-[26%] pb-1 text-left font-semibold">
              <span className="sr-only">Price change</span>
            </th>
            {[-0.3, 0, 0.3].map((c) => (
              <th key={c} scope="col" className="pb-1 font-semibold">
                Catch {pct(c)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="nx-data text-[14.5px]">
          {[0.2, 0, -0.2].map((p) => (
            <tr key={p} className="border-t border-n-line">
              <th scope="row" className="py-1.5 text-left font-ui text-[12.5px] font-semibold text-n-ink-3">
                Price {pct(p)}
              </th>
              {[-0.3, 0, 0.3].map((c) => {
                const v = cell(p, c);
                return (
                  <td key={c} className={`py-1.5 ${p === 0 && c === 0 ? "font-semibold text-n-ink" : v < 0 ? "text-n-down" : "text-n-ink-2"}`} data-testid={`econ-sens-${p}-${c}`}>
                    {usd(v, { signed: true })}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {r.fuelSensitivity && (
        <p className="mt-3 text-[14px] text-n-ink-2" data-testid="econ-fuel-sens">
          Fuel price −20 %: <span className="nx-data">{usd(r.fuelSensitivity.down, { signed: true })}</span>; +20 %: <span className="nx-data">{usd(r.fuelSensitivity.up, { signed: true })}</span>.{" "}
          {r.breakEvenFuelUsdPerGal.value !== null ? (
            <>
              Net reaches $0 at fuel <span className="nx-data">{usd(r.breakEvenFuelUsdPerGal.value, { cents: true })}</span>/gal.
            </>
          ) : r.fuelGal ? (
            "No fuel price brings net to $0 here."
          ) : null}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- scenarios, compare, link, data

function Scenarios({ place, species, draft, scenarioId, setScenarioId, readOnly, onOpen }: { place: { key: string; name: string }; species: string; draft: Draft; scenarioId: string | null; setScenarioId: (id: string | null) => void; readOnly: boolean; onOpen: (d: Draft) => void }) {
  const data = useTripData();
  const list = data?.scenarios ?? [];
  const current = list.find((s) => s.id === scenarioId) ?? null;
  const [name, setName] = useState("");
  const [a, setA] = useState<string>("current");
  const [b, setB] = useState<string>("");
  const say = (res: { ok: boolean; reason?: string }, ok: string) => toast(res.ok ? ok : res.reason === "full" ? `${MAX_SCENARIOS} scenarios kept: delete one first` : res.reason === "readOnly" ? "Read-only here" : "Not saved on this device");

  const save = (asNew: boolean) => {
    const res = tripStore().saveScenario({ id: asNew ? undefined : scenarioId ?? undefined, name: name || current?.name || `${place.name}, ${new Date().toLocaleDateString("en-US", { month: "short", day: "numeric" })}`, placeKey: place.key, species, draft, refs: refsUsed(draft) });
    notifyTrip();
    say(res, "Scenario saved");
    if (res.ok && "id" in res && res.id) {
      setScenarioId(res.id);
      setName("");
    }
  };

  const opts = [{ value: "current", label: "This worksheet" }, ...list.map((s) => ({ value: s.id, label: s.name }))];
  const pick = (id: string): { label: string; draft: Draft } | null => (id === "current" ? { label: "This worksheet", draft } : (list.find((s) => s.id === id) && { label: list.find((s) => s.id === id)!.name, draft: list.find((s) => s.id === id)!.draft }) || null);
  const A = pick(a);
  const B = pick(b);

  return (
    <section aria-labelledby="econ-scen-h" className="flex flex-col gap-4 border-t border-n-line pt-4" data-testid="econ-scenarios">
      <h3 id="econ-scen-h" className="text-[17px] font-semibold">
        Scenarios <span className="font-normal text-n-ink-3">· {list.length} of {MAX_SCENARIOS} kept on this device</span>
      </h3>
      <div className="grid grid-cols-1 gap-3 @[480px]:grid-cols-2">
        <TextField label={current ? "Name" : "Name this scenario"} value={name || current?.name || ""} onChange={setName} testId="econ-scen-name" />
        {list.length > 0 && (
          <Select
            label="Open a saved scenario"
            value={scenarioId ?? ""}
            options={[{ value: "", label: "—" }, ...list.map((s: ScenarioRecord) => ({ value: s.id, label: s.name }))]}
            testId="econ-scen-open"
            onChange={(id) => {
              const s = list.find((x) => x.id === id);
              setScenarioId(s?.id ?? null);
              setName("");
              if (s) onOpen(s.draft);
            }}
          />
        )}
      </div>
      {!readOnly && (
        <div className="flex flex-wrap gap-2">
          <ActionButton primary onClick={() => save(false)} testId="econ-scen-save">
            {current ? "Save changes" : "Save scenario"}
          </ActionButton>
          {current && (
            <>
              <ActionButton
                onClick={() => {
                  const res = tripStore().renameScenario(current.id, name || current.name);
                  notifyTrip();
                  setName("");
                  say(res, "Renamed");
                }}
                testId="econ-scen-rename"
              >
                Rename
              </ActionButton>
              <ActionButton
                onClick={() => {
                  const res = tripStore().duplicateScenario(current.id);
                  notifyTrip();
                  say(res, "Duplicated");
                  if (res.ok && res.id) {
                    setScenarioId(res.id);
                    setName("");
                  }
                }}
                testId="econ-scen-dup"
              >
                Duplicate
              </ActionButton>
              <ActionButton
                onClick={() => {
                  const res = tripStore().deleteScenario(current.id);
                  notifyTrip();
                  say(res, "Scenario deleted");
                  setScenarioId(null);
                }}
                testId="econ-scen-delete"
              >
                Delete
              </ActionButton>
            </>
          )}
        </div>
      )}
      {list.length > 0 && (
        <div className="flex flex-col gap-3" data-testid="econ-compare">
          <h4 className="text-[15px] font-semibold">Compare two</h4>
          <div className="grid grid-cols-1 gap-3 min-[400px]:grid-cols-2">
            <Select label="A" value={a} options={opts} onChange={setA} testId="econ-cmp-a" />
            <Select label="B" value={b} options={[{ value: "", label: "Choose…" }, ...opts]} onChange={setB} testId="econ-cmp-b" />
          </div>
          {A && B && <Compare a={A} b={B} />}
        </div>
      )}
    </section>
  );
}

function ActionButton({ onClick, children, testId, primary }: { onClick: () => void; children: ReactNode; testId: string; primary?: boolean }) {
  return (
    <button type="button" onClick={onClick} className={`nx-press nx-focus min-h-11 rounded-xl px-4 text-[15px] font-semibold ${primary ? "bg-n-ink text-n-bg" : "border border-n-line-strong bg-n-surface text-n-ink"}`} data-testid={testId}>
      {children}
    </button>
  );
}

const fmt = (v: number | null | undefined, d = 2) => (v === null || v === undefined ? "—" : num(v, d));

/** The inputs worth comparing, as text (lb and $/lb: compare reads canonical units). */
function inputRows(d: Draft): [string, string][] {
  const i = d.inputs;
  const rows: [string, string][] = [
    ["Fishery", presetById(d.preset)?.label ?? "Other"],
    ["Round trip, nm", fmt(i.roundTripNm, 1)],
    ["Cruising speed, kt", fmt(i.cruiseKt, 1)],
    ["Hours fishing", fmt(i.fishingHours, 1)],
    ["Burn steaming, gal/h", fmt(i.steamGph, 1)],
    ["Burn fishing, gal/h", fmt(i.fishingGph, 1)],
    ["Gallons for the trip", fmt(i.fuelGal, 1)],
    ["Fuel price, $/gal", fmt(i.fuelUsdPerGal, 3)],
    ["Fuel shared", i.fuelShared === false ? "No" : "Yes"],
    ["Pots", fmt(d.pots, 0)],
    ["Bait per pot, $", fmt(d.baitUsdPerPot)],
    ["Fees, %", fmt(i.feesPctOfGross)],
    ["Fees, $/lb", fmt(i.feesUsdPerLb)],
    ["Crew paid by", CREW_OPTIONS.find((c) => c.value === i.crew?.kind)?.label ?? "—"],
    ["Crew share, %", fmt(i.crew?.pct, 1)],
    ["Day-rate crew", fmt(i.crew?.crew, 0)],
    ["Day rate, $", fmt(i.crew?.usdPerDay)],
    ["Days", fmt(i.crew?.days, 1)],
  ];
  (i.costs ?? []).forEach((c, k) => rows.push([`Cost: ${c.label || k + 1}${c.shared ? " (shared)" : ""}`, fmt(c.usd)]));
  (i.catch ?? []).forEach((c, k) => {
    rows.push([`${c.species || `Catch ${k + 1}`}, lb`, fmt(c.lb, 0)]);
    rows.push([`${c.species || `Catch ${k + 1}`}, $/lb`, fmt(c.usdPerLb)]);
  });
  return rows;
}

function Compare({ a, b }: { a: { label: string; draft: Draft }; b: { label: string; draft: Draft } }) {
  const ra = computeTrip(draftInputs(a.draft));
  const rb = computeTrip(draftInputs(b.draft));
  const ia = new Map(inputRows(a.draft));
  const ib = new Map(inputRows(b.draft));
  const keys = [...new Set([...ia.keys(), ...ib.keys()])].filter((k) => (ia.get(k) ?? "—") !== (ib.get(k) ?? "—"));
  const results: [string, string, string][] = [
    ["Trip cost", usd(ra.tripCostUsd), usd(rb.tripCostUsd)],
    ["Gross", usd(ra.grossUsd), usd(rb.grossUsd)],
    ["Crew", usd(ra.crewUsd), usd(rb.crewUsd)],
    ["Net to the boat", usd(ra.netUsd, { signed: true }), usd(rb.netUsd, { signed: true })],
    ["Break-even catch", beLb(ra, "lb"), beLb(rb, "lb")],
    ["Break-even price", bePrice(ra, "lb"), bePrice(rb, "lb")],
  ];
  return (
    <table className="w-full table-fixed border-collapse text-[14.5px]" data-testid="econ-cmp-table">
      <thead>
        <tr className="text-left text-[13px] text-n-ink-3">
          <th scope="col" className="w-[40%] pb-1 font-semibold">
            <span className="sr-only">Item</span>
          </th>
          <th scope="col" className="truncate pb-1 text-right font-semibold">
            A · {a.label}
          </th>
          <th scope="col" className="truncate pb-1 text-right font-semibold">
            B · {b.label}
          </th>
        </tr>
      </thead>
      <tbody>
        {keys.length === 0 ? (
          <tr className="border-t border-n-line">
            <td colSpan={3} className="py-1.5 text-n-ink-3">
              Same inputs.
            </td>
          </tr>
        ) : (
          keys.map((k) => (
            <tr key={k} className="border-t border-n-line" data-testid="econ-cmp-diff">
              <th scope="row" className="py-1.5 pr-2 text-left font-normal text-n-ink-2">
                {k}
              </th>
              <td className="nx-data py-1.5 text-right">{ia.get(k) ?? "—"}</td>
              <td className="nx-data py-1.5 text-right">{ib.get(k) ?? "—"}</td>
            </tr>
          ))
        )}
        {results.map(([k, x, y], n) => (
          <tr key={k} className={`border-t ${n === 0 ? "border-n-line-strong" : "border-n-line"}`}>
            <th scope="row" className="py-1.5 pr-2 text-left font-semibold">
              {k}
            </th>
            <td className="nx-data py-1.5 text-right">{x}</td>
            <td className="nx-data py-1.5 text-right">{y}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ShareLink({ draft }: { draft: Draft }) {
  const [url, setUrl] = useState<string | null>(null);
  return (
    <div className="flex flex-col gap-2 border-t border-n-line pt-4" data-testid="econ-share">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <ActionButton
          testId="econ-share-copy"
          onClick={async () => {
            const u = `${window.location.origin}${window.location.pathname}${window.location.search}#${SHARE_PARAM}=${encodeShare(draft)}`;
            setUrl(u);
            try {
              await navigator.clipboard.writeText(u);
              toast("Link copied");
            } catch {
              // shown below to copy by hand
            }
          }}
        >
          Copy scenario link
        </ActionButton>
        <span className="text-[13.5px] text-n-ink-3">Numbers only, no names. Only if you press it.</span>
      </div>
      {url && <input readOnly value={url} aria-label="Scenario link" onFocus={(e) => e.currentTarget.select()} className="nx-focus nx-data min-h-11 w-full rounded-xl border border-n-line bg-n-surface-2 px-3 text-[14px] text-n-ink-2" data-testid="econ-share-url" />}
    </div>
  );
}

function StoreStatus() {
  const data = useTripData();
  if (!data) return null;
  const s = data.status;
  const msgs: string[] = [];
  if (!s.available) msgs.push("This browser is not keeping data for this site: the worksheet works, but nothing is kept.");
  if (s.readOnly) msgs.push("Trip data here was saved by a newer version of Neritic. It is shown read-only so nothing is overwritten.");
  if (s.corrupt.length) msgs.push(`Some trip data on this device could not be read (${s.corrupt.join(", ")}). It is kept, and a copy is set aside before anything new is saved. Export it below if you need it.`);
  if (s.invalidRecords) msgs.push(`${s.invalidRecords} saved item${s.invalidRecords === 1 ? "" : "s"} could not be read and ${s.invalidRecords === 1 ? "is" : "are"} kept unchanged.`);
  if (!msgs.length) return null;
  return (
    <div className="flex flex-col gap-1 rounded-xl border border-n-official-line bg-n-official-bg px-4 py-3 text-[14.5px]" role="status" data-testid="econ-store-status">
      {msgs.map((m) => (
        <p key={m}>{m}</p>
      ))}
    </div>
  );
}

function TripDataPanel() {
  const [confirm, setConfirm] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const file = useRef<HTMLInputElement | null>(null);
  return (
    <details className="border-t border-n-line pt-2" data-testid="econ-data">
      <summary className="nx-focus flex min-h-11 cursor-pointer items-center text-[15px] font-semibold text-n-accent">Your trip data</summary>
      <div className="flex flex-col gap-3 pb-2 pt-2">
        <p className="text-[14.5px] text-n-ink-2">Kept on this device only. Never sent to Neritic&rsquo;s server. Export a file to move it to another device.</p>
        <div className="flex flex-wrap gap-2">
          <ActionButton
            testId="econ-export"
            onClick={() => {
              const blob = new Blob([tripStore().exportJson()], { type: "application/json" });
              const a = document.createElement("a");
              a.href = URL.createObjectURL(blob);
              a.download = `neritic-trips-${new Date().toISOString().slice(0, 10)}.json`;
              a.click();
              setTimeout(() => URL.revokeObjectURL(a.href), 1000);
            }}
          >
            Export (JSON)
          </ActionButton>
          <ActionButton testId="econ-import" onClick={() => file.current?.click()}>
            Import
          </ActionButton>
          <input
            ref={file}
            type="file"
            accept="application/json,.json"
            className="sr-only"
            tabIndex={-1}
            aria-hidden
            data-testid="econ-import-file"
            onChange={async (e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (!f) return;
              const res = tripStore().importJson(await f.text());
              notifyTrip();
              setMsg(res.ok ? `Imported: ${res.added} added, ${res.updated} updated${res.skipped ? `, ${res.skipped} skipped (limit reached or unreadable)` : ""}.` : res.reason === "newer" ? "That file is from a newer version of Neritic." : res.reason === "readOnly" ? "Read-only here." : "That file is not a Neritic trip export.");
            }}
          />
          {confirm ? (
            <>
              <ActionButton
                primary
                testId="econ-delete-confirm"
                onClick={() => {
                  const res = tripStore().deleteAll();
                  notifyTrip();
                  setConfirm(false);
                  setMsg(res.ok ? "All trip data on this device was deleted." : "Could not delete here.");
                }}
              >
                Delete everything
              </ActionButton>
              <ActionButton testId="econ-delete-cancel" onClick={() => setConfirm(false)}>
                Keep it
              </ActionButton>
            </>
          ) : (
            <ActionButton testId="econ-delete-all" onClick={() => setConfirm(true)}>
              Delete all trip data
            </ActionButton>
          )}
        </div>
        {msg && (
          <p className="text-[14.5px] text-n-ink-2" role="status" data-testid="econ-data-msg">
            {msg}
          </p>
        )}
      </div>
    </details>
  );
}
