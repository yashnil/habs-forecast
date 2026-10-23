import { FISHERY_PRESETS, SOURCES, presetById, presetForSpecies, presetReferences, type FisheryPreset, type FisheryPresetId, type PresetReference } from "@/content/fisheryPresets";
import { DIESEL_REF, dieselSnapshot, priceRef, priceSnapshot, type RefSnapshot } from "@/content/referenceValues";
import { emptyDraft, type Draft } from "@/lib/tripEconomicsStore";
import { speciesById } from "@/content/species";
import type { CostLine } from "@/lib/tripEconomics";

/** Cost lines every worksheet starts with: labels only, amounts blank (the user's own). */
export const STARTER_COSTS: CostLine[] = [
  { label: "Ice", usd: null, shared: true },
  { label: "Food and supplies", usd: null, shared: true },
  { label: "Gear and repairs (this trip)", usd: null, shared: false },
];

/**
 * A fresh worksheet (v10): the fishery's catch lines and labels only. Nothing numeric is filled in: the
 * fishery's sourced values (burn rates, crew shares, fees) are offered beside their fields as references
 * the user taps to use (decision D1), and how crew is paid starts unchosen (D2). A species with its own
 * CDFW price reference (rock crab) still gets that reference beside its catch line: shown, never filled.
 */
export function draftForPreset(p: FisheryPreset | null, species: string | null = null): Draft {
  const d = emptyDraft(p?.id ?? null);
  d.inputs.costs = STARTER_COSTS.map((c) => ({ ...c }));
  const own = !p && species && priceRef(species) ? { key: species, label: speciesById(species).label } : null;
  d.inputs.catch = (p?.catch ?? [own ?? { key: "", label: species && species !== "any" ? speciesById(species).label : "Main catch" }]).map((c) => ({ species: c.label, ref: c.key || null, lb: null, usdPerLb: null }));
  d.inputs.crew = { kind: null };
  return d;
}

/**
 * Another fishery chosen for a worksheet already in use: its catch lines replace blank ones, and any
 * value that still holds the previous fishery's reference is cleared (it described another fishery).
 * Nothing new is filled in.
 */
export function switchFishery(d: Draft, p: FisheryPreset | null): Draft {
  const prev = presetById(d.preset);
  for (const [key, b] of Object.entries(d.basis)) {
    if (b !== "preset") continue;
    if (prev && presetReferences(prev).some((r) => r.field === key)) {
      setField(d, key, null);
      delete d.basis[key];
    }
  }
  d.preset = p?.id ?? null;
  const blank = (d.inputs.catch ?? []).every((c) => c.lb === null && c.usdPerLb === null);
  if (blank && p) d.inputs.catch = p.catch.map((c) => ({ species: c.label, ref: c.key, lb: null, usdPerLb: null }));
  if (!p?.potTiers) {
    d.pots = null;
    d.baitUsdPerPot = null;
  }
  return d;
}

export function initialDraft(species: string, preset: FisheryPresetId | null = null): Draft {
  return draftForPreset(presetById(preset) ?? presetForSpecies(species), species);
}

export function getField(d: Draft, key: string): number | null {
  const i = d.inputs;
  if (key.startsWith("crew.")) return ((i.crew as Record<string, unknown> | null | undefined)?.[key.slice(5)] as number | null) ?? null;
  if (key === "pots") return d.pots;
  if (key === "baitUsdPerPot") return d.baitUsdPerPot;
  return ((i as Record<string, unknown>)[key] as number | null | undefined) ?? null;
}

export function setField(d: Draft, key: string, v: number | null) {
  if (key.startsWith("crew.")) d.inputs.crew = { kind: d.inputs.crew?.kind ?? null, ...d.inputs.crew, [key.slice(5)]: v };
  else if (key === "pots") d.pots = v;
  else if (key === "baitUsdPerPot") d.baitUsdPerPot = v;
  else (d.inputs as Record<string, unknown>)[key] = v;
}

/** The fishery reference a field offers (and that the field holds when its basis is "preset"). */
export function fieldReferences(d: Draft, key: string): PresetReference[] {
  const p = presetById(d.preset);
  const kind = d.inputs.crew?.kind ?? null;
  return presetReferences(p, key === "crew.pct" ? { crewKind: kind } : {}).filter((r) => r.field === key);
}

/** "NOAA NWFSC-121, Table 31 (2008 survey)": the short name of a reference's source. */
export const sourceShort = (r: PresetReference) => SOURCES[r.source].short;

/** What a saved scenario keeps about the references it used. */
export function refsUsed(d: Draft): RefSnapshot[] {
  const out: RefSnapshot[] = [];
  if (d.basis.fuelUsdPerGal === "eia") out.push(dieselSnapshot());
  (d.inputs.catch ?? []).forEach((c, i) => {
    const r = priceRef(c.ref);
    if (r && d.basis[`catch.${i}.usdPerLb`] === "cdfw") out.push(priceSnapshot(r));
  });
  return out;
}

export const FISHERY_OPTIONS = [...FISHERY_PRESETS.map((p) => ({ value: p.id as string, label: p.label })), { value: "", label: "Other (start blank)" }];

export { DIESEL_REF };
