import { FISHERY_PRESETS, SOURCES, presetById, presetForSpecies, type FisheryPreset, type FisheryPresetId, type Sourced } from "@/content/fisheryPresets";
import { DIESEL_REF, dieselSnapshot, priceRef, priceSnapshot, type RefSnapshot } from "@/content/referenceValues";
import { emptyDraft, type Draft } from "@/lib/tripEconomicsStore";
import type { CostLine } from "@/lib/tripEconomics";

/** Cost lines every worksheet starts with: labels only, amounts blank (the user's own). */
export const STARTER_COSTS: CostLine[] = [
  { label: "Ice", usd: null, shared: true },
  { label: "Food and supplies", usd: null, shared: true },
  { label: "Gear and repairs (this trip)", usd: null, shared: false },
];

/** Field keys whose value came from a preset, and the preset value that sits there. */
function presetFields(p: FisheryPreset): [string, Sourced | null][] {
  return [
    ["fishingGph", p.fishingGph],
    ["steamGph", p.steamGph],
    ["crew.pct", p.crewPct],
    ["feesPctOfGross", p.feesPct],
  ];
}

/** A fresh worksheet for a fishery: its sourced defaults (marked as such), blank catch lines, no price. */
export function draftForPreset(p: FisheryPreset | null): Draft {
  const d = emptyDraft(p?.id ?? null);
  d.inputs.costs = STARTER_COSTS.map((c) => ({ ...c }));
  d.inputs.catch = (p?.catch ?? [{ key: "", label: "Main catch" }]).map((c) => ({ species: c.label, ref: c.key || null, lb: null, usdPerLb: null }));
  d.inputs.crew = { kind: p?.crewKind ?? "afterExpenses", pct: null };
  if (p) applyPreset(d, p);
  return d;
}

/** Fill a fishery's sourced defaults into the fields that are empty or still hold an earlier preset's value. */
export function applyPreset(d: Draft, p: FisheryPreset): Draft {
  d.preset = p.id;
  for (const [key, v] of presetFields(p)) {
    const cur = getField(d, key);
    if (cur !== null && d.basis[key] !== "preset") continue;
    setField(d, key, v ? v.value : null);
    if (v) d.basis[key] = "preset";
    else delete d.basis[key];
  }
  // catch lines that are still blank take the fishery's lines
  const blank = (d.inputs.catch ?? []).every((c) => c.lb === null && c.usdPerLb === null);
  if (blank) d.inputs.catch = p.catch.map((c) => ({ species: c.label, ref: c.key, lb: null, usdPerLb: null }));
  if (!p.potTiers) {
    d.pots = null;
    d.baitUsdPerPot = null;
  }
  return d;
}

export function initialDraft(species: string, preset: FisheryPresetId | null = null): Draft {
  return draftForPreset(presetById(preset) ?? presetForSpecies(species));
}

export function getField(d: Draft, key: string): number | null {
  const i = d.inputs;
  if (key.startsWith("crew.")) return (i.crew as Record<string, unknown>)?.[key.slice(5)] as number | null ?? null;
  if (key === "pots") return d.pots;
  if (key === "baitUsdPerPot") return d.baitUsdPerPot;
  return ((i as Record<string, unknown>)[key] as number | null | undefined) ?? null;
}

export function setField(d: Draft, key: string, v: number | null) {
  if (key.startsWith("crew.")) d.inputs.crew = { kind: d.inputs.crew?.kind ?? "afterExpenses", ...d.inputs.crew, [key.slice(5)]: v };
  else if (key === "pots") d.pots = v;
  else if (key === "baitUsdPerPot") d.baitUsdPerPot = v;
  else (d.inputs as Record<string, unknown>)[key] = v;
}

/** The source line for a field that still holds a preset value. */
export function presetSource(d: Draft, key: string): { text: string; url: string; note?: string } | null {
  if (d.basis[key] !== "preset") return null;
  const p = presetById(d.preset);
  if (!p) return null;
  const v = presetFields(p).find(([k]) => k === key)?.[1];
  if (!v) return null;
  const s = SOURCES[v.source];
  return { text: `${s.short}: ${v.label}`, url: s.url, note: s.note };
}

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
