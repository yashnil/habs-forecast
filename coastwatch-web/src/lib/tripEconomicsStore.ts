import { sanitizeInputs, clampNum, type CostLine } from "@/lib/tripEconomics";
import type { CrewChoice, TripScenario } from "@/lib/tripEvaluation";
import type { RefSnapshot } from "@/content/referenceValues";
import { presetById, type FisheryPresetId } from "@/content/fisheryPresets";

/**
 * Trip scenarios and vessels kept on this device (localStorage, namespace "nx-trip-v1:"). Never sent
 * to the server, never put in a URL unless the user presses "Copy scenario link" (numbers only).
 *
 * Keys: nx-trip-v1:scenarios, nx-trip-v1:vessels (arrays of records) and nx-trip-v1:meta (one object:
 * the working draft per place and species, and the unit). Every record is {schema, id, createdAt,
 * updatedAt, …}.
 * - A record or key written by a newer schema makes the whole store read-only here (nothing is
 *   overwritten that this version does not understand).
 * - A key that does not parse is kept and reported; before the first write over it, its raw text is
 *   copied to "nx-trip-v1:corrupt:<key>", so nothing is ever dropped. A record inside a good list that
 *   does not validate is carried along untouched and counted.
 * Pure apart from the Storage it is given (tests pass a Map-backed one).
 */

export const NS = "nx-trip-v1:";
/**
 * 2 (v10): the crew model may be unchosen (`crew.kind: null`, nothing assumed) or "none" (no hired
 * crew). Records from schema 0/1 are migrated on read: a share-after-expenses model with no share was
 * the old code default, never the user's choice, so it becomes unchosen; every other model is kept.
 */
export const SCHEMA = 2;
export const MAX_VESSELS = 5;
export const MAX_SCENARIOS = 20;
export const MAX_DRAFTS = 20;

export type Unit = "lb" | "kg";

/** What the worksheet holds: the calculator's inputs plus the things that only shape them. */
export type Draft = {
  preset: FisheryPresetId | null;
  /** the calculator's inputs; `crew.kind` is null until the user chooses how crew is paid */
  inputs: TripScenario;
  /** pots fished and bait per pot: become a shared cost line "Bait" */
  pots: number | null;
  baitUsdPerPot: number | null;
  /** where each value came from: "preset" (its source), "cdfw" / "eia" (a reference copied in), else the user's own */
  basis: Record<string, "preset" | "cdfw" | "eia">;
  /** how the worksheet was being filled in, where the inputs alone cannot say (absent = the defaults) */
  view?: DraftView;
};

/**
 * What the calculator's inputs cannot carry. The crew share is always stored as the crew's total; `shareBasis`
 * "each" only says the reader typed it per person. `payerUnset` lists the cost lines whose payer (shared off
 * the top, or the boat) has not been chosen: the stored `shared` flag of those lines means nothing yet.
 */
export type DraftView = { shareBasis?: "each"; fuelRoute?: "distance"; payerUnset?: number[] };

/** The longest name a saved trip keeps. */
export const MAX_NAME = 90;

export type BaseRecord = { schema: number; id: string; createdAt: string; updatedAt: string };
export type ScenarioRecord = BaseRecord & { name: string; placeKey: string; species: string; draft: Draft; refs: RefSnapshot[] };
export type VesselRecord = BaseRecord & { name: string; cruiseKt: number | null; steamGph: number | null; fishingGph: number | null; crew: number | null; pots: number | null };
/** `openId`: the saved trip the working draft was opened from, so saving after a reload updates it instead of adding a copy. */
export type WorkingDraft = { draft: Draft; updatedAt: string; openId?: string };
export type Meta = { schema: number; unit: Unit; drafts: Record<string, WorkingDraft> };

export type StoreStatus = {
  available: boolean;
  readOnly: boolean;
  /** keys whose stored text did not parse (kept; a copy is made before any write) */
  corrupt: ("scenarios" | "vessels" | "meta")[];
  /** records that did not validate inside an otherwise good list (kept, not shown) */
  invalidRecords: number;
};

export type TripData = { scenarios: ScenarioRecord[]; vessels: VesselRecord[]; meta: Meta; status: StoreStatus };

export type KV = Pick<Storage, "getItem" | "setItem" | "removeItem"> & { key?(i: number): string | null; readonly length?: number };

const KEYS = ["scenarios", "vessels", "meta"] as const;
type Key = (typeof KEYS)[number];

export const draftKey = (placeKey: string, species: string) => `${placeKey}|${species}`;

export function emptyDraft(preset: FisheryPresetId | null = null): Draft {
  return { preset, inputs: unchosenFuelShare({ ...sanitizeInputs({ catch: [], costs: [] }), crew: { kind: null } }, undefined), pots: null, baitUsdPerPot: null, basis: {} };
}

/** `sanitizeInputs` fills "fuel off the top" in; a draft keeps it only if the user (or a stored record) really has a value. */
function unchosenFuelShare(inputs: TripScenario, raw: unknown): TripScenario {
  if (typeof raw === "boolean") return { ...inputs, fuelShared: raw };
  const { fuelShared: _unset, ...rest } = inputs;
  void _unset;
  return rest;
}

/** The draft's calculator inputs: pots × bait per pot becomes a shared "Bait" cost. */
export function draftInputs(d: Draft): TripScenario {
  const bait: CostLine[] = d.pots != null && d.baitUsdPerPot != null ? [{ label: `Bait (${d.pots} pots)`, usd: d.pots * d.baitUsdPerPot, shared: true }] : [];
  return { ...d.inputs, costs: [...(d.inputs.costs ?? []), ...bait] };
}

// ---------------------------------------------------------------- validation and migration

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const str = (v: unknown, max = 80) => (typeof v === "string" ? v.slice(0, max) : "");
const iso = (v: unknown, fallback: string) => (typeof v === "string" && !Number.isNaN(Date.parse(v)) ? v : fallback);

/**
 * The crew model a stored draft holds. From schema 2 a missing or unknown kind is unchosen. Before it,
 * the code filled "afterExpenses" whenever nothing was chosen, so that kind with no share is read as
 * unchosen too; a share entered, or any other model, was the user's and is kept.
 */
export function storedCrewKind(raw: unknown, schema: number): CrewChoice | null {
  const c = isObj(raw) ? raw : null;
  const k = c?.kind;
  if (k === "none" || k === "ofGross" || k === "dayRate") return k;
  const pct = clampNum(c?.pct, "pct");
  if (k === "afterExpenses") return schema < 2 && pct === null ? null : "afterExpenses";
  // schema 0/1 without a kind meant share after expenses (the old default): kept only when a share is there
  return schema < 2 && pct !== null ? "afterExpenses" : null;
}

export function sanitizeDraft(v: unknown, schema: number = SCHEMA): Draft {
  const d = isObj(v) ? v : {};
  const preset = presetById(typeof d.preset === "string" ? d.preset : null)?.id ?? null;
  const basis: Draft["basis"] = {};
  if (isObj(d.basis)) for (const [k, b] of Object.entries(d.basis)) if (b === "preset" || b === "cdfw" || b === "eia") basis[k.slice(0, 40)] = b;
  const raw = isObj(d.inputs) ? d.inputs : {};
  const inputs = sanitizeInputs(raw as Parameters<typeof sanitizeInputs>[0]);
  const kind = storedCrewKind(raw.crew, schema);
  const anglers = clampNum(raw.anglers, "people");
  const view = sanitizeView(d.view, inputs.costs?.length ?? 0);
  return {
    preset,
    inputs: unchosenFuelShare({ ...inputs, crew: { ...inputs.crew, kind, ...(kind === "none" ? { pct: null } : {}) }, ...(anglers !== null ? { anglers } : {}) }, raw.fuelShared),
    pots: clampNum(d.pots, "pots"),
    baitUsdPerPot: clampNum(d.baitUsdPerPot, "usdPerDay"),
    basis,
    ...(view ? { view } : {}),
  };
}

function sanitizeView(v: unknown, costLines: number): DraftView | null {
  if (!isObj(v)) return null;
  const out: DraftView = {};
  if (v.shareBasis === "each") out.shareBasis = "each";
  if (v.fuelRoute === "distance") out.fuelRoute = "distance";
  if (Array.isArray(v.payerUnset)) {
    const idx = [...new Set(v.payerUnset.filter((n): n is number => Number.isInteger(n) && n >= 0 && n < costLines))].sort((a, b) => a - b);
    if (idx.length) out.payerUnset = idx;
  }
  return Object.keys(out).length ? out : null;
}

function sanitizeRefs(v: unknown): RefSnapshot[] {
  if (!Array.isArray(v)) return [];
  return v
    .filter(isObj)
    .filter((r) => typeof r.value === "number" && Number.isFinite(r.value))
    .slice(0, 20)
    .map((r) => ({ key: str(r.key), value: r.value as number, unit: str(r.unit, 12), source: str(r.source, 300), period: str(r.period, 80), fetchedAt: str(r.fetchedAt, 40) }));
}

/**
 * One stored record brought to the current schema. A record from before schema 1 (no `schema`, the
 * pre-release drafts) is upgraded: timestamps filled, inputs made safe. Returns "newer" for a record
 * this version cannot read, null when it is not a record at all.
 */
export function migrateRecord(v: unknown, kind: "scenario" | "vessel", now: string): ScenarioRecord | VesselRecord | "newer" | null {
  if (!isObj(v)) return null;
  const schema = v.schema === undefined ? 0 : v.schema;
  if (typeof schema !== "number" || !Number.isInteger(schema) || schema < 0) return null;
  if (schema > SCHEMA) return "newer";
  if (typeof v.id !== "string" || v.id === "") return null;
  const createdAt = iso(v.createdAt, now);
  const base: BaseRecord = { schema: SCHEMA, id: v.id.slice(0, 40), createdAt, updatedAt: iso(v.updatedAt, createdAt) };
  if (kind === "vessel") {
    return { ...base, name: str(v.name, 60) || "Vessel", cruiseKt: clampNum(v.cruiseKt, "kt"), steamGph: clampNum(v.steamGph, "gph"), fishingGph: clampNum(v.fishingGph, "gph"), crew: clampNum(v.crew, "people"), pots: clampNum(v.pots, "pots") };
  }
  // schema 0 kept the inputs at the top level as `inputs`, without a draft wrapper
  const draftSrc = isObj(v.draft) ? v.draft : schema === 0 && isObj(v.inputs) ? { inputs: v.inputs, preset: v.preset } : null;
  if (!draftSrc) return null;
  return { ...base, name: str(v.name, MAX_NAME) || "Scenario", placeKey: str(v.placeKey, 60), species: str(v.species, 30), draft: sanitizeDraft(draftSrc, schema), refs: sanitizeRefs(v.refs) };
}

// ---------------------------------------------------------------- the store

type Raw = { text: string | null; parsed: unknown; corrupt: boolean };

function readRaw(kv: KV, k: Key): Raw {
  let text: string | null = null;
  try {
    text = kv.getItem(NS + k);
  } catch {
    return { text: null, parsed: null, corrupt: false };
  }
  if (text === null) return { text, parsed: null, corrupt: false };
  try {
    return { text, parsed: JSON.parse(text), corrupt: false };
  } catch {
    return { text, parsed: null, corrupt: true };
  }
}

export function defaultMeta(): Meta {
  return { schema: SCHEMA, unit: "lb", drafts: {} };
}

export type TripStore = ReturnType<typeof createTripStore>;

/** "Name (copy)", then "Name (copy 2)", …: never a name already in the list. */
export function copyName(name: string, taken: readonly string[]): string {
  const base = name.replace(/ \(copy(?: \d+)?\)$/, "");
  const fit = (suffix: string) => `${base.slice(0, MAX_NAME - suffix.length)}${suffix}`;
  let out = fit(" (copy)");
  for (let n = 2; taken.includes(out) && n < 100; n++) out = fit(` (copy ${n})`);
  return out;
}

export function createTripStore(kv: KV | null, clock: () => Date = () => new Date()) {
  const nowIso = () => clock().toISOString();
  /** records that did not validate, carried along on every write so they are never dropped */
  const passthrough: Record<"scenarios" | "vessels", unknown[]> = { scenarios: [], vessels: [] };

  function load(): TripData {
    const status: StoreStatus = { available: kv !== null, readOnly: false, corrupt: [], invalidRecords: 0 };
    const out: TripData = { scenarios: [], vessels: [], meta: defaultMeta(), status };
    if (!kv) return out;
    const now = nowIso();
    for (const k of ["scenarios", "vessels"] as const) {
      passthrough[k] = [];
      const r = readRaw(kv, k);
      if (r.corrupt) {
        status.corrupt.push(k);
        continue;
      }
      if (r.parsed === null) continue;
      if (!Array.isArray(r.parsed)) {
        if (isObj(r.parsed) && typeof r.parsed.schema === "number" && r.parsed.schema > SCHEMA) status.readOnly = true;
        else status.corrupt.push(k);
        continue;
      }
      for (const item of r.parsed) {
        const m = migrateRecord(item, k === "scenarios" ? "scenario" : "vessel", now);
        if (m === "newer") {
          status.readOnly = true;
          passthrough[k].push(item);
        } else if (m === null) {
          status.invalidRecords += 1;
          passthrough[k].push(item);
        } else if (k === "scenarios") out.scenarios.push(m as ScenarioRecord);
        else out.vessels.push(m as VesselRecord);
      }
    }
    const rm = readRaw(kv, "meta");
    if (rm.corrupt) status.corrupt.push("meta");
    else if (isObj(rm.parsed)) {
      const schema = typeof rm.parsed.schema === "number" ? rm.parsed.schema : 0;
      if (schema > SCHEMA) status.readOnly = true;
      else {
        const drafts: Meta["drafts"] = {};
        if (isObj(rm.parsed.drafts)) {
          for (const [k, v] of Object.entries(rm.parsed.drafts).slice(0, MAX_DRAFTS)) {
            if (isObj(v)) drafts[k.slice(0, 100)] = { draft: sanitizeDraft(v.draft, schema), updatedAt: iso(v.updatedAt, now), ...(typeof v.openId === "string" && v.openId ? { openId: v.openId.slice(0, 40) } : {}) };
          }
        }
        out.meta = { schema: SCHEMA, unit: rm.parsed.unit === "kg" ? "kg" : "lb", drafts };
      }
    } else if (rm.parsed !== null) status.corrupt.push("meta");
    out.scenarios.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    out.vessels.sort((a, b) => a.createdAt.localeCompare(b.createdAt));
    return out;
  }

  type WriteResult = { ok: true } | { ok: false; reason: "unavailable" | "readOnly" | "full" | "quota" };

  function write(k: Key, value: unknown): WriteResult {
    if (!kv) return { ok: false, reason: "unavailable" };
    try {
      const r = readRaw(kv, k);
      // a damaged value is copied aside before anything replaces it (once: the first copy is the original)
      if (r.corrupt || (k !== "meta" && r.parsed !== null && !Array.isArray(r.parsed)) || (k === "meta" && r.parsed !== null && !isObj(r.parsed))) {
        const aside = `${NS}corrupt:${k}`;
        if (kv.getItem(aside) === null && r.text !== null) kv.setItem(aside, r.text);
      }
      kv.setItem(NS + k, JSON.stringify(value));
      return { ok: true };
    } catch {
      return { ok: false, reason: "quota" };
    }
  }

  function guard(d: TripData): WriteResult | null {
    if (!d.status.available) return { ok: false, reason: "unavailable" };
    if (d.status.readOnly) return { ok: false, reason: "readOnly" };
    return null;
  }

  const putScenarios = (list: ScenarioRecord[]) => write("scenarios", [...list, ...passthrough.scenarios]);
  const putVessels = (list: VesselRecord[]) => write("vessels", [...list, ...passthrough.vessels]);

  const newId = (p: string) => `${p}-${clock().getTime().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;

  return {
    load,

    saveScenario(input: { id?: string; name: string; placeKey: string; species: string; draft: Draft; refs: RefSnapshot[] }): WriteResult & { id?: string } {
      const d = load();
      const g = guard(d);
      if (g) return g;
      const now = nowIso();
      const existing = input.id ? d.scenarios.find((s) => s.id === input.id) : undefined;
      if (!existing && d.scenarios.length >= MAX_SCENARIOS) return { ok: false, reason: "full" };
      const rec: ScenarioRecord = {
        schema: SCHEMA,
        id: existing?.id ?? newId("s"),
        createdAt: existing?.createdAt ?? now,
        updatedAt: now,
        name: input.name.trim().slice(0, MAX_NAME) || "Scenario",
        placeKey: input.placeKey,
        species: input.species,
        draft: sanitizeDraft(input.draft),
        refs: sanitizeRefs(input.refs),
      };
      const r = putScenarios([rec, ...d.scenarios.filter((s) => s.id !== rec.id)]);
      return r.ok ? { ok: true, id: rec.id } : r;
    },

    duplicateScenario(id: string): WriteResult & { id?: string } {
      const d = load();
      const g = guard(d);
      if (g) return g;
      const src = d.scenarios.find((s) => s.id === id);
      if (!src) return { ok: false, reason: "unavailable" };
      if (d.scenarios.length >= MAX_SCENARIOS) return { ok: false, reason: "full" };
      const now = nowIso();
      const rec: ScenarioRecord = { ...src, id: newId("s"), name: copyName(src.name, d.scenarios.map((s) => s.name)), createdAt: now, updatedAt: now };
      const r = putScenarios([rec, ...d.scenarios]);
      return r.ok ? { ok: true, id: rec.id } : r;
    },

    /** Put back a trip that was just deleted, exactly as it was (its id, name and times). Does nothing if it is already there. */
    restoreScenario(rec: ScenarioRecord): WriteResult {
      const d = load();
      const g = guard(d);
      if (g) return g;
      if (d.scenarios.some((s) => s.id === rec.id)) return { ok: true };
      if (d.scenarios.length >= MAX_SCENARIOS) return { ok: false, reason: "full" };
      const back = migrateRecord(rec, "scenario", nowIso());
      if (!back || back === "newer") return { ok: false, reason: "unavailable" };
      return putScenarios([...d.scenarios, back as ScenarioRecord]);
    },

    renameScenario(id: string, name: string): WriteResult {
      const d = load();
      const g = guard(d);
      if (g) return g;
      const now = nowIso();
      return putScenarios(d.scenarios.map((s) => (s.id === id ? { ...s, name: name.trim().slice(0, MAX_NAME) || s.name, updatedAt: now } : s)));
    },

    deleteScenario(id: string): WriteResult {
      const d = load();
      const g = guard(d);
      if (g) return g;
      return putScenarios(d.scenarios.filter((s) => s.id !== id));
    },

    saveVessel(input: Omit<VesselRecord, keyof BaseRecord> & { id?: string }): WriteResult & { id?: string } {
      const d = load();
      const g = guard(d);
      if (g) return g;
      const existing = input.id ? d.vessels.find((v) => v.id === input.id) : undefined;
      if (!existing && d.vessels.length >= MAX_VESSELS) return { ok: false, reason: "full" };
      const now = nowIso();
      const rec = migrateRecord({ ...input, schema: SCHEMA, id: existing?.id ?? newId("v"), createdAt: existing?.createdAt ?? now, updatedAt: now }, "vessel", now) as VesselRecord;
      const r = putVessels([...d.vessels.filter((v) => v.id !== rec.id), rec]);
      return r.ok ? { ok: true, id: rec.id } : r;
    },

    deleteVessel(id: string): WriteResult {
      const d = load();
      const g = guard(d);
      if (g) return g;
      return putVessels(d.vessels.filter((v) => v.id !== id));
    },

    /** The working draft for a place and species (autosaved as it is edited). */
    saveDraft(key: string, draft: Draft, openId?: string | null): WriteResult {
      const d = load();
      const g = guard(d);
      if (g) return g;
      const now = nowIso();
      const entries = Object.entries({ ...d.meta.drafts, [key]: { draft: sanitizeDraft(draft), updatedAt: now, ...(openId ? { openId } : {}) } })
        .sort((a, b) => b[1].updatedAt.localeCompare(a[1].updatedAt))
        .slice(0, MAX_DRAFTS);
      return write("meta", { ...d.meta, drafts: Object.fromEntries(entries) });
    },

    /** Forget the working draft for a place and fishery (Clear). */
    clearDraft(key: string): WriteResult {
      const d = load();
      const g = guard(d);
      if (g) return g;
      if (!(key in d.meta.drafts)) return { ok: true };
      const { [key]: _gone, ...rest } = d.meta.drafts;
      void _gone;
      return write("meta", { ...d.meta, drafts: rest });
    },

    setUnit(unit: Unit): WriteResult {
      const d = load();
      const g = guard(d);
      if (g) return g;
      return write("meta", { ...d.meta, unit });
    },

    exportJson(): string {
      const d = load();
      return JSON.stringify({ app: "neritic-trip", schema: SCHEMA, exportedAt: nowIso(), scenarios: d.scenarios, vessels: d.vessels }, null, 2);
    },

    /** Merge an export: same id → the newer record wins; caps respected (the rest are counted, not added). */
    importJson(text: string): { ok: true; added: number; updated: number; skipped: number } | { ok: false; reason: "unavailable" | "readOnly" | "invalid" | "newer" | "quota" | "full" } {
      const d = load();
      const g = guard(d);
      if (g) return g as { ok: false; reason: "unavailable" | "readOnly" };
      let parsed: unknown;
      try {
        parsed = JSON.parse(text);
      } catch {
        return { ok: false, reason: "invalid" };
      }
      if (!isObj(parsed) || parsed.app !== "neritic-trip") return { ok: false, reason: "invalid" };
      if (typeof parsed.schema === "number" && parsed.schema > SCHEMA) return { ok: false, reason: "newer" };
      const now = nowIso();
      let added = 0;
      let updated = 0;
      let skipped = 0;
      const merge = <T extends ScenarioRecord | VesselRecord>(have: T[], incoming: unknown, kind: "scenario" | "vessel", cap: number): T[] => {
        const list = [...have];
        for (const item of Array.isArray(incoming) ? incoming : []) {
          const m = migrateRecord(item, kind, now);
          if (m === null || m === "newer") {
            skipped += 1;
            continue;
          }
          const i = list.findIndex((x) => x.id === m.id);
          if (i >= 0) {
            if (m.updatedAt > list[i].updatedAt) {
              list[i] = m as T;
              updated += 1;
            }
          } else if (list.length < cap) {
            list.push(m as T);
            added += 1;
          } else skipped += 1;
        }
        return list;
      };
      const scenarios = merge(d.scenarios, parsed.scenarios, "scenario", MAX_SCENARIOS);
      const vessels = merge(d.vessels, parsed.vessels, "vessel", MAX_VESSELS);
      const a = putScenarios(scenarios);
      const b = putVessels(vessels);
      if (!a.ok || !b.ok) return { ok: false, reason: "quota" };
      return { ok: true, added, updated, skipped };
    },

    /** Everything under nx-trip-v1:, including copies of damaged data. */
    deleteAll(): WriteResult {
      if (!kv) return { ok: false, reason: "unavailable" };
      try {
        const keys: string[] = [];
        if (typeof kv.key === "function" && typeof kv.length === "number") {
          for (let i = 0; i < kv.length; i++) {
            const k = kv.key(i);
            if (k && k.startsWith(NS)) keys.push(k);
          }
        }
        for (const k of KEYS) keys.push(NS + k, `${NS}corrupt:${k}`);
        for (const k of new Set(keys)) kv.removeItem(k);
        passthrough.scenarios = [];
        passthrough.vessels = [];
        return { ok: true };
      } catch {
        return { ok: false, reason: "quota" };
      }
    },
  };
}

// ---------------------------------------------------------------- scenario links (numbers only)

/**
 * "Copy scenario link": the draft's numbers in the URL fragment (never sent to a server), base64url
 * JSON. No names: cost lines lose their labels, catch lines keep only their reference key (labels come
 * back from the fishery preset).
 */
export function encodeShare(d: Draft): string {
  const i = d.inputs;
  const n = (v: number | null | undefined) => (typeof v === "number" && Number.isFinite(v) ? Math.round(v * 1e4) / 1e4 : null);
  const payload = {
    v: 1,
    p: d.preset,
    n: [i.roundTripNm, i.cruiseKt, i.steamGph, i.fishingHours, i.fishingGph, i.fuelGal, i.fuelUsdPerGal, i.feesPctOfGross, i.feesUsdPerLb, d.pots, d.baitUsdPerPot].map(n),
    // absent = not chosen yet
    ...(typeof i.fuelShared === "boolean" ? { fs: i.fuelShared ? 1 : 0 } : {}),
    c: (i.costs ?? []).map((c) => [n(c.usd), c.shared ? 1 : 0]),
    k: (i.catch ?? []).map((c) => [c.ref ?? "", n(c.lb), n(c.usdPerLb)]),
    // "" = how crew is paid is not chosen yet
    m: [i.crew?.kind ?? "", n(i.crew?.pct), n(i.crew?.crew), n(i.crew?.usdPerDay), n(i.crew?.days)],
    ...(i.anglers != null ? { a: n(i.anglers) } : {}),
    // how it was filled in (per-person share, distance route, payers not chosen): still numbers and flags only
    ...(d.view ? { w: [d.view.shareBasis === "each" ? 1 : 0, d.view.fuelRoute === "distance" ? 1 : 0, d.view.payerUnset ?? []] } : {}),
  };
  const bytes = new TextEncoder().encode(JSON.stringify(payload));
  let bin = "";
  bytes.forEach((b) => (bin += String.fromCharCode(b)));
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function decodeShare(s: string): Draft | null {
  try {
    if (s.length > 4000) return null;
    const bin = atob(s.replace(/-/g, "+").replace(/_/g, "/"));
    const p = JSON.parse(new TextDecoder().decode(Uint8Array.from(bin, (c) => c.charCodeAt(0))));
    if (!isObj(p) || p.v !== 1 || !Array.isArray(p.n)) return null;
    const [roundTripNm, cruiseKt, steamGph, fishingHours, fishingGph, fuelGal, fuelUsdPerGal, feesPctOfGross, feesUsdPerLb, pots, baitUsdPerPot] = p.n as unknown[];
    const m = Array.isArray(p.m) ? p.m : [];
    const preset = presetById(typeof p.p === "string" ? p.p : null);
    const label = (ref: string, idx: number) => (preset?.catch[idx]?.key === ref ? preset.catch[idx].label : preset?.catch.find((c) => c.key === ref)?.label) ?? (idx === 0 ? "Main catch" : `Catch ${idx + 1}`);
    return sanitizeDraft({
      preset: preset?.id ?? null,
      pots,
      baitUsdPerPot,
      inputs: {
        roundTripNm,
        cruiseKt,
        steamGph,
        fishingHours,
        fishingGph,
        fuelGal,
        fuelUsdPerGal,
        feesPctOfGross,
        feesUsdPerLb,
        fuelShared: p.fs === 0 ? false : p.fs === 1 ? true : undefined,
        costs: (Array.isArray(p.c) ? p.c : []).map((c: unknown, j: number) => (Array.isArray(c) ? { label: `Cost ${j + 1}`, usd: c[0], shared: c[1] !== 0 } : null)).filter(Boolean),
        catch: (Array.isArray(p.k) ? p.k : []).map((c: unknown, j: number) => (Array.isArray(c) ? { species: label(typeof c[0] === "string" ? c[0] : "", j), ref: typeof c[0] === "string" && c[0] ? c[0] : null, lb: c[1], usdPerLb: c[2] } : null)).filter(Boolean),
        crew: { kind: m[0], pct: m[1], crew: m[2], usdPerDay: m[3], days: m[4] },
        anglers: p.a,
      },
      ...(Array.isArray(p.w) ? { view: { shareBasis: p.w[0] === 1 ? "each" : undefined, fuelRoute: p.w[1] === 1 ? "distance" : undefined, payerUnset: p.w[2] } } : {}),
    }, SCHEMA);
  } catch {
    return null;
  }
}

/** The fragment that carries a shared scenario: "#trip=<encoded>". */
export const SHARE_PARAM = "trip";
