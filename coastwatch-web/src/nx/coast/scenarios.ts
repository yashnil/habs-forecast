import { dieselSnapshot, priceRef, priceSnapshot, type RefSnapshot } from "@/content/referenceValues";
import { stableFormat } from "@/lib/time";
import { compareScenarios, fieldLabel, type FieldKey, type ScalarOutputKey } from "@/lib/tripEvaluation";
import { MAX_NAME, MAX_SCENARIOS, SHARE_PARAM, decodeShare, draftInputs, draftKey, encodeShare, sanitizeDraft, type Draft, type ScenarioRecord, type TripData, type TripStore } from "@/lib/tripEconomicsStore";
import { audienceOf, tripFishery, type CoastFishery } from "./fisheries";
import { breakEvenCatchText, breakEvenPriceText, emptyWorkspace, evaluateWorkspace, fromDraft, hasEntry, money, read, toDraft, type Workspace } from "./workspace";

/**
 * Saved trips (v13): what Save, Update, Duplicate, Rename and Delete do, over the trip store
 * (src/lib/tripEconomicsStore.ts, localStorage "nx-trip-v1:*"). Pure apart from the store it is handed.
 *
 * The rules that keep the list honest:
 * - the working draft is autosaved with the id of the trip it was opened from, so a reload or a restart
 *   never turns "save my changes" into a second copy;
 * - Save as new refuses a trip whose inputs are identical to one already saved for this port and fishery,
 *   and names that trip, unless the reader insists;
 * - a name says what the trip was (its numbers, how crew is paid, the minute it was saved) and is never one
 *   already in the list;
 * - the list holds every port's trips, because the device's limit counts them all.
 */

const PACIFIC = "America/Los_Angeles";

// ---------------------------------------------------------------- identical inputs

const key = (d: Draft) => {
  const s = sanitizeDraft(d);
  return JSON.stringify([s.inputs, s.view ?? null]);
};

/** The same numbers and choices (where a reference came from is not an input). */
export const sameInputs = (a: Draft, b: Draft) => key(a) === key(b);

/** A trip already saved for this port and fishery with exactly these inputs. */
export function findIdentical(list: readonly ScenarioRecord[], placeKey: string, species: string, draft: Draft, exceptId?: string | null): ScenarioRecord | null {
  const k = key(draft);
  return list.find((s) => s.id !== exceptId && s.placeKey === placeKey && s.species === species && key(s.draft) === k) ?? null;
}

// ---------------------------------------------------------------- names

const num = (t: string) => t.trim().replace(/^\$/, "");

/** What tells this trip from another: "400 lb at $9, 1 crew 20 % after costs". Empty when nothing is typed. */
export function tripDetail(ws: Workspace, f: CoastFishery | null): string {
  const unit = f?.unit ?? "lb";
  const { n } = read(ws, f);
  const parts: string[] = [];
  if (audienceOf(f) === "recreational") {
    if (n.gallons !== null) parts.push(`${Math.round(n.gallons * 10) / 10} gal`);
    if (n.anglers) parts.push(`${n.anglers} ${n.anglers === 1 ? "person" : "people"}`);
    return parts.join(", ");
  }
  const c = ws.catch[0];
  if (c && num(c.amount) && num(c.price)) parts.push(`${num(c.amount)} ${unit} at $${num(c.price)}`);
  else if (c && num(c.amount)) parts.push(`${num(c.amount)} ${unit}`);
  if (ws.catch.length > 1) parts.push(`${ws.catch.length} species`);
  if (n.crew === 0) parts.push("no crew");
  else if (n.crew !== null && ws.crewKind) {
    const who = `${n.crew} crew`;
    if (ws.crewKind === "dayRate") parts.push(n.dayRate !== null ? `${who} at $${num(ws.dayRate)} a day` : who);
    else if (n.share !== null) parts.push(`${who} ${num(ws.share)} %${ws.shareBasis === "each" && n.crew > 1 ? " each" : ""} ${ws.crewKind === "afterExpenses" ? "after costs" : "of gross"}`);
    else parts.push(who);
  }
  return parts.join(", ");
}

/** "Nov 12, 12:41 AM", Pacific unless told otherwise. */
export function savedAtWords(at: Date | string, timeZone: string = PACIFIC): string {
  return stableFormat(new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone }), new Date(at));
}

/** `name`, or `name (2)`, `name (3)`…: never one already taken, never longer than the store keeps. */
export function uniqueName(name: string, taken: readonly string[]): string {
  const base = name.trim().slice(0, MAX_NAME) || "Trip";
  if (!taken.includes(base)) return base;
  for (let i = 2; i < 200; i++) {
    const suffix = ` (${i})`;
    const out = `${base.slice(0, MAX_NAME - suffix.length)}${suffix}`;
    if (!taken.includes(out)) return out;
  }
  return base;
}

/** A saved trip's name: the fishery, what tells it apart, and the minute it was saved. Unique in the list. */
export function tripName(ws: Workspace, f: CoastFishery | null, o: { now: Date; taken?: readonly string[]; timeZone?: string }): string {
  const detail = tripDetail(ws, f);
  const when = savedAtWords(o.now, o.timeZone);
  const head = [f?.label ?? "Trip", detail].filter(Boolean).join(", ");
  // the time is what makes two trips with the same numbers readable apart, so it is the part that always fits
  const room = MAX_NAME - when.length - 2;
  return uniqueName(`${head.length > room ? `${head.slice(0, room - 1).trimEnd()}…` : head}, ${when}`, o.taken ?? []);
}

// ---------------------------------------------------------------- save, update, duplicate, rename, delete

function refSnapshots(ws: Workspace, f: CoastFishery | null): RefSnapshot[] {
  const out: RefSnapshot[] = [];
  if (ws.taken.fuelPrice) out.push(dieselSnapshot());
  const pr = priceRef(f?.priceKey);
  if (ws.taken.price && pr) out.push(priceSnapshot(pr));
  return out;
}

type Failure = "unavailable" | "readOnly" | "full" | "quota";
export type SaveOutcome =
  | { ok: true; id: string; name: string }
  /** these exact inputs are already saved for this port and fishery, as this trip */
  | { ok: false; reason: "identical"; id: string; name: string }
  /** the trip being updated is no longer in the list (deleted in another tab) */
  | { ok: false; reason: "gone" }
  | { ok: false; reason: "empty" }
  | { ok: false; reason: Failure };

/** What a failed write means, in the reader's words. */
export function failureWords(reason: Exclude<SaveOutcome, { ok: true }>["reason"], name?: string): string {
  switch (reason) {
    case "identical":
      return `These exact numbers are already saved as “${name ?? "another trip"}”.`;
    case "gone":
      return "That trip is no longer saved here. Save this one as new to keep it.";
    case "empty":
      return "Nothing is entered yet.";
    case "full":
      return `${MAX_SCENARIOS} trips are saved on this device, across every port, and that is the most it keeps. Delete one first.`;
    case "readOnly":
      return "A newer Neritic saved trips here; nothing was changed.";
    default:
      return "This browser is not keeping data here (a private window?). Nothing was saved.";
  }
}

export type TripInput = { ws: Workspace; fishery: CoastFishery | null; placeKey: string };

/** Save as new. Refuses an exact copy of a saved trip (and says which) unless `allowIdentical`. */
export function saveNew(store: TripStore, t: TripInput, o: { now: Date; name?: string; allowIdentical?: boolean; timeZone?: string }): SaveOutcome {
  if (!hasEntry(t.ws)) return { ok: false, reason: "empty" };
  const d = store.load();
  const species = t.fishery?.id ?? "";
  const draft = toDraft(t.ws, t.fishery);
  const twin = o.allowIdentical ? null : findIdentical(d.scenarios, t.placeKey, species, draft);
  if (twin) return { ok: false, reason: "identical", id: twin.id, name: twin.name };
  const taken = d.scenarios.map((s) => s.name);
  const name = o.name?.trim() ? uniqueName(o.name, taken) : tripName(t.ws, t.fishery, { now: o.now, taken, timeZone: o.timeZone });
  const r = store.saveScenario({ name, placeKey: t.placeKey, species, draft, refs: refSnapshots(t.ws, t.fishery) });
  return r.ok ? { ok: true, id: r.id!, name } : { ok: false, reason: r.reason };
}

/** Update the open trip in place: its id, name and creation time stay; nothing new is added. */
export function updateTrip(store: TripStore, id: string, t: TripInput): SaveOutcome {
  const rec = store.load().scenarios.find((s) => s.id === id);
  if (!rec) return { ok: false, reason: "gone" };
  const r = store.saveScenario({ id, name: rec.name, placeKey: t.placeKey, species: t.fishery?.id ?? "", draft: toDraft(t.ws, t.fishery), refs: refSnapshots(t.ws, t.fishery) });
  return r.ok ? { ok: true, id, name: rec.name } : { ok: false, reason: r.reason };
}

/** A deliberate copy: "Name (copy)", then "(copy 2)"… */
export function duplicateTrip(store: TripStore, id: string): SaveOutcome {
  const r = store.duplicateScenario(id);
  if (!r.ok) return { ok: false, reason: r.reason === "unavailable" && store.load().status.available ? "gone" : r.reason };
  const rec = store.load().scenarios.find((s) => s.id === r.id);
  return { ok: true, id: r.id!, name: rec?.name ?? "" };
}

/** Rename, never to nothing and never to a name another trip has. */
export function renameTrip(store: TripStore, id: string, name: string): SaveOutcome {
  if (!name.trim()) return { ok: false, reason: "empty" };
  const d = store.load();
  if (!d.scenarios.some((s) => s.id === id)) return { ok: false, reason: "gone" };
  const next = uniqueName(name, d.scenarios.filter((s) => s.id !== id).map((s) => s.name));
  const r = store.renameScenario(id, next);
  return r.ok ? { ok: true, id, name: next } : { ok: false, reason: r.reason };
}

export type DeleteOutcome = { ok: true; record: ScenarioRecord; /** puts the trip back exactly as it was */ undo: () => boolean } | { ok: false; reason: "gone" | Failure };

/** Delete, with the way back: the record is handed over so the page can offer Undo. */
export function deleteTrip(store: TripStore, id: string): DeleteOutcome {
  const record = store.load().scenarios.find((s) => s.id === id);
  if (!record) return { ok: false, reason: "gone" };
  const r = store.deleteScenario(id);
  if (!r.ok) return { ok: false, reason: r.reason };
  return { ok: true, record, undo: () => store.restoreScenario(record).ok };
}

// ---------------------------------------------------------------- the working draft

export type Working = { ws: Workspace; openId: string | null; from: "working" | "fresh" };

/** The worksheet a port and fishery opens with: the draft autosaved here, else a blank one. */
export function loadWorking(data: TripData, placeKey: string, fishery: CoastFishery | null): Working {
  const w = data.meta.drafts[draftKey(placeKey, fishery?.id ?? "")];
  if (!w) return { ws: emptyWorkspace(fishery), openId: null, from: "fresh" };
  // the trip it was opened from may have been deleted since: then it is simply unsaved work
  const openId = w.openId && data.scenarios.some((s) => s.id === w.openId) ? w.openId : null;
  return { ws: fromDraft(w.draft, fishery?.id ?? null), openId, from: "working" };
}

/** Autosave. An untouched worksheet with no trip open leaves nothing behind. */
export function saveWorking(store: TripStore, placeKey: string, fishery: CoastFishery | null, ws: Workspace, openId: string | null): boolean {
  const k = draftKey(placeKey, fishery?.id ?? "");
  if (!hasEntry(ws) && !openId) return store.clearDraft(k).ok;
  return store.saveDraft(k, toDraft(ws, fishery), openId).ok;
}

export type TripState =
  | { kind: "empty"; words: string }
  /** typed, never saved */
  | { kind: "unsaved"; words: string }
  /** a saved trip is open and unchanged */
  | { kind: "saved"; id: string; name: string; words: string }
  /** a saved trip is open and the numbers have moved since */
  | { kind: "changed"; id: string; name: string; words: string };

/** Where the worksheet stands against the saved list: what Save would do, said before it is pressed. */
export function tripState(data: TripData | null, ws: Workspace, fishery: CoastFishery | null, openId: string | null, timeZone?: string): TripState {
  const rec = openId ? data?.scenarios.find((s) => s.id === openId) : undefined;
  if (rec) {
    if (sameInputs(rec.draft, toDraft(ws, fishery))) return { kind: "saved", id: rec.id, name: rec.name, words: `Saved ${savedAtWords(rec.updatedAt, timeZone)}` };
    return { kind: "changed", id: rec.id, name: rec.name, words: "Changes not saved" };
  }
  return hasEntry(ws) ? { kind: "unsaved", words: "Not saved" } : { kind: "empty", words: "Nothing entered" };
}

// ---------------------------------------------------------------- the list

export type TripRow = {
  id: string;
  name: string;
  placeKey: string;
  /** the port's name, or null when this device no longer knows the port */
  port: string | null;
  fishery: string;
  /** saved for the port the reader is at now */
  here: boolean;
  /** "400 lb at $9, 1 crew 20 % after costs" */
  detail: string;
  /** "Net to boat $2,602.24", "Trip cost $247.20", or what it still waits for */
  outcome: string;
  updatedAt: string;
  savedAt: string;
};

export type Capacity = { used: number; max: number; left: number; full: boolean; /** "3 of 20 trips saved on this device, across every port." */ words: string };

export function capacity(data: TripData | null): Capacity {
  const used = data?.scenarios.length ?? 0;
  return { used, max: MAX_SCENARIOS, left: Math.max(0, MAX_SCENARIOS - used), full: used >= MAX_SCENARIOS, words: `${used} of ${MAX_SCENARIOS} trips saved on this device, across every port.` };
}

/** Every saved trip on this device, newest first, each with its port, so none is counted without being shown. */
export function listTrips(data: TripData | null, o: { placeKey: string; portName: (key: string) => string | null; timeZone?: string }): TripRow[] {
  return (data?.scenarios ?? []).map((s) => {
    const f = tripFishery(s.species);
    const ws = fromDraft(s.draft, s.species || null);
    const r = evaluateWorkspace(ws, f);
    const rec = r.audience === "recreational";
    const main = rec ? r.out.operating : r.out.net;
    return {
      id: s.id,
      name: s.name,
      placeKey: s.placeKey,
      port: s.placeKey ? o.portName(s.placeKey) : null,
      fishery: f?.label ?? "Trip",
      here: s.placeKey === o.placeKey,
      detail: tripDetail(ws, f),
      outcome: main.status === "value" ? `${rec ? "Trip cost" : "Net to boat"} ${money(main.value)}` : main.status === "none" ? main.reason : "Not complete",
      updatedAt: s.updatedAt,
      savedAt: savedAtWords(s.updatedAt, o.timeZone),
    };
  });
}

// ---------------------------------------------------------------- compare

export type CompareInput = { label: string; a: string; b: string };
export type CompareOutput = { id: ScalarOutputKey; label: string; a: string; b: string; /** b − a in dollars (or the break-even's unit); null when either side has no value */ delta: number | null; deltaWords: string | null };
export type Comparison = {
  a: { id: string; name: string };
  b: { id: string; name: string };
  /** a commercial trip against a recreational one shares no result to set side by side */
  comparable: boolean;
  /** only what differs, in page order */
  inputs: CompareInput[];
  outputs: CompareOutput[];
};

const CREW_WORDS: Record<string, string> = { afterExpenses: "share after costs", ofGross: "share of gross", dayRate: "day rate", none: "nobody else paid" };
const OUTPUT_ROWS: Array<{ id: ScalarOutputKey; label: string; kind: "usd" | "lb" | "usdPerLb" | "usdPerGal" }> = [
  { id: "gross", label: "Gross", kind: "usd" },
  { id: "fees", label: "Fees", kind: "usd" },
  { id: "fuelCost", label: "Fuel", kind: "usd" },
  { id: "tripCost", label: "Operating costs", kind: "usd" },
  { id: "crewPay", label: "Crew pay", kind: "usd" },
  { id: "netToBoat", label: "Net to boat", kind: "usd" },
  { id: "perPerson", label: "Each person", kind: "usd" },
  { id: "breakEvenCatch", label: "Break-even catch", kind: "lb" },
  { id: "breakEvenPrice", label: "Break-even price", kind: "usdPerLb" },
];

function inputWords(field: FieldKey, v: number | string | boolean | null): string {
  if (v === null || v === "") return "blank";
  if (field === "fuelShared" || /\.shared$/.test(field)) return v ? "shared, off the top" : "the boat";
  if (field === "crew.kind") return CREW_WORDS[String(v)] ?? String(v);
  if (typeof v === "number") {
    const t = String(Number(v.toPrecision(10)));
    return /usdPerLb$|UsdPerLb$|UsdPerGal$|usdPerDay$|\.usd$/.test(field) ? `$${t}` : /pct|Pct/.test(field) ? `${t} %` : t;
  }
  return String(v);
}

/** Two saved trips side by side: only the inputs that differ, and each result with b − a. */
export function compareTrips(a: ScenarioRecord, b: ScenarioRecord, portName: (key: string) => string | null = () => null): Comparison {
  const fa = tripFishery(a.species);
  const fb = tripFishery(b.species);
  const audience = audienceOf(fa);
  const comparable = audience === audienceOf(fb);
  const inputs: CompareInput[] = [];
  if (a.placeKey !== b.placeKey) inputs.push({ label: "Port", a: portName(a.placeKey) ?? "another port", b: portName(b.placeKey) ?? "another port" });
  if (a.species !== b.species) inputs.push({ label: "Fishery", a: fa?.label ?? "Trip", b: fb?.label ?? "Trip" });
  const c = compareScenarios(draftInputs(a.draft), draftInputs(b.draft), { audience });
  // who pays a cost only moves the money when one of the trips shares what is left after costs
  const payersCount = [a, b].some((r) => r.draft.inputs.crew?.kind === "afterExpenses");
  for (const d of c.inputs) {
    if (!payersCount && (d.field === "fuelShared" || /\.shared$/.test(d.field))) continue;
    if (/\.species$/.test(d.field) && a.species !== b.species) continue;
    const label = fieldLabel(d.field);
    inputs.push({ label: label.charAt(0).toUpperCase() + label.slice(1), a: inputWords(d.field, d.a), b: inputWords(d.field, d.b) });
  }
  if ((a.draft.view?.shareBasis ?? "total") !== (b.draft.view?.shareBasis ?? "total")) inputs.push({ label: "Crew share entered as", a: a.draft.view?.shareBasis === "each" ? "each person" : "the crew's total", b: b.draft.view?.shareBasis === "each" ? "each person" : "the crew's total" });
  const say = (kind: (typeof OUTPUT_ROWS)[number]["kind"], v: number | null) => (v === null ? "no value" : kind === "usd" ? money(v) : kind === "lb" ? breakEvenCatchText(v, "lb") : kind === "usdPerLb" ? breakEvenPriceText(v, "lb") : `$${v.toFixed(2)}/gal`);
  const outputs: CompareOutput[] = comparable
    ? OUTPUT_ROWS.flatMap((row) => {
        const o = c.outputs.find((x) => x.output === row.id);
        if (!o) return [];
        const delta = o.delta;
        const deltaWords = delta === null ? null : row.kind === "usd" ? `${delta > 0.005 ? "+" : ""}${money(delta)}` : row.kind === "lb" ? `${delta > 0 ? "+" : delta < 0 ? "−" : ""}${Math.abs(Math.round(delta)).toLocaleString("en-US")} lb` : `${delta > 0 ? "+" : delta < 0 ? "−" : ""}$${Math.abs(delta).toFixed(2)}/lb`;
        return [{ id: row.id, label: row.label, a: say(row.kind, o.a), b: say(row.kind, o.b), delta, deltaWords }];
      })
    : [];
  return { a: { id: a.id, name: a.name }, b: { id: b.id, name: b.name }, comparable, inputs, outputs };
}

// ---------------------------------------------------------------- taking it elsewhere

/**
 * The reader's own backup file: their saved trips and vessels exactly as this device holds them (names and
 * port keys included, because it is theirs to move to another device). Nothing else on the device is in it:
 * no home port, followed fisheries, alerts or account.
 */
export const exportTrips = (store: TripStore): string => store.exportJson();

/**
 * A link that carries the numbers and nothing about the reader: no port, no trip name, no cost or species
 * names (labels come back from the fishery), no vessel. The numbers ride in the fragment, which browsers do
 * not send to any server.
 */
export function shareLink(origin: string, ws: Workspace, fishery: CoastFishery | null): string {
  const q = new URLSearchParams({ view: "trip", ...(fishery ? { fishery: fishery.id } : {}) });
  return `${origin.replace(/\/$/, "")}/coast?${q}#${SHARE_PARAM}=${encodeShare(toDraft(ws, fishery))}`;
}

/** The workspace a shared link carries, for the fishery in its query; null when the fragment is not one. */
export function readShare(hash: string, fisheryId: string | null): Workspace | null {
  const m = new RegExp(`(?:^|[#&])${SHARE_PARAM}=([A-Za-z0-9_-]+)`).exec(hash);
  const d = m ? decodeShare(m[1]) : null;
  if (!d) return null;
  const f = tripFishery(fisheryId);
  const ws = fromDraft(d, f?.id ?? null);
  // a link carries no names: the lines take the fishery's own
  const blank = emptyWorkspace(f);
  return { ...ws, catch: ws.catch.map((c, i) => (i === 0 ? { ...c, label: blank.catch[0].label, ref: blank.catch[0].ref } : c)), costs: ws.costs.map((c, i) => (i === 0 ? { ...c, label: blank.costs[0].label } : c)), taken: {} };
}
