import { describe, expect, it } from "vitest";
import { MAX_NAME, MAX_SCENARIOS, NS, createTripStore, type KV } from "@/lib/tripEconomicsStore";
import { tripFishery } from "@/nx/coast/fisheries";
import {
  capacity,
  compareTrips,
  deleteTrip,
  duplicateTrip,
  exportTrips,
  failureWords,
  findIdentical,
  listTrips,
  loadWorking,
  readShare,
  renameTrip,
  sameInputs,
  saveNew,
  saveWorking,
  shareLink,
  tripDetail,
  tripName,
  tripState,
  uniqueName,
  updateTrip,
} from "@/nx/coast/scenarios";
import { emptyWorkspace, evaluateWorkspace, setCatch, toDraft, type Workspace } from "@/nx/coast/workspace";

function memory(): KV & { map: Map<string, string> } {
  const map = new Map<string, string>();
  return {
    map,
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => void map.set(k, String(v)),
    removeItem: (k) => void map.delete(k),
    key: (i) => [...map.keys()][i] ?? null,
    get length() {
      return map.size;
    },
  };
}

const crab = tripFishery("dungeness_commercial")!;
const sport = tripFishery("dungeness_recreational")!;
const MOSS = "port-592";
const NOW = new Date("2026-11-12T08:41:00Z");
const UTC = { now: NOW, timeZone: "UTC" };

/** The owner's test trip: 400 lb at $9, 30 gal at $8.24, bait and ice $100, 1 crew on 20 % after costs. */
const owner = (o: Partial<Workspace> = {}): Workspace => ({
  ...emptyWorkspace(crab),
  catch: [{ label: "Dungeness crab", ref: "dungeness", amount: "400", price: "9" }],
  fuelGal: "30",
  fuelPrice: "8.24",
  fuelPayer: "shared",
  costs: [{ label: "Bait and ice", usd: "100", payer: "shared" }],
  feesPct: "0",
  crew: "1",
  crewKind: "afterExpenses",
  share: "20",
  ...o,
});
const input = (ws: Workspace = owner(), placeKey = MOSS, fishery = crab) => ({ ws, fishery, placeKey });
const fresh = () => {
  let t = NOW.getTime();
  const kv = memory();
  return { kv, store: createTripStore(kv, () => new Date((t += 1000))) };
};
const ok = <T extends { ok: boolean }>(r: T): Extract<T, { ok: true }> => {
  if (!r.ok) throw new Error(`not ok: ${JSON.stringify(r)}`);
  return r as Extract<T, { ok: true }>;
};

describe("names", () => {
  it("say what the trip was and the minute it was saved", () => {
    expect(tripDetail(owner(), crab)).toBe("400 lb at $9, 1 crew 20 % after costs");
    expect(tripName(owner(), crab, UTC)).toBe("Dungeness crab, 400 lb at $9, 1 crew 20 % after costs, Nov 12, 8:41 AM");
    expect(tripName(owner(), crab, { now: NOW })).toContain("Nov 12, 12:41 AM"); // Pacific by default
    expect(tripDetail(owner({ crew: "2", share: "10", shareBasis: "each" }), crab)).toBe("400 lb at $9, 2 crew 10 % each after costs");
    expect(tripDetail(owner({ crew: "2", crewKind: "dayRate", dayRate: "200", days: "2" }), crab)).toBe("400 lb at $9, 2 crew at $200 a day");
    expect(tripDetail(owner({ crew: "0" }), crab)).toBe("400 lb at $9, no crew");
    expect(tripDetail({ ...emptyWorkspace(sport), fuelGal: "30", anglers: "2" }, sport)).toBe("30 gal, 2 people");
    expect(tripName(emptyWorkspace(null), null, UTC)).toBe("Trip, Nov 12, 8:41 AM");
  });
  it("are never one already in the list, and never longer than the store keeps", () => {
    const first = tripName(owner(), crab, UTC);
    const second = tripName(owner(), crab, { ...UTC, taken: [first] });
    expect(second).toBe(`${first} (2)`);
    expect(tripName(owner(), crab, { ...UTC, taken: [first, second] })).toBe(`${first} (3)`);
    const long = tripName(setCatch(owner(), 0, { amount: "123456.789", price: "1234.5678" }), tripFishery("groundfish"), { ...UTC, taken: [] });
    expect(long.length).toBeLessThanOrEqual(MAX_NAME);
    expect(long.endsWith("Nov 12, 8:41 AM")).toBe(true);
    expect(uniqueName("  ", [])).toBe("Trip");
  });
});

describe("Save as new, Update, and no accidental copies", () => {
  it("saving the same numbers twice does not add a second trip; it names the one already saved", () => {
    const { store } = fresh();
    const a = ok(saveNew(store, input(), UTC));
    const again = saveNew(store, input(), UTC);
    expect(again).toEqual({ ok: false, reason: "identical", id: a.id, name: a.name });
    expect(failureWords("identical", a.name)).toBe(`These exact numbers are already saved as “${a.name}”.`);
    expect(store.load().scenarios).toHaveLength(1);
    // the reader insists: a second trip, told apart by its name
    const forced = ok(saveNew(store, input(), { ...UTC, allowIdentical: true }));
    expect(forced.name).toBe(`${a.name} (2)`);
    expect(store.load().scenarios.map((s) => s.name)).toEqual([forced.name, a.name]);
  });
  it("the same numbers at another port or for another fishery are a different trip", () => {
    const { store } = fresh();
    ok(saveNew(store, input(), UTC));
    ok(saveNew(store, input(owner(), "port-600"), UTC));
    ok(saveNew(store, input(owner(), MOSS, tripFishery("rock_crab")!), UTC));
    expect(store.load().scenarios).toHaveLength(3);
    expect(new Set(store.load().scenarios.map((s) => s.name)).size).toBe(3);
  });
  it("Update changes the open trip in place: same id, nothing added (a typed name stays)", () => {
    const { store } = fresh();
    const a = ok(saveNew(store, input(), { ...UTC, name: "Opener" }));
    const edited = setCatch(owner(), 0, { amount: "450" });
    expect(updateTrip(store, a.id, input(edited))).toEqual({ ok: true, id: a.id, name: a.name });
    const d = store.load();
    expect(d.scenarios).toHaveLength(1);
    expect(d.scenarios[0].draft.inputs.catch![0].lb).toBe(450);
    expect(d.scenarios[0].updatedAt > d.scenarios[0].createdAt).toBe(true);
    expect(updateTrip(store, "s-gone", input())).toEqual({ ok: false, reason: "gone" });
  });
  it("a name Neritic wrote follows the numbers when the trip is updated", () => {
    const { store } = fresh();
    const a = ok(saveNew(store, input(), UTC));
    expect(a.name).toBe("Dungeness crab, 400 lb at $9, 1 crew 20 % after costs, Nov 12, 8:41 AM");
    const later = { now: new Date("2026-11-12T09:05:00Z"), timeZone: "UTC" };
    const up = ok(updateTrip(store, a.id, input(setCatch(owner(), 0, { price: "10" })), later));
    expect(up).toEqual({ ok: true, id: a.id, name: "Dungeness crab, 400 lb at $10, 1 crew 20 % after costs, Nov 12, 9:05 AM" });
    expect(store.load().scenarios.map((s) => [s.id, s.name, s.autoName])).toEqual([[a.id, up.name, true]]);
    // still unique against the other trips, and it keeps following
    const b = ok(saveNew(store, input(setCatch(owner(), 0, { price: "11" })), later));
    const same = ok(updateTrip(store, a.id, input(setCatch(owner(), 0, { price: "11" })), later));
    expect(same.name).toBe(`${b.name} (2)`);
  });
  it("a name the reader typed or edited never changes on its own", () => {
    const { store } = fresh();
    const later = { now: new Date("2026-11-12T09:05:00Z"), timeZone: "UTC" };
    // typed when saving
    const typed = ok(saveNew(store, input(), { ...UTC, name: "Opener" }));
    expect(ok(updateTrip(store, typed.id, input(setCatch(owner(), 0, { price: "10" })), later)).name).toBe("Opener");
    // written by Neritic, then edited by hand
    const auto = ok(saveNew(store, input(setCatch(owner(), 0, { amount: "500" })), UTC));
    ok(renameTrip(store, auto.id, "Second week"));
    expect(ok(updateTrip(store, auto.id, input(setCatch(owner(), 0, { amount: "600" })), later)).name).toBe("Second week");
    expect(store.load().scenarios.every((s) => s.autoName === undefined)).toBe(true);
    // a copy's name is its own; a trip saved before names were tracked is left as it is
    const again = ok(saveNew(store, input(setCatch(owner(), 0, { amount: "700" })), UTC));
    const copy = ok(duplicateTrip(store, again.id));
    expect(ok(updateTrip(store, copy.id, input(setCatch(owner(), 0, { amount: "800" })), later)).name).toBe(copy.name);
  });
  it("an empty worksheet is not saved, and a chosen name is kept (made unique)", () => {
    const { store } = fresh();
    expect(saveNew(store, input(emptyWorkspace(crab)), UTC)).toEqual({ ok: false, reason: "empty" });
    expect(ok(saveNew(store, input(), { ...UTC, name: "Opener" })).name).toBe("Opener");
    expect(ok(saveNew(store, input(setCatch(owner(), 0, { amount: "500" })), { ...UTC, name: "Opener" })).name).toBe("Opener (2)");
  });
  it("identical means the inputs, not where a reference came from", () => {
    const a = toDraft(owner(), crab);
    const b = toDraft(owner({ taken: { fuelPrice: true } }), crab);
    expect(sameInputs(a, b)).toBe(true);
    expect(sameInputs(a, toDraft(owner({ crew: "2", share: "10", shareBasis: "each" }), crab))).toBe(false);
    expect(sameInputs(a, toDraft(owner({ costs: [{ label: "Bait and ice", usd: "100", payer: "boat" }] }), crab))).toBe(false);
    expect(findIdentical([], MOSS, crab.id, a)).toBeNull();
  });
});

describe("the working draft survives leaving, reloading and restarting", () => {
  it("is autosaved with the trip it was opened from, and read back by a new store over the same storage", () => {
    const { kv, store } = fresh();
    const a = ok(saveNew(store, input(), UTC));
    const edited = setCatch(owner(), 0, { amount: "450" });
    expect(saveWorking(store, MOSS, crab, edited, a.id)).toBe(true);

    // a reload or a browser restart: nothing in memory, the same localStorage
    const later = createTripStore(kv, () => new Date(NOW.getTime() + 86_400_000));
    const w = loadWorking(later.load(), MOSS, crab);
    expect(w).toEqual({ ws: edited, openId: a.id, from: "working" });
    expect(tripState(later.load(), w.ws, crab, w.openId, "UTC")).toMatchObject({ kind: "changed", id: a.id, name: a.name, words: "Changes not saved" });
    // so Save means update, not a copy
    ok(updateTrip(later, w.openId!, input(w.ws)));
    expect(later.load().scenarios).toHaveLength(1);
    expect(tripState(later.load(), w.ws, crab, w.openId, "UTC").kind).toBe("saved");
  });
  it("is kept per port and fishery, and an untouched worksheet leaves nothing behind", () => {
    const { kv, store } = fresh();
    saveWorking(store, MOSS, crab, owner(), null);
    expect(loadWorking(store.load(), "port-600", crab).from).toBe("fresh");
    expect(loadWorking(store.load(), MOSS, tripFishery("rock_crab")!).from).toBe("fresh");
    expect(tripState(store.load(), owner(), crab, null)).toEqual({ kind: "unsaved", words: "Not saved" });
    saveWorking(store, MOSS, crab, emptyWorkspace(crab), null);
    expect(loadWorking(store.load(), MOSS, crab).from).toBe("fresh");
    expect(JSON.parse(kv.map.get(`${NS}meta`)!).drafts).toEqual({});
    expect(tripState(store.load(), emptyWorkspace(crab), crab, null).kind).toBe("empty");
  });
  it("a draft whose trip was deleted is unsaved work, not an update of nothing", () => {
    const { store } = fresh();
    const a = ok(saveNew(store, input(), UTC));
    saveWorking(store, MOSS, crab, owner(), a.id);
    ok(deleteTrip(store, a.id));
    expect(loadWorking(store.load(), MOSS, crab)).toMatchObject({ openId: null, from: "working" });
  });
});

describe("Duplicate, Rename, Delete and Undo", () => {
  it("Duplicate is a deliberate, named copy", () => {
    const { store } = fresh();
    const a = ok(saveNew(store, input(), { ...UTC, name: "Opener" }));
    expect(ok(duplicateTrip(store, a.id)).name).toBe("Opener (copy)");
    expect(ok(duplicateTrip(store, a.id)).name).toBe("Opener (copy 2)");
    expect(duplicateTrip(store, "s-gone")).toEqual({ ok: false, reason: "gone" });
  });
  it("Rename keeps names apart and never blanks one", () => {
    const { store } = fresh();
    const a = ok(saveNew(store, input(), { ...UTC, name: "Opener" }));
    const b = ok(saveNew(store, input(setCatch(owner(), 0, { amount: "500" })), { ...UTC, name: "Second week" }));
    expect(renameTrip(store, b.id, "Opener")).toEqual({ ok: true, id: b.id, name: "Opener (2)" });
    expect(renameTrip(store, a.id, "Opener")).toEqual({ ok: true, id: a.id, name: "Opener" });
    expect(renameTrip(store, a.id, "   ")).toEqual({ ok: false, reason: "empty" });
    expect(renameTrip(store, "s-gone", "x")).toEqual({ ok: false, reason: "gone" });
  });
  it("Delete hands back the trip, and Undo restores it exactly", () => {
    const { store } = fresh();
    const a = ok(saveNew(store, input(), UTC));
    const before = store.load().scenarios[0];
    const del = ok(deleteTrip(store, a.id));
    expect(store.load().scenarios).toHaveLength(0);
    expect(del.undo()).toBe(true);
    expect(store.load().scenarios).toEqual([before]);
    expect(del.undo()).toBe(true); // already back: nothing doubled
    expect(store.load().scenarios).toHaveLength(1);
    expect(deleteTrip(store, "s-gone")).toEqual({ ok: false, reason: "gone" });
  });
});

describe("the list is every port's, and the limit is said as it is", () => {
  it("shows every saved trip with its port and outcome, newest first", () => {
    const { store } = fresh();
    ok(saveNew(store, input(), UTC));
    ok(saveNew(store, input(setCatch(owner(), 0, { amount: "" }), "port-600"), UTC));
    ok(saveNew(store, input({ ...emptyWorkspace(sport), fuelGal: "30", fuelPrice: "8.24", costs: [{ label: "Bait and ice", usd: "0", payer: null }], anglers: "2" }, MOSS, sport), UTC));
    const rows = listTrips(store.load(), { placeKey: MOSS, portName: (k) => (k === MOSS ? "Moss Landing" : null), timeZone: "UTC" });
    expect(rows.map((r) => [r.port, r.fishery, r.here, r.outcome])).toEqual([
      ["Moss Landing", "Dungeness crab, recreational", true, "Trip cost $247.20"],
      [null, "Dungeness crab", false, "Not complete"],
      ["Moss Landing", "Dungeness crab", true, "Net to boat $2,602.24"],
    ]);
    expect(rows[2].detail).toBe("400 lb at $9, 1 crew 20 % after costs");
    expect(rows[2].savedAt).toMatch(/^Nov 12, 8:41 AM$/);
    expect(capacity(store.load())).toEqual({ used: 3, max: 20, left: 17, full: false, words: "3 of 20 trips saved on this device, across every port." });
    expect(capacity(null).used).toBe(0);
  });
  it("at the limit, Save as new says the true count and where the trips are", () => {
    const { store } = fresh();
    for (let i = 0; i < MAX_SCENARIOS; i++) ok(saveNew(store, input(setCatch(owner(), 0, { amount: String(100 + i) }), i % 2 ? MOSS : "port-600"), UTC));
    expect(capacity(store.load())).toMatchObject({ used: 20, left: 0, full: true });
    expect(saveNew(store, input(), UTC)).toEqual({ ok: false, reason: "full" });
    expect(failureWords("full")).toBe("20 trips are saved on this device, across every port, and that is the most it keeps. Delete one first.");
    // every one of them is in the list, whichever port it was saved at
    expect(listTrips(store.load(), { placeKey: MOSS, portName: () => null })).toHaveLength(20);
    // updating one of them still works at the limit
    const first = store.load().scenarios[0];
    expect(updateTrip(store, first.id, input()).ok).toBe(true);
  });
});

describe("compare", () => {
  it("lists only the inputs that differ, and each result with b − a", () => {
    const { store } = fresh();
    const a = ok(saveNew(store, input(), { ...UTC, name: "At $9" }));
    const b = ok(saveNew(store, input(setCatch(owner(), 0, { price: "10" })), { ...UTC, name: "At $10" }));
    const recs = store.load().scenarios;
    const c = compareTrips(recs.find((r) => r.id === a.id)!, recs.find((r) => r.id === b.id)!);
    expect(c.comparable).toBe(true);
    expect(c.inputs).toEqual([{ label: "Price per pound, line 1", a: "$9", b: "$10" }]);
    const net = c.outputs.find((o) => o.id === "netToBoat")!;
    // +400 gross, 80 % of it to the boat
    expect(net).toMatchObject({ a: "$2,602.24", b: "$2,922.24", deltaWords: "+$320.00" });
    expect(net.delta).toBeCloseTo(320, 6);
    expect(c.outputs.find((o) => o.id === "crewPay")!.deltaWords).toBe("+$80.00");
    expect(c.outputs.find((o) => o.id === "tripCost")!.deltaWords).toBe("$0.00");
    // 347.2/9 = 38.6 → 39 lb; 347.2/10 = 34.7 → 35 lb
    expect(c.outputs.find((o) => o.id === "breakEvenCatch")).toMatchObject({ a: "39 lb", b: "35 lb", deltaWords: "−4 lb" });
    expect(c.outputs.some((o) => o.id === "perPerson")).toBe(false);
  });
  it("says how crew is paid and who pays in words, and leaves payers out when they move nothing", () => {
    const { store } = fresh();
    ok(saveNew(store, input(), { ...UTC, name: "A" }));
    ok(saveNew(store, input(owner({ fuelPayer: "boat" })), { ...UTC, name: "B" }));
    ok(saveNew(store, input(owner({ crewKind: "ofGross", fuelPayer: "boat" })), { ...UTC, name: "C" }));
    ok(saveNew(store, input(owner({ crewKind: "ofGross" }), "port-600"), { ...UTC, name: "D" }));
    const by = (n: string) => store.load().scenarios.find((r) => r.name === n)!;
    expect(compareTrips(by("A"), by("B")).inputs).toEqual([{ label: "Who pays fuel", a: "shared, off the top", b: "the boat" }]);
    expect(compareTrips(by("A"), by("C")).inputs).toEqual([
      { label: "Who pays fuel", a: "shared, off the top", b: "the boat" },
      { label: "How crew is paid", a: "share after costs", b: "share of gross" },
    ]);
    // both share of gross: who is said to pay fuel changes no figure, so it is not listed as a difference
    expect(compareTrips(by("C"), by("D"), (k) => (k === MOSS ? "Moss Landing" : null)).inputs).toEqual([{ label: "Port", a: "Moss Landing", b: "another port" }]);
  });
  it("a commercial and a recreational trip share no result to compare", () => {
    const { store } = fresh();
    ok(saveNew(store, input(), { ...UTC, name: "A" }));
    ok(saveNew(store, input({ ...emptyWorkspace(sport), fuelGal: "30", fuelPrice: "8.24", anglers: "2" }, MOSS, sport), { ...UTC, name: "R" }));
    const by = (n: string) => store.load().scenarios.find((r) => r.name === n)!;
    const c = compareTrips(by("A"), by("R"));
    expect(c.comparable).toBe(false);
    expect(c.outputs).toEqual([]);
    expect(c.inputs[0]).toEqual({ label: "Fishery", a: "Dungeness crab", b: "Dungeness crab, recreational" });
  });
});

describe("taking it elsewhere without anything about the reader", () => {
  it("the link carries numbers only: no port, no trip name, no cost or species names", () => {
    const ws = owner({ costs: [{ label: "Bait from Phil's", usd: "100", payer: "shared" }], crew: "2", share: "10", shareBasis: "each" });
    const url = shareLink("https://neritic.example/", ws, crab);
    expect(url.startsWith("https://neritic.example/coast?view=trip&fishery=dungeness_commercial#trip=")).toBe(true);
    const payload = Buffer.from(url.split("#trip=")[1].replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8");
    expect(url).not.toContain(MOSS);
    for (const secret of [MOSS, "Phil", "Bait", "Dungeness crab", "Moss"]) expect(payload).not.toContain(secret);
    // and the numbers come back as the same trip
    const back = readShare(new URL(url).hash, crab.id)!;
    expect(back.costs[0].label).toBe("Bait and ice");
    expect(back.shareBasis).toBe("each");
    const net = evaluateWorkspace(back, crab).out.net;
    expect(net.status === "value" && net.value).toBeCloseTo(2602.24, 6);
    expect(readShare("#trip=not-a-trip", crab.id)).toBeNull();
    expect(readShare("", crab.id)).toBeNull();
  });
  it("the export is the reader's saved trips and vessels and nothing else on the device", () => {
    const { kv, store } = fresh();
    kv.setItem("nx-port-v1", JSON.stringify({ place: "port-592", home: "secret-home" }));
    ok(saveNew(store, input(), UTC));
    saveWorking(store, MOSS, crab, setCatch(owner(), 0, { amount: "999" }), null);
    const file = JSON.parse(exportTrips(store));
    expect(Object.keys(file).sort()).toEqual(["app", "exportedAt", "scenarios", "schema", "vessels"]);
    expect(file.scenarios).toHaveLength(1);
    expect(JSON.stringify(file)).not.toContain("secret-home");
    // unsaved work is not exported
    expect(JSON.stringify(file)).not.toContain("999");
  });
});
