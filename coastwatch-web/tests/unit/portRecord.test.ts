import { afterEach, describe, expect, it } from "vitest";
import { EMPTY_PORT, PORT_KEY, readPortRecord, resetPortMemory, updatePortRecord } from "@/lib/portRecord";
import { readHome, writeHome } from "@/v4/coast/home";
import { readHomePlace, readHomePort } from "@/v4/state/homePort";
import { readTodayHref, readTodayPlace, writeTodayPlace } from "@/pilot/today/navPlace";
import { readFisheries, writeFisheries } from "@/pilot/today/prefs";
import { ACCOUNT_KEY, EMPTY_ACCOUNT, forgetDevice, readAccount, updateAccount, writeAccount } from "@/pilot/account/local";
import { __resetMemory } from "@/pilot/account/storage";
import { pushAccount } from "@/pilot/account/sync";

type Mode = "ok" | "throws" | "full";

function device(seed: Record<string, string> = {}, mode: Mode = "ok") {
  const m = new Map(Object.entries(seed));
  const s = {
    getItem: (k: string) => {
      if (mode === "throws") throw new Error("SecurityError");
      return m.get(k) ?? null;
    },
    setItem: (k: string, v: string) => {
      if (mode !== "ok") throw new Error("QuotaExceededError");
      m.set(k, v);
    },
    removeItem: (k: string) => void m.delete(k),
  };
  (globalThis as unknown as { window: unknown }).window = {
    get localStorage() {
      if (mode === "throws") throw new Error("SecurityError");
      return s;
    },
    addEventListener() {},
    removeEventListener() {},
  };
  return m;
}

afterEach(() => {
  delete (globalThis as unknown as { window?: unknown }).window;
  __resetMemory();
});

const stored = (m: Map<string, string>) => JSON.parse(m.get(PORT_KEY)!);

describe("migration", () => {
  it("a device with nothing saved gains nothing (first-visit logic reads the keys)", () => {
    const m = device();
    expect(readPortRecord()).toEqual(EMPTY_PORT);
    expect(m.size).toBe(0);
  });

  it("gathers the three old keys and the account into one record", () => {
    const account = { v: 1, fisheries: ["groundfish"], home_place_id: "port-592", vessel_length_class: "25_40ft", saved_places: [] };
    const m = device({
      "cw-home-port": JSON.stringify({ place: "port-550", for: "commercial", species: "dungeness" }),
      "nx-today-place-v1": "port-592",
      "nx-today-fisheries-v1": "dungeness_commercial,rock_crab",
      [ACCOUNT_KEY]: JSON.stringify(account),
    });
    expect(readPortRecord()).toEqual({ v: 1, place: "port-550", for: "commercial", species: "dungeness", lastPort: "port-592", fisheries: ["dungeness_commercial", "rock_crab"], vessel: "25_40ft" });
    expect(stored(m).place).toBe("port-550");
    // the old keys remain for a rollback
    expect(m.get("cw-home-port")).toBeDefined();
  });

  it("is idempotent: a second read changes nothing, and a later old-key value is ignored", () => {
    const m = device({ "cw-home-port": "port-550" });
    const first = readPortRecord();
    const raw = m.get(PORT_KEY);
    m.set("cw-home-port", "port-201");
    expect(readPortRecord()).toEqual(first);
    expect(m.get(PORT_KEY)).toBe(raw);
  });

  it("a cleared home stays cleared: the old key cannot bring it back", () => {
    const m = device({ "cw-home-port": "port-550" });
    writeHome(null);
    expect(readHome()).toBeNull();
    expect(m.has("cw-home-port")).toBe(false);
    expect(stored(m).place).toBeNull();
  });

  it("keeps a point or a site as the home, and Today's port-only field separate", () => {
    device({ "cw-home-port": JSON.stringify({ place: "point-36.8,-121.9", for: "all", species: "any" }), "nx-today-place-v1": "port-592" });
    expect(readHomePlace()).toBe("point-36.8,-121.9");
    expect(readHomePort()).toBeNull();
    expect(readTodayPlace()).toBe("port-592");
    expect(readTodayHref()).toBe("/today?place=port-592");
  });

  it.each([
    ["cw-home-port", "{not json"],
    ["cw-home-port", '{"for":"commercial"}'],
    ["cw-home-port", '{"place":42}'],
    ["cw-home-port", "<script>"],
    ["cw-home-port", "null"],
    ["nx-today-place-v1", "site-12"],
    ["nx-today-place-v1", "port-99999"],
    ["nx-today-fisheries-v1", "NOT VALID,,"],
    [ACCOUNT_KEY, "{broken"],
    [ACCOUNT_KEY, "[1,2,3]"],
    [ACCOUNT_KEY, '{"fisheries":"groundfish","home_place_id":7,"vessel_length_class":{}}'],
  ])("%s = %s is ignored and leaves no record", (k, v) => {
    const m = device({ [k]: v });
    expect(readPortRecord()).toEqual(EMPTY_PORT);
    expect(m.has(PORT_KEY)).toBe(false);
  });

  it("a partial old value keeps what is good", () => {
    device({ "nx-today-fisheries-v1": "rock_crab, NOT VALID ,rock_crab", "cw-home-port": '{"place":"port-201","for":99}' });
    const r = readPortRecord();
    expect(r.fisheries).toEqual(["rock_crab"]);
    expect(r).toMatchObject({ place: "port-201", for: "all", species: "any" });
  });

  it("falls back to the account's home and list when My Coast and Today have none", () => {
    device({ [ACCOUNT_KEY]: JSON.stringify({ v: 1, saved_places: [], fisheries: ["groundfish", 3], home_place_id: "port-592", vessel_length_class: null }) });
    expect(readPortRecord()).toMatchObject({ place: "port-592", fisheries: ["groundfish"], vessel: null });
  });

  it("a damaged record is rebuilt from the old keys; a partial one keeps its good fields", () => {
    device({ [PORT_KEY]: "{oops", "cw-home-port": "port-550" });
    expect(readPortRecord().place).toBe("port-550");
    device({ [PORT_KEY]: '{"v":1,"place":"port-201","fisheries":"x","lastPort":"site-1"}' });
    expect(readPortRecord()).toEqual({ ...EMPTY_PORT, place: "port-201" });
  });

  it("a record from a newer build is neither read nor replaced", () => {
    const raw = '{"v":2,"place":"port-201","future":true}';
    const m = device({ [PORT_KEY]: raw, "cw-home-port": "port-550" });
    expect(readPortRecord()).toEqual(EMPTY_PORT);
    expect(updatePortRecord((r) => ({ ...r, lastPort: "port-1" }))).toBe(false);
    expect(m.get(PORT_KEY)).toBe(raw);
  });

  it("survives no window and blocked storage", () => {
    expect(readPortRecord()).toEqual(EMPTY_PORT);
    expect(readHome()).toBeNull();
    expect(updatePortRecord((r) => r)).toBe(false);
    device({}, "throws");
    expect(readPortRecord()).toEqual(EMPTY_PORT);
    expect(readFisheries()).toBeNull();
  });
});

describe("the existing readers, through the record", () => {
  it("home keeps its trip and an unknown audience reads as all", () => {
    const m = device();
    writeHome({ place: "site-12", for: "shellfish", species: "mussels" });
    expect(readHome()).toEqual({ place: "site-12", for: "shellfish", species: "mussels" });
    expect(stored(m)).toMatchObject({ place: "site-12", for: "shellfish", species: "mussels" });
    expect(m.has("cw-home-port")).toBe(false);
    updatePortRecord((r) => ({ ...r, for: "tourists" }));
    expect(readHome()!.for).toBe("all");
  });

  it("Today takes ports only, and a list of fisheries the reader can use", () => {
    const m = device();
    expect(writeTodayPlace("site-12")).toBe(false);
    expect(m.has(PORT_KEY)).toBe(false);
    expect(writeTodayPlace("port-592")).toBe(true);
    expect(writeFisheries(["rock_crab", "groundfish"])).toBe(true);
    expect(readFisheries()).toEqual(["rock_crab", "groundfish"]);
    expect(stored(m)).toMatchObject({ lastPort: "port-592", fisheries: ["rock_crab", "groundfish"] });
    updatePortRecord((r) => ({ ...r, fisheries: ["not_a_fishery"] }));
    expect(readFisheries()).toBeNull();
  });

  it("a write the device refuses lasts for this page view", () => {
    device({}, "full");
    expect(writeTodayPlace("port-592")).toBe(false);
    expect(readTodayPlace()).toBe("port-592");
    resetPortMemory();
    expect(readTodayPlace()).toBeNull();
  });
});

describe("the account reads and writes the same record", () => {
  const account = { ...EMPTY_ACCOUNT, age_13_plus: { at: "2026-11-07T00:00:00Z", text: "13+" } };

  it("sees Today's list and My Coast's port home, and writes back to them", () => {
    const m = device({ "nx-today-fisheries-v1": "rock_crab", "cw-home-port": "port-550" });
    expect(readAccount()).toMatchObject({ fisheries: ["rock_crab"], home_place_id: "port-550" });
    updateAccount((a) => ({ ...a, fisheries: ["groundfish"], vessel_length_class: "40_60ft" }));
    expect(readFisheries()).toEqual(["groundfish"]);
    expect(stored(m)).toMatchObject({ fisheries: ["groundfish"], vessel: "40_60ft" });
  });

  it("changing the port resets the trip kept for the old one", () => {
    device({ "cw-home-port": JSON.stringify({ place: "port-550", for: "commercial", species: "dungeness" }) });
    updateAccount((a) => ({ ...a, home_place_id: "port-592" }));
    expect(readHome()).toEqual({ place: "port-592", for: "all", species: "any" });
  });

  it("a point or site saved as home is not the account's: shown as none, never replaced, never sent", async () => {
    const m = device({ "cw-home-port": JSON.stringify({ place: "point-36.8,-121.9", for: "all", species: "any" }) });
    expect(readAccount().home_place_id).toBeNull();
    updateAccount((a) => ({ ...a, home_place_id: "port-592" }));
    expect(stored(m).place).toBe("point-36.8,-121.9");

    writeAccount({ ...account, home_place_id: null });
    const sent: Record<string, unknown>[] = [];
    const chain = { delete: () => chain, update: () => chain, eq: () => chain, not: () => Promise.resolve({ error: null }), upsert: (row: Record<string, unknown>) => (sent.push(row), Promise.resolve({ error: null })) };
    const db = { auth: { getUser: async () => ({ data: { user: { id: "u1" } } }) }, from: () => ({ ...chain, update: () => ({ eq: () => Promise.resolve({ error: null }) }) }) };
    const res = await pushAccount(db as never, readAccount());
    expect(res).toEqual({ ok: true });
    expect(sent[0]).toMatchObject({ user_id: "u1", home_place_id: null });
    expect(JSON.stringify(sent)).not.toContain("36.8");
  });

  it("Delete everything on this device leaves no copy to migrate back", () => {
    const m = device({ "cw-home-port": "port-550", "nx-today-place-v1": "port-592", "nx-today-fisheries-v1": "rock_crab" });
    writeAccount({ ...readAccount(), vessel_length_class: "under_25ft" });
    forgetDevice();
    expect(m.size).toBe(0);
    expect(readAccount()).toEqual(EMPTY_ACCOUNT);
    expect(readHome()).toBeNull();
    expect(readTodayPlace()).toBeNull();
  });
});
