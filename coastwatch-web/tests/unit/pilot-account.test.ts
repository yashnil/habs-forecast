import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { TRIGGERS } from "@/pilot/alerts/diff";
import { realPortToday } from "./fixtures/portToday";
import { readInbox, runLocalCheck, SNAP_KEY, markRead, unreadCount } from "@/pilot/alerts/inbox";
import { pilotAccountConfig } from "@/pilot/account/config";
import { SMS_CONSENT_TEXT, toE164US } from "@/pilot/account/consent";
import { ACCOUNT_KEY, EMPTY_ACCOUNT, readAccount, removePlace, savePlace, setChannel, writeAccount } from "@/pilot/account/local";
import { feedbackFor, saveFeedback } from "@/pilot/account/records";
import { __resetMemory, persistent, readJSON, writeJSON } from "@/pilot/account/storage";
import { readiness } from "@/pilot/readiness";

/** a minimal window with a working (or throwing) localStorage */
function fakeWindow(mode: "ok" | "throws" | "full") {
  const m = new Map<string, string>();
  const storage = {
    getItem: (k: string) => {
      if (mode === "throws") throw new Error("SecurityError");
      return m.get(k) ?? null;
    },
    setItem: (k: string, v: string) => {
      if (mode !== "ok") throw new Error(mode === "full" ? "QuotaExceededError" : "SecurityError");
      m.set(k, v);
    },
    removeItem: (k: string) => void m.delete(k),
  };
  (globalThis as unknown as { window: unknown }).window = {
    get localStorage() {
      if (mode === "throws") throw new Error("SecurityError");
      return storage;
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

describe("storage fallback", () => {
  it("works with storage, and says so", () => {
    const m = fakeWindow("ok");
    expect(persistent()).toBe(true);
    expect(writeJSON("nrt-pilot-x", { a: 1 })).toBe(true);
    expect(m.get("nrt-pilot-x")).toBe('{"a":1}');
    expect(readJSON("nrt-pilot-x", null, (v): v is { a: number } => !!v)).toEqual({ a: 1 });
  });

  for (const mode of ["throws", "full"] as const) {
    it(`storage that ${mode === "throws" ? "is blocked" : "is full"}: keeps the value for this page view and reports not persistent`, () => {
      fakeWindow(mode);
      expect(persistent()).toBe(false);
      expect(writeJSON("nrt-pilot-x", [1, 2])).toBe(false);
      expect(readJSON("nrt-pilot-x", [], (v): v is number[] => Array.isArray(v))).toEqual([1, 2]);
    });
  }

  it("no window at all (server render): defaults, never throws", () => {
    expect(persistent()).toBe(false);
    expect(readAccount()).toEqual(EMPTY_ACCOUNT);
  });

  it("corrupt or foreign values are ignored, unknown fisheries and triggers dropped", () => {
    const m = fakeWindow("ok");
    m.set(ACCOUNT_KEY, "{not json");
    expect(readAccount()).toEqual(EMPTY_ACCOUNT);
    m.set(ACCOUNT_KEY, JSON.stringify({ ...EMPTY_ACCOUNT, fisheries: ["dungeness_commercial", "whales"], saved_places: ["port-592", "<script>"], alerts: [{ place_id: "port-592", triggers: ["status_change", "bogus"], channels: ["in_app", "fax"] }] }));
    const a = readAccount();
    expect(a.fisheries).toEqual(["dungeness_commercial"]);
    expect(a.saved_places).toEqual(["port-592"]);
    expect(a.alerts).toEqual([{ place_id: "port-592", triggers: ["status_change"], channels: ["in_app"] }]);
  });
});

describe("device account", () => {
  it("saving a port makes it home (if none) and turns on in-app alerts for official changes", () => {
    const a = savePlace(EMPTY_ACCOUNT, "port-592");
    expect(a.home_place_id).toBe("port-592");
    expect(a.alerts).toEqual([{ place_id: "port-592", triggers: ["status_change", "new_toxin_result", "ramp_action"], channels: ["in_app"] }]);
    const b = removePlace(savePlace(a, "port-550"), "port-592");
    expect(b.saved_places).toEqual(["port-550"]);
    expect(b.home_place_id).toBe("port-550");
    expect(b.alerts.map((p) => p.place_id)).toEqual(["port-550"]);
  });

  it("feedback: one answer per fact, the newest wins", () => {
    fakeWindow("ok");
    saveFeedback({ fact_id: "f1", place_id: "port-592", useful: true, note: null, at: "2026-11-08T00:00:00Z" });
    saveFeedback({ fact_id: "f1", place_id: "port-592", useful: false, note: "too late", at: "2026-11-09T00:00:00Z" });
    expect(feedbackFor("f1")).toMatchObject({ useful: false, note: "too late" });
  });
});

describe("device alert check (in-app inbox without an account)", () => {
  let today = realPortToday("port-592");
  beforeEach(() => {
    fakeWindow("ok");
    today = realPortToday("port-592");
  });

  it("baseline first, then one inbox item per change, never twice", () => {
    writeAccount(savePlace(EMPTY_ACCOUNT, "port-592"));
    const first = runLocalCheck(readAccount(), [today], new Date("2026-11-08T15:00:00Z"));
    expect(first).toMatchObject({ added: [], baselined: ["port-592"], persisted: true });
    const DUNGENESS = "test:bma-g-monterey-bay-dungeness-crab";
    const changed = { ...today, facts: today.facts.map((f) => (f.id === DUNGENESS ? { ...f, as_of: "2026-11-06", value: "9 ppm avg" } : f)) };
    const second = runLocalCheck(readAccount(), [changed], new Date("2026-11-09T15:00:00Z"));
    expect(second.added.map((a) => a.trigger)).toEqual(["new_toxin_result"]);
    expect(second.added[0].source?.name).toContain("CDPH");
    expect(second.added.every((a) => !a.sample)).toBe(true);

    expect(runLocalCheck(readAccount(), [changed], new Date("2026-11-09T16:00:00Z")).added).toEqual([]);
    expect(readInbox()).toHaveLength(1);
    expect(unreadCount()).toBe(1);
    markRead("all");
    expect(unreadCount()).toBe(0);
    // facts read from the test fixture are marked on the item itself
    const again = { ...changed, facts: changed.facts.map((f) => (f.id === DUNGENESS ? { ...f, as_of: "2026-11-07" } : f)) };
    const sample = runLocalCheck(readAccount(), [again], new Date("2026-11-10T15:00:00Z"), { sample: true });
    expect(sample.added.map((a) => a.sample)).toEqual([true]);
  });

  it("in-app channel off, or the trigger not chosen: the baseline still moves, no inbox item", () => {
    writeAccount(setChannel(savePlace(EMPTY_ACCOUNT, "port-592"), "port-592", "in_app", false));
    runLocalCheck(readAccount(), [today]);
    const changed = { ...today, facts: today.facts.map((f) => (f.kind === "whale" ? { ...f, headline: "RAMP: depth restriction" } : f)) };
    expect(runLocalCheck(readAccount(), [changed]).added).toEqual([]);
    expect(readJSON(SNAP_KEY, {}, (v): v is Record<string, unknown> => !!v)).toHaveProperty("port-592");
  });

  it("blocked storage: the check still runs and says nothing was kept", () => {
    __resetMemory();
    fakeWindow("throws");
    const r = runLocalCheck(savePlace(EMPTY_ACCOUNT, "port-592"), [today]);
    expect(r.persisted).toBe(false);
  });
});

describe("readiness and the config sent to the browser", () => {
  it("no env: everything not configured, nothing secret or public leaks", () => {
    const c = pilotAccountConfig({});
    expect(c).toEqual({ supabase: null, vapidPublicKey: null, readiness: { accounts: "not_configured", alerts: "not_configured", web_push: "not_configured", sms: "not_configured" } });
  });

  it("local stack = demo, cloud = live; the service role key never appears in the config", () => {
    const local = pilotAccountConfig({ NEXT_PUBLIC_SUPABASE_URL: "http://127.0.0.1:56321", NEXT_PUBLIC_SUPABASE_ANON_KEY: "anon", SUPABASE_SERVICE_ROLE_KEY: "service-secret" });
    expect(local.readiness.accounts).toBe("demo");
    expect(JSON.stringify(local)).not.toContain("service-secret");
    const cloud = pilotAccountConfig({ NEXT_PUBLIC_SUPABASE_URL: "https://abc.supabase.co", NEXT_PUBLIC_SUPABASE_ANON_KEY: "anon", NEXT_PUBLIC_VAPID_PUBLIC_KEY: "pub", VAPID_PRIVATE_KEY: "priv-secret" });
    expect(cloud.readiness).toMatchObject({ accounts: "live", alerts: "live", web_push: "live" });
    expect(cloud.vapidPublicKey).toBe("pub");
    expect(JSON.stringify(cloud)).not.toContain("priv-secret");
  });

  it("web push without accounts stays not configured (there is nowhere to deliver from)", () => {
    const c = pilotAccountConfig({ NEXT_PUBLIC_VAPID_PUBLIC_KEY: "pub", VAPID_PRIVATE_KEY: "priv" });
    expect(readiness("web_push", { NEXT_PUBLIC_VAPID_PUBLIC_KEY: "pub", VAPID_PRIVATE_KEY: "priv" })).toBe("live");
    expect(c.readiness.web_push).toBe("not_configured");
    expect(c.vapidPublicKey).toBeNull();
  });
});

describe("consent", () => {
  it("the SMS disclosure carries the carrier-required elements", () => {
    for (const s of ["Neritic Alerts", "Message frequency varies", "Msg & data rates may apply", "Reply STOP to cancel", "HELP for help", "Consent is not a condition", "SMS Terms", "Privacy Policy"]) expect(SMS_CONSENT_TEXT).toContain(s);
  });

  it("US numbers normalise to E.164; anything else is refused", () => {
    expect(toE164US("(831) 555-0100")).toBe("+18315550100");
    expect(toE164US("+1 831.555.0100")).toBe("+18315550100");
    expect(toE164US("555-0100")).toBeNull();
    expect(toE164US("+44 20 7946 0958")).toBeNull();
    expect(toE164US("(131) 555-0100")).toBeNull();
  });

  it("trigger list matches the database's allowed values", () => {
    expect([...TRIGGERS]).toEqual(["status_change", "new_toxin_result", "ramp_action", "bloom_risk_jump", "data_stale"]);
  });
});
