import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { loadToday } from "@/pilot/today/load";
import { CHECK_EVERY_MS, RETRY_AFTER_MS, checkPort, due, portToday, runCheck, type TodayPayload } from "@/nx/alerts/check";
import { DEFAULT_TRIGGERS, FOLLOW_KEY, MAX_FOLLOWS, follow, isFollowed, readFollows, setTrigger, unfollow } from "@/nx/alerts/follow";
import { CHECK_KEY, INBOX_KEY, SNAP_KEY, clearInbox, markRead, mergeInbox, readBaselines, readInbox, readStamp, unreadCount, type InboxItem } from "@/nx/alerts/inbox";
import { isRegistered } from "@/nx/personal/keys";
import { resetMemory } from "@/nx/personal/store";
import { fixtureServerEnv } from "../../fixtures/portToday";
import { device, uninstall } from "../personal/device";

/**
 * Alerts on this device (src/nx/alerts): the follow store, the inbox and the check, on the REAL records Today
 * reads for a port (src/pilot/today/load.ts on the committed fixture), for a pilot port (Moss Landing) and a
 * port outside the pilot (Bodega Bay). Changes are made the way they arrive: in the published records.
 */
const env = fixtureServerEnv();
const saved: Record<string, string | undefined> = {};
const real: Record<string, TodayPayload> = {};
const MOSS = "port-592";
const BODEGA = "port-473";

beforeAll(async () => {
  for (const [k, v] of Object.entries(env)) {
    saved[k] = process.env[k];
    process.env[k] = v;
  }
  for (const p of [MOSS, BODEGA]) {
    const l = await loadToday(p);
    if (!l.ok || !l.inputs) throw new Error(`fixture has no ${p}`);
    real[p] = { inputs: l.inputs, fixture: false };
  }
});
afterAll(() => {
  for (const [k, v] of Object.entries(saved)) {
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
});
afterEach(() => {
  uninstall();
  resetMemory();
});

const copy = (p: string): TodayPayload => JSON.parse(JSON.stringify(real[p])) as TodayPayload;
type Edit = (i: TodayPayload["inputs"]) => void;
const edited = (p: string, ...edits: Edit[]): TodayPayload => {
  const c = copy(p);
  for (const e of edits) e(c.inputs);
  return c;
};

// ---- changes as they arrive in the published records
const statusChanges: Edit = (i) => {
  const r = i.fisheries!.records.find((x) => x.fishery === "dungeness_commercial")!;
  r.level = "no_restriction_on_record";
  r.headline = "Commercial season opens Jan 1, 2027.";
};
const noticeAdded: Edit = (i) => {
  i.notices!.push({ ...i.notices![0], id: "cdfw-2026-new-closure", agency: "CDFW", title: "Razor clam fishery closed", summary: "Closed for domoic acid.", action: "fishery_closure" });
};
const noticeGone: Edit = (i) => {
  i.notices = i.notices!.filter((n) => n.id !== "cdph-2026-annual-mussel-quarantine");
};
const newTest: Edit = (i) => {
  const s = i.biotoxin!.sites.find((x) => x.id === "bma-g-monterey-bay-dungeness-crab")!;
  s.sets.unshift({ ...s.sets[0], date: "2026-10-10", values_ppm: [31, 12, 8, 6, 5, 4], avg_ppm: 11, max_ppm: 31 });
};
const published = (iso: string): Edit => (i) => {
  i.dataGeneratedAt = iso;
};

const T0 = new Date("2026-10-13T16:00:00Z");
const later = (ms: number) => new Date(T0.getTime() + ms);
const loader = (by: Record<string, TodayPayload | null>) => async (port: string) => by[port] ?? null;

describe("following ports", () => {
  it("any CDFW port can be followed, pilot or not, with official changes on by default", () => {
    const d = device();
    expect(follow(BODEGA, T0)).toEqual({ ok: true, persisted: true });
    expect(follow(MOSS, T0)).toEqual({ ok: true, persisted: true });
    expect(readFollows()).toEqual([
      { port: BODEGA, triggers: DEFAULT_TRIGGERS, since: T0.toISOString() },
      { port: MOSS, triggers: DEFAULT_TRIGGERS, since: T0.toISOString() },
    ]);
    expect(d.json(FOLLOW_KEY).v).toBe(1);
    expect(isFollowed(BODEGA)).toBe(true);
    expect(isFollowed("port-550")).toBe(false);
    expect(isFollowed(null)).toBe(false);
  });

  it("following twice changes nothing; a ninth port is refused; what is not a port is refused", () => {
    device();
    for (let i = 0; i < MAX_FOLLOWS; i++) expect(follow(`port-${100 + i}`).ok).toBe(true);
    expect(follow("port-100")).toEqual({ ok: true, persisted: true });
    expect(readFollows()).toHaveLength(MAX_FOLLOWS);
    expect(follow("port-999")).toEqual({ ok: false, why: "full" });
    expect(follow("site-12")).toEqual({ ok: false, why: "not_a_port" });
  });

  it("triggers are turned on and off per port, kept in one order", () => {
    device();
    follow(BODEGA);
    setTrigger(BODEGA, "data_stale", true);
    setTrigger(BODEGA, "status_change", false);
    expect(readFollows()[0].triggers).toEqual(["new_toxin_result", "ramp_action", "data_stale"]);
    expect(setTrigger(MOSS, "status_change", true)).toBe(false);
  });

  it("reads a damaged or tampered record tolerantly", () => {
    device({ local: { [FOLLOW_KEY]: JSON.stringify({ v: 1, ports: [{ port: BODEGA, triggers: ["status_change", "sms_me"], since: 5 }, { port: "site-1" }, null, { port: BODEGA }] }) } });
    expect(readFollows()).toEqual([{ port: BODEGA, triggers: ["status_change"], since: "" }]);
    device({ local: { [FOLLOW_KEY]: "{not json" } });
    expect(readFollows()).toEqual([]);
  });

  it("a browser that keeps nothing still follows for this page view, and says it was not kept", () => {
    const d = device();
    d.setMode("full");
    expect(follow(BODEGA)).toEqual({ ok: true, persisted: false });
    expect(isFollowed(BODEGA)).toBe(true);
    expect(d.local.size).toBe(0);
  });

  it("every key alerts keep is in the registry", () => {
    for (const k of [FOLLOW_KEY, INBOX_KEY, SNAP_KEY, CHECK_KEY]) expect(isRegistered(k, "local"), k).toBe(true);
  });
});

describe("comparing one port (checkPort)", () => {
  const f = { port: BODEGA, triggers: DEFAULT_TRIGGERS };
  const now = T0.toISOString();

  it("the first look keeps a baseline and alerts on nothing", () => {
    const r = checkPort(null, real[BODEGA], f, now);
    expect(r.state).toBe("baselined");
    expect(r.items).toEqual([]);
    expect(Object.keys(r.baseline.snap.facts)).toContain("status:dungeness_commercial");
    expect(r.baseline.data_at).toBe(real[BODEGA].inputs.dataGeneratedAt);
  });

  it("the same records again: nothing", () => {
    const base = checkPort(null, real[BODEGA], f, now).baseline;
    expect(checkPort(base, copy(BODEGA), f, now)).toMatchObject({ state: "compared", items: [] });
  });

  it("a changed season status at a port outside the pilot alerts, with its source and that it is not verified", () => {
    const base = checkPort(null, real[BODEGA], f, now).baseline;
    const r = checkPort(base, edited(BODEGA, statusChanges), f, now);
    expect(r.items).toHaveLength(1);
    expect(r.items[0]).toMatchObject({ place_id: BODEGA, place_label: "Bodega Bay", trigger: "status_change", fact_id: "status:dungeness_commercial", read_at: null, created_at: now });
    expect(r.items[0].title).toMatch(/^Bodega Bay: .*opens Jan 1, 2027/);
    expect(r.items[0].body).toContain("Was: “Dungeness crab (commercial): Commercial season delayed");
    expect(r.items[0].source?.name).toBeTruthy();
    expect(r.items[0].sample).toBeUndefined();
  });

  it("a new official notice and one no longer listed both alert", () => {
    const base = checkPort(null, real[BODEGA], f, now).baseline;
    const r = checkPort(base, edited(BODEGA, noticeAdded, noticeGone), f, now);
    // the quarantine notice also set the shellfish status, so its going changes that status too
    expect(r.items.map((i) => i.fact_id).sort()).toEqual(["notice:cdfw-2026-new-closure", "notice:cdph-2026-annual-mussel-quarantine", "status:bivalve_shellfish"]);
    expect(r.items.find((i) => i.fact_id === "notice:cdph-2026-annual-mussel-quarantine")!.title).toContain("no longer listed");
    expect(r.items.find((i) => i.fact_id === "notice:cdph-2026-annual-mussel-quarantine")!.body).toContain("reason is unknown");
  });

  it("a new CDPH result near the port alerts as a test, never as a status", () => {
    const g = { port: MOSS, triggers: DEFAULT_TRIGGERS };
    const base = checkPort(null, real[MOSS], g, now).baseline;
    const r = checkPort(base, edited(MOSS, newTest), g, now);
    expect(r.items).toHaveLength(1);
    expect(r.items[0]).toMatchObject({ trigger: "new_toxin_result", fact_id: "test:bma-g-monterey-bay-dungeness-crab", as_of: "2026-10-10", basis: "measured" });
  });

  it("only the triggers the reader chose alert; the baseline still moves on", () => {
    const base = checkPort(null, real[MOSS], { port: MOSS, triggers: DEFAULT_TRIGGERS }, now).baseline;
    const next = edited(MOSS, newTest, statusChanges);
    const r = checkPort(base, next, { port: MOSS, triggers: ["new_toxin_result"] }, now);
    expect(r.items.map((i) => i.trigger)).toEqual(["new_toxin_result"]);
    // the status change was seen and is not raised later when the trigger is turned back on
    expect(checkPort(r.baseline, next, { port: MOSS, triggers: DEFAULT_TRIGGERS }, now).items).toEqual([]);
  });

  it("the same change never alerts twice, however often it is compared", () => {
    const base = checkPort(null, real[BODEGA], f, now).baseline;
    const next = edited(BODEGA, statusChanges);
    const once = checkPort(base, next, f, now);
    expect(once.items).toHaveLength(1);
    expect(checkPort(once.baseline, next, f, now).items).toEqual([]);
    // and comparing again from the old baseline gives the same key, which the inbox merges
    expect(checkPort(base, next, f, later(1000).toISOString()).items[0].key).toBe(once.items[0].key);
  });

  it("notices that could not be read are not 'no longer listed', and are not 'new' when they come back", () => {
    const base = checkPort(null, real[MOSS], { port: MOSS, triggers: DEFAULT_TRIGGERS }, now).baseline;
    const r = checkPort(base, edited(MOSS, (i) => void (i.notices = null)), { port: MOSS, triggers: DEFAULT_TRIGGERS }, now);
    expect(r.items).toEqual([]);
    expect(Object.keys(r.baseline.snap.facts)).toContain("notice:cdph-2026-sn26-018-monterey-bivalves");
    // a status can be set by a notice, so status is not compared either while the notices are unreadable
    expect(portToday(edited(MOSS, (i) => void (i.notices = null)).inputs).today.facts.some((x) => x.kind === "status" || x.kind === "notice")).toBe(false);
    expect(checkPort(r.baseline, copy(MOSS), { port: MOSS, triggers: DEFAULT_TRIGGERS }, now).items).toEqual([]);
  });

  it("status and tests that could not be read are left out the same way", () => {
    const g = { port: MOSS, triggers: DEFAULT_TRIGGERS };
    const base = checkPort(null, real[MOSS], g, now).baseline;
    const r = checkPort(base, edited(MOSS, (i) => void ((i.fisheries = null), (i.biotoxin = null))), g, now);
    expect(r.items).toEqual([]);
    expect(portToday(edited(MOSS, (i) => void (i.fisheries = null)).inputs).today.facts.some((x) => x.kind === "status" || x.kind === "whale")).toBe(false);
    expect(checkPort(r.baseline, copy(MOSS), g, now).items).toEqual([]);
    // a change that happened while a part was unreadable is still caught once it reads again
    expect(checkPort(r.baseline, edited(MOSS, statusChanges), g, now).items.map((i) => i.fact_id)).toEqual(["status:dungeness_commercial"]);
  });

  it("whichever part fails to load, alone or together, nothing is raised and nothing is raised when it returns", () => {
    const all = [...DEFAULT_TRIGGERS, "bloom_risk_jump", "data_stale"] as typeof DEFAULT_TRIGGERS;
    const parts = ["fisheries", "notices", "biotoxin", "charm", "outlook"] as const;
    for (const port of [MOSS, BODEGA]) {
      const g = { port, triggers: all };
      const base = checkPort(null, real[port], g, now).baseline;
      for (let mask = 1; mask < 1 << parts.length; mask++) {
        const lost = parts.filter((_, i) => mask & (1 << i));
        const r = checkPort(base, edited(port, (i) => void lost.forEach((p) => ((i as Record<string, unknown>)[p] = null))), g, now);
        expect(r.items.map((x) => x.fact_id), `${port} without ${lost.join("+")}`).toEqual([]);
        expect(checkPort(r.baseline, copy(port), g, now).items.map((x) => x.fact_id), `${port} after ${lost.join("+")} returned`).toEqual([]);
      }
    }
  });

  it("an older copy of the records than the baseline (a saved offline copy) is not compared", () => {
    const base = checkPort(null, edited(BODEGA, published("2026-10-12T18:00:00Z")), f, now).baseline;
    const r = checkPort(base, edited(BODEGA, statusChanges, published("2026-10-08T18:00:00Z")), f, now);
    expect(r).toMatchObject({ state: "older_copy", items: [] });
    expect(r.baseline).toBe(base);
  });

  it("alerts made from the test fixture say so", () => {
    const base = checkPort(null, real[BODEGA], f, now).baseline;
    expect(checkPort(base, { ...edited(BODEGA, statusChanges), fixture: true }, f, now).items[0].sample).toBe(true);
  });

  it("an alert never says waters or seafood are safe", () => {
    const g = { port: MOSS, triggers: [...DEFAULT_TRIGGERS, "bloom_risk_jump", "data_stale"] as typeof DEFAULT_TRIGGERS };
    const base = checkPort(null, real[MOSS], g, now).baseline;
    for (const i of checkPort(base, edited(MOSS, statusChanges, newTest, noticeAdded, noticeGone), g, now).items) expect(`${i.title} ${i.body}`).not.toMatch(/\bsafe\b/i);
  });
});

describe("when a check is due", () => {
  const stamp = (at: Date, ports: string[], failed: string[] = []) => ({ at: at.toISOString(), ports, failed, data_at: null });
  it("never checked: due", () => expect(due(null, [BODEGA], T0)).toBe(true));
  it("checked under 30 minutes ago for the same ports: not due", () => expect(due(stamp(T0, [BODEGA, MOSS]), [MOSS, BODEGA], later(CHECK_EVERY_MS - 1))).toBe(false));
  it("30 minutes on: due", () => expect(due(stamp(T0, [BODEGA]), [BODEGA], later(CHECK_EVERY_MS))).toBe(true));
  it("a port followed since the last check: due now", () => expect(due(stamp(T0, [BODEGA]), [BODEGA, MOSS], later(1000))).toBe(true));
  it("a port that could not be read is tried again after five minutes, not on every page", () => {
    expect(due(stamp(T0, [BODEGA], [MOSS]), [BODEGA, MOSS], later(RETRY_AFTER_MS - 1))).toBe(false);
    expect(due(stamp(T0, [BODEGA], [MOSS]), [BODEGA, MOSS], later(RETRY_AFTER_MS))).toBe(true);
  });
  it("a device clock set back, or an unreadable time: due", () => {
    expect(due(stamp(later(60_000), [BODEGA]), [BODEGA], T0)).toBe(true);
    expect(due({ at: "nonsense", ports: [BODEGA], failed: [], data_at: null }, [BODEGA], T0)).toBe(true);
  });
});

describe("the check a page runs when it opens (runCheck)", () => {
  it("nothing followed: nothing fetched, nothing kept", async () => {
    const d = device({ local: { [SNAP_KEY]: JSON.stringify({ [BODEGA]: { snap: { place_id: BODEGA, seen_at: "x", facts: {} }, data_at: null } }) } });
    let calls = 0;
    const r = await runCheck({ now: T0, online: true, load: async () => (calls++, null) });
    expect(r).toEqual({ state: "nothing_followed" });
    expect(calls).toBe(0);
    // a baseline for a port no longer followed is not kept
    expect(d.local.has(SNAP_KEY)).toBe(false);
  });

  it("first open after following: a baseline, no alerts, and the time of the check", async () => {
    device();
    follow(BODEGA, T0);
    const r = await runCheck({ now: T0, online: true, load: loader({ [BODEGA]: real[BODEGA] }) });
    expect(r).toMatchObject({ state: "checked", added: [], baselined: [BODEGA], persisted: true });
    expect(readInbox()).toEqual([]);
    expect(readStamp()).toEqual({ at: T0.toISOString(), ports: [BODEGA], failed: [], data_at: real[BODEGA].inputs.dataGeneratedAt });
  });

  it("a change at a followed port outside the pilot arrives in the inbox on the next due open, once", async () => {
    device();
    follow(BODEGA, T0);
    await runCheck({ now: T0, online: true, load: loader({ [BODEGA]: real[BODEGA] }) });
    const changed = loader({ [BODEGA]: edited(BODEGA, statusChanges, noticeAdded) });

    // within 30 minutes nothing is fetched
    let calls = 0;
    const early = await runCheck({ now: later(10 * 60_000), online: true, load: async (p) => (calls++, changed(p)) });
    expect(early.state).toBe("throttled");
    expect(calls).toBe(0);

    const r = await runCheck({ now: later(CHECK_EVERY_MS), online: true, load: changed });
    expect(r.state).toBe("checked");
    expect(readInbox().map((i) => i.fact_id).sort()).toEqual(["notice:cdfw-2026-new-closure", "status:dungeness_commercial"]);
    expect(unreadCount()).toBe(2);

    // every later open with the same records adds nothing, forced or not
    await runCheck({ now: later(2 * CHECK_EVERY_MS), online: true, load: changed });
    await runCheck({ now: later(2 * CHECK_EVERY_MS + 1), online: true, force: true, load: changed });
    expect(readInbox()).toHaveLength(2);

    // clearing the inbox does not bring the same alerts back
    clearInbox();
    await runCheck({ now: later(3 * CHECK_EVERY_MS), online: true, load: changed });
    expect(readInbox()).toEqual([]);
  });

  it("offline: nothing is fetched and the last check's time stands", async () => {
    device();
    follow(BODEGA, T0);
    await runCheck({ now: T0, online: true, load: loader({ [BODEGA]: real[BODEGA] }) });
    let calls = 0;
    const r = await runCheck({ now: later(2 * CHECK_EVERY_MS), online: false, load: async () => (calls++, null) });
    expect(r).toMatchObject({ state: "offline", last: { at: T0.toISOString() } });
    expect(calls).toBe(0);
    expect(readStamp()!.at).toBe(T0.toISOString());
  });

  it("a port whose records cannot be read keeps its baseline, is reported, and is not an alert", async () => {
    device();
    follow(BODEGA, T0);
    follow(MOSS, T0);
    await runCheck({ now: T0, online: true, load: loader({ [BODEGA]: real[BODEGA], [MOSS]: real[MOSS] }) });
    const before = readBaselines()[MOSS];
    const r = await runCheck({
      now: later(CHECK_EVERY_MS),
      online: true,
      load: async (p) => {
        if (p === MOSS) throw new Error("network");
        return edited(BODEGA, statusChanges);
      },
    });
    expect(r).toMatchObject({ state: "checked", stamp: { ports: [BODEGA], failed: [MOSS] } });
    expect(readBaselines()[MOSS]).toEqual(before);
    expect(readInbox().map((i) => i.place_id)).toEqual([BODEGA]);
    // when it reads again, a change that happened meanwhile is caught
    await runCheck({ now: later(CHECK_EVERY_MS + RETRY_AFTER_MS), online: true, load: loader({ [BODEGA]: edited(BODEGA, statusChanges), [MOSS]: edited(MOSS, newTest) }) });
    expect(readInbox().map((i) => i.place_id).sort()).toEqual([BODEGA, MOSS]);
  });

  it("two calls at once run one check (two tabs' worth of opens do not double up)", async () => {
    device();
    follow(BODEGA, T0);
    let calls = 0;
    const load = async (p: string) => (calls++, real[p]);
    const [a, b] = await Promise.all([runCheck({ now: T0, online: true, load }), runCheck({ now: T0, online: true, load })]);
    expect(a).toBe(b);
    expect(calls).toBe(1);
  });

  it("unfollowing removes the port's baseline, its alerts and its place in the last check", async () => {
    const d = device();
    follow(BODEGA, T0);
    follow(MOSS, T0);
    await runCheck({ now: T0, online: true, load: loader({ [BODEGA]: real[BODEGA], [MOSS]: real[MOSS] }) });
    await runCheck({ now: later(CHECK_EVERY_MS), online: true, load: loader({ [BODEGA]: edited(BODEGA, statusChanges), [MOSS]: edited(MOSS, newTest) }) });
    expect(readInbox()).toHaveLength(2);
    unfollow(BODEGA);
    expect(readFollows().map((f) => f.port)).toEqual([MOSS]);
    expect(Object.keys(readBaselines())).toEqual([MOSS]);
    expect(readInbox().map((i) => i.place_id)).toEqual([MOSS]);
    expect(readStamp()!.ports).toEqual([MOSS]);
    unfollow(MOSS);
    expect(d.local.has(INBOX_KEY)).toBe(false);
    expect(d.local.has(SNAP_KEY)).toBe(false);
    // following again later starts from a fresh baseline: nothing old is raised
    follow(BODEGA, later(CHECK_EVERY_MS));
    const r = await runCheck({ now: later(CHECK_EVERY_MS + 1000), online: true, load: loader({ [BODEGA]: edited(BODEGA, statusChanges) }) });
    expect(r).toMatchObject({ state: "checked", added: [], baselined: [BODEGA] });
  });

  it("a browser that keeps nothing: the check runs, says it was not kept, and never throws", async () => {
    const d = device();
    follow(BODEGA, T0);
    d.setMode("full");
    const r = await runCheck({ now: T0, online: true, load: loader({ [BODEGA]: real[BODEGA] }) });
    expect(r).toMatchObject({ state: "checked", persisted: false });
  });
});

describe("the inbox", () => {
  const item = (key: string, at: string, read: string | null = null): InboxItem => ({ key, place_id: BODEGA, place_label: "Bodega Bay", fact_id: key, fact_version: "v", trigger: "status_change", severity: "info", title: key, body: "", basis: null, verified: null, source: null, as_of: null, created_at: at, read_at: read });

  it("merges newest first, never the same key twice, at most fifty", () => {
    const cur = [item("a", "2026-10-10T00:00:00Z", "2026-10-10T01:00:00Z")];
    const merged = mergeInbox(cur, [item("a", "2026-10-12T00:00:00Z"), item("b", "2026-10-11T00:00:00Z"), item("b", "2026-10-11T00:00:00Z")]);
    expect(merged.map((i) => i.key)).toEqual(["b", "a"]);
    expect(merged[1].read_at).toBe("2026-10-10T01:00:00Z");
    const many = Array.from({ length: 80 }, (_, i) => item(`k${i}`, `2026-10-${String(1 + (i % 28)).padStart(2, "0")}T00:00:00Z`));
    expect(mergeInbox([], many)).toHaveLength(50);
  });

  it("marks one or all read, keeps the first read time, and counts the unread", () => {
    const d = device({ local: { [INBOX_KEY]: JSON.stringify([item("a", "2026-10-12T00:00:00Z"), item("b", "2026-10-11T00:00:00Z")]) } });
    expect(unreadCount()).toBe(2);
    markRead(["a"], T0);
    expect(unreadCount()).toBe(1);
    markRead("all", later(1000));
    expect(d.json(INBOX_KEY).map((i: InboxItem) => i.read_at)).toEqual([T0.toISOString(), later(1000).toISOString()]);
    clearInbox();
    expect(d.local.has(INBOX_KEY)).toBe(false);
  });

  it("drops what is not an alert when reading a damaged inbox", () => {
    device({ local: { [INBOX_KEY]: JSON.stringify([null, { key: 1 }, item("a", "2026-10-12T00:00:00Z"), "x"]) } });
    expect(readInbox().map((i) => i.key)).toEqual(["a"]);
  });
});
