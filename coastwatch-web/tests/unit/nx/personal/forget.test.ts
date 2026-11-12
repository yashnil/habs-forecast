import { afterEach, describe, expect, it } from "vitest";
import { KEYS } from "@/nx/personal/keys";
import { exportLocalData, forgetEverything, forgetStorage, inventory, thisDevice } from "@/nx/personal/forget";
import { forgetDevice } from "@/pilot/account/local";
import { __resetMemory } from "@/pilot/account/storage";
import { readPortRecord } from "@/lib/portRecord";
import { readFollows } from "@/nx/alerts/follow";
import { readInbox } from "@/nx/alerts/inbox";
import { resetMemory } from "@/nx/personal/store";
import { device, uninstall } from "./device";

afterEach(() => {
  uninstall();
  __resetMemory();
  resetMemory();
});

/** One value under every registered key (a prefix gets a port or an id after it), as a used device would hold. */
function everyKey() {
  const local: Record<string, string> = {};
  const session: Record<string, string> = {};
  const caches: Record<string, string[]> = {};
  for (const e of KEYS) {
    const name = e.match === "prefix" ? `${e.key}port-592` : e.key;
    if (e.store === "local") local[name] = JSON.stringify({ any: name });
    else if (e.store === "session") session[name] = "1";
    else caches[name] = ["/today?place=port-592", "/coast?place=port-592"];
  }
  return { local, session, caches };
}

describe("Delete everything on this device", () => {
  it("removes every registered key from storage and every saved-for-offline cache", async () => {
    const d = device(everyKey());
    expect(d.local.size).toBeGreaterThan(40);
    const gone = await forgetEverything();
    expect([...d.local.keys()]).toEqual([]);
    expect([...d.session.keys()]).toEqual([]);
    // only the app's own code is left: it says nothing about the reader
    expect([...d.caches.keys()]).toEqual(["neritic-static-port-592"]);
    expect(gone.caches.sort()).toEqual(["neritic-kept-v1", "neritic-pack-port-592"]);
    expect(gone.failed).toEqual([]);
  });

  it("removes both offline packs and what the audit found left behind: trips, visit history, Ask's snapshot, the pack record", async () => {
    const d = device({
      local: {
        "nx-port-v1": '{"v":1,"place":"port-592","for":"all","species":"any","lastPort":"port-473","fisheries":["dungeness_commercial"],"vessel":"25_40ft"}',
        "nx-trip-v1:scenarios": '[{"id":"a","name":"Crab, Moss Landing"}]',
        "nx-trip-v1:vessels": '[{"id":"v","name":"F/V Anna"}]',
        "nx-trip-v1:corrupt:scenarios": "{",
        "nx-coast-v1": '{"fishery":"dungeness_commercial"}',
        "nx-today-seen-v1:port-592": '{"at":"2026-10-12T00:00:00Z","facts":[]}',
        "nx-today-seen-v1:port-473": '{"at":"2026-10-12T00:00:00Z","facts":[]}',
        "nx-today-saved-v1": '[{"place":"port-592","label":"Moss Landing","at":"2026-10-12T00:00:00Z"}]',
        "nx-ask-seen-v1:port-592": "{}",
        "nx-field-v1": '{"v":1}',
        "nx-tour-v1": "dismissed",
        "nx-opening-seen-v1": "1",
        "nx-chart-mode-v1": "night",
        "nx-follow-v1": '{"v":1,"ports":[{"port":"port-473","triggers":["status_change"],"since":"2026-10-01T00:00:00Z"}]}',
        "nx-inbox-v1": '[{"key":"k","title":"t","place_id":"port-473","created_at":"2026-10-12T00:00:00Z","read_at":null}]',
        "nx-alert-snap-v1": "{}",
        "nrt-pilot-account-v1": '{"v":1,"saved_places":["port-592"],"sms":{"phone_e164":"+18315550100"}}',
      },
      session: { "nx-today-visit-v1:port-592": "{}" },
      caches: { "neritic-pack-field": ["/today?place=port-592"], "neritic-pack-today": ["/"], "neritic-static-abc": ["/_next/static/x.js"] },
    });
    await forgetEverything();
    expect([...d.local.keys()]).toEqual([]);
    expect([...d.session.keys()]).toEqual([]);
    expect([...d.caches.keys()]).toEqual(["neritic-static-abc"]);
    // and the app reads a first visit again
    expect(readPortRecord()).toMatchObject({ place: null, lastPort: null, fisheries: [] });
    expect(readFollows()).toEqual([]);
    expect(readInbox()).toEqual([]);
  });

  it("removes a key nobody registered too: deleting does not depend on the registry", async () => {
    const d = device({ local: { "nx-something-new-v3": "1", "an-unprefixed-key": "1" }, session: { other: "1" }, caches: { "another-cache": ["/x"] } });
    const gone = await forgetEverything();
    expect(d.local.size + d.session.size + d.caches.size).toBe(0);
    expect(gone.local.sort()).toEqual(["an-unprefixed-key", "nx-something-new-v3"]);
  });

  it("leaves a signed-in account's session: signing out ends it, not clearing the device", async () => {
    const d = device({ local: { "sb-abcd-auth-token": "{}", "nx-port-v1": "{}" } });
    await forgetEverything();
    expect([...d.local.keys()]).toEqual(["sb-abcd-auth-token"]);
  });

  it("the account page's button (forgetDevice) clears storage at once and the caches when its promise settles", async () => {
    const d = device({
      local: { "nx-trip-v1:scenarios": "[]", "nx-today-seen-v1:port-592": "{}", "nrt-pilot-account-v1": '{"v":1,"saved_places":[]}', "nx-port-v1": '{"v":1,"lastPort":"port-592"}' },
      caches: { "neritic-pack-field": ["/a"], "neritic-pack-today": ["/b"] },
    });
    const p = forgetDevice();
    expect([...d.local.keys()]).toEqual([]);
    const gone = await p;
    expect([...d.caches.keys()]).toEqual([]);
    expect(gone.caches.sort()).toEqual(["neritic-pack-field", "neritic-pack-today"]);
  });

  it("never throws: no storage at all, storage that refuses, twice in a row", async () => {
    await expect(forgetEverything({ local: null, session: null, caches: null })).resolves.toEqual({ local: [], session: [], caches: [], failed: [] });
    const d = device({ local: { "nx-port-v1": "{}" } });
    d.setMode("blocked");
    expect(thisDevice().local).toBeNull();
    await expect(forgetEverything()).resolves.toMatchObject({ failed: [] });
    d.setMode("ok");
    expect(forgetStorage()).toEqual(["nx-port-v1"]);
    expect(forgetStorage()).toEqual([]);
  });
});

describe("what is on this device, and a copy of it", () => {
  const seed = {
    local: {
      "nx-port-v1": '{"v":1,"place":"port-592","lastPort":"port-592","fisheries":["rock_crab"]}',
      "nx-trip-v1:scenarios": '[{"id":"a","name":"Crab"}]',
      "nx-today-seen-v1:port-592": '{"at":"2026-10-12T00:00:00Z","facts":[]}',
      "nx-chart-mode-v1": "dusk",
      "nx-gfx-slow-v1": "1",
      "sb-abcd-auth-token": '{"access_token":"secret"}',
      "nx-not-registered-v9": "x",
    },
    session: { "nx-nav-pill-v1": "{}" },
    caches: { "neritic-pack-field": ["/today?place=port-592", "/coast?place=port-592", "/chart/chart-day-ca.webp"], "neritic-static-abc": ["/x.js"] },
  };

  it("lists each thing with what it holds in words, most personal first, and never a sign-in session", async () => {
    device(seed);
    const inv = await inventory();
    expect(inv.map((i) => i.name)).not.toContain("sb-abcd-auth-token");
    expect(inv[0].kind).toBe("personal");
    expect(inv.at(-1)!.kind).toBe("transient");
    expect(inv.find((i) => i.name === "nx-trip-v1:scenarios")).toMatchObject({ kind: "personal", unit: "bytes", unregistered: false });
    expect(inv.find((i) => i.name === "neritic-pack-field")).toMatchObject({ store: "cache", kind: "offline", size: 3, unit: "files" });
    // an unregistered key is still shown (and still deleted), under a plain name
    expect(inv.find((i) => i.name === "nx-not-registered-v9")).toMatchObject({ unregistered: true, holds: "Other data Neritic kept" });
  });

  it("exports the reader's data as parsed values, without housekeeping, the tab's session or a sign-in token", async () => {
    device(seed);
    const x = await exportLocalData(thisDevice(), new Date("2026-10-13T06:00:00Z"));
    expect(x).toMatchObject({ app: "neritic-device", schema: 1, exportedAt: "2026-10-13T06:00:00.000Z" });
    const keys = x.local.map((l) => l.key);
    expect(keys).toEqual(expect.arrayContaining(["nx-port-v1", "nx-trip-v1:scenarios", "nx-today-seen-v1:port-592", "nx-chart-mode-v1", "nx-not-registered-v9"]));
    expect(keys).not.toContain("nx-gfx-slow-v1");
    expect(keys).not.toContain("nx-nav-pill-v1");
    expect(JSON.stringify(x)).not.toContain("secret");
    expect(x.local.find((l) => l.key === "nx-trip-v1:scenarios")!.value).toEqual([{ id: "a", name: "Crab" }]);
    // a value that is not JSON is kept as it is
    expect(x.local.find((l) => l.key === "nx-chart-mode-v1")!.value).toBe("dusk");
    expect(x.caches).toEqual([{ name: "neritic-pack-field", holds: expect.stringMatching(/saved for use without a connection/), files: 3 }]);
  });
});
