import { afterEach, describe, expect, it } from "vitest";
import { ENTRY_SCRIPT } from "@/nx/shell/entry";
import { choosePort, homePortKey, savePort, savedPortKey, setHomePort } from "@/nx/shell/port";
import { PORT_KEY, resetPortMemory } from "@/lib/portRecord";
import { moveFishery, readPrefs, replayOpening, replayTour, setFisheries, setVessel } from "@/nx/personal/prefs";
import { resetMemory } from "@/nx/personal/store";
import { device, uninstall } from "./device";

afterEach(() => {
  uninstall();
  resetPortMemory();
  resetMemory();
});

const record = (o: object) => ({ [PORT_KEY]: JSON.stringify({ v: 1, place: null, for: "all", species: "any", lastPort: null, fisheries: [], vessel: null, ...o }) });

/** The pre-paint script's answer for `/` with this storage (the same stand-in as tests/unit/nx-chart.test.ts). */
function entry(storage: Record<string, string>) {
  const attrs: Record<string, string> = {};
  const store = { ...storage };
  const ls = { getItem: (k: string) => store[k] ?? null, setItem: (k: string, v: string) => (store[k] = v) };
  const win = { localStorage: ls, WebGL2RenderingContext: function () {} };
  const doc = { documentElement: { setAttribute: (k: string, v: string) => (attrs[k] = v), getAttribute: (k: string) => attrs[k] ?? null } };
  const noop = () => 0;
  new Function("document", "localStorage", "location", "matchMedia", "window", "self", "navigator", "addEventListener", "removeEventListener", "setTimeout", ENTRY_SCRIPT)(doc, ls, { pathname: "/", search: "" }, () => ({ matches: false }), win, win, { onLine: true }, noop, noop, noop);
  return attrs["data-nx-port"] ?? null;
}

describe("the home port is the reader's own choice, separate from the port last opened", () => {
  it("the first port a reader chooses becomes their home port", () => {
    const d = device({ search: "" });
    savePort("port-592"); // the front door's finder: the address names no port
    expect(d.json(PORT_KEY)).toMatchObject({ place: "port-592", lastPort: "port-592" });
    expect(homePortKey()).toBe("port-592");
  });

  it("choosing another port later is only the port last opened: `/` still opens home", () => {
    const d = device({ local: record({ place: "port-592", lastPort: "port-592" }), search: "?place=port-592" });
    savePort("port-473"); // the bar's finder, before it navigates
    expect(d.json(PORT_KEY)).toMatchObject({ place: "port-592", lastPort: "port-473" });
    expect(savedPortKey()).toBe("port-592");
    expect(entry(Object.fromEntries(d.local))).toBe("port-592");
  });

  it("a deep link never sets or moves the home port, even on a first visit", () => {
    const d = device({ search: "?place=port-550&utm_source=x" });
    savePort("port-550"); // the page reporting the port its address names
    expect(d.json(PORT_KEY)).toMatchObject({ place: null, lastPort: "port-550" });
    expect(homePortKey()).toBeNull();
    // with no home, `/` opens the port last opened
    expect(savedPortKey()).toBe("port-550");
    // Today's page says so outright
    const e = device({ local: record({ place: "port-592", lastPort: "port-592" }), search: "" });
    savePort("port-550", { chosen: false });
    expect(e.json(PORT_KEY)).toMatchObject({ place: "port-592", lastPort: "port-550" });
  });

  it("choosePort is the explicit form, whatever the address says", () => {
    const d = device({ search: "?place=port-550" });
    choosePort("port-550");
    expect(d.json(PORT_KEY)).toMatchObject({ place: "port-550", lastPort: "port-550" });
  });

  it("setHomePort sets, moves and clears it; nothing else does", () => {
    const d = device({ local: record({ lastPort: "port-473", fisheries: ["rock_crab"] }) });
    expect(setHomePort("port-592")).toBe(true);
    expect(d.json(PORT_KEY)).toMatchObject({ place: "port-592", lastPort: "port-473", fisheries: ["rock_crab"] });
    setHomePort("port-550");
    expect(homePortKey()).toBe("port-550");
    setHomePort(null);
    expect(homePortKey()).toBeNull();
    expect(savedPortKey()).toBe("port-473");
    expect(setHomePort("site-12")).toBe(false);
    expect(setHomePort("nonsense")).toBe(false);
  });

  it("an earlier My Coast home that is a harvest site is not a home port, is not cleared, and is not replaced by a first choice", () => {
    const d = device({ local: record({ place: "site-12", for: "recreational", species: "mussels", lastPort: "port-592" }) });
    expect(homePortKey()).toBeNull();
    expect(savedPortKey()).toBe("port-592");
    setHomePort(null);
    choosePort("port-473");
    expect(d.json(PORT_KEY)).toMatchObject({ place: "site-12", for: "recreational", species: "mussels", lastPort: "port-473" });
    // only the reader setting a home port replaces it
    setHomePort("port-473");
    expect(d.json(PORT_KEY)).toMatchObject({ place: "port-473", for: "all", species: "any" });
  });

  it("ignores what is not a port key", () => {
    const d = device();
    savePort("point-36.8,-121.9");
    savePort("port-99999");
    expect(d.local.size).toBe(0);
  });

  it("the pre-paint script opens the same port the app does: home first, else last opened, in new and old keys", () => {
    const devices: Record<string, string>[] = [
      record({ place: "port-592", lastPort: "port-473" }),
      record({ lastPort: "port-473" }),
      record({ place: "site-12", lastPort: "port-473" }),
      { "cw-home-port": JSON.stringify({ place: "port-593", for: "all" }), "nx-today-place-v1": "port-550" },
      { "nx-today-place-v1": "port-550" },
      { "cw-home-port": "site-12", "nx-today-place-v1": "port-550" },
      {},
    ];
    for (const s of devices) {
      device({ local: s });
      expect(entry(s), JSON.stringify(s)).toBe(savedPortKey());
      uninstall();
      resetPortMemory();
    }
    expect(entry(record({ place: "port-592", lastPort: "port-473" }))).toBe("port-592");
  });

  it("the record's shape is unchanged (an earlier build still reads it)", () => {
    const d = device();
    choosePort("port-592");
    expect(Object.keys(d.json(PORT_KEY)).sort()).toEqual(["fisheries", "for", "lastPort", "place", "species", "v", "vessel"]);
    expect(d.json(PORT_KEY).v).toBe(1);
  });
});

describe("preferences", () => {
  it("reads every preference from the keys the app already uses", () => {
    device({ local: { ...record({ place: "port-592", lastPort: "port-473", fisheries: ["rock_crab", "not_a_fishery", "groundfish"], vessel: "25_40ft" }), "nx-chart-mode-v1": "night", "nx-tour-v1": "dismissed", "nx-opening-seen-v1": "1" } });
    expect(readPrefs()).toEqual({ homePort: "port-592", lastPort: "port-473", fisheries: ["rock_crab", "groundfish"], vessel: "25_40ft", chartMode: "night", tourDismissed: true, openingSeen: true });
  });

  it("a first visit has none", () => {
    device();
    expect(readPrefs()).toEqual({ homePort: null, lastPort: null, fisheries: [], vessel: null, chartMode: null, tourDismissed: false, openingSeen: false });
  });

  it("orders fisheries (the first leads Today) and keeps ids it does not know after them", () => {
    const d = device({ local: record({ fisheries: ["rock_crab", "groundfish", "abalone_legacy"] }) });
    expect(moveFishery("groundfish", -1)).toBe(true);
    expect(d.json(PORT_KEY).fisheries).toEqual(["groundfish", "rock_crab", "abalone_legacy"]);
    expect(moveFishery("groundfish", -1)).toBe(false);
    setFisheries(["market_squid", "market_squid", "groundfish"]);
    expect(d.json(PORT_KEY).fisheries).toEqual(["market_squid", "groundfish", "abalone_legacy"]);
  });

  it("sets the vessel length, and plays the tour and the opening again", () => {
    const d = device({ local: { ...record({ lastPort: "port-592" }), "nx-tour-v1": "dismissed", "nx-opening-seen-v1": "1" } });
    setVessel("40_60ft");
    expect(d.json(PORT_KEY).vessel).toBe("40_60ft");
    replayTour();
    replayOpening();
    expect(d.local.has("nx-tour-v1")).toBe(false);
    expect(d.local.has("nx-opening-seen-v1")).toBe(false);
    expect(readPrefs()).toMatchObject({ tourDismissed: false, openingSeen: false });
  });
});
