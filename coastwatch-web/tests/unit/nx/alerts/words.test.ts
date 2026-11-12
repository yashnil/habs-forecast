import { afterEach, describe, expect, it } from "vitest";
import { CHANNELS, DELIVERY, NOT_AVAILABLE, TRIGGER_WORDS, itemChange, itemTitle, lastCheckedWords, plainDates, reportWords } from "@/nx/alerts/words";
import { FOLLOW_HINT_KEY, dismissFollowHint, dismissedPorts } from "@/nx/alerts/followHint";
import { TRIGGERS } from "@/pilot/alerts/diff";
import { isRegistered } from "@/nx/personal/keys";
import { packWords, type FieldRecord } from "@/nx/offline/pack";
import { resetMemory } from "@/nx/personal/store";
import { device, uninstall } from "../personal/device";

afterEach(() => {
  uninstall();
  resetMemory();
});

const name = (p: string) => ({ "port-473": "Bodega Bay", "port-592": "Moss Landing" })[p] ?? p;
const stamp = (o: object = {}) => ({ at: "2026-10-13T16:00:00.000Z", ports: ["port-473"], failed: [], data_at: null, ...o });

describe("the words alerts are said with", () => {
  it("says when ports are checked and that nothing is sent, in the agreed words", () => {
    expect(DELIVERY).toBe("Checked on this device when you open Neritic. Nothing is checked while it's closed. No texts, emails or phone notifications.");
  });

  it("only the inbox is a way an alert arrives; push, text and email are named as not available", () => {
    expect(CHANNELS.filter((c) => c.available).map((c) => c.id)).toEqual(["in_app"]);
    expect(CHANNELS.filter((c) => !c.available).map((c) => [c.id, c.state])).toEqual([
      ["push", NOT_AVAILABLE],
      ["sms", NOT_AVAILABLE],
      ["email", NOT_AVAILABLE],
    ]);
    expect(NOT_AVAILABLE).toBe("Not available in this preview");
  });

  it("names every trigger, and a model as a model, never as a toxin result or a verdict", () => {
    for (const t of TRIGGERS) expect(TRIGGER_WORDS[t].label.length).toBeGreaterThan(8);
    expect(`${TRIGGER_WORDS.bloom_risk_jump.label} ${TRIGGER_WORDS.bloom_risk_jump.hint}`).toMatch(/model/i);
    expect(TRIGGER_WORDS.bloom_risk_jump.hint).toContain("not a toxin test");
    for (const t of TRIGGERS) expect(`${TRIGGER_WORDS[t].label} ${TRIGGER_WORDS[t].hint}`).not.toMatch(/\bsafe\b|bloom risk|toxicity/i);
  });

  it("an alert's title drops the port the page already names, and says what kind of change it is", () => {
    expect(itemTitle({ title: "Bodega Bay: Dungeness crab (commercial): opens Jan 1.", place_label: "Bodega Bay" })).toBe("Dungeness crab (commercial): opens Jan 1.");
    expect(itemTitle({ title: "Bodega Bay: no longer listed — CDPH: Annual quarantine.", place_label: "Bodega Bay" })).toBe("CDPH: Annual quarantine.");
    expect(itemChange({ change: "changed", was: "Delayed to Jan 1.", trigger: "status_change" })).toBe("Before: Delayed to Jan 1.");
    expect(itemChange({ change: "new", was: null, trigger: "status_change" })).toBe("Newly listed for this port.");
    expect(itemChange({ change: "new", was: null, trigger: "new_toxin_result" })).toBe("A new result.");
    expect(itemChange({ change: "gone", was: "x", trigger: "status_change" })).toContain("The reason is unknown");
    // an alert kept by an earlier build has neither field: nothing is made up for it
    expect(itemChange({ trigger: "status_change" })).toBe("");
  });

  it("puts the pilot rules' ISO days in the reader's terms", () => {
    expect(plainDates("Changed since 2026-10-13. Source: CDFW, 2026-10-12.", "2026-10-14T00:00:00Z")).toBe("Changed since Oct 13. Source: CDFW, Oct 12.");
    expect(plainDates("No dates here.")).toBe("No dates here.");
  });

  it("says what a check did: a baseline, new alerts, nothing new, a port that could not be read, offline", () => {
    expect(reportWords({ state: "nothing_followed" }, name)).toBe("Follow a port to have it checked.");
    expect(reportWords({ state: "offline", last: null }, name)).toContain("nothing was checked");
    expect(reportWords({ state: "throttled", last: stamp() }, name)).toContain("Checked a moment ago");
    expect(reportWords({ state: "checked", stamp: stamp(), added: [], baselined: ["port-473"], persisted: true }, name)).toBe("Checked: nothing has changed. Bodega Bay: noted as it is now. Changes from here on will be listed.");
    expect(reportWords({ state: "checked", stamp: stamp(), added: [], baselined: [], persisted: true }, name)).toBe("Checked: nothing has changed.");
    const one = { state: "checked" as const, stamp: stamp({ ports: ["port-473"], failed: ["port-592"] }), added: [{} as never], baselined: [], persisted: false };
    expect(reportWords(one, name)).toBe("1 new alert. Moss Landing could not be read just now and will be tried again. This browser is not keeping data, so this lasts only until you close the page.");
    // nothing read at all is never "nothing has changed"
    expect(reportWords({ state: "checked", stamp: stamp({ ports: [], failed: ["port-473"] }), added: [], baselined: [], persisted: true }, name)).toBe("Bodega Bay could not be read just now and will be tried again.");
  });

  it("'Last checked' names the time and what could not be read; a device that never checked says nothing", () => {
    const when = () => "Oct 13, 9:00 AM PT";
    expect(lastCheckedWords(null, when, name)).toBeNull();
    expect(lastCheckedWords(stamp(), when, name)).toBe("Last checked Oct 13, 9:00 AM PT.");
    expect(lastCheckedWords(stamp({ failed: ["port-592"] }), when, name)).toBe("Last checked Oct 13, 9:00 AM PT. Moss Landing could not be read then.");
  });
});

describe("the Follow line's dismissal", () => {
  it("is kept per port, on the device, and is registered", () => {
    const d = device();
    expect(dismissedPorts()).toEqual([]);
    dismissFollowHint("port-592");
    dismissFollowHint("port-473");
    dismissFollowHint("port-592");
    expect(dismissedPorts()).toEqual(["port-592", "port-473"]);
    expect(d.json(FOLLOW_HINT_KEY)).toEqual(["port-592", "port-473"]);
    expect(isRegistered(FOLLOW_HINT_KEY, "local")).toBe(true);
  });
  it("reads a damaged list as empty", () => {
    device({ local: { [FOLLOW_HINT_KEY]: '{"not":"a list"}' } });
    expect(dismissedPorts()).toEqual([]);
  });
});

describe("Alerts offline", () => {
  const rec: FieldRecord = { v: 1, port: { key: "port-592", name: "Moss Landing" }, savedAt: "2026-10-13T06:00:00Z", pages: [{ id: "today", label: "Today", path: "/today?place=port-592", ok: true }, { id: "alerts", label: "Alerts", path: "/alerts?place=port-592", ok: true }], frames: [], station: null, files: 2, bytes: 10, missing: [] };
  it("says the alerts on the device are listed and that a new check waits for a connection", () => {
    const w = packWords(rec, "alerts");
    expect(w.notSaved).toBe("The alerts already on this device are listed. A new check of the ports you follow waits for a connection.");
    expect(w.saved).toBe("Saved for Moss Landing: Today.");
  });
});
