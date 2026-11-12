import { afterEach, describe, expect, it } from "vitest";
import type { PortFact } from "@/pilot/contracts";
import { sinceWords } from "@/nx/today/change";
import { VISIT_IDLE_MS, VISIT_PREFIX, browserStore, openVisit, type VisitStore } from "@/nx/today/useChangeLine";
import type { Seen } from "@/pilot/today/prefs";
import { isRegistered } from "@/nx/personal/keys";
import { realPortToday } from "../../fixtures/portToday";
import { device, uninstall } from "../personal/device";

/**
 * "Since your last visit" (src/nx/today/useChangeLine.ts): one comparison per visit. The facts are the real
 * ones for Moss Landing (fixture sidecars); a change is a changed fact, as it arrives.
 */
afterEach(uninstall);

const facts = realPortToday("port-592").facts;
const changed: PortFact[] = facts.map((f) => (f.id === "status:dungeness_commercial" ? { ...f, severity: "amber", headline: "Dungeness crab (commercial): Season opens Jan 1 with a 30-fathom depth limit.", value: "Restricted" } : f));

function memoryStore(): VisitStore & { seen: Map<string, Seen>; visits: Map<string, { prev: Seen | null; touched: number }> } {
  const seen = new Map<string, Seen>();
  const visits = new Map<string, { prev: Seen | null; touched: number }>();
  return { seen, visits, readVisit: (p) => visits.get(p) ?? null, writeVisit: (p, v) => void visits.set(p, v), readSeen: (p) => seen.get(p) ?? null, writeSeen: (p, s) => void seen.set(p, s) };
}

const DAY1 = "2026-10-12T15:00:00.000Z";
const DAY2 = "2026-10-13T15:00:00.000Z";
const t = (iso: string) => Date.parse(iso);

describe("which earlier visit Today compares with", () => {
  it("a port opened for the first time says nothing, however often the page mounts", () => {
    const s = memoryStore();
    expect(openVisit("port-473", facts, DAY1, t(DAY1), s)).toBeNull();
    // the page mounting again a moment later (a client navigation does this) must not compare with itself
    expect(openVisit("port-473", facts, DAY1, t(DAY1) + 400, s)).toBeNull();
    // nor a reload a minute later, nor a second look within the visit
    expect(openVisit("port-473", facts, DAY1, t(DAY1) + 60_000, s)).toBeNull();
    expect(s.seen.get("port-473")!.at).toBe(DAY1);
  });

  it("the next visit compares with the last one and says what changed", () => {
    const s = memoryStore();
    openVisit("port-592", facts, DAY1, t(DAY1), s);
    s.visits.clear(); // a new tab, or the browser restarted
    const line = openVisit("port-592", changed, DAY2, t(DAY2), s)!;
    expect(line.since).toBe(DAY1);
    expect(line.text).toBe("Since Oct 12: Dungeness crab (commercial) is now restricted.");
    expect(line.changes.map((c) => c.id)).toEqual(["status:dungeness_commercial"]);
  });

  it("a reload or a second mount in the same visit shows the same line, not 'no change'", () => {
    const s = memoryStore();
    openVisit("port-592", facts, DAY1, t(DAY1), s);
    s.visits.clear();
    const first = openVisit("port-592", changed, DAY2, t(DAY2), s)!;
    const again = openVisit("port-592", changed, DAY2, t(DAY2) + 5_000, s)!;
    const reload = openVisit("port-592", changed, DAY2, t(DAY2) + 10 * 60_000, s)!;
    expect(again.text).toBe(first.text);
    expect(reload.text).toBe(first.text);
    expect(reload.since).toBe(DAY1);
  });

  it("a tab left open for more than thirty minutes starts a new visit", () => {
    const s = memoryStore();
    openVisit("port-592", facts, DAY1, t(DAY1), s);
    const laterIso = new Date(t(DAY1) + VISIT_IDLE_MS + 60_000).toISOString();
    const line = openVisit("port-592", changed, laterIso, t(laterIso), s)!;
    expect(line.since).toBe(DAY1);
    expect(line.changes).toHaveLength(1);
    // just inside thirty minutes it is still the same visit: the first visit had no earlier one
    const s2 = memoryStore();
    openVisit("port-592", facts, DAY1, t(DAY1), s2);
    expect(openVisit("port-592", changed, DAY1, t(DAY1) + VISIT_IDLE_MS - 1, s2)).toBeNull();
  });

  it("nothing changed since a real earlier visit: it says so", () => {
    const s = memoryStore();
    openVisit("port-592", facts, DAY1, t(DAY1), s);
    s.visits.clear();
    expect(openVisit("port-592", facts, DAY2, t(DAY2), s)!.text).toBe("Since Oct 12: no change in season status, tests, notices or models.");
  });

  it("each port has its own visit: opening a second port never borrows the first one's", () => {
    const s = memoryStore();
    openVisit("port-592", facts, DAY1, t(DAY1), s);
    expect(openVisit("port-473", changed, DAY1, t(DAY1) + 1000, s)).toBeNull();
    expect([...s.seen.keys()].sort()).toEqual(["port-473", "port-592"]);
  });

  it("what the page shows now is always kept for next time, so the visit after compares with the latest", () => {
    const s = memoryStore();
    openVisit("port-592", facts, DAY1, t(DAY1), s);
    s.visits.clear();
    openVisit("port-592", changed, DAY2, t(DAY2), s);
    s.visits.clear();
    const day3 = "2026-10-14T15:00:00.000Z";
    expect(openVisit("port-592", changed, day3, t(day3), s)!.changes).toEqual([]);
  });

  it("a device clock that jumped backwards does not keep an old visit alive", () => {
    const s = memoryStore();
    openVisit("port-592", facts, DAY2, t(DAY2), s);
    // the tab's visit was stamped later than the device now says
    const line = openVisit("port-592", changed, DAY2, t(DAY2) - 3 * 3_600_000, s)!;
    expect(line.changes).toHaveLength(1);
  });
});

describe("never a 'since' in the future", () => {
  it("words a visit dated after the page's clock as 'your last visit'", () => {
    expect(sinceWords("2026-10-14T00:00:00Z", "2026-10-13T06:00:00Z")).toBe("your last visit");
    expect(sinceWords("not a date", "2026-10-13T06:00:00Z")).toBe("your last visit");
    expect(sinceWords("2026-10-08T15:00:00Z", "2026-10-13T06:00:00Z")).toBe("Oct 8");
  });

  it("the line then says so instead of a date", () => {
    const s = memoryStore();
    // the earlier visit was stamped by a page whose clock was ahead of this one's (a saved offline copy opened later)
    s.seen.set("port-592", { at: "2026-11-01T00:00:00.000Z", facts: [] });
    const line = openVisit("port-592", facts.filter((f) => f.kind === "status").slice(0, 1), DAY2, t(DAY2), s)!;
    expect(line.text.startsWith("Since your last visit: ")).toBe(true);
    expect(line.text).not.toMatch(/Nov/);
  });
});

describe("where the visit is kept in a browser", () => {
  it("the earlier visit lives in the tab's session, the last view on the device, both registered", () => {
    const d = device();
    openVisit("port-592", facts, DAY1, t(DAY1));
    expect([...d.session.keys()]).toEqual([`${VISIT_PREFIX}port-592`]);
    expect([...d.local.keys()]).toEqual(["nx-today-seen-v1:port-592"]);
    expect(isRegistered(`${VISIT_PREFIX}port-592`, "session")).toBe(true);
    expect(JSON.parse(d.session.get(`${VISIT_PREFIX}port-592`)!).prev).toBeNull();
    // a second mount reads it back
    expect(openVisit("port-592", changed, DAY1, t(DAY1) + 500)).toBeNull();
  });

  it("a tab whose session storage is blocked still compares once per page load", () => {
    const d = device({ local: { "nx-today-seen-v1:port-550": JSON.stringify({ at: DAY1, facts: [] }) } });
    Object.defineProperty((globalThis as unknown as { window: object }).window, "sessionStorage", {
      get() {
        throw new Error("SecurityError");
      },
    });
    const one = facts.filter((f) => f.kind === "whale");
    const first = openVisit("port-550", one, DAY2, t(DAY2));
    const again = openVisit("port-550", one, DAY2, t(DAY2) + 500);
    expect(first?.since).toBe(DAY1);
    expect(again?.text).toBe(first?.text);
    expect(d.session.size).toBe(0);
    expect(browserStore.readVisit("port-550")?.prev?.at).toBe(DAY1);
  });
});
