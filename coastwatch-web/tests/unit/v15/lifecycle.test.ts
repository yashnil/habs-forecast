import { describe, expect, it } from "vitest";
import { autoName, available, cancel, complete, daysUntil, departure, draftTrip, edit, limitValue, localDate, MAX_HISTORY, nextAction, plan, planNeeds, recordOutcome, reopen, type NextCtx } from "@/nx/trips/lifecycle";
import type { Change, Recheck, Trip } from "@/nx/trips/types";
import type { Draft } from "@/lib/tripEconomicsStore";

const NOW = new Date("2026-11-13T18:00:00Z"); // 10:00 Pacific, Nov 13
const words = { port: "Moss Landing", fishery: "Dungeness crab" };

const base = (o: Partial<Parameters<typeof draftTrip>[0]> = {}) => draftTrip({ id: "t1", activity: "commercial", fishery: "dungeness_commercial", placeKey: "port-592", words, ...o }, NOW);
const planned = (date = "2026-11-20", o: Partial<Parameters<typeof draftTrip>[0]> = {}) => {
  const r = plan(base({ depart: departure(date, null), limits: { windKt: 20, seasFt: 8 }, ...o }), NOW);
  if (!r.ok) throw new Error(r.words);
  return r.trip;
};
const ctx = (o: Partial<NextCtx> = {}): NextCtx => ({ changes: null, recheck: null, now: NOW, ...o });
const change = (priority: 1 | 2 | 3): Change => ({ id: `c${priority}`, priority, category: "forecast", kind: "revised", title: "Marine forecast updated since you planned.", detail: null, before: null, after: null, where: null, when: null, url: null, periodKey: null, evidenceKey: null });
const recheck = (at: string, complete = true): Recheck => ({ at, mode: "live", complete, changed: [], same: [], unchecked: [], stale: [], verify: [], baseId: "s1", snapshotId: "s2" });
const draftEcon = { preset: null, inputs: {}, pots: null, baitUsdPerPot: null, basis: {} } as unknown as Draft;

describe("creating a trip", () => {
  it("starts as a draft with an automatic name and no snapshot", () => {
    const t = base({ depart: departure("2026-11-14", "05:30") });
    expect(t.state).toBe("draft");
    expect(t.name).toBe("Dungeness crab from Moss Landing, Nov 14");
    expect(t.autoName).toBe(true);
    expect(t.depart).toEqual({ date: "2026-11-14", time: "05:30", timeZone: "America/Los_Angeles" });
    expect(t.snapshots).toEqual({ planning: null, rechecks: [], superseded: [] });
    expect(t.economics).toBeNull();
  });
  it("keeps the reader's own name", () => {
    const t = base({ name: "  Pots off the canyon " });
    expect(t.name).toBe("Pots off the canyon");
    expect(t.autoName).toBe(false);
  });
  it("blank, zero and nonsense limits are none set, never 0", () => {
    expect(limitValue("")).toBeNull();
    expect(limitValue("0")).toBeNull();
    expect(limitValue("abc")).toBeNull();
    expect(limitValue(-3)).toBeNull();
    expect(limitValue("18.25")).toBe(18.3);
    expect(base({ limits: { windKt: 0, seasFt: null } }).limits).toEqual({ windKt: null, seasFt: null });
  });
  it("a departure needs a calendar date; a bad time is 'some time that day'", () => {
    expect(departure("Nov 14", null)).toBeNull();
    expect(departure("2026-11-14", "5am")).toEqual({ date: "2026-11-14", time: null, timeZone: "America/Los_Angeles" });
  });
  it("area words are approximate free text, trimmed and bounded", () => {
    expect(base({ area: "   " }).area).toBeNull();
    expect(base({ area: "x".repeat(300) }).area).toHaveLength(120);
  });
});

describe("transitions and guards", () => {
  it("plan needs port, fishery and date", () => {
    const r = plan(base(), NOW);
    expect(r.ok).toBe(false);
    if (!r.ok) {
      expect(r.reason).toBe("missing");
      expect(r.words).toBe("Needs a date.");
    }
    expect(planNeeds({ placeKey: "", fishery: "" as never, depart: null })).toEqual(["port", "fishery", "date"]);
    expect(planned().state).toBe("planned");
  });
  it("a planned trip completes; a draft does not", () => {
    expect(complete(base(), NOW).ok).toBe(false);
    const r = complete(planned(), NOW);
    expect(r.ok && r.trip.state).toBe("completed");
  });
  it("cancel from draft or planned, never twice, never a completed trip", () => {
    expect(cancel(base(), NOW).ok).toBe(true);
    const c = cancel(planned(), NOW);
    expect(c.ok && c.trip.state).toBe("cancelled");
    if (c.ok) expect(cancel(c.trip, NOW).ok).toBe(false);
    const done = complete(planned(), NOW);
    if (done.ok) expect(cancel(done.trip, NOW).ok).toBe(false);
  });
  it("reopen: cancelled → planned when it still has a date, else → draft; cancel keeps snapshots", () => {
    const p = { ...planned(), snapshots: { planning: "s1", rechecks: ["s2"], superseded: [] } };
    const c = cancel(p, NOW);
    if (!c.ok) throw new Error();
    expect(c.trip.snapshots.planning).toBe("s1");
    const r = reopen(c.trip, NOW);
    expect(r.ok && r.trip.state).toBe("planned");
    const d = cancel(base(), NOW);
    if (!d.ok) throw new Error();
    const r2 = reopen(d.trip, NOW);
    expect(r2.ok && r2.trip.state).toBe("draft");
    expect(reopen(planned(), NOW).ok).toBe(false);
  });
  it("recording an outcome completes a planned trip and stamps when it changed", () => {
    const o = { departedOn: "2026-11-20", returnedOn: "2026-11-20", landed: [{ label: "Dungeness crab", lb: 400, usdPerLb: 9 }], fuelGal: 30, fuelUsdPerGal: 8.24, costs: [], crewPaidUsd: 650.56, notes: "", conditions: "", landingRef: null };
    const r = recordOutcome(planned(), o, NOW);
    if (!r.ok) throw new Error();
    expect(r.trip.state).toBe("completed");
    expect(r.trip.outcome?.recordedAt).toBe(NOW.toISOString());
    const later = new Date(NOW.getTime() + 3_600_000);
    const r2 = recordOutcome(r.trip, { ...o, notes: "rough" }, later);
    if (!r2.ok) throw new Error();
    expect(r2.trip.outcome?.recordedAt).toBe(NOW.toISOString());
    expect(r2.trip.outcome?.updatedAt).toBe(later.toISOString());
    const c = cancel(planned(), NOW);
    if (c.ok) expect(recordOutcome(c.trip, o, NOW).ok).toBe(false);
  });
  it("available lists the transitions on offer", () => {
    expect(available(base())).toEqual(["cancel"]);
    expect(available(base({ depart: departure("2026-11-20", null) }))).toEqual(["plan", "cancel"]);
    expect(available(planned())).toEqual(["complete", "cancel"]);
  });
});

describe("edit", () => {
  it("an automatic name follows the details; the reader's own does not", () => {
    const r = edit(planned(), { depart: departure("2026-11-21", null) }, NOW, words);
    expect(r.ok && r.trip.name).toBe("Dungeness crab from Moss Landing, Nov 21");
    const named = edit(planned(), { name: "Canyon run" }, NOW, words);
    if (!named.ok) throw new Error();
    const r2 = edit(named.trip, { depart: departure("2026-11-22", null) }, NOW, words);
    expect(r2.ok && r2.trip.name).toBe("Canyon run");
    const back = edit(named.trip, { name: "" }, NOW, words);
    expect(back.ok && back.trip.autoName).toBe(true);
  });
  it("a planned trip that loses its date is a draft again", () => {
    const r = edit(planned(), { depart: null }, NOW);
    expect(r.ok && r.trip.state).toBe("draft");
  });
  it("changed economics keep the earlier version, bounded", () => {
    let t: Trip = { ...planned(), economics: draftEcon };
    for (let i = 0; i < MAX_HISTORY + 3; i++) {
      const r = edit(t, { economics: { ...draftEcon, pots: i } }, NOW);
      if (!r.ok) throw new Error();
      t = r.trip;
    }
    expect(t.history).toHaveLength(MAX_HISTORY);
    expect(t.history[0].economics.pots).toBe(MAX_HISTORY + 1);
    // the same economics again is not a new version
    const same = edit(t, { economics: t.economics }, NOW);
    expect(same.ok && same.trip.history).toHaveLength(MAX_HISTORY);
  });
  it("a completed trip keeps its plan; its name and notes still change", () => {
    const d = complete(planned(), NOW);
    if (!d.ok) throw new Error();
    expect(edit(d.trip, { limits: { windKt: 30, seasFt: null } }, NOW).ok).toBe(false);
    expect(edit(d.trip, { notes: "slow day", name: "First of the season" }, NOW).ok).toBe(true);
  });
});

describe("days and dates", () => {
  it("counts calendar days in the trip's zone", () => {
    expect(localDate(new Date("2026-11-14T06:00:00Z"), "America/Los_Angeles")).toBe("2026-11-13");
    expect(daysUntil(planned("2026-11-13"), NOW)).toBe(0);
    expect(daysUntil(planned("2026-11-14"), NOW)).toBe(1);
    expect(daysUntil(planned("2026-11-10"), NOW)).toBe(-3);
    expect(daysUntil(base(), NOW)).toBeNull();
  });
  it("autoName without a port or date", () => {
    expect(autoName({ depart: null }, { port: null, fishery: null })).toBe("Trip");
  });
});

describe("nextAction: one primary action", () => {
  it("draft without a date: plan this trip by setting a date", () => {
    expect(nextAction(base(), ctx())?.id).toBe("set-date");
    expect(nextAction(base({ depart: departure("2026-11-20", null) }), ctx())?.id).toBe("plan");
  });
  it("cancelled: reopen; completed: record outcome, then edit it", () => {
    const c = cancel(planned(), NOW);
    if (c.ok) expect(nextAction(c.trip, ctx())?.id).toBe("reopen");
    const d = complete(planned(), NOW);
    if (!d.ok) throw new Error();
    expect(nextAction(d.trip, ctx())?.id).toBe("record-outcome");
    expect(nextAction({ ...d.trip, outcome: {} as never }, ctx())?.id).toBe("edit-outcome");
  });
  it("past departure asks for the outcome before anything else", () => {
    expect(nextAction(planned("2026-11-12"), ctx({ changes: [change(1)] }))?.id).toBe("record-outcome");
  });
  it("meaningful changes come first; minor ones alone do not take the action; reviewed ones step aside", () => {
    expect(nextAction(planned(), ctx({ changes: [change(1)] }))).toMatchObject({ id: "review-changes", why: "1 since you planned." });
    expect(nextAction(planned("2026-11-16"), ctx({ changes: [change(3)] }))?.id).toBe("compare-departures");
    expect(nextAction(planned("2026-11-16"), ctx({ changes: [change(2)], reviewed: true }))?.id).toBe("compare-departures");
  });
  it("today or tomorrow: recheck, again when stale or incomplete", () => {
    expect(nextAction(planned("2026-11-14"), ctx())).toMatchObject({ id: "recheck", why: "Departure is tomorrow." });
    expect(nextAction(planned("2026-11-13"), ctx())).toMatchObject({ id: "recheck", why: "Departure is today." });
    expect(nextAction(planned("2026-11-14"), ctx({ recheck: recheck("2026-11-13T17:00:00Z") }))?.id).toBe("compare-departures");
    expect(nextAction(planned("2026-11-14"), ctx({ recheck: recheck("2026-11-13T17:00:00Z", false) }))).toMatchObject({ id: "recheck", why: "The last recheck was incomplete." });
    expect(nextAction(planned("2026-11-14"), ctx({ recheck: recheck("2026-11-12T17:00:00Z") }))?.id).toBe("recheck");
  });
  it("a check that could not read NWS asks for a forecast recheck before any comparison", () => {
    const failed = { ...recheck("2026-11-13T17:00:00Z", false), unchecked: [{ source: "nwsForecast" as const, why: "did not answer", required: true }] };
    expect(nextAction(planned("2026-11-16"), ctx({ recheck: failed }))).toMatchObject({ id: "recheck", label: "Recheck the forecast" });
    const cdphOnly = { ...recheck("2026-11-13T17:00:00Z", false), unchecked: [{ source: "cdphTest" as const, why: "did not answer", required: true }] };
    expect(nextAction(planned("2026-11-16"), ctx({ recheck: cdphOnly }))?.id).toBe("compare-departures");
  });
  it("the known forecast decides: an older one or none asks for a recheck before comparing", () => {
    expect(nextAction(planned("2026-11-16"), ctx({ forecast: "older" }))).toMatchObject({ id: "recheck", why: "The latest check could not read NWS." });
    expect(nextAction(planned("2026-11-16"), ctx({ forecast: "none" }))).toMatchObject({ id: "recheck", why: "No forecast is saved for this trip." });
    expect(nextAction(planned("2026-11-16"), ctx({ forecast: "latest" }))?.id).toBe("compare-departures");
    expect(nextAction(planned("2026-11-30"), ctx({ forecast: "none" }))?.id).toBe("edit-assumptions");
  });
  it("no limits of their own: edit assumptions; in forecast reach: compare; further out: edit assumptions", () => {
    expect(nextAction(planned("2026-11-16", { limits: {} }), ctx())?.id).toBe("edit-assumptions");
    expect(nextAction(planned("2026-11-16"), ctx())?.id).toBe("compare-departures");
    expect(nextAction(planned("2026-11-30"), ctx())).toMatchObject({ id: "edit-assumptions", why: "The forecast does not reach Nov 30 yet." });
  });
  it("never words a verdict", () => {
    const all = [planned(), planned("2026-11-14"), planned("2026-11-12"), base()].map((t) => nextAction(t, ctx({ changes: [change(1)] })));
    for (const a of all) expect(`${a?.label} ${a?.why}`).not.toMatch(/\b(safe|unsafe|legal|approved|profit|all clear)\b/i);
  });
});
