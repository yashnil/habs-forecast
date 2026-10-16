import { describe, expect, it } from "vitest";
import type { LayerArtifact } from "@/generated/schema";
import { charmDays, defaultLead, roleWords, runAgeWords, runNotes, runWords } from "@/lib/charmDays";
import { relativeDay, relativeDayShort } from "@/lib/time";

/**
 * Scientific-communication regressions for the C-HARM day vocabulary (design audit P0: "forecast day
 * labels contradict each other"). A run issued Oct 14 holds a nowcast valid Oct 13 and forecasts for
 * Oct 14–16; the chips must name each day by its date and distance from today and nothing else.
 */
const lead = (valid: string) => ({ time: { valid_date: valid } }) as unknown as LayerArtifact;
const RUN = [lead("2026-10-13"), lead("2026-10-14"), lead("2026-10-15"), lead("2026-10-16")];
const NOW = new Date("2026-10-15T20:00:00Z"); // Oct 15, 1 pm Pacific

describe("C-HARM day chips", () => {
  it("name each day by its date and its distance from today, in one vocabulary", () => {
    const days = charmDays(RUN, NOW);
    expect(days.map((d) => d.short)).toEqual(["Oct 13", "Oct 14", "Oct 15", "Oct 16"]);
    expect(days.map((d) => d.relative)).toEqual(["2 days ago", "yesterday", "today", "tomorrow"]);
    // never "nowcast" beside "today": the role is a separate field, said in the run line
    for (const d of days) expect(d.relative).not.toMatch(/nowcast|forecast|\+\d/);
    expect(days[0].role).toBe("nowcast");
    expect(days.slice(1).every((d) => d.role === "forecast")).toBe(true);
  });

  it("every chip's relative word agrees with its own date, whatever the clock", () => {
    for (const iso of ["2026-10-13T08:00:00Z", "2026-10-15T06:59:00Z", "2026-10-15T07:01:00Z", "2026-10-19T20:00:00Z"]) {
      const now = new Date(iso);
      for (const d of charmDays(RUN, now)) expect(d.relative).toBe(relativeDay(d.date!, now));
    }
  });

  it("carries a compact form for a chip's width that names the same day", () => {
    const days = charmDays(RUN, NOW);
    expect(days.map((d) => d.relativeShort)).toEqual(["2 d ago", "yesterday", "today", "tomorrow"]);
    expect(relativeDayShort("2026-10-18", NOW)).toBe("in 3 d");
    expect(relativeDayShort("2026-10-11", NOW)).toBe("4 d ago");
    // the short and long forms always agree on the day
    for (const iso of ["2026-10-13T08:00:00Z", "2026-10-15T06:59:00Z", "2026-10-19T20:00:00Z"]) {
      const now = new Date(iso);
      for (const d of RUN) {
        const long = relativeDay(d.time.valid_date!, now);
        const short = relativeDayShort(d.time.valid_date!, now);
        expect(short.replace(/(\d+) d\b/, "$1 days")).toBe(long);
        expect(short.length).toBeLessThanOrEqual(9);
      }
    }
  });

  it("a day the run did not issue is marked, not invented", () => {
    const days = charmDays([RUN[0], RUN[1], null, RUN[3]], NOW);
    expect(days[2]).toMatchObject({ lead: 2, date: null, short: null, relative: null, relativeShort: null, issued: false });
  });

  it("opens on today when the run reaches it, never on yesterday's nowcast", () => {
    expect(defaultLead(charmDays(RUN, NOW), "2026-10-15")).toBe(2);
    // the morning after the run: today is the +1 day
    expect(defaultLead(charmDays(RUN, NOW), "2026-10-14")).toBe(1);
    // a stale run that stops before today: its last day (the note says it is old)
    expect(defaultLead(charmDays(RUN, NOW), "2026-10-19")).toBe(3);
    // today not issued (a run missing lead 2): the first day after today
    expect(defaultLead(charmDays([RUN[0], RUN[1], null, RUN[3]], NOW), "2026-10-15")).toBe(3);
    // a run from the future (clock skew): its first day
    expect(defaultLead(charmDays(RUN, NOW), "2026-10-10")).toBe(0);
    expect(defaultLead(charmDays([null, null, null, null], NOW), "2026-10-15")).toBe(0);
  });

  it("the run line explains why the nowcast's date precedes the run's", () => {
    expect(runWords({ issued_date: "2026-10-14", issued_date_derived: true }, charmDays(RUN, NOW))).toBe("Run of Oct 14 (date inferred): nowcast Oct 13, forecasts Oct 14–Oct 16");
    expect(runWords({ issued_date: "2026-10-14", issued_date_derived: false }, charmDays([RUN[0], null, null, null], NOW))).toBe("Run of Oct 14: nowcast Oct 13");
    expect(roleWords(0)).toBe("nowcast");
    expect(roleWords(2)).toBe("forecast +2 days");
  });
});

describe("the run's state, in one voice", () => {
  it("says nothing about a current run", () => {
    expect(runNotes({ state: "current", basisDate: "2026-10-14", ageDays: 1 }, { outcome: "updated", last_attempt_at: "2026-10-15T17:03:14Z" })).toEqual([]);
  });

  it("treats a run a few days old as ordinary latency (caution), named by its date, not as a failure", () => {
    const [n] = runNotes({ state: "stale", basisDate: "2026-10-12", ageDays: 3 }, { outcome: "updated", last_attempt_at: "2026-10-15T17:03:14Z" });
    expect(n.tone).toBe("caution");
    expect(n.testid).toBe("stale-note");
    expect(n.head).toBe("No newer run since Oct 12 (3 days ago)");
    expect(n.text).toMatch(/often missing a day or more/);
    expect(`${n.head} ${n.text}`).not.toMatch(/fail|error|could not|stale/i);
  });

  it("calls a run past the policy window out of date, and a failed update a failure, with the time tried", () => {
    const notes = runNotes({ state: "historical", basisDate: "2026-10-02", ageDays: 13 }, { outcome: "failed", last_attempt_at: "2026-10-15T17:03:14Z" });
    expect(notes.map((n) => n.testid)).toEqual(["historical-note", "update-failed"]);
    expect(notes.every((n) => n.tone === "problem")).toBe(true);
    expect(notes[0].text).toMatch(/does not describe current conditions/);
    expect(notes[1].text).toMatch(/could not reach NOAA at Oct 15, 10:03 AM PDT, so a newer run may exist/);
  });

  it("briefing words: 'run of Oct 12, 3 days old', and 'out-of-date' only past the window", () => {
    expect(runAgeWords({ state: "stale", basisDate: "2026-10-12", ageDays: 3 })).toBe("run of Oct 12, 3 days old");
    expect(runAgeWords({ state: "stale", basisDate: "2026-10-14", ageDays: 1 })).toBe("run of Oct 14, 1 day old");
    expect(runAgeWords({ state: "historical", basisDate: "2026-10-02", ageDays: 13 })).toBe("out-of-date run of Oct 2, 13 days old");
  });
});
