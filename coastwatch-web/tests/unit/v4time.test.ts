import { describe, expect, it } from "vitest";
import { dayLabel, pacificDay, relTime } from "@/v4/ui/time";
import { coverageWhen } from "@/v4/explore/layers";
import { hoursAgo } from "@/lib/currents";

describe("v4 relTime", () => {
  it("counts a calendar date in Pacific days, also in the evening when UTC is already tomorrow", () => {
    expect(relTime("2026-10-14", new Date("2026-10-16T17:00:00Z"))).toBe("2 days ago"); // 10 am PDT
    expect(relTime("2026-10-14", new Date("2026-10-17T06:30:00Z"))).toBe("2 days ago"); // 11:30 pm PDT
    expect(relTime("2026-10-14", new Date("2026-10-17T07:30:00Z"))).toBe("3 days ago"); // 12:30 am PDT
    expect(relTime("2026-10-16", new Date("2026-10-17T03:00:00Z"))).toBe("today");
    expect(relTime("2026-10-15", new Date("2026-10-17T03:00:00Z"))).toBe("yesterday");
  });
  it("keeps clock times relative to the instant", () => {
    const now = new Date("2026-10-17T00:00:00Z");
    expect(relTime("2026-10-16T23:30:00Z", now)).toBe("30 min ago");
    expect(relTime("2026-10-16T19:00:00Z", now)).toBe("5 h ago");
    expect(relTime("not a date", now)).toBe("");
    // one age function: whole hours elapsed, as the map's source stamp counts them (lib/currents hoursAgo)
    expect(relTime("2026-10-16T10:35:00Z", now)).toBe(`${hoursAgo("2026-10-16T10:35:00Z", now)} h ago`);
    expect(relTime("2026-10-16T10:35:00Z", now)).toBe("13 h ago");
  });
});

describe("v4 dates of instants", () => {
  it("dates a timestamp by its Pacific day, not its UTC day", () => {
    // HF radar hour 02:00 UTC Oct 17 = 7 PM PDT Oct 16 (the walkthrough's wrong timebar chip)
    expect(dayLabel("2026-10-17T02:00:00Z")).toBe("Oct 16");
    expect(dayLabel("2026-10-17T08:00:00Z")).toBe("Oct 17");
    expect(dayLabel("2026-10-16")).toBe("Oct 16");
    expect(pacificDay("2026-10-17T02:00:00Z")).toBe("2026-10-16");
  });
  it("gives one observation one age, whether given as a date or a time", () => {
    const now = new Date("2026-10-17T09:00:00Z"); // 2 am PDT Oct 17
    expect(relTime("2026-10-14", now)).toBe("3 days ago");
    expect(relTime("2026-10-14T18:44:40Z", now)).toBe("3 days ago"); // was "3"/"2" depending on hours
    expect(relTime("2026-10-14T23:59:00Z", now)).toBe(relTime("2026-10-14", now));
  });
});

describe("coverage dates", () => {
  const t = (o: Partial<{ observed_date: string; observed_times: string[] }>) => ({ observed_date: null, observed_times: [], ...o }) as never;
  it("names a multi-sensor view's span, not one overpass", () => {
    const l = { composite: null, multisensor: {} as never, time: t({ observed_date: "2026-10-15", observed_times: ["2026-10-10T17:57:56Z", "2026-10-13T17:28:21Z", "2026-10-15T19:08:58Z"] }) };
    expect(coverageWhen(l)).toBe("pixels observed Oct 10–Oct 15, each its latest clear day");
  });
  it("names a composite's oldest and newest days", () => {
    const l = { composite: { oldest_observed_date: "2026-10-10", newest_observed_date: "2026-10-15" } as never, multisensor: null, time: t({ observed_date: "2026-10-15" }) };
    expect(coverageWhen(l)).toBe("pixels observed Oct 10–Oct 15, each its latest clear day");
  });
  it("keeps 'overpass of' for a single day", () => {
    const l = { composite: null, multisensor: null, time: t({ observed_date: "2026-10-15", observed_times: ["2026-10-15T16:47:48Z", "2026-10-15T19:08:58Z"] }) };
    expect(coverageWhen(l)).toBe("overpass of Oct 15");
  });
});
