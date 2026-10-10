import { describe, expect, it } from "vitest";
import type { FreshnessPolicy, Manifest, SourceStatus } from "@/generated/schema";
import { ageInDays, classifyDate, classifySource, classifyTime } from "@/lib/freshness";
import { formatDate, relativeDay } from "@/lib/time";
import { worstState } from "@/lib/health";
import manifest from "../fixture-data/v1/manifest.json";
import failedManifest from "../fixture-data-failed/v1/manifest.json";

const CHARM: FreshnessPolicy = { basis: "issued_date", current_max_age_days: 1, stale_max_age_days: 7, note: "" };
const at = (iso: string) => new Date(iso);

describe("freshness classification", () => {
  it.each([
    // ages count California (Pacific) calendar days, like the "today"/"yesterday" labels
    ["2026-10-08T08:00:00Z", "current", 0], // Oct 8, 01:00 PDT
    ["2026-10-10T06:59:00Z", "current", 1], // Oct 9, 23:59 PDT
    ["2026-10-10T01:00:00Z", "current", 1], // Oct 9, 18:00 PDT (UTC is already Oct 10)
    ["2026-10-10T07:00:00Z", "stale", 2], // Oct 10, 00:00 PDT
    ["2026-10-15T19:00:00Z", "stale", 7],
    ["2026-10-16T07:00:00Z", "historical", 8],
    ["2027-01-01T08:00:00Z", "historical", 85], // Jan 1, 00:00 PST
  ])("issued 2026-10-08 viewed at %s is %s", (now, state, age) => {
    const f = classifyDate(CHARM, "2026-10-08", at(now));
    expect(f.state).toBe(state);
    expect(f.ageDays).toBe(age);
  });

  it("missing or malformed dates are unavailable, never current", () => {
    expect(classifyDate(CHARM, null, at("2026-10-08T00:00:00Z")).state).toBe("unavailable");
    expect(classifyDate(CHARM, "Oct 8", at("2026-10-08T00:00:00Z")).state).toBe("unavailable");
    expect(classifyTime(CHARM, undefined, at("2026-10-08T00:00:00Z")).state).toBe("unavailable");
  });

  it("uses the policy basis field", () => {
    const obs: FreshnessPolicy = { basis: "observed_date", current_max_age_days: 4, stale_max_age_days: 10, note: "" };
    const time = { issued_date: "2026-09-30", observed_date: "2026-10-06", valid_date: "2026-10-06", issued_date_derived: false };
    expect(classifyTime(obs, time, at("2026-10-08T12:00:00Z")).state).toBe("current");
    expect(classifyTime(CHARM, time, at("2026-10-08T12:00:00Z")).state).toBe("historical");
  });

  it("future dates (clock skew) count as current, age 0 or less", () => {
    expect(ageInDays("2026-10-09", at("2026-10-08T19:00:00Z"))).toBe(-1);
    expect(classifyDate(CHARM, "2026-10-09", at("2026-10-08T12:00:00Z")).state).toBe("current");
  });

  it("classifies source status by its issued or valid date", () => {
    const s = (manifest as unknown as Manifest).sources.find((x) => x.source_id === "charm") as SourceStatus;
    expect(classifySource(s, at("2026-10-08T20:00:00Z")).state).toBe("current");
    expect(classifySource(s, at("2026-10-12T20:00:00Z")).state).toBe("stale");
  });

  it("a failed latest update makes the overall health at least stale", () => {
    const m = failedManifest as unknown as Manifest;
    const st = m.sources.find((x) => x.source_id === "charm")!;
    expect(st.outcome).toBe("failed");
    // same day as the last good run: data itself is current, but the failure must show
    expect(worstState(m, at("2026-10-08T20:00:00Z"))).toBe("stale");
    expect(worstState(manifest as unknown as Manifest, at("2026-10-08T20:00:00Z"))).toBe("current");
    expect(worstState(null, at("2026-10-08T20:00:00Z"))).toBe("unavailable");
  });
});

describe("date display", () => {
  it("formats calendar dates without time-zone shifting", () => {
    expect(formatDate("2026-10-08")).toBe("Thu, Oct 8");
    expect(formatDate("2026-10-08", { year: true })).toBe("Thu, Oct 8, 2026");
  });
  it("relative day uses Pacific time", () => {
    // 05:00Z on Oct 9 is still Oct 8 in California
    expect(relativeDay("2026-10-08", at("2026-10-09T05:00:00Z"))).toBe("today");
    expect(relativeDay("2026-10-09", at("2026-10-09T05:00:00Z"))).toBe("tomorrow");
    expect(relativeDay("2026-10-07", at("2026-10-09T05:00:00Z"))).toBe("yesterday");
    expect(relativeDay("2026-10-10", at("2026-10-09T05:00:00Z"))).toBe("in 2 days");
  });
});
