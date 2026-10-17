import { describe, expect, it } from "vitest";
import { relTime } from "@/v4/ui/time";

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
  });
});
