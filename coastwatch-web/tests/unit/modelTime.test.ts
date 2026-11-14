import { describe, expect, it } from "vitest";
import { allPast, isPastRun, runKindWord, runTiming } from "@/lib/modelTime";

/** The one rule for a model run's dates (T3): every valid day before today (Pacific) is a past run. */
describe("model freshness by valid dates", () => {
  it("a window that ended yesterday is past; one that holds today is current; one that starts tomorrow is upcoming", () => {
    expect(runTiming("2026-10-04", "2026-10-11", "2026-10-12")).toBe("past");
    expect(runTiming("2026-10-04", "2026-10-12", "2026-10-12")).toBe("current");
    expect(runTiming("2026-10-13", "2026-10-16", "2026-10-12")).toBe("upcoming");
    expect(isPastRun("2026-10-10", "2026-10-10", "2026-10-12")).toBe(true);
    expect(isPastRun(null, null, "2026-10-12")).toBe(false);
  });

  it("a run is past only when every lead is before today, and an empty run is not past", () => {
    expect(allPast(["2026-10-07", "2026-10-08", "2026-10-09", "2026-10-10"], "2026-10-12")).toBe(true);
    expect(allPast(["2026-10-11", "2026-10-12"], "2026-10-12")).toBe(false);
    expect(allPast(["2026-10-11", null], "2026-10-12")).toBe(false);
    expect(allPast([], "2026-10-12")).toBe(false);
  });

  it("is worded the same everywhere: 'past run (date)', else the kind", () => {
    expect(runKindWord("outlook", true, "Oct 8")).toBe("past run (Oct 8)");
    expect(runKindWord("model", true)).toBe("past run");
    expect(runKindWord("outlook", false, "Oct 8")).toBe("outlook");
    expect(runKindWord("model", false)).toBe("model");
  });
});
